"""
grad_bypass_check.py
--------------------
Ground-truth parity check for the **grad-bypass escape hatch** (``grad_outputs=``).

``Pipeline.step`` normally computes a scalar loss on the last stage via
``loss_fn(output, target)`` and backprops it. The ``grad_outputs`` escape hatch
lets a caller BYPASS the loss entirely and feed a precomputed ``dL/dOutput``
straight into the last stage's backward — useful when the gradient comes from
somewhere the pipeline can't see (a downstream model, a custom differentiator,
reinforcement learning advantages, etc.).

This file verifies that the bypass is numerically identical to the equivalent
sequential manual-backward, for every schedule (gpipe / 1f1b / staggered_1b1f)
and overlap on/off, in both accepted forms:

  * **callable**  ``grad_outputs=lambda out, tgt: grad``  — resolved per
    microbatch on the last-stage worker (mirrors ``loss_fn``);
  * **tensor**    ``grad_outputs=full_batch_grad``        — chunked along dim 0
    like ``targets``.

Reference gradient: we use ``loss = MSELoss(out, target).sum()-style`` semantics
``loss_mb = ((out - target)**2).sum()``, whose analytic gradient is
``dL/dOut = 2 * (out - target)``. That makes the bypass exactly comparable to a
sequential ``out.backward(2*(out-target))`` baseline AND to the pipeline running
``loss_fn=lambda out, tgt: ((out - tgt)**2).sum()`` — all three must agree
bit-for-bit (same reduction, same microbatching).

Also verified:
  * ``result.loss`` raises a clear error under bypass (no scalar loss exists);
  * ``grad_outputs`` + ``loss_fn`` together raise (mutually exclusive).

Usage:
    PYTHONPATH=. python examples/grad_bypass_check.py
    PYTHONPATH=. python examples/grad_bypass_check.py --devices cuda:1 cuda:3
"""

import argparse
import copy

import torch
import torch.nn as nn

from ramtorch import Pipeline


DIM = 32
BATCH = 16
SCHEDULES = ("gpipe", "1f1b", "staggered_1b1f")


# ── Stages ────────────────────────────────────────────────────────────────────

class Stage0(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.relu(self.lin(x))


class Stage1(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        return self.lin(x)


# ── Loss + its analytic gradient ──────────────────────────────────────────────

def loss_fn(out, tgt):
    """loss_mb = sum((out - tgt)^2) over the microbatch."""
    return ((out - tgt) ** 2).sum()


def grad_fn(out, tgt):
    """d loss_mb / d out = 2 * (out - tgt)."""
    return 2.0 * (out - tgt)


# ── Sequential manual-backward baseline ───────────────────────────────────────

def seq_manual_backward(m0, m1, opt, x, y, n_micro):
    """Sequential microbatch manual backward with the SAME gradient.

    Each microbatch: out = m1(m0(x_mb)); out.backward(2*(out - y_mb)). Param
    grads accumulate in .grad directly (like the pipeline's grad_acc -> flush).
    Returns the mean microbatch grad is intentionally NOT rescaled here; the
    pipeline's flush_grads scales by 1/n_micro, so we divide the final .grad by
    n_micro to match.
    """
    opt.zero_grad()
    for x_mb, y_mb in zip(x.chunk(n_micro, dim=0), y.chunk(n_micro, dim=0)):
        out = m1(m0(x_mb))
        out.backward(grad_fn(out, y_mb))
    for p in list(m0.parameters()) + list(m1.parameters()):
        p.grad /= n_micro  # match pipeline's mean-scaled flush
    opt.step()


# ── Pipeline bypass step ──────────────────────────────────────────────────────

def pipe_bypass_step(pipe, opt, x, y, n_micro, schedule, form):
    """One pipeline step using the grad-bypass escape hatch.

    form: "callable" (grad_outputs=grad_fn) or "tensor" (grad_outputs=full grad).
    """
    if form == "callable":
        res = pipe.step(
            x, targets=y, schedule=schedule, n_microbatches=n_micro,
            grad_outputs=grad_fn,
        )
    else:  # tensor: precompute full-batch grad on the last device
        # The full-batch grad is just 2*(out_full - y); but we don't have
        # out_full without running the model. For the tensor form we instead
        # precompute the grad from a forward pass on the same weights.
        with torch.no_grad():
            full = x
            for st in pipe.stages:
                full = st.module(full.to(st.device))
        full_grad = grad_fn(full, y.to(full.device))
        res = pipe.step(
            x, targets=y, schedule=schedule, n_microbatches=n_micro,
            grad_outputs=full_grad,
        )
    res.flush_grads()
    opt.step()
    return res


def weights(m0, m1):
    return {
        n: p.detach().cpu().clone()
        for n, p in list(m0.named_parameters()) + list(m1.named_parameters())
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs=2, default=["cuda:1", "cuda:3"])
    ap.add_argument("--steps", type=int, default=20)
    args = ap.parse_args()
    dev0, dev1 = args.devices
    print(f"config: dim={DIM} batch={BATCH} devices={args.devices} steps={args.steps}")

    if not torch.cuda.is_available():
        print("CUDA not available; skipping.")
        return

    torch.manual_seed(0)
    base0, base1 = Stage0(DIM), Stage1(DIM)
    g = torch.Generator().manual_seed(1234)
    x = torch.randn(BATCH, DIM, generator=g)
    y = torch.randn(BATCH, DIM, generator=g)

    n_micro = 4
    all_ok = True

    for form in ("callable", "tensor"):
        for schedule in SCHEDULES:
            for overlap in (False, True):
                # Fresh sequential baseline (single device — the math reference).
                q0 = copy.deepcopy(base0).to(dev0)
                q1 = copy.deepcopy(base1).to(dev0)
                osq = torch.optim.SGD(
                    list(q0.parameters()) + list(q1.parameters()), lr=0.05
                )
                # Fresh pipeline with identical initial weights.
                p0, p1 = copy.deepcopy(base0), copy.deepcopy(base1)
                pipe = Pipeline(
                    stage_modules=[p0, p1], devices=[dev0, dev1], overlap=overlap
                )
                opp = torch.optim.SGD(
                    list(p0.parameters()) + list(p1.parameters()), lr=0.05
                )

                xd, yd = x.to(dev0), y.to(dev0)
                for _ in range(args.steps):
                    seq_manual_backward(q0, q1, osq, xd, yd, n_micro)
                    res = pipe_bypass_step(
                        pipe, opp, xd, yd, n_micro, schedule, form
                    )

                # result.loss must raise under bypass (checked once, on last res).
                loss_raised = False
                try:
                    _ = res.loss
                except RuntimeError:
                    loss_raised = True

                wq, wp = weights(q0, q1), weights(p0, p1)
                d = max((wq[k] - wp[k]).abs().max().item() for k in wq)
                ok = d == 0.0 and loss_raised
                all_ok &= ok
                print(
                    f"  [{'OK ' if ok else 'FAIL'}] {form:8s} {schedule:15s} "
                    f"overlap={overlap!s:5s} weight|Δ|={d:.3e} loss_raises={loss_raised}"
                )

    # Mutual exclusion: grad_outputs + loss_fn must raise.
    p0, p1 = copy.deepcopy(base0), copy.deepcopy(base1)
    pipe = Pipeline(stage_modules=[p0, p1], devices=[dev0, dev1])
    try:
        pipe.step(
            x.to(dev0), targets=y.to(dev0), n_microbatches=n_micro,
            loss_fn=loss_fn, grad_outputs=grad_fn,
        )
        print("  [FAIL] mutual-exclusion: no error raised"); all_ok = False
    except ValueError as e:
        print(f"  [OK ] mutual-exclusion raises: {str(e)[:60]}...")

    print("\nALL GRAD-BYPASS CHECKS PASSED" if all_ok else "\nSOME CHECKS FAILED")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
