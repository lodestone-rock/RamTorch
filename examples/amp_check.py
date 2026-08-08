"""
amp_check.py
------------
Mixed precision (autocast) parity check for the pipeline.

``torch.autocast`` is THREAD-LOCAL, so wrapping ``pipe.step(...)`` in an
autocast context in the caller thread does nothing — each stage runs in its own
worker thread. The pipeline instead takes ``autocast=`` and each stage enters
the context itself around its forward + loss (backward runs outside autocast,
params/grads stay fp32).

This file verifies that against a plain sequential baseline running under the
same autocast, reusing the tuple/no-grad stages so mixed precision is exercised
together with a multi-arg stage boundary (float grad arg + forward-only float
arg + bool mask):

  1. Single-step loss/grad parity (all schedules, overlap on/off) vs sequential
     microbatch grad-accum under the same autocast — expected BIT-EXACT (0.0).
  2. Multi-step bit-identity training check (SGD) — final weights expected 0.0,
     for both the bf16 leg and the fp32 (autocast=None) regression leg.
  3. fp16 guard: ``step()`` must raise (fp16 training needs loss scaling,
     unsupported); ``infer()`` with fp16 must run and return finite fp16 logits.

Usage:
    PYTHONPATH=. python examples/amp_check.py
    PYTHONPATH=. python examples/amp_check.py --devices cuda:1 cuda:3 --steps 5
"""

import argparse
import contextlib
import copy

import torch
import torch.nn as nn

from ramtorch import Pipeline


DIM = 32
BATCH = 16


# ── Stages (same shape as tuple_no_grad_check: tuple output, no-grad args) ────

class TupleStage0(nn.Module):
    out_no_grad = (1,)  # aux_logits is forward-only

    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.aux = nn.Linear(dim, dim)

    def forward(self, x):
        feat = torch.tanh(self.lin(x))
        aux = self.aux(x)
        mask = feat > 0  # bool mask (non-float -> auto no-grad)
        return feat, aux, mask


class TupleStage1(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, feat, aux, mask):
        aux = aux.detach()  # forward-only; never differentiate it
        pooled = (feat * mask.to(feat.dtype)).sum(dim=1, keepdim=True) / mask.sum(
            dim=1, keepdim=True
        ).clamp(min=1).to(feat.dtype)
        h = self.proj(feat) + 0.01 * aux.tanh() + pooled
        return self.head(torch.relu(h))


# ── Sequential microbatch-accum reference (under the same autocast) ──────────

def _autocast_ctx(device: torch.device, dtype):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=dtype)


def _seq_microbatch_grads(s0, s1, x, y, n_micro, dtype):
    """One sequential microbatch-accum backward pass; grads left in .grad
    (scaled by 1/n_micro, exactly like PipelineResult.flush_grads)."""
    device = next(s0.parameters()).device
    for p in list(s0.parameters()) + list(s1.parameters()):
        p.grad = None
    mb_losses = []
    for xmb, ymb in zip(x.chunk(n_micro), y.chunk(n_micro)):
        # Forward + loss under autocast; backward outside (standard AMP).
        with _autocast_ctx(device, dtype):
            feat, aux, mask = s0(xmb)
            logits = s1(feat, aux, mask)
            loss = nn.functional.cross_entropy(logits, ymb)
        loss.backward()
        mb_losses.append(loss.detach())
    with torch.no_grad():
        for m in (s0, s1):
            for p in m.parameters():
                if p.grad is not None:
                    p.grad.mul_(1.0 / n_micro)
    return torch.stack(mb_losses).mean()


def _seq_microbatch_step(s0, s1, opt, x, y, n_micro, dtype):
    _seq_microbatch_grads(s0, s1, x, y, n_micro, dtype)
    opt.step()


def _pipeline_step(pipe, opt, x, y, n_micro, schedule):
    result = pipe.step(x, targets=y, schedule=schedule, n_microbatches=n_micro,
                       loss_fn=nn.functional.cross_entropy)
    result.flush_grads()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return float(result.loss)


def collect_grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters()
            if p.grad is not None}


# ── Checks ────────────────────────────────────────────────────────────────────

def check_grad_parity(base0, base1, devices, x, y, n_micro, dtype, label):
    """Pipeline grads/loss vs sequential microbatch-accum under same autocast.
    Same kernels on the same GPU model -> expected bit-exact (0.0)."""
    ref_dev = torch.device(devices[0])
    q0 = copy.deepcopy(base0).to(ref_dev)
    q1 = copy.deepcopy(base1).to(ref_dev)
    ref_loss = _seq_microbatch_grads(q0, q1, x.to(ref_dev), y.to(ref_dev),
                                     n_micro, dtype)
    ref = {**{f"s0.{n}": g for n, g in collect_grads(q0).items()},
           **{f"s1.{n}": g for n, g in collect_grads(q1).items()}}

    all_ok = True
    for schedule in ("gpipe", "1f1b", "staggered_1b1f"):
        for overlap in (False, True):
            p0 = copy.deepcopy(base0)
            p1 = copy.deepcopy(base1)
            pipe = Pipeline(stage_modules=[p0, p1], devices=devices,
                            overlap=overlap, autocast=dtype)
            result = pipe.step(x, targets=y, schedule=schedule,
                               n_microbatches=n_micro,
                               loss_fn=nn.functional.cross_entropy)
            result.flush_grads()
            got = {**{f"s0.{n}": g for n, g in collect_grads(p0).items()},
                   **{f"s1.{n}": g for n, g in collect_grads(p1).items()}}

            loss_d = abs(float(result.loss) - float(ref_loss))
            worst = ("", 0.0)
            ok = loss_d == 0.0 and set(ref) <= set(got)
            for name, gp in got.items():
                gref = ref.get(name)
                if gref is None:
                    # Forward-only params (e.g. s0.aux): the baseline leaves
                    # .grad = None while flush_grads writes zeros — require
                    # the pipeline grad to be exactly zero.
                    d = gp.abs().max().item()
                else:
                    d = (gp.cpu() - gref.cpu()).abs().max().item()
                if d > worst[1]:
                    worst = (name, d)
                if d != 0.0:
                    ok = False
            status = "OK " if ok else "FAIL"
            print(f"  [{status}] {label:<5} {schedule:<15} overlap={overlap!s:<5} "
                  f"lossΔ={loss_d:.2e}  gradΔ={worst[1]:.2e}({worst[0]})")
            all_ok &= ok
    return all_ok


def check_bit_identity(base0, base1, devices, x, y, n_micro, steps, lr, dtype,
                       label):
    """Multi-step training: pipeline vs sequential microbatch-accum, both under
    the same autocast. Final weights expected bit-identical (0.0)."""
    ref_dev = torch.device(devices[0])
    all_ok = True
    for schedule in ("gpipe", "1f1b", "staggered_1b1f"):
        q0 = copy.deepcopy(base0).to(ref_dev)
        q1 = copy.deepcopy(base1).to(ref_dev)
        xq, yq = x.to(ref_dev), y.to(ref_dev)
        opt_seq = torch.optim.SGD(list(q0.parameters()) + list(q1.parameters()),
                                  lr=lr)
        p0 = copy.deepcopy(base0)
        p1 = copy.deepcopy(base1)
        pipe = Pipeline(stage_modules=[p0, p1], devices=devices, overlap=True,
                        autocast=dtype)
        opt_pipe = torch.optim.SGD(list(p0.parameters()) + list(p1.parameters()),
                                   lr=lr)
        for _ in range(steps):
            _seq_microbatch_step(q0, q1, opt_seq, xq, yq, n_micro, dtype)
            _pipeline_step(pipe, opt_pipe, x, y, n_micro, schedule)

        worst = ("", 0.0)
        for (qn, qp), (pn, pp) in list(zip(q0.named_parameters(),
                                           p0.named_parameters())) + \
                                  list(zip(q1.named_parameters(),
                                           p1.named_parameters())):
            assert qn == pn
            d = (qp.detach().cpu() - pp.detach().cpu()).abs().max().item()
            if d > worst[1]:
                worst = (qn, d)
        ok = worst[1] == 0.0
        print(f"  [{'OK ' if ok else 'DIFF'}] {label:<5} {schedule:<15} "
              f"final-weight max|Δ|={worst[1]:.3e} ({worst[0]})")
        all_ok &= ok
    return all_ok


def check_fp16_guard(base0, base1, devices, x, y, n_micro):
    """fp16 must be rejected for training (no loss scaling) but usable in infer."""
    pipe = Pipeline(
        stage_modules=[copy.deepcopy(base0), copy.deepcopy(base1)],
        devices=devices, autocast="fp16",
    )
    try:
        pipe.step(x, targets=y, n_microbatches=n_micro,
                  loss_fn=nn.functional.cross_entropy)
        print("  [FAIL] fp16 step() did NOT raise")
        step_ok = False
    except ValueError as e:
        print(f"  [OK ] fp16 step() raises: {e}")
        step_ok = True

    out = pipe.infer(x, n_microbatches=n_micro)
    infer_ok = out.dtype == torch.float16 and bool(torch.isfinite(out).all())
    print(f"  [{'OK ' if infer_ok else 'FAIL'}] fp16 infer() runs: "
          f"dtype={out.dtype}, finite={bool(torch.isfinite(out).all())}")
    return step_ok and infer_ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs=2, default=["cuda:0", "cuda:1"])
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    base0 = TupleStage0(DIM)
    base1 = TupleStage1(DIM)

    g = torch.Generator().manual_seed(1234)
    x = torch.randn(BATCH, DIM, generator=g)
    y = torch.randint(0, DIM, (BATCH,), generator=g)

    print(f"config: dim={DIM} batch={BATCH} micro={args.micro} "
          f"devices={args.devices}")

    all_ok = True

    print("\nsingle-step grad parity vs sequential microbatch-accum "
          "(same autocast, expected bit-exact):")
    all_ok &= check_grad_parity(base0, base1, args.devices, x, y, args.micro,
                                torch.bfloat16, "bf16")
    all_ok &= check_grad_parity(base0, base1, args.devices, x, y, args.micro,
                                None, "fp32")

    print(f"\nbit-identity training check: {args.steps} steps, SGD lr={args.lr}, "
          f"micro={args.micro}:")
    all_ok &= check_bit_identity(base0, base1, args.devices, x, y, args.micro,
                                 args.steps, args.lr, torch.bfloat16, "bf16")
    all_ok &= check_bit_identity(base0, base1, args.devices, x, y, args.micro,
                                 args.steps, args.lr, None, "fp32")

    print("\nfp16 guard (training rejected, inference allowed):")
    all_ok &= check_fp16_guard(base0, base1, args.devices, x, y, args.micro)

    print("\nALL AMP CHECKS PASSED" if all_ok else "\nFAILURES — see above")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
