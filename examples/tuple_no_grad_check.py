"""
tuple_no_grad_check.py
----------------------
Ground-truth parity check for **tuple stage outputs with per-arg no-grad flags**.

A stage may return ``(a, b, c)`` where some args are forward-only:

  * non-floating-point outputs (bool/int/long, e.g. attention masks) are
    AUTO-skipped in backward — they can never require grad;
  * any output index listed in the module's ``out_no_grad`` attribute is
    explicitly forward-only, even if it is floating-point.

The backward must then flow ONLY through the grad-needing args. This file
verifies that against a plain sequential baseline that does the same masking:

  Stage0 (cuda:0): returns ``(feat, logits_aux, bool_mask)``
      - ``feat``       float, feeds stage 1          -> grad
      - ``logits_aux`` float, detached in stage 1    -> explicit ``out_no_grad=(1,)``
      - ``bool_mask``  bool                          -> auto no-grad (non-float)
  Stage1 (cuda:1): consumes ``feat`` (and receives the other two args, using
      the mask for pooling), returns logits.

Checks, for every schedule (gpipe / 1f1b / staggered_1b1f, overlap on/off):
  1. loss parity vs sequential baseline
  2. per-parameter gradient parity vs sequential baseline

Usage:
    PYTHONPATH=. python examples/tuple_no_grad_check.py
    PYTHONPATH=. python examples/tuple_no_grad_check.py --devices cuda:1 cuda:3
"""

import argparse
import copy

import torch
import torch.nn as nn

from ramtorch import Pipeline


DIM = 32
BATCH = 16


# ── Stages ────────────────────────────────────────────────────────────────────

class TupleStage0(nn.Module):
    """First stage: returns (feat, aux_logits, bool_mask).

    - ``feat``       (B, DIM) float — differentiable, consumed by stage 1.
    - ``aux_logits`` (B, DIM) float — explicitly forward-only (out_no_grad).
    - ``bool_mask``  (B, DIM) bool  — auto forward-only (non-float).
    """

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
    """Last stage: takes (feat, aux_logits, bool_mask) -> logits.

    Uses the bool mask for masked mean pooling; uses aux detached (so even if
    the relay accidentally gave it a grad path, the numerics would differ from
    the baseline, which also detaches — keeping the parity honest).
    """

    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.head = nn.Linear(dim, dim)

    def forward(self, feat, aux, mask):
        aux = aux.detach()  # aux is forward-only; never differentiate it
        pooled = (feat * mask.to(feat.dtype)).sum(dim=1, keepdim=True) / mask.sum(
            dim=1, keepdim=True
        ).clamp(min=1).to(feat.dtype)
        h = self.proj(feat) + self.aux_bias(aux) + pooled
        return self.head(torch.relu(h))

    def aux_bias(self, aux):
        return 0.01 * aux.tanh()


# ── Sequential baseline ───────────────────────────────────────────────────────

def run_sequential(s0, s1, x, y):
    feat, aux, mask = s0(x)
    logits = s1(feat, aux, mask)
    loss = nn.functional.cross_entropy(logits, y)
    loss.backward()
    return loss.detach()


# ── Pipeline check ────────────────────────────────────────────────────────────

def run_pipeline(s0, s1, devices, schedule, overlap, x, y, n_micro):
    pipe = Pipeline(stage_modules=[s0, s1], devices=devices, overlap=overlap)
    result = pipe.step(
        x, targets=y, schedule=schedule, n_microbatches=n_micro,
        loss_fn=nn.functional.cross_entropy,
    )
    result.flush_grads()
    return result


def collect_grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


# ── Multi-step bit-identity training check ────────────────────────────────────
# The pipeline should be BIT-IDENTICAL (0.0) to *sequential microbatch grad
# accumulation* — NOT to a single full-batch backward. Both do mean-of-microbatch
# -means with the same reduction order; the only difference is pipeline overlap.
# This trains both for --steps steps with SGD and compares final weights.

def _seq_microbatch_step(s0, s1, opt, x, y, n_micro):
    """One sequential microbatch-accum step (the bit-identity reference)."""
    opt.zero_grad(set_to_none=True)
    mb_losses = []
    for xmb, ymb in zip(x.chunk(n_micro), y.chunk(n_micro)):
        feat, aux, mask = s0(xmb)
        logits = s1(feat, aux, mask)
        loss = nn.functional.cross_entropy(logits, ymb)  # mean over microbatch
        loss.backward()                                   # accumulate into .grad
        mb_losses.append(loss.item())
    # Scale accumulated grads by 1/n_micro == mean of microbatch means,
    # exactly like PipelineResult.flush_grads().
    with torch.no_grad():
        for m in (s0, s1):
            for p in m.parameters():
                if p.grad is not None:
                    p.grad.mul_(1.0 / n_micro)
    opt.step()
    return sum(mb_losses) / len(mb_losses)


def _pipeline_step(pipe, opt, x, y, n_micro, schedule):
    result = pipe.step(x, targets=y, schedule=schedule, n_microbatches=n_micro,
                       loss_fn=nn.functional.cross_entropy)
    result.flush_grads()
    opt.step()
    opt.zero_grad(set_to_none=True)
    return float(result.loss)


def run_bit_identity(base0, base1, devices, schedule, x, y, n_micro, steps, lr):
    """Train pipeline vs sequential microbatch-accum; compare final weights."""
    # Sequential microbatch-accum reference on ONE GPU (devices[0]). It must
    # run on the same device type as the pipeline: CPU and CUDA matmuls differ
    # in the last ULP, so a CPU reference can never be bit-identical.
    ref_dev = torch.device(devices[0])
    q0 = copy.deepcopy(base0).to(ref_dev)
    q1 = copy.deepcopy(base1).to(ref_dev)
    xq, yq = x.to(ref_dev), y.to(ref_dev)
    opt_seq = torch.optim.SGD(
        list(q0.parameters()) + list(q1.parameters()), lr=lr
    )
    # Pipeline (fresh identical init).
    p0, p1 = copy.deepcopy(base0), copy.deepcopy(base1)
    pipe = Pipeline(stage_modules=[p0, p1], devices=devices, overlap=True)
    opt_pipe = torch.optim.SGD(
        list(p0.parameters()) + list(p1.parameters()), lr=lr
    )

    for _ in range(steps):
        _seq_microbatch_step(q0, q1, opt_seq, xq, yq, n_micro)
        _pipeline_step(pipe, opt_pipe, x, y, n_micro, schedule)

    # Compare final weights (stage0 vs stage0, stage1 vs stage1).
    worst = ("", 0.0)
    for (qn, qp), (pn, pp) in zip(q0.named_parameters(), p0.named_parameters()):
        assert qn == pn
        d = (qp.detach().cpu() - pp.detach().cpu()).abs().max().item()
        if d > worst[1]:
            worst = (f"s0.{qn}", d)
    for (qn, qp), (pn, pp) in zip(q1.named_parameters(), p1.named_parameters()):
        assert qn == pn
        d = (qp.detach().cpu() - pp.detach().cpu()).abs().max().item()
        if d > worst[1]:
            worst = (f"s1.{qn}", d)
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--devices", nargs=2, default=["cuda:0", "cuda:1"])
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=0,
                    help="if >0, also run the multi-step bit-identity training "
                         "check (pipeline vs sequential microbatch-accum, SGD)")
    ap.add_argument("--lr", type=float, default=0.05)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    base0 = TupleStage0(DIM)
    base1 = TupleStage1(DIM)

    g = torch.Generator().manual_seed(1234)
    x = torch.randn(BATCH, DIM, generator=g)
    y = torch.randint(0, DIM, (BATCH,), generator=g)

    # ── Sequential ground truth (on devices[0], same device type as the
    # pipeline — a CPU baseline differs from CUDA in the last ULP) ───────────
    ref_dev = torch.device(args.devices[0])
    seq0 = copy.deepcopy(base0).to(ref_dev)
    seq1 = copy.deepcopy(base1).to(ref_dev)
    gt_loss = run_sequential(seq0, seq1, x.to(ref_dev), y.to(ref_dev))
    gt0 = collect_grads(seq0)
    gt1 = collect_grads(seq1)

    print(f"config: dim={DIM} batch={BATCH} micro={args.micro} devices={args.devices}")
    print(f"ground-truth loss: {gt_loss.item():.6f}")

    all_ok = True
    for schedule in ("gpipe", "1f1b", "staggered_1b1f"):
        for overlap in (False, True):
            p0 = copy.deepcopy(base0)
            p1 = copy.deepcopy(base1)
            result = run_pipeline(
                p0, p1, args.devices, schedule, overlap, x, y, args.micro
            )
            loss = float(result.loss)
            g0 = collect_grads(p0)
            g1 = collect_grads(p1)

            loss_d = abs(loss - gt_loss.item())
            ok = loss_d < 1e-5
            worst = ("", 0.0)
            for name, gref in list(gt0.items()) + list(gt1.items()):
                got = g0.get(name, g1.get(name))
                if got is None:
                    ok = False
                    worst = (name, float("inf"))
                    continue
                d = (got.cpu() - gref.cpu()).abs().max().item()
                if d > worst[1]:
                    worst = (name, d)
                if d > 1e-5:
                    ok = False
            status = "OK " if ok else "FAIL"
            print(f"  [{status}] {schedule:<15} overlap={overlap!s:<5} "
                  f"lossΔ={loss_d:.2e}  gradΔ={worst[1]:.2e}({worst[0]})")
            all_ok &= ok

    # ── Multi-step bit-identity vs sequential microbatch-accum ──────────────
    if args.steps > 0:
        print(f"\nbit-identity training check: {args.steps} steps, SGD lr={args.lr}, "
              f"micro={args.micro} (pipeline vs sequential microbatch-accum)")
        for schedule in ("gpipe", "1f1b", "staggered_1b1f"):
            name, d = run_bit_identity(
                base0, base1, args.devices, schedule, x, y, args.micro,
                args.steps, args.lr,
            )
            # Bit-identical means exactly 0.0; allow a tiny floor for any
            # non-deterministic GPU kernel reduction (cublas/attention).
            ok = d == 0.0
            print(f"  [{'OK ' if ok else 'DIFF'}] {schedule:<15} "
                  f"final-weight max|Δ|={d:.3e} ({name})")
            all_ok &= ok

    print("\nALL TUPLE/NO-GRAD CHECKS PASSED" if all_ok else "\nFAILURES — see above")
    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
