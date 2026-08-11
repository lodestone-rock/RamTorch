"""
offload_vs_plain_demo.py
------------------------
How to use OffloadModel, side by side with a plain full-resident model.

The model is diced into chunks by YOU (the pipeline ``stage_modules``
convention) — no tracing, no module surgery::

    chunks = [make_block() for _ in range(16)]          # your own dicing
    model  = OffloadModel(chunks, device="cuda:0",
                          window=2,                      # streaming slots
                          pin=4,                         # evenly-spaced pinned
                          keep_activations=True)         # skip recompute

    out = model(x)                                       # streamed inference
    res = model.step(x, targets=y, loss_fn=F.mse_loss)   # fwd+bwd
    model.flush_grads()                                  # acc -> .grad
    opt.step()                                           # CPU+GPU params

What this demo shows:
  1. BIT-IDENTITY over a 100-step SGD training run: at every step the
     streamed model must produce exactly the same output and loss as the
     plain model — ``torch.equal``, not allclose — with gradients compared at
     sampled steps and the full weight trajectory compared at the end. Both
     models take real optimizer updates each step (SGD is mul+sub only,
     IEEE-exact, so the mixed CPU/GPU update preserves bits), so a single-ULP
     divergence anywhere would compound and fail. (Recompute mode is
     reported as max-abs-diff; it re-runs the same deterministic kernels so
     it is normally bit-identical too.)
  2. Wall-time and peak-GPU-memory comparison: plain vs streamed
     (recompute / keep / keep+pin).
  3. Perfetto traces: ``model.step(..., profile_path="....trace.json")``
     captures compute ops (F3/B3), H2D loads (L3), D2H writebacks, and stall
     waits (wait L3) along with every CUDA kernel and memcpy. Drop the JSON
     into https://ui.perfetto.dev.

Two regimes:
  * default: transfer-bound — the FF chunks are so fast that PCIe can't keep
    up, so streaming stalls on 'wait L*' and offload is slower than plain.
  * --compute-bound: each chunk additionally burns vector-core time with
    ``--spam`` rounds of pointwise mul/add on the activation. Pointwise work
    is bandwidth/launch bound — far slower per element than tensor-core
    matmul — and it adds ZERO parameters, so per-chunk compute grows while
    H2D bytes stay fixed. Loads then hide behind compute and the streamed
    step approaches plain speed at a fraction of the memory. (Scalar mul/add
    is chosen deliberately: its backward saves no activations, so keep mode
    stays cheap.)

Run:  PYTHONPATH=. python examples/offload_vs_plain_demo.py [--compute-bound]
"""

import argparse
import copy
import sys
import time

import torch
import torch.nn as nn

from ramtorch import OffloadModel

LOSS = nn.functional.mse_loss


class Block(nn.Module):
    """FF block + optional vector-core burner (``spam`` pointwise rounds)."""

    def __init__(self, d: int, spam: int = 0):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d)
        )
        self.spam = spam

    def forward(self, x):
        x = self.ff(x)
        # scalar pointwise chain: mul ~1.0 / add ~0.0 keeps values stable and
        # bit-deterministic while soaking up vector-core / launch bandwidth
        for _ in range(self.spam):
            x = x.mul(0.999977).add(2**-20)
        return x


def build_chunks(n: int, d: int, seed: int = 0, spam: int = 0):
    torch.manual_seed(seed)
    return [Block(d, spam) for _ in range(n)]


def plain_step(model: nn.Sequential, x, y):
    model.zero_grad(set_to_none=True)
    out = model(x)
    loss = LOSS(out, y)
    loss.backward()
    return out, loss


def bit_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.detach().cpu(), b.detach().cpu())


# ── 1. bit-identity (keep_activations mode) ──────────────────────────────────
#
# Multi-step TRAINING TRAJECTORY check: each step draws fresh data, runs
# fwd+bwd on both models, compares, then applies an SGD update to both.
# SGD (no momentum) is only mul + sub — IEEE-exact single-rounding ops — so
# updating streamed params on the CPU matches plain's GPU update bit-for-bit,
# and any divergence at any step would compound and get caught immediately.

def _grad_check_steps(steps: int):
    return {0, steps - 1} | set(range(0, steps, 25))


def check_bit_identity(device, n, d, batch, window, pin, spam=0,
                       steps=1) -> bool:
    chunks = build_chunks(n, d, spam=spam)
    plain = nn.Sequential(*copy.deepcopy(chunks)).to(device)
    model = OffloadModel(chunks, device=device, window=window, pin=pin,
                         keep_activations=True)
    opt_ref = torch.optim.SGD(plain.parameters(), lr=1e-3, foreach=False)
    opt_off = torch.optim.SGD(model.parameters(), lr=1e-3, foreach=False)
    gen = torch.Generator().manual_seed(1)

    first_bad = None
    grad_steps = _grad_check_steps(steps)
    for step in range(steps):
        x = torch.randn(batch, d, generator=gen)
        y = torch.randn(batch, d, generator=gen)
        ref_out, ref_loss = plain_step(plain, x.to(device), y.to(device))
        res = model.step(x, targets=y, loss_fn=LOSS)
        model.flush_grads()

        step_ok = (bit_equal(res.output, ref_out)
                   and bit_equal(res.loss, ref_loss))
        if step in grad_steps:  # full grad compare is 0.5 GB — sample it
            for i in range(n):
                got = dict(model.chunks[i].named_parameters())
                for nme, p in dict(plain[i].named_parameters()).items():
                    step_ok &= bit_equal(got[nme].grad, p.grad)
        opt_ref.step()
        opt_off.step()
        if not step_ok and first_bad is None:
            first_bad = step

    # cumulative: the entire weight trajectory must have stayed identical
    weights_ok = all(
        bit_equal(dict(model.chunks[i].named_parameters())[nme], p)
        for i in range(n)
        for nme, p in dict(plain[i].named_parameters()).items()
    )
    ok = first_bad is None and weights_ok
    status = ("BIT-IDENTICAL" if ok else
              f"DIVERGED (first at step {first_bad})" if first_bad is not None
              else "final weights DIFFER")
    print(f"  [keep W={window} pin={pin}] {steps} SGD steps "
          f"(loss+output every step, grads @ {len(grad_steps)} steps, "
          f"final weights): {status}")
    model.close()
    return ok


def check_recompute_closeness(device, n, d, batch, spam=0, steps=1) -> bool:
    chunks = build_chunks(n, d, spam=spam)
    plain = nn.Sequential(*copy.deepcopy(chunks)).to(device)
    model = OffloadModel(chunks, device=device, window=2,
                         keep_activations=False)
    opt_ref = torch.optim.SGD(plain.parameters(), lr=1e-3, foreach=False)
    opt_off = torch.optim.SGD(model.parameters(), lr=1e-3, foreach=False)
    gen = torch.Generator().manual_seed(1)

    worst, identical = 0.0, True
    grad_steps = _grad_check_steps(steps)
    for step in range(steps):
        x = torch.randn(batch, d, generator=gen)
        y = torch.randn(batch, d, generator=gen)
        _, ref_loss = plain_step(plain, x.to(device), y.to(device))
        res = model.step(x, targets=y, loss_fn=LOSS)
        model.flush_grads()
        worst = max(worst, (res.loss.cpu() - ref_loss.cpu()).abs().item())
        identical &= bit_equal(res.loss, ref_loss)
        if step in grad_steps:
            for i in range(n):
                got = dict(model.chunks[i].named_parameters())
                for nme, p in dict(plain[i].named_parameters()).items():
                    worst = max(worst, (got[nme].grad.cpu()
                                        - p.grad.cpu()).abs().max().item())
                    identical &= bit_equal(got[nme].grad, p.grad)
        opt_ref.step()
        opt_off.step()
    identical &= all(
        bit_equal(dict(model.chunks[i].named_parameters())[nme], p)
        for i in range(n)
        for nme, p in dict(plain[i].named_parameters()).items()
    )
    print(f"  [recompute W=2] {steps} SGD steps: max |diff| vs plain = "
          f"{worst:.3e}"
          f"{'  (bit-identical incl. final weights)' if identical else ''}")
    model.close()
    return worst < 1e-5


# ── 2. wall time + peak memory ────────────────────────────────────────────────

def bench(fn, device, warmup=1, iters=3) -> float:
    is_cuda = torch.device(device).type == "cuda"
    for _ in range(warmup):
        fn()
    if is_cuda:
        torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    if is_cuda:
        torch.cuda.synchronize(device)
    return (time.perf_counter() - t0) / iters


def compare_speed_memory(device, n, d, batch, spam=0):
    is_cuda = torch.device(device).type == "cuda"
    torch.manual_seed(1)
    x, y = torch.randn(batch, d), torch.randn(batch, d)
    chunks = build_chunks(n, d, spam=spam)

    rows = []

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
    plain = nn.Sequential(*copy.deepcopy(chunks)).to(device)
    xd, yd = x.to(device), y.to(device)
    t = bench(lambda: plain_step(plain, xd, yd), device)
    peak = torch.cuda.max_memory_allocated(device) if is_cuda else 0
    rows.append(("plain (full-resident)", t, peak, "-", "-"))
    del plain
    if is_cuda:
        torch.cuda.empty_cache()

    for name, kw in [
        ("offload recompute W=2", dict(window=2, keep_activations=False)),
        ("offload keep      W=2", dict(window=2, keep_activations=True)),
        ("offload keep W=2 pin=8", dict(window=2, pin=8,
                                        keep_activations=True)),
    ]:
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
        model = OffloadModel(copy.deepcopy(chunks), device=device, **kw)
        t = bench(lambda: model.step(x, targets=y, loss_fn=LOSS), device)
        peak = torch.cuda.max_memory_allocated(device) if is_cuda else 0
        loads = model.stats["loads"]
        wait = model.stats["acquire_wait_s"]
        rows.append((name, t, peak, loads, f"{wait * 1e3:.0f} ms"))
        model.close()
        if is_cuda:
            torch.cuda.empty_cache()

    print(f"\n  {'config':<24} {'step time':>10} {'peak GPU':>10} "
          f"{'loads':>6} {'total wait':>10}")
    for name, t, peak, loads, wait in rows:
        peak_s = f"{peak / 2**20:.0f} MiB" if is_cuda else "n/a"
        print(f"  {name:<24} {t * 1e3:>8.1f}ms {peak_s:>10} "
              f"{loads!s:>6} {wait!s:>10}")


# ── 3. Perfetto traces ────────────────────────────────────────────────────────

def export_traces(device, n, d, batch, spam=0, suffix=""):
    torch.manual_seed(1)
    x, y = torch.randn(batch, d), torch.randn(batch, d)
    paths = []
    for name, keep in [("recompute", False), ("keep", True)]:
        model = OffloadModel(build_chunks(n, d, spam=spam), device=device,
                             window=2, pin=4, keep_activations=keep)
        model.step(x, targets=y, loss_fn=LOSS)  # warm the window first
        path = f"examples/offload_step_{name}{suffix}.trace.json"
        model.step(x, targets=y, loss_fn=LOSS, profile_path=path)
        model.close()
        paths.append(path)
        print(f"  wrote {path}")
    print("  open them at https://ui.perfetto.dev — look for F*/B* compute "
          "spans, L* h2d loads,\n  'wait L*' stalls, G d2h writebacks, and "
          "the Memcpy HtoD/DtoH rows overlapping compute.")
    return paths


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--compute-bound", action="store_true",
                    help="make chunks compute-heavy (pointwise burner) so "
                         "H2D loads hide behind compute")
    ap.add_argument("--spam", type=int, default=None,
                    help="pointwise mul/add rounds per chunk "
                         "(default 0, or 150 with --compute-bound)")
    ap.add_argument("--steps", type=int, default=100,
                    help="SGD steps for the bit-identity trajectory check")
    args = ap.parse_args()
    spam = args.spam if args.spam is not None else (
        150 if args.compute_bound else 0
    )

    if torch.cuda.is_available():
        device, n, d, batch = "cuda:0", 16, 1024, 64
        if spam:
            # bigger batch so each pointwise kernel is bandwidth-bound on the
            # GPU (tens of µs) rather than launch-bound on the CPU — with
            # tiny kernels both models just measure launch overhead
            batch = 4096
    else:
        device, n, d, batch = "cpu", 6, 128, 16
        spam = min(spam, 30)  # CPU pointwise is slow enough already
    regime = "compute-bound" if spam else "transfer-bound"
    print(f"device={device}, {n} chunks of Linear({d},{4 * d})+GELU+"
          f"Linear({4 * d},{d}), batch={batch}")
    print(f"regime: {regime}"
          + (f" (spam={spam} pointwise rounds per chunk)\n" if spam
             else " (raw FF chunks, PCIe can't keep up)\n"))

    print(f"bit-identity vs plain model ({args.steps}-step SGD training "
          "run):")
    ok = check_bit_identity(device, n, d, batch, window=2, pin=0, spam=spam,
                            steps=args.steps)
    ok &= check_bit_identity(device, n, d, batch, window=2, pin=4, spam=spam,
                             steps=args.steps)
    ok &= check_recompute_closeness(device, n, d, batch, spam=spam,
                                    steps=args.steps)

    print("\nspeed / memory (one training step, mean of 3):")
    compare_speed_memory(device, n, d, batch, spam=spam)

    print("\nperfetto traces:")
    export_traces(device, n, d, batch, spam=spam,
                  suffix="_compute" if spam else "")

    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
