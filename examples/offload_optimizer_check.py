"""
offload_optimizer_check.py
--------------------------
Correctness + speed check for ``ramtorch.offload_optimizer.OffloadAdamW`` —
the windowed CPU->GPU streaming AdamW (PRIVATE / educational, not exported).

Why this module is educational rather than recommended: the benchmark below
shows it. A streamed optimizer step is PCIe-bound (~28 B/param through a
20-50 GB/s link), while ``torch.optim.AdamW(fused=True)`` on CPU params is a
single multithreaded pass over host RAM at DDR bandwidth (100+ GB/s on
workstation boards). No overlap trick lets PCIe beat one DDR pass, so
**use ``fused=True`` for OffloadModel training** (see docs/offload.md and
offload_quickstart.py). The streamed design still beats the *non-fused*
foreach CPU path and the legacy per-param-bounce ``ramtorch.AdamW``, and
remains a worked example of the window/stream/event pattern on an optimizer.

1. PARITY: train an OffloadModel (streamed weights, mixed CPU/GPU params)
   with OffloadAdamW against a full-resident reference trained with
   ``torch.optim.AdamW`` — same data, same steps. On a single deterministic
   device the trajectories should be bit-identical: the streamed executor
   computes the same math (validated in offload_vs_plain_demo) and
   OffloadAdamW replicates torch's multi-tensor AdamW op order in fp32 on
   the GPU.

2. BENCH: one optimizer step at the same param count, comparing
     * OffloadAdamW (fp32 state, streamed through the GPU window)
     * OffloadAdamW (bf16 state + stochastic rounding — half the PCIe bytes)
     * torch.optim.AdamW on CPU params (foreach, and fused — the winner)
     * legacy ramtorch.AdamW (per-param GPU bounce, no overlap)
     * torch.optim.AdamW with params RESIDENT on the GPU (foreach + fused) —
       a reference point only: it needs params+grads+state (4x weight bytes)
       in GPU memory, which is exactly what offloading avoids spending.
   Effective GB/s = optimizer-state bytes moved over PCIe per step / time
   (fp32 state: 16 B/elem up + 12 B/elem down = 28; bf16 state: 20).

Run:
    PYTHONPATH=. python examples/offload_optimizer_check.py
    PYTHONPATH=. python examples/offload_optimizer_check.py \
        --device cuda:1 --steps 20 --bench-params-m 128
"""

import argparse
import copy
import gc
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ramtorch import AdamW as LegacyAdamW
from ramtorch import OffloadModel
from ramtorch.offload_optimizer import OffloadAdamW  # private / educational

DIM = 256
N_CHUNKS = 12
BATCH = 32


def make_block(d: int) -> nn.Module:
    return nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))


# ── 1. parity ─────────────────────────────────────────────────────────────────

def check_parity(dev: str, steps: int, window: int, pin: int) -> bool:
    print(f"\nPARITY  device={dev} chunks={N_CHUNKS} dim={DIM} "
          f"window={window} pin={pin} steps={steps}")
    torch.manual_seed(0)
    chunks = [make_block(DIM) for _ in range(N_CHUNKS)]
    reference = nn.Sequential(*copy.deepcopy(chunks)).to(dev)

    model = OffloadModel(chunks, device=dev, window=window, pin=pin,
                         keep_activations=True)
    opt = OffloadAdamW(model.parameters(), lr=1e-3, weight_decay=0.01,
                       device=dev, bucket_mb=8)
    opt_ref = torch.optim.AdamW(reference.parameters(), lr=1e-3,
                                weight_decay=0.01, foreach=True)

    gen = torch.Generator().manual_seed(1)
    worst = 0.0
    for step in range(steps):
        xb = torch.randn(BATCH, DIM, generator=gen)
        yb = torch.randn(BATCH, DIM, generator=gen)

        res = model.step(xb, targets=yb, loss_fn=F.mse_loss)
        model.flush_grads()          # pinned .grad + residency invalidation
        opt.step()
        model.zero_grad_acc()

        opt_ref.zero_grad(set_to_none=True)
        ref_loss = F.mse_loss(reference(xb.to(dev)), yb.to(dev))
        ref_loss.backward()
        opt_ref.step()

        d = (res.loss.cpu() - ref_loss.cpu()).abs().item()
        worst = max(worst, d)
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            print(f"  step {step:3d}  loss={float(res.loss):.6f}  "
                  f"|loss diff|={d:.3e}")

    weight_diff = max(
        (dict(model.chunks[i].named_parameters())[name].detach().cpu()
         - p.detach().cpu()).abs().max().item()
        for i in range(N_CHUNKS)
        for name, p in dict(reference[i].named_parameters()).items()
    )
    print(f"  worst |loss diff| over run:        {worst:.3e}")
    print(f"  max |weight diff| over all chunks: {weight_diff:.3e}  "
          f"(bit-identical: {worst == 0.0 and weight_diff == 0.0})")
    model.close()
    # bit-identity is expected on one deterministic device; keep a small
    # tolerance for exotic kernel-selection paths
    return worst < 1e-3 and weight_diff < 1e-4


# ── 2. benchmark ──────────────────────────────────────────────────────────────

def _make_params(numel_total: int, where: str, dev: str):
    """A pile of contiguous fp32 params + grads (roughly 16M each).

    ``where``: "pinned" (CPU pinned), "cpu" (pageable), or "gpu".
    """
    chunk = 4096 * 4096
    params = []
    left = numel_total
    gen = torch.Generator().manual_seed(5)
    while left > 0:
        n = min(chunk, left)
        p = nn.Parameter(torch.randn(n, generator=gen) * 0.01)
        g = torch.randn(n, generator=gen) * 0.001
        if where == "pinned":
            p.data = p.data.pin_memory()
            g = g.pin_memory()
        elif where == "gpu":
            p.data = p.data.to(dev)
            g = g.to(dev)
        p.grad = g
        params.append(p)
        left -= n
    return params


def _time_steps(opt, warmup=2, iters=5) -> float:
    for _ in range(warmup):
        opt.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        opt.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def bench(dev: str, params_m: int, window: int, bucket_mb: float):
    numel = params_m * 2**20
    print(f"\nBENCH  device={dev}  params={params_m}M fp32 "
          f"({numel * 4 / 2**30:.2f} GiB)  window={window} "
          f"bucket_mb={bucket_mb:g}")
    if torch.cuda.is_available():
        torch.cuda.set_device(torch.device(dev))

    contenders = [
        ("OffloadAdamW (fp32 state)", "pinned", 28,
         lambda ps: OffloadAdamW(ps, lr=1e-3, device=dev, window=window,
                                 bucket_mb=bucket_mb)),
        ("OffloadAdamW (bf16 state)", "pinned", 20,
         lambda ps: OffloadAdamW(ps, lr=1e-3, device=dev, window=window,
                                 bucket_mb=bucket_mb,
                                 state_dtype=torch.bfloat16,
                                 stochastic_rounding=True)),
        ("torch AdamW CPU (foreach)", "cpu", 0,
         lambda ps: torch.optim.AdamW(ps, lr=1e-3, foreach=True)),
        ("torch AdamW CPU (fused)", "pinned", 0,
         lambda ps: torch.optim.AdamW(ps, lr=1e-3, fused=True)),
        ("legacy ramtorch.AdamW", "pinned", 0,
         lambda ps: LegacyAdamW(ps, lr=1e-3)),
        ("torch AdamW GPU-resident (foreach)", "gpu", 0,
         lambda ps: torch.optim.AdamW(ps, lr=1e-3, foreach=True)),
        ("torch AdamW GPU-resident (fused)", "gpu", 0,
         lambda ps: torch.optim.AdamW(ps, lr=1e-3, fused=True)),
    ]

    rows = []
    for name, where, bytes_per_elem, ctor in contenders:
        params = _make_params(numel, where, dev)
        try:
            opt = ctor(params)
            ms = _time_steps(opt) * 1e3
            gbs = (bytes_per_elem * numel / (ms / 1e3) / 1e9
                   if bytes_per_elem else None)
            rows.append((name, ms, gbs))
        except Exception as e:  # noqa: BLE001 — e.g. fused CPU unsupported
            rows.append((name, None, None))
            print(f"  {name}: skipped ({type(e).__name__}: {e})")
        finally:
            del params
            try:
                del opt
            except NameError:
                pass
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print(f"\n  {'optimizer':<36} {'ms/step':>10} {'eff PCIe GB/s':>14}")
    for name, ms, gbs in rows:
        ms_s = f"{ms:8.1f}" if ms is not None else "  n/a  "
        gbs_s = f"{gbs:10.1f}" if gbs is not None else "       -"
        print(f"  {name:<36} {ms_s:>10} {gbs_s:>14}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--device",
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--pin", type=int, default=3)
    ap.add_argument("--bench-params-m", type=int, default=128,
                    help="benchmark size in millions of fp32 params")
    ap.add_argument("--bucket-mb", type=float, default=32.0)
    ap.add_argument("--no-bench", action="store_true")
    args = ap.parse_args()

    ok = check_parity(args.device, args.steps, args.window, args.pin)
    if torch.cuda.is_available() and not args.no_bench:
        bench(args.device, args.bench_params_m, args.window, args.bucket_mb)

    print("\nOFFLOAD OPTIMIZER CHECK OK" if ok else "\nMISMATCH — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
