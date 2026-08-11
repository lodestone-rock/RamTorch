"""
offload_quickstart.py
---------------------
The minimal runnable intro to :class:`ramtorch.OffloadModel` — single-GPU
CPU→GPU weight streaming with a sliding window + pinned layers.

You dice your model into chunk modules (the same convention as
``Pipeline(stage_modules=...)``); ``OffloadModel`` streams their weights from
CPU pinned memory through a small GPU window while a loader thread prefetches
ahead. Peak GPU weight memory ≈ ``(window + pin)`` chunks, no matter how many
chunks the model has.

This script:
  1. builds a chunked model and wraps it in OffloadModel,
  2. runs streamed inference,
  3. runs a short training loop (step + flush_grads + optimizer) with
     ``torch.optim.AdamW(fused=True)`` — the recommended optimizer: torch
     groups params by device, so the streamed CPU masters get the fused CPU
     kernel (one multithreaded pass at DDR bandwidth) and the pinned GPU
     chunks get the fused CUDA kernel,
  4. verifies the streamed model matches a plain full-resident reference
     (loss + final weights after several AdamW steps).

Numerics note: the streamed model and the reference compute the SAME math, so
on a single deterministic device (e.g. cuda:0) they are bit-identical. Two
caveats this script handles: (a) across different GPUs cuBLAS may pick
different reduction orders (pure fp noise, reported against a tolerance);
(b) the fused CPU and fused CUDA AdamW kernels round differently, so the
reference mirrors the offload's per-layer optimizer device placement
(``MirroredFusedAdamW``) to keep the trajectories comparable bit-for-bit.

Run:
    PYTHONPATH=. python examples/offload_quickstart.py
    PYTHONPATH=. python examples/offload_quickstart.py --device cuda:0 --steps 20
"""

import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from ramtorch import OffloadModel

DIM = 256
N_CHUNKS = 12
BATCH = 32


def make_block(d: int) -> nn.Module:
    """One chunk: a small feed-forward block. You decide the granularity."""
    return nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))


class MirroredFusedAdamW:
    """Reference-side optimizer that mirrors OffloadModel's device placement.

    The streamed model's optimizer runs the fused CPU kernel on the CPU
    masters and the fused CUDA kernel on the pinned chunks. Those two kernels
    round differently (~1e-7 per step), so a plain all-GPU AdamW reference
    would slowly drift. This mirror steps the reference's streamed layers
    with the SAME fused CPU kernel — on CPU master copies, via the same data
    path the offload uses (grad D2H, CPU step, weights H2D; every copy is
    bit-exact) — and the pinned layers with the fused CUDA kernel in place.
    Result: bit-identical trajectories. (Only needed for this comparison;
    real training just uses one AdamW(fused=True) over model.parameters().)
    """

    def __init__(self, layers: nn.Sequential, pinned_layers, lr: float):
        self._mirror = []  # (gpu_param, cpu_master)
        cpu_masters, gpu_params = [], []
        for i, layer in enumerate(layers):
            for p in layer.parameters():
                if i in pinned_layers:
                    gpu_params.append(p)
                else:
                    m = nn.Parameter(p.detach().cpu())
                    self._mirror.append((p, m))
                    cpu_masters.append(m)
        self._opts = [
            torch.optim.AdamW(ps, lr=lr, fused=True)
            for ps in (cpu_masters, gpu_params) if ps
        ]

    @torch.no_grad()
    def step(self):
        for p, m in self._mirror:
            m.grad = p.grad.detach().cpu()   # grad D2H, like the writeback
        for opt in self._opts:
            opt.step()
        for p, m in self._mirror:
            p.copy_(m)                       # weights H2D, like the loader

    def zero_grad(self, set_to_none: bool = True):
        for opt in self._opts:
            opt.zero_grad(set_to_none=set_to_none)
        # the mirrored GPU params belong to no inner optimizer — clear them
        # too, or backward() would accumulate across steps
        for p, _ in self._mirror:
            p.grad = None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--pin", type=int, default=3)
    args = ap.parse_args()
    dev = args.device

    print(f"device={dev}  chunks={N_CHUNKS}  dim={DIM}  batch={BATCH}  "
          f"window={args.window}  pin={args.pin}")

    # ── 1. Dice + wrap ────────────────────────────────────────────────────
    torch.manual_seed(0)
    chunks = [make_block(DIM) for _ in range(N_CHUNKS)]
    # Deepcopy the reference BEFORE constructing OffloadModel: the constructor
    # relocates chunk params in place (pinned -> GPU, streamed -> CPU pinned
    # memory), and deep-copying a pinned-memory tensor can diverge in the last
    # ULP from the GPU copy. Copy the pristine CPU chunks for an exact reference.
    reference = nn.Sequential(*copy.deepcopy(chunks)).to(dev)
    model = OffloadModel(
        chunks,
        device=dev,
        window=args.window,          # streaming slots (>=2 overlaps load/compute)
        pin=args.pin,                # evenly-spaced chunks pinned resident
        keep_activations=True,       # skip recompute in backward (ok: no dropout)
    )
    print(f"pinned layers: {sorted(model.pinned_layers)}  "
          f"(peak GPU weight memory ~ {args.window + args.pin} chunks)")

    # ── 2. Streamed inference ─────────────────────────────────────────────
    gen = torch.Generator().manual_seed(1)
    x = torch.randn(BATCH, DIM, generator=gen)
    with torch.no_grad():
        out = model(x)
        ref_out = reference(x.to(dev))
    inf_diff = (out.cpu() - ref_out.cpu()).abs().max().item()
    print(f"inference: out{tuple(out.shape)}  "
          f"max|diff| vs reference = {inf_diff:.3e}  "
          f"(bit-identical: {inf_diff == 0.0})")

    # ── 3. Training loop ──────────────────────────────────────────────────
    # One torch.optim.AdamW(fused=True) covers the mixed devices: torch
    # groups params by device, so the streamed (CPU) masters get the fused
    # CPU kernel — a single multithreaded pass at DDR bandwidth, cheaper
    # than any PCIe round trip — and the pinned (GPU) chunks get the fused
    # CUDA kernel.
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)

    # Reference side: the fused CPU and CUDA kernels round differently, so a
    # plain all-GPU AdamW would drift ~1e-7 from the streamed trajectory.
    # Mirror the offload placement instead — step each layer with the SAME
    # kernel the streamed model uses — and the comparison stays bit-exact.
    opt_ref = MirroredFusedAdamW(reference, model.pinned_layers, lr=1e-3)

    # ── 4. Trajectory match over multiple steps ───────────────────────────
    # The streamed model and the reference compute the same math; track the
    # worst abs diff over the run. On a single deterministic device (cuda:0)
    # they are bit-identical (0.0). Across GPUs cuBLAS may pick different
    # reduction orders, so tiny fp noise appears; MSE *squares* the activation
    # diff, so the loss tolerance is looser than the (tight) weight tolerance.
    LOSS_TOL = 1e-2    # loss is a squared/reduced quantity — keep this loose
    WEIGHT_TOL = 1e-4  # weights are the real correctness signal — keep tight
    worst = 0.0
    for step in range(args.steps):
        xb = torch.randn(BATCH, DIM, generator=gen)
        yb = torch.randn(BATCH, DIM, generator=gen)

        # streamed step
        res = model.step(xb, targets=yb, loss_fn=F.mse_loss)
        model.flush_grads()          # accumulated grads -> .grad
        opt.step()
        model.zero_grad_acc()        # reset accumulators (or opt.zero_grad + flush)

        # reference step
        opt_ref.zero_grad(set_to_none=True)
        ref_loss = F.mse_loss(reference(xb.to(dev)), yb.to(dev))
        ref_loss.backward()
        opt_ref.step()

        d = (res.loss.cpu() - ref_loss.cpu()).abs().item()
        worst = max(worst, d)
        if step % max(1, args.steps // 5) == 0 or step == args.steps - 1:
            print(f"  step {step:3d}  loss={float(res.loss):.6f}  "
                  f"|loss diff|={d:.3e}")

    # Final weight trajectory comparison across every chunk.
    weight_diff = max(
        (dict(model.chunks[i].named_parameters())[name].detach().cpu()
         - p.detach().cpu()).abs().max().item()
        for i in range(N_CHUNKS)
        for name, p in dict(reference[i].named_parameters()).items()
    )
    print(f"\nworst |loss diff| over run:      {worst:.3e}  (tol {LOSS_TOL:.0e})")
    print(f"max |weight diff| over all chunks: {weight_diff:.3e}  "
          f"(tol {WEIGHT_TOL:.0e})")
    print(f"total stall (acquire_wait): {model.stats['acquire_wait_s']*1e3:.1f} ms  "
          f"over {model.stats['loads']} chunk loads")

    model.close()  # join the loader/writeback threads
    ok = worst < LOSS_TOL and weight_diff < WEIGHT_TOL
    print("\nOFFLOAD QUICKSTART OK" if ok else "\nMISMATCH — see above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
