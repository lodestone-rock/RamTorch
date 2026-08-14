"""
mnist_pipeline_offload.py
-------------------------
End-to-end pipeline parallelism WITH weight offloading: each pipeline stage's
model is a list of chunks whose master weights live in CPU pinned RAM and
stream through a small GPU window, prefetched in the schedule's exact chunk
order (F -> chunks 0..L-1, B -> L-1..0; the staggered_1b1f F<->B turnaround
gets echo reuse for free).

The point: GPU weight memory per stage drops from ALL chunks to
~(window + pin) chunks, letting a pipeline train models whose stage shards
don't fit in VRAM — while the schedule's bubbles and compute hide most of the
PCIe traffic (validated in ramtorch/pipeline_offload_simulator.py; measured
here for real).

What it shows, end to end:
  1. Build ONE flat list of chunk modules and hand it to
     Pipeline(chunk_modules=...) — the pipeline splits it across the devices
     (evenly, or per chunks_per_stage=[...]) into offloaded stages. The
     baseline uses the SAME flat list with offload=False (resident stages).
  2. Train with pipe.step() + flush_grads() + one AdamW(fused=True): torch
     groups params by device, so CPU masters get the fused CPU kernel and any
     GPU-pinned chunks the fused CUDA kernel.
  3. Report per-device peak GPU memory, engine H2D load counts and
     acquire-stall time vs a full-resident baseline of the same model.
  4. --profile writes a Perfetto/kineto trace where the loader's H2D copies
     overlap the compute stream.

The model is deliberately wider than MNIST needs (default ~135M params:
16 residual MLP blocks of dim 1024) so the weight traffic is visible.

Honest expectations: this MLP is a WORST case for streaming — it has very
little compute per weight byte, so the run is transfer-bound (the whole
model crosses PCIe ~2x(m-ish) times per step for weights + once for grads)
and the offloaded step is several times slower than full-resident despite
acquire stalls being tiny (~15 ms/step: the prefetcher does overlap; there
is simply more traffic than compute). Measured here (2x ~100GB/s-class
GPUs over PCIe, m=8): pin=0 -> 709 ms/step at 4.5x less peak GPU memory vs
79 ms full-resident; pin=5 -> 354 ms/step at 1.4x less. Loss curves and
test accuracy are IDENTICAL to full-resident in all cases (bit-parity is
asserted in examples/pipeline_offload_check.py). For models with real
arithmetic intensity (transformers at useful sequence lengths) the compute
hides the traffic — see ramtorch/pipeline_offload_simulator.py for the
regime map. Use --pin to trade GPU memory back for traffic when a stage
almost fits.

Requirements: 2+ GPUs, torchvision. Run:
    PYTHONPATH=. python examples/mnist_pipeline_offload.py
    PYTHONPATH=. python examples/mnist_pipeline_offload.py \
        --devices cuda:1 cuda:3 --steps 200 --window 2 --pin 0 \
        --keep-activations checkpoint --profile trace.json
"""

from __future__ import annotations

import argparse
import itertools
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ramtorch import Pipeline
from ramtorch.pipeline_offload import OffloadStage


class Block(nn.Module):
    """Residual MLP block (LayerNorm — buffer-free, offload-safe)."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, 4 * dim)
        self.fc2 = nn.Linear(4 * dim, dim)

    def forward(self, x):
        return x + self.fc2(F.gelu(self.fc1(self.norm(x))))


class Head(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.fc = nn.Linear(28 * 28, dim)

    def forward(self, x):
        return self.fc(x.view(x.size(0), -1))


def build_chunks(dim: int, blocks: int, seed: int):
    """ONE flat chunk list for the whole model: Head, B, ..., B, classifier.
    The Pipeline splits it across the devices itself (chunk_modules=)."""
    torch.manual_seed(seed)
    return [Head(dim)] + [Block(dim) for _ in range(blocks)] \
        + [nn.Linear(dim, 10)]


def gpu_peaks(devices):
    return [torch.cuda.max_memory_allocated(d) / 2**20 for d in devices]


def reset_peaks(devices):
    for d in devices:
        torch.cuda.reset_peak_memory_stats(d)


def train(pipe, loader, args, label, profile_path=None):
    loss_fn = nn.CrossEntropyLoss()
    # One optimizer over all stages: offloaded stages contribute CPU-pinned
    # masters, resident stages GPU params. fused=True gives each device group
    # its fused kernel (CPU masters included).
    params = itertools.chain(*(st.params for st in pipe.stages))
    opt = torch.optim.AdamW(params, lr=args.lr, fused=True)
    reset_peaks(args.devices)

    step = 0
    running = 0.0
    losses = []
    t_start = time.perf_counter()
    done = False
    while not done:
        for x, y in loader:
            prof = profile_path if (profile_path and step == 10) else None
            res = pipe.step(x, targets=y, schedule=args.schedule,
                            n_microbatches=args.micro, loss_fn=loss_fn,
                            profile_path=prof)
            res.flush_grads()
            opt.step()
            opt.zero_grad(set_to_none=True)
            running += res.loss.item()
            losses.append(res.loss.item())
            step += 1
            if step % 50 == 0:
                print(f"  [{label}] step {step:>4}: loss={running / 50:.4f}")
                running = 0.0
            if prof:
                print(f"  [{label}] wrote profiler trace -> {prof}")
            if step >= args.steps:
                done = True
                break
    for d in args.devices:
        torch.cuda.synchronize(d)
    dt = time.perf_counter() - t_start

    report = {
        "time_per_step_ms": dt / step * 1e3,
        "peak_mib": gpu_peaks(args.devices),
        "final_loss": sum(losses[-50:]) / len(losses[-50:]),
    }
    for s, st in enumerate(pipe.stages):
        if isinstance(st, OffloadStage):
            stats = st.engine.stats
            report[f"stage{s}"] = (
                f"H2D loads={stats['loads']} "
                f"({stats['loads'] / step:.1f}/step), "
                f"acquire stall={stats['acquire_wait_s']:.2f}s total "
                f"({stats['acquire_wait_s'] / step * 1e3:.1f} ms/step)"
            )
    return report


def evaluate(pipe, loader, micro):
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = pipe.infer(x, n_microbatches=micro).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--micro", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--blocks", type=int, default=16,
                    help="residual blocks total (split across stages)")
    ap.add_argument("--window", type=int, default=2,
                    help="streamed GPU chunk slots per stage")
    ap.add_argument("--pin", type=int, default=0,
                    help="chunks pinned permanently on each stage's GPU")
    ap.add_argument("--keep-activations", default="keep",
                    choices=["keep", "checkpoint"])
    ap.add_argument("--schedule", default="staggered_1b1f",
                    choices=["gpipe", "staggered_1b1f"])
    ap.add_argument("--devices", nargs="+", default=["cuda:1", "cuda:3"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", type=str, default="",
                    help="write a Perfetto/kineto trace of step 10 here")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="skip the full-resident comparison run")
    args = ap.parse_args()

    if torch.cuda.device_count() < 1:
        raise SystemExit("this example needs CUDA")
    n_stages = len(args.devices)
    keep = True if args.keep_activations == "keep" else "checkpoint"

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=tf),
        batch_size=args.batch_size, shuffle=True, num_workers=2,
        drop_last=True,
    )
    test_loader = DataLoader(
        datasets.MNIST("./data", train=False, download=True, transform=tf),
        batch_size=512, shuffle=False, num_workers=2,
    )

    # ── offloaded pipeline: ONE flat chunk list, split across the GPUs ──────
    chunks = build_chunks(args.dim, args.blocks, args.seed)
    n_params = sum(p.numel() for m in chunks for p in m.parameters())
    print(f"model: {n_params / 1e6:.0f}M params, {len(chunks)} chunks over "
          f"devices {args.devices}")
    print(f"offload: window={args.window} pin={args.pin} mode={keep} "
          f"-> ~{args.window + args.pin} chunks of weights on each GPU")

    pipe = Pipeline(
        chunk_modules=chunks,
        devices=args.devices,
        offload_window=args.window,
        offload_pin=args.pin,
        offload_keep_activations=keep,
    )
    print(f"  split: {[len(st.engine.chunks) for st in pipe.stages]} "
          "chunks per stage (even; pass chunks_per_stage=[...] to weight)")
    off = train(pipe, train_loader, args, "offloaded",
                profile_path=args.profile or None)
    acc = evaluate(pipe, test_loader, args.micro)
    pipe.close()

    print(f"\n[offloaded] {off['time_per_step_ms']:.0f} ms/step, "
          f"final loss {off['final_loss']:.4f}, test acc {acc:.4f}")
    for d, p in zip(args.devices, off["peak_mib"]):
        print(f"  {d}: peak {p:.0f} MiB")
    for s in range(n_stages):
        if f"stage{s}" in off:
            print(f"  stage {s}: {off[f'stage{s}']}")

    if args.skip_baseline:
        return 0

    # ── full-resident baseline: SAME flat chunk list, offload=False ─────────
    ref = Pipeline(chunk_modules=build_chunks(args.dim, args.blocks,
                                              args.seed),
                   devices=args.devices, offload=False)
    base = train(ref, train_loader, args, "full-resident")
    ref_acc = evaluate(ref, test_loader, args.micro)
    ref.close()

    print(f"\n[full-resident] {base['time_per_step_ms']:.0f} ms/step, "
          f"final loss {base['final_loss']:.4f}, test acc {ref_acc:.4f}")
    for d, p in zip(args.devices, base["peak_mib"]):
        print(f"  {d}: peak {p:.0f} MiB")

    print("\n── comparison ──────────────────────────────────────────────")
    slow = off["time_per_step_ms"] / base["time_per_step_ms"]
    print(f"  step time : {off['time_per_step_ms']:.0f} vs "
          f"{base['time_per_step_ms']:.0f} ms  "
          f"({(slow - 1) * 100:+.1f}% offloaded)")
    for i, d in enumerate(args.devices):
        ratio = base["peak_mib"][i] / max(off["peak_mib"][i], 1e-9)
        print(f"  {d} peak  : {off['peak_mib'][i]:.0f} vs "
              f"{base['peak_mib'][i]:.0f} MiB  ({ratio:.1f}x smaller)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
