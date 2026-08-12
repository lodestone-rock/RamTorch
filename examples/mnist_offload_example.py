"""
mnist_offload_example.py
------------------------
A complete, self-contained reference for single-GPU weight streaming.

This is the canonical "how do I train with OffloadModel?" example. It trains
an MLP on MNIST with the model's weights living in CPU pinned memory and
streaming through a small GPU window — one GPU, one process, no torchrun.

What it shows (the whole workflow, end to end):
  1. Dice a model into an ordered list of chunk modules (you choose the
     granularity — same convention as ``Pipeline(stage_modules=...)``).
  2. Wrap the chunks in OffloadModel (window / pin / keep_activations knobs).
  3. Train with model.step() + flush_grads() + torch.optim.AdamW(fused=True)
     — the recommended optimizer: torch groups params by device, so the
     streamed CPU masters get the fused CPU kernel (one multithreaded pass
     at DDR bandwidth) and the pinned GPU chunks get the fused CUDA kernel.
  4. Evaluate with a streamed forward (arbitrary batch size).
  5. Save / load a standard state_dict checkpoint.
  6. Report peak GPU memory vs the full-resident weight footprint, plus the
     streaming stats (loads / stall time) that tell you whether you are
     transfer-bound.
  7. Optionally push some chunks one tier further down with --nvme K
     --nvme-path FILE: their masters move from CPU RAM into a scratch file
     (mmap-backed, interleaved placement) and stream disk -> pinned staging
     -> GPU. Training works unchanged — the optimizer updates the mapped
     masters in place. Saves host RAM for those chunks at the cost of
     slower loads (see docs/offload.md, "The NVMe tier").

The MLP here is deliberately deeper/wider than MNIST needs (default: 12
FF blocks of Linear(512, 2048) -> GELU -> Linear(2048, 512)) so the streaming
actually has something to do; scale --hidden/--blocks up to see the memory
gap grow while accuracy stays the same.

Backward strategy is a flag: --backward {recompute,keep,checkpoint} picks
keep_activations, and --selective marks each block's heavy FF with
ramtorch.offload_checkpoint inside the block's own forward (implies keep:
norm/residual activations stay live, only the marked 4x expansion is
recomputed at backward). All choices train to the same accuracy — they trade
activation memory vs recompute time; see offload_checkpoint_study.py.

Requirements: 1 GPU (any), torchvision for MNIST. Run:
    python examples/mnist_offload_example.py
    python examples/mnist_offload_example.py --device cuda:1 --steps 500 \
        --hidden 1024 --blocks 24 --window 2 --pin 4
    python examples/mnist_offload_example.py --device cuda:1 --selective
    # NVMe tier: 6 chunks' masters on disk (put the file on a real drive,
    # NOT /tmp — that is usually tmpfs, i.e. RAM)
    python examples/mnist_offload_example.py --nvme 6 \
        --nvme-path ./mnist_nvme_masters.bin
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ramtorch import OffloadModel, offload_checkpoint


# ── 1. Dice the model into chunks ─────────────────────────────────────────────
# Chunks exchange a SINGLE tensor: chunk i+1 consumes chunk i's output. The
# first chunk owns the flatten + input projection, the last chunk owns the
# classifier head, and everything in between is a pre-norm residual FF block
# (a plain 12-deep MLP without residuals diverges — nothing offload-specific,
# depth just needs skips). Residuals are fine because they live INSIDE a
# chunk. No dropout/BatchNorm on purpose: recompute mode would resample
# dropout (use keep_activations=True or "checkpoint" for stochastic
# chunks), and buffer
# mutations (BatchNorm running stats) are not written back — LayerNorm is
# buffer-free and safe.

class Block(nn.Module):
    """x + FF(LayerNorm(x)) — one streamable chunk.

    With ``selective=True`` the heavy 4x expansion is marked with
    ``offload_checkpoint`` in the forward itself: its internal activations
    are dropped and recomputed at backward, while the norm output and the
    residual stay live. (A bare ``torch.utils.checkpoint`` would break under
    the streaming engine — the helper re-applies the streamed GPU weights
    for the recompute.) Run with keep_activations=True.
    """

    def __init__(self, hidden: int, selective: bool = False):
        super().__init__()
        self.selective = selective
        self.norm = nn.LayerNorm(hidden)
        self.ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(),
            nn.Linear(4 * hidden, hidden),
        )

    def forward(self, x):
        h = self.norm(x)
        if self.selective:
            return x + offload_checkpoint(self.ff, h)
        return x + self.ff(h)


def build_chunks(hidden: int, blocks: int,
                 selective: bool = False) -> list[nn.Module]:
    head = nn.Sequential(nn.Flatten(), nn.Linear(28 * 28, hidden), nn.GELU())
    tail = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 10))
    return ([head] + [Block(hidden, selective) for _ in range(blocks)]
            + [tail])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--blocks", type=int, default=12, help="FF chunks between head and tail")
    ap.add_argument("--window", type=int, default=2, help="GPU streaming slots")
    ap.add_argument("--pin", type=int, default=2, help="chunks pinned resident on the GPU")
    ap.add_argument("--nvme", type=int, default=0,
                    help="chunks whose masters live on disk instead of CPU "
                         "RAM (interleaved placement; needs --nvme-path)")
    ap.add_argument("--nvme-path", type=str, default="",
                    help="scratch file for the NVMe-tier masters — put it on "
                         "an actual drive, /tmp is usually tmpfs (RAM). "
                         "Deleted on close")
    ap.add_argument("--backward", default="recompute",
                    choices=["recompute", "keep", "checkpoint"],
                    help="keep_activations strategy: engine recompute "
                         "(cheapest memory), keep (fastest, all activations "
                         "live), or torch-checkpoint per chunk (recompute "
                         "memory, dropout-safe)")
    ap.add_argument("--selective", action="store_true",
                    help="mark each block's heavy FF with offload_checkpoint "
                         "in its own forward (implies --backward keep): "
                         "marked parts recompute, the rest keeps activations")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="", help="optional checkpoint path")
    args = ap.parse_args()

    if args.selective and args.backward != "keep":
        print(f"--selective implies --backward keep (was {args.backward})")
        args.backward = "keep"
    keep_activations = {"recompute": False, "keep": True,
                        "checkpoint": "checkpoint"}[args.backward]
    if args.nvme > 0 and not args.nvme_path:
        default = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "_mnist_nvme_masters.bin",
        )
        print(f"--nvme without --nvme-path: using {default}")
        args.nvme_path = default
    if args.nvme > 0 and os.environ.get("RAMTORCH_NVME_ACKNOWLEDGE") != "1":
        ap.error(
            "--nvme is locked: training with on-disk masters rewrites them "
            "every optimizer step and can wear out an SSD. If you accept "
            "responsibility for drive wear, re-run with "
            "RAMTORCH_NVME_ACKNOWLEDGE=1 (see docs/offload.md, "
            "'Drive-endurance caution')."
        )

    torch.manual_seed(args.seed)
    dev = torch.device(args.device)
    is_cuda = dev.type == "cuda"

    # ── Data ──────────────────────────────────────────────────────────────────
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=tf),
        batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True,
    )
    test_loader = DataLoader(
        datasets.MNIST("./data", train=False, download=True, transform=tf),
        batch_size=512, shuffle=False, num_workers=2,
    )

    # ── 2. Wrap the chunks in OffloadModel ────────────────────────────────────
    chunks = build_chunks(args.hidden, args.blocks, selective=args.selective)
    n_params = sum(p.numel() for c in chunks for p in c.parameters())
    model = OffloadModel(
        chunks,
        device=dev,
        window=args.window,
        pin=args.pin,
        nvme=args.nvme,
        nvme_path=args.nvme_path or None,
        keep_activations=keep_activations,
    )
    print(f"model: {len(chunks)} chunks, {n_params / 1e6:.1f}M params "
          f"({n_params * 4 / 2**20:.0f} MiB fp32 full-resident)")
    print(f"offload: device={dev}  window={args.window}  pin={args.pin} "
          f"(pinned layers {sorted(model.pinned_layers)})  "
          f"backward={args.backward}"
          f"{' + selective FF marks' if args.selective else ''}")
    print(f"  -> peak GPU weight memory ~ {args.window + args.pin} of "
          f"{len(chunks)} chunks")
    if model.nvme_layers:
        nvme_bytes = sum(
            p.numel() * p.element_size()
            for i in model.nvme_layers
            for p in model.chunks[i].parameters()
        )
        print(f"nvme tier: {len(model.nvme_layers)} chunks "
              f"(layers {sorted(model.nvme_layers)}) — "
              f"{nvme_bytes / 2**20:.0f} MiB of masters moved from CPU RAM "
              f"to {args.nvme_path}")

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(dev)

    # ── 3. Train: step + flush_grads + fused AdamW ────────────────────────────
    # One AdamW(fused=True) covers the mixed devices (CPU masters + pinned GPU
    # chunks). flush_grads() writes accumulated grads into persistent pinned
    # .grad buffers AND invalidates the streamed GPU weight cache so the next
    # step sees the updated masters.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, fused=True)

    model.train()
    step, running = 0, 0.0
    done = False
    while not done:
        for x, y in train_loader:
            res = model.step(x, targets=y, loss_fn=F.cross_entropy)
            model.flush_grads()
            opt.step()
            model.zero_grad_acc()

            running += res.loss.item()
            step += 1
            if step % 50 == 0:
                print(f"  step {step:>4}: loss={running / 50:.4f}")
                running = 0.0
            if step >= args.steps:
                done = True
                break

    # ── 4. Evaluate — streamed forward, any batch size ────────────────────────
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = model(x).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"\ntest accuracy after {step} steps: {correct / total:.4f}")

    # ── 6. Memory + streaming report ──────────────────────────────────────────
    if is_cuda:
        peak = torch.cuda.max_memory_allocated(dev)
        # Full-resident fp32 training keeps weights + grads + Adam m/v on the
        # GPU (4 tensors the size of the model) plus activations. Here only
        # window+pin weight chunks are resident and grads + Adam state live
        # in CPU RAM.
        full = 4 * n_params * 4
        print(f"peak GPU memory: {peak / 2**20:.0f} MiB incl. activations "
              f"(full-resident training needs ~{full / 2**20:.0f} MiB for "
              f"weights+grads+Adam state before activations)")
    nvme_part = (f" ({model.stats['nvme_loads']} from the nvme tier)"
                 if model.nvme_layers else "")
    print(f"streaming: {model.stats['loads']} chunk loads{nvme_part}, "
          f"{model.stats['acquire_wait_s'] * 1e3:.0f} ms total stall "
          f"(large stall = transfer-bound; try more --pin, a bigger model, "
          f"or check with the offload_simulator)")

    # ── 5. Checkpoint — standard state_dict, chunks are model.chunks[i] ───────
    if args.save:
        torch.save(model.state_dict(), args.save)
        print(f"saved checkpoint -> {args.save}")
        # Reload later:
        #   m = OffloadModel(build_chunks(args.hidden, args.blocks), ...)
        #   m.load_state_dict(torch.load(args.save))

    model.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
