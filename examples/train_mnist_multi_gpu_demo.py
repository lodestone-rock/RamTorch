"""train_mnist_multi_gpu_demo.py — Dummy MNIST trainer using MultiGPUWrapper.

Demonstrates how the new wrapper collapses all the multi-GPU boilerplate
(model lists, NCCL calls, ThreadPoolExecutor, per-GPU optimizer loops) into
something that reads almost identically to a plain single-GPU training loop.

Run:
    python train_mnist_multi_gpu_demo.py
    python train_mnist_multi_gpu_demo.py --epochs 3 --batch-size 256 --accum 2
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR
from torch.profiler import ProfilerActivity, profile, record_function, schedule
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from ramtorch.multi_gpu import MultiGPUWrapper

torch.manual_seed(0)


# ---------------------------------------------------------------------------
# Profiler helper
# ---------------------------------------------------------------------------


def make_profiler_ctx(args, trace_path: str):
    """Return a torch.profiler.profile context (or nullcontext if --profile is off).

    The profiler is driven by a schedule so it only records between
    ``--profile-start`` and ``--profile-stop`` (global step numbers).

        wait   = profile_start          # idle for this many steps
        warmup = 1                      # one warm-up step (traces discarded)
        active = profile_stop - profile_start  # steps actually recorded
        repeat = 1                      # run the schedule once then stop

    Example: --profile-start 20 --profile-stop 23
        steps 0-19  : profiler idle
        step  20    : warmup  (CUDA kernels launched but trace discarded)
        steps 21-23 : active  (3 steps recorded)
        step  24+   : profiler stopped
    """
    if not args.profile:
        return nullcontext()

    start = args.profile_start
    stop = args.profile_stop
    if stop <= start:
        raise ValueError(
            f"--profile-stop ({stop}) must be greater than --profile-start ({start})"
        )

    return profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=schedule(wait=start, warmup=1, active=stop - start, repeat=1),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
        on_trace_ready=lambda p: (
            print(p.key_averages().table(sort_by="cuda_time_total", row_limit=20)),
            p.export_chrome_trace(trace_path),
            print(f"[profiler] Chrome trace saved to {trace_path}"),
        ),
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class MNISTNet(nn.Module):
    """Tiny CNN — nothing fancy, just enough to show the wrapper pattern."""

    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(7),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.conv(x))


# ---------------------------------------------------------------------------
# Per-GPU callables
# ---------------------------------------------------------------------------


def forward_fn(
    gpu_id: int,
    model: MNISTNet,
    images: torch.Tensor,
    labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward only — returns (logits, labels_on_device) for use in backward."""
    device = f"cuda:{gpu_id}"
    logits = model(images.to(device))
    return logits, labels.to(device)


def backward_fn(
    gpu_id: int,
    model: MNISTNet,
    output: tuple[torch.Tensor, torch.Tensor],
    accum_steps: int = 1,
) -> float:
    """Backward only — receives the output from forward_fn."""
    logits, labels = output
    loss = F.cross_entropy(logits, labels)
    (loss / accum_steps).backward()
    return loss.item()


def forward_backward(
    gpu_id: int,
    model: MNISTNet,
    images: torch.Tensor,
    labels: torch.Tensor,
    accum_steps: int = 1,
) -> float:
    """Combined forward + backward — used by the fast-path wrapper.step()."""
    logits, labels_d = forward_fn(gpu_id, model, images, labels)
    return backward_fn(gpu_id, model, (logits, labels_d), accum_steps=accum_steps)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(args):
    n_gpus = torch.cuda.device_count()
    print(f"Found {n_gpus} GPU(s)")

    # ------------------------------------------------------------------
    # Dataset — plain torchvision MNIST
    # ------------------------------------------------------------------
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_dataset = datasets.MNIST(
        "./data", train=True, download=True, transform=transform
    )
    val_dataset = datasets.MNIST(
        "./data", train=False, download=True, transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size * n_gpus,  # full batch; wrapper splits it
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False, num_workers=2)

    # ------------------------------------------------------------------
    # Build the wrapper — this is the only multi-GPU-specific code you write
    # ------------------------------------------------------------------
    wrapper = MultiGPUWrapper(
        model_factory=lambda: MNISTNet(),
        optimizer_factory=lambda params: AdamW(params, lr=args.lr, weight_decay=1e-4),
        forward_backward_fn=forward_backward,
        scheduler_factory=lambda opt: LinearLR(
            opt, start_factor=0.1, end_factor=1.0, total_iters=args.warmup
        ),
        gradient_accumulation_steps=args.accum,
        max_grad_norm=1.0,
    )
    wrapper.setup()
    wrapper.save_checkpoint("mnist_demo_untrained.safetensors")
    # ------------------------------------------------------------------
    # Training loop — looks like single-GPU
    # ------------------------------------------------------------------
    global_step = 0

    with make_profiler_ctx(args, "mnist_demo_trace.json") as prof:
        for epoch in range(1, args.epochs + 1):
            for m in wrapper.models:  # set all replicas to train mode
                m.train()
            pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

            for i, (images, labels) in enumerate(pbar):
                # split_batch returns list of (img_chunk, lbl_chunk) tuples
                chunks = wrapper.split_batch(images, labels)
                # chunks[gpu_id] == (images[s:e], labels[s:e])

                with record_function("step"):
                    loss = wrapper.step(
                        chunks,
                        accum_steps=args.accum,  # forwarded as **kwargs to forward_backward
                    )

                lr = wrapper.last_lr
                pbar.set_postfix(loss=f"{loss:.4f}", lr=f"{lr:.2e}", step=global_step)
                global_step += 1

                if prof is not None:
                    prof.step()

            # ------------------------------------------------------------------
            # Validation — forward(eval_mode=True) runs all replicas concurrently
            # in eval + no_grad, then restores train mode automatically.
            # ------------------------------------------------------------------
            correct = total = 0
            for images, labels in val_loader:
                # Split across GPUs just like training
                chunks = wrapper.split_batch(images, labels)
                outputs = wrapper.forward(chunks, forward_fn=forward_fn, eval_mode=True)
                # outputs[gpu_id] == (logits, labels_on_device)
                for logits, labels_d in outputs:
                    correct += (logits.argmax(1) == labels_d).sum().item()
                    total += labels_d.size(0)

            acc = correct / total
            print(f"  [epoch {epoch}] val accuracy: {acc:.4f}  lr: {wrapper.last_lr:.2e}")

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    wrapper.save_checkpoint("mnist_demo.safetensors")
    wrapper.cleanup()
    print("Done.")


# ---------------------------------------------------------------------------
# Advanced demo — manual accumulation loop (for when you need more control)
# ---------------------------------------------------------------------------


def train_manual_accum(args):
    """Same training but with the low-level primitives instead of wrapper.step().

    Shows how to reach into the wrapper when you need custom logic between
    the backward and the optimizer step (e.g. logging per-micro-step loss,
    custom grad scaling, EMA updates, etc.).
    """
    n_gpus = torch.cuda.device_count()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size * n_gpus,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    wrapper = MultiGPUWrapper(
        model_factory=lambda: MNISTNet(),
        optimizer_factory=lambda params: AdamW(params, lr=args.lr),
        forward_backward_fn=forward_backward,
        gradient_accumulation_steps=args.accum,
        max_grad_norm=1.0,
    )
    wrapper.setup()
    wrapper.save_checkpoint("mnist_demo_manual_untrained.safetensors")

    accum_loss = 0.0
    with make_profiler_ctx(args, "mnist_demo_manual_trace.json") as prof:
        for step, (images, labels) in enumerate(tqdm(loader, desc="Manual accum")):
            chunks = wrapper.split_batch(images, labels)

            # 1. Forward + backward only (no optimizer step yet)
            with record_function("forward_backward"):
                loss = wrapper.forward_backward_only(chunks, accum_steps=args.accum)
            accum_loss += loss

            # 2. Every accum_steps, sync grads and step
            if (step + 1) % args.accum == 0:
                with record_function("reduce_grads"):
                    wrapper.reduce_grads()    # ZeRO-1: NCCL reduce each grad to its owner GPU
                wrapper.clip_grads()          # uses wrapper.max_grad_norm
                with record_function("optimizer_step"):
                    wrapper.optimizer_step()  # step owned params + broadcast weights to all replicas

                # --- tinker here: e.g. EMA, custom logging, grad inspection ---
                # wrapper.models[0].some_param.grad  # still accessible before zero_grad
                # wrapper.optimizers[2].param_groups[0]["lr"] = new_lr

                print(
                    f"  step {step+1:5d}  loss={accum_loss/args.accum:.4f}  "
                    f"lr={wrapper.current_lr:.2e}"
                )
                accum_loss = 0.0

            if prof is not None:
                prof.step()

    wrapper.save_checkpoint("mnist_demo_manual.safetensors")
    wrapper.cleanup()


# ---------------------------------------------------------------------------
# Split forward/backward demo — inspect outputs between passes
# ---------------------------------------------------------------------------


def train_split_fwd_bwd(args):
    """Shows wrapper.forward() + wrapper.backward() as separate calls.

    Useful when you need to inspect, log, or post-process model outputs
    before committing to the backward pass (e.g. auxiliary losses, EMA
    target updates, logging predictions mid-step, etc.).
    """
    n_gpus = torch.cuda.device_count()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size * n_gpus,
        shuffle=True,
        num_workers=4,
        drop_last=True,
    )

    wrapper = MultiGPUWrapper(
        model_factory=lambda: MNISTNet(),
        optimizer_factory=lambda params: AdamW(params, lr=args.lr),
        gradient_accumulation_steps=args.accum,
        max_grad_norm=1.0,
    )
    wrapper.setup()
    wrapper.save_checkpoint("mnist_demo_split_untrained.safetensors")

    with make_profiler_ctx(args, "mnist_demo_split_trace.json") as prof:
        for step, (images, labels) in enumerate(tqdm(loader, desc="Split fwd/bwd")):
            chunks = wrapper.split_batch(images, labels)

            # 1. Forward — all GPUs run concurrently, outputs collected
            with record_function("forward"):
                outputs = wrapper.forward(chunks, forward_fn=forward_fn)
            # outputs[gpu_id] == (logits, labels_on_device)

            # --- inspect here: log predictions, compute auxiliary metrics, etc. ---
            if step % 100 == 0:
                all_logits = torch.cat([o[0].cpu() for o in outputs])
                all_labels = torch.cat([o[1].cpu() for o in outputs])
                acc = (all_logits.argmax(1) == all_labels).float().mean().item()
                print(f"  step {step:5d}  train acc (this batch): {acc:.3f}")

            # 2. Backward — all GPUs run concurrently, returns summed loss
            with record_function("backward"):
                loss = wrapper.backward(
                    outputs, backward_fn=backward_fn, accum_steps=args.accum
                )

            # 3. Sync + step (manual here so we can slot things in between)
            if (step + 1) % args.accum == 0:
                with record_function("reduce_grads"):
                    wrapper.reduce_grads()    # ZeRO-1: reduce each grad to its owner GPU
                wrapper.clip_grads()
                with record_function("optimizer_step"):
                    wrapper.optimizer_step()  # step owned params + broadcast weights

            if prof is not None:
                prof.step()

    wrapper.save_checkpoint("mnist_demo_split.safetensors")
    wrapper.cleanup()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument(
        "--batch-size", type=int, default=128, help="per-GPU batch size"
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--accum", type=int, default=1, help="gradient accumulation steps"
    )
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument(
        "--manual", action="store_true", help="use manual accumulation demo"
    )
    parser.add_argument(
        "--split", action="store_true", help="use split forward/backward demo"
    )
    parser.add_argument(
        "--profile", action="store_true", default=False,
        help="enable torch.profiler (CPU+CUDA, memory, shapes, flops); exports a Chrome trace",
    )
    parser.add_argument(
        "--profile-start", type=int, default=20, metavar="STEP",
        help="global step at which profiling begins (default: 20)",
    )
    parser.add_argument(
        "--profile-stop", type=int, default=23, metavar="STEP",
        help="global step at which profiling ends, exclusive (default: 23)",
    )
    args = parser.parse_args()

    if args.manual:
        train_manual_accum(args)
    elif args.split:
        train_split_fwd_bwd(args)
    else:
        train(args)
