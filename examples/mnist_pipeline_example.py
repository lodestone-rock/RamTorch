"""
mnist_pipeline_example.py
--------------------------
A complete, self-contained reference for RamTorch pipeline parallelism.

This is the canonical "how do I use the pipeline?" example. It trains a small
MLP on MNIST pipeline-parallel across two GPUs, then runs inference — using the
high-level `PipelineModel` API that feels like a single-GPU model.

What it shows (the whole workflow, end to end):
  1. Define a plain nn.Module (nothing pipeline-specific in the model itself).
  2. Wrap it in PipelineModel — auto-split into balanced stages, one line.
  3. Train with pipe.step() + flush_grads() + a single Adam optimizer.
  4. Evaluate with pipe.forward() (arbitrary batch size).
  5. Save / load a standard state_dict checkpoint.

Requirements: 2+ GPUs (edit DEVICES below to the free ones), torchvision for
MNIST. Run:
    python examples/mnist_pipeline_example.py
    python examples/mnist_pipeline_example.py --steps 500 --devices cuda:0 cuda:1

For lower-level control (manual split points, schedule choice, profiling), see
`examples/pipeline_easy_demo.py` and the ramtorch.pipeline_relay docs.
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ramtorch import PipelineModel


# ── 1. A plain model — nothing pipeline-specific here ─────────────────────────
# PipelineModel splits on the model's TOP-LEVEL children (fc1..fc4 below), so
# any nn.Module structured as a sequence of submodules works out of the box.

class MNISTMLP(nn.Module):
    def __init__(self, hidden: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(28 * 28, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--micro", type=int, default=4, help="microbatches per step")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--schedule", default="staggered_1b1f",
                    choices=["gpipe", "staggered_1b1f"],
                    help="recommended schedules. ('1f1b' is also accepted for "
                         "comparison but is educational only — see README.)")
    ap.add_argument("--devices", nargs="+", default=["cuda:1", "cuda:3"],
                    help="one device per pipeline stage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="", help="optional checkpoint path")
    args = ap.parse_args()

    n_stages = len(args.devices)
    if n_stages < 1:
        raise SystemExit("need at least one device")
    if n_stages > 1 and torch.cuda.device_count() < 2:
        raise SystemExit("this example uses multiple GPUs; pass --devices cuda:0 for single")

    torch.manual_seed(args.seed)

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

    # ── 2. Wrap in PipelineModel (auto-split into balanced stages) ────────────
    # The tracer needs ONE example microbatch (batch_size // micro samples).
    example = torch.randn(args.batch_size // args.micro, 1, 28, 28)
    pipe = PipelineModel(
        MNISTMLP(hidden=args.hidden),
        example,
        devices=args.devices,
        # device_weights=[2.0, 1.0],  # <- uncomment for heterogeneous GPUs
    )
    print(f"pipeline: {pipe.num_stages} stages on {[str(d) for d in pipe.devices]}")
    print(f"  auto-split at: {list(pipe.split_spec.keys()) or '(single stage)'}")
    print(f"  schedule={args.schedule}  microbatches={args.micro}  Adam lr={args.lr}")

    # ── 3. Train ──────────────────────────────────────────────────────────────
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(pipe.parameters(), lr=args.lr)  # one optimizer, whole model

    pipe.train()
    step = 0
    running = 0.0
    done = False
    while not done:
        for x, y in train_loader:
            # One pipeline forward+backward step over the batch (split into micros).
            result = pipe.step(
                x,
                targets=y,
                schedule=args.schedule,
                n_microbatches=args.micro,
                loss_fn=loss_fn,
            )
            result.flush_grads()      # mean-scale microbatch grads into .grad
            opt.step()                # update all params (stages share them)
            opt.zero_grad(set_to_none=True)

            running += result.loss.item()
            step += 1
            if step % 50 == 0:
                print(f"  step {step:>4}: loss={running/50:.4f}")
                running = 0.0
            if step >= args.steps:
                done = True
                break

    # ── 4. Evaluate — pipe.forward() handles arbitrary batch sizes ────────────
    pipe.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = pipe.forward(x).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"\ntest accuracy after {step} steps: {acc:.4f}")

    # ── 5. Checkpoint — standard state_dict, no pipeline-specific format ──────
    if args.save:
        torch.save(pipe.state_dict(), args.save)
        print(f"saved checkpoint -> {args.save}")
        # Reload into a fresh single-GPU model (state_dict is the plain model's):
        #   m = MNISTMLP(); m.load_state_dict(torch.load(args.save)); m.to("cuda:0")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
