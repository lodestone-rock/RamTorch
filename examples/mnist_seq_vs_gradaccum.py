"""
mnist_seq_vs_gradaccum.py
--------------------------
Liability check: does plain *sequential* microbatch gradient accumulation —
with NO pipeline code at all — drift from full-batch training the same way our
pipeline does?

If yes, the pipeline introduces no NEW numerical liability: the drift is the
well-known microbatch-vs-fullbatch float effect that ANY gradient-accumulation
setup already has (single GPU included). If sequential-accum were somehow
cleaner than the pipeline, that would point at a pipeline-specific bug.

Three pure-PyTorch runs on ONE GPU, same init, same data order, same SGD steps:
  A. full-batch        — model(x).backward() on the whole batch (the reference)
  B. microbatch-accum  — split batch into N_MICRO chunks, backward each (mean
                         loss), accumulate .grad, scale by 1/N_MICRO, step.
                         This is EXACTLY what our pipeline does, but sequentially.
  C. (optional) pipeline — our staggered_1b1f, to confirm it matches B.

We compare final weights + loss trajectory of B (and C) against A.

Usage:
    python examples/mnist_seq_vs_gradaccum.py --batches 1000
"""

from __future__ import annotations

import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.pipelining import SplitPoint
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ramtorch import Pipeline


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


SPLIT_SPEC = {"fc3": SplitPoint.BEGINNING}
loss_fn = nn.CrossEntropyLoss()


def get_loader(batch_size, max_batches, seed):
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    ds = datasets.MNIST("./data", train=True, download=True, transform=tf)
    n = batch_size * max_batches
    base = len(ds)
    idx = [i % base for i in range(n)]
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False,
                      num_workers=0, drop_last=True)


def train_full_batch(model, loader, device, lr, max_batches):
    """A: plain full-batch training (reference)."""
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_microbatch_accum(model, loader, device, lr, max_batches, n_micro):
    """B: sequential microbatch grad-accum — mirrors the pipeline's reduction."""
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        mb_losses = []
        # Split into microbatches, backward each (mean), accumulate into .grad.
        for xmb, ymb in zip(x.chunk(n_micro), y.chunk(n_micro)):
            loss = loss_fn(model(xmb), ymb)   # mean over microbatch
            loss.backward()                    # accumulate into .grad
            mb_losses.append(loss.item())
        # Scale accumulated grads by 1/n_micro (== mean of microbatch means),
        # exactly like PipelineResult.flush_grads().
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(1.0 / n_micro)
        opt.step()
        losses.append(sum(mb_losses) / len(mb_losses))  # mean-of-means, like pipeline
    return losses, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def train_pipeline(model, loader, devices, lr, max_batches, n_micro):
    """C: our pipeline (staggered_1b1f)."""
    example = next(iter(loader))[0][:64 // n_micro]
    pipe = Pipeline(model, example_input=example, split_spec=SPLIT_SPEC,
                    devices=devices, overlap=True)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        result = pipe.step(x, targets=y, schedule="staggered_1b1f",
                           n_microbatches=n_micro, loss_fn=loss_fn)
        result.flush_grads()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(result.loss.item())
    return losses, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def _max_diff(a, b):
    return (a - b).abs().max().item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--devices", nargs=2, default=["cuda:1", "cuda:3"])
    ap.add_argument("--with-pipeline", action="store_true",
                    help="also run the pipeline (C) to confirm it matches B")
    args = ap.parse_args()

    loader = get_loader(args.batch_size, args.batches, args.seed)
    torch.manual_seed(args.seed)
    base = MNISTMLP(hidden=args.hidden)

    print(f"config: batches={args.batches} batch={args.batch_size} micro={args.micro} "
          f"lr={args.lr} seed={args.seed}  (single GPU {args.devices[0]})\n")

    losses_a, sd_a = train_full_batch(copy.deepcopy(base), loader, args.devices[0],
                                      args.lr, args.batches)
    print("A (full-batch)        done")

    losses_b, sd_b = train_microbatch_accum(copy.deepcopy(base), loader,
                                            args.devices[0], args.lr,
                                            args.batches, args.micro)
    print("B (microbatch-accum)  done")

    # Compare B vs A
    print("\n=== B (sequential microbatch-accum) vs A (full-batch) ===")
    print("step    A(full)      B(accum)      Δloss")
    for i in range(0, args.batches, max(1, args.batches // 10)):
        print(f"{i:>5}  {losses_a[i]:>10.6f}  {losses_b[i]:>10.6f}  "
              f"{abs(losses_a[i]-losses_b[i]):.2e}")
    worst_loss = max(abs(a - b) for a, b in zip(losses_a, losses_b))
    worst_w = max(_max_diff(sd_a[k], sd_b[k]) for k in sd_a)
    print(f"\nmax|Δloss|={worst_loss:.3e}   max|Δweight|={worst_w:.3e}")

    result_b = (worst_loss, worst_w)

    if args.with_pipeline:
        losses_c, sd_c = train_pipeline(copy.deepcopy(base), loader, args.devices,
                                        args.lr, args.batches, args.micro)
        print("\n=== C (pipeline staggered_1b1f) vs B (sequential accum) ===")
        worst_loss_c = max(abs(b - c) for b, c in zip(losses_b, losses_c))
        worst_w_c = max(_max_diff(sd_b[k], sd_c[k]) for k in sd_b)
        print(f"max|Δloss(B,C)|={worst_loss_c:.3e}   max|Δweight(B,C)|={worst_w_c:.3e}")
        print("(C should match B almost exactly — both are microbatch-accum)")

    print("\n" + "=" * 60)
    print("If B drifts from A by the SAME amount as the pipeline does, the")
    print("pipeline adds NO new liability — it's just standard microbatching.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
