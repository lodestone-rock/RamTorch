"""
mnist_three_way_plot.py
------------------------
Three-way MNIST training comparison for the team:

  A. full-batch          — model(x).backward() on the whole batch
  B. microbatch-accum    — sequential gradient accumulation (no pipeline)
  C. pipeline 1B1F       — our staggered_1b1f pipeline-parallel executor

All three: same weight init, same data order, same Adam optimizer + LR, same
number of steps. We track train loss per step and test accuracy periodically,
then plot both so it's obvious the three runs are numerically equivalent (B and
C overlap almost exactly; A is the full-batch reference).

Uses Adam (not SGD) since that's what people actually train with; LR is tuned
for Adam (much smaller than SGD's).

Outputs:
    mnist_three_way.png
    (optionally prints final test accuracies)

Usage:
    python examples/mnist_three_way_plot.py --batches 1000
    python examples/mnist_three_way_plot.py --batches 2000 --lr 1e-3
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


# ── Data ──────────────────────────────────────────────────────────────────────

def get_train_loader(batch_size, max_batches, seed):
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    ds = datasets.MNIST("./data", train=True, download=True, transform=tf)
    n = batch_size * max_batches
    base = len(ds)
    idx = [i % base for i in range(n)]
    return DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False,
                      num_workers=0, drop_last=True)


def get_test_loader():
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    ds = datasets.MNIST("./data", train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=512, shuffle=False, num_workers=0)


@torch.no_grad()
def test_accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    model.train()
    return correct / total


# ── Trainers (all use Adam) ──────────────────────────────────────────────────

def train_full_batch(model, loader, test_loader, device, lr, max_batches, eval_every):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses, accs = [], []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (i + 1) % eval_every == 0:
            accs.append((i + 1, test_accuracy(model, test_loader, device)))
    return losses, accs, model


def train_microbatch_accum(model, loader, test_loader, device, lr, max_batches,
                           n_micro, eval_every):
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    losses, accs = [], []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        mb_losses = []
        for xmb, ymb in zip(x.chunk(n_micro), y.chunk(n_micro)):
            loss = loss_fn(model(xmb), ymb)
            loss.backward()
            mb_losses.append(loss.item())
        with torch.no_grad():
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.mul_(1.0 / n_micro)
        opt.step()
        losses.append(sum(mb_losses) / len(mb_losses))
        if (i + 1) % eval_every == 0:
            accs.append((i + 1, test_accuracy(model, test_loader, device)))
    return losses, accs, model


def train_pipeline(model, loader, test_loader, devices, lr, max_batches,
                   n_micro, eval_every, eval_template):
    example = next(iter(loader))[0][:64 // n_micro]
    pipe = Pipeline(model, example_input=example, split_spec=SPLIT_SPEC,
                    devices=devices, overlap=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    # The pipeline splits the model across `devices`, so a plain model(x) eval
    # won't run (mat1/mat2 on different GPUs). Keep a single-device eval copy and
    # sync it from the pipeline's state_dict() (which gathers params to CPU).
    eval_device = devices[0]
    eval_model = copy.deepcopy(eval_template).to(eval_device)
    losses, accs = [], []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        result = pipe.step(x, targets=y, schedule="staggered_1b1f",
                           n_microbatches=n_micro, loss_fn=loss_fn)
        result.flush_grads()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(result.loss.item())
        if (i + 1) % eval_every == 0:
            eval_model.load_state_dict(
                {k: v.cpu() for k, v in model.state_dict().items()}
            )
            accs.append((i + 1, test_accuracy(eval_model, test_loader, eval_device)))
    return losses, accs, model


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3, help="Adam lr (tuned)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--devices", nargs=2, default=["cuda:1", "cuda:3"])
    ap.add_argument("--out", type=str, default="mnist_three_way.png")
    args = ap.parse_args()

    train_loader = get_train_loader(args.batch_size, args.batches, args.seed)
    test_loader = get_test_loader()

    torch.manual_seed(args.seed)
    base = MNISTMLP(hidden=args.hidden)

    print(f"config: batches={args.batches} batch={args.batch_size} micro={args.micro} "
          f"Adam lr={args.lr} seed={args.seed}  eval_every={args.eval_every}")
    print(f"devices: A/B on {args.devices[0]}, C pipeline on {args.devices}\n")

    print("[A] full-batch ...", flush=True)
    la, aa, _ = train_full_batch(copy.deepcopy(base), train_loader, test_loader,
                                 args.devices[0], args.lr, args.batches, args.eval_every)
    print("[B] microbatch-accum ...", flush=True)
    lb, ab, _ = train_microbatch_accum(copy.deepcopy(base), train_loader, test_loader,
                                       args.devices[0], args.lr, args.batches,
                                       args.micro, args.eval_every)
    print("[C] pipeline staggered_1b1f ...", flush=True)
    lc, ac, _ = train_pipeline(copy.deepcopy(base), train_loader, test_loader,
                               args.devices, args.lr, args.batches, args.micro,
                               args.eval_every, eval_template=base)

    # ── Numeric summary ───────────────────────────────────────────────────────
    print("\n=== final test accuracy ===")
    print(f"  A full-batch       : {aa[-1][1]:.4f}")
    print(f"  B microbatch-accum : {ab[-1][1]:.4f}")
    print(f"  C pipeline 1B1F    : {ac[-1][1]:.4f}")
    print(f"\n  max|Δloss(A,B)|={max(abs(x-y) for x,y in zip(la,lb)):.2e}")
    print(f"  max|Δloss(B,C)|={max(abs(x-y) for x,y in zip(lb,lc)):.2e}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    steps = range(1, len(la) + 1)
    # Smooth loss with a small window for readability
    def smooth(v, k=15):
        if len(v) < k:
            return v
        out = torch.tensor(v, dtype=torch.float32)
        return out.unfold(0, k, 1).mean(dim=1).tolist() + [float('nan')]*(k-1)

    ax1.plot(steps, smooth(la), label="A full-batch", color="#2ca02c", lw=1.6)
    ax1.plot(steps, smooth(lb), label="B microbatch-accum", color="#1f77b4", lw=1.4)
    ax1.plot(steps, smooth(lc), label="C pipeline 1B1F", color="#d62728", lw=1.2,
             linestyle="--", alpha=0.9)
    ax1.set_xlabel("step"); ax1.set_ylabel("train loss (smoothed)")
    ax1.set_title("Train loss — 3 runs overlap")
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot([s for s, _ in aa], [v for _, v in aa], marker="o", ms=4,
             label="A full-batch", color="#2ca02c")
    ax2.plot([s for s, _ in ab], [v for _, v in ab], marker="s", ms=4,
             label="B microbatch-accum", color="#1f77b4")
    ax2.plot([s for s, _ in ac], [v for _, v in ac], marker="^", ms=4,
             label="C pipeline 1B1F", color="#d62728", linestyle="--")
    ax2.set_xlabel("step"); ax2.set_ylabel("test accuracy")
    ax2.set_title("Test accuracy — 3 runs overlap")
    ax2.legend(); ax2.grid(alpha=0.3); ax2.set_ylim(0.9, 1.0)

    fig.suptitle(f"MNIST 3-way equivalence — Adam lr={args.lr}, {args.batches} steps, "
                 f"micro={args.micro}")
    fig.tight_layout()
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
