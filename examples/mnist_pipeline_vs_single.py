"""
mnist_pipeline_vs_single.py
---------------------------
Head-to-head training validation: classical single-GPU training vs our
pipeline-parallel executor (staggered_1b1f), starting from the SAME weight init
and feeding the SAME batches in the SAME order.

If the pipeline (model split + manual autograd + relay + grad accumulation) is
correct, the two runs must produce:
  * the same loss trajectory (to float tolerance), and
  * the same final weights (to float tolerance),
because they perform mathematically identical computation.

Fair-comparison contract (the subtle parts that make or break equivalence):
  * Identical init: seed torch before building each model; deepcopy the seeded
    model for the second run so both start from the exact same weights.
  * Identical data order: a fixed-seed sampler yields the same batch sequence
    to both runs.
  * Identical loss reduction: pipeline splits each batch into N_MICRO equal
    microbatches and flush_grads() averages their grads by 1/N_MICRO. With
    cross_entropy(mean) per equal-sized microbatch, mean-of-means == full-batch
    mean, so the effective gradient matches a plain full-batch backward.
  * Identical optimizer + LR + step count: SGD with fixed lr, one step per
    batch, no momentum/weight-decay/warmup (keep the moving parts minimal so any
    divergence is attributable to the pipeline, not the optimizer).

GPUs: pipeline uses 2 stages on --devices (default cuda:1,cuda:3 — the free
ones); the single-GPU control runs on the first of those devices.

Usage:
    python examples/mnist_pipeline_vs_single.py
    python examples/mnist_pipeline_vs_single.py --batches 200 --devices cuda:1 cuda:3
"""

from __future__ import annotations

import argparse
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ramtorch import Pipeline
from torch.distributed.pipelining import SplitPoint


# ── Model ─────────────────────────────────────────────────────────────────────
# A plain MLP so the pipeline split is clean (linear stacks split predictably).
# Flattened MNIST (784) -> hidden -> hidden -> hidden -> 10.

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


# 2 stages: [fc1, fc2] | [fc3, fc4]  -> split at fc3
SPLIT_SPEC = {"fc3": SplitPoint.BEGINNING}

loss_fn = nn.CrossEntropyLoss()  # mean reduction over each microbatch


# ── Data ──────────────────────────────────────────────────────────────────────

def get_loader(batch_size: int, seed: int, max_batches: int):
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    ds = datasets.MNIST("./data", train=True, download=True, transform=tf)
    # Fixed index sequence so both runs see the SAME samples in the SAME order.
    # Indices wrap (mod dataset size) so we can run more batches than one epoch.
    n = batch_size * max_batches
    base = len(ds)
    indices = [i % base for i in range(n)]
    ds = Subset(ds, indices)
    g = torch.Generator().manual_seed(seed)
    # shuffle=False keeps order deterministic and identical across runs.
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=0, drop_last=True, generator=g)


# ── Single-GPU classical training ─────────────────────────────────────────────

def train_single_gpu(model, loader, device, lr, max_batches):
    """Plain training loop. Returns (losses, final_state_dict)."""
    model = model.to(device)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ── Pipeline (staggered_1b1f) training ────────────────────────────────────────

def train_pipeline(model, loader, devices, lr, max_batches, n_micro, overlap=True):
    """Pipeline training loop via a reusable Pipeline (split once, step per batch).

    Returns (losses, final_state_dict).
    """
    # Build the pipeline ONCE: splits the model and moves stages to devices.
    # example_input must match the (CPU) model's device at trace time.
    example = next(iter(loader))[0][:64 // n_micro]
    pipe = Pipeline(
        model,
        example_input=example,
        split_spec=SPLIT_SPEC,
        devices=devices,
        overlap=overlap,
    )
    # One optimizer over the *full* model's params; pipeline stages share these
    # same param objects after the split, so a single SGD sees all grads.
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        result = pipe.step(
            x,
            targets=y,
            schedule="staggered_1b1f",
            n_microbatches=n_micro,
            loss_fn=loss_fn,
        )
        # Write accumulated microbatch grads (mean-scaled) into .grad, then step.
        result.flush_grads()  # default scale = 1/n_micro
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(result.loss.item())
    return losses, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ── Comparison ────────────────────────────────────────────────────────────────

def _max_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item() if a.numel() else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--devices", nargs=2, default=["cuda:1", "cuda:3"])
    args = ap.parse_args()

    assert args.batch_size % args.micro == 0
    loader = get_loader(args.batch_size, args.seed, args.batches)

    # ── Identical init for both runs ──────────────────────────────────────────
    torch.manual_seed(args.seed)
    base_model = MNISTMLP(hidden=args.hidden)
    model_single = copy.deepcopy(base_model)
    model_pipe = copy.deepcopy(base_model)

    # Sanity: identical starting weights
    for (n1, p1), (n2, p2) in zip(
        model_single.state_dict().items(), model_pipe.state_dict().items()
    ):
        assert n1 == n2 and torch.equal(p1, p2), f"init mismatch at {n1}"
    print(f"init: identical weights confirmed ({sum(p.numel() for p in base_model.parameters())} params)")
    print(f"config: batches={args.batches} batch_size={args.batch_size} "
          f"micro={args.micro} lr={args.lr} seed={args.seed}")
    print(f"devices: single-GPU on {args.devices[0]}, pipeline stages on {args.devices}\n")

    # ── Run both ──────────────────────────────────────────────────────────────
    print("training single-GPU (control) ...", flush=True)
    losses_sg, sd_sg = train_single_gpu(
        model_single, loader, args.devices[0], args.lr, args.batches
    )

    print("training pipeline staggered_1b1f ...", flush=True)
    losses_pp, sd_pp = train_pipeline(
        model_pipe, loader, args.devices, args.lr, args.batches, args.micro
    )

    # ── Compare loss trajectories ─────────────────────────────────────────────
    print("\nstep  single-GPU   pipeline      Δloss")
    worst_loss = 0.0
    for i in range(0, args.batches, max(1, args.batches // 10)):
        a, b = losses_sg[i], losses_pp[i]
        d = abs(a - b)
        worst_loss = max(worst_loss, d)
        print(f"{i:>5}  {a:>10.6f}  {b:>10.6f}  {d:.2e}")
    # full-trajectory max diff
    worst_loss = max(abs(a - b) for a, b in zip(losses_sg, losses_pp))
    print(f"\nmax |Δloss| over all {args.batches} steps: {worst_loss:.3e}")

    # ── Compare final weights ─────────────────────────────────────────────────
    print("\nfinal-weight parity per tensor:")
    worst_w, worst_name = 0.0, None
    for k in sd_sg:
        d = _max_diff(sd_sg[k], sd_pp[k])
        if d > worst_w:
            worst_w, worst_name = d, k
        print(f"  {k:<12} max|Δ|={d:.3e}")
    print(f"\nworst weight diff: {worst_w:.3e}  ({worst_name})")

    # ── Verdict ───────────────────────────────────────────────────────────────
    LOSS_TOL = 1e-3   # loss trajectory (fp32 accumulation-order noise)
    W_TOL = 1e-3      # final weights
    loss_ok = worst_loss < LOSS_TOL
    w_ok = worst_w < W_TOL
    print("\n" + "=" * 60)
    print(f"loss trajectory match : {'OK' if loss_ok else 'FAIL'} (tol {LOSS_TOL})")
    print(f"final weights match   : {'OK' if w_ok else 'FAIL'} (tol {W_TOL})")
    ok = loss_ok and w_ok
    print("VERDICT:", "PIPELINE EQUIVALENT TO SINGLE-GPU" if ok else "DIVERGENCE — INVESTIGATE")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
