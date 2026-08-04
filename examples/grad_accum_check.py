"""
grad_accum_check.py
-------------------
Isolate the gradient-reduction path from weight updates.

Hypothesis to test: the pipeline's gradient *reduction* (sum per-microbatch
means, then scale by 1/n_micro) drifts from a plain sequential full-batch
backward over many accumulation steps.

Method: run K accumulation steps where each step computes the gradient of a
batch and ADDS it into a running gradient buffer, for BOTH:
  (a) sequential  — plain model(x).backward() on the full batch, and
  (b) pipeline    — run_pipeline_relay + flush_grads(scale=1) to ADD grads.

NO optimizer step is taken, so weights stay frozen and identical — any
divergence in the accumulated gradient is purely from the reduction, not from
compounding through weight updates.

We compare two reduction conventions for the pipeline:
  * scale=1.0         -> accumulated SUM of per-microbatch means
  * scale=1/n_micro   -> accumulated MEAN of per-microbatch means
and the matching sequential equivalents (sum of full-batch grads, or mean).

If the pipeline reduction is correct, the matching conventions agree to fp32
round-off and the MISMATCHED ones differ by exactly a factor of n_micro.

Usage:
    python examples/grad_accum_check.py --steps 1000
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
loss_fn = nn.CrossEntropyLoss()  # mean over each (micro)batch


def get_batches(batch_size, n_steps, seed):
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    ds = datasets.MNIST("./data", train=True, download=True, transform=tf)
    n = batch_size * n_steps
    base = len(ds)
    idx = [i % base for i in range(n)]
    loader = DataLoader(Subset(ds, idx), batch_size=batch_size, shuffle=False,
                        num_workers=0, drop_last=True)
    # Materialize the exact batch sequence so both paths see identical data.
    return [(x, y) for x, y in loader]


@torch.no_grad()
def _zero_acc(model):
    return {n: torch.zeros_like(p, device="cpu") for n, p in model.named_parameters()}


def sequential_accum(model, batches, device, scale):
    """Accumulate full-batch grads (sum over steps). Returns {name: grad_sum}."""
    model = model.to(device)
    acc = _zero_acc(model)
    for x, y in batches:
        x, y = x.to(device), y.to(device)
        model.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        for n, p in model.named_parameters():
            acc[n] += p.grad.detach().cpu() * scale
    return acc


def pipeline_accum(pipe, model, batches, n_micro, scale):
    """Accumulate pipeline grads (sum over steps). Returns {name: grad_sum}."""
    acc = _zero_acc(model)
    for x, y in batches:
        result = pipe.step(x, targets=y, schedule="staggered_1b1f",
                           n_microbatches=n_micro, loss_fn=loss_fn)
        result.flush_grads(scale=scale)   # write scaled grad into .grad
        for n, p in model.named_parameters():
            acc[n] += p.grad.detach().cpu()
    return acc


def _max_diff(a, b):
    return (a - b).abs().max().item()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--devices", nargs=2, default=["cuda:1", "cuda:3"])
    args = ap.parse_args()

    batches = get_batches(args.batch_size, args.steps, args.seed)
    n_micro = args.micro

    torch.manual_seed(args.seed)
    base = MNISTMLP(hidden=args.hidden)
    model_seq = copy.deepcopy(base)
    model_pipe = copy.deepcopy(base)

    example = batches[0][0][: args.batch_size // n_micro]
    pipe = Pipeline(model_pipe, example_input=example, split_spec=SPLIT_SPEC,
                    devices=args.devices, overlap=True)

    print(f"accumulating {args.steps} steps  batch={args.batch_size} micro={n_micro}")
    print(f"devices: seq on {args.devices[0]}, pipeline on {args.devices}\n")

    # ── Matching convention: both compute SUM over steps of full-batch-mean grad ──
    # Sequential: full-batch mean grad, summed over steps (scale=1).
    acc_seq = sequential_accum(model_seq, batches, args.devices[0], scale=1.0)
    # Pipeline: mean-of-microbatch-means per step (scale=1/n_micro), summed over steps.
    acc_pipe = pipeline_accum(pipe, model_pipe, batches, n_micro, scale=1.0 / n_micro)

    print("=== matching convention (both = sum of full-batch-mean grads) ===")
    worst, worst_n = 0.0, None
    for n in acc_seq:
        d = _max_diff(acc_seq[n], acc_pipe[n])
        if d > worst:
            worst, worst_n = d, n
        print(f"  {n:<12} max|Δ|={d:.3e}")
    print(f"worst: {worst:.3e} ({worst_n})")

    # ── Show the factor-of-n_micro relationship explicitly ──
    # If we DON'T scale the pipeline (scale=1), it should be n_micro times larger.
    acc_pipe_unscaled = pipeline_accum(pipe, model_pipe, batches, n_micro, scale=1.0)
    print("\n=== check factor: unscaled pipeline vs scaled pipeline ===")
    for n in list(acc_seq)[:3]:
        ratio = (acc_pipe_unscaled[n] / (acc_pipe[n] + 1e-12)).median().item()
        print(f"  {n:<12} median(unscaled/scaled)={ratio:.3f} (expect ~{n_micro})")

    # Verdict on the matching convention
    tol = max(1e-6, 1e-5 * args.steps / 1000)
    ok = worst < tol
    print("\n" + "=" * 60)
    print(f"matched-reduction agreement: {'OK' if ok else 'DRIFT'} (worst {worst:.2e}, tol {tol:.1e})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
