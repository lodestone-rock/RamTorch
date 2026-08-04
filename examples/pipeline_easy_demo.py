"""
pipeline_easy_demo.py
---------------------
The single-GPU-feel pipeline API: PipelineModel.

Shows both sides of the ergonomic wrapper:
  * auto-split (no manual split_spec) across the available GPUs
  * pipe.forward(x)  -> inference, one call
  * pipe.step() + flush_grads() + a single optimizer -> training

Run:
    python examples/pipeline_easy_demo.py
"""

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from ramtorch import PipelineModel


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
    if torch.cuda.device_count() < 2:
        raise SystemExit("need 2 GPUs for this demo")
    devices = ["cuda:1", "cuda:3"]  # the free ones on this box

    # ── Data ──────────────────────────────────────────────────────────────────
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_ds = datasets.MNIST("./data", train=True, download=True, transform=tf)
    idx = [i % len(train_ds) for i in range(64 * 200)]
    train_loader = DataLoader(Subset(train_ds, idx), batch_size=64, shuffle=False)
    test_loader = DataLoader(
        datasets.MNIST("./data", train=False, transform=tf), batch_size=512
    )

    # ── Build the pipeline model — auto-split, one line ──────────────────────
    torch.manual_seed(0)
    example = next(iter(train_loader))[0][:16]  # one microbatch for the tracer
    pipe = PipelineModel(MNISTMLP(), example, devices=devices)
    print(f"auto-split into {pipe.num_stages} stages on {pipe.devices}")
    print(f"  split_spec: {pipe.split_spec}")

    # ── Inference — just .forward() ───────────────────────────────────────────
    pipe.eval()
    x_test, _ = next(iter(test_loader))
    logits = pipe.forward(x_test)
    print(f"inference: forward({tuple(x_test.shape)}) -> {tuple(logits.shape)}")

    # Sanity: matches a plain single-GPU forward exactly
    ref = copy.deepcopy(pipe._model).to(devices[0])
    ref_logits = ref(x_test.to(devices[0]))
    print(f"  max|Δ| vs single-GPU forward: {(logits.cpu()-ref_logits.cpu()).abs().max().item():.2e}")

    # ── Training — step + flush + single optimizer ────────────────────────────
    pipe.train()
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(pipe.parameters(), lr=1e-3)  # one optimizer over the wrapper

    print("\ntraining (200 steps, staggered_1b1f, 4 microbatches) ...")
    for i, (x, y) in enumerate(train_loader):
        if i >= 200:
            break
        result = pipe.step(x, targets=y, schedule="staggered_1b1f",
                           n_microbatches=4, loss_fn=loss_fn)
        result.flush_grads()
        opt.step()
        opt.zero_grad(set_to_none=True)
        if (i + 1) % 50 == 0:
            print(f"  step {i+1:>3}: loss={result.loss.item():.4f}")

    # ── Final test accuracy via the eval-friendly forward ────────────────────
    pipe.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            pred = pipe.forward(x).argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.size(0)
    print(f"\nfinal test accuracy: {correct/total:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
