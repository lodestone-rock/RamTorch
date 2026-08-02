"""
pipeline_relay_demo.py
----------------------
Validation for the relay executor (:func:`ramtorch.run_pipeline_relay`).

Checks, against a plain sequential baseline on the full batch:
  * loss parity          — GPipe and 1F1B, overlap True/False
  * gradient parity      — per-param max|Δgrad| after flush_grads
  * schedule restoration — writes Perfetto traces so you can confirm 1F1B now
                           interleaves F/B across stages instead of collapsing
                           into GPipe (all-F-then-all-B).

Open the traces at https://ui.perfetto.dev (drag & drop the .json files).

Usage:
    python examples/pipeline_relay_demo.py
Outputs:
    relay_trace_gpipe.json, relay_trace_1f1b.json
"""

import copy
import time

import torch
import torch.nn as nn
from torch.distributed.pipelining import SplitPoint

from ramtorch import run_pipeline_relay


# ── Model (same as pipeline_single_process_demo.py) ──────────────────────────

class Layer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(torch.relu(self.lin(x)))


class MyModel(nn.Module):
    def __init__(self, dim: int = 16, n_layers: int = 4):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.layers = nn.ModuleList([Layer(dim) for _ in range(n_layers)])
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


# ── Config ────────────────────────────────────────────────────────────────────

DIM = 16
N_LAYERS = 4
BATCH = 16
N_MICRO = 4
SPLIT_SPEC = {
    "layers.1": SplitPoint.BEGINNING,
    "layers.3": SplitPoint.BEGINNING,
}  # → 3 stages: [embed+L0], [L1+L2], [L3+head]

loss_fn = nn.MSELoss()


def sequential_baseline(model, data, targets):
    """Plain fwd/bwd on the full batch; returns (loss, {name: grad})."""
    model.zero_grad(set_to_none=True)
    out = model(data)
    loss = loss_fn(out, targets)
    loss.backward()
    grads = {n: p.grad.clone() for n, p in model.named_parameters()}
    return loss.detach(), grads


def run_schedule(schedule, model, data, targets, devices, overlap=True,
                 trace_path=None):
    m = copy.deepcopy(model)
    t0 = time.perf_counter()
    result = run_pipeline_relay(
        m,
        example_input=data[: BATCH // N_MICRO],
        split_spec=SPLIT_SPEC,
        data=data,
        targets=targets,
        schedule=schedule,
        n_microbatches=N_MICRO,
        loss_fn=loss_fn,
        devices=devices,
        overlap=overlap,
        trace_path=trace_path,
    )
    dt = time.perf_counter() - t0
    result.flush_grads()  # mean-scaled grads into .grad
    grads = {n: p.grad.detach().cpu().clone() for n, p in m.named_parameters()}
    return result.loss.detach().cpu(), grads, dt


def check_parity(name, ref_loss, ref_grads, loss, grads):
    loss_ok = torch.allclose(ref_loss, loss, rtol=1e-4, atol=1e-6)
    max_diff = 0.0
    for n, g in grads.items():
        d = (g - ref_grads[n]).abs().max().item()
        max_diff = max(max_diff, d)
    grad_ok = max_diff < 1e-4
    status = "OK " if (loss_ok and grad_ok) else "FAIL"
    print(f"  [{status}] {name:<26} loss={loss.item():.6f}  max|Δgrad|={max_diff:.2e}")
    return loss_ok and grad_ok


def main():
    torch.manual_seed(0)
    model = MyModel(dim=DIM, n_layers=N_LAYERS)
    data = torch.randn(BATCH, DIM)
    targets = torch.randn(BATCH, 1)

    n_cuda = torch.cuda.device_count()
    n_stages = 3
    devices = (
        [f"cuda:{i % n_cuda}" for i in range(n_stages)] if n_cuda else ["cpu"] * n_stages
    )
    print(f"devices: {devices}")

    ref_loss, ref_grads = sequential_baseline(model, data, targets)
    print(f"sequential baseline loss: {ref_loss.item():.6f}\n")

    all_ok = True
    for sched in ["gpipe", "1f1b"]:
        for overlap in [False, True]:
            trace = f"relay_trace_{sched}.json" if overlap else None
            loss, grads, dt = run_schedule(
                sched, model, data, targets, devices, overlap, trace_path=trace
            )
            all_ok &= check_parity(
                f"{sched} overlap={overlap}", ref_loss, ref_grads, loss, grads
            )

    print("\ntraces written: relay_trace_gpipe.json, relay_trace_1f1b.json")
    print("open at https://ui.perfetto.dev — 1F1B should show steady-state F/B")
    print("interleave across stages, NOT all-forwards-then-all-backwards.")

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
