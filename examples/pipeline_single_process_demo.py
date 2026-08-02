"""
pipeline_single_process_demo.py
--------------------------------
Single-process pipeline parallelism demo (no torchrun, no process groups).

Runs GPipe / 1F1B / Interleaved-1F1B via ramtorch.run_pipeline and checks
gradient parity against a plain sequential baseline. Also demonstrates
fake-compute mode for simulating schedule timing without loading the GPUs.

Usage:
    python examples/pipeline_single_process_demo.py
"""

import copy
import time

import torch
import torch.nn as nn
from torch.distributed.pipelining import SplitPoint

from ramtorch import run_pipeline


# ── Model (same as explore_pipeline_split.py) ─────────────────────────────────

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


def run_schedule(schedule, model, data, targets, devices, overlap=True, fake=None,
                 pp_group_size=None):
    m = copy.deepcopy(model)
    t0 = time.perf_counter()
    result = run_pipeline(
        m,
        example_input=data[: BATCH // N_MICRO],
        split_spec=SPLIT_SPEC,
        data=data,
        targets=targets,
        schedule=schedule,
        n_microbatches=N_MICRO,
        loss_fn=loss_fn,
        devices=devices,
        fake_compute=fake,
        overlap=overlap,
        pp_group_size=pp_group_size,
    )
    dt = time.perf_counter() - t0
    result.flush_grads()  # mean-scaled grads into .grad
    grads = {n: p.grad.detach().cpu().clone() for n, p in m.named_parameters()}
    return result.loss.detach().cpu(), grads, dt


def check_parity(name, ref_loss, ref_grads, loss, grads):
    ok = torch.allclose(ref_loss, loss, rtol=1e-4, atol=1e-6)
    max_diff = 0.0
    for n, g in grads.items():
        d = (g - ref_grads[n]).abs().max().item()
        max_diff = max(max_diff, d)
    grad_ok = max_diff < 1e-4
    status = "OK " if (ok and grad_ok) else "FAIL"
    print(f"  [{status}] {name:<28} loss={loss.item():.6f}  max|Δgrad|={max_diff:.2e}")
    return ok and grad_ok


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
            loss, grads, dt = run_schedule(sched, model, data, targets, devices, overlap)
            all_ok &= check_parity(f"{sched} overlap={overlap}", ref_loss, ref_grads, loss, grads)

    # Interleaved: treat the 3 stages as p=1, v=3 (single "rank" view)
    loss, grads, dt = run_schedule(
        "interleaved", model, data, targets, devices, pp_group_size=1
    )
    all_ok &= check_parity("interleaved p=1", ref_loss, ref_grads, loss, grads)

    # ── Fake-compute timing simulation (no GPU load) ──────────────────────────
    print("\nfake-compute timing (fwd=20ms, bwd=40ms per stage):")
    fake = {"fwd": 0.02, "bwd": 0.04}
    p, m = n_stages, N_MICRO
    tf, tb = 0.02, 0.04
    # Critical path (perfect overlap): GPipe = fill + drain; 1F1B = warmup fwd
    # chain + first bwd chain + steady-state bottleneck + cooldown drain.
    crit = {
        "gpipe": p * m * tf + p * m * tb - (p - 1) * (m - 1) * min(tf, tb) * 0
        # simpler: last mb finishes fwd at p*tf + (m-1)*tf, then bwd chain
    }
    # exact critical paths for these schedules:
    crit_gpipe = (p + m - 1) * tf + (p + m - 1) * tb - (p - 1) * tb  # fwd pipe + bwd pipe
    crit_1f1b = p * tf + p * tb + (m - 1) * max(tf, tb) + (p - 1) * tb
    for sched in ["gpipe", "1f1b"]:
        _, _, dt = run_schedule(sched, model, data, targets, ["cpu"] * n_stages,
                                overlap=True, fake=fake)
        c = crit_gpipe if sched == "gpipe" else crit_1f1b
        serial = m * p * (tf + tb)
        print(f"  {sched:<8} wall={dt*1e3:7.1f}ms  critical-path≈{c*1e3:5.0f}ms  "
              f"serial={serial*1e3:.0f}ms")

    print("\n" + ("ALL CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
