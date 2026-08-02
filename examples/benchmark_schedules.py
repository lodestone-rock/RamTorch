"""
benchmark_schedules.py
----------------------
Big-model schedule comparison with kineto profiling.

Uses a large transformer-ish FFN so per-op kernels are substantial and the
schedule's overlap pattern dominates launch overhead (the thing that hid the
pattern at small sizes). Warmup iterations run before the profiled iteration so
we measure steady state, not CUDA init / JIT.

GPUs 1 and 3 are free on this box (0 and 2 are busy), so this uses a 2-stage
pipeline on cuda:1 / cuda:3. More microbatches + a wide model make the
forward/backward packing visible in the profiler timeline.

Usage:
    python examples/benchmark_schedules.py
Outputs:
    big_profile_gpipe.json, big_profile_1f1b.json, big_profile_staggered_1b1f.json
    big_trace_<sched>.json   (op-level spans)
Open the profiles at https://ui.perfetto.dev
"""

import time

import torch
import torch.nn as nn
from torch.distributed.pipelining import SplitPoint

from ramtorch import run_pipeline_relay

# ── Model ─────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    """Pre-norm FFN block: Linear(d,4d)->GELU->Linear(4d,d)+residual."""

    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d * 4)
        self.fc2 = nn.Linear(d * 4, d)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc2(torch.nn.functional.gelu(self.fc1(h)))
        return x + h


class Net(nn.Module):
    def __init__(self, d: int, n_layers: int, out_dim: int):
        super().__init__()
        self.proj = nn.Linear(d, d)
        self.layers = nn.ModuleList([FFN(d) for _ in range(n_layers)])
        self.head = nn.Linear(d, out_dim)

    def forward(self, x):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


# ── Config ────────────────────────────────────────────────────────────────────

DIM = 4096          # wide model -> big matmuls
N_LAYERS = 20       # 10 layers per stage (2 stages)
BATCH = 256
N_MICRO = 16        # many microbatches -> clear steady-state pattern
WARMUP_ITERS = 2    # run a couple of un-profiled steps first
DEVICES = ["cuda:1", "cuda:3"]

# 20 layers -> 10 per stage: split at layers.10
SPLIT_SPEC = {"layers.10": SplitPoint.BEGINNING}
loss_fn = nn.MSELoss()


def fresh_model() -> nn.Module:
    torch.manual_seed(1234)
    return Net(d=DIM, n_layers=N_LAYERS, out_dim=DIM)


def main() -> int:
    free = [i for i in (1, 3) if i < torch.cuda.device_count()]
    if len(free) < 2:
        raise SystemExit("need cuda:1 and cuda:3 free for this benchmark")

    n_params = sum(p.numel() for p in fresh_model().parameters())
    print(f"model params: {n_params/1e6:.0f}M ({n_params*4/1e9:.2f} GB fp32)  "
          f"({N_LAYERS} layers, dim {DIM})")
    print(f"pipeline: 2 stages on {DEVICES}, {N_MICRO} microbatches, batch {BATCH}\n")

    torch.manual_seed(0)
    data = torch.randn(BATCH, DIM)
    targets = torch.randn(BATCH, DIM)

    ref_loss = None
    for sched in ["gpipe", "1f1b", "staggered_1b1f"]:
        # Warmup (steady state, primes CUDA/cudnn autotune) — not profiled.
        for _ in range(WARMUP_ITERS):
            m = fresh_model()
            run_pipeline_relay(
                m, example_input=data[: BATCH // N_MICRO], split_spec=SPLIT_SPEC,
                data=data, targets=targets, schedule=sched, n_microbatches=N_MICRO,
                loss_fn=loss_fn, devices=DEVICES, overlap=True,
            )

        # Profiled + traced run.
        m = fresh_model()
        for d in set(torch.device(x) for x in DEVICES):
            torch.cuda.synchronize(d)
        t0 = time.perf_counter()
        result = run_pipeline_relay(
            m, example_input=data[: BATCH // N_MICRO], split_spec=SPLIT_SPEC,
            data=data, targets=targets, schedule=sched, n_microbatches=N_MICRO,
            loss_fn=loss_fn, devices=DEVICES, overlap=True,
            trace_path=f"big_trace_{sched}.json",
            profile_path=f"big_profile_{sched}.json",
        )
        for d in set(torch.device(x) for x in DEVICES):
            torch.cuda.synchronize(d)
        dt = (time.perf_counter() - t0) * 1e3

        loss = result.loss.item()
        if ref_loss is None:
            ref_loss = loss
        ok = "OK " if abs(loss - ref_loss) < 1e-4 * abs(ref_loss) else "DIFF"
        print(f"[{ok}] {sched:<16} loss={loss:.6f}  wall={dt:7.1f}ms  "
              f"-> big_profile_{sched}.json")

    print("\nOpen big_profile_*.json at https://ui.perfetto.dev")
    print("Compare the steady-state packing of the three schedules.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
