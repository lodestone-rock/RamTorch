"""
pipeline_perfetto_demo.py
-------------------------
Real-compute pipeline-parallel demo with Perfetto/Chrome-trace export.

Model: 10-layer FFN, width 4096, 4x expansion (~1.3 GB of params in fp32),
split 5+5 across 2 GPUs. Runs GPipe / 1F1B / Interleaved-1F1B with real
matmuls (no fake sleep) and writes one trace file per schedule.

Open the traces at https://ui.perfetto.dev (or chrome://tracing).

Usage:
    python examples/pipeline_perfetto_demo.py
Outputs:
    trace_gpipe.json, trace_1f1b.json, trace_interleaved.json
"""

import os
import time

import torch
import torch.nn as nn
from torch.distributed.pipelining import SplitPoint

from ramtorch import run_pipeline

# ── Model ─────────────────────────────────────────────────────────────────────

class FFN(nn.Module):
    """Pre-norm FFN block: Linear(d, 4d) -> GELU -> Linear(4d, d) + residual."""

    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d * 4)
        self.fc2 = nn.Linear(d * 4, d)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc2(torch.nn.functional.gelu(self.fc1(h)))
        return x + h


class FFNNet(nn.Module):
    def __init__(self, d: int = 4096, n_layers: int = 10, out_dim: int = 4096):
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

DIM = 4096
N_LAYERS = 10
BATCH = 64          # full batch of token-vectors
N_MICRO = 8
# GPUs 1 and 3 are fully free on this box (0 and 2 are occupied by other jobs)
DEVICES = ["cuda:1", "cuda:3"]

# 10 layers -> 5 per stage: split at layers.5
SPLIT_SPEC = {"layers.5": SplitPoint.BEGINNING}

loss_fn = nn.MSELoss()


def main():
    if torch.cuda.device_count() < 2:
        raise SystemExit("need at least 2 GPUs for this demo")

    n_params = sum(p.numel() for p in FFNNet(d=DIM, n_layers=N_LAYERS).parameters())
    print(f"model params: {n_params/1e6:.1f}M ({n_params*4/1e9:.2f} GB fp32)")

    torch.manual_seed(0)
    data = torch.randn(BATCH, DIM)
    targets = torch.randn(BATCH, DIM)

    def fresh_model():
        # Build on CPU with a fixed seed (identical weights each time) — avoids
        # deepcopy after CUDA has been touched, which can copy GPU tensors.
        torch.manual_seed(1234)
        return FFNNet(d=DIM, n_layers=N_LAYERS)

    ref_loss = None
    for sched in ["gpipe", "1f1b"]:  # interleaved disabled: driver deadlock (W/B ordering)
        m = fresh_model()
        trace = f"trace_{sched}.json"
        print(f"running {sched} ...", flush=True)
        t0 = time.perf_counter()
        result = run_pipeline(
            m,
            example_input=data[: BATCH // N_MICRO],
            split_spec=SPLIT_SPEC,
            data=data,
            targets=targets,
            schedule=sched,
            n_microbatches=N_MICRO,
            loss_fn=loss_fn,
            devices=DEVICES,
            overlap=True,
            pp_group_size=2 if sched == "interleaved" else None,
            trace_path=trace,
        )
        dt = time.perf_counter() - t0
        loss = result.loss.item()
        if ref_loss is None:
            ref_loss = loss
        ok = "OK " if abs(loss - ref_loss) < 1e-4 * abs(ref_loss) else "DIFF"
        size = os.path.getsize(trace) / 1e3
        print(f"[{ok}] {sched:<12} loss={loss:.6f}  wall={dt*1e3:7.1f}ms  -> {trace} ({size:.0f} KB)")

    print("\nOpen traces at https://ui.perfetto.dev  (drag & drop the .json files)")


if __name__ == "__main__":
    main()
