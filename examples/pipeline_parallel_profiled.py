"""pipeline_parallel_profiled.py — 1F1B pipeline-parallel residual FFN with torch.profiler.

Splits a large residual feed-forward network across GPUs using
torch.distributed.pipelining with a 1F1B (one-forward-one-backward) schedule,
runs mixed-precision (bf16) training with random inputs/targets, and profiles
each rank in steady state.

Run (2 GPUs):
    torchrun --nproc_per_node=2 examples/pipeline_parallel_profiled.py

Run (4 GPUs):
    torchrun --nproc_per_node=4 examples/pipeline_parallel_profiled.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.pipelining import PipelineStage, Schedule1F1B
from torch.profiler import ProfilerActivity, profile, schedule, tensorboard_trace_handler


# ---------------------------------------------------------------------------
# Model definition — large residual FFN
# ---------------------------------------------------------------------------

HIDDEN = 4096
N_LAYERS_PER_STAGE = 6  # layers assigned to each pipeline stage


class ResidualFFNBlock(nn.Module):
    """Single residual feed-forward block: Linear -> GELU -> Linear + skip."""

    def __init__(self, dim: int):
        super().__init__()
        self.up = nn.Linear(dim, dim * 4, bias=False)
        self.down = nn.Linear(dim * 4, dim, bias=False)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        x = self.up(x)
        x = nn.functional.gelu(x)
        x = self.down(x)
        return x + residual


class ResidualFFNStage(nn.Module):
    """A stack of ResidualFFNBlocks — one pipeline stage."""

    def __init__(self, dim: int, n_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([ResidualFFNBlock(dim) for _ in range(n_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    # --- Distributed setup ---
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)

    print(f"[Rank {rank}/{world_size}] Pipeline parallel profiled example starting.")

    # --- Build per-stage model ---
    torch.manual_seed(42 + rank)
    stage_model = ResidualFFNStage(HIDDEN, N_LAYERS_PER_STAGE).to(device)

    # --- Mixed precision: use bfloat16 for compute ---
    stage_model = stage_model.to(torch.bfloat16)

    # --- Optimizer ---
    optimizer = torch.optim.AdamW(stage_model.parameters(), lr=1e-4, weight_decay=0.01)

    # --- Pipeline stage setup ---
    # Microbatch shape: (microbatch_size, seq_len, hidden)
    BATCH_SIZE = 32
    SEQ_LEN = 512
    N_MICROBATCHES = 4  # split batch into microbatches for pipeline
    MICROBATCH_SIZE = BATCH_SIZE // N_MICROBATCHES

    # Loss function (only meaningful on last stage, but we define it for the schedule)
    def loss_fn(output, target):
        return nn.functional.mse_loss(output, target)

    # Create the PipelineStage wrapper
    # input_args is needed for shape inference so the stage knows the activation
    # tensor shapes for P2P communication setup.
    example_input = torch.randn(MICROBATCH_SIZE, SEQ_LEN, HIDDEN, device=device, dtype=torch.bfloat16)
    stage = PipelineStage(
        stage_model,
        stage_index=rank,
        num_stages=world_size,
        device=device,
        input_args=(example_input,),
        output_args=(example_input,),  # same shape in/out for residual FFN
    )

    # Create the 1F1B schedule (loss_fn goes in the constructor)
    pipe_schedule = Schedule1F1B(
        stage,
        n_microbatches=N_MICROBATCHES,
        loss_fn=loss_fn if rank == world_size - 1 else None,
    )

    # Ensure all ranks are ready before starting the loop
    dist.barrier()
    if rank == 0:
        print("All ranks ready, starting training loop.", flush=True)

    # --- Profiler setup ---
    # We run a few warmup steps outside the profiler, then profile steady-state.
    WARMUP_STEPS = 3   # unprofiled warmup to stabilize CUDA caches / NCCL
    PROFILED_STEPS = 10  # steps recorded by the profiler
    TOTAL_STEPS = WARMUP_STEPS + PROFILED_STEPS

    trace_dir = f"tb_trace/pipeline_rank{rank}"
    os.makedirs(trace_dir, exist_ok=True)

    prof_schedule = schedule(
        wait=WARMUP_STEPS,       # skip the warmup steps
        warmup=1,                # 1 profiler-internal warmup step
        active=PROFILED_STEPS - 1,  # record the rest
        repeat=1,
    )

    # --- Training loop with profiling ---
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler(trace_dir),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,  # avoid massive overhead that causes hangs on trace export
    ) as prof:
        for step in range(TOTAL_STEPS):
            # Random input (only first stage feeds real data)
            if rank == 0:
                x = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN, device=device, dtype=torch.bfloat16)
            else:
                x = None

            # Random target for last stage
            if rank == world_size - 1:
                target = torch.randn(BATCH_SIZE, SEQ_LEN, HIDDEN, device=device, dtype=torch.bfloat16)
            else:
                target = None

            # --- Forward + Backward through 1F1B pipeline ---
            optimizer.zero_grad()

            losses = []
            # Only the first stage passes input data as a positional arg.
            # Other stages must pass NO positional args (not even None) so
            # _split_inputs sees an empty tuple and creates empty microbatch
            # placeholders — the actual activations arrive via P2P comms.
            if rank == 0:
                pipe_schedule.step(x, target=target, losses=losses)
            else:
                pipe_schedule.step(target=target, losses=losses)

            # --- Optimizer step (all ranks have their local grads) ---
            optimizer.step()

            # Sync all CUDA work before signalling profiler — prevents NCCL
            # deadlocks when one rank is busy exporting a trace while the
            # other is already in the next pipeline step.
            torch.cuda.synchronize()
            prof.step()

            if rank == world_size - 1 and losses:
                avg_loss = sum(l.item() for l in losses) / len(losses)
                print(f"  [Step {step}] loss = {avg_loss:.4f}", flush=True)
            elif rank == 0:
                print(f"  [Step {step}] done", flush=True)

    print(f"[Rank {rank}] Profiling complete. Traces saved to {trace_dir}/")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
