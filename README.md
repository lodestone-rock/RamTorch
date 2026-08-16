# RamTorch

**RAM is All You Need** — train big models on a single node. No datacenter, no NVLink, no `torchrun`.

RamTorch is a PyTorch library for people who have **a workstation with a GPU or two and a lot of RAM**, not a cluster. It lets you train and run models that don't fit in GPU memory by using the RAM you already have — on **one GPU** by streaming weights from RAM, and on **multiple plain PCIe GPUs** by pipelining across them in a single process.

If you've ever thought *"this model is almost small enough for my GPU"*, RamTorch is for you. The corporate-scale solutions (FSDP, DeepSpeed, Megatron) already exist and target NVLink/InfiniBand clusters; RamTorch targets the rest of us.

## The two ways RamTorch helps

| Your hardware | RamTorch feature | Idea |
|---|---|---|
| **One GPU** (+ lots of RAM) | **Weight streaming** (`OffloadModel`) | Keep weights in pinned RAM, stream them through a small sliding GPU window. Peak GPU weight memory ≈ `window + pin` chunks, no matter how big the model is. |
| **A few PCIe GPUs** (one node, no NVLink) | **Pipeline parallelism** (`PipelineModel` / `Pipeline`) | Split the model across GPUs and train/infer in a single process — no process groups, no NCCL, no launcher. One worker thread per GPU; activations/gradients relay between stages. |

Both are built for **PCIe bandwidth**, not NVLink — which is exactly what a single-node enthusiast box has.

## Key Features

- **Single-GPU Weight Streaming (`OffloadModel`)**: weights live in CPU pinned memory and stream through a sliding GPU window (with optional pinned layers). Train/infer models too big for one GPU.
- **Single-Process Pipeline Parallelism**: split a model across GPUs with GPipe / 1F1B / staggered-1B1F schedules — no `torchrun`, no process groups, no NCCL.
- **Pipeline + weight streaming combined**: a stage passed as a list of chunks streams its weights from CPU RAM through a small GPU window, prefetched in schedule order — for stages that don't fit their GPU.
- **Mixed precision (`autocast=`)**, **tuple stage outputs**, and a **grad-bypass escape hatch** for custom losses.
- **Design simulators**: predict makespan / GPU-utilization / memory *before* you run — `ramtorch.schedule_simulator` (pipeline), `ramtorch.offload_simulator` (streaming), and `ramtorch.pipeline_offload_simulator` (both combined).
- **ZeRO-1 / ZeRO-2 sharding** and the original per-layer `ramtorch.Linear` are kept for backward compatibility (see [Legacy](#legacy-zero-12--per-layer-ramtorchlinear)).

## Installation

```bash
pip install ramtorch
```

Or from source:

```bash
git clone https://github.com/lodestone-rock/RamTorch.git
cd RamTorch
pip install -e .
```

## Quick Start

### One GPU — stream weights from RAM

Dice your model into chunk modules (you choose the granularity — the same convention as `Pipeline(stage_modules=...)`), then let `OffloadModel` stream them through the GPU:

```python
import torch, torch.nn as nn, torch.nn.functional as F
from ramtorch import OffloadModel

# 1. Dice your model into chunks.
chunks = [nn.Sequential(nn.Linear(1024, 4096), nn.GELU(), nn.Linear(4096, 1024))
          for _ in range(24)]

# 2. Wrap: stream through a 2-slot GPU window, keep 4 chunks pinned resident.
model = OffloadModel(chunks, device="cuda:0", window=2, pin=4,
                     keep_activations=True)

# 3. Inference / training.
out = model(x)                                                  # streamed forward
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)  # fused CPU+GPU kernels
for x, y in loader:
    res = model.step(x, targets=y, loss_fn=F.cross_entropy)     # fwd+bwd
    model.flush_grads()                                         # acc -> .grad
    opt.step(); opt.zero_grad()
model.close()
```

**Peak GPU weight memory ≈ `(window + pin)` chunks.** A loader thread prefetches the next chunk over a dedicated H2D stream so the copy overlaps compute. Guide: **[docs/offload.md](docs/offload.md)** · Quickstart: `examples/offload_quickstart.py` · MNIST end-to-end: `examples/mnist_offload_example.py` · full study: `examples/offload_vs_plain_demo.py`.

### A few PCIe GPUs — pipeline across them

The easiest way: dice your model into an ordered list of chunk modules (same convention as `OffloadModel`) and hand the flat list to `Pipeline` — it splits the chunks across your GPUs for you, no tracer involved:

```python
import itertools, torch
from ramtorch import Pipeline

chunks = [Embed(...), *[Block(...) for _ in range(24)], Head(...)]  # your dicing

pipe = Pipeline(chunk_modules=chunks, devices=["cuda:0", "cuda:1"],
                offload=False)                    # resident stages; offload=True streams weights
opt = torch.optim.Adam(itertools.chain(*(s.params for s in pipe.stages)), lr=1e-3)

for x, y in train_loader:
    result = pipe.step(x, targets=y,
                       schedule="staggered_1b1f",   # or "gpipe" / "1f1b"
                       n_microbatches=4,
                       loss_fn=torch.nn.CrossEntropyLoss())
    result.flush_grads(); opt.step(); opt.zero_grad()
```

For **simple models**, `PipelineModel` auto-splits with `torch.export` and gives you a normal-looking `nn.Module` (`pipe.parameters()` works directly, `pipe.forward(images)` infers any batch size):

```python
from ramtorch import PipelineModel

pipe = PipelineModel(model, example_input, devices=["cuda:0", "cuda:1"])
logits = pipe.forward(images)                       # inference (any batch size)
result = pipe.step(x, targets=y, n_microbatches=4, loss_fn=loss_fn)  # training
```

For **complex architectures** the tracer can fail — the chunk-list form above already bypasses it, or partition whole stages yourself with `Pipeline(stage_modules=[stage0, stage1], ...)`.

Guide: **[docs/pipeline_parallel.md](docs/pipeline_parallel.md)** · Examples: `examples/mnist_pipeline_example.py` (auto-split MLP), `examples/mnist_pipeline_big_transformer_manual.py` (~85M transformer, chunk-diced).

---

## Single-GPU Weight Streaming (`OffloadModel`)

> **Full guide: [docs/offload.md](docs/offload.md)** — knobs, backward strategies, transfer- vs compute-bound regimes, the design simulator, and profiling.

Weights live in CPU **pinned** memory. A **loader thread** prefetches upcoming chunks into a GPU window of `window` slots over a dedicated H2D stream, so the copy for chunk `i+1` overlaps the compute of chunk `i`. `pin` evenly-spaced chunks stay on the GPU permanently (never loaded, never evicted), easing PCIe traffic at the cost of their memory. Eviction is farthest-next-use (Belady — optimal, since the itinerary is known). Gradients accumulate **on the GPU** too (`grad_accum="stream"`, default): each streamed chunk's accumulator holds one of `acc_slots` GPU slots and spills/reloads over the copy streams like a weight — zero CPU arithmetic, and when slots suffice the grads cross PCIe only once per step at `flush_grads()` (legacy `grad_accum="cpu"` keeps the old per-microbatch D2H writeback + CPU-add path). Chunks exchange a single tensor or a **tuple** (elements become the next chunk's positional args; non-float extras like masks pass through grad-free).

### The three knobs

- **`window`** (default `2`): streaming slots on the GPU. `window=1` never overlaps a load with compute; `window≥2` lets the loader run ahead. Total weight memory ≈ `window + pin` chunks.
- **`pin`** (default `0`): evenly-spaced chunks pinned resident. `pin_layers=[...]` overrides with explicit indices.
- **`keep_activations`**: backward strategy —
  - `False` (*recompute*, default): forward runs under `no_grad` caching only chunk-boundary activations; backward reloads each chunk and recomputes its forward (per-chunk gradient checkpointing). Cheapest memory, but forward runs twice and **dropout would resample**.
  - `True` (*keep*): forward builds each chunk's graph and keeps its activations, so backward skips the recompute. Weights still stream (their storage is freed on eviction and refilled before backward — the FSDP resharding trick). Fastest, works with dropout, costs activation memory for all chunks.
  - `"checkpoint"`: keep-mode plumbing but each chunk runs under non-reentrant `torch.utils.checkpoint` — recompute-level memory that is **also dropout-safe** (checkpoint stashes/restores RNG), and it measured *faster* than the engine's own recompute. Head-to-head numbers: `examples/offload_checkpoint_study.py` and [docs/offload.md](docs/offload.md).
  - Selective: mark regions inside your own `forward` with `ramtorch.offload_checkpoint(self.heavy, x)` (+ `keep_activations=True`) — unmarked parts keep activations, marked parts recompute. A bare `torch.utils.checkpoint` breaks under the streaming engine; the helper re-applies the streamed weights for the recompute (see [docs/offload.md](docs/offload.md)).

### The optimizer step: use `fused=True`

The optimizer sees mixed devices (streamed masters on CPU, pinned chunks on the GPU). One `torch.optim.AdamW(model.parameters(), fused=True)` handles both: torch groups params by device, so the CPU masters get the **fused CPU kernel** — one multithreaded pass at DDR bandwidth, ~5x faster than the default eager CPU path, and faster than any PCIe-streaming scheme (we built one to check; it lives on as the private, educational `ramtorch.offload_optimizer.OffloadAdamW` — see [docs/offload.md](docs/offload.md) and the benchmark in `examples/offload_optimizer_check.py`).

### Mixed precision & grad bypass

- **Autocast needs no engine support** (unlike the pipeline's `autocast=` param): offload compute runs on the calling thread, so just wrap the step — `with torch.autocast("cuda", dtype=torch.bfloat16): model.step(...)`. Verified bit-identical to a full-resident model on the same recipe, in all three backward modes.
- **`step(x, grad_outputs=...)`** mirrors the pipeline's escape hatch: skip the loss and backprop a precomputed `dL/dOutput` (tensor, or callable resolved with the live output). Mutually exclusive with `loss_fn`; `res.loss` raises.

### Will it actually be faster? (the simulator)

Offload trades **PCIe bandwidth for GPU memory**. It wins when there's enough per-chunk compute to hide the H2D copy (the *compute-bound* regime) and loses when chunks are tiny/fast (*transfer-bound*). Predict which regime you're in *before* building anything:

```bash
# sweep window 1..N, print makespan / GPU% / stall / peak-resident table
python -m ramtorch.offload_simulator --layers 24 --tf 1.0 --tb 2.0 --th2d 0.5
# one window, Gantt chart
python -m ramtorch.offload_simulator --layers 24 --window 2 --pin 4 --plot gantt.png
```

Profile a real step with `model.step(..., profile_path="offload.trace.json")` and open it at <https://ui.perfetto.dev> — you get compute spans, H2D loads, D2H writebacks, and stall waits.

---

## Pipeline Parallelism

> **Full guide: [docs/pipeline_parallel.md](docs/pipeline_parallel.md)** — torch.export gotchas, manual splitting, tuple/pre-diced inputs, mixed precision, grad-bypass, schedules, profiling, and numerics.

Single-process pipeline parallelism for a single node of plain PCIe GPUs. Each stage runs on its own GPU driven by one worker thread; activations and gradients are relayed between stages through lightweight thread-safe handoffs (one mailbox per pipeline edge × microbatch, with a CUDA-event handshake).

### Schedules

| Schedule | Bubble | Peak in-flight activations | Notes |
|---|---|---|---|
| `staggered_1b1f` (default) | **lowest** | ~`num_stages` | Backward-eager + staggered warmup; GPipe-level throughput at 1F1B-class memory. **Recommended.** |
| `gpipe` | fill + drain | all `n_microbatches` | Simple, highest memory; correctness baseline |
| `1f1b` | steady-state | ~`num_stages` | Textbook forward-first. **Educational only.** |

`staggered_1b1f` and `1f1b` compute the *same* math, but `1f1b` forwards before backwarding, leaving a large steady-state bubble — on a 10.9 GB model ~50% GPU utilization vs ~92-98% for `staggered_1b1f`. Execution order matters. Compare them with `python -m ramtorch.schedule_simulator`.

### Auto-split & heterogeneous GPUs

Stages are balanced by parameter count by default. For GPUs of different speed, weight the split so faster GPUs take more layers:

```python
pipe = PipelineModel(model, example, devices=["cuda:0", "cuda:1"],
                     device_weights=[2.0, 1.0])   # cuda:0 is ~2x faster
```

Or pass an explicit `split_spec` for full manual control over where the model is cut.

### Mixed precision, tuple outputs, grad-bypass

- **Mixed precision**: `PipelineModel(..., autocast="bf16")` (or `Pipeline(..., autocast=torch.bfloat16)`). Each stage worker enters autocast around forward+loss; params/grads stay fp32. bf16 training is bit-identical to a sequential autocast baseline; fp16 is inference-only.
- **Tuple stage outputs & no-grad args**: a stage can return a tuple; non-float outputs (e.g. bool masks) are auto forward-only, and a module `out_no_grad=(i,...)` attribute marks float outputs forward-only. Backward flows only through grad-needing args.
- **Grad-bypass**: `pipe.step(..., grad_outputs=...)` backprops a supplied `dL/dOutput` directly into the last stage, skipping `loss_fn` — for downstream-model gradients, RL advantages, custom differentiators.

### Weight streaming inside a stage (pipeline + offload)

If a stage's shard doesn't fit its GPU, pass that stage as a **list of chunk modules** — it becomes an `OffloadStage`: masters live in CPU pinned RAM and stream through a `offload_window`-chunk GPU window, prefetched in the schedule's exact chunk order (the executor's static op list is announced to the loader before the step runs, so H2D copies overlap compute *and* pipeline bubbles; the `staggered_1b1f` F↔B turnaround reuses the resident chunk for free).

```python
pipe = Pipeline(stage_modules=[[Block() for _ in range(12)],  # list -> streamed
                               nn.Sequential(...)],           # module -> resident
                devices=["cuda:0", "cuda:1"],
                offload_window=2, offload_pin=0)

# or simplest: one flat chunk list, split across the GPUs for you
pipe = Pipeline(chunk_modules=[Block() for _ in range(32)],
                devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"],  # 8 each
                chunks_per_stage=[7, 8, 8, 9])  # optional explicit weighting

# the same dicing without streaming: offload=False -> resident stages
pipe = Pipeline(chunk_modules=chunks, devices=[...], offload=False)
```

GPU weight memory per streamed stage ≈ `(window + pin)` chunks. Bit-identical to the full-resident pipeline (`examples/pipeline_offload_check.py`); simulate your regime first with `python -m ramtorch.pipeline_offload_simulator`. No NVMe tier here, deliberately — pipeline training from disk would thrash the drive. See [docs/pipeline_parallel.md](docs/pipeline_parallel.md).

### Notes & gotchas

- **Inference batch size**: traced stages specialize to the example microbatch's batch size; `forward()` chunks and pads arbitrary batches (emitting a silence-able `PipelinePaddingWarning`).
- **Numerics**: microbatch grad accumulation is *mean-of-microbatch-means*, bit-identical to sequential grad accumulation (differs from a single full-batch backward only by normal fp32 reduction-order noise).
- **Which pipeline API**: `Pipeline(chunk_modules=[...])` (flat chunk list) is the least code and the recommended default; `PipelineModel` (traced auto-split) is convenient for simple models; `Pipeline(stage_modules=...)` gives full manual stage control for exotic architectures.

---

## Which feature should I reach for?

- **Model fits on one GPU but barely / OOMs** → `OffloadModel`, tune `window`/`pin` with the simulator.
- **Model too big for one GPU, you have 2+ PCIe GPUs on one node** → `Pipeline(chunk_modules=[...])` (recommended) or `PipelineModel` (simple auto-split). Per-GPU memory drops by ~`1/num_gpus`.
- **Even a pipeline stage doesn't fit its GPU** → pass that stage as a *list of chunks* (`Pipeline` + weight streaming): per-GPU weight memory drops to ~`(offload_window + offload_pin)` chunks. Check the regime with `python -m ramtorch.pipeline_offload_simulator` first — compute-bound stages hide the traffic (~0.4% simulated overhead), transfer-bound ones pay real slowdown for the memory.
- **Multi-node cluster with fast interconnect** → you probably want FSDP/DeepSpeed/Megatron instead; RamTorch targets single-node PCIe.

## Performance Considerations

**Best suited for:**
- Models that don't fit in GPU memory but fit in (or stream from) CPU RAM
- Single-node, multi-GPU PCIe machines without NVLink
- Training/inference with limited GPU memory but abundant CPU memory and PCIe bandwidth

**Less suitable for:**
- Small models that fit comfortably in GPU memory (offload/streaming overhead dominates)
- Tiny/fast layers where PCIe can't keep up with compute (transfer-bound — check with the simulator)
- Latency-critical single-batch inference
- Multi-node clusters (use FSDP/DeepSpeed/Megatron)

**Tips:**
1. Larger batch sizes amortize transfer costs.
2. Use bf16 autocast for extra memory savings.
3. For offload, tune `window`/`pin` and prefer the *compute-bound* regime (the simulator tells you).
4. Combine activation checkpointing with offload for the deepest memory cuts.

## Architecture

### Single-GPU weight streaming

```
                 ┌──────────────────────────────────┐
                 │   CPU pinned memory (weights)    │
                 └───────────────┬──────────────────┘
                                 │ H2D (loader thread, prefetch)
                    ┌────────────▼────────────┐
                    │  GPU window (W slots)   │◄──── pinned chunks (resident)
                    │  F0 F1 ... B(n-1) ... B0│
                    └────────────┬────────────┘
                                 │ backward adds (GPU, compute stream)
                    ┌────────────▼────────────┐
                    │ GPU grad accumulators   │
                    │     (acc_slots)         │
                    └────────────┬────────────┘
                                 │ D2H spill / H2D reload (slot pressure
                                 │ + one flush_grads() spill per step)
                    ┌────────────▼────────────┐
                    │  CPU grad accumulators  │
                    └─────────────────────────┘
```

### Pipeline parallelism

```
 GPU 0        GPU 1        GPU 2        GPU 3
┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐
│ stage0 │─►│ stage1 │─►│ stage2 │─►│ stage3 │   activations relay ──►
│  F B   │◄─│  F B   │◄─│  F B   │◄─│  F B   │   gradients  ◄── relay
└────────┘  └────────┘  └────────┘  └────────┘
 one worker thread per stage; one mailbox per (edge, microbatch)
```

---

## Legacy: ZeRO-1/2 & per-layer `ramtorch.Linear`

These are kept for backward compatibility but are **no longer the focus** — for a single-node PCIe box, prefer `OffloadModel` (one GPU) and pipeline parallelism (several GPUs).

### ZeRO-1 / ZeRO-2 (multi-process sharding)

ZeRO-1 shards optimizer state and ZeRO-2 shards gradients across multiple **process** workers that share the same CPU parameter storage. This is a `torch.distributed` (NCCL) multi-process design — powerful, but heavier than the single-process pipeline above and really aimed at many-GPU/multi-node setups. The full example lives in `examples/`; the short version:

```python
from ramtorch import AdamW
from ramtorch.helpers import replace_linear_with_ramtorch
from ramtorch.zero1 import create_zero_param_groups, broadcast_zero_params
from ramtorch.zero2 import setup_grad_sharding_hooks

model = replace_linear_with_ramtorch(model, rank).to(rank)   # shared CPU storage
param_groups = [{'params': list(model.parameters()), 'lr': 1e-3}]
rank_groups = create_zero_param_groups(param_groups, world_size)  # ZeRO-1
setup_grad_sharding_hooks(rank_groups, rank)                      # ZeRO-2
optimizer = AdamW(rank_groups[rank])
# ... train, then broadcast_zero_params(rank_groups) each step
```

### Per-layer `ramtorch.Linear` (deprecated)

`ramtorch.Linear` / `CPUBouncingLinear` bounced *individual* linear layers between CPU and GPU on CUDA streams. It's **deprecated** (`DeprecationWarning` on construction): `OffloadModel` supersedes it by streaming *user-diced chunks* (you control the granularity) with a proper loader/writeback overlap engine and pinned layers. Migrate by dicing your model into chunk modules and wrapping them in `OffloadModel`.

## Contributing

We welcome contributions! Please see our contributing guidelines for details.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use RamTorch in your research, please cite:

```bibtex
@software{ramtorch2025,
  author = {Lodestone},
  title = {RamTorch: Memory-Efficient Deep Learning with CPU-GPU Hybrid Architecture},
  url = {https://github.com/lodestone-rock/RamTorch},
  year = {2025}
}
```

## Acknowledgments

Built on top of PyTorch's excellent automatic differentiation and CUDA stream management capabilities. Inspired by Microsoft's ZeRO optimizer and DeepSpeed library.
