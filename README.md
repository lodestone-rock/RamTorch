# RamTorch

**RAM is All You Need** - A PyTorch library for memory-efficient deep learning that enables training and inference of large models that don't fit in GPU memory.

## Overview

RamTorch provides CPU-GPU hybrid implementations of neural network components that keep parameters in CPU memory and transfer them to GPU on-demand. This approach dramatically reduces GPU memory usage while maintaining computational efficiency through asynchronous CUDA streams and intelligent batching.

## Key Features

- **Memory-Efficient Linear Layers**: Parameters stored on CPU with on-demand GPU transfer
- **Asynchronous CUDA Streams**: Overlaps computation with data transfer for minimal latency
- **ZeRO-Style Distributed Training**: 
  - **ZeRO-1**: Optimizer state sharding across multiple GPUs
  - **ZeRO-2**: Gradient sharding with automatic reduction
- **Pipeline Parallelism**: Single-process pipeline-parallel training & inference across GPUs — no `torchrun`, no process groups — with GPipe / 1F1B / staggered-1B1F schedules
- **Shared CPU Memory**: Multi-GPU workers share the same CPU tensor storage
- **Drop-in Replacement**: Compatible with existing PyTorch code

## Installation

```bash
pip install ramtorch
```

Or install from source:

```bash
git clone https://github.com/lodestone-rock/RamTorch.git
cd RamTorch
pip install -e .
```

## Quick Start

### Basic Usage

Replace `torch.nn.Linear` with `ramtorch.Linear` for automatic memory optimization:

```python
import torch
from ramtorch import Linear

# Standard PyTorch approach (high GPU memory usage)
# linear = torch.nn.Linear(1000, 1000)

# RamTorch approach (low GPU memory usage)
linear = Linear(1000, 1000, device="cuda")

# Use exactly like a normal PyTorch layer
x = torch.randn(32, 1000, device="cuda")
output = linear(x)  # Parameters automatically transferred from CPU to GPU
```

### Building Models

```python
import torch.nn as nn
from ramtorch import Linear

class MemoryEfficientModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            Linear(1000, 2000),
            nn.ReLU(),
            Linear(2000, 2000),
            nn.ReLU(),
            Linear(2000, 100)
        )
    
    def forward(self, x):
        return self.layers(x)

model = MemoryEfficientModel()
```

### Converting Existing Models

Use the helper function to automatically replace all Linear layers:

```python
from ramtorch.helpers import replace_linear_with_ramtorch

# Your existing PyTorch model
model = YourExistingModel()

# Replace all nn.Linear layers with RamTorch Linear layers
model = replace_linear_with_ramtorch(model, rank=0)
model = model.to("cuda:0")
```

### Multi-GPU Training with ZeRO-1 and ZeRO-2

RamTorch implements ZeRO-style optimizations where multiple GPU workers share the same CPU parameter storage, dramatically reducing memory usage:

```python
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, DistributedSampler

from ramtorch import AdamW
from ramtorch.helpers import replace_linear_with_ramtorch
from ramtorch.zero1 import create_zero_param_groups, broadcast_zero_params
from ramtorch.zero2 import setup_grad_sharding_hooks

def train(rank, world_size, model):
    # Setup distributed process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # Replace Linear layers with RamTorch (shares CPU memory across workers)
    model = replace_linear_with_ramtorch(model, rank)
    model.to(rank)
    
    # Setup ZeRO-1: Shard optimizer states across workers
    # Each worker only maintains optimizer states for a subset of parameters
    all_params = list(model.parameters())
    param_groups = [{'params': all_params, 'lr': 1e-3, 'weight_decay': 0.01}]
    rank_param_groups = create_zero_param_groups(param_groups, world_size)
    
    # Setup ZeRO-2: Shard gradients across workers
    # Gradients are partitioned and only linked on the worker responsible for them
    setup_grad_sharding_hooks(rank_param_groups, rank)
    
    # Each worker's optimizer only handles its shard
    optimizer = AdamW(rank_param_groups[rank])
    
    # Scheduler works normally
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Setup distributed data loading
    dataset = YourDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    loader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    # Training loop
    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # Important for proper shuffling
        
        for batch in loader:
            inputs, targets = batch
            inputs = inputs.to(rank)
            targets = targets.to(rank)
            
            # Forward and backward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets) / world_size  # Scale loss
            
            # Synchronize before backward to ensure all workers are ready
            torch.cuda.synchronize()
            loss.backward()
            torch.cuda.synchronize()
            
            # Update parameters (each worker updates only its shard)
            optimizer.step()
            
            # Broadcast updated parameters to all workers
            # RamTorch parameters are already shared via CPU memory,
            # but standard PyTorch parameters need explicit broadcasting
            broadcast_zero_params(rank_param_groups)
            
            scheduler.step()
            
            # IMPORTANT: Use model.zero_grad(), not optimizer.zero_grad()
            # Each worker handles partial gradients, so we need to zero
            # gradients at the model level to properly clear all workers' buffers
            model.zero_grad()
            
            torch.cuda.synchronize()
    
    dist.destroy_process_group()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    
    # Model must be instantiated BEFORE spawning GPU workers
    # RamTorch shares CPU tensors across workers, so the model needs to exist
    # in the parent process before forking to enable proper memory sharing
    model = YourModel()
    
    mp.spawn(train, args=(world_size, model), nprocs=world_size)
```

## Pipeline Parallelism

RamTorch also provides **single-process pipeline parallelism**: split a model across multiple GPUs and train/infer pipeline-parallel **without `torchrun`, without process groups, without NCCL**. Each stage runs on its own GPU driven by one worker thread; activations and gradients are relayed between stages through lightweight thread-safe handoffs.

### Why

Pipeline parallelism splits a model *vertically* (by layers) so each GPU holds only a slice — complementary to RamTorch's CPU-offload (which reduces per-GPU memory) and ZeRO (which shards optimizer/gradients). Use it when the model is too big for one GPU but you want a simple single-process program instead of a distributed launcher.

### The `PipelineModel` API (single-GPU feel)

The easiest way in. Auto-splits your model into balanced stages and gives you a normal-looking `nn.Module`:

```python
import torch
from ramtorch import PipelineModel

# Your ordinary model (any nn.Module with top-level children to split on)
model = MyModel()

# One example microbatch for the tracer (same device as the model)
example = torch.randn(16, *input_shape)

# Auto-split across your GPUs (balanced by parameter count)
pipe = PipelineModel(model, example, devices=["cuda:0", "cuda:1"])

# Inference — one call, returns logits (arbitrary batch size OK)
logits = pipe.forward(images)

# Training — step + flush + a single optimizer over the wrapper
opt = torch.optim.Adam(pipe.parameters(), lr=1e-3)
for x, y in train_loader:
    result = pipe.step(x, targets=y,
                       schedule="staggered_1b1f",   # or "gpipe" / "1f1b"
                       n_microbatches=4,
                       loss_fn=torch.nn.CrossEntropyLoss())
    result.flush_grads()   # write mean-scaled microbatch grads into .grad
    opt.step()
    opt.zero_grad()
```

### Schedules

Two schedules are recommended for everyday use:

| Schedule | Bubble | Peak in-flight activations | Notes |
|---|---|---|---|
| `staggered_1b1f` (default) | **lowest** | ~`num_stages` | Backward-eager + staggered warmup; GPipe-level throughput at 1F1B-class memory. **Recommended.** |
| `gpipe` | fill + drain | all `n_microbatches` | Simple, highest memory; useful as a correctness baseline |

A third schedule, `1f1b` (textbook forward-first one-forward-one-backward), is kept **for education and comparison only** — it is *not* recommended for real runs. It exists to make the importance of execution order concrete: `1f1b` and `staggered_1b1f` compute the *same* math, but `1f1b` forwards before backwarding, which leaves a large steady-state bubble. On a 10.9 GB model `1f1b` reached only ~50% GPU utilization vs ~92-98% for `staggered_1b1f` — identical work, very different wall-clock, purely from op ordering. Compare them with the schedule simulator (`python -m ramtorch.schedule_simulator`) or `examples/benchmark_schedules.py`.

### Auto-split & heterogeneous GPUs

By default stages are balanced by parameter count. For machines with GPUs of different speed, weight the split so faster GPUs take more layers:

```python
pipe = PipelineModel(model, example,
                     devices=["cuda:0", "cuda:1"],
                     device_weights=[2.0, 1.0])   # cuda:0 is ~2x faster
```

Or pass an explicit `split_spec` for full manual control over where the model is cut.

### Notes & gotchas

- **Batch size for inference**: the traced stages are specialized to the example microbatch's batch size. `forward()` automatically chunks larger/smaller batches and pads the final partial chunk (emitting a silence-able `PipelinePaddingWarning` when padding is used).
- **Numerics**: microbatch gradient accumulation is *mean-of-microbatch-means*, bit-identical to sequential gradient accumulation. It differs from a single full-batch backward only by normal fp32 reduction-order noise (same as any gradient-accumulation setup).
- **Example reference**: see `examples/mnist_pipeline_example.py` for a complete, self-contained MNIST training + inference script.

## Performance Considerations

### When to Use RamTorch

**Best suited for:**
- Large models that don't fit in GPU memory
- Multi-GPU training where memory is the bottleneck
- Inference scenarios with memory constraints
- Training with limited GPU memory but abundant CPU memory and bandwidth
- Distributed training with many parameters

**Less suitable for:**
- Small models that fit comfortably in GPU memory
- Scenarios where CPU-GPU bandwidth is the bottleneck
- Real-time applications requiring minimal latency
- Single-batch inference where transfer overhead dominates

### Optimization Tips

1. **Use Larger Batch Sizes**: Helps amortize transfer costs across more computation
2. **Mixed Precision Training**: Combine with `torch.cuda.amp` for additional memory savings
3. **Strategic Placement**: Use RamTorch layers for the largest components only
4. **Gradient Checkpointing**: Combine with activation checkpointing to further reduce memory
5. **Multi-GPU Setup**: RamTorch's shared CPU memory makes multi-GPU training particularly efficient

## Architecture

### CPU-Offloaded Linear Layer

The core innovation of RamTorch:

1. Stores parameters on CPU memory with `share_memory_()` for zero-copy sharing across processes
2. Asynchronously transfers weights to GPU during forward pass using dedicated CUDA streams
3. Uses CUDA events for proper stream synchronization
4. Automatically cleans up GPU memory after computation

### Memory Flow

```
                    ┌─────────────────────────┐
                    │   CPU Memory (Shared)   │
                    │  Parameters stored once │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
            ┌───────▼────────┐       ┌────────▼────────┐
            │  GPU Worker 0  │       │  GPU Worker 1   │
            │ (Async Stream) │       │ (Async Stream)  │
            └───────┬────────┘       └────────┬────────┘
                    │                         │
                    │   Compute on GPU        │
                    │                         │
            ┌───────▼────────┐       ┌───────▼─────────┐
            │    Result 0    │       │    Result 1     │
            └────────────────┘       └─────────────────┘
                    │                         │
                    └────────────┬────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   Cleanup GPU Memory    │
                    └─────────────────────────┘
```

### ZeRO-Style Sharding

```
┌─────────────────────────────────────────────────────┐
│              Model Parameters (CPU Shared)          │
│  [P₀, P₁, P₂, P₃, P₄, P₅, P₆, P₇, P₈, P₉, ...]      │
└─────────────────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        │                │                │
┌───────▼────────┐ ┌─────▼────────┐ ┌─────▼────────┐
│ GPU Worker 0   │ │ GPU Worker 1 │ │ GPU Worker 2 │
│ (CPU mem map)  │ │(CPU mem map) │ │(CPU mem map) │
│ Optimizer for: │ │ Optimizer for│ │ Optimizer for│
│  P₀, P₁, P₂    │ │  P₃, P₄, P₅  │ │  P₆, P₇, P₈  │
│                │ │              │ │              │
│ Gradients for: │ │ Gradients for│ │ Gradients for│
│  P₀, P₁, P₂    │ │  P₃, P₄, P₅  │ │  P₆, P₇, P₈  │
└────────────────┘ └──────────────┘ └──────────────┘
```
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
