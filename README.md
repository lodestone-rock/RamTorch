# RamTorch

**RAM is All You Need** - A PyTorch library for memory-efficient deep learning that enables training and inference of large models that don't fit in GPU memory.

## Overview

RamTorch provides CPU-GPU hybrid implementations of neural network components that keep parameters in CPU memory and transfer them to GPU on-demand. This approach dramatically reduces GPU memory usage while maintaining computational efficiency through asynchronous CUDA streams and intelligent batching.

## Key Features

- **Memory-Efficient Linear Layers**: CPU-stored parameters with on-demand GPU transfer
- **Asynchronous CUDA Streams**: Overlap computation with data transfer for minimal latency
- **ZeRO-1 Optimizer Support**: Distributed optimizer state sharding across multiple GPUs
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

Replace `torch.nn.Linear` with `ramtorch.modules.Linear` for automatic memory optimization:

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

### ZeRO-1 Optimizer Sharding

For distributed training with optimizer state sharding:

```python
import torch.distributed as dist
from ramtorch.zero1 import create_zero_param_groups, broadcast_zero_params

# Initialize distributed training
dist.init_process_group(backend='nccl')
model = YourModel()
all_params = list(model.parameters())
rank = dist.get_rank()
world_size = dist.get_world_size()

# Create ZeRO-1 sharded optimizer
param_groups = [{'params': all_params, 'lr': 1e-3, 'weight_decay': 0.01}]
rank_param_groups = create_zero_param_groups(param_groups, world_size)
optimizer = torch.optim.AdamW(sharded_groups[rank]) # only optimize the shard

# Scheduler works normally with sharded optimizer
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# Training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward/backward with gradient accumulation
        for micro_batch in split_batch(batch):
            loss = model(micro_batch)
            loss.backward()

        # All-reduce gradients across ranks (you need to implement this)
        all_reduce_gradients(all_params)
        
        # Each rank updates only its owned parameters
        optimizer.step()
        
        # Broadcast updated parameters from owners to all ranks
        broadcast_zero_params(rank_param_groups)
        
        # It has to be model.zero_grad()! because optimizer on each rank only handles its own shard
        model.zero_grad()
        scheduler.step()
```

## Performance Considerations

### When to Use RamTorch

**Best suited for:**
- Large models that don't fit in GPU memory
- Inference scenarios with memory constraints
- Training with limited GPU memory but abundant CPU memory
- Distributed training with many parameters

**Less suitable for:**
- Small models that fit comfortably in GPU memory
- Scenarios where CPU-GPU bandwidth is the bottleneck
- Real-time applications requiring minimal latency

### Optimization Tips

1. **Use Larger Batch Sizes**: Helps amortize transfer costs
2. **Mixed Precision**: Combine with `torch.cuda.amp` for additional memory savings
3. **Strategic Placement**: Use RamTorch layers for the largest components only

## Architecture

### CPU Bouncing Linear Layer


1. Stores parameters on CPU memory (with `share_memory_()` for multiprocessing)
2. Asynchronously transfers weights to GPU during forward pass
3. Uses CUDA events for proper stream synchronization

### Memory Flow

```
CPU Memory (Parameters) → Transfer Stream → GPU Memory (Computation) → Result
                     ↑                                                      ↓
                     └────── Cleanup after computation ←──────────────────┘
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

Built on top of PyTorch's excellent automatic differentiation and CUDA stream management capabilities.
