"""
CPU Linear Module

A memory-efficient linear layer implementation that keeps parameters on CPU
and transfers them to GPU on-demand using asynchronous CUDA streams.

This approach interleave compute and data transfer, making it useful for:
- Very large models that don't fit in GPU memory
- Scenarios where GPU memory is limited but CPU memory is abundant
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

# Global CUDA stream for asynchronous weight transfers
# Using a dedicated stream allows transfers to overlap with computation
TRANSFER_STREAM = torch.cuda.Stream()

# --- Forward Pass Synchronization Primitives ---
TRANSFER_FORWARD_FINISHED_EVENT = torch.cuda.Event()
COMPUTE_FORWARD_START_EVENT = torch.cuda.Event()
W_BUFFERS = [None, None]
B_BUFFERS = [None, None]

# --- Backward Pass Synchronization Primitives ---
TRANSFER_BACKWARD_FINISHED_EVENT = torch.cuda.Event()
COMPUTE_BACKWARD_START_EVENT = torch.cuda.Event()
W_GRAD_BUFFERS = [None, None]

# buffer clock, tick toc!
FORWARD_BUFFER_CLK = 0
BACKWARD_BUFFER_CLK = 0


class BouncingLinearFn(torch.autograd.Function):
    """
    Custom autograd function implementing the bouncing linear operation.

    This function handles:
    1. Asynchronous transfer of weights from CPU to GPU
    2. Throttling of concurrent transfers to manage memory
    3. Proper synchronization between transfer and compute streams
    """

    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device="cuda"):
        """
        Forward pass of bouncing linear layer.

        Args:
            ctx: PyTorch autograd context for saving backward pass info
            x (torch.Tensor): Input tensor on GPU
            weight_cpu (torch.Tensor): Weight matrix stored on CPU
            bias_cpu (torch.Tensor, optional): Bias vector stored on CPU
            device (str): Target GPU device for computation

        Returns:
            torch.Tensor: Linear transformation output (x @ weight.T + bias)

        Flow:
            1. Initiate async transfer of weights to GPU
            2. Record completion event and add to pending queue
            3. Throttle if too many transfers are in-flight
            4. Wait for transfer completion before computation
            5. Perform linear operation and return result
        """
        global TRANSFER_STREAM, TRANSFER_FORWARD_FINISHED_EVENT, COMPUTE_FORWARD_START_EVENT, FORWARD_BUFFER_CLK, W_BUFFERS, B_BUFFERS

        # get index from clock
        selected_buffer = FORWARD_BUFFER_CLK

        # enqueue transfer on transfer stream
        with torch.cuda.stream(TRANSFER_STREAM):
            # if it's a first time, it's a no-op
            # wait for compute event to finish first
            TRANSFER_STREAM.wait_event(COMPUTE_FORWARD_START_EVENT)

            # alternate between buffers to prevent race condition where the transfer stream
            # overwriting the weight buffers before the main stream finish calculating the value 
            W_BUFFERS[selected_buffer] = weight_cpu.to(device, non_blocking=True)
            B_BUFFERS[selected_buffer] = (
                bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None
            )

            # flip the clock!
            FORWARD_BUFFER_CLK ^= 1
            # record event after transfer is done
            TRANSFER_FORWARD_FINISHED_EVENT.record()

        # make compute stream wait for this transfer
        torch.cuda.current_stream().wait_event(TRANSFER_FORWARD_FINISHED_EVENT)

        # mark the start of compute event
        COMPUTE_FORWARD_START_EVENT.record()
        out = F.linear(x, W_BUFFERS[selected_buffer], B_BUFFERS[selected_buffer])

        # save for backward
        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.device = device
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        Backward pass for gradient computation.

        Args:
            ctx: Autograd context containing saved forward pass data
            grad_out (torch.Tensor): Gradient w.r.t. layer output

        Returns:
            tuple: Gradients w.r.t. (input, weight, bias, device)
                  Device gradient is None (not differentiable)

        Note:
            Weights need to be transferred again for gradient computation
            since they're not kept on GPU between forward and backward passes.
        """
        global TRANSFER_STREAM, TRANSFER_BACKWARD_FINISHED_EVENT, COMPUTE_BACKWARD_START_EVENT, BACKWARD_BUFFER_CLK, W_GRAD_BUFFERS

        # get index from clock
        selected_buffer = BACKWARD_BUFFER_CLK

        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device

        # transfer weights on transfer stream
        with torch.cuda.stream(TRANSFER_STREAM):
            # if it's a first time, it's a no-op
            # wait for compute event to finish first
            TRANSFER_STREAM.wait_event(COMPUTE_BACKWARD_START_EVENT)

            # alternate between buffers to prevent race condition where the transfer stream
            # overwriting the weight buffers before the main stream finish calculating the value 
            W_GRAD_BUFFERS[selected_buffer] = weight_cpu.to(device, non_blocking=True)

            # flip the clock!
            BACKWARD_BUFFER_CLK ^= 1
            # record when transfer is done
            TRANSFER_BACKWARD_FINISHED_EVENT.record()

        # Make the compute stream wait for the weight transfer to complete
        torch.cuda.current_stream().wait_event(TRANSFER_BACKWARD_FINISHED_EVENT)

        # mark the start of compute event
        COMPUTE_BACKWARD_START_EVENT.record()

        # Compute gradients
        grad_input = grad_out @ W_GRAD_BUFFERS[selected_buffer]
        # TODO: maybe stream this
        grad_weight = (grad_out.mT @ x).to("cpu")
        grad_bias = grad_out.sum(dim=0).to("cpu") if bias_cpu is not None else None
        return grad_input, grad_weight, grad_bias, None


class CPUBouncingLinear(nn.Module):
    """
    Linear layer with CPU-stored parameters that bounce to GPU on demand.

    This module provides a drop-in replacement for nn.Linear but with different
    memory characteristics:
    - Parameters stored on CPU (using shared memory for multiprocessing)
    - Transferred to GPU only during forward/backward passes
    - Automatic cleanup after each operation

    Trade-offs:
    + Drastically reduced GPU memory usage
    + Enables training much larger models
    - Requires batching to mask the latency

    Best suited for:
    - Models too large for GPU memory
    - Inference scenarios with memory constraints
    """

    def __init__(self, in_features, out_features, bias=True, device="cuda"):
        """
        Initialize CPU linear layer.

        Args:
            in_features (int): Input feature dimension
            out_features (int): Output feature dimension
            bias (bool): Whether to include learnable bias term
            device (str): Target GPU device for computation

        Note:
            Parameters are initialized on CPU with proper weight initialization.
            share_memory_() enables efficient sharing in multiprocessing contexts.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        # parameters live on CPU
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device="cpu").share_memory_().pin_memory()
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, device="cpu").share_memory_().pin_memory())
            if bias
            else None
        )

        # init
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = in_features
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        """
        Forward pass through CPU linear layer.

        Args:
            x (torch.Tensor): Input tensor (should be on GPU)

        Returns:
            torch.Tensor: Linear transformation output

        Note:
            Input tensor should already be on the target GPU device.
            The autograd function handles all weight transfer logic.
        """
        return BouncingLinearFn.apply(x, self.weight, self.bias, self.device)


Linear = CPUBouncingLinear
