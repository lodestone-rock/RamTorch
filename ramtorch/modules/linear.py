"""
CPU Linear Module

A memory-efficient linear layer implementation that keeps parameters on CPU
and transfers them to GPU on-demand using asynchronous CUDA streams.

This approach interleave compute and data transfer, making it useful for:
- Very large models that don't fit in GPU memory
- Scenarios where GPU memory is limited but CPU memory is abundant
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Per-device global state registry ---
_DEVICE_STATE = {}


def _get_device_state(device=torch.cuda.current_device()):
    """Get or initialize per-device state."""
    if isinstance(device, str):
        device = torch.device(device)

    if device not in _DEVICE_STATE:
        with torch.cuda.device(device):
            _DEVICE_STATE[device] = {
                # streams & events
                "transfer_stream": torch.cuda.Stream(device=device),
                "transfer_forward_finished_event": torch.cuda.Event(),
                "compute_forward_start_event": torch.cuda.Event(),
                "transfer_backward_finished_event": torch.cuda.Event(),
                "compute_backward_start_event": torch.cuda.Event(),
                # buffers
                "w_buffers": [None, None],
                "b_buffers": [None, None],
                "w_grad_buffers": [None, None],
                # clocks
                "forward_clk": 0,
                "backward_clk": 0,
            }
    return _DEVICE_STATE[device]


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
        state = _get_device_state(device)
        transfer_stream = state["transfer_stream"]
        w_buffers = state["w_buffers"]
        b_buffers = state["b_buffers"]
        transfer_forward_finished_event = state["transfer_forward_finished_event"]
        compute_forward_start_event = state["compute_forward_start_event"]

        # get index from clock
        selected_buffer = state["forward_clk"]

        # enqueue transfer on transfer stream
        with torch.cuda.stream(transfer_stream):
            # if it's a first time, it's a no-op
            # wait for compute event to finish first
            transfer_stream.wait_event(compute_forward_start_event)

            # alternate between buffers to prevent race condition where the transfer stream
            # overwriting the weight buffers before the main stream finish calculating the value
            w_buffers[selected_buffer] = weight_cpu.to(device, non_blocking=True)
            b_buffers[selected_buffer] = (
                bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None
            )

            # flip the clock!
            state["forward_clk"] ^= 1
            # record event after transfer is done
            transfer_forward_finished_event.record()

        # make compute stream wait for this transfer
        torch.cuda.current_stream().wait_event(transfer_forward_finished_event)

        # mark the start of compute event
        compute_forward_start_event.record()
        out = F.linear(x, w_buffers[selected_buffer], b_buffers[selected_buffer])

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
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device
        state = _get_device_state(device)
        transfer_stream = state["transfer_stream"]
        w_grad_buffers = state["w_grad_buffers"]
        transfer_backward_finished_event = state["transfer_backward_finished_event"]
        compute_backward_start_event = state["compute_backward_start_event"]

        # get index from clock
        selected_buffer = state["backward_clk"]

        # transfer weights on transfer stream
        with torch.cuda.stream(transfer_stream):
            # if it's a first time, it's a no-op
            # wait for compute event to finish first
            transfer_stream.wait_event(compute_backward_start_event)

            # alternate between buffers to prevent race condition where the transfer stream
            # overwriting the weight buffers before the main stream finish calculating the value
            w_grad_buffers[selected_buffer] = weight_cpu.to(device, non_blocking=True)

            # flip the clock!
            state["backward_clk"] ^= 1
            # record when transfer is done
            transfer_backward_finished_event.record()

        # Make the compute stream wait for the weight transfer to complete
        torch.cuda.current_stream().wait_event(transfer_backward_finished_event)

        # mark the start of compute event
        compute_backward_start_event.record()

        # Compute gradients
        grad_input = grad_out @ w_grad_buffers[selected_buffer]
        grad_weight = grad_out.flatten(0, -2).T @ x.flatten(0, -2)
        if bias_cpu is not None:
            # sum over all batch-like dims, keep only last dim (Out)
            reduce_dims = tuple(range(grad_out.ndim - 1))
            grad_bias = grad_out.sum(dim=reduce_dims)
        else:
            grad_bias = None

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

    def __init__(self, in_features, out_features, bias=True, dtype=None, device="cuda"):
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
        if dtype is None:
            dtype = torch.float32

        # parameters live on CPU
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=dtype, device="cpu")
            .share_memory_()
            .pin_memory()
        )
        self.bias = (
            nn.Parameter(
                torch.empty(out_features, dtype=dtype, device="cpu")
                .share_memory_()
                .pin_memory()
            )
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
