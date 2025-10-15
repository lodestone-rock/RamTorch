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

from torch.profiler import record_function  # just for profiling

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
                "transfer_grad_stream": torch.cuda.Stream(device=device),
                "transfer_forward_finished_event": torch.cuda.Event(),
                "compute_forward_start_event": torch.cuda.Event(),
                "transfer_backward_finished_event": torch.cuda.Event(),
                "transfer_weight_backward_start_event": torch.cuda.Event(),
                "transfer_weight_backward_finished_event": torch.cuda.Event(),
                "compute_backward_start_event": torch.cuda.Event(),
                "compute_backward_finished_event": torch.cuda.Event(),
                # buffers
                "w_buffers": [None, None],
                "b_buffers": [None, None],
                "w_bwd_buffers": [None, None],
                "w_grad_buffers": [None, None],
                "b_grad_buffers": [None, None],
                "w_grad_accum_buffers": [None, None],
                "b_grad_accum_buffers": [None, None],
                # clocks
                "forward_clk": 0,
                "backward_clk": 0,
            }
    return _DEVICE_STATE[device]


def _invoke_tensor_hooks(tensor, grad):
    """
    Invoke backward hooks registered on a tensor.

    Args:
        tensor: The parameter tensor that may have hooks
        grad: The gradient tensor to pass through hooks

    Returns:
        The potentially modified gradient after all hooks
    """
    # Standard backward hooks
    if hasattr(tensor, "_ramtorch_backward_hooks") and tensor._ramtorch_backward_hooks:
        for hook_id, hook_fn in tensor._ramtorch_backward_hooks.items():
            result = hook_fn(grad)
            if result is not None:
                grad = result

    return grad


def _invoke_post_accum_tensor_hooks(tensor):
    """
    Invoke post accumulate grad hooks registered on a tensor.

    Args:
        tensor: The parameter tensor that may have hooks

    """
    # Post-accumulate grad hooks (PyTorch 2.0+)
    if (
        hasattr(tensor, "_ramtorch_post_accumulate_grad_hooks")
        and tensor._ramtorch_post_accumulate_grad_hooks
    ):
        for hook_id, hook_fn in tensor._ramtorch_post_accumulate_grad_hooks.items():
            hook_fn(tensor)


class BouncingLinearFn(torch.autograd.Function):
    """
    Custom autograd function implementing the bouncing linear operation.

    This function handles:
    1. Asynchronous transfer of weights from CPU to GPU
    2. Proper synchronization between transfer and compute streams
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
            1. Select buffer using ping-pong clock (alternates between 0 and 1)
            2. On transfer stream: wait for previous compute to finish, then
               initiate async transfer of weights/bias to selected GPU buffer
            3. Record transfer completion event
            4. On compute stream: wait for transfer completion, then perform
               linear operation using the transferred weights
            5. Record compute start event for next iteration's synchronization

        The ping-pong buffering prevents race conditions by ensuring the
        transfer stream never overwrites buffers that the compute stream
        is still using.
        """
        state = _get_device_state(device)
        transfer_stream = state["transfer_stream"]
        w_buffers = state["w_buffers"]
        b_buffers = state["b_buffers"]
        transfer_forward_finished_event = state["transfer_forward_finished_event"]
        compute_forward_start_event = state["compute_forward_start_event"]

        # get index from clock
        selected_buffer = state["forward_clk"]
        # flip the clock!
        state["forward_clk"] ^= 1

        # enqueue transfer on transfer stream
        with torch.cuda.stream(transfer_stream):
            # if it's a first time, it's a no-op
            # wait for compute event to finish first
            transfer_stream.wait_event(compute_forward_start_event)

            with record_function(
                "forward_weight_bias_transfer"
            ):  # for profiling and easy debugging

                # alternate between buffers to prevent race condition where the transfer stream
                # overwriting the weight buffers before the main stream finish calculating the value
                w_buffers[selected_buffer] = weight_cpu.to(device, non_blocking=True)
                b_buffers[selected_buffer] = (
                    bias_cpu.to(device, non_blocking=True)
                    if bias_cpu is not None
                    else None
                )

            # record event after transfer is done
            transfer_forward_finished_event.record()

        with record_function(
            "forward_linear_compute"
        ):  # for profiling and easy debugging
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
            tuple: Gradients w.r.t. (input, weight, bias, device).
                   The gradients for weight and bias are returned as None
                   because they are handled manually.

        Gradient Handling Note:
            This function manually computes and accumulates gradients for the
            weight and bias parameters, which reside on the CPU. The process is:
            1. Asynchronously transfer the current gradients (if any) from
               CPU to GPU for accumulation.
            2. Compute new gradients on the GPU.
            3. Add the existing gradients to the new ones on the GPU.
            4. Invoke any registered backward hooks on the GPU gradients.
            5. Asynchronously transfer the final accumulated gradients back to
               the .grad attribute of the CPU parameters.

            Because this process bypasses the standard autograd mechanism for
            parameters, this function returns None for the weight and bias
            gradients. Hooks are manually invoked to maintain compatibility.
        """
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device
        state = _get_device_state(device)
        transfer_stream = state["transfer_stream"]
        transfer_grad_stream = state["transfer_grad_stream"]
        w_bwd_buffers = state["w_bwd_buffers"]
        w_grad_buffers = state["w_grad_buffers"]
        b_grad_buffers = state["b_grad_buffers"]
        w_grad_accum_buffers = state["w_grad_accum_buffers"]
        b_grad_accum_buffers = state["b_grad_accum_buffers"]
        transfer_backward_finished_event = state["transfer_backward_finished_event"]
        transfer_weight_backward_finished_event = state[
            "transfer_weight_backward_finished_event"
        ]
        compute_backward_start_event = state["compute_backward_start_event"]
        compute_backward_finished_event = state["compute_backward_finished_event"]

        # get index from clock
        selected_buffer = state["backward_clk"]
        # flip the clock!
        state["backward_clk"] ^= 1

        # transfer weights on transfer stream
        with torch.cuda.stream(transfer_stream):
            with record_function(
                "backward_weight_transfer"
            ):  # for profiling and easy debugging
                # if it's a first time, it's a no-op
                # wait for compute event to finish first
                transfer_stream.wait_event(compute_backward_start_event)

                # alternate between buffers to prevent race condition where the transfer stream
                # overwriting the weight buffers before the main stream finish calculating the value
                w_bwd_buffers[selected_buffer] = weight_cpu.to(
                    device, non_blocking=True
                )

            # load gradient for manual accumulation
            with record_function(
                "backward_grad_accumulator_transfer"
            ):  # for profiling and easy debugging
                w_grad_accum_buffers[selected_buffer] = (
                    weight_cpu.grad.to(device, non_blocking=True)
                    if weight_cpu.grad is not None
                    else None
                )
                b_grad_accum_buffers[selected_buffer] = (
                    bias_cpu.grad.to(device, non_blocking=True)
                    if bias_cpu is not None and bias_cpu.grad is not None
                    else None
                )
                # record when transfer is done
                transfer_backward_finished_event.record()

        # Make the compute stream wait for the weight transfer to complete
        torch.cuda.current_stream().wait_event(transfer_backward_finished_event)

        # mark the start of compute event
        compute_backward_start_event.record()

        with record_function(
            "backward_linear_compute"
        ):  # for profiling and easy debugging
            # compute input grad
            grad_input = grad_out @ w_bwd_buffers[selected_buffer]

            # this must launch after the transfer stream is done
            torch.cuda.current_stream().wait_event(
                transfer_weight_backward_finished_event
            )
            w_grad_buffers[selected_buffer] = grad_out.flatten(0, -2).T @ x.flatten(
                0, -2
            )

            # gradient accumulation is being performed directly
            with record_function(
                "backward_weight_grad_accumulate"
            ):  # for profiling and easy debugging

                # invoke grad hooks on weight gradient
                w_grad_buffers[selected_buffer] = _invoke_tensor_hooks(
                    weight_cpu, w_grad_buffers[selected_buffer]
                )

                if w_grad_accum_buffers[selected_buffer] is not None:

                    # accumulate weight
                    w_grad_buffers[selected_buffer] += w_grad_accum_buffers[
                        selected_buffer
                    ]

                # invoke post accum hooks on weight gradient
                # use temporary reference to simplify post accum ipl
                weight_cpu.ramtorch_grad = w_grad_buffers[selected_buffer]
                _invoke_post_accum_tensor_hooks(weight_cpu)
                del weight_cpu.ramtorch_grad

            if bias_cpu is not None:
                # sum over all batch-like dims, keep only last dim (Out)
                reduce_dims = tuple(range(grad_out.ndim - 1))
                b_grad_buffers[selected_buffer] = grad_out.sum(dim=reduce_dims)

                with record_function(
                    "backward_bias_grad_accumulate"
                ):  # for profiling and easy debugging

                    # invoke grad hooks on bias gradient
                    b_grad_buffers[selected_buffer] = _invoke_tensor_hooks(
                        bias_cpu, b_grad_buffers[selected_buffer]
                    )

                    if b_grad_accum_buffers[selected_buffer] is not None:

                        # accumulate bias
                        b_grad_buffers[selected_buffer] += b_grad_accum_buffers[
                            selected_buffer
                        ]

                    # invoke post accum hooks on bias gradient
                    # use temporary reference to simplify post accum ipl
                    bias_cpu.ramtorch_grad = b_grad_buffers[selected_buffer]
                    _invoke_post_accum_tensor_hooks(bias_cpu)
                    del bias_cpu.ramtorch_grad
            else:
                grad_bias = None
            compute_backward_finished_event.record()

        with record_function(
            "backward_grad_transfer"
        ):  # for profiling and easy debugging
            # directly populate the grad and do grad accumulation here instead of relying on autograd
            with torch.cuda.stream(transfer_grad_stream):
                transfer_grad_stream.wait_event(compute_backward_finished_event)

                # transfer_weight_backward_start_event.record()
                weight_cpu.grad = w_grad_buffers[selected_buffer].to(
                    "cpu", non_blocking=True
                )

                if bias_cpu is not None:
                    bias_cpu.grad = (
                        b_grad_buffers[selected_buffer].to("cpu", non_blocking=True)
                        if bias_cpu is not None
                        else None
                    )

                # record when transfer is done
                transfer_weight_backward_finished_event.record()

        return grad_input, None, None, None


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

    Gradient Handling and Autograd Hooks:
        This module uses a custom autograd function that manually computes and
        accumulates gradients for the weight and bias parameters. Backward hooks
        registered on weight/bias tensors ARE supported and will be invoked on
        the GPU before transferring gradients back to CPU.

        Hooks are called in this order:
        1. Gradient computation on GPU
        2. Gradient accumulation on GPU
        3. Backward hooks invoked on GPU (tensor._ramtorch_backward_hooks)
        4. Post-accumulate hooks invoked on GPU (tensor._ramtorch_post_accumulate_grad_hooks)
        5. Transfer modified gradient back to CPU
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

        # create a flag to indicate if the tensors are ramtorch tensors
        self.weight.is_ramtorch = True
        if bias:
            self.bias.is_ramtorch = True

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
