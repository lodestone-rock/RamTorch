import os
import torch
import torch.nn as nn
import torch.nn.functional as F

TRANSFER_STREAM = torch.cuda.Stream()
MAX_INFLIGHT = int(os.getenv("MAX_INFLIGHT", 2))
PENDING_EVENTS = []


class BouncingLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight_cpu, bias_cpu, device="cuda"):
        global PENDING_EVENTS

        # enqueue transfer on transfer stream
        with torch.cuda.stream(TRANSFER_STREAM):
            w = weight_cpu.to(device, non_blocking=True)
            b = bias_cpu.to(device, non_blocking=True) if bias_cpu is not None else None

            # record event after transfer
            evt = torch.cuda.Event()
            evt.record(TRANSFER_STREAM)
            PENDING_EVENTS.append(evt)

        # throttle: wait if too many inflight
        if len(PENDING_EVENTS) > MAX_INFLIGHT:
            PENDING_EVENTS[0].synchronize()
            PENDING_EVENTS.pop(0)

        # make compute stream wait for this transfer
        torch.cuda.current_stream().wait_event(evt)

        out = F.linear(x, w, b)

        # save for backward
        ctx.save_for_backward(x, weight_cpu, bias_cpu)
        ctx.device = device
        return out

    @staticmethod
    def backward(ctx, grad_out):
        global PENDING_EVENTS
        x, weight_cpu, bias_cpu = ctx.saved_tensors
        device = ctx.device

        # transfer weights on transfer stream
        with torch.cuda.stream(TRANSFER_STREAM):
            w = weight_cpu.to(device, non_blocking=True)
            evt = torch.cuda.Event()
            evt.record(TRANSFER_STREAM)
            PENDING_EVENTS.append(evt)

        if len(PENDING_EVENTS) > MAX_INFLIGHT:
            PENDING_EVENTS[0].synchronize()
            PENDING_EVENTS.pop(0)

        torch.cuda.current_stream().wait_event(evt)

        grad_input = grad_out @ w
        grad_weight = grad_out.t() @ x
        grad_bias = grad_out.sum(0) if bias_cpu is not None else None

        return grad_input, grad_weight, grad_bias, None


class CPUBouncingLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, device="cuda"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.device = device

        # parameters live on CPU
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, device="cpu").share_memory_()
        )
        self.bias = (
            nn.Parameter(torch.empty(out_features, device="cpu").share_memory_())
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
        return BouncingLinearFn.apply(x, self.weight, self.bias, self.device)
