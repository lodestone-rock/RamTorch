import torch
from torch.optim import Optimizer


# @torch.compile
def copy_stochastic_(target: torch.Tensor, source: torch.Tensor):
    # thanks to Nerogar for fast stochastic pytorch implementation
    # https://github.com/pytorch/pytorch/issues/120376#issuecomment-1974828905
    with torch.no_grad():
        # create a random 16 bit integer
        result = torch.randint_like(
            source,
            dtype=torch.int32,
            low=0,
            high=(1 << 16),
        )

        # add the random number to the lower 16 bit of the mantissa
        result.add_(source.view(dtype=torch.int32))

        # mask off the lower 16 bit of the mantissa
        result.bitwise_and_(-65536)  # -65536 = FFFF0000 as a signed int32

        # copy the higher 16 bit into the target tensor
        target.copy_(result.view(dtype=torch.float32))


class AdamW(Optimizer):
    r"""
    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float):
            Learning rate parameter (default 0.0025)
        betas (Tuple[float, float], optional):
            coefficients used for computing running averages of
            gradient and its square (default: (0.9, 0.999)).
        eps (float):
            Term added to the denominator outside of the root operation to
            improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        centralization (float):
            center model grad (default: 0).
        chunk_size (int):
            Number of parameters to process before synchronizing.
            A larger chunk size can improve performance but uses more
            temporary GPU memory. (default: 16)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        centralization=0,
        chunk_size=64,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            centralization=centralization,
        )
        super(AdamW, self).__init__(params, defaults)

        self.chunk_size = chunk_size

        # Initialize state in pinned memory for faster async transfers
        for group in self.param_groups:
            for p in group["params"]:
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["ema"] = torch.zeros_like(
                        p.data, dtype=torch.bfloat16, device="cpu"
                    ).pin_memory()
                    state["ema_squared"] = torch.zeros_like(
                        p.data, dtype=torch.bfloat16, device="cpu"
                    ).pin_memory()

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            # Enumerate to keep track of the parameter index for chunking
            for i, p in enumerate(group["params"]):
                if p.grad is None:
                    continue

                assert p.dtype == torch.bfloat16, "only bfloat 16 is supported."
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Compass does not support sparse gradients")

                state = self.state[p]
                device = p.device

                # Lazy state initialization
                if not state:
                    state["step"] = 0
                    state["ema"] = torch.zeros_like(
                        p.data, dtype=torch.bfloat16, device="cpu"
                    ).pin_memory()
                    state["ema_squared"] = torch.zeros_like(
                        p.data, dtype=torch.bfloat16, device="cpu"
                    ).pin_memory()

                # ========= Asynchronously queue all operations for this parameter =========

                # 1. Queue Host-to-Device copy
                ema_gpu = state["ema"].to(device, non_blocking=True)
                ema_squared_gpu = state["ema_squared"].to(device, non_blocking=True)

                # 2. Queue computations on the GPU
                grad = grad.to(torch.float32)
                p_fp32 = p.clone().to(torch.float32)
                ema_fp32 = ema_gpu.to(torch.float32)
                ema_squared_fp32 = ema_squared_gpu.to(torch.float32)

                beta1, beta2 = group["betas"]
                lr = group["lr"]
                weight_decay = group["weight_decay"]
                centralization = group["centralization"]
                state["step"] += 1

                if centralization != 0:
                    grad.sub_(
                        grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True).mul_(
                            centralization
                        )
                    )

                bias_correction = 1 - beta1 ** state["step"]
                bias_correction_sqrt = (1 - beta2 ** state["step"]) ** (1 / 2)
                step_size = lr / bias_correction

                ema_fp32.mul_(beta1).add_(grad, alpha=1 - beta1)
                ema_squared_fp32.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                denom = (ema_squared_fp32.sqrt() / bias_correction_sqrt).add_(
                    group["eps"]
                )

                if weight_decay != 0:
                    p_fp32.data.mul_(1 - step_size * weight_decay)

                p_fp32.data.addcdiv_(ema_fp32, denom, value=-step_size)

                copy_stochastic_(p, p_fp32)
                copy_stochastic_(ema_gpu, ema_fp32)
                copy_stochastic_(ema_squared_gpu, ema_squared_fp32)

                # 3. Queue Device-to-Host copy
                state["ema"].copy_(ema_gpu, non_blocking=True)
                state["ema_squared"].copy_(ema_squared_gpu, non_blocking=True)

                # ========= Check if we need to synchronize =========
                # We synchronize after processing a chunk of parameters.
                # The (i + 1) ensures we sync after the 1st, 2nd, ... chunk.
                if (i + 1) % self.chunk_size == 0:
                    torch.cuda.synchronize()

            # Final synchronization to handle the last partial chunk
            # This ensures all operations for the group are complete before exiting.
            torch.cuda.synchronize()

        return loss
