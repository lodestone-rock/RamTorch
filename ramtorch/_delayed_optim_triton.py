"""Optional fused pointwise Adam(W)/Lion + update-level EF kernels.

Original RamTorch kernels, not a distributed optimizer. Imported only for an
explicit ``use_triton=True`` group. One launch per nonempty active parameter
updates moments, weights, and (only when enabled) uncorrected update history.
No parameter-sized update workspace, autotuning, or device synchronization.

Cold JIT launches are serialized. Each worker thread owns its JIT objects, so
warmed launches cannot race with another worker's mutable compiler/launcher
state. Runtime scalar bit patterns retain FP64 hyperparameter precision without
specializing on learning rates or per-parameter bias-correction counters.
"""
import math
import struct
import threading
import types

import torch

_IMPORT_ERROR = None
try:
    import triton
    import triton.language as tl
    import triton.language.extra.cuda.libdevice as libdevice
except ImportError as exc:
    _IMPORT_ERROR = exc
    TRITON_AVAILABLE = False
    triton = types.ModuleType("triton")
    triton.jit = lambda *args, **kwargs: (lambda fn: fn)
    tl = types.ModuleType("triton.language")
    tl.constexpr = int
else:
    TRITON_AVAILABLE = True


@triton.jit()
def _scalar(bits, dtype: tl.constexpr):
    # Python float kernel arguments are normally marshalled as FP32. Carry the
    # IEEE bits as an integer instead; convert only after reconstructing FP64.
    return bits.to(tl.int64).to(tl.float64, bitcast=True).to(dtype)


@triton.jit(do_not_specialize=["N", "LR", "B1", "B2", "WD", "EPS", "STEP_SIZE", "BIAS2_SQRT", "EF"])
def _update_kernel(P, G, M, V, H, N, LR, B1, B2, WD, EPS, STEP_SIZE, BIAS2_SQRT, EF,
                   ADAM: tl.constexpr, DECOUPLED: tl.constexpr, DECAY: tl.constexpr,
                   HISTORY: tl.constexpr, CORRECT: tl.constexpr,
                   BLOCK: tl.constexpr):
    offsets = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < N
    p = tl.load(P + offsets, mask=mask, other=0)
    g = tl.load(G + offsets, mask=mask, other=0)
    m = tl.load(M + offsets, mask=mask, other=0)
    dtype: tl.constexpr = P.dtype.element_ty
    lr, b1, b2 = _scalar(LR, dtype), _scalar(B1, dtype), _scalar(B2, dtype)
    wd = _scalar(WD, dtype)
    # Eager computes 1-beta in Python double before casting to parameter dtype;
    # subtracting already-rounded FP32 beta2 would badly amplify near-one error.
    one_minus_b1 = (1. - _scalar(B1, tl.float64)).to(dtype)
    one_minus_b2 = (1. - _scalar(B2, tl.float64)).to(dtype)
    if ADAM:
        if DECAY and not DECOUPLED:
            g = libdevice.fma(p, wd, g)
        # Moments advance exactly once. libdevice operations retain FP64 and
        # avoid approximate sqrt/div for the FP32 Adam denominator.
        delta = g - m
        m = tl.where(one_minus_b1 < 0.5,
                     libdevice.fma(delta, one_minus_b1, m),
                     libdevice.fma(-delta, 1 - one_minus_b1, g))
        v = tl.load(V + offsets, mask=mask, other=0)
        v = v * b2 + (g * g) * one_minus_b2
        denominator = libdevice.div_rn(libdevice.sqrt_rn(v), _scalar(BIAS2_SQRT, dtype)) + _scalar(EPS, dtype)
        update = libdevice.div_rn(m, denominator) * _scalar(STEP_SIZE, dtype)
        if DECAY and DECOUPLED:
            decay_lr = (_scalar(LR, tl.float64) * _scalar(WD, tl.float64)).to(dtype)
            update = libdevice.fma(p, decay_lr, update)
        tl.store(V + offsets, v, mask=mask)
    else:
        direction = libdevice.fma(g, one_minus_b1, m * b1)
        direction = tl.where(direction > 0, 1., tl.where(direction < 0, -1., 0.)).to(dtype)
        if DECAY:
            direction = libdevice.fma(p, wd, direction)
        update = direction * lr
        m = libdevice.fma(g, one_minus_b2, m * b2)
    tl.store(M + offsets, m, mask=mask)
    applied = update
    if HISTORY:
        if CORRECT:
            previous = tl.load(H + offsets, mask=mask, other=0)
            applied = update + _scalar(EF, dtype) * (update - previous)
        tl.store(H + offsets, update, mask=mask)
    tl.store(P + offsets, p - applied, mask=mask)


_INITIALIZE_LOCK = threading.Lock()
_WORKER = threading.local()


def require_backend(parameter):
    if parameter.device.type != "cuda" or torch.version.hip is not None:
        raise ValueError("use_triton=True requires NVIDIA CUDA parameters; use eager for CPU/ROCm")
    if type(parameter) not in (torch.Tensor, torch.nn.Parameter):
        raise ValueError("Triton optimizers require ordinary local tensors")
    if parameter.dtype not in (torch.float32, torch.float64) or not parameter.is_contiguous():
        raise ValueError("Triton optimizers require contiguous FP32/FP64 parameters")
    if not TRITON_AVAILABLE:
        raise ImportError("use_triton=True requires the optional triton package") from _IMPORT_ERROR


def _bits(value):
    return struct.unpack("q", struct.pack("d", float(value)))[0]


def step_parameter(parameter, state, group, *, adam):
    """Called under no_grad after whole-optimizer preflight validation."""
    if not state:
        state["exp_avg"] = torch.zeros_like(parameter)
        if adam:
            state["step"] = 0
            state["exp_avg_sq"] = torch.zeros_like(parameter)
    if adam:
        state["step"] += 1
    history = group["ef_coefficient"] > 0
    if history and "previous_update" not in state:
        # A parameter first active after group startup corrects against zero.
        state["previous_update"] = torch.zeros_like(parameter)
    if parameter.numel() == 0:
        return
    b1, b2 = group["betas"]
    step_size = group["lr"] / (1 - b1 ** state["step"]) if adam else 0.
    bias2_sqrt = math.sqrt(1 - b2 ** state["step"]) if adam else 1.
    scalars = tuple(_bits(v) for v in (group["lr"], b1, b2, group["weight_decay"],
                                     group.get("eps", 0.), step_size, bias2_sqrt,
                                     group["ef_coefficient"]))
    tensors = (parameter, parameter.grad, state["exp_avg"],
               state["exp_avg_sq"] if adam else state["exp_avg"],
               state["previous_update"] if history else parameter)
    flags = dict(ADAM=adam, DECOUPLED=group.get("decoupled_weight_decay", True),
                 DECAY=group["weight_decay"] != 0,
                 HISTORY=history, CORRECT=history and group["ef_step"] > 1, BLOCK=256)
    # Triton also specializes on argument types/alignment, but not scalar values.
    # Distinguish int32/int64 runtime marshaling (zero double bits fit int32).
    key = (parameter.device.index, parameter.dtype, tuple(flags.values()),
           tuple(t.data_ptr() % 16 for t in tensors),
           tuple(-(2**31) <= s < 2**31 for s in (parameter.numel(), *scalars)))
    cache = getattr(_WORKER, "kernels", None)
    if cache is None:
        cache = _WORKER.kernels = {}
    kernel = cache.get(key)
    grid = (triton.cdiv(parameter.numel(), 256),)
    args = (*tensors, parameter.numel(), *scalars)
    with torch.cuda.device(parameter.device):
        if kernel is None:
            with _INITIALIZE_LOCK:
                kernel = triton.jit(_update_kernel.fn, do_not_specialize=[
                    "N", "LR", "B1", "B2", "WD", "EPS", "STEP_SIZE", "BIAS2_SQRT", "EF"])
                kernel[grid](*args, **flags, enable_fp_fusion=False)
                cache[key] = kernel
        else:
            kernel[grid](*args, **flags, enable_fp_fusion=False)
