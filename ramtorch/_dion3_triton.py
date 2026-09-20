"""Optional CUDA/BF16 Polar Express backend for Dion3.

Vendored from microsoft/dion, revision
1b4cd8b60f33add3e8753bd03c5281db176ab336:
https://github.com/microsoft/dion/blob/1b4cd8b60f33add3e8753bd03c5281db176ab336/dion/newton_schulz_triton.py
https://github.com/microsoft/dion/blob/1b4cd8b60f33add3e8753bd03c5281db176ab336/dion/polar_express.py
https://github.com/microsoft/dion/blob/1b4cd8b60f33add3e8753bd03c5281db176ab336/LICENSE

Only the two symmetric kernels and their support are vendored. Their arithmetic
and autotuning configurations are unchanged. RamTorch adds optional-import and
CUDA checks, an optimizer-only 2D entry point, and thread-safe cold initialization.
Unlike upstream, there is no torch.compile wrapper; epsilon defaults to 1e-8.
The caller should disable autocast, as for the other Dion3 orthogonalizers.

Import this private module only when the Triton backend is requested. Importing
it does not require Triton to be installed and does not initialize CUDA.
TRITON_AVAILABLE describes the package import, not the selected GPU's capability.

MIT License

Copyright (c) Microsoft Corporation.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE
"""

import math
import threading
import types

import torch
from torch import Tensor

__all__ = ["TRITON_AVAILABLE", "polar_express_triton"]

_TRITON_IMPORT_ERROR = None
try:
    import triton
    import triton.language as tl
except ImportError as exc:
    _TRITON_IMPORT_ERROR = exc
    TRITON_AVAILABLE = False
    # Parse the vendored kernel definitions without requiring Triton on CPU.
    triton = types.ModuleType("triton")
    triton.jit = lambda fn: fn
    tl = types.ModuleType("triton.language")
    tl.constexpr = int
else:
    TRITON_AVAILABLE = True


# Exactly upstream's five-step Polar Express coefficients (safety_factor=2e-2,
# cushion=2), from https://arxiv.org/pdf/2505.16932.
_POLAR_EXPRESS_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def _get_autotune_configs():
    return [
        triton.Config(
            {
                "BLOCK_SIZE_M": bm,
                "BLOCK_SIZE_N": bn,
                "BLOCK_SIZE_K": bk,
                "GROUP_SIZE_M": 8,
                "LOWER_UPPER": 1,
            },
            num_stages=stages,
            num_warps=warps,
        )
        for bm in [64, 128]
        for bn in [64, 128, 256]
        for bk in [64, 128]
        for stages, warps in [(3, 4), (3, 8), (4, 4)]
        if bm // bn <= 2 and bn // bm <= 2
    ]


@triton.jit
def _batch_offset(batch_idx, batch_stride):
    return batch_idx.to(tl.int64) * batch_stride


@triton.jit
def _pid_to_block(
    pid,
    M,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """Map a program ID to (batch, row, column) of the output matrix."""
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_N)

    batch_idx = pid // (num_pid_m * num_pid_n)
    pid = pid % (num_pid_m * num_pid_n)

    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    pid_m, pid_n = tl.swizzle2d(pid_m, pid_n, num_pid_m, num_pid_n, GROUP_SIZE_M)

    m_idx = pid_m * BLOCK_SIZE_M
    n_idx = pid_n * BLOCK_SIZE_N
    return batch_idx, m_idx, n_idx


# Autotuners are constructed on the cold path, not shared between workers.
@triton.jit
def ns_line_1_kernel(
    A_ptr,
    C_ptr,
    M,
    K,
    a_stride_b,
    a_stride_r,
    a_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
    INPUT_PRECISION: tl.constexpr = "tf32",
):
    """Compute C = A @ A.T, where A has shape (M, K)."""
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    A_ptr += _batch_offset(batch_idx, a_stride_b)
    C_ptr += _batch_offset(batch_idx, c_stride_b)

    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator, input_precision=INPUT_PRECISION)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


@triton.jit
def ns_line_2_kernel(
    A_ptr,
    C_ptr,
    M,
    a_stride_b,
    a_stride_r,
    a_stride_c,
    c_stride_b,
    c_stride_r,
    c_stride_c,
    alpha,
    beta,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    LOWER_UPPER: tl.constexpr,
    INPUT_PRECISION: tl.constexpr = "tf32",
):
    """Compute C = alpha * A @ A.T + beta * A for symmetric square A."""
    pid = tl.program_id(axis=0)
    batch_idx, m_idx, n_idx = _pid_to_block(
        pid, M, BLOCK_SIZE_M, BLOCK_SIZE_N, GROUP_SIZE_M
    )

    skip_block_below_diag = (LOWER_UPPER == 0) and (n_idx + BLOCK_SIZE_N <= m_idx)
    skip_block_above_diag = (LOWER_UPPER != 0) and (m_idx + BLOCK_SIZE_M <= n_idx)
    if skip_block_below_diag or skip_block_above_diag:
        return

    A_ptr += _batch_offset(batch_idx, a_stride_b)
    C_ptr += _batch_offset(batch_idx, c_stride_b)

    offs_m = (m_idx + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (n_idx + tl.arange(0, BLOCK_SIZE_N)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A_ptr + (offs_m[:, None] * a_stride_r + offs_k[None, :] * a_stride_c)
    at_ptrs = A_ptr + (offs_k[:, None] * a_stride_c + offs_n[None, :] * a_stride_r)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in tl.range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < M - k * BLOCK_SIZE_K, other=0.0)
        at = tl.load(at_ptrs, mask=offs_k[:, None] < M - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, at, accumulator, input_precision=INPUT_PRECISION)
        a_ptrs += BLOCK_SIZE_K * a_stride_c
        at_ptrs += BLOCK_SIZE_K * a_stride_c

    offs_am = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_an = n_idx + tl.arange(0, BLOCK_SIZE_N)
    a_add_ptrs = A_ptr + (offs_am[:, None] * a_stride_r + offs_an[None, :] * a_stride_c)
    a_add_mask = (offs_am[:, None] < M) & (offs_an[None, :] < M)
    a_add = tl.load(a_add_ptrs, mask=a_add_mask, other=0.0).to(tl.float32)

    accumulator *= alpha
    accumulator += a_add * beta

    out_dtype = C_ptr.dtype.element_ty
    output = accumulator.to(out_dtype)

    offs_cm = m_idx + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = n_idx + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = C_ptr + (offs_cm[:, None] * c_stride_r + offs_cn[None, :] * c_stride_c)
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, output, mask=c_mask)

    c_ptrs_t = C_ptr + (offs_cn[:, None] * c_stride_r + offs_cm[None, :] * c_stride_c)
    c_mask_t = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
    tl.store(c_ptrs_t, output.T, mask=c_mask_t)


# Only cold signatures acquire this lock. Each cache entry owns a separate JIT
# object, so a new shape/device cannot mutate the JIT state of a running entry.
# Warm launches bypass Autotuner.run entirely: even cache-hit autotuner calls
# write shared nargs/best_config fields and can race with another worker.
_INITIALIZE_LOCK = threading.Lock()
_KERNEL_CACHE = {}
_SUPPORTED_DEVICES = set()


def _require_backend(G: Tensor):
    if not TRITON_AVAILABLE:
        raise ImportError(
            "Dion3's Triton backend requires the optional 'triton' package. "
            "Install a Triton version compatible with your CUDA PyTorch build, "
            "or select the PyTorch backend."
        ) from _TRITON_IMPORT_ERROR
    if G.device.type != "cuda" or torch.version.hip is not None:
        raise RuntimeError(
            "Dion3's Triton backend requires an NVIDIA CUDA tensor; "
            "use the PyTorch backend on CPU or ROCm."
        )
    if G.device.index not in _SUPPORTED_DEVICES:
        with _INITIALIZE_LOCK:
            if G.device.index not in _SUPPORTED_DEVICES:
                if not torch.cuda.is_available():
                    raise RuntimeError("Dion3's Triton backend requires available CUDA.")
                # BF16 emulation is insufficient for the kernels' tensor-core dots.
                if torch.cuda.get_device_capability(G.device)[0] < 8:
                    raise RuntimeError(
                        "Dion3's Triton backend requires native CUDA BF16 support "
                        "(NVIDIA Ampere / compute capability 8.0 or newer)."
                    )
                _SUPPORTED_DEVICES.add(G.device.index)


def _launch(kernel, grid, A: Tensor, out: Tensor, **kwargs):
    """Autotune once per device/layout; never share a mutable autotuner."""
    signature = (
        kernel.__name__,
        A.device.index,
        A.dtype,
        tuple(A.shape),
        tuple(A.stride()),
        tuple(out.shape),
        tuple(out.stride()),
        # Triton specializes pointer alignment as well as shapes and strides.
        A.data_ptr() % 16,
        out.data_ptr() % 16,
    )
    entry = _KERNEL_CACHE.get(signature)
    if entry is None:
        with _INITIALIZE_LOCK:
            entry = _KERNEL_CACHE.get(signature)
            if entry is None:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Warm up Dion3's Triton backend on each device/shape/stride "
                        "before CUDA graph capture; cold signatures require autotuning."
                    )
                # Use distinct JIT/autotuner objects for each signature. The cold
                # lock also protects shared helper JIT hashes and driver setup.
                jit_kernel = triton.jit(kernel.fn)
                key = ["M", "a_stride_r", "a_stride_c", "c_stride_r", "c_stride_c"]
                if "K" in kwargs:
                    key.insert(1, "K")
                autotuner = triton.autotune(
                    configs=_get_autotune_configs(), key=key
                )(jit_kernel)
                autotuner[grid](A_ptr=A, C_ptr=out, **kwargs)
                config = autotuner.best_config.all_kwargs()
                # Publish only after compilation AND the first actual launch,
                # which initializes lazy CUDA module/function/launcher handles.
                _KERNEL_CACHE[signature] = (jit_kernel, config)
                return
    jit_kernel, config = entry
    jit_kernel[grid](A_ptr=A, C_ptr=out, **kwargs, **config)


def _check_kernel_input(A: Tensor, out: Tensor):
    _require_backend(A)
    if A.ndim not in (2, 3) or A.numel() == 0:
        raise ValueError("Dion3 symmetric kernels require nonempty 2D or 3D tensors.")
    if A.dtype != torch.bfloat16:
        raise TypeError("Dion3 symmetric kernels require BF16 input.")
    if out is not None:
        expected = (*A.shape[:-1], A.size(-2))
        if tuple(out.shape) != expected or out.device != A.device or out.dtype != A.dtype:
            raise ValueError("Output must have the expected shape, device, and BF16 dtype.")


def ns_line_1(A: Tensor, *, out: Tensor = None):
    """Compute C = A @ A.T using the upstream symmetric Triton kernel."""
    _check_kernel_input(A, out)
    M, K = A.shape[-2:]
    if out is None:
        out = torch.empty((*A.shape[:-1], M), device=A.device, dtype=A.dtype)
    batch_size = A.size(0) if A.ndim == 3 else 1
    grid = lambda meta: (
        batch_size
        * triton.cdiv(M, meta["BLOCK_SIZE_M"])
        * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    with torch.cuda.device(A.device):
        _launch(
            ns_line_1_kernel, grid, A, out,
            M=M,
            K=K,
            a_stride_b=A.stride(0) if A.ndim == 3 else 0,
            a_stride_r=A.stride(-2),
            a_stride_c=A.stride(-1),
            c_stride_b=out.stride(0) if out.ndim == 3 else 0,
            c_stride_r=out.stride(-2),
            c_stride_c=out.stride(-1),
            INPUT_PRECISION="tf32",
        )
    return out


def ns_line_2(A: Tensor, alpha: float, beta: float, *, out: Tensor = None):
    """Compute C = alpha * A @ A.T + beta * A for symmetric square A."""
    _check_kernel_input(A, out)
    M, K = A.shape[-2:]
    if M != K:
        raise ValueError(f"Input must be symmetric square matrix, got {A.shape}.")
    if out is None:
        out = torch.empty((*A.shape[:-1], M), device=A.device, dtype=A.dtype)
    batch_size = A.size(0) if A.ndim == 3 else 1
    grid = lambda meta: (
        batch_size
        * triton.cdiv(M, meta["BLOCK_SIZE_M"])
        * triton.cdiv(M, meta["BLOCK_SIZE_N"]),
    )
    with torch.cuda.device(A.device):
        _launch(
            ns_line_2_kernel, grid, A, out,
            M=M,
            a_stride_b=A.stride(0) if A.ndim == 3 else 0,
            a_stride_r=A.stride(-2),
            a_stride_c=A.stride(-1),
            c_stride_b=out.stride(0) if out.ndim == 3 else 0,
            c_stride_r=out.stride(-2),
            c_stride_c=out.stride(-1),
            alpha=float(alpha),
            beta=float(beta),
            INPUT_PRECISION="tf32",
        )
    return out


@torch.no_grad()
def polar_express_triton(G: Tensor, epsilon: float = 1e-8) -> Tensor:
    """Orthogonalize one real 2D matrix with five upstream BF16 polynomials.

    Returns a BF16 matrix on G's device, with the same shape; G is not modified.
    This optimizer helper is not differentiable. Native NVIDIA BF16 support and
    Triton are required; errors never silently select another algorithm.

    Cold device/layout signatures serialize JIT/autotuning, whose upstream
    benchmark machinery may synchronize the device. No explicit synchronization
    is added here, and warmed signatures take no initialization lock. Prewarm
    all expected signatures before timing, profiling, or CUDA graph capture.
    """
    if not isinstance(G, Tensor):
        raise TypeError("polar_express_triton expects a torch.Tensor.")
    if G.ndim != 2 or G.numel() == 0:
        raise ValueError("polar_express_triton expects a nonempty 2D matrix.")
    if G.layout != torch.strided or not G.is_floating_point():
        raise TypeError("polar_express_triton expects a dense real floating-point matrix.")
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive.")
    _require_backend(G)

    with torch.cuda.device(G.device):
        X = G.to(dtype=torch.bfloat16)
        if G.size(-2) > G.size(-1):
            X = X.mT

        # Preserve upstream's BF16 normalization and 1.02 safety factor.
        X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + epsilon)
        X = X.contiguous()
        A = torch.empty((X.size(-2), X.size(-2)), device=X.device, dtype=X.dtype)
        B = torch.empty_like(A)
        C = torch.empty_like(X)

        for a, b, c in _POLAR_EXPRESS_COEFFS:
            ns_line_1(X, out=A)
            ns_line_2(A, alpha=c, beta=b, out=B)
            torch.addmm(X, B, X, beta=a, out=C)
            X, C = C, X

        if G.size(-2) > G.size(-1):
            X = X.mT
        return X
