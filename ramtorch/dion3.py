"""Single-process Dion3 (Microsoft's NorDion2) with optional delay-update EF.

Algorithm adapted from https://github.com/microsoft/dion at revision
1b4cd8b60f33add3e8753bd03c5281db176ab336: dion2.py, nordion2.py,
normuon.py and polar_express.py. No distributed optimizer or
Triton dependency is imported on the ordinary PyTorch path.

Upstream material is used under the MIT License:
Copyright (c) Microsoft Corporation.
Copyright (c) 2024 Keller Jordan
Copyright (c) 2025 Moonshot AI

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
SOFTWARE.
"""
from __future__ import annotations

import math

import torch

from .delayed_optim import _UpdateEFOptimizer, _adam_update, _nonnegative

__all__ = ["Dion3"]

# Upstream Polar Express, five iterations; safety factor baked into all but last.
_POLAR_EXPRESS_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def _polar_express(direction, group):
    if group["use_triton"]:
        from ._dion3_triton import polar_express_triton
        return polar_express_triton(direction, epsilon=group["epsilon"])
    matrix = direction.to(dtype=group["ns_dtype"])
    matrix = matrix / (matrix.norm() * 1.02 + group["epsilon"])
    tall = matrix.shape[0] > matrix.shape[1]
    for a, b, c in _POLAR_EXPRESS_COEFFS:
        gram = matrix.T @ matrix if tall else matrix @ matrix.T
        polynomial = b * gram + c * (gram @ gram)
        matrix = a * matrix + (matrix @ polynomial if tall else polynomial @ matrix)
    return matrix


class Dion3(_UpdateEFOptimizer):
    r"""Dion3 row-selection + NorMuon normalization, not the original Dion.

    Parameters default to ``algorithm='dion3'``: nonempty 2D matrices only.
    Explicit ``algorithm='adamw'`` or ``'lion'`` groups cover biases, embeddings,
    heads, and other nonmatrix roles, with independently configurable LRs.
    ``'nordion2'`` is accepted as the upstream spelling of ``'dion3'``.

    Matrix update: accumulate ``M += g``; select ``ceil(fraction * rows)`` rows
    by largest L1 norm (torch.topk, sorted=False); orthogonalize these rows with
    five Polar Express iterations and retain ``mu * M`` at selected rows.
    Unselected rows retain their complete accumulated residual. Update only the
    selected row variances using ``muon_beta2``; normalize by sqrt(variance)
    plus 1e-8, then restore the submatrix's Frobenius norm. Scatter the direction
    back into the full update. There is no Adam bias correction on row variance.
    This built-in compression error feedback is ALWAYS active, even at c=0.

    ``adjust_lr='spectral_norm'`` scales the matrix direction by sqrt(rows/cols),
    ``'rms_norm'`` by 0.2*sqrt(max(rows, cols)), or None leaves it unscaled.
    Decay uses base LR and CURRENT weights on ALL rows. Optional delay EF applies
    to the complete update ``u``: ``p -= u + c*(u-u_prev)`` after the first group
    call. History is the uncorrected full update, not the compression residual.
    Fallback groups use RamTorch's missing-gradient/per-parameter Adam counters.

    Params, gradients, residuals, row variances and EF history are FP32/FP64.
    ``ns_dtype`` defaults to BF16 as upstream; FP32/FP64 are explicit reference
    alternatives. Normalization runs in parameter dtype with autocast disabled.
    ``use_triton=True`` uses vendored upstream BF16 CUDA Polar Express kernels;
    it requires CUDA BF16 and Triton, not a process group, NCCL, or torch.compile.
    Triton and eager arithmetic need not be bit-identical. Upstream batched /
    compiled arithmetic and weight-decay ordering can also round differently.

    State is lazy: one full ``momentum`` plus ``variance_neuron`` of (rows, 1),
    plus one full ``previous_update`` iff c>0. Missing/frozen parameters skip
    residual/variance/weight updates and clear existing delay EF history.
    Standard state_dict supports continuation and same-device/dtype bank
    rebinding, not full 2BW checkpoints. No DTensor, 3D flattening, head-splitting,
    GradScaler, fused/foreach, capturable or differentiable optimizer support.
    """

    def __init__(self, params, lr=0.01, fraction=0.25, mu=0.95, muon_beta2=0.95,
                 betas=(0.9, 0.95), weight_decay=0.01, epsilon=1e-8,
                 adjust_lr="spectral_norm", ef_coefficient=0,
                 ns_dtype=torch.bfloat16, use_triton=False):
        super().__init__(params, dict(
            lr=lr, fraction=fraction, mu=mu, muon_beta2=muon_beta2,
            betas=betas, weight_decay=weight_decay, epsilon=epsilon,
            adjust_lr=adjust_lr, ef_coefficient=ef_coefficient,
            ns_dtype=ns_dtype, use_triton=use_triton, algorithm="dion3"))

    def _validate_group(self, group):
        super()._validate_group(group)
        _nonnegative("fraction", group["fraction"])
        if not 0 < group["fraction"] <= 1:
            raise ValueError("fraction must be in (0, 1]")
        for name in ("mu", "muon_beta2"):
            _nonnegative(name, group[name])
            if group[name] >= 1:
                raise ValueError(f"{name} must be less than 1")
        _nonnegative("epsilon", group["epsilon"])
        if group["epsilon"] == 0:
            raise ValueError("epsilon must be positive")
        if group["algorithm"] not in ("dion3", "nordion2", "adamw", "lion"):
            raise ValueError("algorithm must be 'dion3', 'nordion2', 'adamw' or 'lion'")
        if group["adjust_lr"] not in ("spectral_norm", "rms_norm", None):
            raise ValueError("adjust_lr must be 'spectral_norm', 'rms_norm' or None")
        if group["ns_dtype"] not in (torch.float32, torch.float64, torch.bfloat16):
            raise ValueError("ns_dtype must be torch.float32, torch.float64 or torch.bfloat16")
        if not isinstance(group["use_triton"], bool):
            raise ValueError("use_triton must be a bool")
        if group["use_triton"] and group["ns_dtype"] != torch.bfloat16:
            raise ValueError("Dion3 Triton kernels require ns_dtype=torch.bfloat16")
        # Avoid silently accepting unsupported upstream configuration in groups.
        for key in ("flatten", "num_heads", "split_sizes", "distributed_mesh",
                    "newton_schulz_func", "use_gram_newton_schulz", "selection_scope",
                    "use_polar_express", "triton_post_ortho"):
            if key in group:
                raise ValueError(f"Dion3 does not support group option {key!r}")

    def _validate_parameter(self, group, parameter):
        super()._validate_parameter(group, parameter)
        if type(parameter) not in (torch.Tensor, torch.nn.Parameter):
            raise ValueError("Dion3 requires ordinary local tensors, not DTensor/subclasses")
        if group["algorithm"] in ("dion3", "nordion2"):
            if parameter.ndim != 2 or parameter.numel() == 0:
                raise ValueError("Dion3 requires nonempty 2D matrices; use explicit fallback groups")
            if group["use_triton"]:
                if parameter.device.type != "cuda":
                    raise ValueError("Dion3 Triton kernels require CUDA parameters")
                # Validate native NVIDIA BF16 (not emulation/ROCm) before
                # advancing counters or touching the compression residual.
                from ._dion3_triton import _require_backend
                _require_backend(parameter)

    def _state_spec(self, group):
        if group["algorithm"] in ("dion3", "nordion2"):
            return ("momentum", "variance_neuron"), False
        if group["algorithm"] == "adamw":
            return ("exp_avg", "exp_avg_sq"), True
        return ("exp_avg",), False

    def _state_shape(self, group, parameter, name):
        if name == "variance_neuron":
            return (parameter.shape[0], 1)
        return parameter.shape

    @staticmethod
    def _matrix_update(parameter, state, group):
        if not state:
            state["momentum"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
            state["variance_neuron"] = torch.zeros_like(parameter[:, :1])
        momentum, variance = state["momentum"], state["variance_neuron"]
        momentum.add_(parameter.grad)
        count = max(1, math.ceil(group["fraction"] * parameter.shape[0]))
        indices = torch.topk(momentum.norm(p=1, dim=1), count, sorted=False).indices
        selected = momentum.index_select(0, indices)
        momentum.index_copy_(0, indices, selected * group["mu"])
        direction = _polar_express(selected, group).to(parameter.dtype)
        norm = direction.norm()
        selected_variance = torch.lerp(variance.index_select(0, indices),
                                      direction.square().mean(dim=1, keepdim=True),
                                      1 - group["muon_beta2"])
        variance.index_copy_(0, indices, selected_variance)
        direction = direction / (selected_variance.sqrt() + 1e-8)
        direction = direction * (norm / direction.norm().clamp(min=1e-8))
        rows, cols = parameter.shape
        scale = (math.sqrt(rows / cols) if group["adjust_lr"] == "spectral_norm" else
                 0.2 * math.sqrt(max(rows, cols)) if group["adjust_lr"] == "rms_norm" else 1.0)
        update = torch.zeros_like(parameter, memory_format=torch.preserve_format)
        update.index_copy_(0, indices, direction * (group["lr"] * scale))
        return update

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._prepare_step()
        for group in self.param_groups:
            group["ef_step"] += 1
            for parameter in group["params"]:
                if not parameter.requires_grad or parameter.grad is None:
                    self._skip_parameter(parameter)
                    continue
                state = self.state[parameter]
                # Autocast belongs to each caller/worker, not the main thread.
                with torch.autocast(device_type=parameter.device.type, enabled=False):
                    if group["algorithm"] in ("dion3", "nordion2"):
                        update = self._matrix_update(parameter, state, group)
                    elif group["algorithm"] == "adamw":
                        update = _adam_update(parameter, parameter.grad, state, lr=group["lr"],
                                              betas=group["betas"], eps=group["epsilon"])
                    else:
                        if not state:
                            state["exp_avg"] = torch.zeros_like(parameter)
                        beta1, beta2 = group["betas"]
                        moment = state["exp_avg"]
                        update = moment.mul(beta1).add_(parameter.grad, alpha=1-beta1).sign_()
                        update.mul_(group["lr"])
                        moment.mul_(beta2).add_(parameter.grad, alpha=1-beta2)
                    if group["weight_decay"]:
                        update.add_(parameter, alpha=group["lr"] * group["weight_decay"])
                    self._apply_update(parameter, state, group, update)
        return loss
