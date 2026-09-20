"""Dense Adam and Lion with optional *update-level* extrapolation (EF).

For the ordinary optimizer's full update ``u`` at the CURRENT weights, apply
``p <- p - u - c * (u - previous_update)``. Moments are advanced exactly once;
this is not gradient extrapolation, a second optimizer step, or a base-optimizer
wrapper. The first completed ``step()`` is ordinary; correction starts on the
second call, matching a paper indexing convention with an initial no-op and
``t <= 1`` startup. Each parameter group saves its logical call count as
``ef_step``. Only actual calls count: an externally skipped call does nothing.

Parameters, gradients, moments, and saved updates use float32 or float64 on the
parameter's device (including CPU). No low-precision/master-weight conversion,
sparse/complex parameters, AMP GradScaler integration, fused/foreach kernels,
capturable execution, or differentiable optimizer steps are supported. Finite
precision arithmetic can differ from native optimizers by rounding order; in
particular, an update is computed explicitly, never recovered by subtracting
rounded old/new parameters. Tiny updates may leave a parameter unchanged while
still being present in EF history.

With ``ef_coefficient > 0``, the only additional persistent tensor per active
parameter is ``previous_update``: the UNCORRECTED update, including its learning
rate and weight decay. A scheduler change therefore compares the new update to
the update at the previous call's learning rate, without rescaling history.
With coefficient zero, no history tensor is allocated. Missing gradients (and
frozen parameters, even with an attached gradient) leave weights, moments, and
per-parameter Adam steps unchanged, but zero any existing history. The logical
group counter still advances, so a first active parameter on a later call uses
zero history and receives correction immediately.

State is keyed by stable Parameter identities, not their storage pointers, so
same-shape/dtype/device storage rebinding by PipeDream-2BW is supported. Standard
optimizer ``state_dict`` / ``load_state_dict`` preserve moments, update history,
and startup counters. Parameters must be restored separately; this is NOT full
pipeline checkpoint/resume support.
"""
from __future__ import annotations

import math
from numbers import Integral, Real

import torch
from torch.optim import Optimizer

__all__ = ["AdamEF", "Lion"]


def _nonnegative(name, value):
    if (isinstance(value, bool) or not isinstance(value, Real)
            or not math.isfinite(value) or value < 0):
        raise ValueError(f"{name} must be a finite nonnegative real scalar, got {value!r}")


def _counter(name, value):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer, got {value!r}")


def _dense_real(name, tensor):
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.layout != torch.strided or tensor.dtype not in (torch.float32, torch.float64):
        raise ValueError(f"{name} must be dense real float32/float64; "
                         f"got {tensor.layout}, {tensor.dtype}")


class _UpdateEFOptimizer(Optimizer):
    """Shared validation/history, without an inner optimizer or weight backup."""

    _moment_names = ()
    _has_adam_step = False

    def __init__(self, params, defaults):
        defaults = dict(defaults, ef_step=0)
        self._validate_group(defaults)
        super().__init__(params, defaults)

    def _validate_group(self, group):
        for name in ("lr", "weight_decay", "ef_coefficient"):
            _nonnegative(name, group[name])
        betas = group["betas"]
        if not isinstance(betas, (tuple, list)) or len(betas) != 2:
            raise ValueError("betas must contain exactly two real scalars in [0, 1)")
        for i, beta in enumerate(betas):
            _nonnegative(f"betas[{i}]", beta)
            if beta >= 1:
                raise ValueError(f"betas[{i}] must be less than 1, got {beta!r}")
        _counter("ef_step", group["ef_step"])
        for name in ("fused", "foreach", "capturable", "differentiable", "maximize", "amsgrad"):
            if group.get(name, False):
                raise ValueError(f"{type(self).__name__} does not support {name}=True")

    def add_param_group(self, param_group):
        """Add a validated group with its own fresh logical startup counter."""
        if not isinstance(param_group, dict):
            raise TypeError("param_group must be a dict")
        group = dict(param_group)
        params = group["params"]
        if isinstance(params, torch.Tensor):
            params = [params]
        elif isinstance(params, set):
            raise TypeError("optimizer parameters must have a deterministic order, not a set")
        else:
            params = list(params)
        if len({id(parameter) for parameter in params}) != len(params):
            raise ValueError("each parameter must occur only once in an optimizer group")
        group["params"] = params
        candidate = dict(self.defaults, **group)
        self._validate_group(candidate)
        for parameter in params:
            _dense_real("parameter", parameter)
        super().add_param_group(group)

    def _validate_state(self, group, parameter, state, *, loading=False):
        if not isinstance(state, dict):
            raise ValueError("parameter optimizer state must be a dict")
        if not state:
            return
        expected = set(self._moment_names)
        if self._has_adam_step:
            expected.add("step")
            _counter("Adam step", state.get("step"))
            if not 0 < state["step"] <= group["ef_step"]:
                raise ValueError("Adam step must be positive and no greater than ef_step")
        if group["ef_coefficient"] > 0:
            expected.add("previous_update")
        if set(state) != expected:
            raise ValueError(f"invalid {type(self).__name__} state keys: "
                             f"expected {sorted(expected)}, got {sorted(state)}")
        if group["ef_step"] == 0:
            raise ValueError("initialized optimizer state requires a positive ef_step")
        for name in expected - {"step"}:
            value = state[name]
            _dense_real(f"state[{name!r}]", value)
            if value.shape != parameter.shape:
                raise ValueError(f"state[{name!r}] shape {tuple(value.shape)} does not "
                                 f"match parameter shape {tuple(parameter.shape)}")
            if not loading and (value.dtype != parameter.dtype or value.device != parameter.device):
                raise ValueError(f"state[{name!r}] must match the parameter dtype/device; "
                                 "only same-dtype/device storage rebinding is supported")

    def load_state_dict(self, state_dict):
        """Restore this optimizer's state, validating counters and tensor shapes.

        As with PyTorch optimizers, parameters are matched by group order, not
        names. Floating state is cast/moved to each destination parameter by
        PyTorch; unsupported saved dtypes/layouts are rejected before casting.
        This accepts this optimizer's format, not native Adam/AdamW checkpoints.
        """
        if not isinstance(state_dict, dict) or not {"state", "param_groups"} <= state_dict.keys():
            raise ValueError("optimizer state_dict requires 'state' and 'param_groups'")
        saved_groups = state_dict["param_groups"]
        saved_state = state_dict["state"]
        if not isinstance(saved_groups, (list, tuple)) or not isinstance(saved_state, dict):
            raise ValueError("invalid optimizer state_dict structure")
        if len(saved_groups) != len(self.param_groups):
            raise ValueError("loaded state_dict has a different number of parameter groups")
        parameter_ids = set()
        for group, current in zip(saved_groups, self.param_groups):
            if not isinstance(group, dict) or not (set(self.defaults) | {"params"}) <= group.keys():
                raise ValueError("loaded parameter group is missing hyperparameters or ef_step")
            self._validate_group(group)
            if len(group["params"]) != len(current["params"]):
                raise ValueError("loaded parameter group has a different number of parameters")
            for key, parameter in zip(group["params"], current["params"]):
                if not isinstance(key, Integral) or key in parameter_ids:
                    raise ValueError("loaded parameter IDs must be unique integers")
                parameter_ids.add(key)
                _dense_real("parameter", parameter)
                self._validate_state(group, parameter, saved_state.get(key, {}), loading=True)
        if not set(saved_state) <= parameter_ids:
            raise ValueError("loaded optimizer state contains unknown parameter IDs")
        return super().load_state_dict(state_dict)

    def _prepare_step(self):
        # Validate all groups/gradients before mutating any weights or counters.
        if hasattr(self, "grad_scale") or hasattr(self, "found_inf"):
            raise RuntimeError("AMP GradScaler integration is not supported")
        for group in self.param_groups:
            self._validate_group(group)
            for parameter in group["params"]:
                _dense_real("parameter", parameter)
                state = self.state.get(parameter, {})
                self._validate_state(group, parameter, state)
                if parameter.requires_grad and parameter.grad is not None:
                    _dense_real("gradient", parameter.grad)

    def _apply_update(self, parameter, state, group, update):
        coefficient = group["ef_coefficient"]
        if coefficient == 0:
            parameter.sub_(update)
            return
        previous = state.get("previous_update")
        if group["ef_step"] > 1:
            # Keep 'update' unmodified: history is u, never the corrected update
            # or the displacement recovered from rounded parameter arithmetic.
            correction = update.clone() if previous is None else update - previous
            correction.mul_(coefficient).add_(update)
            parameter.sub_(correction)
        else:
            parameter.sub_(update)
        if previous is None:
            state["previous_update"] = update.detach().clone(memory_format=torch.preserve_format)
        else:
            previous.copy_(update)

    def _skip_parameter(self, parameter):
        state = self.state.get(parameter)
        if state and "previous_update" in state:
            state["previous_update"].zero_()


class AdamEF(_UpdateEFOptimizer):
    r"""Adam with optional update-level EF, not extrapolated gradients.

    Args:
        params: Parameters or parameter-group dictionaries.
        lr: Finite nonnegative learning rate (default: 1e-3).
        betas: Moment coefficients in [0, 1) (default: (0.9, 0.999)).
        eps: Finite nonnegative denominator epsilon (default: 1e-8).
        weight_decay: Finite nonnegative decay coefficient (default: 0).
        ef_coefficient: Finite nonnegative EF coefficient (default: 1).
            Zero is ordinary Adam, with no update-history allocation.
        decoupled_weight_decay: If False, add decay to the gradient before
            updating either moment, as in Adam. If True, add ``lr * wd * p``
            at CURRENT weights to the uncorrected update, as in AdamW.

    For each active parameter, advance the Adam step and moments once, then
    compute ``u = lr * m_hat / (sqrt(v_hat) + eps)`` (plus decoupled decay if
    enabled). Apply ``p -= u + c * (u - u_prev)`` from the SECOND logical group
    call; the first call applies only ``p -= u``. With ``c=0``, math matches
    native Adam (or AdamW for decoupled decay), to arithmetic tolerance rather
    than bitwise equivalence. All arithmetic/history uses the parameter dtype.

    Missing/frozen parameters neither update nor advance their Adam step;
    existing history is zeroed, but group calls still count. The saved update
    includes that call's LR, so LR scheduling does not rescale old history.
    State follows Parameter identity through same-shape/dtype/device storage
    rebinding, and can live on CPU. See module docs for serialization/startup
    semantics and the unsupported low-precision/AMP/capturable/fused features.
    """

    _moment_names = ("exp_avg", "exp_avg_sq")
    _has_adam_step = True

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, ef_coefficient=1.0, decoupled_weight_decay=False):
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps,
                                     weight_decay=weight_decay,
                                     ef_coefficient=ef_coefficient,
                                     decoupled_weight_decay=decoupled_weight_decay))

    def _validate_group(self, group):
        super()._validate_group(group)
        _nonnegative("eps", group["eps"])
        if not isinstance(group["decoupled_weight_decay"], bool):
            raise ValueError("decoupled_weight_decay must be a bool")

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one logical call; evaluate an optional closure with gradients."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._prepare_step()
        for group in self.param_groups:
            group["ef_step"] += 1
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if not parameter.requires_grad or parameter.grad is None:
                    self._skip_parameter(parameter)
                    continue
                grad = parameter.grad
                if group["weight_decay"] and not group["decoupled_weight_decay"]:
                    grad = grad.add(parameter, alpha=group["weight_decay"])
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                state["step"] += 1
                moment, variance = state["exp_avg"], state["exp_avg_sq"]
                moment.lerp_(grad, 1 - beta1)
                variance.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                bias1 = 1 - beta1 ** state["step"]
                bias2 = 1 - beta2 ** state["step"]
                denominator = variance.sqrt().div_(math.sqrt(bias2)).add_(group["eps"])
                update = moment.div(denominator).mul_(group["lr"] / bias1)
                if group["decoupled_weight_decay"] and group["weight_decay"]:
                    update.add_(parameter, alpha=group["lr"] * group["weight_decay"])
                self._apply_update(parameter, state, group, update)
        return loss


class Lion(_UpdateEFOptimizer):
    r"""Lion with optional update-level EF (ordinary Lion by default).

    Args:
        params: Parameters or parameter-group dictionaries.
        lr: Finite nonnegative learning rate (default: 1e-4).
        betas: Sign/momentum coefficients in [0, 1) (default: (0.9, 0.99)).
        weight_decay: Finite nonnegative decoupled decay (default: 0).
        ef_coefficient: Finite nonnegative EF coefficient (default: 0).
            Zero allocates no update history; one enables standard EF.

    Compute ``u = lr * (sign(beta1*m + (1-beta1)*g) + wd*p)`` at
    CURRENT weights using the old momentum, then advance momentum exactly once
    as ``m = beta2*m + (1-beta2)*g``. Apply ``p -= u`` on the first logical
    call and ``p -= u + c*(u-u_prev)`` thereafter. The persistent history is
    the uncorrected ``u`` including that call's LR and decay, not rounded weight
    displacement; scheduler changes do not rescale it. Arithmetic/history uses
    the parameter's float32/float64 dtype, and may differ from other Lion
    implementations by rounding order.

    Missing/frozen parameters keep their weights/momentum and zero any existing
    history. Group calls still advance: a first active parameter after startup
    uses zero history and immediately receives correction. State follows stable
    Parameter identities through storage rebinding and supports CPU placement.
    ``state_dict`` includes optimizer startup/history, not pipeline checkpoints.
    No low-precision, sparse/complex, AMP GradScaler, fused/foreach, capturable,
    or differentiable step support is provided; only actual step calls count.
    """

    _moment_names = ("exp_avg",)

    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0,
                 ef_coefficient=0):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay,
                                     ef_coefficient=ef_coefficient))

    @torch.no_grad()
    def step(self, closure=None):
        """Perform one logical call; evaluate an optional closure with gradients."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._prepare_step()
        for group in self.param_groups:
            group["ef_step"] += 1
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if not parameter.requires_grad or parameter.grad is None:
                    self._skip_parameter(parameter)
                    continue
                state = self.state[parameter]
                if not state:
                    state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                moment = state["exp_avg"]
                update = moment.mul(beta1).add_(parameter.grad, alpha=1 - beta1).sign_()
                if group["weight_decay"]:
                    update.add_(parameter, alpha=group["weight_decay"])
                update.mul_(group["lr"])
                moment.mul_(beta2).add_(parameter.grad, alpha=1 - beta2)
                self._apply_update(parameter, state, group, update)
        return loss
