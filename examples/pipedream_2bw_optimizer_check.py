"""Independent AdamEF/Lion/Muon/Dion3 equations and exact PipeDream-2BW checks.

Run from the repository root, using a Python environment with working PyTorch::

    PYTHONPATH=. python examples/pipedream_2bw_optimizer_check.py
    PYTHONPATH=. python examples/pipedream_2bw_optimizer_check.py --bf16
    PYTHONPATH=. python examples/pipedream_2bw_optimizer_check.py --devices cuda:0,cuda:1 --bf16

--bf16 ADDS autocast checks to the FP32 bank/pipeline suite; optimizer parameters
and moments remain FP32. --unit-only and --pipeline-only isolate the two suites.
--optimizers dion3 selects just Dion3; --dion3-triton uses its vendored backend
only with CUDA optimizer placement (CPU masters still use eager PyTorch).
--triton independently selects AdamEF/AdamW/Lion CUDA kernels and adds their
standalone backend checks; CPU groups explicitly select use_triton=False.
No production optimizer helper is used to compute expected optimizer updates:
Python scalar equations and an independent Float64 Newton-Schulz polynomial
check weights, moments and full update history; plain AdamEF and Muon's explicit
fallback are also compared with native Adam/AdamW. The copied-history reference
uses the same factory ONLY for independent bank/schedule comparisons.
"""
from __future__ import annotations

import argparse
import copy
import io
import math
from collections import OrderedDict

# Import before torch: the reference sets deterministic CUDA environment flags.
from pipedream_2bw_reference import (
    assert_exact, configure_determinism, make_batches, make_model, run_reference,
)
import torch
from torch import nn
from torch.nn import functional as F

from ramtorch import Pipeline
from ramtorch.pipeline import Stage
from ramtorch.pipeline_2bw import _VersionedStage


KINDS = ("adam", "adamw", "lion", "muon", "dion3")
INITIAL = ((0.8, -1.3, 0.0), (-0.6, 0.2, 1.1), (1.7, -0.4, 0.3), (3.0, 2.0, 1.0))
GRADIENTS = (
    ((0.3, -0.7, 0.0), (-0.2, 0.4, 0.8), None, None),
    ((-0.1, 0.5, 0.0), None, None, None),
    ((0.0, 0.0, 0.0), (0.6, -0.3, 0.1), (0.4, -0.2, 0.9), None),
    ((0.8, -0.1, 0.3), (0.0, 0.0, 0.0), None, None),
    (None, None, None, None),
    ((-0.6, 0.2, -0.5), (-0.7, 0.4, 0.1), (0.1, -0.3, 0.2), None),
    ((0.1, 0.0, 0.2), None, (0.0, 0.0, 0.0), None),
)
LEARNING_RATES = (0.07, 0.023, 0.0, 0.11, 0.017, 0.035, 0.009)


def optimizer_factory(kind, coefficient, **overrides):
    """One factory shared by the scheduled and copied-history models."""
    from ramtorch.delayed_optim import AdamEF, Lion

    options = dict(lr=0.017, betas=(0.6, 0.8), weight_decay=0.13,
                   ef_coefficient=coefficient)
    if kind in ("adam", "adamw"):
        options["eps"] = 0.031
        # Omit the flag for Adam: coupled decay must be the default.
        if kind == "adamw":
            options["decoupled_weight_decay"] = True
        cls = AdamEF
    elif kind == "lion":
        cls = Lion
    elif kind == "muon":
        from ramtorch.delayed_optim import Muon

        options.update(momentum=0.7, nesterov=True, ns_steps=5, eps=1e-7,
                       adamw_eps=0.031, adjust_lr_fn="original", ns_dtype=torch.float32)
        options.update(overrides)

        def make_muon(parameters):
            # One optimizer per stage; vectors are explicit AdamW, never Muon.
            parameters = list(parameters)
            groups = [{"params": [p for p in parameters if p.ndim == 2], "use_muon": True},
                      {"params": [p for p in parameters if p.ndim != 2], "use_muon": False}]
            return Muon([group for group in groups if group["params"]], **options)

        return make_muon
    elif kind == "dion3":
        from ramtorch.dion3 import Dion3

        options.update(fraction=0.5, mu=0.7, muon_beta2=0.8, epsilon=1e-7,
                       adjust_lr="spectral_norm", ns_dtype=torch.bfloat16)
        options.update(overrides)

        def make_dion3(parameters):
            # Assign roles from the supplied parameters, including CPU masters.
            parameters = list(parameters)
            groups = [{"params": [p for p in parameters if p.ndim == 2], "algorithm": "dion3"},
                      {"params": [p for p in parameters if p.ndim != 2], "algorithm": "adamw",
                       "epsilon": 0.031}]
            # A CUDA pipeline may place its optimizer on CPU. Triton is only
            # requested for actual CUDA optimizer parameters, never CPU masters.
            local_options = dict(options)
            local_options["use_triton"] = bool(options.get("use_triton", False) and
                                                all(p.device.type == "cuda" for p in parameters))
            return Dion3([group for group in groups if group["params"]], **local_options)

        return make_dion3
    else:
        raise ValueError(f"unknown optimizer {kind!r}")
    options.update(use_triton=False)
    options.update(overrides)

    def make_dense(parameters):
        # The factory, not the strict backend, opts CPU masters into eager.
        # Materialize generators and preserve explicit per-group hyperparameters.
        parameters = list(parameters)
        groups = parameters if parameters and isinstance(parameters[0], dict) else [{"params": parameters}]
        local_groups = []
        for group in groups:
            group = dict(group)
            values = group["params"]
            group["params"] = [values] if isinstance(values, torch.Tensor) else list(values)
            requested = group.get("use_triton", options["use_triton"])
            group["use_triton"] = bool(requested and group["params"] and
                                       all(p.device.type == "cuda" for p in group["params"]))
            local_groups.append(group)
        return cls(local_groups, **options)

    return make_dense


def _parameters(device="cpu", dtype=torch.float64):
    return [nn.Parameter(torch.tensor(values, dtype=dtype, device=device),
                         requires_grad=index != 3)
            for index, values in enumerate(INITIAL)]


def _set_gradients(parameters, gradients):
    for parameter, values in zip(parameters, gradients):
        parameter.grad = (None if values is None else
                          torch.tensor(values, device=parameter.device, dtype=parameter.dtype))


def _close(actual, expected, path, *, rtol=2e-12, atol=2e-13):
    if not isinstance(actual, torch.Tensor):
        actual = torch.tensor(actual, dtype=torch.float64)
    expected = torch.tensor(expected, dtype=torch.float64, device=actual.device)
    torch.testing.assert_close(actual.detach().to(torch.float64), expected,
                               rtol=rtol, atol=atol, msg=lambda msg: f"{path}: {msg}")


class ScalarEquations:
    """Ordinary Python arithmetic, with no torch optimizer/tensor update helpers.

    Adam: m <- b1*m+(1-b1)*g, v <- b2*v+(1-b2)*g*g, then bias correction.
    Lion: direction = sign(b1*m+(1-b1)*g), m <- b2*m+(1-b2)*g.
    Decoupled decay contributes lr*wd*x to u; coupled decay enters Adam's g.
    The first optimizer CALL is ordinary, even if no parameter has a gradient.
    Later calls subtract u+c*(u-previous_u), using zero for missing history.
    """

    def __init__(self, kind, initial, coefficients):
        self.kind = kind
        self.weights = [list(values) for values in initial]
        self.coefficients = list(coefficients)
        self.states = [None for _ in initial]
        self.calls = 0

    def step(self, gradients, learning_rates, decays, betas=(0.6, 0.8), eps=0.031):
        beta1, beta2 = betas
        for index, (gradient, lr, decay) in enumerate(zip(gradients, learning_rates, decays)):
            state, values = self.states[index], self.weights[index]
            coefficient = self.coefficients[index]
            if gradient is None:
                if state is not None and "previous_update" in state:
                    state["previous_update"] = [0.0] * len(values)
                continue
            if state is None:
                state = {"exp_avg": [0.0] * len(values), "step": 0}
                if self.kind != "lion":
                    state["exp_avg_sq"] = [0.0] * len(values)
                if coefficient > 0:
                    state["previous_update"] = [0.0] * len(values)
                self.states[index] = state
            state["step"] += 1
            for coordinate, (value, raw_gradient) in enumerate(zip(values, gradient)):
                moment = state["exp_avg"][coordinate]
                if self.kind == "lion":
                    blend = beta1 * moment + (1.0 - beta1) * raw_gradient
                    direction = 1.0 if blend > 0 else -1.0 if blend < 0 else 0.0
                    update = lr * (direction + decay * value)
                    state["exp_avg"][coordinate] = beta2 * moment + (1.0 - beta2) * raw_gradient
                else:
                    effective = raw_gradient + decay * value if self.kind == "adam" else raw_gradient
                    moment = beta1 * moment + (1.0 - beta1) * effective
                    variance = beta2 * state["exp_avg_sq"][coordinate] + (1.0 - beta2) * effective**2
                    state["exp_avg"][coordinate] = moment
                    state["exp_avg_sq"][coordinate] = variance
                    corrected_moment = moment / (1.0 - beta1**state["step"])
                    corrected_variance = variance / (1.0 - beta2**state["step"])
                    update = lr * corrected_moment / (math.sqrt(corrected_variance) + eps)
                    if self.kind == "adamw":
                        update += lr * decay * value
                previous = state.get("previous_update", [0.0] * len(values))[coordinate]
                correction = coefficient * (update - previous) if self.calls else 0.0
                values[coordinate] = value - update - correction
                if coefficient > 0:
                    state["previous_update"][coordinate] = update
        self.calls += 1

    def check(self, parameters, optimizer, path):
        for index, parameter in enumerate(parameters):
            location = f"{path}.parameter{index}"
            _close(parameter, self.weights[index], location)
            actual, expected = optimizer.state.get(parameter, {}), self.states[index]
            if expected is None:
                assert not actual, f"{location}: missing/frozen parameter allocated state: {actual.keys()}"
                continue
            _close(actual["exp_avg"], expected["exp_avg"], location + ".exp_avg")
            if self.kind != "lion":
                _close(actual["exp_avg_sq"], expected["exp_avg_sq"], location + ".exp_avg_sq")
                assert float(actual["step"]) == expected["step"], location + ": moments advanced twice"
            if self.coefficients[index] == 0:
                assert "previous_update" not in actual, location + ": c=0 allocated history"
            else:
                _close(actual["previous_update"], expected["previous_update"], location + ".previous_update")
            expected_tensors = {"exp_avg"}
            if self.kind != "lion":
                expected_tensors.add("exp_avg_sq")
            if self.coefficients[index] > 0:
                expected_tensors.add("previous_update")
            sized_tensors = {key for key, value in actual.items()
                             if isinstance(value, torch.Tensor) and value.shape == parameter.shape}
            assert sized_tensors == expected_tensors, f"{location}: unexpected parameter-sized state {sized_tensors}"


def check_equations(kind, coefficient, device, *, empty_start=False):
    parameters = _parameters(device)
    optimizer = optimizer_factory(kind, coefficient)([
        {"params": parameters[:2]},
        {"params": parameters[2:], "weight_decay": 0.04},
    ])
    equations = ScalarEquations(kind, INITIAL, [coefficient] * len(parameters))
    calls = ((None,) * len(parameters),) + GRADIENTS if empty_start else GRADIENTS
    for call, gradients in enumerate(calls):
        lr = LEARNING_RATES[call % len(LEARNING_RATES)]
        optimizer.param_groups[0]["lr"] = lr
        optimizer.param_groups[1]["lr"] = lr * 0.7
        _set_gradients(parameters, gradients)
        before_parameters = [parameter.detach().clone() for parameter in parameters]
        before_gradients = [None if p.grad is None else p.grad.clone() for p in parameters]
        before_state = [copy.deepcopy(optimizer.state.get(p, {})) for p in parameters]
        optimizer.step()
        equations.step(gradients, [lr, lr, lr * 0.7, lr * 0.7], [0.13, 0.13, 0.04, 0.04])
        path = f"equations.{kind}.c{coefficient}.{device}.empty{empty_start}.call{call}"
        equations.check(parameters, optimizer, path)
        for index, parameter in enumerate(parameters):
            assert_exact(parameter.grad, before_gradients[index], path=path + ".gradient")
            if gradients[index] is None:
                assert_exact(parameter, before_parameters[index], path=path + ".missing_weight")
                state = optimizer.state.get(parameter, {})
                assert state.keys() == before_state[index].keys(), path + ": missing gradient initialized state"
                for key, value in before_state[index].items():
                    if key == "previous_update":
                        assert torch.count_nonzero(state[key]).item() == 0, path + ": missing update retained history"
                    else:
                        assert_exact(state[key], value, path=path + ".missing_state." + key)


def check_group_overrides(kind, device):
    """Mixed c=0/c>0 groups keep independent settings and share logical startup."""
    parameters = _parameters(device)
    coefficients, decays = [0.0, 1.0, 0.35, 1.0], [0.0, 0.13, 0.04, 0.09]
    optimizer = optimizer_factory(kind, 0.0)([
        {"params": [parameter], "ef_coefficient": coefficient, "weight_decay": decay}
        for parameter, coefficient, decay in zip(parameters, coefficients, decays)
    ])
    equations = ScalarEquations(kind, INITIAL, coefficients)
    calls = ((None,) * len(parameters),) + GRADIENTS
    for call, gradients in enumerate(calls):
        learning_rates = [LEARNING_RATES[call % len(LEARNING_RATES)] * (1.0 + index * 0.2)
                          for index in range(len(parameters))]
        for group, lr in zip(optimizer.param_groups, learning_rates):
            group["lr"] = lr
        _set_gradients(parameters, gradients)
        optimizer.step()
        equations.step(gradients, learning_rates, decays)
        equations.check(parameters, optimizer, f"mixed_groups.{kind}.call{call}")
        if call in (0, 3):
            # A brand new optimizer must remember even a gradient-free first call.
            state = copy.deepcopy(optimizer.state_dict())
            restored = optimizer_factory(kind, 2.0)([{"params": [p]} for p in parameters])
            restored.load_state_dict(state)
            assert_exact(restored.state_dict(), state, path="mixed_groups.resume")
            optimizer = restored


def check_native_adam(device):
    for kind, native_class in (("adam", torch.optim.Adam), ("adamw", torch.optim.AdamW)):
        for dtype in (torch.float32, torch.float64):
            for decay in (0.0, 0.13):
                actual, expected = _parameters(device, dtype), _parameters(device, dtype)
                options = dict(lr=0.07, betas=(0.6, 0.8), eps=0.031, weight_decay=decay)
                optimizer = optimizer_factory(kind, 0.0, **options)(actual)
                native = native_class(expected, **options, foreach=False, fused=False)
                tolerance = 3e-6 if dtype == torch.float32 else 3e-13
                for call, gradients in enumerate(GRADIENTS):
                    optimizer.param_groups[0]["lr"] = native.param_groups[0]["lr"] = LEARNING_RATES[call]
                    _set_gradients(actual, gradients)
                    _set_gradients(expected, gradients)
                    optimizer.step()
                    native.step()
                    for left, right in zip(actual, expected):
                        torch.testing.assert_close(left, right, rtol=tolerance, atol=tolerance)
                        ours, theirs = optimizer.state.get(left, {}), native.state.get(right, {})
                        assert "previous_update" not in ours, "plain AdamEF allocated EF history"
                        for key in ("exp_avg", "exp_avg_sq", "step"):
                            assert (key in ours) == (key in theirs)
                            if key in ours:
                                torch.testing.assert_close(torch.as_tensor(ours[key]).cpu().double(),
                                                           torch.as_tensor(theirs[key]).cpu().double(),
                                                           rtol=tolerance, atol=tolerance)
    print(f"PASS native Adam/AdamW parity, FP32/FP64, coupled/decoupled/zero decay: {device}")


def check_state_budget(kind, device):
    optimizers, parameters = [], []
    for coefficient in (0.0, 1.0):
        params = _parameters(device)
        optimizer = optimizer_factory(kind, coefficient)(params)
        assert not optimizer.state, "constructor allocated per-parameter optimizer state"
        _set_gradients(params, GRADIENTS[0])
        optimizer.step()
        parameters.append(params)
        optimizers.append(optimizer)
    for index in range(len(INITIAL)):
        plain = optimizers[0].state.get(parameters[0][index], {})
        corrected = optimizers[1].state.get(parameters[1][index], {})
        assert "previous_update" not in plain
        if not plain:
            assert not corrected, "unused/frozen parameter allocated EF state"
            continue
        assert corrected.keys() == plain.keys() | {"previous_update"}, "EF must add exactly one state entry"
        history = corrected["previous_update"]
        parameter = parameters[1][index]
        assert history.shape == parameter.shape and history.dtype == parameter.dtype
        assert history.device == parameter.device and not history.requires_grad
        assert history.untyped_storage().data_ptr() != parameter.untyped_storage().data_ptr()
        for key, value in corrected.items():
            if key != "previous_update" and isinstance(value, torch.Tensor):
                assert history.untyped_storage().data_ptr() != value.untyped_storage().data_ptr()
        plain_bytes = sum(value.numel() * value.element_size() for value in plain.values()
                          if isinstance(value, torch.Tensor))
        ef_bytes = sum(value.numel() * value.element_size() for value in corrected.values()
                       if isinstance(value, torch.Tensor))
        assert ef_bytes - plain_bytes == parameter.numel() * parameter.element_size()
        # Startup is ordinary for both coefficients, including moments.
        assert_exact(parameter, parameters[0][index], path="state_budget.startup.weight")
        assert_exact({key: corrected[key] for key in plain}, plain, path="state_budget.startup.state")


def check_closure(kind, coefficient, device):
    actual, expected = _parameters(device), _parameters(device)
    optimizer, direct = (optimizer_factory(kind, coefficient)(params) for params in (actual, expected))
    for step in range(3):
        calls, returned = [], []

        def closure():
            assert torch.is_grad_enabled(), "closure ran with gradients disabled"
            calls.append(True)
            optimizer.zero_grad(set_to_none=True)
            loss = sum((parameter.square() * (step + 1)).sum() for parameter in actual[:3])
            loss.backward()
            returned.append(loss)
            return loss

        result = optimizer.step(closure)
        assert len(calls) == 1 and result is returned[0], "closure must run once and return its loss"
        direct.zero_grad(set_to_none=True)
        loss = sum((parameter.square() * (step + 1)).sum() for parameter in expected[:3])
        loss.backward()
        direct.step()
        assert_exact(actual, expected, path="closure.weights")
        assert_exact(optimizer.state_dict(), direct.state_dict(), path="closure.state")


def check_resume(kind, coefficient, device):
    """Serialize before startup, after an empty startup, and after real/missing updates."""
    gradients = ((None,) * len(INITIAL),) + GRADIENTS
    original = _parameters(device)
    optimizer = optimizer_factory(kind, coefficient)(original)
    snapshots = [(copy.deepcopy(original), copy.deepcopy(optimizer.state_dict()))]
    for call, values in enumerate(gradients):
        optimizer.param_groups[0]["lr"] = LEARNING_RATES[call % len(LEARNING_RATES)]
        _set_gradients(original, values)
        optimizer.step()
        snapshots.append((copy.deepcopy(original), copy.deepcopy(optimizer.state_dict())))
    for split in (0, 1, 2, 4, 6):
        buffer = io.BytesIO()
        weights, state = snapshots[split]
        torch.save({"weights": [parameter.detach().clone() for parameter in weights],
                    "optimizer": state}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, map_location=device, weights_only=True)
        resumed = [nn.Parameter(value, requires_grad=index != 3)
                   for index, value in enumerate(saved["weights"])]
        # Deliberately different construction defaults: loading must restore c.
        restored = optimizer_factory(kind, 0.37, lr=0.9)(resumed)
        restored.load_state_dict(saved["optimizer"])
        assert_exact(restored.state_dict(), state, path=f"resume.{kind}.split{split}.loaded")
        for call in range(split, len(gradients)):
            restored.param_groups[0]["lr"] = LEARNING_RATES[call % len(LEARNING_RATES)]
            _set_gradients(resumed, gradients[call])
            restored.step()
            expected_weights, expected_state = snapshots[call + 1]
            assert_exact(resumed, expected_weights, path=f"resume.{kind}.split{split}.call{call}.weights")
            assert_exact(restored.state_dict(), expected_state,
                         path=f"resume.{kind}.split{split}.call{call}.state")


def _reject(action, label, exceptions=(ValueError, TypeError, RuntimeError)):
    try:
        action()
    except exceptions:
        return
    raise AssertionError(f"accepted invalid {label}")


def check_rejections(kind, device):
    invalid = [
        ("lr", -0.1), ("lr", math.nan), ("lr", math.inf),
        ("weight_decay", -0.1), ("weight_decay", math.nan), ("weight_decay", math.inf),
        ("ef_coefficient", -0.1), ("ef_coefficient", math.nan), ("ef_coefficient", math.inf),
        ("ef_coefficient", "1.0"),
        ("betas", (-0.1, 0.9)), ("betas", (0.9, -0.1)),
        ("betas", (1.0, 0.9)), ("betas", (0.9, 1.0)),
        ("betas", (math.nan, 0.9)), ("betas", (0.9, math.nan)),
        ("betas", (math.inf, 0.9)), ("betas", (0.9,)), ("betas", (0.1, 0.2, 0.3)),
    ]
    if kind != "lion":
        invalid += [("eps", -0.1), ("eps", math.nan), ("eps", math.inf)]
    for key, value in invalid:
        def defaults_case():
            optimizer_factory(kind, 1.0, **{key: value})(_parameters(device))
        _reject(defaults_case, f"{kind} default {key}={value!r}")

        def group_case():
            optimizer_factory(kind, 1.0)([{"params": _parameters(device), key: value}])
        _reject(group_case, f"{kind} group {key}={value!r}")
    options = dict(lr=0.0, betas=(0.0, 0.0), weight_decay=0.0)
    if kind != "lion":
        options["eps"] = 0.0
    optimizer_factory(kind, 0.0, **options)(_parameters(device))  # Valid inclusive boundaries.
    for dtype in (torch.bfloat16, torch.float16, torch.complex64):
        def unsupported_dtype():
            parameter = nn.Parameter(torch.ones(3, dtype=dtype, device=device))
            optimizer = optimizer_factory(kind, 1.0)([parameter])
            parameter.grad = torch.ones_like(parameter)
            optimizer.step()
        _reject(unsupported_dtype, f"{kind} parameter dtype {dtype}")

    def sparse_gradient():
        parameter = nn.Parameter(torch.ones(3, dtype=torch.float64, device=device))
        optimizer = optimizer_factory(kind, 1.0)([parameter])
        parameter.grad = torch.sparse_coo_tensor([[0, 2]], [0.1, -0.2], (3,),
                                                 device=device, dtype=torch.float64,
                                                 check_invariants=True)
        optimizer.step()
    _reject(sparse_gradient, kind + " sparse gradient", (ValueError, TypeError, RuntimeError, NotImplementedError))


def _muon_polynomial(matrix, steps, eps):
    """Independent double-precision scalar matrix algebra, not a production helper.

    Keeping dot products in Python also avoids sharing torch's matmul kernels,
    autocast, or a production transpose/normalization path with the unit oracle.
    """
    values = matrix.detach().cpu().double().tolist()
    transpose = len(values) > len(values[0])
    if transpose:
        values = [list(column) for column in zip(*values)]
    norm = math.sqrt(sum(value * value for row in values for value in row))
    values = [[value / (norm + eps) for value in row] for row in values]

    def multiply(left, right):
        columns = list(zip(*right))
        return [[sum(a * b for a, b in zip(row, column)) for column in columns]
                for row in left]

    for _ in range(steps):
        gram = multiply(values, list(zip(*values)))
        gram_squared = multiply(gram, gram)
        polynomial = [[-4.775 * a + 2.0315 * b for a, b in zip(row_a, row_b)]
                      for row_a, row_b in zip(gram, gram_squared)]
        product = multiply(polynomial, values)
        values = [[3.4445 * x + y for x, y in zip(row_x, row_y)]
                  for row_x, row_y in zip(values, product)]
    if transpose:
        values = [list(column) for column in zip(*values)]
    return torch.tensor(values, dtype=torch.float64)


def _muon_parameters(device="cpu", dtype=torch.float64):
    # Tall, wide, square, zero direction, late, unused, frozen; then explicit
    # AdamW vector/scalar/matrix/frozen/unused fallbacks in the SAME optimizer.
    shapes = ((5, 3), (3, 5), (3, 3), (2, 3), (2, 2), (2, 2), (2, 2),
              (3,), (), (2, 3), (3,), (2,))
    result = []
    for index, shape in enumerate(shapes):
        values = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)
        values = ((values + index * 0.7).cos() * 0.6).to(device=device, dtype=dtype)
        result.append(nn.Parameter(values, requires_grad=index not in (6, 10)))
    return result


def _muon_gradients(parameters, call):
    gradients = []
    for index, parameter in enumerate(parameters):
        if (call == 4 or index in (5, 11) or (index == 4 and call < 2)
                or (call == 1 and index in (1, 7)) or (call == 3 and index == 2)):
            gradients.append(None)
            continue
        values = torch.arange(parameter.numel(), dtype=torch.float64).reshape(parameter.shape)
        values = (values * 0.43 + index * 0.37 + call * 0.61).sin() * 0.8
        if index == 3 or (call in (2, 6) and index == 0):
            values.zero_()
        gradients.append(values.to(device=parameter.device, dtype=parameter.dtype))
    return gradients


def _muon_optimizer(parameters, coefficient, **overrides):
    from ramtorch.delayed_optim import Muon

    options = dict(lr=0.017, momentum=0.7, nesterov=True, weight_decay=0.13,
                   ns_steps=5, eps=1e-7, ef_coefficient=coefficient, betas=(0.6, 0.8),
                   adamw_eps=0.031, adjust_lr_fn="original", ns_dtype=torch.float64)
    options.update(overrides)
    return Muon([
        {"params": parameters[:7]},  # use_muon=True must be the default.
        {"params": parameters[7:9], "use_muon": False, "weight_decay": 0.04,
         "betas": (0.4, 0.75), "adamw_eps": 0.013},
        {"params": parameters[9:], "use_muon": False, "weight_decay": 0.09,
         "betas": (0.3, 0.85), "adamw_eps": 0.007,
         "ef_coefficient": coefficient * 0.35},
    ], **options)


def _muon_assign(parameters, gradients):
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = None if gradient is None else gradient.clone()


def _muon_lrs(optimizer, call):
    rates = [LEARNING_RATES[call % len(LEARNING_RATES)] * factor for factor in (1.0, 0.7, 1.3)]
    for group, lr in zip(optimizer.param_groups, rates):
        group["lr"] = lr
    return rates


class MuonEquations:
    """Independent Float64 updates, current-weight decay, and per-group EF clocks."""

    def __init__(self, parameters, optimizer):
        self.weights = [parameter.detach().cpu().double().clone() for parameter in parameters]
        self.active = [parameter.requires_grad for parameter in parameters]
        indices = {id(parameter): index for index, parameter in enumerate(parameters)}
        self.groups = [dict(copy.deepcopy({key: value for key, value in group.items() if key != "params"}),
                            indices=[indices[id(p)] for p in group["params"]])
                       for group in optimizer.param_groups]
        self.states = [{} for _ in parameters]

    def step(self, gradients, rates):
        for group, lr in zip(self.groups, rates):
            group["ef_step"] += 1
            coefficient = group["ef_coefficient"]
            for index in group["indices"]:
                state, weight = self.states[index], self.weights[index]
                if gradients[index] is None or not self.active[index]:
                    if "previous_update" in state:
                        state["previous_update"].zero_()
                    continue
                gradient = gradients[index].detach().cpu().double()
                if group["use_muon"]:
                    momentum = group["momentum"]
                    old = state.get("momentum_buffer", torch.zeros_like(weight))
                    buffer = momentum * old + gradient
                    state["momentum_buffer"] = buffer
                    direction = gradient + momentum * buffer if group["nesterov"] else buffer
                    rows, columns = weight.shape
                    scale = (math.sqrt(max(1.0, rows / columns)) if group["adjust_lr_fn"] == "original"
                             else 0.2 * math.sqrt(max(rows, columns)))
                    update = lr * scale * _muon_polynomial(direction, group["ns_steps"], group["eps"])
                else:
                    beta1, beta2 = group["betas"]
                    state["step"] = state.get("step", 0) + 1
                    moment = beta1 * state.get("exp_avg", torch.zeros_like(weight)) + (1 - beta1) * gradient
                    variance = beta2 * state.get("exp_avg_sq", torch.zeros_like(weight)) + (1 - beta2) * gradient.square()
                    state["exp_avg"], state["exp_avg_sq"] = moment, variance
                    corrected_moment = moment / (1 - beta1 ** state["step"])
                    corrected_variance = variance / (1 - beta2 ** state["step"])
                    update = lr * corrected_moment / (corrected_variance.sqrt() + group["adamw_eps"])
                # Decay is neither orthogonalized nor multiplied by Muon's scale.
                update = update + lr * group["weight_decay"] * weight
                corrected = update
                if group["ef_step"] > 1 and coefficient:
                    corrected = update + coefficient * (update - state.get("previous_update", torch.zeros_like(weight)))
                self.weights[index] = weight - corrected
                if coefficient:
                    state["previous_update"] = update.clone()

    def check(self, parameters, optimizer, path):
        for actual, expected in zip(optimizer.param_groups, self.groups):
            assert actual["ef_step"] == expected["ef_step"], path + ": wrong logical group clock"
        for index, parameter in enumerate(parameters):
            location = f"{path}.parameter{index}"
            torch.testing.assert_close(parameter.detach().cpu(), self.weights[index],
                                       rtol=2e-12, atol=2e-13, msg=location)
            actual, expected = optimizer.state.get(parameter, {}), self.states[index]
            assert actual.keys() == expected.keys(), location + ": unexpected persistent state"
            for key, value in expected.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(actual[key].cpu(), value, rtol=2e-12, atol=2e-13,
                                               msg=location + "." + key)
                else:
                    assert actual[key] == value, location + ": fallback moments advanced twice"


def check_muon_equations(device):
    for coefficient in (0.0, 1.0):
        for steps in (1, 5):
            for nesterov in (False, True):
                for scaling in ("original", "match_rms_adamw"):
                    for empty_start in (False, True):
                        parameters = _muon_parameters(device)
                        optimizer = _muon_optimizer(parameters, coefficient, ns_steps=steps,
                                                    nesterov=nesterov, adjust_lr_fn=scaling)
                        equations = MuonEquations(parameters, optimizer)
                        for call in range(7 + int(empty_start)):
                            rates = _muon_lrs(optimizer, call)
                            gradients = ([None] * len(parameters) if empty_start and call == 0 else
                                         _muon_gradients(parameters, call - int(empty_start)))
                            _muon_assign(parameters, gradients)
                            before = [p.detach().clone() for p in parameters]
                            before_state = [copy.deepcopy(optimizer.state.get(p, {})) for p in parameters]
                            optimizer.step()
                            equations.step(gradients, rates)
                            path = f"muon.c{coefficient}.ns{steps}.nesterov{nesterov}.{scaling}.empty{empty_start}.call{call}"
                            equations.check(parameters, optimizer, path)
                            for index, parameter in enumerate(parameters):
                                assert_exact(parameter.grad, gradients[index], path=path + ".gradient")
                                if gradients[index] is None or not parameter.requires_grad:
                                    assert_exact(parameter, before[index], path=path + ".missing_or_frozen")
                                    state = optimizer.state.get(parameter, {})
                                    assert state.keys() == before_state[index].keys()
                                    for key, value in before_state[index].items():
                                        if key == "previous_update":
                                            assert not torch.count_nonzero(state[key]).item()
                                        else:
                                            assert_exact(state[key], value, path=path + ".skipped." + key)
    print(f"PASS Muon Float64 independent polynomial: tall/wide/square/zero, NS 1/5, both momentum/scales, "
          f"current decay, mixed AdamW, EF, missing/frozen/late/empty startup: {device}")


def check_muon_native_adamw(device):
    from ramtorch.delayed_optim import Muon

    for dtype in (torch.float32, torch.float64):
        for decay in (0.0, 0.13):
            actual, expected = _muon_parameters(device, dtype)[7:], _muon_parameters(device, dtype)[7:]
            options = dict(lr=0.07, betas=(0.6, 0.8), weight_decay=decay)
            optimizer = Muon([{"params": actual, "use_muon": False}], adamw_eps=0.031, **options)
            native = torch.optim.AdamW(expected, eps=0.031, foreach=False, fused=False, **options)
            tolerance = 3e-6 if dtype == torch.float32 else 3e-13
            for call in range(7):
                optimizer.param_groups[0]["lr"] = native.param_groups[0]["lr"] = LEARNING_RATES[call]
                gradients = _muon_gradients(_muon_parameters(device, dtype), call)[7:]
                # Native torch optimizers do not themselves skip frozen attached gradients.
                gradients = [g if p.requires_grad else None for p, g in zip(actual, gradients)]
                _muon_assign(actual, gradients)
                _muon_assign(expected, gradients)
                optimizer.step()
                native.step()
                for left, right in zip(actual, expected):
                    torch.testing.assert_close(left, right, rtol=tolerance, atol=tolerance)
                    ours, theirs = optimizer.state.get(left, {}), native.state.get(right, {})
                    assert "previous_update" not in ours
                    assert ours.keys() == theirs.keys()
                    for key in ours:
                        torch.testing.assert_close(torch.as_tensor(ours[key]).cpu().double(),
                                                   torch.as_tensor(theirs[key]).cpu().double(),
                                                   rtol=tolerance, atol=tolerance)
    print(f"PASS Muon explicit AdamW fallback: native FP32/FP64 parity, vectors/scalars/matrices: {device}")


def check_muon_budget_resume(device):
    plain, corrected = _muon_parameters(device), _muon_parameters(device)
    ordinary, ef = _muon_optimizer(plain, 0.0), _muon_optimizer(corrected, 1.0)
    assert not ordinary.state and not ef.state, "Muon constructor allocated parameter state"
    for parameters, optimizer in ((plain, ordinary), (corrected, ef)):
        _muon_assign(parameters, _muon_gradients(parameters, 0))
        optimizer.step()
    for p, q in zip(plain, corrected):
        base, extra = ordinary.state.get(p, {}), ef.state.get(q, {})
        assert_exact(p, q, path="muon.budget.startup")
        if not base:
            assert not extra
            continue
        assert extra.keys() == base.keys() | {"previous_update"}
        assert_exact({key: extra[key] for key in base}, base, path="muon.budget.moments")
        history = extra["previous_update"]
        assert history.shape == q.shape and history.dtype == q.dtype and history.device == q.device
        assert not history.requires_grad
        for tensor in [q, q.grad] + [value for value in base.values() if isinstance(value, torch.Tensor)] + [
                value for key, value in extra.items() if key != "previous_update" and isinstance(value, torch.Tensor)]:
            assert history.untyped_storage().data_ptr() != tensor.untyped_storage().data_ptr()
        nbytes = lambda state: sum(value.numel() * value.element_size() for value in state.values()
                                  if isinstance(value, torch.Tensor))
        assert nbytes(extra) - nbytes(base) == q.numel() * q.element_size(), "EF must add one tensor only"

    for coefficient in (0.0, 1.0):
        original = _muon_parameters(device)
        optimizer = _muon_optimizer(original, coefficient)
        snapshots = [(copy.deepcopy(original), copy.deepcopy(optimizer.state_dict()))]
        calls = [[None] * len(original)] + [_muon_gradients(original, call) for call in range(7)]
        for call, gradients in enumerate(calls):
            _muon_lrs(optimizer, call)
            _muon_assign(original, gradients)
            optimizer.step()
            snapshots.append((copy.deepcopy(original), copy.deepcopy(optimizer.state_dict())))
        for split in (0, 1, 2, 4, 6):
            buffer = io.BytesIO()
            weights, state = snapshots[split]
            torch.save({"weights": [p.detach().clone() for p in weights], "optimizer": state}, buffer)
            buffer.seek(0)
            saved = torch.load(buffer, map_location=device, weights_only=True)
            resumed = [nn.Parameter(value, requires_grad=p.requires_grad)
                       for value, p in zip(saved["weights"], weights)]
            restored = _muon_optimizer(resumed, 0.37, lr=0.9, momentum=0.1, ns_steps=1,
                                       nesterov=False, ns_dtype=torch.bfloat16, adjust_lr_fn="match_rms_adamw")
            restored.load_state_dict(saved["optimizer"])
            assert_exact(restored.state_dict(), state, path=f"muon.resume{split}.loaded")
            for call in range(split, len(calls)):
                _muon_lrs(restored, call)
                _muon_assign(resumed, calls[call])
                restored.step()
                expected_weights, expected_state = snapshots[call + 1]
                assert_exact(resumed, expected_weights, path=f"muon.resume{split}.call{call}.weights")
                assert_exact(restored.state_dict(), expected_state, path=f"muon.resume{split}.call{call}.state")
    print(f"PASS Muon one-history-tensor bytes, changing LR and byte-exact serialized continuation: {device}")


def check_muon_autocast(device):
    from ramtorch.delayed_optim import Muon

    if torch.device(device).type == "cuda" and not torch.cuda.is_bf16_supported():
        print(f"SKIP Muon BF16 ambient autocast: unsupported device {device}")
        return
    for dtype in (torch.float32, torch.float64):
        histories = {}
        for ns_dtype in (torch.float32, torch.float64, torch.bfloat16):
            actual = _muon_parameters(device, dtype)[:4]
            expected = [nn.Parameter(p.detach().clone()) for p in actual]
            options = dict(lr=0.07, ef_coefficient=1.0, weight_decay=0.13, ns_dtype=ns_dtype)
            ambient, ordinary = Muon(actual, **options), Muon(expected, **options)
            for call in range(3):
                gradients = _muon_gradients(actual, call)
                _muon_assign(actual, gradients)
                _muon_assign(expected, gradients)
                with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                    ambient.step()
                ordinary.step()
                assert_exact(actual, expected, path=f"muon.autocast.{dtype}.{ns_dtype}.weights")
                assert_exact(ambient.state_dict(), ordinary.state_dict(), path="muon.autocast.state")
                for parameter in actual:
                    for value in ambient.state.get(parameter, {}).values():
                        if isinstance(value, torch.Tensor):
                            assert value.dtype == dtype, "NS precision leaked into persistent state"
            histories[ns_dtype] = [ambient.state[p]["previous_update"].clone() for p in actual]
        # Matching ambient/off runs alone would also pass an implementation
        # that ignores ns_dtype entirely. These nontrivial fixtures must differ.
        for first, second in ((torch.float32, torch.float64), (torch.float32, torch.bfloat16)):
            assert any(not torch.equal(a, b) for a, b in zip(histories[first], histories[second])), \
                f"Muon ignored explicit NS precision {first} versus {second}"
    print(f"PASS Muon explicit FP32/FP64/BF16 NS is byte-exact with/without ambient BF16 autocast: {device}")


def check_muon_rejections(device):
    from ramtorch.delayed_optim import Muon

    invalid = [
        ("lr", -0.1), ("lr", math.nan), ("lr", math.inf),
        ("weight_decay", -0.1), ("weight_decay", math.nan), ("weight_decay", math.inf),
        ("ef_coefficient", -0.1), ("ef_coefficient", math.nan), ("ef_coefficient", math.inf),
        ("ef_coefficient", "1"),
        ("momentum", -0.1), ("momentum", 1.0), ("momentum", math.nan), ("momentum", math.inf),
        ("nesterov", 1), ("nesterov", "true"),
        ("ns_steps", 0), ("ns_steps", -1), ("ns_steps", 1.5), ("ns_steps", True),
        ("eps", -1e-7), ("eps", 0.0), ("eps", math.nan), ("eps", math.inf),
        ("adamw_eps", -1e-8), ("adamw_eps", math.nan), ("adamw_eps", math.inf),
        ("betas", (-0.1, 0.9)), ("betas", (0.9, 1.0)), ("betas", (math.nan, 0.9)),
        ("betas", (0.9,)), ("betas", (0.1, 0.2, 0.3)),
        ("adjust_lr_fn", "unknown"), ("adjust_lr_fn", None),
        ("ns_dtype", torch.float16), ("ns_dtype", torch.int64), ("ns_dtype", "float32"),
    ]
    for key, value in invalid:
        _reject(lambda: Muon([_muon_parameters(device)[0]], **{key: value}), f"Muon default {key}={value!r}")
        _reject(lambda: Muon([{"params": [_muon_parameters(device)[0]], key: value}]),
                f"Muon group {key}={value!r}")
    for key, value in (("use_muon", 1), ("use_muon", "false"), ("ef_step", -1),
                       ("ef_step", 0.5), ("fused", True), ("foreach", True)):
        _reject(lambda: Muon([{"params": [_muon_parameters(device)[0]], key: value}]),
                f"Muon group {key}={value!r}")
    for shape in ((), (3,), (2, 2, 2), (0, 3), (3, 0)):
        parameter = nn.Parameter(torch.ones(shape, device=device, dtype=torch.float64))
        _reject(lambda: Muon([parameter]), f"Muon implicit matrix ndim={len(shape)}")
        _reject(lambda: Muon([{"params": [parameter], "use_muon": True}]),
                f"Muon explicit matrix ndim={len(shape)}")
        fallback = Muon([{"params": [parameter], "use_muon": False}])
        parameter.grad = torch.ones_like(parameter)
        fallback.step()  # Nonmatrices are legal only with explicit fallback.
    for dtype in (torch.float16, torch.bfloat16, torch.complex64):
        _reject(lambda: Muon([nn.Parameter(torch.ones(2, 3, device=device, dtype=dtype))]),
                f"Muon parameter dtype={dtype}")

    def sparse_gradient():
        parameter = _muon_parameters(device)[0]
        optimizer = Muon([parameter])
        parameter.grad = torch.sparse_coo_tensor([[0, 2], [1, 0]], [0.1, -0.2], parameter.shape,
                                                 device=device, dtype=parameter.dtype,
                                                 check_invariants=True)
        optimizer.step()
    _reject(sparse_gradient, "Muon sparse gradient", (ValueError, TypeError, RuntimeError, NotImplementedError))
    # Live group edits are validated too, before any earlier valid group mutates.
    parameters = _muon_parameters(device)
    optimizer = _muon_optimizer(parameters, 1.0)
    _muon_assign(parameters, _muon_gradients(parameters, 0))
    before = [p.detach().clone() for p in parameters]
    optimizer.param_groups[-1]["lr"] = -0.1
    _reject(optimizer.step, "Muon mutated invalid group")
    assert_exact(parameters, before, path="muon.invalid_group.atomic_weights")
    assert not optimizer.state and all(group["ef_step"] == 0 for group in optimizer.param_groups)
    print(f"PASS Muon default/group validation, explicit fallback and dense matrix restrictions: {device}")


def check_muon_closure(device):
    for coefficient in (0.0, 1.0):
        actual, expected = _muon_parameters(device), _muon_parameters(device)
        optimizer, direct = (_muon_optimizer(params, coefficient) for params in (actual, expected))
        for call in range(3):
            returned = []

            def closure():
                assert torch.is_grad_enabled(), "Muon closure disabled gradients"
                optimizer.zero_grad(set_to_none=True)
                loss = sum(p.square().sum() * (call + 1) for p in actual if p.requires_grad)
                loss.backward()
                returned.append(loss)
                return loss

            result = optimizer.step(closure)
            assert len(returned) == 1 and result is returned[0]
            direct.zero_grad(set_to_none=True)
            loss = sum(p.square().sum() * (call + 1) for p in expected if p.requires_grad)
            loss.backward()
            direct.step()
            assert_exact(actual, expected, path="muon.closure.weights")
            assert_exact(optimizer.state_dict(), direct.state_dict(), path="muon.closure.state")
    print(f"PASS Muon closure executes once with gradients and returns its loss: {device}")


def check_muon_units(device):
    check_muon_equations(device)
    check_muon_native_adamw(device)
    check_muon_budget_resume(device)
    check_muon_autocast(device)
    check_muon_rejections(device)
    check_muon_closure(device)


def check_units(device, kinds=KINDS, *, dion3_triton=False, triton=False):
    if triton and any(kind in ("adam", "adamw", "lion") for kind in kinds):
        from triton_optimizer_check import check_triton_units

        check_triton_units(device, kinds=tuple(kind for kind in kinds if kind in ("adam", "adamw", "lion")))
    if any(kind in ("adam", "adamw") for kind in kinds):
        check_native_adam(device)
    for kind in kinds:
        if kind == "dion3":
            from dion3_optimizer_check import check_dion3_units

            check_dion3_units(device, triton=dion3_triton and torch.device(device).type == "cuda")
            continue
        if kind == "muon":
            check_muon_units(device)
            continue  # Scalar/vector fixtures are not legal implicit Muon matrices.
        check_state_budget(kind, device)
        check_rejections(kind, device)
        check_group_overrides(kind, device)
        for coefficient in (0.0, 0.35, 1.0, 2.0):
            check_equations(kind, coefficient, device)
            check_equations(kind, coefficient, device, empty_start=True)
        for coefficient in (0.0, 1.0):
            check_closure(kind, coefficient, device)
            check_resume(kind, coefficient, device)
        print(f"PASS {kind}: scalar equations, one-step moments, full-update EF, missing/zero/frozen, "
              f"late/global startup, changing lr, allocation, closure, rejection, exact resume: {device}")


def _model(dim, layers):
    model = make_model(dim=dim, layers=layers, seed=41)
    for block in model:
        block.register_parameter("unused", nn.Parameter(torch.tensor([1.1, -0.7])))
        block.register_parameter("frozen", nn.Parameter(torch.tensor([0.9, -0.3]), requires_grad=False))
    return model


def _expected_snapshot(update, names):
    return {"grads": {name: update.gradients[name] for name in names},
            "weights": {name: update.weights[name] for name in names},
            "optimizer": {name: update.optimizer_state[name] for name in names}}


def _check_owners(bank, coefficient, *, use_triton=None):
    if use_triton is not None:
        for group in bank.optimizer.param_groups:
            expected = bool(use_triton and all(p.device.type == "cuda" for p in group["params"]))
            assert group["use_triton"] is expected, "factory selected the wrong optimizer backend"
    for name, parameter in bank.optimizer_named.items():
        assert parameter.dtype == torch.float32, "autocast changed master dtype"
        assert parameter.device == (torch.device("cpu") if bank.cpu_optimizer else bank.stage.device)
        state = bank.optimizer.state.get(parameter, {})
        if name.endswith(("unused", "frozen")):
            assert not state, "unused/frozen pipeline parameter allocated state"
            continue
        assert ("previous_update" in state) == (coefficient > 0)
        for value in state.values():
            if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                assert value.dtype == torch.float32 and value.device == parameter.device


def check_banks(kind, coefficient, device, optimizer_device, bf16, updates, *, dion3_triton=False, triton=False):
    model = _model(dim=8, layers=2)
    groups = make_batches(updates=updates, microbatches=1, dim=8, seed=43)
    overrides = ({"use_triton": dion3_triton} if kind == "dion3" else
                 {"use_triton": triton} if kind in ("adam", "adamw", "lion") else {})
    factory = optimizer_factory(kind, coefficient, **overrides)
    expected = run_reference(model, groups, factory, device=device,
                             optimizer_device=optimizer_device, autocast_bf16=bf16)
    stage = Stage(copy.deepcopy(model), 0, 1, device=device,
                  autocast_dtype=torch.bfloat16 if bf16 else None)
    bank = _VersionedStage(stage, factory, optimizer_device=optimizer_device)
    records = {}
    pointers = [{name: value.untyped_storage().data_ptr() for name, value in slot.items()}
                for slot in bank.banks]

    def observe(stage_index, group, snapshot):
        assert stage_index == 0 and group not in records
        records[group] = snapshot

    try:
        # The next old-weight graph is alive while the first update changes banks.
        bank.forward(0, groups[0][0][0], 1)
        bank.forward(1, groups[1][0][0], 1)
        for group in range(updates):
            if group > 1:
                bank.forward(group, groups[group][0][0], 1)
            bank.backward(group, target=groups[group][0][1].to(device), loss_fn=F.mse_loss)
            bank.update(group, 1, observe)
            for slot in range(2):
                actual = {name: value.untyped_storage().data_ptr() for name, value in bank.banks[slot].items()}
                assert actual == pointers[slot], "bank storage was replaced"
            for name, parameter in bank.named.items():
                assert pointers[0][name] != pointers[1][name]
                assert parameter.untyped_storage().data_ptr() in (pointers[0][name], pointers[1][name])
            _check_owners(bank, coefficient, use_triton=triton if kind in ("adam", "adamw", "lion") else None)
            assert_exact(records[group], _expected_snapshot(expected.updates[group], bank.named),
                         path=f"banks.{kind}.c{coefficient}.group{group}")
    finally:
        bank.release()
    print(f"BIT-EXACT banks: {kind} c={coefficient} {device} optimizer_device={optimizer_device} bf16={bf16}")


def _loader(groups):
    for group in groups:
        yield torch.cat([inputs for inputs, _ in group]), torch.cat([targets for _, targets in group])


def check_pipeline(kind, coefficient, devices, optimizer_device, bf16, updates, *, dion3_triton=False, triton=False):
    depth, microbatches = len(devices), max(3, len(devices))
    model = _model(dim=8, layers=2 * depth)
    groups = make_batches(updates=updates, microbatches=microbatches, dim=8, seed=47)
    overrides = ({"use_triton": dion3_triton} if kind == "dion3" else
                 {"use_triton": triton} if kind in ("adam", "adamw", "lion") else {})
    factory = optimizer_factory(kind, coefficient, **overrides)
    expected = run_reference(model, groups, factory, device=devices[0],
                             optimizer_device=optimizer_device, autocast_bf16=bf16)
    assert [update.eval_version for update in expected.updates] == [max(group - 1, 0) for group in range(updates)]
    all_records = []
    for continuation in (False, True):
        modules = [nn.Sequential(OrderedDict((str(index), copy.deepcopy(model[index]))
                                            for index in range(2 * stage, 2 * stage + 2)))
                   for stage in range(depth)]
        pipe = Pipeline(stage_modules=modules, devices=devices,
                        autocast=torch.bfloat16 if bf16 else None)
        records = {}

        def observe(stage, group, snapshot):
            assert (stage, group) not in records
            records[stage, group] = snapshot

        try:
            with pipe.train_session(optimizer_factory=factory, n_microbatches=microbatches,
                                    loss_fn=F.mse_loss, optimizer_device=optimizer_device) as trainer:
                if continuation:
                    first = trainer.run(_loader(groups[:1]), updates=1, observer=observe)
                    assert first.total_updates == 1
                    result = trainer.run(_loader(groups[1:]), updates=updates - 1, observer=observe)
                else:
                    result = trainer.run(_loader(groups), updates=updates, observer=observe)
                assert result.total_updates == updates
                assert result.peak_inflight <= trainer.max_inflight
                assert len(records) == depth * updates
                for group, update in enumerate(expected.updates):
                    for stage, module in enumerate(modules):
                        names = dict(module.named_parameters())
                        assert_exact(records[stage, group], _expected_snapshot(update, names),
                                     path=f"pipeline.{kind}.c{coefficient}.continued{continuation}.stage{stage}.group{group}")
                for bank in trainer.stages:
                    _check_owners(bank, coefficient, use_triton=triton if kind in ("adam", "adamw", "lion") else None)
            for module in modules:
                for name, parameter in module.named_parameters():
                    assert_exact(parameter, expected.updates[-1].weights[name], path="closed." + name)
                    assert parameter.grad is None
        finally:
            pipe.close()
        all_records.append(records)
    assert_exact(all_records[0], all_records[1], path="continuous_vs_1_plus_remaining")
    print(f"BIT-EXACT pipeline + 1/{updates - 1} continuation: {kind} c={coefficient} "
          f"{devices} optimizer_device={optimizer_device} bf16={bf16}; weights/grads/state at every update")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--devices", default="cpu,cpu", help="comma-separated CPU or CUDA stage devices")
    parser.add_argument("--bf16", action="store_true", help="also test BF16 autocast; never BF16 optimizer parameters")
    parser.add_argument("--updates", type=int, default=6, help="pipeline updates (at least 3; default 6)")
    parser.add_argument("--optimizers", nargs="+", choices=KINDS, default=KINDS,
                        help="optimizer kinds to check (default: all)")
    parser.add_argument("--dion3-triton", action="store_true",
                        help="use vendored Dion3 kernels for CUDA optimizer placement only")
    parser.add_argument("--triton", action="store_true",
                        help="check AdamEF/AdamW/Lion Triton on CUDA; explicitly keep CPU masters eager")
    suite = parser.add_mutually_exclusive_group()
    suite.add_argument("--unit-only", action="store_true", help="only independent optimizer tests")
    suite.add_argument("--pipeline-only", action="store_true", help="only exact banks/pipeline/continuation tests")
    args = parser.parse_args()
    devices = [value.strip() for value in args.devices.split(",") if value.strip()]
    if not devices or args.updates < 3:
        parser.error("provide at least one device and at least three updates")
    types = {torch.device(device).type for device in devices}
    if types not in ({"cpu"}, {"cuda"}):
        parser.error("devices must be uniformly CPU or CUDA")
    if types == {"cuda"} and not torch.cuda.is_available():
        parser.error("CUDA devices requested but CUDA is unavailable")
    if args.dion3_triton and (types != {"cuda"} or "dion3" not in args.optimizers):
        parser.error("--dion3-triton requires CUDA devices and --optimizers including dion3")
    if args.triton and not any(kind in ("adam", "adamw", "lion") for kind in args.optimizers):
        parser.error("--triton requires --optimizers including adam, adamw or lion")
    configure_determinism()
    if not args.pipeline_only:
        for device in dict.fromkeys(["cpu", devices[0]]):
            check_units(device, args.optimizers, dion3_triton=args.dion3_triton, triton=args.triton)
    if not args.unit_only:
        placements = (None, "cpu") if types == {"cuda"} else (None,)
        for bf16 in ((False, True) if args.bf16 else (False,)):
            for optimizer_device in placements:
                for kind in args.optimizers:
                    for coefficient in (0.0, 1.0):
                        check_banks(kind, coefficient, devices[0], optimizer_device, bf16, args.updates,
                                    dion3_triton=args.dion3_triton, triton=args.triton)
                        check_pipeline(kind, coefficient, devices, optimizer_device, bf16, args.updates,
                                       dion3_triton=args.dion3_triton, triton=args.triton)
    print("ALL REQUESTED OPTIMIZER CHECKS PASSED")


if __name__ == "__main__":
    main()
