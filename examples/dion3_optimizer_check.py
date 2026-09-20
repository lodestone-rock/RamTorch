"""Independent Dion3 equations, state/precision checks, and optional CUDA kernels.

Run from the repository root with a working PyTorch environment::

    PYTHONPATH=. python examples/dion3_optimizer_check.py --device cpu
    PYTHONPATH=. python examples/dion3_optimizer_check.py --device cuda:0 --triton

The Float64 oracle uses Python row ranking and dot products, hardcoded Polar
Express coefficients, and independent residual/row-EMA/normalization/EF equations.
It never calls production optimizer helpers to calculate an expected update.
Equation comparisons allow arithmetic rounding; serialization, storage rebinding,
ambient autocast isolation, and the separate 2BW schedule checks remain bit-exact.
CPU checks do not validate CUDA transfers, overlap, or Triton kernels.
"""
from __future__ import annotations

import argparse
import copy
import io
import math
from unittest.mock import patch

# The reference configures CUDA environment flags before importing torch.
from pipedream_2bw_reference import assert_exact, configure_determinism
import torch
from torch import nn

from ramtorch.dion3 import Dion3


RATES = (0.07, 0.023, 0.0, 0.11, 0.017, 0.035, 0.009)


def _polar_equations(matrix, epsilon):
    """Independent Float64 polynomial, including tall/wide Gram orientation."""
    values = matrix.detach().cpu().double().tolist()
    tall = len(values) > len(values[0])
    if tall:
        values = [list(column) for column in zip(*values)]
    norm = math.sqrt(sum(x * x for row in values for x in row))
    values = [[x / (1.02 * norm + epsilon) for x in row] for row in values]

    def multiply(left, right):
        columns = list(zip(*right))
        return [[sum(x * y for x, y in zip(row, column)) for column in columns]
                for row in left]

    # Deliberately not imported from ramtorch.dion3 or the vendored kernels.
    coefficients = (
        (8.156554524902461, -22.48329292557795, 15.878769915207462),
        (4.042929935166739, -2.808917465908714, 0.5000178451051316),
        (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
        (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
        (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
    )
    for a, b, c in coefficients:
        gram = multiply(values, list(zip(*values)))
        squared = multiply(gram, gram)
        polynomial = [[b * x + c * y for x, y in zip(row, row2)]
                      for row, row2 in zip(gram, squared)]
        product = multiply(polynomial, values)
        values = [[a * x + y for x, y in zip(row, row2)]
                  for row, row2 in zip(values, product)]
    if tall:
        values = [list(column) for column in zip(*values)]
    return torch.tensor(values, dtype=torch.float64)


def _parameters(device="cpu", dtype=torch.float64):
    # Matrices: tall/wide/square/always-zero/late/unused/frozen. Fallbacks
    # deliberately include 2D heads as well as scalars and vectors.
    shapes = ((7, 3), (3, 7), (4, 4), (3, 4), (5, 2), (2, 2), (3, 2),
              (4,), (), (2, 3), (4,), (2, 3), (3,), (2,), (3,))
    result = []
    for index, shape in enumerate(shapes):
        values = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)
        values = 0.6 * (values * 0.73 + index * 0.31).cos()
        result.append(nn.Parameter(values.to(device=device, dtype=dtype),
                                   requires_grad=index not in (6, 12)))
    return result


def _gradients(parameters, call):
    result = []
    for index, parameter in enumerate(parameters):
        if (call == 4 or index in (5, 13) or (index in (4, 14) and call < 2)
                or (call == 1 and index in (1, 7, 10)) or (call == 3 and index in (2, 8))):
            result.append(None)
            continue
        values = torch.arange(parameter.numel(), dtype=torch.float64).reshape(parameter.shape)
        values = 0.8 * (values * 0.43 + index * 0.37 + call * 0.61 + 0.2).sin()
        if index == 3 or (call in (2, 6) and index in (0, 9, 11)):
            values.zero_()
        result.append(values.to(device=parameter.device, dtype=parameter.dtype))
    return result


def _assign(parameters, gradients):
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = None if gradient is None else gradient.clone()


def _optimizer(parameters, coefficient, **overrides):
    options = dict(lr=0.017, fraction=0.5, mu=0.7, muon_beta2=0.8,
                   weight_decay=0.13, epsilon=1e-7, ef_coefficient=coefficient,
                   betas=(0.6, 0.8), adjust_lr="spectral_norm", ns_dtype=torch.float64)
    options.update(overrides)
    return Dion3([
        {"params": parameters[:7]},  # Default matrix role is intentional.
        {"params": parameters[7:10], "algorithm": "adamw", "weight_decay": 0.04,
         "betas": (0.4, 0.75), "epsilon": 0.013},
        {"params": parameters[10:], "algorithm": "lion", "weight_decay": 0.09,
         "betas": (0.3, 0.85), "ef_coefficient": coefficient * 0.35},
    ], **options)


def _set_lrs(optimizer, call):
    rates = [RATES[call % len(RATES)] * scale for scale in (1.0, 0.7, 1.3)]
    for group, lr in zip(optimizer.param_groups, rates):
        group["lr"] = lr
    return rates


class Dion3Equations:
    """Independent selected-row residual, variance, full-decay and delay EF."""

    def __init__(self, parameters, optimizer):
        self.weights = [p.detach().cpu().double().clone() for p in parameters]
        self.active = [p.requires_grad for p in parameters]
        indices = {id(parameter): index for index, parameter in enumerate(parameters)}
        self.groups = [dict(copy.deepcopy({k: v for k, v in group.items() if k != "params"}),
                            indices=[indices[id(p)] for p in group["params"]])
                       for group in optimizer.param_groups]
        self.states = [{} for _ in parameters]

    def step(self, gradients, rates):
        for group, lr in zip(self.groups, rates):
            group["ef_step"] += 1
            coefficient = group["ef_coefficient"]
            for index in group["indices"]:
                weight, state = self.weights[index], self.states[index]
                if gradients[index] is None or not self.active[index]:
                    if "previous_update" in state:
                        state["previous_update"] = torch.zeros_like(weight)
                    continue
                gradient = gradients[index].detach().cpu().double()
                if group["algorithm"] in ("dion3", "nordion2"):
                    rows, columns = weight.shape
                    residual = state.get("momentum", torch.zeros_like(weight)) + gradient
                    scores = [sum(abs(x) for x in row) for row in residual.tolist()]
                    count = math.ceil(group["fraction"] * rows)
                    selected = sorted(range(rows), key=lambda row: scores[row], reverse=True)[:count]
                    # No topk or production row-selection helper in this oracle.
                    direction = _polar_equations(residual[selected], group["epsilon"])
                    state["momentum"] = residual.clone()
                    for row in selected:
                        state["momentum"][row] = group["mu"] * residual[row]
                    variance = state.get("variance_neuron", torch.zeros(rows, 1, dtype=torch.float64)).clone()
                    original_norm = math.sqrt(sum(x * x for row in direction.tolist() for x in row))
                    normalized = direction.clone()
                    for slot, row in enumerate(selected):
                        row_mean = sum(x * x for x in direction[slot].tolist()) / columns
                        variance[row, 0] = (group["muon_beta2"] * float(variance[row, 0])
                                            + (1 - group["muon_beta2"]) * row_mean)
                        normalized[slot] = direction[slot] / (math.sqrt(float(variance[row, 0])) + 1e-8)
                    normalized_norm = math.sqrt(sum(x * x for row in normalized.tolist() for x in row))
                    normalized = normalized * (original_norm / max(normalized_norm, 1e-8))
                    torch.testing.assert_close(normalized.norm(), direction.norm(), rtol=2e-14, atol=2e-14)
                    state["variance_neuron"] = variance
                    scale = (math.sqrt(rows / columns) if group["adjust_lr"] == "spectral_norm" else
                             0.2 * math.sqrt(max(rows, columns)) if group["adjust_lr"] == "rms_norm" else 1.0)
                    update = torch.zeros_like(weight)
                    for slot, row in enumerate(selected):
                        update[row] = lr * scale * normalized[slot]
                else:
                    beta1, beta2 = group["betas"]
                    moment = state.get("exp_avg", torch.zeros_like(weight))
                    if group["algorithm"] == "lion":
                        update = lr * (beta1 * moment + (1 - beta1) * gradient).sign()
                        state["exp_avg"] = beta2 * moment + (1 - beta2) * gradient
                    else:
                        state["step"] = state.get("step", 0) + 1
                        moment = beta1 * moment + (1 - beta1) * gradient
                        variance = beta2 * state.get("exp_avg_sq", torch.zeros_like(weight)) + (1 - beta2) * gradient.square()
                        state["exp_avg"], state["exp_avg_sq"] = moment, variance
                        update = lr * (moment / (1 - beta1 ** state["step"])) / (
                            (variance / (1 - beta2 ** state["step"])).sqrt() + group["epsilon"])
                # ALL rows decay at CURRENT weights, using unadjusted base LR.
                update = update + lr * group["weight_decay"] * weight
                correction = (coefficient * (update - state.get("previous_update", torch.zeros_like(weight)))
                              if group["ef_step"] > 1 else torch.zeros_like(weight))
                self.weights[index] = weight - (update + correction)
                if coefficient:
                    state["previous_update"] = update.clone()

    def check(self, parameters, optimizer, path):
        for actual, expected in zip(optimizer.param_groups, self.groups):
            assert actual["ef_step"] == expected["ef_step"], path + ": group clock"
        for index, parameter in enumerate(parameters):
            location = f"{path}.parameter{index}"
            # Five degree-five polynomials and a different row ordering change
            # rounding; this is an equation oracle, not a schedule comparison.
            torch.testing.assert_close(parameter.detach().cpu(), self.weights[index],
                                       rtol=2e-11, atol=2e-12, msg=location)
            actual, expected = optimizer.state.get(parameter, {}), self.states[index]
            assert actual.keys() == expected.keys(), location + ": state keys"
            for key, value in expected.items():
                if isinstance(value, torch.Tensor):
                    torch.testing.assert_close(actual[key].cpu(), value, rtol=2e-11, atol=2e-12,
                                               msg=location + "." + key)
                else:
                    assert actual[key] == value, location + ": fallback counter"


def check_equations(device):
    for coefficient in (0.0, 1.0):
        for fraction in (0.25, 0.5, 1.0):
            for scale in (None, "spectral_norm", "rms_norm"):
                for empty_start in (False, True):
                    parameters = _parameters(device)
                    optimizer = _optimizer(parameters, coefficient, fraction=fraction, adjust_lr=scale)
                    equations = Dion3Equations(parameters, optimizer)
                    for call in range(7 + int(empty_start)):
                        gradients = ([None] * len(parameters) if empty_start and call == 0 else
                                     _gradients(parameters, call - int(empty_start)))
                        rates = _set_lrs(optimizer, call)
                        _assign(parameters, gradients)
                        before = [p.detach().clone() for p in parameters]
                        states = [copy.deepcopy(optimizer.state.get(p, {})) for p in parameters]
                        optimizer.step()
                        equations.step(gradients, rates)
                        path = f"dion3.c{coefficient}.fraction{fraction}.{scale}.empty{empty_start}.call{call}"
                        equations.check(parameters, optimizer, path)
                        for index, parameter in enumerate(parameters):
                            assert_exact(parameter.grad, gradients[index], path=path + ".gradient")
                            if gradients[index] is None or not parameter.requires_grad:
                                assert_exact(parameter, before[index], path=path + ".skipped_weight")
                                state = optimizer.state.get(parameter, {})
                                assert state.keys() == states[index].keys()
                                for key, value in states[index].items():
                                    if key == "previous_update":
                                        assert not torch.count_nonzero(state[key]).item()
                                    else:
                                        assert_exact(state[key], value, path=path + ".skipped." + key)
    print(f"PASS Dion3 Float64 equations: partial/full rows, tall/wide, residual, row EMA, norm, "
          f"full decay, EF 0/1, LR changes, zero/missing/frozen/late/empty startup: {device}")


def check_fallbacks(device):
    from ramtorch.delayed_optim import Lion

    for algorithm, bounds in (("adamw", slice(7, 10)), ("lion", slice(10, None))):
        for dtype in (torch.float32, torch.float64):
            for decay in (0.0, 0.13):
                for coefficient in ((0.0,) if algorithm == "adamw" else (0.0, 1.0)):
                    all_parameters = _parameters(device, dtype)
                    actual = all_parameters[bounds]
                    expected = [nn.Parameter(p.detach().clone(), requires_grad=p.requires_grad) for p in actual]
                    options = dict(lr=0.07, betas=(0.6, 0.8), weight_decay=decay)
                    optimizer = Dion3([{"params": actual, "algorithm": algorithm}], epsilon=0.031,
                                      ef_coefficient=coefficient, **options)
                    native = (torch.optim.AdamW(expected, eps=0.031, foreach=False, fused=False, **options)
                              if algorithm == "adamw" else Lion(expected, ef_coefficient=coefficient, **options))
                    tolerance = 3e-6 if dtype == torch.float32 else 3e-13
                    for call in range(7):
                        optimizer.param_groups[0]["lr"] = native.param_groups[0]["lr"] = RATES[call]
                        gradients = _gradients(all_parameters, call)[bounds]
                        gradients = [g if p.requires_grad else None for p, g in zip(actual, gradients)]
                        _assign(actual, gradients)
                        _assign(expected, gradients)
                        optimizer.step()
                        native.step()
                        for left, right in zip(actual, expected):
                            ours, theirs = optimizer.state.get(left, {}), native.state.get(right, {})
                            # Lion groups form lr*sign + (lr*wd)*p, whereas
                            # standalone Lion uses lr*(sign + wd*p). The
                            # moments match exactly, but full updates can round.
                            torch.testing.assert_close(left, right, rtol=tolerance, atol=tolerance)
                            assert ours.keys() == theirs.keys()
                            for key in ours:
                                if algorithm == "lion" and key == "exp_avg":
                                    assert_exact(ours[key], theirs[key], path="dion3.lion.moment")
                                else:
                                    torch.testing.assert_close(torch.as_tensor(ours[key]).cpu().double(),
                                                               torch.as_tensor(theirs[key]).cpu().double(),
                                                               rtol=tolerance, atol=tolerance)
    print(f"PASS Dion3 explicit fallback: native AdamW and standalone Lion FP32/FP64, Lion EF and exact moments: {device}")


def check_budget_resume(device):
    plain, corrected = _parameters(device), _parameters(device)
    ordinary, ef = _optimizer(plain, 0.0), _optimizer(corrected, 1.0)
    assert not ordinary.state and not ef.state, "constructor allocated state"
    for parameters, optimizer in ((plain, ordinary), (corrected, ef)):
        optimizer.step()  # Even an empty logical call must not allocate state.
        assert not optimizer.state
        _assign(parameters, _gradients(parameters, 0))
        optimizer.step()
    for index, (p, q) in enumerate(zip(plain, corrected)):
        base, extra = ordinary.state.get(p, {}), ef.state.get(q, {})
        if not base:
            assert not extra
            continue
        assert extra.keys() == base.keys() | {"previous_update"}
        assert_exact({key: extra[key] for key in base}, base, path="dion3.budget.base")
        if index < 7:
            assert base.keys() == {"momentum", "variance_neuron"}
            assert base["momentum"].shape == p.shape
            assert base["variance_neuron"].shape == (p.shape[0], 1)
        tensors = [q, q.grad] + [value for value in extra.values() if isinstance(value, torch.Tensor)]
        pointers = [tensor.untyped_storage().data_ptr() for tensor in tensors]
        assert len(set(pointers)) == len(pointers), "state/history aliases weights, gradients, or row state"
        for value in extra.values():
            if isinstance(value, torch.Tensor):
                assert value.dtype == q.dtype and value.device == q.device and not value.requires_grad
        nbytes = lambda state: sum(v.numel() * v.element_size() for v in state.values() if isinstance(v, torch.Tensor))
        assert nbytes(extra) - nbytes(base) == q.numel() * q.element_size()
        if index < 7:
            assert nbytes(base) == (p.numel() + p.shape[0]) * p.element_size()

    for coefficient in (0.0, 1.0):
        parameters = _parameters(device)
        optimizer = _optimizer(parameters, coefficient)
        snapshots = [(copy.deepcopy(parameters), copy.deepcopy(optimizer.state_dict()))]
        calls = [[None] * len(parameters)] + [_gradients(parameters, call) for call in range(7)]
        for call, gradients in enumerate(calls):
            _set_lrs(optimizer, call)
            _assign(parameters, gradients)
            optimizer.step()
            snapshots.append((copy.deepcopy(parameters), copy.deepcopy(optimizer.state_dict())))
        for split in (0, 1, 2, 4, 6):
            weights, state = snapshots[split]
            buffer = io.BytesIO()
            torch.save({"weights": [p.detach().clone() for p in weights], "optimizer": state}, buffer)
            buffer.seek(0)
            saved = torch.load(buffer, map_location=device, weights_only=True)
            resumed = [nn.Parameter(value, requires_grad=p.requires_grad) for value, p in zip(saved["weights"], weights)]
            restored = _optimizer(resumed, 0.37, lr=0.9, fraction=1.0, mu=0.1,
                                  muon_beta2=0.2, ns_dtype=torch.bfloat16, adjust_lr=None)
            restored.load_state_dict(saved["optimizer"])
            assert_exact(restored.state_dict(), state, path=f"dion3.resume{split}.loaded")
            for call in range(split, len(calls)):
                _set_lrs(restored, call)
                _assign(resumed, calls[call])
                restored.step()
                expected_weights, expected_state = snapshots[call + 1]
                assert_exact(resumed, expected_weights, path=f"dion3.resume{split}.call{call}.weights")
                assert_exact(restored.state_dict(), expected_state, path=f"dion3.resume{split}.call{call}.state")
    print(f"PASS Dion3 lazy full-residual + row-variance budget, one EF tensor, exact BytesIO continuation: {device}")


def _reject(operation, label):
    try:
        operation()
    except (ValueError, TypeError, RuntimeError):
        return
    raise AssertionError(f"Dion3 accepted {label}")


def check_rejections(device):
    for key, value in (("fraction", 0.0), ("fraction", 1.01), ("fraction", math.nan),
                       ("mu", 1.0), ("muon_beta2", -0.1), ("epsilon", 0.0),
                       ("ef_coefficient", -1.0), ("adjust_lr", "unknown"),
                       ("ns_dtype", torch.float16), ("use_triton", 1)):
        _reject(lambda: Dion3([_parameters(device)[0]], **{key: value}), f"option {key}={value}")
    for shape in ((), (3,), (2, 2, 2), (0, 3), (3, 0)):
        _reject(lambda: Dion3([nn.Parameter(torch.ones(shape, device=device))]), f"implicit matrix {shape}")
    for dtype in (torch.float16, torch.bfloat16, torch.complex64):
        _reject(lambda: Dion3([nn.Parameter(torch.ones(2, 3, device=device, dtype=dtype))]), f"parameter {dtype}")
    if torch.device(device).type == "cpu":
        _reject(lambda: Dion3([_parameters(device)[0]], use_triton=True), "CPU Triton matrix")

    parameters = _parameters(device)
    optimizer = _optimizer(parameters, 1.0)
    _assign(parameters, _gradients(parameters, 0))
    optimizer.step()
    pristine = copy.deepcopy(optimizer.state_dict())
    parameter_id = pristine["param_groups"][0]["params"][0]
    row = pristine["state"][parameter_id]["variance_neuron"]
    bad_rows = (row.flatten(), torch.zeros_like(parameters[0]), row.to(torch.float16),
                row.to_sparse(), "not a tensor")
    for bad_row in bad_rows:
        malformed = copy.deepcopy(pristine)
        malformed["state"][parameter_id]["variance_neuron"] = bad_row
        _reject(lambda: optimizer.load_state_dict(malformed), "malformed saved row state")
        assert_exact(optimizer.state_dict(), pristine, path="dion3.reject.atomic_load")
    for key in ("variance_neuron", "momentum", "previous_update"):
        malformed = copy.deepcopy(pristine)
        del malformed["state"][parameter_id][key]
        _reject(lambda: optimizer.load_state_dict(malformed), "missing saved " + key)
        assert_exact(optimizer.state_dict(), pristine, path="dion3.reject.missing_state")
    # A malformed later parameter cannot partially update earlier valid ones.
    later = parameters[1]
    valid_row = optimizer.state[later]["variance_neuron"]
    for bad_row in (valid_row.flatten(), valid_row.float()):
        optimizer.state[later]["variance_neuron"] = bad_row
        before_weights = [p.detach().clone() for p in parameters]
        before_state = copy.deepcopy(optimizer.state_dict())
        _reject(optimizer.step, "malformed live row state")
        assert_exact(parameters, before_weights, path="dion3.reject.atomic_weights")
        assert_exact(optimizer.state_dict(), before_state, path="dion3.reject.atomic_state")
    optimizer.state[later]["variance_neuron"] = valid_row
    print(f"PASS Dion3 parameter/options validation and atomic malformed saved/live row-state rejection: {device}")


def check_rebind(device):
    for coefficient in (0.0, 1.0):
        actual, expected = _parameters(device), _parameters(device)
        rebound, ordinary = _optimizer(actual, coefficient), _optimizer(expected, coefficient)
        for call in range(6):
            _set_lrs(rebound, call)
            _set_lrs(ordinary, call)
            old_storage = [p.detach() for p in actual]
            old_values = [p.detach().clone() for p in actual]
            state_pointers = {id(p): {key: value.untyped_storage().data_ptr() for key, value in state.items()
                                      if isinstance(value, torch.Tensor)} for p, state in rebound.state.items()}
            with torch.no_grad():
                for p, q in zip(actual, expected):
                    p.data = p.detach().clone().add_(0.013 * (call + 1))
                    q.add_(0.013 * (call + 1))
            # Freeze an initialized parameter with an attached gradient, then
            # unfreeze it. Existing EF history must clear, unlike its residual.
            actual[0].requires_grad_(call != 3)
            expected[0].requires_grad_(call != 3)
            gradients = _gradients(actual, call)
            _assign(actual, gradients)
            _assign(expected, gradients)
            before = copy.deepcopy(rebound.state.get(actual[0], {}))
            rebound.step()
            ordinary.step()
            assert_exact(actual, expected, path="dion3.rebind.weights")
            assert_exact(rebound.state_dict(), ordinary.state_dict(), path="dion3.rebind.state")
            assert_exact(old_storage, old_values, path="dion3.rebind.old_banks")
            for p, state in rebound.state.items():
                for key, pointer in state_pointers.get(id(p), {}).items():
                    assert state[key].untyped_storage().data_ptr() == pointer, "rebinding replaced optimizer state"
            if call == 3:
                for key, value in before.items():
                    if key == "previous_update":
                        assert not torch.count_nonzero(rebound.state[actual[0]][key]).item()
                    else:
                        assert_exact(rebound.state[actual[0]][key], value, path="dion3.freeze_initialized")
    print(f"PASS Dion3 same-identity storage rebinding, preserved old banks/state storage, frozen-history clearing: {device}")


def check_precision(device):
    if torch.device(device).type == "cuda" and not torch.cuda.is_bf16_supported():
        print(f"SKIP Dion3 BF16 ambient autocast: unsupported device {device}")
        return
    for dtype in (torch.float32, torch.float64):
        histories = {}
        for ns_dtype in (torch.float32, torch.float64, torch.bfloat16):
            actual, expected = _parameters(device, dtype), _parameters(device, dtype)
            ambient, ordinary = (_optimizer(p, 1.0, ns_dtype=ns_dtype) for p in (actual, expected))
            for call in range(3):
                gradients = _gradients(actual, call)
                _assign(actual, gradients)
                _assign(expected, gradients)
                with torch.autocast(torch.device(device).type, dtype=torch.bfloat16):
                    ambient.step()
                ordinary.step()
                assert_exact(actual, expected, path=f"dion3.autocast.{dtype}.{ns_dtype}.weights")
                assert_exact(ambient.state_dict(), ordinary.state_dict(), path="dion3.autocast.state")
                for parameter in actual:
                    for value in ambient.state.get(parameter, {}).values():
                        if isinstance(value, torch.Tensor):
                            assert value.dtype == dtype and value.device == parameter.device
            histories[ns_dtype] = [ambient.state[p]["previous_update"].clone() for p in actual[:3]]
        for first, second in ((torch.float32, torch.float64), (torch.float32, torch.bfloat16)):
            assert any(not torch.equal(a, b) for a, b in zip(histories[first], histories[second])), \
                f"Dion3 ignored explicit NS precision {first} versus {second}"
    default = Dion3([_parameters(device)[0]])
    assert default.param_groups[0]["ns_dtype"] == torch.bfloat16
    print(f"PASS Dion3 FP32/FP64/BF16 working precision, FP32/FP64 state, exact ambient-autocast isolation: {device}")


def check_triton(device):
    """Explicit opt-in only: execute vendored kernels, never silently fall back."""
    if torch.device(device).type != "cuda" or not torch.cuda.is_bf16_supported():
        raise ValueError("--triton requires a BF16-capable CUDA device")
    from ramtorch import _dion3_triton

    # Exercise multiple tiles and masked boundaries, not just tiny matrices.
    # Check each fused kernel against FP64 equations rounded once to BF16;
    # this is stronger than only comparing final, small parameter updates.
    generator = torch.Generator().manual_seed(59)
    for rows, columns in ((1, 1), (67, 131), (257, 129)):
        matrix = torch.randn(rows, columns, generator=generator).to(device, torch.bfloat16)
        for source in (matrix, matrix.T.contiguous().T):
            expected_gram = (source.double() @ source.double().T).to(torch.bfloat16)
            gram = _dion3_triton.ns_line_1(source)
            torch.testing.assert_close(gram, expected_gram, rtol=0.008, atol=0.008)
            alpha, beta = 0.42323551169305323, -1.7097828382687081
            expected_poly = (alpha * (gram.double() @ gram.double().T) + beta * gram.double()).to(torch.bfloat16)
            polynomial = _dion3_triton.ns_line_2(gram, alpha=alpha, beta=beta)
            torch.testing.assert_close(polynomial, expected_poly, rtol=0.008, atol=0.008)
        zero = _dion3_triton.polar_express_triton(torch.zeros_like(matrix))
        assert torch.isfinite(zero).all() and not torch.count_nonzero(zero)
    print(f"PASS Dion3 Triton symmetric kernels: FP64 equations, multi-tile/odd/strided/zero: {device}")

    for shape in ((32, 16), (16, 32)):
        for fraction in (0.25, 1.0):
            generator = torch.Generator().manual_seed(53)
            values = torch.randn(shape, generator=generator).to(device)
            actual, duplicate = nn.Parameter(values.clone()), nn.Parameter(values.clone())
            eager_parameter = nn.Parameter(values.clone())
            options = dict(lr=0.02, fraction=fraction, ef_coefficient=1.0, ns_dtype=torch.bfloat16)
            optimized = Dion3([actual], use_triton=True, **options)
            repeated = Dion3([duplicate], use_triton=True, **options)
            eager = Dion3([eager_parameter], **options)
            # A wrapper proves the requested backend is actually called. The
            # implementation imports the entrypoint on each matrix update.
            with patch.object(_dion3_triton, "polar_express_triton",
                              wraps=_dion3_triton.polar_express_triton) as kernel, \
                 patch.object(_dion3_triton, "ns_line_1", wraps=_dion3_triton.ns_line_1) as gram_kernel, \
                 patch.object(_dion3_triton, "ns_line_2", wraps=_dion3_triton.ns_line_2) as polynomial_kernel:
                for call in range(3):
                    gradient = torch.randn(shape, generator=generator).to(device)
                    for parameter in (actual, duplicate, eager_parameter):
                        parameter.grad = gradient.clone()
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        optimized.step()
                    repeated.step()
                    eager.step()
                    assert_exact(actual, duplicate, path="dion3.triton.repeat.weights")
                    assert_exact(optimized.state_dict(), repeated.state_dict(), path="dion3.triton.repeat.state")
                    assert_exact(optimized.state[actual]["momentum"], eager.state[eager_parameter]["momentum"],
                                 path="dion3.triton.residual")
                    # BF16 fused kernels and eager matmuls round differently.
                    # Compare complete update RMS, not weight-relative error
                    # (which could hide a completely missing small update).
                    update = optimized.state[actual]["previous_update"]
                    reference = eager.state[eager_parameter]["previous_update"]
                    relative_error = (update - reference).norm() / reference.norm().clamp_min(1e-12)
                    assert float(relative_error) < 0.12, f"Triton/eager update RMS error {float(relative_error):.5f}"
                assert kernel.call_count == 6, "vendored kernel path was not executed"
                assert gram_kernel.call_count == polynomial_kernel.call_count == 30, \
                    "both five-iteration symmetric Triton kernels must execute"
            torch.cuda.synchronize(device)
    print(f"PASS Dion3 vendored Triton tall/wide partial/full: actual backend calls, exact repeat/autocast, "
          f"residual parity and BF16 eager update RMS error <12%: {device}")


def check_dion3_units(device, *, triton=False):
    torch.set_num_threads(1)
    check_equations(device)
    check_fallbacks(device)
    check_budget_resume(device)
    check_rejections(device)
    check_rebind(device)
    check_precision(device)
    if triton:
        check_triton(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="CPU or CUDA device (default: cpu)")
    parser.add_argument("--triton", action="store_true", help="also execute optional vendored CUDA BF16 kernels")
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("device must be CPU or CUDA")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.triton and device.type != "cuda":
        parser.error("--triton requires --device cuda:N")
    configure_determinism()
    check_dion3_units(args.device, triton=args.triton)
    print("ALL REQUESTED DION3 OPTIMIZER CHECKS PASSED")


if __name__ == "__main__":
    main()
