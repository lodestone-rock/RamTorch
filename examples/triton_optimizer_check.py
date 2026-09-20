"""Independent AdamEF/AdamW/Lion equations and strict optional Triton checks.

Run from the repository root with a working PyTorch environment::

    CUDA_VISIBLE_DEVICES= PYTHONPATH=. python examples/triton_optimizer_check.py --device cpu
    PYTHONPATH=. python examples/triton_optimizer_check.py --device cuda:0

CUDA selects use_triton=True and verifies calls to the real kernel wrapper;
CPU runs eager numerical/serialization checks plus strict CUDA-only rejection
and a fresh-process optional-import check. CPU success is NOT GPU validation.
FP32 comparisons allow 4e-6 and FP64 2e-12 (rtol and atol), declared in advance.
The independent Python Float64 equations, eager and native Adam references allow
rounding differences. Same-backend resume, rebinding and autocast remain exact.
"""
from __future__ import annotations

import argparse
import copy
import io
import math
import subprocess
import sys
from contextlib import nullcontext
from unittest.mock import patch

# Import the reference first: it sets deterministic CUDA environment flags.
from pipedream_2bw_reference import assert_exact, configure_determinism
from pipedream_2bw_optimizer_check import ScalarEquations
import torch
from torch import nn

from ramtorch.delayed_optim import AdamEF, Lion


KINDS = ("adam", "adamw", "lion")
TOLERANCES = {torch.float32: 4e-6, torch.float64: 2e-12}
RATES = (0.07, 0.023, 0.0, 0.11, 0.017, 0.035, 0.009)
DECAYS = (0.13, 0.0, 0.04, 0.19, 0.03, 0.07, 0.0)
# An empty tensor, scalar, masked odd tails, matrix, and >typical-block tensor;
# then late, unused, frozen-with-attached-gradient, and temporarily frozen.
SHAPES = ((0,), (), (17,), (2, 3), (8193,), (5,), (3,), (3,), (7,))


def _parameters(device, dtype):
    result = []
    for index, shape in enumerate(SHAPES):
        values = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)
        values = 0.6 * (values * 0.17 + index * 0.31).cos()
        if index == 1:
            values.zero_()  # Exact zero Lion direction and Adam numerator at startup.
        result.append(nn.Parameter(values.to(device=device, dtype=dtype), requires_grad=index != 7))
    return result


def _optimizer(kind, parameters, coefficient, *, triton=False, **overrides):
    options = dict(lr=0.017, betas=(0.6, 0.8), weight_decay=0.13,
                   ef_coefficient=coefficient, use_triton=triton)
    options.update(overrides)
    if kind != "lion":
        options.update(eps=0.031, decoupled_weight_decay=kind == "adamw")
    return (Lion if kind == "lion" else AdamEF)(parameters, **options)


def _groups(parameters):
    return [{"params": parameters[:5]}, {"params": parameters[5:], "weight_decay": 0.04}]


def _gradients(parameters, call):
    result = []
    for index, parameter in enumerate(parameters):
        if (call < 0 or call == 4 or index == 6 or (index == 5 and call < 2)
                or (call == 1 and index in (1, 3))):
            result.append(None)
            continue
        values = torch.arange(parameter.numel(), dtype=torch.float64).reshape(parameter.shape)
        values = 0.8 * (values * 0.43 + index * 0.37 + call * 0.61 + 0.2).sin()
        if (index == 2 and call in (2, 6)) or (index == 1 and call == 0):
            values.zero_()
        result.append(values.to(device=parameter.device, dtype=parameter.dtype))
    return result


def _assign(parameters, gradients, call):
    parameters[8].requires_grad_(call not in (3, 4))
    for parameter, gradient in zip(parameters, gradients):
        parameter.grad = None if gradient is None else gradient.clone()


def _schedule(optimizer, call):
    rates, decays = [], []
    for index, group in enumerate(optimizer.param_groups):
        group["lr"] = RATES[call % len(RATES)] * (1.0 + 0.2 * index)
        group["weight_decay"] = DECAYS[(call + index) % len(DECAYS)]
        rates.append(group["lr"])
        decays.append(group["weight_decay"])
    return rates, decays


def _close(actual, expected, dtype, path):
    expected = torch.as_tensor(expected, dtype=torch.float64, device=actual.device).reshape(actual.shape)
    tolerance = TOLERANCES[dtype]
    torch.testing.assert_close(actual.detach().double(), expected, rtol=tolerance, atol=tolerance,
                               msg=lambda message: f"{path}: {message}")


def _check_equations(equations, parameters, optimizer, dtype, path):
    for index, parameter in enumerate(parameters):
        location = f"{path}.parameter{index}"
        _close(parameter, equations.weights[index], dtype, location)
        actual, expected = optimizer.state.get(parameter, {}), equations.states[index]
        if expected is None:
            assert not actual, location + ": lazy state allocated for unused/frozen parameter"
            continue
        keys = {"exp_avg"}
        if equations.kind != "lion":
            keys.update(("exp_avg_sq", "step"))
            assert actual["step"] == expected["step"], location + ": moments advanced twice"
        if equations.coefficients[index] > 0:
            keys.add("previous_update")
        assert actual.keys() == keys, location + ": unexpected persistent state"
        tensors = []
        for name in keys - {"step"}:
            value = actual[name]
            _close(value, expected[name], dtype, location + "." + name)
            assert value.shape == parameter.shape and value.dtype == parameter.dtype
            assert value.device == parameter.device and value.is_contiguous() and not value.requires_grad
            tensors.append(value)
        expected_count = (1 if equations.kind == "lion" else 2) + (equations.coefficients[index] > 0)
        nbytes = sum(value.numel() * value.element_size() for value in tensors)
        assert nbytes == expected_count * parameter.numel() * parameter.element_size()
        if parameter.numel():
            pointers = [tensor.untyped_storage().data_ptr() for tensor in [parameter] + tensors]
            if parameter.grad is not None:
                pointers.append(parameter.grad.untyped_storage().data_ptr())
            assert len(set(pointers)) == len(pointers), location + ": state aliases parameter/gradient"
    assert all(group["ef_step"] == equations.calls for group in optimizer.param_groups)


def check_equations(kind, coefficient, device, dtype, *, empty_start=False, mixed_groups=False):
    triton = torch.device(device).type == "cuda"
    parameters = _parameters(device, dtype)
    eager_parameters = _parameters(device, dtype)
    groups, eager_groups = _groups(parameters), _groups(eager_parameters)
    second_coefficient = 0.0 if mixed_groups else coefficient
    if mixed_groups:
        groups[1]["ef_coefficient"] = eager_groups[1]["ef_coefficient"] = second_coefficient
    optimizer = _optimizer(kind, groups, coefficient, triton=triton)
    eager = _optimizer(kind, eager_groups, coefficient)
    assert not optimizer.state, "constructor allocated state"
    equations = ScalarEquations(kind, [p.detach().cpu().flatten().double().tolist() for p in parameters],
                                [coefficient] * 5 + [second_coefficient] * 4)
    calls = [-1] + list(range(7)) if empty_start else list(range(7))
    for call in calls:
        gradients = _gradients(parameters, call)
        _assign(parameters, gradients, call)
        _assign(eager_parameters, gradients, call)
        rates, decays = _schedule(optimizer, call)
        _schedule(eager, call)
        before = [p.detach().clone() for p in parameters]
        before_state = [copy.deepcopy(optimizer.state.get(p, {})) for p in parameters]
        optimizer.step()
        eager.step()
        scalar_gradients = [None if g is None or not p.requires_grad else g.flatten().cpu().double().tolist()
                            for p, g in zip(parameters, gradients)]
        equations.step(scalar_gradients, [rates[0]] * 5 + [rates[1]] * 4,
                       [decays[0]] * 5 + [decays[1]] * 4)
        path = f"{kind}.c{coefficient}.{dtype}.empty{empty_start}.mixed{mixed_groups}.call{call}"
        _check_equations(equations, parameters, optimizer, dtype, path)
        for index, (actual, expected) in enumerate(zip(parameters, eager_parameters)):
            _close(actual, expected, dtype, path + ".eager.weight")
            actual_state, eager_state = optimizer.state.get(actual, {}), eager.state.get(expected, {})
            assert actual_state.keys() == eager_state.keys()
            for key, value in actual_state.items():
                if isinstance(value, torch.Tensor):
                    _close(value, eager_state[key], dtype, path + ".eager." + key)
                else:
                    assert value == eager_state[key]
            assert_exact(actual.grad, gradients[index], path=path + ".unchanged_gradient")
            if scalar_gradients[index] is None:
                assert_exact(actual, before[index], path=path + ".skipped_weight")
                assert actual_state.keys() == before_state[index].keys()
                for key, value in before_state[index].items():
                    if key == "previous_update":
                        assert not torch.count_nonzero(actual_state[key]).item()
                    else:
                        assert_exact(actual_state[key], value, path=path + ".skipped." + key)


def check_native(kind, device):
    if kind == "lion":
        return
    for dtype in TOLERANCES:
        parameters, expected = _parameters(device, dtype), _parameters(device, dtype)
        optimizer = _optimizer(kind, _groups(parameters), 0.0, triton=torch.device(device).type == "cuda")
        native = (torch.optim.AdamW if kind == "adamw" else torch.optim.Adam)(
            _groups(expected), lr=0.017, betas=(0.6, 0.8), eps=0.031, weight_decay=0.13,
            foreach=False, fused=False)
        for call in range(7):
            gradients = _gradients(parameters, call)
            _assign(parameters, gradients, call)
            _assign(expected, gradients, call)
            # Native optimizers don't themselves skip frozen attached gradients.
            for parameter in expected:
                if not parameter.requires_grad:
                    parameter.grad = None
            _schedule(optimizer, call)
            _schedule(native, call)
            optimizer.step()
            native.step()
            for left, right in zip(parameters, expected):
                _close(left, right, dtype, "native.weight")
                ours, theirs = optimizer.state.get(left, {}), native.state.get(right, {})
                assert ours.keys() == theirs.keys()
                for name, value in ours.items():
                    if name == "step":
                        assert value == int(theirs[name])
                    else:
                        _close(value, theirs[name], dtype, "native." + name)


def check_resume_rebind(kind, coefficient, device, dtype):
    triton = torch.device(device).type == "cuda"
    parameters = _parameters(device, dtype)
    optimizer = _optimizer(kind, _groups(parameters), coefficient, triton=triton)
    calls = [-1] + list(range(7))
    snapshots = [(copy.deepcopy(parameters), copy.deepcopy(optimizer.state_dict()))]
    for call in calls:
        _assign(parameters, _gradients(parameters, call), call)
        _schedule(optimizer, call)
        optimizer.step()
        snapshots.append((copy.deepcopy(parameters), copy.deepcopy(optimizer.state_dict())))
    for split in (0, 1, 2, 4, 6):
        weights, state = snapshots[split]
        buffer = io.BytesIO()
        torch.save({"weights": [p.detach().clone() for p in weights], "optimizer": state}, buffer)
        buffer.seek(0)
        saved = torch.load(buffer, map_location=device, weights_only=True)
        resumed = [nn.Parameter(value, requires_grad=p.requires_grad)
                   for value, p in zip(saved["weights"], weights)]
        # Default eager constructor must restore the saved CUDA backend flag.
        restored = _optimizer(kind, _groups(resumed), 0.37, triton=False, lr=0.9)
        restored.load_state_dict(saved["optimizer"])
        assert_exact(restored.state_dict(), state, path="resume.loaded")
        old_storage = []
        for offset, call in enumerate(calls[split:], split):
            if offset % 2 == 0:
                for parameter in resumed:
                    previous = parameter.detach()
                    parameter.data = previous.clone()
                    if parameter.numel():
                        assert parameter.data_ptr() != previous.data_ptr()
                    old_storage.append((previous, previous.clone()))
            _assign(resumed, _gradients(resumed, call), call)
            _schedule(restored, call)
            restored.step()
            expected_weights, expected_state = snapshots[offset + 1]
            assert_exact(resumed, expected_weights, path=f"resume.{kind}.{split}.{call}.weights")
            assert_exact(restored.state_dict(), expected_state, path="resume.state")
            for previous, unchanged in old_storage:
                assert_exact(previous, unchanged, path="rebind.old_storage_unchanged")


def check_closure_autocast(kind, device, dtype):
    parameters, expected = _parameters(device, dtype), _parameters(device, dtype)
    triton = torch.device(device).type == "cuda"
    optimizer = _optimizer(kind, parameters, 0.35, triton=triton)
    direct = _optimizer(kind, expected, 0.35, triton=triton)
    autocast_dtype = torch.float16 if triton else torch.bfloat16
    for call in range(3):
        returned = []

        def closure():
            assert torch.is_grad_enabled(), "closure disabled gradients"
            optimizer.zero_grad(set_to_none=True)
            loss = sum(p.square().sum() * (call + 1) for p in parameters if p.requires_grad)
            loss.backward()
            returned.append(loss)
            return loss

        with torch.autocast(torch.device(device).type, dtype=autocast_dtype):
            result = optimizer.step(closure)
        assert len(returned) == 1 and result is returned[0]
        direct.zero_grad(set_to_none=True)
        loss = sum(p.square().sum() * (call + 1) for p in expected if p.requires_grad)
        loss.backward()
        direct.step()
        assert_exact(parameters, expected, path="closure.autocast.weights")
        assert_exact(optimizer.state_dict(), direct.state_dict(), path="closure.autocast.state")


def _reject(action, label):
    try:
        action()
    except (ValueError, TypeError, RuntimeError, ImportError):
        return
    raise AssertionError(f"accepted invalid {label}")


def _atomic_reject(optimizer, label):
    parameters = [p for group in optimizer.param_groups for p in group["params"]]
    weights = [p.detach().clone() for p in parameters]
    def gradient_snapshot():
        result = []
        for parameter in parameters:
            gradient = parameter.grad
            if gradient is not None and gradient.layout == torch.sparse_coo:
                gradient = gradient.coalesce()
                result.append((gradient.indices().clone(), gradient.values().clone(), tuple(gradient.shape)))
            else:
                result.append(None if gradient is None else gradient.clone())
        return result

    gradients = gradient_snapshot()
    state = copy.deepcopy(optimizer.state_dict())
    _reject(optimizer.step, label)
    assert_exact(parameters, weights, path=label + ".atomic_weights")
    assert_exact(gradient_snapshot(), gradients, path=label + ".atomic_gradients")
    assert_exact(optimizer.state_dict(), state, path=label + ".atomic_state_and_counters")


def check_rejections(kind, device):
    for flag in (1, 0, "true", None):
        _reject(lambda: _optimizer(kind, [nn.Parameter(torch.ones(2))], 1.0, triton=flag),
                f"{kind} nonboolean use_triton={flag!r}")
        _reject(lambda: _optimizer(kind, [{"params": [nn.Parameter(torch.ones(2))],
                                         "use_triton": flag}], 1.0), "group backend flag")
    # True on CPU must never silently dispatch eager, even without gradients.
    _reject(lambda: _optimizer(kind, [nn.Parameter(torch.ones(2))], 1.0, triton=True).step(),
            kind + " CPU Triton")
    cpu = nn.Parameter(torch.ones(2))
    optimizer = _optimizer(kind, [cpu], 1.0)
    cpu.grad = torch.ones_like(cpu)
    optimizer.param_groups[0]["use_triton"] = True
    _atomic_reject(optimizer, kind + ".live_cpu_backend")
    if torch.device(device).type != "cuda":
        return
    for dtype in (torch.float16, torch.bfloat16, torch.complex64):
        _reject(lambda: _optimizer(kind, [nn.Parameter(torch.ones(3, device=device, dtype=dtype))],
                                   1.0, triton=True), "unsupported Triton dtype")
    noncontiguous = nn.Parameter(torch.ones(3, 2, device=device).t())
    assert not noncontiguous.is_contiguous()
    _reject(lambda: _optimizer(kind, [noncontiguous], 1.0, triton=True), "noncontiguous parameter")
    failures = ["parameter", "gradient", "exp_avg", "previous_update", "cpu_parameter",
                "state_dtype", "state_device", "sparse_gradient"]
    if kind != "lion":
        failures.append("exp_avg_sq")
    for warmed in (False, True):
        for failure in failures:
            parameters = [nn.Parameter(torch.ones(2, 3, device=device)) for _ in range(2)]
            optimizer = _optimizer(kind, [{"params": [p]} for p in parameters], 1.0, triton=True)
            for parameter in parameters:
                parameter.grad = torch.ones_like(parameter)
            if warmed or failure in ("exp_avg", "exp_avg_sq", "previous_update", "state_dtype", "state_device"):
                optimizer.step()
            late = parameters[-1]
            bad = torch.ones(3, 2, device=device).t()
            assert not bad.is_contiguous()
            if failure == "parameter":
                late.data = bad
            elif failure == "gradient":
                late.grad = bad
            elif failure == "cpu_parameter":
                late.grad = None
                late.data = late.detach().cpu()
                late.grad = torch.ones_like(late)
            elif failure == "state_dtype":
                optimizer.state[late]["exp_avg"] = optimizer.state[late]["exp_avg"].to(torch.bfloat16)
            elif failure == "state_device":
                optimizer.state[late]["exp_avg"] = optimizer.state[late]["exp_avg"].cpu()
            elif failure == "sparse_gradient":
                late.grad = torch.sparse_coo_tensor([[0, 1], [1, 2]], [0.1, -0.2], late.shape,
                                                   device=device, dtype=late.dtype)
            else:
                optimizer.state[late][failure] = bad
            _atomic_reject(optimizer, f"{kind}.{failure}.warmed{warmed}")
    # Eager remains intentionally permissive for noncontiguous dense tensors.
    eager = _optimizer(kind, [noncontiguous], 1.0)
    noncontiguous.grad = torch.ones_like(noncontiguous)
    eager.step()


def check_old_checkpoints(kind, device):
    for active in (False, True):
        parameters = _parameters(device, torch.float64)
        original = _optimizer(kind, _groups(parameters), 1.0)
        if active:
            _assign(parameters, _gradients(parameters, 0), 0)
            original.step()
        state = copy.deepcopy(original.state_dict())
        for group in state["param_groups"]:
            del group["use_triton"]
        untouched = copy.deepcopy(state)
        resumed = [nn.Parameter(p.detach().clone(), requires_grad=p.requires_grad) for p in parameters]
        restored = _optimizer(kind, _groups(resumed), 0.35, triton=torch.device(device).type == "cuda")
        restored.load_state_dict(state)
        assert all(group["use_triton"] is False for group in restored.param_groups), "old checkpoint didn't select eager"
        assert_exact(state, untouched, path="old_checkpoint.input_unchanged")
        for call in (1, 2, 3):
            gradients = _gradients(parameters, call)
            _assign(parameters, gradients, call)
            _assign(resumed, gradients, call)
            _schedule(original, call)
            _schedule(restored, call)
            original.step()
            restored.step()
            assert_exact(resumed, parameters, path="old_checkpoint.weights")
            assert_exact(restored.state_dict(), original.state_dict(), path="old_checkpoint.state")


def check_optional_imports():
    # Fresh interpreter: a previously imported JIT module cannot mask eagerness.
    code = r'''
import importlib.abc
import sys
import torch
class NoOptionalBackend(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "ramtorch._delayed_optim_triton":
            raise AssertionError("eager path imported optional backend: " + fullname)
        if fullname == "triton" or fullname.startswith("triton."):
            # PyTorch itself probes this optional package; emulate its absence
            # rather than forbidding a correctly caught availability check.
            raise ModuleNotFoundError("optional Triton package unavailable", name=fullname)
sys.meta_path.insert(0, NoOptionalBackend())
from ramtorch import AdamEF, Lion
for cls in (AdamEF, Lion):
    p = torch.nn.Parameter(torch.ones(3))
    optimizer = cls([p], use_triton=False)
    p.grad = torch.ones_like(p)
    optimizer.step()
    saved = optimizer.state_dict()
    del saved["param_groups"][0]["use_triton"]
    restored = cls([p], use_triton=False)
    restored.load_state_dict(saved)
    restored.step()
assert "ramtorch._delayed_optim_triton" not in sys.modules
print("PASS fresh-process eager import without optional backend")
'''
    result = subprocess.run([sys.executable, "-c", code], text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    print(result.stdout.strip())


def check_factory_routing(device):
    from pipedream_2bw_optimizer_check import optimizer_factory

    for kind in KINDS:
        cpu = [nn.Parameter(torch.ones(3))]
        optimizer = optimizer_factory(kind, 1.0, use_triton=True)(iter(cpu))
        assert all(group["use_triton"] is False for group in optimizer.param_groups)
        if torch.device(device).type == "cuda":
            gpu = [nn.Parameter(torch.ones(3, device=device))]
            optimizer = optimizer_factory(kind, 1.0, use_triton=True)(iter(gpu))
            assert all(group["use_triton"] is True for group in optimizer.param_groups)
            mixed = optimizer_factory(kind, 1.0, use_triton=True)([{"params": cpu}, {"params": gpu}])
            assert [group["use_triton"] for group in mixed.param_groups] == [False, True]


def check_triton_units(device, kinds=KINDS):
    device = torch.device(device)
    triton = device.type == "cuda"
    check_factory_routing(device)
    if not triton:
        check_optional_imports()
    for kind in kinds:
        check_rejections(kind, device)
        check_old_checkpoints(kind, device)
        if triton:
            from ramtorch import _delayed_optim_triton as backend

            calls = patch.object(backend, "step_parameter", wraps=backend.step_parameter)
        else:
            calls = nullcontext(None)
        with calls as launched:
            for dtype in TOLERANCES:
                for coefficient in (0.0, 0.35, 1.0):
                    for empty_start in (False, True):
                        check_equations(kind, coefficient, device, dtype, empty_start=empty_start)
                    check_resume_rebind(kind, coefficient, device, dtype)
                check_equations(kind, 0.35, device, dtype, empty_start=True, mixed_groups=True)
                check_closure_autocast(kind, device, dtype)
            check_native(kind, device)
            if triton:
                torch.cuda.synchronize(device)  # Correctness observer only, not timing.
                assert launched.call_count > 0, kind + ": CUDA backend never launched"
                assert any(call.args[0].numel() > 8192 for call in launched.call_args_list), "large kernel untested"
                assert {call.kwargs["adam"] for call in launched.call_args_list} == {kind != "lion"}
                assert all(call.args[0].device.type == "cuda" and call.args[2]["use_triton"] is True
                           for call in launched.call_args_list), "backend received eager/CPU group"
        print(f"PASS {kind}: scalar oracle/eager/native tolerance, EF 0/.35/1, LR/decay schedules, "
              f"lazy budgets, empty/scalar/odd/large, skipped/frozen/late, exact resume/rebind/autocast: {device}")
    if not triton:
        print("SKIP CUDA launch/layout checks: CPU validation does not exercise Triton")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="cpu for API/eager checks; cuda:N for real Triton checks")
    parser.add_argument("--optimizers", nargs="+", choices=KINDS, default=KINDS)
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("only CPU and CUDA are supported")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable; no eager fallback")
    configure_determinism()
    check_triton_units(device, kinds=args.optimizers)
    print("ALL REQUESTED TRITON OPTIMIZER CHECKS PASSED")


if __name__ == "__main__":
    main()
