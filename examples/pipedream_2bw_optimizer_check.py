"""Independent AdamEF/Lion equations and exact PipeDream-2BW schedule checks.

Run from the repository root, using a Python environment with working PyTorch::

    PYTHONPATH=. python examples/pipedream_2bw_optimizer_check.py
    PYTHONPATH=. python examples/pipedream_2bw_optimizer_check.py --bf16
    PYTHONPATH=. python examples/pipedream_2bw_optimizer_check.py --devices cuda:0,cuda:1 --bf16

--bf16 ADDS autocast checks to the FP32 bank/pipeline suite; optimizer parameters
and moments remain FP32. --unit-only and --pipeline-only isolate the two suites.
No production optimizer helper is used to compute expected optimizer updates:
Python scalar equations check Float64 weights, moments and full update history;
plain AdamEF is also compared with native Adam/AdamW. The copied-history
reference uses the same factory ONLY for independent bank/schedule comparisons.
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


KINDS = ("adam", "adamw", "lion")
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
    else:
        raise ValueError(f"unknown optimizer {kind!r}")
    options.update(overrides)
    return lambda parameters: cls(parameters, **options)


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


def check_units(device):
    check_native_adam(device)
    for kind in KINDS:
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


def _check_owners(bank, coefficient):
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


def check_banks(kind, coefficient, device, optimizer_device, bf16, updates):
    model = _model(dim=8, layers=2)
    groups = make_batches(updates=updates, microbatches=1, dim=8, seed=43)
    factory = optimizer_factory(kind, coefficient)
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
            _check_owners(bank, coefficient)
            assert_exact(records[group], _expected_snapshot(expected.updates[group], bank.named),
                         path=f"banks.{kind}.c{coefficient}.group{group}")
    finally:
        bank.release()
    print(f"BIT-EXACT banks: {kind} c={coefficient} {device} optimizer_device={optimizer_device} bf16={bf16}")


def _loader(groups):
    for group in groups:
        yield torch.cat([inputs for inputs, _ in group]), torch.cat([targets for _, targets in group])


def check_pipeline(kind, coefficient, devices, optimizer_device, bf16, updates):
    depth, microbatches = len(devices), max(3, len(devices))
    model = _model(dim=8, layers=2 * depth)
    groups = make_batches(updates=updates, microbatches=microbatches, dim=8, seed=47)
    factory = optimizer_factory(kind, coefficient)
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
                    _check_owners(bank, coefficient)
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
    configure_determinism()
    # Fail clearly if the independently implemented classes are not present yet.
    optimizer_factory("adam", 0.0)
    if not args.pipeline_only:
        for device in dict.fromkeys(["cpu", devices[0]]):
            check_units(device)
    if not args.unit_only:
        placements = (None, "cpu") if types == {"cuda"} else (None,)
        for bf16 in ((False, True) if args.bf16 else (False,)):
            for optimizer_device in placements:
                for kind in KINDS:
                    for coefficient in (0.0, 1.0):
                        check_banks(kind, coefficient, devices[0], optimizer_device, bf16, args.updates)
                        check_pipeline(kind, coefficient, devices, optimizer_device, bf16, args.updates)
    print("ALL REQUESTED OPTIMIZER CHECKS PASSED")


if __name__ == "__main__":
    main()
