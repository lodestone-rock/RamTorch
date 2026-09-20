"""Standalone, deliberately unscheduled PipeDream-2BW stale-gradient oracle.

Number weights by completed updates: w[0] is the initial model. Group g evaluates
all its microbatches at w[max(g - 1, 0)], accumulates their gradients in ascending
order, then multiplies each accumulated gradient by 1/m exactly once. The
optimizer applies that gradient to w[g], using its CURRENT state, producing
w[g + 1]. In particular, groups 0 and 1 both evaluate w[0].

There are two ordinary models: a latest-weight model owned by the optimizer and
an evaluation model loaded from copied historical state dictionaries. The latest
model can live on CPU (including CPU-fused AdamW) while evaluation stays on CUDA.
No pipeline, functional model API, schedule, optimizer-model weight swapping, or
optimizer-state rewind is involved. The intended models have stateless forwards
(no running-statistic buffers or stochastic layers).

All returned tensors are detached CPU copies. UpdateResult.weights and
UpdateResult.optimizer_state are POST-update; gradients are the scaled,
PRE-optimizer gradients (before optimizer weight decay). Unused/frozen parameters
retain None gradients and empty optimizer state, rather than invented zeros.
Exact repeatability means the same device, dtype, and PyTorch environment, not
bitwise equality between CPU, CUDA, or different PyTorch versions.

Examples:
    env/bin/python examples/pipedream_2bw_reference.py --device cpu --self-test
    env/bin/python examples/pipedream_2bw_reference.py --optimizer adamw --bf16
    env/bin/python examples/pipedream_2bw_reference.py --device cuda:0 --updates 8
    env/bin/python examples/pipedream_2bw_reference.py --device cuda:0 --optimizer adamw --optimizer-device cpu --fused
"""

from __future__ import annotations

import os

# These must precede the first torch import in the process. When importing this
# module into an already-running application, configure its environment first.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import copy
import math
from collections import deque
from dataclasses import dataclass, fields, is_dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

Device = Union[str, torch.device]
Batch = Tuple[Tensor, Tensor]
NamedTensors = Dict[str, Tensor]
NamedGradients = Dict[str, Optional[Tensor]]
NamedOptimizerState = Dict[str, Dict[str, Any]]
OptimizerFactory = Callable[[Iterable[nn.Parameter]], torch.optim.Optimizer]


def configure_determinism(seed: int = 0) -> None:
    """Set process-wide deterministic controls; do not silently allow fallback."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        # If an importing application already started interop work, fail clearly
        # instead of claiming that the requested configuration was established.
        torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(seed)


class ResidualMLPBlock(nn.Module):
    """Small residual around Linear(d, 4d), GELU, Linear(4d, d)."""

    def __init__(self, dim: int, residual_scale: float = 0.1):
        super().__init__()
        self.up = nn.Linear(dim, 4 * dim)
        self.activation = nn.GELU()
        self.down = nn.Linear(4 * dim, dim)
        self.residual_scale = residual_scale

    def forward(self, x: Tensor) -> Tensor:
        return x + self.down(self.activation(self.up(x))) * self.residual_scale


def make_model(
    dim: int = 16,
    layers: int = 4,
    *,
    device: Device = "cpu",
    seed: int = 0,
    residual_scale: float = 0.1,
    dtype: torch.dtype = torch.float32,
) -> nn.Module:
    """Build on CPU using a private RNG scope, then copy to the requested device."""
    if dim < 1 or layers < 1:
        raise ValueError("dim and layers must be positive")
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        model = nn.Sequential(
            *(ResidualMLPBlock(dim, residual_scale) for _ in range(layers))
        )
    return model.to(device=device, dtype=dtype)


def make_batches(
    updates: int = 5,
    microbatches: int = 4,
    dim: int = 16,
    *,
    batch_size: int = 2,
    device: Device = "cpu",
    seed: int = 1,
    dtype: torch.dtype = torch.float32,
) -> List[List[Batch]]:
    """Return batches[group][microbatch] = (input, target), in execution order.

    batch_size is the number of samples PER microbatch. CPU generation with a
    private generator gives identical starting data regardless of destination.
    """
    if min(updates, microbatches, dim, batch_size) < 1:
        raise ValueError("updates, microbatches, dim, and batch_size must be positive")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    groups = []
    for _ in range(updates):
        group = []
        for _ in range(microbatches):
            x = torch.randn(batch_size, dim, generator=generator, dtype=dtype)
            y = torch.randn(batch_size, dim, generator=generator, dtype=dtype)
            group.append((x.to(device), y.to(device)))
        groups.append(group)
    return groups


def optimizer_factory(
    name: str = "sgd",
    *,
    lr: float = 0.01,
    momentum: float = 0.9,
    weight_decay: float = 0.01,
    betas: Tuple[float, float] = (0.9, 0.999),
    eps: float = 1e-8,
    fused: bool = False,
) -> OptimizerFactory:
    """Return a fresh optimizer, disabling foreach and opting into fused only on request.

    SGD uses momentum without dampening or Nesterov. AdamW uses decoupled weight
    decay and ordinary bias-corrected moments. momentum is SGD-only; betas/eps
    are AdamW-only. Optimizer hyperparameter validation is left to PyTorch.
    """
    name = name.lower()
    if name not in ("sgd", "adamw"):
        raise ValueError("optimizer must be 'sgd' or 'adamw'")

    def build(parameters: Iterable[nn.Parameter]) -> torch.optim.Optimizer:
        if name == "sgd":
            return torch.optim.SGD(
                parameters, lr=lr, momentum=momentum, weight_decay=weight_decay,
                foreach=False, fused=fused,
            )
        return torch.optim.AdamW(
            parameters, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
            foreach=False, fused=fused,
        )

    return build


@dataclass
class UpdateResult:
    group: int
    eval_version: int
    losses: Tuple[Tensor, ...]
    gradients: NamedGradients
    weights: NamedTensors
    optimizer_state: NamedOptimizerState


@dataclass
class ReferenceResult:
    initial_weights: NamedTensors
    updates: List[UpdateResult]

    @property
    def weight_history(self) -> List[NamedTensors]:
        """w[0], ..., w[number_of_updates]; entries are the recorded snapshots."""
        return [self.initial_weights] + [update.weights for update in self.updates]


def _snapshot(value: Any) -> Any:
    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _snapshot(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot(item) for item in value)
    return copy.deepcopy(value)


def _weights(model: nn.Module) -> NamedTensors:
    return {name: _snapshot(parameter) for name, parameter in model.named_parameters()}


def run_reference(
    model: nn.Module,
    batches: Sequence[Sequence[Batch]],
    make_optimizer: OptimizerFactory,
    *,
    device: Optional[Device] = None,
    optimizer_device: Optional[Device] = None,
    loss_fn: Optional[Callable[[Tensor, Tensor], Tensor]] = None,
    autocast_bf16: bool = False,
    seed: int = 0,
    observer: Optional[Callable[[UpdateResult], None]] = None,
    retain_updates: bool = True,
) -> ReferenceResult:
    """Run the copied-history oracle without modifying model or batches.

    loss_fn must return one scalar microbatch-mean loss; default is MSE. Each
    nonempty group is equally weighted by its own microbatch count. BF16 uses
    autocast only, with no GradScaler, retaining the model's parameter dtype.
    Pass optimizer_factory(...) as make_optimizer, not an existing optimizer.
    With retain_updates=False, consume observer(update) instead of retaining the
    trajectory. Only the two newest copied historical state dictionaries remain;
    this is still ordinary model loading, not the runtime's aliasing/bank logic.
    optimizer_device=None keeps the optimizer on the evaluation device (the
    original behavior). With optimizer_device='cpu', the latest model and ALL
    optimizer math/state live on CPU, while evaluation still runs on device.
    Gradients are scaled once on the evaluation device, then copied to CPU.
    Compare a CPU-fused runtime with optimizer_factory('adamw', fused=True),
    not a CUDA/nonfused optimizer whose arithmetic may round differently.
    """
    configure_determinism(seed)
    if not batches or any(not group for group in batches):
        raise ValueError("provide at least one update and one microbatch per update")
    first_parameter = next(model.parameters(), None)
    if first_parameter is None:
        raise ValueError("model must have parameters")
    device = torch.device(device) if device is not None else first_parameter.device
    if device.type not in ("cpu", "cuda"):
        raise ValueError("the reference supports CPU and CUDA devices")
    loss_fn = F.mse_loss if loss_fn is None else loss_fn

    optimizer_device = device if optimizer_device is None else torch.device(optimizer_device)
    if optimizer_device != device and optimizer_device.type != "cpu":
        raise ValueError("optimizer_device must be CPU or the evaluation device")
    latest = copy.deepcopy(model).to(optimizer_device)
    evaluation = copy.deepcopy(model).to(device)
    optimizer = make_optimizer(latest.parameters())
    latest_parameters = dict(latest.named_parameters())
    evaluation_parameters = dict(evaluation.named_parameters())
    # Full state copies load into the evaluator, never into the optimizer model.
    history = deque([_snapshot(latest.state_dict())], maxlen=2)
    result = ReferenceResult(initial_weights=_weights(latest), updates=[])

    for group_index, group in enumerate(batches):
        eval_version = max(group_index - 1, 0)
        # Before group g, the oldest retained snapshot is w[max(g-1, 0)].
        evaluation.load_state_dict(history[0])
        evaluation.zero_grad(set_to_none=True)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for x, y in group:  # Deliberately ascending; no reordering or tree reduce.
            with torch.autocast(
                device_type=device.type, dtype=torch.bfloat16, enabled=autocast_bf16
            ):
                loss = loss_fn(evaluation(x.to(device)), y.to(device))
            if loss.ndim != 0:
                raise ValueError("loss_fn must return a scalar microbatch-mean loss")
            loss.backward()  # Accumulate unscaled microbatch gradients.
            losses.append(_snapshot(loss))

        scale = 1.0 / len(group)
        for name, parameter in evaluation_parameters.items():
            if parameter.grad is None:
                latest_parameters[name].grad = None
            else:
                parameter.grad.mul_(scale)  # The ONLY gradient mean scaling.
                # copy=True gives the optimizer its own gradient even when both
                # ordinary models live on CPU. No runtime helpers or banks.
                latest_parameters[name].grad = parameter.grad.detach().to(
                    optimizer_device, copy=True
                )
        gradients = {
            name: _snapshot(parameter.grad)
            for name, parameter in latest_parameters.items()
        }
        optimizer.step()  # Mutates w[g], with optimizer state from update g-1.
        update = UpdateResult(
            group=group_index,
            eval_version=eval_version,
            losses=tuple(losses),
            gradients=gradients,
            weights=_weights(latest),
            optimizer_state={
                name: _snapshot(optimizer.state.get(parameter, {}))
                for name, parameter in latest_parameters.items()
            },
        )
        if observer is not None:
            observer(update)
        if retain_updates:
            result.updates.append(update)
        del update
        history.append(_snapshot(latest.state_dict()))
    return result


def assert_exact(actual: Any, expected: Any, *, path: str = "result") -> None:
    """Assert zero-tolerance equality recursively, reporting the failing path.

    Supports result dataclasses, named dictionaries, sequences, tensors, and
    scalars/None. Tensor dtype and shape must match; devices may differ. NaNs
    never count as equal. No allclose tolerances hide divergent trajectories.
    """
    if isinstance(actual, Tensor) or isinstance(expected, Tensor):
        if not isinstance(actual, Tensor) or not isinstance(expected, Tensor):
            raise AssertionError(f"{path}: tensor/non-tensor mismatch")
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise AssertionError(f"{path}: nonfinite tensor")
        torch.testing.assert_close(
            actual, expected, rtol=0, atol=0, check_device=False,
            msg=lambda message: f"{path}: {message}",
        )
        a = actual.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        b = expected.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
        if not torch.equal(a, b):
            raise AssertionError(f"{path}: tensor bytes differ (including signed zero)")
        return
    if type(actual) is not type(expected):
        raise AssertionError(f"{path}: type mismatch {type(actual)} != {type(expected)}")
    if is_dataclass(actual):
        for field in fields(actual):
            assert_exact(getattr(actual, field.name), getattr(expected, field.name),
                         path=f"{path}.{field.name}")
    elif isinstance(actual, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"{path}: different keys {actual.keys()} != {expected.keys()}")
        for key in actual:
            assert_exact(actual[key], expected[key], path=f"{path}[{key!r}]")
    elif isinstance(actual, (tuple, list)):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: different lengths {len(actual)} != {len(expected)}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            assert_exact(left, right, path=f"{path}[{index}]")
    elif actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


# Self-tests intentionally use a separate, Python-scalar recurrence: no autograd,
# model forward, optimizer, or oracle helper calculates the expected trajectory.
class _ScalarModel(nn.Module):
    def __init__(self, dtype: torch.dtype = torch.float64):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0, dtype=dtype))
        self.unused = nn.Parameter(torch.tensor(7.0, dtype=dtype))
        self.frozen = nn.Parameter(torch.tensor(3.0, dtype=dtype), requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.weight * x


def _manual_scalar_recurrence(
    pairs: Sequence[Sequence[Tuple[float, float]]],
    name: str,
    *,
    momentum: float,
    weight_decay: float,
    fresh_grad: bool = False,
    wrong_base: bool = False,
) -> List[Dict[str, Any]]:
    weights = [1.0]
    buffer = first_moment = second_moment = 0.0
    records = []
    lr, beta1, beta2, eps = 0.125, 0.5, 0.75, 0.125
    for group_index, group in enumerate(pairs):
        version = group_index if fresh_grad else max(group_index - 1, 0)
        evaluation_weight = weights[version]
        gradient = 0.0
        for x, y in group:
            gradient += (evaluation_weight * x - y) * x
        gradient *= 1.0 / len(group)
        base = evaluation_weight if wrong_base else weights[-1]
        state = {}
        if name == "sgd":
            effective_gradient = gradient + weight_decay * base
            if momentum:
                buffer = momentum * buffer + effective_gradient
                effective_gradient = buffer
                state = {"momentum_buffer": buffer}
            updated = base - lr * effective_gradient
        else:
            step = group_index + 1
            first_moment = beta1 * first_moment + (1.0 - beta1) * gradient
            second_moment = beta2 * second_moment + (1.0 - beta2) * gradient ** 2
            denominator = math.sqrt(second_moment) / math.sqrt(1.0 - beta2 ** step) + eps
            updated = base * (1.0 - lr * weight_decay)
            updated -= (lr / (1.0 - beta1 ** step)) * first_moment / denominator
            state = {"step": float(step), "exp_avg": first_moment,
                     "exp_avg_sq": second_moment}
        weights.append(updated)
        records.append({"gradient": gradient, "weight": updated, "state": state})
    return records


def _must_differ(actual: Any, wrong: Any, description: str) -> None:
    try:
        assert_exact(actual, wrong)
    except AssertionError:
        return
    raise AssertionError(f"negative control did not fail: {description}")


def run_self_tests(device: Device = "cpu", *, autocast_bf16: bool = False) -> List[str]:
    """Raise on failure; return descriptions of successful independent checks."""
    configure_determinism()
    passed = []
    for name in ("sgd", "adamw"):
        model = make_model(dim=4, layers=2, device=device, seed=13)
        batches = make_batches(updates=5, microbatches=3, dim=4, seed=17)
        original_model, original_batches = _snapshot(model.state_dict()), _snapshot(batches)
        factory = optimizer_factory(name)
        first = run_reference(model, batches, factory, autocast_bf16=autocast_bf16)
        second = run_reference(model, batches, factory, autocast_bf16=autocast_bf16)
        assert_exact(first, second)
        streamed_groups = []
        def compare_stream(update):
            assert_exact(update, first.updates[update.group], path="streamed")
            streamed_groups.append(update.group)
        streamed = run_reference(model, batches, factory, autocast_bf16=autocast_bf16,
                                 observer=compare_stream, retain_updates=False)
        assert_exact(streamed.initial_weights, first.initial_weights)
        assert_exact(streamed.updates, [])
        assert_exact(streamed_groups, list(range(len(batches))))
        assert_exact(dict(model.state_dict()), original_model, path="caller.model")
        assert_exact(batches, original_batches, path="caller.batches")
        assert_exact([u.eval_version for u in first.updates], [0, 0, 1, 2, 3])
        # Independently recreate fixtures after disturbing the global RNG.
        torch.randn(19)
        assert_exact(_weights(make_model(4, 2, device=device, seed=13)), original_model)
        assert_exact(make_batches(5, 3, 4, seed=17), original_batches)
        passed.append(f"{name}: exact dense repeatability and immutable deterministic fixtures")

    # Constant first two groups make the bootstrap rule directly observable.
    pairs = [[(2.0, 0.0), (1.0, 0.5)] for _ in range(5)]
    scalar_batches = [[
        (torch.tensor([x], dtype=torch.float64), torch.tensor([y], dtype=torch.float64))
        for x, y in group
    ] for group in pairs]
    for name, momentum, decay in (("sgd", 0.0, 0.0), ("sgd", 0.5, 0.125),
                                  ("adamw", 0.0, 0.125)):
        factory = optimizer_factory(name, lr=0.125, momentum=momentum,
                                    weight_decay=decay, betas=(0.5, 0.75), eps=0.125)
        result = run_reference(
            _ScalarModel(), scalar_batches, factory, device=device,
            loss_fn=lambda output, target: 0.5 * (output - target).square().mean(),
        )
        manual = _manual_scalar_recurrence(pairs, name, momentum=momentum, weight_decay=decay)
        for update, expected in zip(result.updates, manual):
            actual = {"gradient": update.gradients["weight"], "weight": update.weights["weight"],
                      "state": update.optimizer_state["weight"]}
            # Dyadic SGD arithmetic is exact. AdamW's independently expressed
            # sqrt/division operations can round differently, so use a tight
            # FP64 tolerance ONLY for this analytic check, never repeatability.
            if name == "sgd":
                expected_tensors = _snapshot(expected)
                expected_tensors["gradient"] = torch.tensor(expected["gradient"], dtype=torch.float64)
                expected_tensors["weight"] = torch.tensor(expected["weight"], dtype=torch.float64)
                expected_tensors["state"] = {
                    key: torch.tensor(value, dtype=torch.float64)
                    for key, value in expected["state"].items()
                }
                assert_exact(actual, expected_tensors, path=f"scalar.sgd[{update.group}]")
            else:
                assert_exact(set(actual["state"]), set(expected["state"]))
                for key in ("gradient", "weight"):
                    torch.testing.assert_close(actual[key].item(), expected[key], rtol=1e-14, atol=1e-14)
                for key, value in expected["state"].items():
                    torch.testing.assert_close(actual["state"][key].item(), value, rtol=1e-14, atol=1e-14)
                assert_exact(actual["state"]["step"].item(), float(update.group + 1))
            for unused in ("unused", "frozen"):
                assert_exact(update.gradients[unused], None)
                assert_exact(update.optimizer_state[unused], {})
                assert_exact(update.weights[unused], result.initial_weights[unused])
        assert_exact(result.updates[0].gradients, result.updates[1].gradients)
        observed = [update.weights["weight"].item() for update in result.updates]
        for flag in ("fresh_grad", "wrong_base"):
            wrong = _manual_scalar_recurrence(
                pairs, name, momentum=momentum, weight_decay=decay, **{flag: True}
            )
            wrong_weights = [item["weight"] for item in wrong]
            _must_differ(observed, wrong_weights, f"{name}: {flag}")
            if not any(abs(left - right) > 1e-6 for left, right in zip(observed, wrong_weights)):
                raise AssertionError(f"{name}: {flag} differs only by rounding noise")
        passed.append(f"{name} momentum={momentum}: analytic stale recurrence, optimizer state, "
                      "None gradients, and fresh-gradient/wrong-base negative controls")

    # A cancellation witness distinguishes sum-then-scale from both early loss
    # scaling and a reordered reduction. Each microbatch's gradient is its y.
    coefficients = (1e8, 5.0, -1e8)
    cancellation_batches = [[
        (torch.ones(1), torch.tensor([coefficient])) for coefficient in coefficients
    ]]
    cancellation = run_reference(
        _ScalarModel(dtype=torch.float32), cancellation_batches,
        optimizer_factory("sgd", momentum=0, weight_decay=0), device=device,
        loss_fn=lambda output, target: (output * target).mean(),
    )
    expected_gradient = torch.tensor(8.0).mul_(1.0 / 3)
    assert_exact(cancellation.updates[0].gradients["weight"], expected_gradient)
    early_scaled = torch.tensor(0.0)
    for coefficient in coefficients:
        early_scaled.add_(torch.tensor(coefficient).mul_(1.0 / 3))
    _must_differ(expected_gradient, early_scaled, "scale each microbatch early")
    _must_differ(expected_gradient, torch.tensor(5.0).mul_(1.0 / 3), "reorder microbatches")
    _must_differ({"unused": None}, {"unused": torch.tensor(0.0)}, "replace None with zero")
    passed.append("ascending accumulation and single final scaling: cancellation/None negative controls")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="cpu", help="cpu or a CUDA device such as cuda:0")
    parser.add_argument("--updates", type=int, default=5)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--dim", type=int, default=16)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2, help="samples per microbatch")
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--optimizer-device", choices=("cpu",), default=None,
                        help="keep latest weights, optimizer state and optimizer math on CPU")
    parser.add_argument("--fused", action="store_true", help="use the requested optimizer's fused math")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--stream", action="store_true",
                        help="print updates as they finish without retaining tensor snapshots")
    parser.add_argument("--bf16", action="store_true", help="enable BF16 autocast, keeping FP32 weights")
    parser.add_argument("--self-test", action="store_true", help="run independent self-tests and exit")
    args = parser.parse_args()
    configure_determinism()
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        parser.error("--device must be cpu or a CUDA device such as cuda:0")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is unavailable")
    if min(args.updates, args.microbatches, args.dim, args.layers, args.batch_size) < 1:
        parser.error("--updates, --microbatches, --dim, --layers, and --batch-size must be positive")
    if args.self_test:
        checks = run_self_tests(device, autocast_bf16=args.bf16)
        for check in checks:
            print(f"PASS {check}")
        print(f"All {len(checks)} self-test groups passed on {device} (BF16={args.bf16}).")
        return 0
    def print_update(update):
        loss = torch.stack(update.losses).mean().item()
        print(f"group={update.group} evaluate=w[{update.eval_version}] "
              f"apply=w[{update.group}]->w[{update.group + 1}] mean_loss={loss:.9g}", flush=True)
    print(f"Independent 2BW reference: device={device}, optimizer={args.optimizer}, "
          f"optimizer_device={args.optimizer_device or device}, fused={args.fused}, "
          f"updates={args.updates}, microbatches={args.microbatches}, BF16={args.bf16}")
    result = run_reference(
        make_model(args.dim, args.layers, device=device),
        make_batches(args.updates, args.microbatches, args.dim, batch_size=args.batch_size),
        optimizer_factory(args.optimizer, lr=args.lr, fused=args.fused),
        optimizer_device=args.optimizer_device, autocast_bf16=args.bf16,
        observer=print_update if args.stream else None, retain_updates=not args.stream,
    )
    if args.stream:
        print("Streamed all updates; historical tensor snapshots were not retained.")
    else:
        for update in result.updates:
            print_update(update)
        print("Recorded named gradients, post-update weights, and current optimizer state for every update.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
