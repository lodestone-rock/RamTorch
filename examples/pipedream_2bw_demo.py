r"""Benchmark resident 2BW against synchronous 1B1F or CPU fused AdamW.

Run from the repository root (or install RamTorch first)::

    env/bin/python examples/pipedream_2bw_demo.py \
        --devices cuda:0,cuda:1 --dim 512 --layers 8 --microbatches 8 \
        --updates 16 --batch-size 32 --optimizer adamw --bf16

Each mode gets an independent short warmup, fresh clean-timing models, and
another fresh model for a separate full-run Kineto capture. Inputs are generated
one pinned CPU update group at a time; loading and transfers count toward time.
The modes share initial weights, data, loss, precision, and optimizer settings,
NOT update semantics: 2BW uses one-update-stale gradients, while the synchronous
baseline uses current weights. Their final weights are not expected to match.

With --compare-cpu-optimizer --optimizer adamw, compare the SAME 2BW schedule:
nonfused GPU AdamW versus fused CPU AdamW with pinned CPU masters/gradient buffers,
CPU optimizer state, and two resident GPU weight banks. With --compare-all, also
run synchronous 1F1B with each optimizer placement. The synchronous CPU baseline
uses one GPU weight bank and concurrent per-stage CPU updates AFTER each full
drain; all uploads finish before the next batch begins.
Storage validation hooks run only in the separate profiles, never clean timing.

Outputs are one .json.gz trace per mode, metadata.json, and a deflated ZIP containing
those artifacts. Missing real kernel events on any requested GPU make the
command fail AFTER saving the diagnostic bundle. No correctness observer runs
in any benchmark or profile. Profiling overhead never enters reported speedup.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import gc
import json
import math
import os
import platform
import sys
import time
import zipfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from datetime import datetime, timezone
from pathlib import Path

# The reference configures CUDA/BLAS environment variables BEFORE importing
# torch. Support both direct execution and `python -m examples....`.
if __package__:
    from .pipedream_2bw_reference import (
        configure_determinism, make_model, optimizer_factory,
    )
else:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from pipedream_2bw_reference import (
        configure_determinism, make_model, optimizer_factory,
    )

import torch
from torch import nn
from torch.nn import functional as F

from ramtorch import Pipeline
from ramtorch.pipeline_2bw_trace import TraceCapture, inspect_trace

MODEL_SEED = 0
DATA_SEED = 1
MODES = ("pipedream_2bw", "staggered_1b1f")
CPU_OPTIMIZER_MODES = ("pipedream_2bw", "pipedream_2bw_cpu_fused")
ALL_MODES = (*CPU_OPTIMIZER_MODES, "staggered_1b1f", "staggered_1b1f_cpu_fused")


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--devices", default="cuda:0",
                        help="comma-separated, distinct indexed CUDA devices (default: cuda:0)")
    parser.add_argument("--dim", type=positive_int, default=512)
    parser.add_argument("--layers", type=positive_int, default=8)
    parser.add_argument("--microbatches", type=positive_int, default=8)
    parser.add_argument("--updates", type=positive_int, default=16)
    parser.add_argument("--batch-size", type=positive_int, default=32,
                        help="samples PER microbatch (default: 32)")
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="sgd")
    parser.add_argument("--lr", type=float, default=0.01, help="learning rate (default: 0.01)")
    parser.add_argument("--compare-cpu-optimizer", action="store_true",
                        help="compare GPU nonfused vs CPU fused AdamW on the same 2BW schedule; "
                             "requires --optimizer adamw; skips synchronous baseline")
    parser.add_argument("--compare-all", action="store_true",
                        help="compare 2BW and synchronous 1F1B with GPU and CPU-fused AdamW")
    parser.add_argument("--cpu-threads", type=positive_int, default=4,
                        help="PyTorch intra-op threads for both modes (default: 4)")
    parser.add_argument("--bf16", action="store_true", help="BF16 autocast; FP32 parameters")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("scratchpad/pipedream_2bw/runs/runtime"))
    args = parser.parse_args(argv)
    if not math.isfinite(args.lr) or args.lr < 0:
        parser.error("--lr must be finite and nonnegative")
    if (args.compare_cpu_optimizer or args.compare_all) and args.optimizer != "adamw":
        parser.error("CPU optimizer comparisons require --optimizer adamw")
    if args.compare_cpu_optimizer and args.compare_all:
        parser.error("use either --compare-all or --compare-cpu-optimizer")
    try:
        devices = [torch.device(part.strip()) for part in args.devices.split(",")]
    except (RuntimeError, ValueError) as error:
        parser.error("invalid --devices: {}".format(error))
    if any(device.type != "cuda" or device.index is None for device in devices):
        parser.error("--devices must contain indexed CUDA devices, e.g. cuda:0,cuda:1")
    if len(set(devices)) != len(devices):
        parser.error("--devices must not repeat a GPU")
    if args.layers < len(devices):
        parser.error("--layers must be at least the number of devices")
    if args.microbatches < len(devices):
        parser.error("--microbatches must be at least the pipeline depth for 2BW")
    return args, devices


def group_batches(args, updates, *, pin_memory=True):
    """Yield full update groups lazily, matching the reference's RNG order.

    Each iterator owns its CPU generator, so rebuilding a run exactly replays
    the data, independent of interleaving or global RNG state. Never reuse an
    emitted pinned buffer: an asynchronous consumer may still be copying it.
    Only the current group is allocated here; 2BW bounds its own live window.
    """
    generator = torch.Generator(device="cpu").manual_seed(DATA_SEED)
    shape = (args.microbatches * args.batch_size, args.dim)
    for _ in range(updates):
        x = torch.empty(shape, dtype=torch.float32, device="cpu", pin_memory=pin_memory)
        y = torch.empty_like(x, pin_memory=pin_memory)
        for mb in range(args.microbatches):
            rows = slice(mb * args.batch_size, (mb + 1) * args.batch_size)
            # Alternating x/y draws with per-microbatch shapes also matches
            # make_batches(), unlike two whole-group randn calls.
            torch.randn(args.batch_size, args.dim, generator=generator, out=x[rows])
            torch.randn(args.batch_size, args.dim, generator=generator, out=y[rows])
        yield x, y


def make_pipe(args, devices):
    model = make_model(dim=args.dim, layers=args.layers, seed=MODEL_SEED)
    cuts = [args.layers * stage // len(devices) for stage in range(len(devices) + 1)]
    modules = [nn.Sequential(OrderedDict(
        (str(index), model[index]) for index in range(cuts[stage], cuts[stage + 1])
    )) for stage in range(len(devices))]
    return Pipeline(stage_modules=modules, devices=devices, offload=False,
                    overlap=True, autocast=torch.bfloat16 if args.bf16 else None)


def synchronize(devices):
    for device in devices:
        if device.type == "cuda":
            torch.cuda.synchronize(device)


@contextlib.contextmanager
def baseline_stage_spans(pipe, trace, enabled, group_context):
    """Annotate fresh baseline stages only during profiling, not clean timing.

    Stage's optional Perfetto tracer is not a Kineto record_function. Instance
    wrappers give TraceCapture worker-thread F/B spans without editing Stage or
    inventing GPU events. Pipeline.step joins the workers before restoration.
    """
    originals = []

    def wrap(method, kind, stage):
        @functools.wraps(method)
        def annotated(mb, *args, **kwargs):
            with trace.span(kind, stage=stage, mb=mb, group=group_context["group"]):
                return method(mb, *args, **kwargs)
        return annotated

    try:
        if enabled:
            for stage in pipe.stages:
                for name, kind in (("forward_one_chunk", "F"), ("backward_one_chunk", "B")):
                    method = getattr(stage, name)
                    originals.append((stage, name, method))
                    setattr(stage, name, wrap(method, kind, stage.stage_index))
        yield
    finally:
        for stage, name, method in originals:
            setattr(stage, name, method)


class _SynchronousCPUOptimizer:
    """Benchmark-only CPU update for a drained Stage; no second GPU bank.

    Each stage owns a transfer stream. All stage updates run concurrently after
    flush_grads, and all uploads complete before the next Pipeline.step begins.
    This retains fresh-gradient synchronous semantics, not 2BW's stale rule.
    """
    def __init__(self, stage, factory):
        self.stage = stage
        self.named = dict(stage.module.named_parameters())
        self.banks = [self.named]
        self.cpu_grads = {}
        self.optimizer_named = {}
        cuda = stage.device.type == "cuda"
        for name, parameter in self.named.items():
            master = torch.empty_like(parameter, device="cpu", pin_memory=cuda)
            master.copy_(parameter.detach())
            self.optimizer_named[name] = nn.Parameter(master, requires_grad=parameter.requires_grad)
        self.optimizer = factory(self.optimizer_named.values())
        self.stream = torch.cuda.Stream(device=stage.device) if cuda else None

    @torch.no_grad()
    def step(self, group, trace, ready=None):
        device = self.stage.device
        cuda = self.stream is not None
        with (torch.cuda.device(device) if cuda else contextlib.nullcontext()), \
             (torch.cuda.stream(self.stream) if cuda else contextlib.nullcontext()):
            if ready is not None:
                self.stream.wait_event(ready)
            def span(name, **kwargs):
                return trace.span(name, stage=self.stage.stage_index, group=group, **kwargs)
            with span("U"):
                with span("optimizer_d2h", bytes=sum(p.grad.numel()*p.grad.element_size()
                          for p in self.named.values() if p.grad is not None)):
                    for name, parameter in self.named.items():
                        master = self.optimizer_named[name]
                        if parameter.grad is None:
                            master.grad = None
                        else:
                            if name not in self.cpu_grads:
                                self.cpu_grads[name] = torch.empty_like(master, device="cpu", pin_memory=cuda)
                            # PipelineResult.flush_grads already applied 1/m.
                            self.cpu_grads[name].copy_(parameter.grad, non_blocking=cuda)
                            master.grad = self.cpu_grads[name]
                    if cuda:
                        copied = torch.cuda.Event()
                        copied.record(self.stream)
                with span("optimizer_d2h_wait"):
                    if cuda:
                        copied.synchronize()
                with span("optimizer_cpu", fused=True):
                    self.optimizer.step()
                with span("optimizer_h2d", bytes=sum(p.numel()*p.element_size() for p in self.named.values())):
                    for name, parameter in self.named.items():
                        parameter.copy_(self.optimizer_named[name], non_blocking=cuda)
                with span("optimizer_h2d_wait"):
                    if cuda:
                        self.stream.synchronize()
                self.optimizer.zero_grad(set_to_none=True)
                for parameter in self.named.values():
                    parameter.grad = None


def run_synchronous(pipe, optimizers, loader, updates, microbatches, profile_path=None):
    """Capture the ENTIRE baseline, including loading, flush, and optimizer work."""
    iterator = iter(loader)
    group_context = {"group": 0}
    cpu_updates = all(isinstance(opt, _SynchronousCPUOptimizer) for opt in optimizers)
    # Do not serialize CPU updates across stages: unlike GPU kernel enqueueing,
    # optimizer.step on CPU blocks the calling thread until arithmetic finishes.
    with (ThreadPoolExecutor(max_workers=len(optimizers), thread_name_prefix="1f1b-cpu")
          if cpu_updates else contextlib.nullcontext()) as pool, TraceCapture(profile_path) as trace:
        with baseline_stage_spans(pipe, trace, profile_path is not None, group_context):
            try:
                for group in range(updates):
                    group_context["group"] = group
                    with trace.span("loader_next", group=group):
                        inputs, targets = next(iterator)
                    with trace.span("step", group=group, schedule="staggered_1b1f"):
                        # No per-step profiler: it would miss the flush/optimizer
                        # and nest inside the full-session Kineto capture.
                        result = pipe.step(inputs, targets=targets,
                                           schedule="staggered_1b1f",
                                           n_microbatches=microbatches, loss_fn=F.mse_loss)
                    with trace.span("flush_grads", group=group):
                        result.flush_grads()  # Exactly one 1/m mean scaling.
                    if cpu_updates:
                        futures = []
                        for adapter in optimizers:
                            ready = None
                            if adapter.stream is not None:
                                ready = torch.cuda.Event()
                                ready.record(torch.cuda.current_stream(adapter.stage.device))
                            futures.append(pool.submit(adapter.step, group, trace, ready))
                        # Wait for all workers even on error before profiler exit.
                        errors = []
                        for future in futures:
                            try:
                                future.result()
                            except BaseException as error:
                                errors.append(error)
                        if errors:
                            raise errors[0]
                    else:
                        for stage, optimizer in enumerate(optimizers):
                            with trace.span("optimizer", stage=stage, group=group):
                                optimizer.step()
                                optimizer.zero_grad(set_to_none=True)
                    del result, inputs, targets
            finally:
                # Kineto must remain active until every optimizer kernel ends.
                synchronize(pipe.devices)


def make_optimizer_factory(args, mode):
    if mode.endswith("cpu_fused"):
        return functools.partial(torch.optim.AdamW, lr=args.lr, foreach=False, fused=True)
    return optimizer_factory(args.optimizer, lr=args.lr)


def tensor_state(value):
    """Yield every tensor, including nested optimizer state without copying it."""
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensor_state(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensor_state(item)


def verify_stage_storage(stage, cpu_optimizer, *, during_step=False):
    """Check placement and count logical tensor bytes (not allocator/RSS bytes)."""
    optimizer = stage.optimizer
    named = stage.optimizer_named
    parameters = list(named.values())
    actual = [p for group in optimizer.param_groups for p in group["params"]]
    if len(actual) != len(parameters) or {id(p) for p in actual} != {id(p) for p in parameters}:
        raise AssertionError("optimizer must own exactly optimizer_named parameters")
    expected = torch.device("cpu") if cpu_optimizer else stage.stage.device
    if any(p.device != expected for p in parameters):
        raise AssertionError("optimizer parameter on the wrong device")
    attached_gradients = [p.grad for p in parameters if p.grad is not None]
    gradient_buffers = list(getattr(stage, "cpu_grads", {}).values())
    # The CPU runtime retains pinned buffers after zero_grad(set_to_none=True).
    # During a step these same buffers are attached as Parameter.grad: count once.
    gradients = list({id(tensor): tensor for tensor in
                      [*attached_gradients, *gradient_buffers]}.values())
    if any(grad.device != expected for grad in gradients):
        raise AssertionError("optimizer gradient on the wrong device")
    if during_step and len(attached_gradients) != sum(p.requires_grad for p in parameters):
        raise AssertionError("demo expected every trainable parameter to have a gradient")
    states = list(tensor_state(optimizer.state))
    if cpu_optimizer:
        if type(optimizer) is not torch.optim.AdamW or any(
            group.get("fused") is not True or group.get("foreach") is not False
            for group in optimizer.param_groups
        ):
            raise AssertionError("CPU comparison requires fused=True, foreach=False AdamW")
        if any(tensor.device.type != "cpu" for tensor in states):
            raise AssertionError("every CPU optimizer state tensor must be on CPU")
        if stage.stage.device.type == "cuda" and not all(
            tensor.is_pinned() for tensor in [*parameters, *gradients]
        ):
            raise AssertionError("CPU masters and gradient buffers must use pinned memory")
        if during_step and any(
            not {"step", "exp_avg", "exp_avg_sq"}.issubset(optimizer.state.get(p, {}))
            for p in parameters if p.requires_grad
        ):
            raise AssertionError("CPU AdamW state was not initialized for every parameter")
    expected_banks = 1 if isinstance(stage, _SynchronousCPUOptimizer) else 2
    if len(stage.banks) != expected_banks or any(
        set(bank) != set(named) or any(t.device != stage.stage.device for t in bank.values())
        for bank in stage.banks
    ):
        raise AssertionError("complete weight banks must remain on the stage device")
    if expected_banks == 2 and any(stage.banks[0][name].data_ptr() == stage.banks[1][name].data_ptr() for name in named):
        raise AssertionError("weight banks must use distinct storage")
    by_device = {}
    for category, tensors in (
        ("parameter_bytes", parameters), ("gradient_bytes", gradients),
        ("state_bytes", states),
        ("weight_bank_bytes", [tensor for bank in stage.banks for tensor in bank.values()]),
    ):
        for tensor in tensors:
            counts = by_device.setdefault(str(tensor.device), dict(
                parameter_bytes=0, gradient_bytes=0, state_bytes=0, weight_bank_bytes=0))
            counts[category] += tensor.numel() * tensor.element_size()
    return {
        "stage": stage.stage.stage_index, "verified": True,
        "optimizer_device": str(expected),
        "fused": all(group.get("fused") is True for group in optimizer.param_groups),
        "parameter_tensors": len(parameters), "gradient_tensors": len(gradients),
        "attached_gradient_tensors": len(attached_gradients),
        "retained_gradient_buffer_tensors": len(gradient_buffers),
        "state_tensors": len(states), "bytes_by_device": by_device,
    }


def summarize_storage(stages):
    totals = {}
    for stage in stages:
        for device, counts in stage["bytes_by_device"].items():
            combined = totals.setdefault(device, {name: 0 for name in counts})
            for name, count in counts.items():
                combined[name] += count
    return {"stages": stages, "bytes_by_device": totals, "verified": True}


@contextlib.contextmanager
def profile_storage_checks(trainer, cpu_optimizer, enabled):
    """Inspect live grads/state after each step, before runtime zero_grad.

    Hooks only inspect metadata, never copy tensors or change optimizer math.
    They are deliberately absent from clean timing and warmup.
    """
    checks = [{"steps_verified": 0} for _ in trainer.stages] if enabled else []
    handles = []
    try:
        for stage, check in zip(trainer.stages, checks):
            def after_step(optimizer, args, kwargs, stage=stage, check=check):
                check["last_step_storage"] = verify_stage_storage(
                    stage, cpu_optimizer, during_step=True)
                check["steps_verified"] += 1
            handles.append(stage.optimizer.register_step_post_hook(after_step))
        yield checks
    finally:
        for handle in handles:
            handle.remove()


def measure_run(args, devices, mode, updates, profile_path=None):
    """Build fresh state; measure execution, not model/session construction."""
    if mode not in ALL_MODES:
        raise ValueError("unknown mode: {}".format(mode))
    cpu_optimizer = mode.endswith("cpu_fused")
    gc.collect()
    synchronize(devices)
    for device in devices:
        if device.type == "cuda":
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
    pipe = make_pipe(args, devices)
    trainer = None
    try:
        factory = make_optimizer_factory(args, mode)
        if mode in CPU_OPTIMIZER_MODES:
            trainer = pipe.train_session(optimizer_factory=factory,
                                         n_microbatches=args.microbatches, loss_fn=F.mse_loss,
                                         optimizer_device="cpu" if cpu_optimizer else None)
        elif cpu_optimizer:
            optimizers = [_SynchronousCPUOptimizer(stage, factory) for stage in pipe.stages]
        else:
            optimizers = [factory(stage.module.parameters()) for stage in pipe.stages]
        storage_owner = trainer if trainer is not None else (
            SimpleNamespace(stages=optimizers) if cpu_optimizer else None)
        loader = group_batches(args, updates,
                               pin_memory=any(device.type == "cuda" for device in devices))
        synchronize(devices)
        memory = {}
        for device in devices:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                memory[str(device)] = {
                    "allocated_before_bytes": torch.cuda.memory_allocated(device),
                    "reserved_before_bytes": torch.cuda.memory_reserved(device),
                }
        validation = (profile_storage_checks(storage_owner, cpu_optimizer, profile_path is not None)
                      if storage_owner is not None else contextlib.nullcontext([]))
        with validation as storage_checks:
            start = time.perf_counter()
            if trainer is not None:
                result = trainer.run(loader, updates=updates, profile_path=profile_path)
            else:
                run_synchronous(pipe, optimizers, loader, updates, args.microbatches, profile_path)
            synchronize(devices)
            elapsed = time.perf_counter() - start
        for device in devices:
            if device.type == "cuda":
                memory[str(device)].update(
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
                    allocated_after_bytes=torch.cuda.memory_allocated(device),
                    reserved_after_bytes=torch.cuda.memory_reserved(device),
                )
        samples = updates * args.microbatches * args.batch_size
        stats = {
            "updates": updates,
            "microbatches": updates * args.microbatches,
            "samples": samples,
            "wall_seconds": elapsed,
            "profiled": profile_path is not None,
            "memory_per_device": memory,
        }
        if profile_path is None:
            stats.update(updates_per_second=updates / elapsed,
                         microbatches_per_second=updates * args.microbatches / elapsed,
                         samples_per_second=samples / elapsed)
        if trainer is not None:
            stats["peak_inflight_microbatches"] = result.peak_inflight
            stats["max_inflight_microbatches"] = trainer.max_inflight
        if storage_owner is not None:
            stats["optimizer_storage_after_run"] = summarize_storage([
                verify_stage_storage(stage, cpu_optimizer) for stage in storage_owner.stages
            ])
            if storage_checks:
                if any(check["steps_verified"] != updates for check in storage_checks):
                    raise AssertionError("not every profiled optimizer step was storage-verified")
                stats["optimizer_storage_during_step"] = summarize_storage([
                    dict(check["last_step_storage"], steps_verified=check["steps_verified"])
                    for check in storage_checks
                ])
        return stats
    finally:
        if trainer is not None:
            trainer.close()
        pipe.close()


def environment_metadata(devices):
    gpus = []
    for device in devices:
        properties = torch.cuda.get_device_properties(device)
        gpus.append({
            "device": str(device), "name": properties.name,
            "compute_capability": [properties.major, properties.minor],
            "total_memory_bytes": properties.total_memory,
            "multiprocessor_count": properties.multi_processor_count,
            "uuid": str(getattr(properties, "uuid", "unavailable")),
        })
    return {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "hostname": platform.node(),
        "torch": str(torch.__version__), "torch_git_version": torch.version.git_version,
        "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
        "gpus": gpus, "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "allow_bf16_reduced_precision_reduction":
            torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        "environment_variables": {name: os.environ.get(name) for name in (
            "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG", "NVIDIA_TF32_OVERRIDE",
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
        )},
    }


def verify_profile(path, devices, *, mode=None, updates=None, microbatches=None):
    summary = inspect_trace(path)
    counts = summary["gpu_kernels_per_device"]
    missing = [str(device) for device in devices if counts.get(str(device.index), 0) < 1]
    summary.update(missing_cuda_devices=missing, all_cuda_devices_have_kernels=not missing)
    errors = []
    if (mode or "").endswith("cpu_fused"):
        expected = {name: updates * len(devices) for name in (
            "U", "optimizer_d2h", "optimizer_d2h_wait", "optimizer_cpu", "optimizer_h2d")}
        expected.update(F=updates * microbatches * len(devices),
                        B=updates * microbatches * len(devices))
        actual = summary["cpu_annotations_by_name"]
        mismatches = {name: {"expected": count, "actual": actual.get(name, 0)}
                      for name, count in expected.items() if actual.get(name, 0) != count}
        summary.update(expected_cpu_annotations=expected, annotation_mismatches=mismatches)
        if mismatches:
            errors.append("CPU optimizer annotation counts do not match the workload")
        copies = summary["gpu_memcopies_by_direction"]
        # Whole-trace counts include inputs/targets/relay; they are not attributed
        # to individual optimizer spans. No exact per-parameter count is assumed.
        if summary["gpu_memcopies"]:
            absent = [direction for direction in ("D2H", "H2D") if not copies.get(direction)]
            summary["optimizer_memcpy_validation"] = {
                "status": "failed" if absent else "passed", "missing_directions": absent,
                "scope": "whole trace; includes input/target/relay copies",
            }
            if absent:
                errors.append("real CUDA memcpy events missing directions: " + ", ".join(absent))
        else:
            summary["optimizer_memcpy_validation"] = {
                "status": "unavailable", "reason": "Kineto exposed no real CUDA memcpy events",
            }
    summary["validation_errors"] = errors
    summary["validation_passed"] = not missing and not errors
    return summary


def write_bundle(output_dir, metadata, traces):
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, allow_nan=False)
        stream.write("\n")
    bundle_path = output_dir / "pipedream_2bw_bundle.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED,
                         compresslevel=6) as bundle:
        for path in [metadata_path, *traces]:
            bundle.write(path, arcname=path.name)
    with zipfile.ZipFile(bundle_path) as bundle:
        bad_file = bundle.testzip()
        if bad_file is not None:
            raise RuntimeError("ZIP integrity check failed for {}".format(bad_file))
    return metadata_path, bundle_path


def main(argv=None):
    args, devices = parse_args(argv)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this GPU benchmark; no CUDA device is available.")
    for device in devices:
        if device.index >= torch.cuda.device_count():
            raise SystemExit("Requested {} is not a visible CUDA device".format(device))
        if args.bf16:
            with torch.cuda.device(device):
                if not torch.cuda.is_bf16_supported():
                    raise SystemExit("BF16 is not supported on {}".format(device))
    configure_determinism(MODEL_SEED)
    torch.set_num_threads(args.cpu_threads)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    modes = ALL_MODES if args.compare_all else (CPU_OPTIMIZER_MODES if args.compare_cpu_optimizer else MODES)
    warmup_updates = min(3, args.updates)
    optimizer_settings = dict(
        name=args.optimizer, lr=args.lr, weight_decay=0.01, foreach=False, fused=False,
        **({"momentum": 0.9, "dampening": 0, "nesterov": False} if args.optimizer == "sgd"
           else {"betas": [0.9, 0.999], "eps": 1e-8}))
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "argv": list(sys.argv if argv is None else [sys.argv[0], *argv]),
        "cli": dict(vars(args), output_dir=str(args.output_dir)),
        "environment": dict(environment_metadata(devices), requested_cpu_threads=args.cpu_threads),
        "comparison": "schedule_and_optimizer_matrix" if args.compare_all else (
            "same_schedule_cpu_optimizer" if args.compare_cpu_optimizer else "synchronous"),
        "modes": modes,
        "workload": {
            "model": "ResidualMLPBlock: Linear(d,4d), GELU, Linear(4d,d), residual scale 0.1",
            "model_seed": MODEL_SEED, "data_seed": DATA_SEED,
            "loss": "MSE microbatch mean, accumulated then scaled once by 1/microbatches",
            "group_batch_size": args.microbatches * args.batch_size,
            "parameter_dtype": "float32", "input_dtype": "float32",
            "autocast_dtype": "bfloat16" if args.bf16 else None,
            "layers_per_stage": [args.layers * (i + 1) // len(devices)
                                 - args.layers * i // len(devices) for i in range(len(devices))],
            "optimizer": optimizer_settings,
            "optimizer_per_mode": {
                mode: dict(optimizer_settings, fused=mode.endswith("cpu_fused"),
                           device="cpu" if mode.endswith("cpu_fused") else "stage_device")
                for mode in modes
            },
        },
        "methodology": {
            "warmup_updates_per_mode": warmup_updates,
            "synchronous_cpu_updates": "concurrent per-stage worker threads after full F/B drain; "
                                       "one GPU weight bank; all uploads finish before next step",
            "fresh_models_for_warmup_timing_and_profile": True,
            "lazy_pinned_cpu_groups": True, "correctness_observer": False,
            "timing_includes": "lazy data generation, lazy optimizer-state/gradient-buffer allocation, "
                               "all H2D/D2H/relay "
                               "transfers and waits, forward/backward, optimizer updates, final drain",
            "timing_excludes": "model/session construction (including CPU master allocation/initial copy), "
                               "warmup, profiling, artifact export, storage validation, teardown",
            "profile_wall_seconds_include_capture_and_export": True,
            "profile_wall_seconds_include_storage_validation": True,
            "weight_comparison": ("not performed; same 2BW update semantics and data, but fused CPU "
                                  "and nonfused GPU AdamW can differ numerically"
                                  if args.compare_cpu_optimizer else
                                  "not performed; stale-gradient 2BW and synchronous updates differ"),
            "memory": "PyTorch CUDA allocator bytes only; CPU optimizer tensor bytes are separate, "
                      "not process RSS or total pinned-host allocation",
            "optimizer_storage": "logical tensor bytes by device/category after run; profile step hooks "
                                 "verify live grads and every tensor state on every update and report "
                                 "the final step before zero_grad. Gradient bytes include retained CPU "
                                 "transfer buffers, counted once when attached as Parameter.grad. "
                                 "GPU after-run gradients can be zero bytes. "
                                 "GPU optimizer parameters alias a weight bank: do not sum categories "
                                 "as unique allocation bytes. No copies or correctness observer used.",
        },
        "timings": {}, "profiles": {}, "bundle": "pipedream_2bw_bundle.zip",
    }
    print("Independent warmups: {} updates per mode".format(warmup_updates), flush=True)
    for mode in modes:
        measure_run(args, devices, mode, warmup_updates)
    for mode in modes:
        print("Clean timing: {}".format(mode), flush=True)
        stats = measure_run(args, devices, mode, args.updates)
        metadata["timings"][mode] = stats
        print("  {:.3f}s; {:.2f} updates/s; {:.1f} samples/s".format(
            stats["wall_seconds"], stats["updates_per_second"], stats["samples_per_second"]), flush=True)
    if args.compare_all:
        ratios = {}
        for suffix in ("", "_cpu_fused"):
            speedup = (metadata["timings"]["staggered_1b1f"+suffix]["wall_seconds"]
                       / metadata["timings"]["pipedream_2bw"+suffix]["wall_seconds"])
            ratios["cpu_fused" if suffix else "gpu"] = speedup
            print("2BW / synchronous throughput ({}) : {:.3f}x (different update semantics)".format(
                "CPU fused" if suffix else "GPU optimizer", speedup), flush=True)
        metadata["pipedream_2bw_speedup_vs_synchronous_by_optimizer"] = ratios
    elif args.compare_cpu_optimizer:
        speedup = (metadata["timings"]["pipedream_2bw"]["wall_seconds"]
                   / metadata["timings"]["pipedream_2bw_cpu_fused"]["wall_seconds"])
        metadata["cpu_fused_speedup_vs_gpu_optimizer"] = speedup
        print("CPU fused / GPU optimizer throughput: {:.3f}x (same 2BW schedule)".format(
            speedup), flush=True)
    else:
        speedup = (metadata["timings"]["staggered_1b1f"]["wall_seconds"]
                   / metadata["timings"]["pipedream_2bw"]["wall_seconds"])
        metadata["pipedream_2bw_speedup_vs_synchronous"] = speedup
        print("2BW / synchronous throughput: {:.3f}x (different update semantics)".format(
            speedup), flush=True)
    traces = []
    for mode in modes:
        path = args.output_dir / (mode + ".json.gz")
        print("Fresh-model profile: {}".format(path), flush=True)
        stats = measure_run(args, devices, mode, args.updates, profile_path=path)
        inspection = verify_profile(path, devices, mode=mode, updates=args.updates,
                                    microbatches=args.microbatches)
        metadata["profiles"][mode] = dict(
            path=path.name, bytes=path.stat().st_size, run=stats, inspection=inspection)
        traces.append(path)
        print("  Real kernel counts by CUDA ordinal: {}".format(
            inspection["gpu_kernels_per_device"]), flush=True)
        if mode.endswith("cpu_fused"):
            print("  Real CUDA memcpy counts by direction: {} ({})".format(
                inspection["gpu_memcopies_by_direction"],
                inspection["optimizer_memcpy_validation"]["status"]), flush=True)
            cpu_bytes = stats["optimizer_storage_during_step"]["bytes_by_device"]["cpu"]
            print("  CPU optimizer bytes: parameters={parameter_bytes:,}; "
                  "gradients={gradient_bytes:,}; state={state_bytes:,}".format(**cpu_bytes), flush=True)
    valid = all(profile["inspection"]["validation_passed"]
                for profile in metadata["profiles"].values())
    metadata["trace_validation_passed"] = valid
    metadata_path, bundle_path = write_bundle(args.output_dir, metadata, traces)
    print("Metadata: {}\nVerified ZIP: {}".format(metadata_path, bundle_path), flush=True)
    if not valid:
        missing = {mode: profile["inspection"]["missing_cuda_devices"]
                   for mode, profile in metadata["profiles"].items()
                   if profile["inspection"]["missing_cuda_devices"]}
        errors = {mode: profile["inspection"]["validation_errors"]
                  for mode, profile in metadata["profiles"].items()
                  if profile["inspection"]["validation_errors"]}
        print("ERROR: trace validation failed; missing GPU kernels: {}; other errors: {}. "
              "Check Kineto/CUPTI availability; diagnostic artifacts were saved.".format(
                  missing, errors), file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
