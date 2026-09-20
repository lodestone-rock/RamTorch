"""Exact per-update checks for the resident PipeDream-2BW session.

PYTHONPATH=. python examples/pipedream_2bw_check.py --devices cuda:0,cuda:1,cuda:2,cuda:3

Add --long-run --updates 100 --bf16 to compare the base 1024-wide, eight-block
workload against the unsplit single-GPU reference with bounded debug snapshots.

--cpu-optimizer-only runs small fp32 AND bf16 CPU-fused AdamW checks, including
four-stage validation when given four GPUs. Its independent oracle evaluates on
CUDA but updates ordinary CPU parameters with CPU-fused AdamW, not GPU AdamW.
For a custom workload use --long-run --optimizer-device cpu --fused.

--sync-cpu-optimizer-only checks the synchronous benchmark's CPU-fused AdamW
against a separate fresh-weight oracle (NOT the delayed PipeDream reference).
It covers fp32 AND bf16, one resident weight bank, and continued drained runs.
"""
import argparse
import copy
import gc
import json
import queue
from pathlib import Path
import threading
import time
import weakref
from unittest import mock

from pipedream_2bw_reference import (
    assert_exact, configure_determinism, make_batches, make_model,
    optimizer_factory, run_reference,
)
import torch
from torch import nn
from torch.nn import functional as F
from ramtorch import Pipeline
from ramtorch.pipeline import Stage
from ramtorch.pipeline_2bw import _VersionedStage, _rank_ops


def loader_batches(groups):
    for group in groups:
        yield torch.cat([x for x, _ in group]), torch.cat([y for _, y in group])


def check_long_run(devices, updates=100, dim=1024, layers=8, microbatches=4,
                   batch_size=128, optimizer="adamw", bf16=True, lr=0.001,
                   report_path=None, optimizer_device=None, fused=False):
    """Compare every update directly, with bounded CPU snapshots, no hashes.

    The independent one-GPU oracle runs on the caller thread. A separate thread
    runs ONE continuous pipeline run. Each stage supplies one CPU snapshot at a
    time; the oracle consumes and releases it immediately. This debug-only
    backpressure is intentionally not used for performance measurements.
    """
    from collections import OrderedDict
    if min(updates, dim, layers, microbatches, batch_size) < 1:
        raise ValueError("all workload sizes must be positive")
    if layers < len(devices) or microbatches < len(devices):
        raise ValueError("layers and microbatches must be >= pipeline depth")
    configure_determinism()
    model = make_model(dim=dim, layers=layers)
    groups = make_batches(updates=updates, microbatches=microbatches, dim=dim,
                          batch_size=batch_size)
    factory = optimizer_factory(optimizer, lr=lr, fused=fused)
    session_options = {} if optimizer_device is None else {"optimizer_device": optimizer_device}
    cuts = [layers*i//len(devices) for i in range(len(devices)+1)]
    modules = [nn.Sequential(OrderedDict((str(j), copy.deepcopy(model[j]))
               for j in range(cuts[i], cuts[i+1]))) for i in range(len(devices))]
    pipe = Pipeline(stage_modules=modules, devices=devices,
                    autocast=torch.bfloat16 if bf16 else None)
    mailboxes = [queue.Queue(maxsize=1) for _ in devices]
    cancelled, finished = threading.Event(), threading.Event()
    errors, run_results = [], []
    compared = 0
    versions = []
    losses = []
    began = time.perf_counter()

    def publish(stage, group, snapshot):
        while not cancelled.is_set():
            try:
                mailboxes[stage].put((group, snapshot), timeout=0.1)
                return
            except queue.Full:
                pass
        raise RuntimeError("long exact comparison cancelled")

    def compare(update):
        nonlocal compared
        for stage, box in enumerate(mailboxes):
            while True:
                try:
                    group, actual = box.get(timeout=0.1)
                    break
                except queue.Empty:
                    if errors:
                        raise RuntimeError("pipeline failed during comparison") from errors[0]
                    if finished.is_set():
                        raise AssertionError("pipeline ended without expected snapshot")
            assert group == update.group, (stage, group, update.group)
            names = dict(modules[stage].named_parameters())
            expected = {
                "grads": {n: update.gradients[n] for n in names},
                "weights": {n: update.weights[n] for n in names},
                "optimizer": {n: update.optimizer_state[n] for n in names},
            }
            assert_exact(actual, expected, path=f"long.group{group}.stage{stage}")
            del actual, expected
        compared += 1
        versions.append(update.eval_version)
        losses.append(sum(float(loss) for loss in update.losses)/len(update.losses))
        if compared == 1 or compared % 10 == 0 or compared == updates:
            print(f"BIT-EXACT {compared}/{updates}: gradients + weights + optimizer state; "
                  f"eval=w{update.eval_version} latest=w{compared} loss={losses[-1]:.7g}", flush=True)

    try:
        with pipe.train_session(optimizer_factory=factory, n_microbatches=microbatches,
                                loss_fn=F.mse_loss, **session_options) as trainer:
            def pipeline_run():
                try:
                    run_results.append(trainer.run(loader_batches(groups), updates=updates,
                                                   observer=publish))
                except BaseException as error:
                    errors.append(error)
                finally:
                    finished.set()
            worker = threading.Thread(target=pipeline_run, daemon=True, name="long-check-pipeline")
            worker.start()
            try:
                result = run_reference(model, groups, factory, device=devices[0],
                                       optimizer_device=optimizer_device,
                                       autocast_bf16=bf16, observer=compare,
                                       retain_updates=False)
                assert result.updates == [], "streaming oracle retained its history"
            finally:
                cancelled.set()
                worker.join(120)
                if worker.is_alive():
                    raise RuntimeError("pipeline failed to stop after comparison")
            if errors:
                raise RuntimeError("pipeline comparison failed") from errors[0]
            assert compared == updates
            assert run_results[0].total_updates == updates
    finally:
        pipe.close()
    report = {
        "status": "BIT_EXACT", "comparison": "direct bytewise, no tolerances or hashes",
        "updates_compared": compared, "microbatches_per_update": microbatches,
        "total_microbatches": compared*microbatches, "dim": dim, "layers": layers,
        "batch_size_per_microbatch": batch_size, "optimizer": optimizer, "lr": lr,
        "bf16": bf16, "reference_device": devices[0], "pipeline_devices": devices,
        "optimizer_device": optimizer_device or "stage", "fused": fused,
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "gpu_names": [torch.cuda.get_device_name(d) if torch.device(d).type == "cuda" else "cpu"
                      for d in devices],
        "eval_versions": versions, "reference_mean_losses": losses,
        "compared_each_update": ["scaled_gradients", "post_update_weights", "optimizer_state"],
        "wall_seconds_with_debug_comparison": time.perf_counter()-began,
    }
    if report_path is not None:
        path = Path(report_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2)+"\n")
    print(f"LONG RUN PASS: {compared} updates, {compared*microbatches} microbatches, "
          f"{optimizer}, {'bf16' if bf16 else 'fp32'}, all tensors byte-identical", flush=True)
    return report


def check_banks(device, optimizer="adamw"):
    """Keep next group's old-version graph alive across the first update."""
    model = make_model(dim=8, layers=2)
    groups = make_batches(updates=8, microbatches=1, dim=8)
    factory = optimizer_factory(optimizer)
    expected = run_reference(model, groups, factory, device=device)
    bank = _VersionedStage(Stage(copy.deepcopy(model), 0, 1, device=device), factory)
    records = []
    pointers = [{n: t.untyped_storage().data_ptr() for n, t in b.items()} for b in bank.banks]
    bank.forward(0, groups[0][0][0], 1)
    bank.forward(1, groups[1][0][0], 1)
    for group in range(len(groups)):
        if group > 1:
            bank.forward(group, groups[group][0][0], 1)
        bank.backward(group, target=groups[group][0][1].to(device), loss_fn=F.mse_loss)
        bank.update(group, 1, lambda s, g, snap: records.append(snap))
        for slot in range(2):
            assert pointers[slot] == {n: t.untyped_storage().data_ptr() for n, t in bank.banks[slot].items()}
        for n, p in bank.named.items():
            assert pointers[0][n] != pointers[1][n]
            assert p.untyped_storage().data_ptr() in (pointers[0][n], pointers[1][n])
    for group, actual in enumerate(records):
        ref = expected.updates[group]
        assert_exact(actual, {"grads": ref.gradients, "weights": ref.weights,
                              "optimizer": ref.optimizer_state}, path=f"bank.{optimizer}.{group}")
    bank.release()
    print(f"PASS bank lifetime/storage/optimizer {device} {optimizer}")


def check_pipeline(devices, m, updates, optimizer, bf16=False, split_run=False,
                   slow=False, dim=16, layers=None, max_inflight=None):
    depth = len(devices)
    layers = layers or depth * 2
    model = make_model(dim=dim, layers=layers)
    groups = make_batches(updates=updates, microbatches=m, dim=dim)
    factory = optimizer_factory(optimizer)
    expected = run_reference(model, groups, factory, device=devices[0], autocast_bf16=bf16)
    # Keep global block names so state snapshots map directly to the oracle.
    from collections import OrderedDict
    cuts = [layers * i // depth for i in range(depth + 1)]
    modules = [nn.Sequential(OrderedDict((str(j), copy.deepcopy(model[j]))
               for j in range(cuts[i], cuts[i+1]))) for i in range(depth)]
    pipe = Pipeline(stage_modules=modules, devices=devices,
                    autocast=torch.bfloat16 if bf16 else None)
    records = {}
    def observe(stage, group, snapshot):
        assert (stage, group) not in records
        records[stage, group] = snapshot
    def source(part):
        for item in loader_batches(part):
            if slow:
                time.sleep(0.002)
            yield item
    with pipe.train_session(optimizer_factory=factory, n_microbatches=m,
                            loss_fn=F.mse_loss, max_inflight=max_inflight) as trainer:
        if split_run and updates > 1:
            trainer.run(source(groups[:1]), updates=1, observer=observe)
            result = trainer.run(source(groups[1:]), updates=updates-1, observer=observe)
        else:
            result = trainer.run(source(groups), updates=updates, observer=observe)
        assert result.peak_inflight <= trainer.max_inflight
        assert result.total_updates == updates
        for group in range(updates):
            for stage in range(depth):
                ref = expected.updates[group]
                names = dict(modules[stage].named_parameters())
                want = {"grads": {n: ref.gradients[n] for n in names},
                        "weights": {n: ref.weights[n] for n in names},
                        "optimizer": {n: ref.optimizer_state[n] for n in names}}
                assert_exact(records[stage, group], want, path=f"stage{stage}.group{group}")
        # Public APIs may not race the session's banks.
        try:
            pipe.infer(torch.randn(m, dim), n_microbatches=m)
        except RuntimeError:
            pass
        else:
            raise AssertionError("inference accepted an active trainer")
    for module in modules:
        for name, parameter in module.named_parameters():
            assert_exact(parameter, expected.updates[-1].weights[name], path="closed."+name)
    pipe.close()
    print(f"PASS pipeline {devices} m={m} updates={updates} {optimizer} bf16={bf16} continuation={split_run} peak={result.peak_inflight}")


def check_unobserved(devices):
    """No debug CPU snapshots/synchronization between updates: catch async races."""
    from collections import OrderedDict
    depth, m, updates = len(devices), len(devices) + 1, 9
    model = make_model(dim=64, layers=2*depth)
    groups = make_batches(updates=updates, microbatches=m, dim=64, batch_size=8)
    factory = optimizer_factory("adamw")
    reference = run_reference(model, groups, factory, device=devices[0]).updates[-1]
    for capacity in (depth, 2*m+depth):
        modules = [nn.Sequential(OrderedDict((str(j), copy.deepcopy(model[j]))
                   for j in range(2*i, 2*i+2))) for i in range(depth)]
        pipe = Pipeline(stage_modules=modules, devices=devices)
        with pipe.train_session(optimizer_factory=factory, n_microbatches=m,
                               loss_fn=F.mse_loss, max_inflight=capacity) as trainer:
            if capacity == depth and torch.device(devices[0]).type == "cuda":
                # Inputs/targets produced on non-default caller streams, with
                # targets originating on a different device when available.
                xs = torch.cuda.Stream(device=devices[0])
                ys = torch.cuda.Stream(device=devices[-1])
                def gpu_source():
                    for x, y in loader_batches(groups):
                        with torch.cuda.stream(xs), torch.cuda.stream(ys):
                            yield x.to(devices[0], non_blocking=True) + 0, y.to(devices[-1], non_blocking=True) + 0
                source = gpu_source()
            else:
                source = loader_batches(groups)
            trainer.run(source, updates=updates)
            for stage in trainer.stages:
                for name, parameter in stage.named.items():
                    assert_exact(parameter, reference.weights[name], path="async.weight."+name)
                    assert_exact(stage.optimizer.state.get(parameter, {}),
                                 reference.optimizer_state[name], path="async.optimizer."+name)
        pipe.close()
    print("PASS unobserved asynchronous final weights/state at minimum/default capacity")


def check_empty_gradient_fences(devices, optimizer_device=None, fused=False):
    """Detached boundaries still need a CUDA completion fence without tensors."""
    session_options = {} if optimizer_device is None else {"optimizer_device": optimizer_device}
    from ramtorch.pipeline_2bw import PipeDream2BWTrainer
    from ramtorch.pipeline_relay import _Mailbox
    cuda = [torch.device(d) for d in devices if torch.device(d).type == "cuda"]
    if cuda:
        source, destination = cuda[-1], cuda[0]
        producer = torch.cuda.Stream(device=source)
        consumer = torch.cuda.Stream(device=destination)
        for value in (None, (None, None)):
            box = _Mailbox()
            with torch.cuda.device(source), torch.cuda.stream(producer):
                tensor = torch.ones(1024, 1024, device=source)
                tensor = tensor @ tensor
                complete = torch.cuda.Event()
                complete.record(producer)
                PipeDream2BWTrainer._publish(box, value, producer)
            assert box._cuda_events, "empty payload lost producer-completion fence"
            with torch.cuda.device(destination), torch.cuda.stream(consumer):
                assert box.get() == value
            consumer.synchronize()
            assert complete.query(), "consumer retired before producer work"

    class Detached(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(4, 4)
        def forward(self, x):
            return self.linear(x.detach())
    actual_devices = devices if len(devices) > 1 else devices * 2
    model = nn.Sequential(nn.Linear(4, 4), Detached())
    groups = make_batches(updates=7, microbatches=2, dim=4)
    factory = optimizer_factory("adamw", weight_decay=0.3, fused=fused)
    expected = run_reference(model, groups, factory, device=actual_devices[0],
                             optimizer_device=optimizer_device).updates[-1]
    pipe = Pipeline(stage_modules=[copy.deepcopy(model[0]), copy.deepcopy(model[1])],
                    devices=[actual_devices[0], actual_devices[-1]])
    with pipe.train_session(optimizer_factory=factory, n_microbatches=2,
                            loss_fn=F.mse_loss, max_inflight=2, **session_options) as trainer:
        trainer.run(loader_batches(groups), updates=len(groups))
        for i, stage in enumerate(trainer.stages):
            for n, p in stage.named.items():
                assert_exact(p, expected.weights[f"{i}.{n}"], path="detached.weight")
                owner = stage.optimizer_named[n] if optimizer_device is not None else p
                assert_exact(stage.optimizer.state.get(owner, {}),
                             expected.optimizer_state[f"{i}.{n}"], path="detached.state")
                if optimizer_device is not None:
                    assert owner.device.type == "cpu"
                    assert_exact(owner, expected.weights[f"{i}.{n}"], path="detached.master")
    pipe.close()
    if cuda:
        # CPU-only smoke mode can receive CUDA data, but D2H must be complete
        # before the CPU forward/loss reads it.
        groups = make_batches(updates=3, microbatches=2, dim=4)
        model = make_model(dim=4, layers=1)
        expected = run_reference(model, groups, factory, device="cpu",
                                 optimizer_device=optimizer_device).updates[-1]
        stream = torch.cuda.Stream(device=cuda[0])
        def gpu_loader():
            for x, y in loader_batches(groups):
                with torch.cuda.device(cuda[0]), torch.cuda.stream(stream):
                    yield x.to(cuda[0]) + 0, y.to(cuda[0]) + 0
        pipe = Pipeline(stage_modules=[copy.deepcopy(model)], devices=["cpu"])
        with pipe.train_session(optimizer_factory=factory, n_microbatches=2,
                                loss_fn=F.mse_loss, **session_options) as trainer:
            trainer.run(gpu_loader(), updates=3)
        for n, p in pipe.stages[0].module.named_parameters():
            assert_exact(p, expected.weights[n], path="d2h.weight")
        pipe.close()
    print("PASS None-gradient producer fences, detached boundaries, and D2H inputs")


def check_schedule():
    for p in range(1, 7):
        for m in (p, p+1, 2*p+1):
            for groups in (1, 2, 9):
                for stage in range(p):
                    ops = list(_rank_ops(stage, p, m*groups, m))
                    latest, versions, live, back = 0, {0}, {}, 0
                    forwards = []
                    for kind, mb in ops:
                        version = max(mb//m-1, 0)
                        if kind == "F":
                            assert version in versions
                            live[mb] = version
                            forwards.append(mb)
                        elif kind == "B":
                            assert mb == back and live.pop(mb) == version
                            back += 1
                        else:
                            assert back == (latest+1)*m
                            latest += 1
                            versions = {max(latest-1, 0), latest}
                            assert set(live.values()) <= versions
                    assert not live and forwards == list(range(m*groups))
                    assert latest == groups
    print("PASS schedule versions/bootstrap/drain matrix")


def check_failures(device):
    def build():
        return Pipeline(stage_modules=[make_model(dim=4, layers=1)], devices=[device])
    for loader, loss in [([], F.mse_loss), ([(torch.ones(3,4), torch.ones(3,4))], F.mse_loss),
                         ([(torch.ones(2,4), torch.ones(2,4))], lambda x,y: (_ for _ in ()).throw(ValueError("loss failure")))]:
        pipe = build()
        with pipe.train_session(optimizer_factory=optimizer_factory(), n_microbatches=2, loss_fn=loss) as trainer:
            try:
                trainer.run(loader, updates=1)
            except RuntimeError:
                pass
            else:
                raise AssertionError("invalid run unexpectedly succeeded")
            try:
                trainer.run(loader, updates=1)
            except RuntimeError:
                pass
            else:
                raise AssertionError("poisoned trainer was reused")
        pipe.close()
    print("PASS loader/worker failure cleanup and poisoning")


def expect_error(kind, call, text=None):
    try:
        call()
    except kind as error:
        if text is not None:
            assert text in str(error), str(error)
        return error
    raise AssertionError(f"expected {kind.__name__}")


def bounded(call, timeout=30):
    """A broken worker/close must fail the check, not hang the test process."""
    result, errors = [], []
    def run():
        try:
            result.append(call())
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    thread.join(timeout)
    assert not thread.is_alive(), f"operation hung for {timeout}s"
    if errors:
        raise errors[0]
    return result[0]


def assert_drained(trainer, closed=False):
    assert not trainer._tickets and not trainer._retired
    for stage in trainer.stages:
        assert not stage.cache
        assert all(value is None for value in stage.acc.values())
        assert all(p.grad is None for p in stage.named.values())
        assert all(p.grad is None for p in getattr(stage, "optimizer_named", {}).values())
        assert not stage.stage._cache and not stage.stage.grad_acc
        if closed:
            assert not stage.banks
            assert not getattr(stage, "cpu_grads", {})
        else:
            assert stage.live == [0, 0]
    if closed:
        assert not any(worker.is_alive() for worker in trainer._workers)
        assert trainer.pipe._train_sess is None


class _UnusedFrozen(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.unused = nn.Parameter(torch.ones(4))
        self.frozen = nn.Parameter(torch.ones(4), requires_grad=False)
        self.zero = nn.Parameter(torch.ones(4))

    def forward(self, x):
        return self.linear(x) + self.frozen + self.zero * 0


def check_unused_frozen(device):
    groups = make_batches(updates=5, microbatches=2, dim=4)
    for name in ("sgd", "adamw"):
        model = _UnusedFrozen()
        factory = optimizer_factory(name, weight_decay=0.3)
        expected = run_reference(model, groups, factory, device=device)
        pipe = Pipeline(stage_modules=[copy.deepcopy(model)], devices=[device])
        records = []
        try:
            with pipe.train_session(optimizer_factory=factory, n_microbatches=2,
                                    loss_fn=F.mse_loss) as trainer:
                trainer.run(loader_batches(groups), updates=len(groups),
                            observer=lambda s, g, snap: records.append(snap))
                assert_drained(trainer)
            assert_drained(trainer, closed=True)
            for group, snapshot in enumerate(records):
                ref = expected.updates[group]
                assert_exact(snapshot, {"grads": ref.gradients, "weights": ref.weights,
                                        "optimizer": ref.optimizer_state})
                for parameter in ("unused", "frozen"):
                    assert snapshot["grads"][parameter] is None
                    assert snapshot["optimizer"][parameter] == {}
                    assert_exact(snapshot["weights"][parameter],
                                 expected.initial_weights[parameter])
                assert torch.count_nonzero(snapshot["grads"]["zero"]) == 0
                assert snapshot["optimizer"]["zero"]
                assert not torch.equal(snapshot["weights"]["zero"],
                                       expected.initial_weights["zero"])
            assert len(records) == len(groups)
        finally:
            pipe.close()
    print("PASS unused/frozen None gradients versus zero-gradient weight decay")


class _CPUOptimizerEdge(_UnusedFrozen):
    def forward(self, x):
        # Unlike zero * 0, cancellation always produces a +0 gradient, even
        # when every upstream microbatch gradient is negative. This avoids an
        # incidental -0 versus accumulator +0 without relaxing bytewise checks.
        return self.linear(x) + self.frozen + (self.zero - self.zero)


class _CPUOptimizerStorage:
    """Metadata-only hooks: never synchronize CUDA or copy values in async tests."""

    def __init__(self, stage):
        self.stage = stage
        self.pointers = {}
        self.calls = 0
        self._check()
        self.handles = [stage.optimizer.register_step_pre_hook(self._before),
                        stage.optimizer.register_step_post_hook(self._after)]

    def _storage(self, key, tensor):
        pointer = tensor.untyped_storage().data_ptr()
        assert self.pointers.setdefault(key, pointer) == pointer, f"storage churn: {key}"

    def _check(self):
        stage = self.stage
        assert type(stage.optimizer) is torch.optim.AdamW
        assert all(group["fused"] is True for group in stage.optimizer.param_groups)
        assert stage.named.keys() == stage.optimizer_named.keys()
        actual = [p for group in stage.optimizer.param_groups for p in group["params"]]
        assert len(actual) == len(stage.optimizer_named)
        assert {id(p) for p in actual} == {id(p) for p in stage.optimizer_named.values()}
        assert len(stage.banks) == 2
        for name, master in stage.optimizer_named.items():
            public = stage.named[name]
            assert master is not public
            assert master.device.type == "cpu"
            assert public.device == stage.stage.device
            assert master.dtype == public.dtype
            assert master.requires_grad == public.requires_grad
            assert public.grad is None
            assert dict(stage.module.named_parameters())[name] is public
            self._storage(("master", name), master)
            for slot, bank in enumerate(stage.banks):
                assert bank[name].device == stage.stage.device
                self._storage(("bank", slot, name), bank[name])
            assert self.pointers["bank", 0, name] != self.pointers["bank", 1, name]
            assert public.untyped_storage().data_ptr() in (
                self.pointers["bank", 0, name], self.pointers["bank", 1, name])
            if public.is_cuda:
                assert master.is_pinned(), f"unpinned CPU master: {name}"
            if master.grad is not None:
                assert master.grad.device.type == "cpu"
                assert master.grad.dtype == master.dtype
                if public.is_cuda:
                    assert master.grad.is_pinned(), f"unpinned CPU gradient: {name}"
                self._storage(("gradient", name), master.grad)
            for key, value in stage.optimizer.state.get(master, {}).items():
                if isinstance(value, torch.Tensor):
                    assert value.device.type == "cpu", f"non-CPU optimizer state: {name}.{key}"
                    self._storage(("state", name, key), value)

    def _before(self, optimizer, args, kwargs):
        self._check()
        assert any(p.grad is not None for p in self.stage.optimizer_named.values())

    def _after(self, optimizer, args, kwargs):
        self._check()
        self.calls += 1

    def close(self):
        for handle in self.handles:
            handle.remove()


def check_cpu_optimizer_banks(device, bf16=False):
    """An old-version graph survives CPU AdamW and upload into the OTHER bank."""
    model = _CPUOptimizerEdge()
    m, updates = 3, 7
    groups = make_batches(updates=updates, microbatches=m, dim=4)
    factory = optimizer_factory("adamw", fused=True, weight_decay=0.3)
    expected = run_reference(model, groups, factory, device=device,
                             optimizer_device="cpu", autocast_bf16=bf16)
    bank = _VersionedStage(Stage(copy.deepcopy(model), 0, 1, device=device,
                                autocast_dtype=torch.bfloat16 if bf16 else None),
                           factory, optimizer_device="cpu")
    storage = _CPUOptimizerStorage(bank)
    records = []
    def forward(group):
        for mb, (x, _) in enumerate(groups[group]):
            bank.forward(group*m+mb, x, m)
    try:
        forward(0)
        forward(1)
        for group in range(updates):
            if group > 1:
                forward(group)
            for mb, (_, target) in enumerate(groups[group]):
                bank.backward(group*m+mb, target=target.to(device), loss_fn=F.mse_loss)
            bank.update(group, m, lambda s, g, snap: records.append(snap))
            storage._check()
        assert storage.calls == updates
        for group, actual in enumerate(records):
            ref = expected.updates[group]
            assert_exact(actual, {"grads": ref.gradients, "weights": ref.weights,
                                  "optimizer": ref.optimizer_state}, path=f"cpu.bank.{group}")
        for name, master in bank.optimizer_named.items():
            assert_exact(master, expected.updates[-1].weights[name], path="cpu.master."+name)
    finally:
        storage.close()
        bank.release()
    assert not bank.banks and not bank.cache
    assert all(p.grad is None for p in bank.named.values())
    assert all(p.grad is None for p in bank.optimizer_named.values())
    print(f"PASS CPU-fused AdamW live banks/resident storage {device} bf16={bf16}")


def check_cpu_optimizer_pipeline(devices, *, bf16=False, edge_parameters=False,
                                 observed=True, capacity=None):
    """Exact CPU-FUSED oracle, including continuation and truly unobserved runs."""
    from collections import OrderedDict
    depth = len(devices)
    m, updates = depth + 1, 9
    dim = 4 if edge_parameters else 32
    model = (nn.Sequential(*(_CPUOptimizerEdge() for _ in devices)) if edge_parameters
             else make_model(dim=dim, layers=2*depth))
    groups = make_batches(updates=updates, microbatches=m, dim=dim, batch_size=4)
    factory = optimizer_factory("adamw", fused=True, weight_decay=0.3)
    reference = run_reference(model, groups, factory, device=devices[0],
                              optimizer_device="cpu", autocast_bf16=bf16)
    cuts = [len(model)*i//depth for i in range(depth+1)]
    modules = [nn.Sequential(OrderedDict((str(j), copy.deepcopy(model[j]))
               for j in range(cuts[i], cuts[i+1]))) for i in range(depth)]
    pipe = Pipeline(stage_modules=modules, devices=devices,
                    autocast=torch.bfloat16 if bf16 else None)
    seen = set()
    def observe(stage_index, group, snapshot):
        assert (stage_index, group) not in seen
        seen.add((stage_index, group))
        ref = reference.updates[group]
        names = dict(modules[stage_index].named_parameters())
        assert_exact(snapshot, {
            "grads": {n: ref.gradients[n] for n in names},
            "weights": {n: ref.weights[n] for n in names},
            "optimizer": {n: ref.optimizer_state[n] for n in names},
        }, path=f"cpu.stage{stage_index}.group{group}")
        if edge_parameters:
            for name in names:
                leaf = name.rsplit(".", 1)[-1]
                if leaf in ("unused", "frozen"):
                    assert snapshot["grads"][name] is None
                    assert snapshot["optimizer"][name] == {}
                    assert_exact(snapshot["weights"][name], reference.initial_weights[name])
                elif leaf == "zero":
                    assert torch.count_nonzero(snapshot["grads"][name]) == 0
                    assert snapshot["optimizer"][name]
                    assert not torch.equal(snapshot["weights"][name], reference.initial_weights[name])
    storages = []
    try:
        with pipe.train_session(optimizer_factory=factory, optimizer_device="cpu",
                                n_microbatches=m, loss_fn=F.mse_loss,
                                max_inflight=capacity) as trainer:
            storages = [_CPUOptimizerStorage(stage) for stage in trainer.stages]
            # The observed case drains after bootstrap and resumes twice. The
            # unobserved case has NO observer or value-reading hook at any update.
            parts = (groups[:1], groups[1:3], groups[3:]) if observed else (groups,)
            completed = 0
            for part in parts:
                result = trainer.run(loader_batches(part), updates=len(part),
                                     observer=observe if observed else None)
                completed += len(part)
                assert result.total_updates == completed
                assert result.peak_inflight <= trainer.max_inflight
                assert_drained(trainer)
                for storage in storages:
                    storage._check()
                    assert storage.calls == completed
            assert len(seen) == (depth*updates if observed else 0)
            final = reference.updates[-1]
            for stage in trainer.stages:
                for name, master in stage.optimizer_named.items():
                    assert_exact(master, final.weights[name], path="cpu.final.master."+name)
                    assert_exact(stage.named[name], final.weights[name], path="cpu.final.gpu."+name)
                    assert_exact(stage.optimizer.state.get(master, {}), final.optimizer_state[name],
                                 path="cpu.final.optimizer."+name)
            for storage in storages:
                storage.close()
        assert_drained(trainer, closed=True)
        for module in modules:
            for name, parameter in module.named_parameters():
                assert_exact(parameter, final.weights[name], path="cpu.closed."+name)
        # CPU masters must not keep a graph, gradient, or worker alive on close.
        for stage in trainer.stages:
            assert all(p.grad is None for p in stage.optimizer_named.values())
        pipe.close()
        pipe.close()
    finally:
        for storage in storages:
            storage.close()
        pipe.close()
    print(f"PASS CPU-fused AdamW {devices} bf16={bf16} edge={edge_parameters} "
          f"observed={observed} capacity={capacity} updates={updates}")


def check_cpu_optimizer_suite(devices):
    """Small focused suite; --devices cuda:0,cuda:1,cuda:2,cuda:3 covers four GPUs."""
    for bf16 in (False, True):
        bounded(lambda: check_cpu_optimizer_banks(devices[0], bf16), timeout=120)
        for edge_parameters in (False, True):
            bounded(lambda: check_cpu_optimizer_pipeline(
                devices, bf16=bf16, edge_parameters=edge_parameters), timeout=120)
        for capacity in (len(devices), 3*len(devices)+2):
            bounded(lambda: check_cpu_optimizer_pipeline(
                devices, bf16=bf16, observed=False, capacity=capacity), timeout=120)
    check_extended_failures(devices, optimizer_device="cpu", fused=True)
    bounded(lambda: check_empty_gradient_fences(devices, optimizer_device="cpu", fused=True),
            timeout=120)
    print("ALL CPU-FUSED OPTIMIZER EXACT CHECKS PASSED (fp32 AND bf16)")


def _synchronous_cpu_reference(model, groups, factory, device, bf16):
    """Ordinary latest-weight CPU AdamW, with a separately copied evaluator.

    This deliberately does not call run_reference: that oracle's one-update
    delay is correct for 2BW and would be wrong for the synchronous baseline.
    """
    device = torch.device(device)
    latest = copy.deepcopy(model).cpu()
    evaluation = copy.deepcopy(model).to(device)
    latest_named = dict(latest.named_parameters())
    evaluation_named = dict(evaluation.named_parameters())
    optimizer = factory(latest.parameters())
    records = []
    for group in groups:
        # Copy CURRENT weights on every update, never a historical snapshot.
        evaluation.load_state_dict({name: value.detach().clone()
                                    for name, value in latest.state_dict().items()})
        optimizer.zero_grad(set_to_none=True)
        accumulated = {name: torch.zeros_like(parameter)
                       for name, parameter in evaluation_named.items()}
        for x, y in group:  # Ascending accumulation, like Pipeline.step.
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=bf16):
                output = evaluation(x.to(device).detach().requires_grad_(True))
                loss = F.mse_loss(output, y.to(device))
            gradients = torch.autograd.grad(loss, tuple(evaluation_named.values()))
            for name, gradient in zip(evaluation_named, gradients):
                accumulated[name].add_(gradient.detach())
        for name, gradient in accumulated.items():
            gradient.mul_(1.0 / len(group))  # Exactly one mean scaling.
            latest_named[name].grad = gradient.to("cpu", copy=True)
        optimizer.step()
        records.append({
            "weights": {name: parameter.detach().clone()
                        for name, parameter in latest_named.items()},
            "optimizer": {name: copy.deepcopy(optimizer.state.get(parameter, {}))
                          for name, parameter in latest_named.items()},
        })
    return records


def check_synchronous_cpu_optimizer(devices, *, bf16=False):
    """Compare drained synchronous runs, including multiple updates per run."""
    from collections import OrderedDict
    from pipedream_2bw_demo import _SynchronousCPUOptimizer, run_synchronous

    depth = len(devices)
    dim, layers, updates, m = 16, 2 * depth, 5, depth + 1
    model = make_model(dim=dim, layers=layers)
    groups = make_batches(updates=updates, microbatches=m, dim=dim)
    factory = optimizer_factory("adamw", fused=True)
    reference = _synchronous_cpu_reference(model, groups, factory, devices[0], bf16)
    modules = [nn.Sequential(OrderedDict((str(j), copy.deepcopy(model[j]))
               for j in range(2 * stage, 2 * stage + 2))) for stage in range(depth)]
    pipe = Pipeline(stage_modules=modules, devices=devices, offload=False,
                    overlap=True, autocast=torch.bfloat16 if bf16 else None)
    pointers = {}
    handles = []
    calls = [0] * depth

    def storage(key, tensor):
        pointer = tensor.untyped_storage().data_ptr()
        assert pointers.setdefault(key, pointer) == pointer, f"storage churn: {key}"

    def cpu_state(value):
        if isinstance(value, torch.Tensor):
            assert value.device.type == "cpu", "synchronous optimizer state left CPU"
        elif isinstance(value, dict):
            for item in value.values():
                cpu_state(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                cpu_state(item)

    def check_storage(adapter, stage_index):
        stage = pipe.stages[stage_index]
        assert type(adapter.optimizer) is torch.optim.AdamW
        assert all(group["fused"] is True and group["foreach"] is False
                   for group in adapter.optimizer.param_groups)
        assert len(adapter.banks) == 1, "synchronous baseline allocated a second bank"
        assert adapter.banks[0] is adapter.named, "weight bank must be the live parameters"
        public_named = dict(stage.module.named_parameters())
        assert adapter.named.keys() == adapter.optimizer_named.keys() == public_named.keys()
        if adapter.optimizer.state:
            assert adapter.cpu_grads.keys() == adapter.optimizer_named.keys()
        else:
            assert adapter.cpu_grads.keys() <= adapter.optimizer_named.keys()
        owned = [p for group in adapter.optimizer.param_groups for p in group["params"]]
        assert len(owned) == len(adapter.optimizer_named)
        assert {id(p) for p in owned} == {id(p) for p in adapter.optimizer_named.values()}
        for name, master in adapter.optimizer_named.items():
            public = adapter.named[name]
            assert public is public_named[name]
            assert public.device == stage.device
            assert master is not public and master.device.type == "cpu"
            assert master.dtype == public.dtype and master.shape == public.shape
            assert master.requires_grad == public.requires_grad
            if public.is_cuda:
                assert master.is_pinned(), name
            for category, tensor in (("master", master), ("bank", public)):
                storage((stage_index, category, name), tensor)
            assert master.untyped_storage().data_ptr() != public.untyped_storage().data_ptr()
            gradient = adapter.cpu_grads.get(name)  # Allocated lazily on the first update.
            if gradient is not None:
                assert gradient.device.type == "cpu"
                assert gradient.dtype == master.dtype and gradient.shape == master.shape
                if public.is_cuda:
                    assert gradient.is_pinned(), name
                storage((stage_index, "grad", name), gradient)
            if master.grad is not None:
                assert master.grad is gradient, "optimizer must use its retained CPU gradient"
        cpu_state(adapter.optimizer.state)

    def after_step(adapter, stage_index):
        def observe(optimizer, args, kwargs):
            # Metadata-only checks: do not synchronize/copy CUDA tensors here.
            check_storage(adapter, stage_index)
            calls[stage_index] += 1
        return observe

    try:
        adapters = [_SynchronousCPUOptimizer(stage, factory) for stage in pipe.stages]
        for stage_index, adapter in enumerate(adapters):
            check_storage(adapter, stage_index)
            handles.append(adapter.optimizer.register_step_post_hook(
                after_step(adapter, stage_index)))
        completed = 0
        # Bootstrap, resume twice, and test barriers between successive updates
        # INSIDE a run as well as across calls. No value-reading optimizer hooks.
        for part in (groups[:1], groups[1:3], groups[3:]):
            run_synchronous(pipe, adapters, loader_batches(part), len(part), m)
            completed += len(part)
            expected = reference[completed - 1]
            assert calls == [completed] * depth
            for stage_index, adapter in enumerate(adapters):
                check_storage(adapter, stage_index)
                assert not pipe.stages[stage_index]._cache, "run retained an autograd graph"
                assert all(p.grad is None for p in adapter.named.values())
                assert all(p.grad is None for p in adapter.optimizer_named.values())
                for name, master in adapter.optimizer_named.items():
                    prefix = f"sync.cpu.update{completed}.stage{stage_index}.{name}"
                    assert_exact(master, expected["weights"][name], path=prefix + ".master")
                    assert_exact(adapter.named[name], expected["weights"][name],
                                 path=prefix + ".resident")
                    assert_exact(adapter.optimizer.state.get(master, {}),
                                 expected["optimizer"][name], path=prefix + ".optimizer")
            print(f"BIT-EXACT synchronous CPU AdamW {devices} bf16={bf16} "
                  f"update={completed}/{updates}: masters + state + resident weights", flush=True)
    finally:
        for handle in handles:
            handle.remove()
        pipe.close()
    print(f"PASS synchronous CPU-fused AdamW {devices} bf16={bf16} "
          f"dim={dim} layers={layers} m={m} updates={updates} continuation=1+2+2 single-bank")


def check_synchronous_cpu_optimizer_suite(devices):
    """Use every supplied stage/device; --devices cpu,cpu also tests the relay."""
    for bf16 in (False, True):
        bounded(lambda: check_synchronous_cpu_optimizer(devices, bf16=bf16), timeout=120)
    print("ALL SYNCHRONOUS CPU-FUSED OPTIMIZER EXACT CHECKS PASSED (fp32 AND bf16)")


class _TupleStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, x, mask, ids):
        assert mask.dtype == torch.bool and ids.dtype == torch.int64
        return self.linear(x) * mask[:, None] + ids[:, None].to(x.dtype) * 0.01, mask, ids


class _PackedTupleModel(nn.Module):
    """Let the tensor-input oracle exercise exactly the same tuple modules."""
    def __init__(self, depth):
        super().__init__()
        self.stages = nn.ModuleList([_TupleStage() for _ in range(depth)])

    def forward(self, packed):
        value = (packed[:, :4], packed[:, 4].bool(), packed[:, 5].long())
        for stage in self.stages:
            value = stage(*value)
        return value


def check_tuples(devices):
    # Exercise a real tuple relay even for the default one-device CPU command.
    devices = devices if len(devices) > 1 else devices * 2
    m = len(devices)
    model = _PackedTupleModel(m)
    groups = make_batches(updates=5, microbatches=m, dim=4)
    packed_groups = []
    for group in groups:
        packed_groups.append([(torch.cat((x, torch.tensor([[1., 2.], [0., 3.]])), dim=1), y)
                              for x, y in group])
    def loss(output, target):
        x, mask, ids = output
        assert mask.dtype == torch.bool and ids.dtype == torch.int64
        if isinstance(target, tuple):
            target, target_mask, target_ids = target
            assert torch.equal(mask, target_mask) and torch.equal(ids, target_ids)
        return F.mse_loss(x, target)
    factory = optimizer_factory("adamw")
    expected = run_reference(model, packed_groups, factory, device=devices[0], loss_fn=loss)
    modules = [copy.deepcopy(stage) for stage in model.stages]
    pipe = Pipeline(stage_modules=modules, devices=devices)
    def source():
        for packed, target in loader_batches(packed_groups):
            mask, ids = packed[:, 4].bool(), packed[:, 5].long()
            yield (packed[:, :4], mask, ids), (target, mask, ids)
    observed = set()
    def observe(stage, group, snapshot):
        assert (stage, group) not in observed
        observed.add((stage, group))
        ref = expected.updates[group]
        want = {}
        for key, values in (("grads", ref.gradients), ("weights", ref.weights),
                            ("optimizer", ref.optimizer_state)):
            want[key] = {name: values[f"stages.{stage}.{name}"]
                         for name, _ in modules[stage].named_parameters()}
        assert_exact(snapshot, want, path=f"tuple.stage{stage}.group{group}")
    try:
        with pipe.train_session(optimizer_factory=factory, n_microbatches=m,
                                loss_fn=loss, max_inflight=m) as trainer:
            trainer.run(source(), updates=len(groups), observer=observe)
            assert_drained(trainer)
        assert len(observed) == m * len(groups)
    finally:
        pipe.close()
    print("PASS tuple inputs/intermediates/targets with bool masks and integer ids")


def check_invalid_configs(device):
    def reject(label, modules=None, pipe_options=None, **options):
        modules = [nn.Linear(4, 4)] if modules is None else modules
        pipe = Pipeline(stage_modules=modules, devices=[device] * len(modules),
                        **(pipe_options or {}))
        kwargs = dict(optimizer_factory=optimizer_factory(),
                      n_microbatches=len(modules), loss_fn=F.mse_loss)
        kwargs.update(options)
        try:
            expect_error(ValueError, lambda: pipe.train_session(**kwargs))
            assert pipe._train_sess is None, label
        finally:
            pipe.close()
    reject("offload", [[nn.Linear(4, 4)]], {"offload": True})
    reject("m < p", [nn.Linear(4, 4), nn.Linear(4, 4)], n_microbatches=1)
    for value in (0, -1, 1.5, True):
        reject("invalid m", n_microbatches=value)
        reject("invalid inflight", max_inflight=value)
    reject("inflight < p", [nn.Linear(4, 4), nn.Linear(4, 4)], max_inflight=1)
    buffered = nn.Linear(4, 4)
    buffered.register_buffer("counter", torch.zeros(()))
    reject("buffer", [buffered])
    reject("dropout", [nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.1))])
    reject("forward weight mutation", [nn.Embedding(8, 4, max_norm=1.)])
    reject("embeddingbag mutation", [nn.EmbeddingBag(8, 4, max_norm=1.)])
    reject("attention internal dropout", [nn.MultiheadAttention(4, 1, dropout=0.1)])
    reject("RNN internal dropout", [nn.LSTM(4, 4, num_layers=2, dropout=0.1)])
    if torch.device(device).type == "cuda":
        mixed = Pipeline(stage_modules=[nn.Linear(4, 4), nn.Linear(4, 4)],
                         devices=[device, "cpu"])
        try:
            expect_error(ValueError, lambda: mixed.train_session(
                optimizer_factory=optimizer_factory(), n_microbatches=2,
                loss_fn=F.mse_loss), "mixed devices")
        finally:
            mixed.close()
    tied = nn.Linear(4, 4)
    reject("tied within stage", [nn.Sequential(tied, tied)])
    shared = nn.Linear(4, 4)
    reject("shared across stages", [shared, shared])
    frozen = nn.Linear(4, 4).requires_grad_(False)
    reject("all frozen", [frozen])
    reject("unsupported schedule", schedule="1f1b")
    reject("noncallable loss", loss_fn=None)
    reject("fp16", pipe_options={"autocast": torch.float16})
    class CustomSGD(torch.optim.SGD):
        pass
    factories = [
        [], [optimizer_factory(), optimizer_factory()],
        lambda ps: object(),
        lambda ps: torch.optim.RMSprop(ps, lr=0.01),
        lambda ps: CustomSGD(ps, lr=0.01),
        lambda ps: torch.optim.SGD(ps[:-1], lr=0.01),
        lambda ps: torch.optim.SGD(ps + [nn.Parameter(torch.ones(1, device=device))], lr=0.01),
        lambda ps: torch.optim.SGD([nn.Parameter(p.detach().clone()) for p in ps], lr=0.01),
        lambda ps: torch.optim.SGD(ps, lr=0.01, differentiable=True),
        lambda ps: torch.optim.AdamW(ps, lr=0.01, capturable=True),
    ]
    for factory in factories:
        reject("bad optimizer factory", optimizer_factory=factory)
    print("PASS invalid stages/configuration/optimizer factory restrictions")


def check_extended_failures(devices, optimizer_device=None, fused=False):
    session_options = {} if optimizer_device is None else {"optimizer_device": optimizer_device}
    depth = len(devices)
    m = max(2, depth)
    groups = make_batches(updates=3, microbatches=m, dim=4)
    good = list(loader_batches(groups))
    marker = ValueError("injected loader failure")
    def broken_loader():
        yield from good
        raise marker
    cases = [
        ("scalar inputs", [(torch.tensor(1.), good[0][1])], 1, None),
        ("scalar targets", [(good[0][0], torch.tensor(1.))], 1, None),
        ("scalar tuple member", [((good[0][0], torch.tensor(True)), good[0][1])], 1, None),
        ("loader after valid groups", broken_loader(), 4, marker),
        ("optimizer step hook", good, 3, ValueError("injected optimizer failure")),
    ]
    for label, source, updates, cause in cases:
        def check_case():
            pipe = Pipeline(stage_modules=[nn.Linear(4, 4) for _ in devices], devices=devices)
            trainer = pipe.train_session(optimizer_factory=optimizer_factory("adamw", fused=fused),
                                         n_microbatches=m, loss_fn=F.mse_loss,
                                         max_inflight=depth, **session_options)
            hook = None
            observed = []
            try:
                if label == "optimizer step hook":
                    def fail_step(*args, **kwargs):
                        raise cause
                    hook = trainer.optimizers[-1].register_step_pre_hook(fail_step)
                error = expect_error(RuntimeError, lambda: trainer.run(source, updates=updates,
                                     observer=lambda s, g, snap: observed.append((s, g))))
                if cause is not None:
                    while error is not cause and error.__cause__ is not None:
                        error = error.__cause__
                    assert error is cause, label
                if label == "loader after valid groups":
                    assert observed, "loader failed before any optimizer update"
                expect_error(RuntimeError, lambda: trainer.run(good, updates=1))
                assert trainer._error is None
                assert trainer._poison.__traceback__ is None
                assert trainer._poison.__cause__ is None
                assert_drained(trainer)
            finally:
                if hook is not None:
                    hook.remove()
                pipe.close()
                pipe.close()
            assert_drained(trainer, closed=True)
        bounded(check_case)
    print("PASS scalar batches, late loader errors, optimizer-hook failure and cleanup")


def check_session_lifecycle(device):
    pipe = Pipeline(stage_modules=[nn.Linear(4, 4)], devices=[device])
    kwargs = dict(optimizer_factory=optimizer_factory(), n_microbatches=1, loss_fn=F.mse_loss)
    trainer = pipe.train_session(**kwargs)
    entered, release = threading.Event(), threading.Event()
    errors = []
    batch = (torch.ones(2, 4), torch.zeros(2, 4))
    def slow_source():
        entered.set()
        assert release.wait(10), "test did not release loader"
        yield batch
    def run():
        try:
            trainer.run(slow_source(), updates=1)
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=run, daemon=True)
    try:
        # Rejected arguments must not poison an otherwise healthy session.
        for updates in (0, -1, True, 1.5):
            expect_error(ValueError, lambda: trainer.run([batch], updates=updates))
        expect_error(RuntimeError, lambda: pipe.train_session(**kwargs))
        thread.start()
        assert entered.wait(10), "run did not reach loader"
        try:
            bounded(lambda: expect_error(RuntimeError, lambda: trainer.run([batch], updates=1)))
            bounded(lambda: expect_error(RuntimeError, trainer.close))
            bounded(lambda: expect_error(RuntimeError, pipe.close))
        finally:
            release.set()
            thread.join(30)
        assert not thread.is_alive(), "active run did not drain"
        if errors:
            raise errors[0]
        assert trainer.run([batch], updates=1).total_updates == 2
        assert_drained(trainer)
        bounded(trainer.close)
        trainer.close()
        expect_error(RuntimeError, lambda: trainer.run([batch], updates=1))
        assert_drained(trainer, closed=True)
        with pipe.train_session(**kwargs) as fresh:
            assert fresh.run([batch], updates=1).total_updates == 1
        assert_drained(fresh, closed=True)
    finally:
        release.set()
        bounded(pipe.close)
    print("PASS concurrent run/close rejection, continued runs, idempotent close and reuse")


def check_memory_lifecycle(devices):
    from ramtorch import pipeline_2bw
    depth = len(devices)
    pipe = Pipeline(stage_modules=[nn.Linear(4, 4) for _ in devices], devices=devices)
    tickets = weakref.WeakSet()
    original_ticket = pipeline_2bw._Ticket
    def track_ticket(*args, **kwargs):
        # Admission accounting must be bounded independently of the returned peak.
        assert len(trainer._tickets) < trainer.max_inflight
        ticket = original_ticket(*args, **kwargs)
        tickets.add(ticket)
        return ticket
    def source():
        for _ in range(24):
            time.sleep(0.001)
            yield torch.ones(depth * 2, 4), torch.zeros(depth * 2, 4)
    try:
        with pipe.train_session(optimizer_factory=optimizer_factory("adamw"),
                                n_microbatches=depth, loss_fn=F.mse_loss,
                                max_inflight=depth) as trainer:
            with mock.patch.object(pipeline_2bw, "_Ticket", track_ticket):
                for _ in range(2):
                    result = bounded(lambda: trainer.run(source(), updates=24))
                    assert result.peak_inflight <= depth
                    assert_drained(trainer)
                    gc.collect()
                    assert not tickets, "completed run retained input/target/relay tickets"
            assert result.total_updates == 48
        assert_drained(trainer, closed=True)
    finally:
        bounded(pipe.close)
    print("PASS minimum-inflight slow loader, bounded tickets and graph-cache lifetime")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="cpu")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--bank-only", action="store_true")
    modes.add_argument("--cpu-optimizer-only", action="store_true",
                       help="focused CPU-fused AdamW suite: fp32 AND bf16, exact CPU oracle")
    modes.add_argument("--sync-cpu-optimizer-only", action="store_true",
                       help="synchronous CPU-fused AdamW: fp32 AND bf16, fresh-weight CPU oracle")
    parser.add_argument("--optimizer-device", choices=("cpu",), default=None,
                        help="long-run: CPU optimizer instead of the default stage device")
    parser.add_argument("--fused", action="store_true", help="long-run: opt into fused optimizer math")
    modes.add_argument("--long-run", action="store_true",
                       help="bounded per-update comparison against the single-device oracle")
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--optimizer", choices=("sgd", "adamw"), default="adamw")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--report", type=Path, help="write long-run results as JSON")
    args = parser.parse_args()
    configure_determinism()
    devices = [d.strip() for d in args.devices.split(",")]
    if (args.optimizer_device is not None or args.fused) and not args.long_run:
        parser.error("--optimizer-device/--fused are long-run options; --cpu-optimizer-only sets both")
    if args.cpu_optimizer_only:
        check_cpu_optimizer_suite(devices)
        return
    if args.sync_cpu_optimizer_only:
        check_synchronous_cpu_optimizer_suite(devices)
        return
    if args.long_run:
        check_long_run(devices, updates=args.updates, dim=args.dim, layers=args.layers,
                       microbatches=args.microbatches, batch_size=args.batch_size,
                       optimizer=args.optimizer, bf16=args.bf16, lr=args.lr,
                       report_path=args.report, optimizer_device=args.optimizer_device,
                       fused=args.fused)
        return
    for optimizer in ("sgd", "adamw"):
        check_banks(devices[0], optimizer)
    if args.bank_only:
        return
    check_schedule()
    p = len(devices)
    check_pipeline(devices, p, 1, "sgd")
    check_pipeline(devices, p, 2, "adamw")
    check_pipeline(devices, p+1, 7, "adamw", split_run=True, slow=True, max_inflight=p)
    check_pipeline(devices, 2*p+1, 12, "sgd", layers=p*2+1)
    check_pipeline(devices, p, 4, "adamw", bf16=True)
    check_failures(devices[0])
    bounded(lambda: check_unused_frozen(devices[0]))
    bounded(lambda: check_tuples(devices))
    bounded(lambda: check_invalid_configs(devices[0]))
    check_extended_failures(devices)
    check_session_lifecycle(devices[0])
    check_memory_lifecycle(devices)
    check_unobserved(devices)
    check_empty_gradient_fences(devices)
    print("ALL EXACT CHECKS PASSED")


if __name__ == "__main__":
    main()
