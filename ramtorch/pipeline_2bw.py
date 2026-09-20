"""Resident PipeDream-2BW: two weight banks and stage-local optimizer updates.

This is deliberately separate from the flush-per-batch Pipeline.step executor.
Only deterministic, buffer-free resident modules and the explicit optimizer
allowlist (torch.optim.SGD/Adam/AdamW and ramtorch.AdamEF/Lion/Muon) are supported.
Optimizer masters/state can optionally live and update on CPU; both compute
weight banks and all forward/backward work remain on their stage devices.
"""
from __future__ import annotations

import contextlib
import collections
import queue
import threading
from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn

from .delayed_optim import AdamEF, Lion, Muon

__all__ = ["PipeDream2BWTrainer"]


def _rank_ops(stage, depth, microbatches, accumulation, start=0):
    """Lazy continuous staggered 1F1B, with a local update after each group."""
    warmup = min(depth - 1 - stage, microbatches)
    for mb in range(warmup):
        yield "F", start + mb
    forward = warmup
    for backward in range(microbatches):
        if forward < microbatches:
            yield "F", start + forward
            forward += 1
        yield "B", start + backward
        if (backward + 1) % accumulation == 0:
            yield "U", start + backward


def _snapshot(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: _snapshot(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_snapshot(v) for v in value)
    return value


class _VersionedStage:
    """Two storages per parameter; optimizer handles never enter autograd graphs.

    ``Parameter.data`` yields a storage alias with an independent version counter.
    Graphs use bank leaves, NOT the stable optimizer Parameters. Rebinding those
    Parameters therefore cannot invalidate a graph on the bank being preserved.
    All bank mutations and backward reads must run on the same owning stream.
    """

    def __init__(self, stage, optimizer_factory: Callable, optimizer_device=None):
        if optimizer_device is not None and torch.device(optimizer_device) != torch.device("cpu"):
            raise ValueError("optimizer_device must be None (stage device) or 'cpu'")
        self.cpu_optimizer = optimizer_device is not None
        self.stage = stage
        self.module = stage.module
        self.named = dict(self.module.named_parameters())
        if not any(p.requires_grad for p in self.named.values()):
            raise ValueError("2BW stages must contain trainable parameters")
        self.banks = [
            {n: p.data.requires_grad_(p.requires_grad) for n, p in self.named.items()},
            {n: p.detach().clone().requires_grad_(p.requires_grad)
             for n, p in self.named.items()},
        ]
        self.versions = [0, None]
        self.latest = 0
        self.live = [0, 0]
        self.cache = {}
        self.acc = {n: None for n, p in self.named.items() if p.requires_grad}
        # CPU masters are the authoritative latest weights. The two GPU banks
        # still serve forward/backward, including the one-update-stale graph.
        # Only masters and gradient transfer buffers need to be pinned; moments
        # never leave the CPU and are allocated by the ordinary optimizer.
        self.optimizer_named = self.named
        self.cpu_grads = {}
        if self.cpu_optimizer:
            self.optimizer_named = {}
            for name, parameter in self.named.items():
                master = torch.empty_like(parameter, device="cpu",
                                          pin_memory=stage.device.type == "cuda")
                master.copy_(parameter.detach())  # Blocking initial D2H.
                self.optimizer_named[name] = nn.Parameter(master, requires_grad=parameter.requires_grad)
        self.optimizer = optimizer_factory(list(self.optimizer_named.values()))
        if type(self.optimizer) not in (torch.optim.SGD, torch.optim.Adam,
                                       torch.optim.AdamW, AdamEF, Lion, Muon):
            raise ValueError("2BW supports only torch.optim.SGD/Adam/AdamW and ramtorch.AdamEF/Lion/Muon")
        actual = [p for group in self.optimizer.param_groups for p in group["params"]]
        if len(actual) != len(self.optimizer_named) or {id(p) for p in actual} != {
            id(p) for p in self.optimizer_named.values()
        }:
            raise ValueError("optimizer factory must own exactly this stage's parameters")
        if any(g.get("differentiable", False) or g.get("capturable", False)
               for g in self.optimizer.param_groups):
            raise ValueError("differentiable/capturable optimizers are unsupported")
        self.backward_count = 0

    def slot(self, version):
        if version not in self.versions:
            raise RuntimeError(f"weight version {version} unavailable: {self.versions}")
        return self.versions.index(version)

    def forward(self, mb, value, accumulation):
        version = max(mb // accumulation - 1, 0)
        slot = self.slot(version)
        is_tuple = isinstance(value, (tuple, list))
        values = tuple(value) if is_tuple else (value,)
        inputs = tuple(t.to(self.stage.device, non_blocking=True).detach()
                       .requires_grad_(t.is_floating_point()) for t in values)
        leaves = self.banks[slot]
        # Disable the AMP weight cache: storage/version rebinding must not reuse
        # a cast belonging to another bank/version.
        with self.autocast():
            out = torch.func.functional_call(self.module, leaves, inputs, strict=True)
        outputs = tuple(out) if isinstance(out, (tuple, list)) else (out,)
        no_grad = frozenset(getattr(self.module, "out_no_grad", ()) or ())
        mask = tuple(o.is_floating_point() and i not in no_grad
                     for i, o in enumerate(outputs))
        self.cache[mb] = (inputs, outputs, mask, slot, is_tuple)
        self.live[slot] += 1
        return out

    def autocast(self):
        if self.stage.autocast_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.stage.device.type,
                              dtype=self.stage.autocast_dtype, cache_enabled=False)

    def backward(self, mb, grad_output=None, target=None, loss_fn=None):
        inputs, outputs, mask, slot, is_tuple = self.cache.pop(mb)
        names = list(self.acc)
        leaves = [self.banks[slot][n] for n in names]
        required = [x for x in inputs if x.requires_grad]
        loss = None
        if self.stage.is_last:
            out = outputs[0] if len(outputs) == 1 else outputs
            with self.autocast():
                loss = loss_fn(out, target)
            if loss.ndim != 0:
                raise ValueError("loss_fn must return a scalar mean loss")
            grads = torch.autograd.grad(loss, required + leaves, allow_unused=True)
        else:
            incoming = tuple(grad_output) if isinstance(grad_output, (tuple, list)) else (grad_output,)
            if len(incoming) != len(outputs):
                raise ValueError("gradient/output arity mismatch")
            outs, seeds = [], []
            for out, needed, grad in zip(outputs, mask, incoming):
                if needed and grad is not None and out.requires_grad:
                    outs.append(out)
                    seeds.append(grad.to(self.stage.device, non_blocking=True))
            grads = (torch.autograd.grad(outs, required + leaves, grad_outputs=seeds,
                                         allow_unused=True) if outs else
                     (None,) * (len(required) + len(leaves)))
        for name, grad in zip(names, grads[len(required):]):
            if grad is not None:
                if self.acc[name] is None:
                    self.acc[name] = torch.zeros_like(grad)
                self.acc[name].add_(grad.detach())
        it = iter(grads[:len(required)])
        input_grads = tuple(next(it) if x.requires_grad else None for x in inputs)
        self.live[slot] -= 1
        self.backward_count += 1
        return (input_grads if is_tuple else input_grads[0]), loss

    @torch.no_grad()
    def update(self, group, accumulation, observer=None, trace=None):
        if group != self.latest or self.backward_count != (group + 1) * accumulation:
            raise RuntimeError("out-of-order 2BW update")
        source = self.slot(self.latest)
        destination = 1 - source
        if self.live[destination]:
            raise RuntimeError("attempted overwrite of a bank with live graphs")
        # Bank leaves have their own counters; mutate only the retired bank.
        if self.cpu_optimizer:
            self._cpu_update(group, accumulation, destination, trace)
        else:
            with (trace.span("weight_copy", stage=self.stage.stage_index, group=group,
                             source_slot=source, destination_slot=destination,
                             source_version=self.latest, new_version=self.latest+1)
                  if trace is not None else contextlib.nullcontext()):
                for name, parameter in self.named.items():
                    self.banks[destination][name].copy_(self.banks[source][name])
                    parameter.data = self.banks[destination][name]
                    acc = self.acc.get(name)
                    parameter.grad = None if acc is None else acc.mul_(1.0 / accumulation)
            with (trace.span("optimizer", stage=self.stage.stage_index, group=group)
                  if trace is not None else contextlib.nullcontext()):
                self.optimizer.step()
        self.latest += 1
        self.versions[destination] = self.latest
        if observer is not None:
            # Explicit debugging path, intentionally synchronous and not used in
            # timings. Observer sees detached immutable CPU copies.
            observer(self.stage.stage_index, group, {
                "grads": {n: _snapshot(p.grad) for n, p in self.optimizer_named.items()},
                "weights": {n: _snapshot(p) for n, p in self.named.items()},
                "optimizer": {n: _snapshot(self.optimizer.state.get(p, {}))
                              for n, p in self.optimizer_named.items()},
            })
        self.optimizer.zero_grad(set_to_none=True)
        for name in self.acc:
            self.acc[name] = None

    @torch.no_grad()
    def _cpu_update(self, group, accumulation, destination, trace):
        def span(name, **args):
            return (trace.span(name, stage=self.stage.stage_index, group=group, **args)
                    if trace is not None else contextlib.nullcontext())

        cuda = self.stage.device.type == "cuda"
        gradient_bytes = sum(g.numel() * g.element_size() for g in self.acc.values() if g is not None)
        with span("optimizer_d2h", bytes=gradient_bytes):
            for name, master in self.optimizer_named.items():
                acc = self.acc.get(name)
                if acc is None:
                    master.grad = None  # Unused/frozen is NOT a zero gradient.
                    continue
                if name not in self.cpu_grads:
                    self.cpu_grads[name] = torch.empty_like(master, device="cpu", pin_memory=cuda)
                acc.mul_(1.0 / accumulation)
                self.cpu_grads[name].copy_(acc, non_blocking=cuda)
                master.grad = self.cpu_grads[name]
            if cuda:
                # Recorded even for all-None gradients: it also fences the last
                # upload's read of CPU masters before this step can mutate them.
                ready = torch.cuda.Event()
                ready.record(torch.cuda.current_stream(self.stage.device))
        with span("optimizer_d2h_wait"):
            if cuda:
                ready.synchronize()  # Stage-local only, never a device barrier.
        with span("optimizer_cpu", fused=bool(self.optimizer.defaults.get("fused", False))):
            self.optimizer.step()
        with span("optimizer_h2d", destination_slot=destination,
                  new_version=self.latest+1,
                  bytes=sum(p.numel() * p.element_size() for p in self.optimizer_named.values())):
            for name, parameter in self.named.items():
                self.banks[destination][name].copy_(self.optimizer_named[name], non_blocking=cuda)
                parameter.data = self.banks[destination][name]
            # Subsequent F/B and the next D2H fence use this same stage stream.
            # Pinned masters remain owned (and unmodified) until that fence.

    def release(self):
        """Leave public module Parameters on the latest bank, clear graphs."""
        slot = self.slot(self.latest)
        for name, parameter in self.named.items():
            parameter.data = self.banks[slot][name]
            parameter.grad = None
        self.cache.clear()
        self.acc.clear()
        self.banks.clear()
        self.cpu_grads.clear()
        self.optimizer.zero_grad(set_to_none=True)


@dataclass
class TrainRunResult:
    """Small run summary; no per-microbatch GPU outputs are retained."""
    updates: int
    total_updates: int
    peak_inflight: int


class _Ticket:
    def __init__(self, depth, inputs, targets):
        from .pipeline_relay import _Mailbox
        self.inputs, self.targets = _Mailbox(), _Mailbox()
        self.inputs.put_from_any_device(inputs)
        self.targets.put_from_any_device(targets)
        self.forward = [_Mailbox() for _ in range(depth - 1)]
        self.backward = [_Mailbox() for _ in range(depth - 1)]

    def release(self):
        for box in [self.inputs, self.targets, *self.forward, *self.backward]:
            box.release()


class PipeDream2BWTrainer:
    """Own a resident pipeline for continuous one-update-stale training.

    ``run(loader, updates=N)`` consumes N full ``(inputs, targets)`` batches,
    each split into ``n_microbatches`` equally sized pieces. Workers update
    locally without a group barrier. Each run drains, but version/optimizer
    history survives into the next run. Close before ordinary Pipeline calls.

    ``observer(stage, group, snapshot)`` is a synchronous correctness/debug hook,
    NOT a performance path. ``profile_path`` exports a compressed Kineto trace.
    External mutation of module parameters/optimizers during ownership is forbidden.
    With ``optimizer_device='cpu'`` the factory owns separate CPU masters and
    updates them on CPU; compute banks remain resident. Factories must return
    exactly torch.optim.SGD/Adam/AdamW or ramtorch.AdamEF/Lion/Muon (not subclasses).
    AdamEF/Lion/Muon require FP32/FP64 optimizer parameters and keep update history
    attached to stable Parameter identities across bank switches. Native fused
    AdamW remains available, e.g. ``torch.optim.AdamW(params, fused=True)``.
    CPU thread settings belong to the application, not this trainer.
    """

    def __init__(self, pipe, *, optimizer_factory, n_microbatches=4,
                 loss_fn, max_inflight=None, optimizer_device=None):
        from .pipeline import Stage
        if getattr(pipe, "_train_sess", None) is not None:
            raise RuntimeError("pipeline already has a training session")
        if pipe._infer_sess is not None:
            raise RuntimeError("close streaming inference before training")
        if not pipe._manual:
            raise ValueError("2BW initially requires manual resident stages/chunks")
        if not isinstance(n_microbatches, int) or isinstance(n_microbatches, bool) or n_microbatches < pipe.num_stages:
            raise ValueError("n_microbatches must be an integer >= pipeline depth")
        if not callable(loss_fn):
            raise ValueError("loss_fn must be callable")
        self.pipe, self.m, self.loss_fn = pipe, n_microbatches, loss_fn
        self.depth = pipe.num_stages
        self.max_inflight = max_inflight if max_inflight is not None else 2 * self.m + self.depth
        if not isinstance(self.max_inflight, int) or isinstance(self.max_inflight, bool) or self.max_inflight < self.depth:
            raise ValueError("max_inflight must be an integer >= pipeline depth")
        device_types = {stage.device.type for stage in pipe.stages}
        if device_types not in ({"cpu"}, {"cuda"}):
            raise ValueError("2BW requires all-CUDA stages (or all-CPU for testing); mixed devices are unsupported")
        seen = set()
        for stage in pipe.stages:
            if type(stage) is not Stage or stage.fake is not None:
                raise ValueError("2BW requires resident, non-fake Stage objects; no offloading")
            if stage.autocast_dtype not in (None, torch.bfloat16):
                raise ValueError("2BW supports fp32 or bf16 autocast, not fp16")
            if list(stage.module.buffers()):
                raise ValueError("2BW initially requires buffer-free modules")
            for module in stage.module.modules():
                if isinstance(module, (nn.modules.dropout._DropoutNd, nn.RReLU,
                                       nn.modules.batchnorm._BatchNorm)):
                    raise ValueError("2BW does not support stochastic/stateful modules")
                if isinstance(module, (nn.Embedding, nn.EmbeddingBag)) and module.max_norm is not None:
                    raise ValueError("2BW does not support forward-mutating embedding max_norm")
                if isinstance(module, (nn.MultiheadAttention, nn.RNNBase)) and module.dropout:
                    raise ValueError("2BW does not support internal attention/RNN dropout")
            for _, parameter in stage.module.named_parameters(remove_duplicate=False):
                if id(parameter) in seen:
                    raise ValueError("shared/tied parameters are not supported by 2BW")
                seen.add(id(parameter))
            if stage._cache or stage.grad_acc or any(p.grad is not None for p in stage.params):
                raise ValueError("start 2BW with clean stages (no cached graphs/gradients)")
        factories = [optimizer_factory] * self.depth if callable(optimizer_factory) else list(optimizer_factory)
        if len(factories) != self.depth:
            raise ValueError("need one optimizer factory per stage")
        self.stages = [_VersionedStage(s, factory, optimizer_device=optimizer_device)
                       for s, factory in zip(pipe.stages, factories)]
        self.optimizers = [s.optimizer for s in self.stages]
        self.streams = [torch.cuda.Stream(device=s.stage.device) if s.stage.device.type == "cuda" else None
                        for s in self.stages]
        for stream, stage in zip(self.streams, pipe.stages):
            if stream is not None:
                stream.wait_stream(torch.cuda.current_stream(stage.device))
        self.total_updates = 0
        self._closed = False
        self._poison = None
        self._running = threading.Lock()
        self._condition = threading.Condition()
        self._abort = threading.Event()
        self._error = None
        self._tickets = {}
        self._retired = collections.deque()
        self._commands = [queue.Queue() for _ in self.stages]
        self._done = [threading.Event() for _ in self.stages]
        self._workers = [threading.Thread(target=self._worker, args=(i,), daemon=True,
                                          name=f"2bw-stage-{i}") for i in range(self.depth)]
        pipe._train_sess = self
        for worker in self._workers:
            worker.start()

    def _fail(self, error):
        with self._condition:
            if self._error is None:
                self._error = error
            self._abort.set()
            for ticket in self._tickets.values():
                ticket.release()
            self._condition.notify_all()

    def _check(self):
        if self._abort.is_set():
            raise RuntimeError("2BW run aborted") from self._error

    def _ticket(self, mb):
        with self._condition:
            self._condition.wait_for(lambda: mb in self._tickets or self._abort.is_set())
            self._check()
            return self._tickets[mb]

    def _worker(self, index):
        stage = self.stages[index]
        dev, stream = stage.stage.device, self.streams[index]
        device_context = torch.cuda.device(dev) if stream is not None else contextlib.nullcontext()
        with device_context:
            while True:
                command = self._commands[index].get()
                if command is None:
                    return
                start, count, observer, trace = command
                try:
                    with torch.enable_grad(), (torch.cuda.stream(stream) if stream is not None else contextlib.nullcontext()):
                        for kind, mb in _rank_ops(index, self.depth, count, self.m, start):
                            self._check()
                            group = mb // self.m
                            version = max(group - 1, 0)
                            with trace.span(kind, stage=index, mb=mb, group=group,
                                            version=version, slot=stage.slot(version) if kind != "U" else 1-stage.slot(stage.latest)):
                                if kind == "U":
                                    stage.update(group, self.m, observer, trace)
                                    continue
                                with trace.span("admission_wait", stage=index, mb=mb):
                                    ticket = self._ticket(mb)
                                if kind == "F":
                                    with trace.span("forward_relay_wait", stage=index, mb=mb):
                                        value = (ticket.inputs if index == 0 else ticket.forward[index-1]).get()
                                        self._check()
                                    if index == 0:
                                        # Cross-device .to() launches on the source device's
                                        # current stream too. Order that stream, not only the
                                        # receiving compute stream, after caller production.
                                        self._order_sources(ticket.inputs, value)
                                    with trace.span("input_transfer", stage=index, mb=mb):
                                        value = self._move(value, dev)
                                    out = stage.forward(mb, value, self.m)
                                    if index < self.depth - 1:
                                        with trace.span("activation_send", stage=index, mb=mb):
                                            sent = self._move(self._detach(out), self.stages[index+1].stage.device)
                                            self._publish(ticket.forward[index], sent, stream)
                                            sent = None
                                else:
                                    if index == self.depth - 1:
                                        with trace.span("target_transfer", stage=index, mb=mb):
                                            target = ticket.targets.get()
                                            self._check()
                                            self._order_sources(ticket.targets, target)
                                            target = self._move(target, dev)
                                        grad, _ = stage.backward(mb, target=target, loss_fn=self.loss_fn)
                                    else:
                                        with trace.span("backward_relay_wait", stage=index, mb=mb):
                                            incoming = ticket.backward[index].get()
                                            self._check()
                                        grad, _ = stage.backward(mb, grad_output=incoming)
                                    if index:
                                        with trace.span("gradient_send", stage=index, mb=mb):
                                            sent = self._move(grad, self.stages[index-1].stage.device)
                                            self._publish(ticket.backward[index-1], sent, stream)
                                            sent = None
                                    else:
                                        event = torch.cuda.Event() if stream is not None else None
                                        if event is not None:
                                            event.record(stream)
                                        with self._condition:
                                            self._retired.append((mb, event))
                                            self._condition.notify_all()
                                # No worker-local last-ticket reference survives a run.
                                del ticket
                                value = out = grad = incoming = target = None
                except BaseException as error:
                    self._fail(error)
                finally:
                    # Persistent workers must not keep the last graph, observer
                    # closure, or a potentially huge profiler capture alive.
                    ticket = value = out = grad = incoming = target = sent = _ = None
                    command = observer = trace = None
                    self._done[index].set()

    @staticmethod
    def _publish(mailbox, value, stream):
        # A payload may contain NO CUDA tensors (e.g. all input gradients None).
        # Still propagate the producing compute stream's completion dependency:
        # stage-0 retirement must fence downstream work even across detached edges.
        # Append before publishing readiness, never after waking the consumer.
        if stream is not None:
            event = torch.cuda.Event()
            event.record(stream)
            mailbox._cuda_events.append(event)
        mailbox.put_from_any_device(value)

    @staticmethod
    def _detach(value):
        if isinstance(value, torch.Tensor):
            return value.detach()
        if isinstance(value, (tuple, list)):
            return tuple(PipeDream2BWTrainer._detach(v) for v in value)
        return value

    @staticmethod
    def _order_sources(mailbox, value):
        tensors = value if isinstance(value, (tuple, list)) else (value,)
        for tensor in tensors:
            if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
                for event in mailbox._cuda_events:
                    torch.cuda.current_stream(tensor.device).wait_event(event)

    @staticmethod
    def _move(value, device):
        if isinstance(value, torch.Tensor):
            # CPU consumers cannot wait on CUDA events through a CPU tensor.
            # Make device-to-host inputs/targets ready before returning them.
            non_blocking = not (value.is_cuda and torch.device(device).type == "cpu")
            return value.to(device, non_blocking=non_blocking)
        if isinstance(value, (tuple, list)):
            return tuple(PipeDream2BWTrainer._move(v, device) for v in value)
        if value is None:
            return None
        raise ValueError("inputs/targets must be tensors or flat tuples of tensors")

    def _split(self, inputs, targets):
        def pieces(value, optional=False):
            if value is None and optional:
                return [None] * self.m
            ts = tuple(value) if isinstance(value, (tuple, list)) else (value,)
            if not ts or any(not isinstance(t, torch.Tensor) or t.ndim == 0 for t in ts):
                raise ValueError("expected tensor or flat tuple of batched tensors")
            size = ts[0].shape[0]
            if size < self.m or size % self.m or any(t.shape[0] != size for t in ts):
                raise ValueError("batch must split into complete equal-size microbatches")
            chunks = [t.chunk(self.m) for t in ts]
            return [tuple(c[i] for c in chunks) if isinstance(value, (tuple, list)) else chunks[0][i]
                    for i in range(self.m)]
        xs, ys = pieces(inputs), pieces(targets, optional=True)
        if targets is not None:
            first_x = inputs[0] if isinstance(inputs, (tuple, list)) else inputs
            first_y = targets[0] if isinstance(targets, (tuple, list)) else targets
            if first_x.shape[0] != first_y.shape[0]:
                raise ValueError("input/target batch sizes differ")
        return zip(xs, ys)

    def _make_room(self, trace):
        while len(self._tickets) >= self.max_inflight:
            with trace.span("backpressure_wait", stage=-1):
                with self._condition:
                    self._condition.wait_for(lambda: self._retired or self._abort.is_set())
                    self._check()
                    mb, event = self._retired.popleft()
                # Bound actual CUDA lifetime, not merely host enqueue count.
                # This waits on one completed microbatch, never all devices.
                if event is not None:
                    event.synchronize()
                with self._condition:
                    del self._tickets[mb]

    def run(self, loader, *, updates, observer=None, profile_path=None):
        """Run N complete update groups; observer copies are opt-in debug only."""
        from .pipeline_2bw_trace import TraceCapture
        if self._closed or self._poison is not None:
            raise RuntimeError("2BW session is closed or failed") from self._poison
        if not isinstance(updates, int) or isinstance(updates, bool) or updates < 1:
            raise ValueError("updates must be a positive integer")
        if not self._running.acquire(blocking=False):
            raise RuntimeError("concurrent run/close is not supported")
        peak = 0
        dispatched = False
        try:
            self._tickets.clear()
            self._retired.clear()
            self._abort.clear()
            self._error = None
            with TraceCapture(profile_path) as trace:
                try:
                    for done in self._done:
                        done.clear()
                    start = self.total_updates * self.m
                    for commands in self._commands:
                        commands.put((start, updates * self.m, observer, trace))
                    dispatched = True
                    iterator = iter(loader)
                    for group in range(updates):
                        self._check()
                        with trace.span("loader_next", stage=-1, group=self.total_updates+group):
                            try:
                                batch = next(iterator)
                            except StopIteration as error:
                                raise ValueError("loader ended before requested update count") from error
                        if not isinstance(batch, (tuple, list)) or len(batch) != 2:
                            raise ValueError("loader must yield (inputs, targets)")
                        for local, (inputs, targets) in enumerate(self._split(*batch)):
                            self._make_room(trace)
                            self._check()
                            mb = start + group * self.m + local
                            ticket = _Ticket(self.depth, inputs, targets)
                            with self._condition:
                                self._tickets[mb] = ticket
                                peak = max(peak, len(self._tickets))
                                self._condition.notify_all()
                except BaseException as error:
                    self._fail(error)
                finally:
                    if dispatched:
                        for done in self._done:
                            done.wait()
                    # A run boundary is deliberately drained; never a group boundary.
                    for stream in self.streams:
                        if stream is not None:
                            stream.synchronize()
                    self._tickets.clear()
                    self._retired.clear()
                if self._error is not None:
                    # GPU reads are drained above. A failed run is unusable, so
                    # discard partial graphs/gradients rather than retaining them
                    # through the poisoned session until a later close.
                    for stage in self.stages:
                        stage.cache.clear()
                        stage.live[:] = [0, 0]
                        for name in stage.acc:
                            stage.acc[name] = None
                        stage.optimizer.zero_grad(set_to_none=True)
                        stage.cpu_grads.clear()
                    raise RuntimeError("2BW training failed; close this session") from self._error
            self.total_updates += updates
            return TrainRunResult(updates, self.total_updates, peak)
        except BaseException as error:
            # Keep only a lightweight poison marker on the persistent trainer.
            # The exception delivered to the caller retains its original chain,
            # but the trainer must not own failed frames and their CUDA tensors.
            self._poison = RuntimeError(f"previous run failed: {type(error).__name__}")
            raise
        finally:
            self._error = None
            self._running.release()

    def close(self):
        """Join workers and expose latest weights. Does not roll back failures."""
        if self._closed:
            return
        if not self._running.acquire(blocking=False):
            raise RuntimeError("cannot close while run is active; let run finish first")
        try:
            for commands in self._commands:
                commands.put(None)
            for worker in self._workers:
                worker.join()
            for stream in self.streams:
                if stream is not None:
                    stream.synchronize()
            for stage in self.stages:
                stage.release()
            self._closed = True
            if self.pipe._train_sess is self:
                self.pipe._train_sess = None
        finally:
            self._running.release()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
