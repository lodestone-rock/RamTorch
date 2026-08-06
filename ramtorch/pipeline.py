"""
ramtorch.pipeline
-----------------
Single-process pipeline parallelism.

Splits a model with ``torch.distributed.pipelining.pipeline()`` + ``SplitPoint``,
places stage *i* on ``devices[i]`` (default ``cuda:i``), and executes GPipe,
1F1B, or Interleaved-1F1B schedules locally — no ``torchrun``, no process
groups. P2P communication is a plain ``.to(device)``.

Autograd is fully manual: every stage forward/backward is an independent
``torch.autograd.grad`` call, so each (stage, microbatch, fwd|bwd) op is a
separately schedulable unit. Parameter gradients are returned as tensors and
accumulated explicitly by the orchestrator (deterministic order).

Overlap: one worker thread per stage with a FIFO op queue. CUDA kernel
launches release the GIL and each device has its own default stream, so
stages on different GPUs genuinely overlap. ``overlap=False`` runs the same
schedule sequentially for debugging.

Fake-compute mode (``fake_compute={"fwd": s, "bwd": s}``) wraps each op in
``time.sleep`` so schedule timing/bubbles can be simulated without loading
the GPUs. ``fake_compute="replace"`` additionally skips the real kernels and
returns zeros of the correct shape (pure-logic CPU simulation).
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from queue import Queue
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

__all__ = ["Stage", "run_pipeline", "PipelineResult", "PerfettoTracer"]


# ── Perfetto / Chrome-trace instrumentation ──────────────────────────────────

class PerfettoTracer:
    """
    Records pipeline ops and exports a Chrome-trace JSON file that can be
    dropped into https://ui.perfetto.dev (or chrome://tracing).

    Two kinds of spans per op:
      * CPU span  — time on the worker thread (kernel launches + host work),
        track "stage s / cpu".
      * GPU span  — measured with torch.cuda.Event around the op, track
        "stage s / cuda:<idx>". Only for CUDA stages; reflects when the work
        actually ran on the device (events are recorded in stream order).

    Cross-thread safe; ``export()`` resolves all CUDA event pairs first.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.t0 = time.perf_counter()
        self.events: List[dict] = []
        self._pending: List[dict] = []   # unresolved CUDA-event pairs

    def _ts(self) -> float:
        return (time.perf_counter() - self.t0) * 1e6  # microseconds

    def record_cpu(self, name: str, cat: str, track: str, t0_us: float, t1_us: float):
        with self._lock:
            self.events.append({
                "name": name, "cat": cat, "ph": "X", "pid": track,
                "tid": "cpu", "ts": t0_us, "dur": max(t1_us - t0_us, 0.01),
            })

    def record_gpu(self, name: str, cat: str, track: str, dev_idx: int,
                   ev0, ev1):
        with self._lock:
            self._pending.append({
                "name": name, "cat": cat, "pid": track, "tid": f"cuda:{dev_idx}",
                "ev0": ev0, "ev1": ev1, "ts0": self._ts(),
            })

    def export(self, path: str):
        # Resolve CUDA events: ts of event pair + elapsed gives span.
        for rec in self._pending:
            rec["ev1"].synchronize()
            dur_ms = rec["ev0"].elapsed_time(rec["ev1"])
            self.events.append({
                "name": rec["name"], "cat": rec["cat"], "ph": "X",
                "pid": rec["pid"], "tid": rec["tid"], "ts": rec["ts0"],
                "dur": max(dur_ms * 1e3, 0.01),
            })
        self._pending.clear()
        import json
        with open(path, "w") as f:
            json.dump({"traceEvents": self.events,
                       "displayTimeUnit": "ms"}, f)


# ── Fake-compute config ───────────────────────────────────────────────────────

class _FakeCompute:
    """Parsed fake_compute config."""

    def __init__(self, cfg, num_stages: int):
        self.replace = cfg == "replace"
        if cfg is None or cfg == "replace":
            self.fwd = [0.0] * num_stages
            self.bwd = [0.0] * num_stages
            return

        def _expand(v):
            if isinstance(v, (int, float)):
                return [float(v)] * num_stages
            v = list(v)
            assert len(v) == num_stages
            return [float(x) for x in v]

        self.fwd = _expand(cfg.get("fwd", 0.0))
        self.bwd = _expand(cfg.get("bwd", 0.0))


# ── Stage ─────────────────────────────────────────────────────────────────────

class Stage:
    """
    Wraps one nn.Module partition on one device.

    Manual-autograd contract:
      * ``forward_one_chunk`` caches ``(input_leaf, output)`` per microbatch.
        The input of a non-first stage is detached into a fresh leaf, so each
        microbatch's graph is strictly per-stage.
      * ``backward_one_chunk`` pops the cached pair and calls
        ``torch.autograd.grad(output, [input_leaf] + params, grad_output)``.
        It returns ``(input_grad, param_grads)`` — nothing is written to
        ``.grad`` implicitly. Each cached graph is consumed exactly once, so
        no ``retain_graph`` is ever needed.

    Thread safety: a per-stage lock guards the caches; a stage's ops are
    executed by a single worker thread in FIFO order anyway.
    """

    def __init__(
        self,
        module: nn.Module,
        stage_index: int,
        num_stages: int,
        device: Union[torch.device, str] = "cpu",
        fake: Optional[_FakeCompute] = None,
        tracer: Optional["PerfettoTracer"] = None,
    ):
        self.device = torch.device(device)
        self.module = module.to(self.device)
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.is_first = stage_index == 0
        self.is_last = stage_index == num_stages - 1
        self.fake = fake
        self.tracer = tracer
        self.dev_idx = self.device.index if self.device.index is not None else 0
        self._track = f"stage {stage_index}"
        self.losses: Dict[int, torch.Tensor] = {}   # mb_index -> scalar loss (last stage)

        self.params: List[nn.Parameter] = list(self.module.parameters())
        # Explicit gradient accumulators (deterministic add order).
        self.grad_acc: Dict[nn.Parameter, torch.Tensor] = {
            p: torch.zeros_like(p) for p in self.params
        }

        self._lock = threading.Lock()
        self._cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    # ------------------------------------------------------------------
    def _sleep(self, kind: str):
        if self.fake is None:
            return
        t = self.fake.fwd[self.stage_index] if kind == "fwd" else self.fake.bwd[self.stage_index]
        if t > 0:
            time.sleep(t)

    # ------------------------------------------------------------------
    def forward_one_chunk(self, mb_index: int, x) -> torch.Tensor:
        t0 = self.tracer._ts() if self.tracer else 0.0
        ev0 = ev1 = None
        if self.tracer and self.device.type == "cuda":
            ev0 = torch.cuda.Event(enable_timing=True)
            ev0.record()

        # ``x`` may be a single tensor or a tuple of tensors (multi-input stage).
        # Normalize to a tuple internally; single-tensor input becomes (x,).
        is_tuple = isinstance(x, (tuple, list))
        xs = tuple(x) if is_tuple else (x,)

        def _prep(t):
            t = t.to(self.device)
            return t.detach().requires_grad_(t.is_floating_point())

        xs = tuple(_prep(t) for t in xs)

        if self.fake is not None and self.fake.replace:
            self._sleep("fwd")
            out = self._fake_output(xs[0])
        else:
            self._sleep("fwd")
            out = self.module(*xs) if is_tuple else self.module(xs[0])

        with self._lock:
            # Cache the (possibly multi-tensor) input for the backward pass.
            self._cache[mb_index] = (xs if is_tuple else xs[0], out)

        if self.tracer:
            name = f"F mb{mb_index}"
            if ev0 is not None:
                ev1 = torch.cuda.Event(enable_timing=True)
                ev1.record()
                self.tracer.record_gpu(name, "fwd", self._track, self.dev_idx, ev0, ev1)
            self.tracer.record_cpu(name, "fwd", self._track, t0, self.tracer._ts())
        return out

    def _fake_output(self, x: torch.Tensor) -> torch.Tensor:
        """Shape-correct dummy output for replace mode (linear: last-dim preserved)."""
        # Best effort: mirror input shape; real numerics don't matter in replace mode.
        out = torch.zeros_like(x, requires_grad=True)
        return out + x.sum() * 0  # keep a graph edge so autograd.grad works

    # ------------------------------------------------------------------
    def backward_one_chunk(
        self,
        mb_index: int,
        grad_output: Optional[torch.Tensor] = None,
        loss: Optional[torch.Tensor] = None,
        loss_fn: Optional[Callable] = None,
        target: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Run backward for one microbatch via manual autograd.

        Last stage: provide ``loss_fn`` + ``target`` (loss is computed here, on
        the worker, from the cached output) OR a precomputed ``loss``. Other
        stages: provide ``grad_output``. Returns the gradient w.r.t. this
        stage's input (None for the first stage); accumulates param grads into
        ``self.grad_acc``. Also stashes the scalar loss in ``self.losses[mb]``.
        """
        with self._lock:
            inp, out = self._cache.pop(mb_index)

        t0 = self.tracer._ts() if self.tracer else 0.0
        ev0 = None
        if self.tracer and self.device.type == "cuda":
            ev0 = torch.cuda.Event(enable_timing=True)
            ev0.record()

        self._sleep("bwd")

        # ``inp`` may be a single tensor or a tuple of tensors (multi-input stage).
        # Flatten tuple inputs into the autograd inputs list; the returned
        # input-grads then come back as a matching tuple.
        inp_is_tuple = isinstance(inp, (tuple, list))
        inp_list = list(inp) if inp_is_tuple else [inp]
        inputs = inp_list + self.params
        n_inp = len(inp_list)

        if self.is_last:
            if loss is None:
                assert loss_fn is not None, "last stage needs loss_fn+target or loss"
                loss = loss_fn(out, target)
            self.losses[mb_index] = loss.detach()
            grads = torch.autograd.grad(loss, inputs, allow_unused=True)
        else:
            assert grad_output is not None, "non-last stage needs grad_output"
            grads = torch.autograd.grad(
                out, inputs, grad_outputs=grad_output.to(self.device),
                allow_unused=True,
            )

        input_grads, param_grads = grads[:n_inp], grads[n_inp:]
        # Single-tensor input -> single grad; tuple input -> tuple of grads.
        input_grad = input_grads if inp_is_tuple else input_grads[0]
        with self._lock:
            for p, g in zip(self.params, param_grads):
                if g is not None:
                    self.grad_acc[p] += g.detach()

        if self.tracer:
            name = f"B mb{mb_index}"
            if ev0 is not None:
                ev1 = torch.cuda.Event(enable_timing=True)
                ev1.record()
                self.tracer.record_gpu(name, "bwd", self._track, self.dev_idx, ev0, ev1)
            self.tracer.record_cpu(name, "bwd", self._track, t0, self.tracer._ts())

        return None if self.is_first else input_grad

    # ------------------------------------------------------------------
    def flush_grads(self, scale: float = 1.0):
        """Write accumulated grads into ``.grad`` (for optimizer compatibility) and reset."""
        with self._lock:
            for p, acc in self.grad_acc.items():
                p.grad = acc * scale
                acc.zero_()

    def zero_grad_acc(self):
        with self._lock:
            for acc in self.grad_acc.values():
                acc.zero_()

    def clear(self):
        with self._lock:
            self._cache.clear()


# ── Overlap engine ────────────────────────────────────────────────────────────

class _Op:
    __slots__ = ("kind", "stage", "mb", "payload", "dep", "prev", "done", "result")

    def __init__(self, kind: str, stage: int, mb: int, payload=None, dep=None, prev=None):
        self.kind = kind          # 'F' or 'B'
        self.stage = stage
        self.mb = mb
        self.payload = payload    # input tensor (F) or (grad_output|loss) (B)
        self.dep = dep            # optional _Op whose result is the real payload
        self.prev = prev          # optional prior op on the SAME stage (ordering)
        self.done = threading.Event()
        self.result = None


class _StageWorker(threading.Thread):
    """FIFO op queue for one stage; runs under its stage's device context."""

    def __init__(self, stage: Stage):
        super().__init__(daemon=True)
        self.stage = stage
        self.q: Queue[Optional[_Op]] = Queue()
        self.error: Optional[BaseException] = None

    def run(self):
        dev = self.stage.device
        while True:
            op = self.q.get()
            if op is None:
                return
            try:
                if dev.type == "cuda":
                    torch.cuda.set_device(dev)
                # Enforce per-stage serial order: wait for the prior op on this
                # stage before starting (but a forward's prev may finish long
                # before its cross-stage dep, so this doesn't serialize stages).
                if op.prev is not None:
                    op.prev.done.wait()
                if op.kind == "F":
                    x = op.payload
                    if op.dep is not None:
                        op.dep.done.wait()
                        x = op.dep.result
                    op.result = self.stage.forward_one_chunk(op.mb, x)
                else:
                    loss, grad, loss_fn, target = op.payload
                    if op.dep is not None:
                        op.dep.done.wait()
                        grad = op.dep.result
                    op.result = self.stage.backward_one_chunk(
                        op.mb, grad_output=grad, loss=loss,
                        loss_fn=loss_fn, target=target,
                    )
            except BaseException as e:  # propagate to orchestrator
                self.error = e
            finally:
                op.done.set()


class _Engine:
    """Owns workers; dispatches ops and waits for results."""

    def __init__(self, stages: List[Stage], overlap: bool):
        self.stages = stages
        self.overlap = overlap
        self.workers: Optional[List[_StageWorker]] = None
        if overlap:
            self.workers = [_StageWorker(s) for s in stages]
            for w in self.workers:
                w.start()

    def close(self):
        if self.workers:
            for w in self.workers:
                w.q.put(None)
            for w in self.workers:
                w.join()

    def _check_errors(self):
        if self.workers:
            for w in self.workers:
                if w.error is not None:
                    raise RuntimeError(f"stage worker failed: {w.error}") from w.error

    def fwd_async(self, stage_idx: int, mb: int, x=None, dep: Optional[_Op] = None,
                  prev: Optional[_Op] = None) -> _Op:
        """Dispatch a forward; returns a handle to wait on later."""
        if not self.overlap:
            op = _Op("F", stage_idx, mb)
            xx = dep.result if dep is not None else x
            op.result = self.stages[stage_idx].forward_one_chunk(mb, xx)
            op.done.set()
            return op
        op = _Op("F", stage_idx, mb, x, dep=dep, prev=prev)
        self.workers[stage_idx].q.put(op)
        return op

    def bwd_async(self, stage_idx: int, mb: int, *, loss=None, grad=None,
                  loss_fn=None, target=None, dep: Optional[_Op] = None,
                  prev: Optional[_Op] = None) -> _Op:
        """Dispatch a backward; returns a handle to wait on later."""
        if not self.overlap:
            op = _Op("B", stage_idx, mb)
            gg = dep.result if dep is not None else grad
            op.result = self.stages[stage_idx].backward_one_chunk(
                mb, grad_output=gg, loss=loss, loss_fn=loss_fn, target=target
            )
            op.done.set()
            return op
        op = _Op("B", stage_idx, mb, (loss, grad, loss_fn, target), dep=dep, prev=prev)
        self.workers[stage_idx].q.put(op)
        return op

    def wait(self, op: _Op):
        op.done.wait()
        self._check_errors()
        return op.result


# ── Schedules ─────────────────────────────────────────────────────────────────

# ── Generic per-rank op-list driver ──────────────────────────────────────────
#
# A schedule is expressed as one FIFO op list per stage (mirroring torch's
# _get_pipeline_order). Ops: ('F', mb) | ('B', mb) | ('W', mb) where 'W' waits
# for the same-stage forward of mb to finish before computing the loss locally
# (torch sends losses across ranks instead; we keep them in shared memory).
#
# The driver dispatches ops whose input dependency is ready; each stage's ops
# remain strictly FIFO on its worker. Independent stages therefore overlap.

_WAIT_POLL_S = 0.0005


def _run_schedule_ops(
    rank_ops: List[List[Tuple[str, int]]],
    stages: List[Stage],
    engine: _Engine,
    micro_batches: List[torch.Tensor],
    targets: Optional[List[torch.Tensor]],
    loss_fn: Callable,
    tracer: Optional["PerfettoTracer"] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    p = len(stages)
    m = len(micro_batches)
    outputs: Dict[int, torch.Tensor] = {}
    losses: Dict[int, torch.Tensor] = {}

    fwd_handles: Dict[Tuple[int, int], _Op] = {}   # (stage, mb) -> fwd handle
    bwd_handles: Dict[Tuple[int, int], _Op] = {}   # (stage, mb) -> bwd handle
    activations: Dict[Tuple[int, int], torch.Tensor] = {}  # (mb, stage) -> out
    grads: Dict[Tuple[int, int], torch.Tensor] = {}        # (mb, stage) -> input grad

    # ── Dispatch in dependency order, not strict per-stage FIFO ──────────────
    #
    # Key insight (the zig-zag fix): a FORWARD for microbatch t+1 must NOT wait
    # behind a BACKWARD for microbatch t on the same stage. Forwards only depend
    # on the previous stage's forward of the same microbatch. So we dispatch all
    # forwards first (in mb order per stage), then all backwards.
    #
    # Per-stage ordering is still enforced: each stage's worker is FIFO, and we
    # chain each op to the previous op on the same stage via `prev_op` so a
    # stage never runs two of its own ops concurrently. But because forwards are
    # enqueued before the backwards that would otherwise block them, a stage's
    # forward for mb t+1 is already queued and ready the moment its own prior
    # op finishes — it doesn't wait for a different microbatch's backward.
    prev_op: List[Optional[_Op]] = [None] * p   # last op dispatched per stage

    def _dispatch_fwd(s: int, mb: int):
        dep = None if s == 0 else fwd_handles.get((s - 1, mb))
        if s > 0 and dep is None:
            return False
        x = micro_batches[mb] if s == 0 else None
        h = engine.fwd_async(s, mb, x=x, dep=dep, prev=prev_op[s])
        fwd_handles[(s, mb)] = h
        prev_op[s] = h
        return True

    def _dispatch_bwd(s: int, mb: int):
        if s == p - 1:
            fh = fwd_handles.get((s, mb))
            if fh is None:
                return False
            tgt = targets[mb] if targets else None
            h = engine.bwd_async(s, mb, loss_fn=loss_fn, target=tgt, dep=fh,
                                 prev=prev_op[s])
        else:
            down = bwd_handles.get((s + 1, mb))
            if down is None:
                return False
            h = engine.bwd_async(s, mb, dep=down, prev=prev_op[s])
        bwd_handles[(s, mb)] = h
        prev_op[s] = h
        return True

    # Collect the F and B ops per stage from the (validated) op lists.
    fwd_ops = [ [mb for kind, mb in ops if kind == "F"] for ops in rank_ops ]
    bwd_ops = [ [mb for kind, mb in ops if kind == "B"] for ops in rank_ops ]

    # Phase 1: dispatch all forwards (dependency = previous stage's fwd).
    # Repeat sweeps until every forward is enqueued.
    fwd_pending = [list(q) for q in fwd_ops]
    while any(fwd_pending):
        progress = False
        for s in range(p):
            while fwd_pending[s] and _dispatch_fwd(s, fwd_pending[s][0]):
                fwd_pending[s].pop(0)
                progress = True
        if not progress:
            time.sleep(_WAIT_POLL_S)

    # Phase 2: dispatch all backwards (dependency = next stage's bwd, or own fwd
    # for the last stage). Last stage goes first to seed the chain.
    bwd_pending = [list(q) for q in bwd_ops]
    while any(bwd_pending):
        progress = False
        for s in range(p - 1, -1, -1):
            while bwd_pending[s] and _dispatch_bwd(s, bwd_pending[s][0]):
                bwd_pending[s].pop(0)
                progress = True
        if not progress:
            time.sleep(_WAIT_POLL_S)

    # All ops dispatched; wait for completion and harvest outputs/losses.
    # Wait on the *backward* handles (which populate losses), not just forwards.
    for mb in range(m):
        outputs[mb] = engine.wait(fwd_handles[(p - 1, mb)])
    for mb in range(m):
        engine.wait(bwd_handles[(p - 1, mb)])
        losses[mb] = stages[p - 1].losses[mb]

    return [outputs[i] for i in range(m)], [losses[i] for i in range(m)]


def _build_gpipe_rank_ops(p: int, m: int) -> List[List[Tuple[str, int]]]:
    ops = [[("F", mb) for mb in range(m)] for _ in range(p)]
    ops[p - 1].extend(("W", mb) for mb in range(m))
    for s in range(p):
        ops[s].extend(("B", mb) for mb in range(m))
    return ops


def _build_1f1b_rank_ops(p: int, m: int) -> List[List[Tuple[str, int]]]:
    """Mirrors torch Schedule1F1B._get_pipeline_order."""
    warmup = min(p - 1, m)
    steady = m - warmup
    ops: List[List[Tuple[str, int]]] = []
    for s in range(p):
        ro: List[Tuple[str, int]] = [("F", mb) for mb in range(warmup)]
        for k in range(steady):
            ro.append(("F", warmup + k))
            if s == p - 1:
                ro.append(("W", k))
            ro.append(("B", k))
        for mb in range(steady, m):
            if s == p - 1:
                ro.append(("W", mb))
            ro.append(("B", mb))
        ops.append(ro)
    return ops


def _build_staggered_1b1f_rank_ops(p: int, m: int) -> List[List[Tuple[str, int]]]:
    """
    Staggered-warmup, backward-eager schedule ("1B1F").

    Two differences from textbook (forward-first) 1F1B:

    1. Staggered warmup: stage ``s`` runs ``(p - 1 - s)`` warmup forwards, so the
       first stage crams the most microbatches and the LAST stage does zero
       warmup — it backwards each microbatch immediately after its forward.
    2. Backward-eager steady state: after warmup, a stage issues a backward as
       soon as its microbatch is ready (its own forward done + downstream grad,
       resolved by the executor), rather than always forwarding first.

    Effect (validated in ramtorch/schedule_simulator.py): the last stage pulls
    gradients back up the pipeline with minimal latency, eliminating the
    steady-state bubble of forward-first 1F1B. Matches GPipe makespan while
    bounding in-flight activations to ~p microbatches (1F1B memory).

    Pattern (p=2, m=4):
        s0: F0 F1 .  B0 F2 B1 F3 B2 .  B3
        s1: .  F0 B0 F1 B1 F2 B2 F3 B3
    """
    ops: List[List[Tuple[str, int]]] = []
    for s in range(p):
        warmup = max(0, min(p - 1 - s, m))
        ro: List[Tuple[str, int]] = []
        fwd = 0
        bwd = 0
        for _ in range(warmup):
            ro.append(("F", fwd))
            fwd += 1
        while bwd < m:
            if fwd < m:
                ro.append(("F", fwd))
                fwd += 1
            if bwd < fwd:
                if s == p - 1:
                    ro.append(("W", bwd))
                ro.append(("B", bwd))
                bwd += 1
        ops.append(ro)
    return ops


def _step_gpipe(
    stages: List[Stage],
    engine: _Engine,
    micro_batches: List[torch.Tensor],
    targets: Optional[List[torch.Tensor]],
    loss_fn: Callable,
    tracer: Optional["PerfettoTracer"] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    rank_ops = _build_gpipe_rank_ops(len(stages), len(micro_batches))
    return _run_schedule_ops(rank_ops, stages, engine, micro_batches, targets, loss_fn, tracer)


def _step_1f1b(
    stages: List[Stage],
    engine: _Engine,
    micro_batches: List[torch.Tensor],
    targets: Optional[List[torch.Tensor]],
    loss_fn: Callable,
    tracer: Optional["PerfettoTracer"] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    rank_ops = _build_1f1b_rank_ops(len(stages), len(micro_batches))
    return _run_schedule_ops(rank_ops, stages, engine, micro_batches, targets, loss_fn, tracer)


def _step_interleaved(
    stages: List[Stage],
    engine: _Engine,
    micro_batches: List[torch.Tensor],
    targets: Optional[List[torch.Tensor]],
    loss_fn: Callable,
    pp_group_size: int,
    tracer: Optional["PerfettoTracer"] = None,
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """
    Interleaved 1F1B (Megatron-v2). ``stages`` holds all p*v virtual stages in
    logical order; the op ordering reproduces torch's
    ``_calculate_single_rank_operations`` for the single-process view.
    """
    m = len(micro_batches)
    nv = len(stages)
    p = pp_group_size
    v = nv // p
    assert nv % p == 0, "num virtual stages must be divisible by pp_group_size"

    number_of_rounds = max(1, m // p)
    if m % number_of_rounds != 0:
        raise ValueError(
            f"interleaved requires n_microbatches ({m}) divisible by "
            f"number_of_rounds ({number_of_rounds})"
        )
    mpr = m // number_of_rounds

    # ── Build op list (ported from torch schedules.py, rank-0 view) ──────────
    def get_rank_warmup_ops(rank):
        warmups_last = (v - 1) * mpr
        return min(warmups_last + 2 * ((p - 1) - rank), m * v)

    rank = 0
    warmup_ops = get_rank_warmup_ops(rank)
    microbatch_ops = v * m
    fwd_bwd_ops = microbatch_ops - warmup_ops
    cooldown_ops = microbatch_ops - fwd_bwd_ops
    total_ops = warmup_ops + fwd_bwd_ops + cooldown_ops

    def fwd_stage_idx(step):
        return ((step // mpr) % v) * p + rank

    def bwd_stage_idx(step):
        return (v - 1 - ((step - warmup_ops) // mpr) % v) * p + rank

    fwd_mb_ctr: Dict[int, int] = defaultdict(int)
    bwd_mb_ctr: Dict[int, int] = defaultdict(int)
    rank_ops: List[List[Tuple[str, int]]] = [[] for _ in range(nv)]

    for op_i in range(total_ops):
        if op_i < warmup_ops:
            si = fwd_stage_idx(op_i)
            rank_ops[si].append(("F", fwd_mb_ctr[si]))
            fwd_mb_ctr[si] += 1
        elif op_i < warmup_ops + fwd_bwd_ops:
            fsi = fwd_stage_idx(op_i)
            rank_ops[fsi].append(("F", fwd_mb_ctr[fsi]))
            fwd_mb_ctr[fsi] += 1
            bsi = bwd_stage_idx(op_i)
            bmb = bwd_mb_ctr[bsi]
            if bsi == nv - 1:
                rank_ops[bsi].append(("W", bmb))
            rank_ops[bsi].append(("B", bmb))
            bwd_mb_ctr[bsi] += 1
        else:
            bsi = bwd_stage_idx(op_i)
            bmb = bwd_mb_ctr[bsi]
            if bsi == nv - 1:
                rank_ops[bsi].append(("W", bmb))
            rank_ops[bsi].append(("B", bmb))
            bwd_mb_ctr[bsi] += 1

    return _run_schedule_ops(
        rank_ops, stages, engine, micro_batches, targets, loss_fn, tracer
    )


# ── Public API ────────────────────────────────────────────────────────────────

class PipelineResult:
    """Result of a ``run_pipeline`` step."""

    def __init__(self, outputs, losses, stages):
        self.outputs = outputs          # last-stage output per microbatch
        self.losses = losses            # scalar loss per microbatch
        self.stages = stages            # Stage objects (own grad_acc)

    @property
    def loss(self) -> torch.Tensor:
        return torch.stack([l.detach() for l in self.losses]).mean()

    def flush_grads(self, scale: Optional[float] = None):
        """Write accumulated grads into ``.grad`` (default: scaled by 1/n_microbatches)."""
        s = scale if scale is not None else 1.0 / len(self.losses)
        for st in self.stages:
            st.flush_grads(scale=s)


def run_pipeline(
    model: nn.Module,
    example_input: torch.Tensor,
    split_spec: dict,
    data: torch.Tensor,
    *,
    targets: Optional[torch.Tensor] = None,
    schedule: str = "1f1b",
    n_microbatches: int = 4,
    loss_fn: Optional[Callable] = None,
    devices: Optional[Sequence[Union[str, torch.device]]] = None,
    fake_compute=None,
    overlap: bool = True,
    pp_group_size: Optional[int] = None,
    trace_path: Optional[str] = None,
    profile_path: Optional[str] = None,
) -> PipelineResult:
    """
    Split ``model`` and run one pipeline-parallel step in a single process.

    Parameters
    ----------
    model          : full model (will be traced/split; stage modules are moved
                     to their devices)
    example_input  : one example microbatch input for the tracer
    split_spec     : ``{layer_name: SplitPoint.BEGINNING|END}`` for pipeline()
    data           : full batch input tensor (dim 0 is split into microbatches)
    targets        : optional full-batch targets, split the same way
    schedule       : "gpipe" | "1f1b" | "interleaved"
    n_microbatches : number of microbatches
    loss_fn        : callable(output, target) -> scalar loss (default: sum)
    devices        : one device per stage (default: cuda:i, falling back to
                     fewer devices round-robin / CPU)
    fake_compute   : None | "replace" | {"fwd": s|[s...], "bwd": s|[s...]}
    overlap        : per-stage worker threads (True) or sequential (False)
    pp_group_size  : virtual pipeline depth p for "interleaved"
                     (default: number of devices); total stages = p * v where
                     v = num_stages / p
    trace_path     : if set, write a Chrome-trace JSON (Perfetto) of
                     op-level fwd/bwd spans to this path after the step
    profile_path   : if set, capture a full torch.profiler (kineto) trace of
                     the whole step — every CPU op, kernel dispatch, CUDA
                     kernel, and memcpy — and write it to this path. This is
                     the toggleable "see everything incl. dispatch" view and
                     is independent of trace_path.

    Returns
    -------
    PipelineResult with per-microbatch outputs/losses and stages holding
    accumulated param grads (``result.flush_grads()`` writes them to ``.grad``).
    """
    from torch.distributed.pipelining import pipeline as _split_pipeline

    loss_fn = loss_fn or (lambda out, _: out.sum())

    pipe = _split_pipeline(module=model, mb_args=(example_input,), split_spec=split_spec)
    num_stages = pipe.num_stages

    # ── Devices ───────────────────────────────────────────────────────────────
    if devices is None:
        n_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if n_cuda > 0:
            devices = [torch.device("cuda", i % n_cuda) for i in range(num_stages)]
        else:
            devices = [torch.device("cpu")] * num_stages
    else:
        devices = [torch.device(d) for d in devices]
        assert len(devices) == num_stages, (
            f"need {num_stages} devices, got {len(devices)}"
        )

    fake = _FakeCompute(fake_compute, num_stages) if fake_compute is not None else None
    tracer = PerfettoTracer() if trace_path else None

    stages = [
        Stage(pipe.get_stage_module(i), i, num_stages, device=devices[i],
              fake=fake, tracer=tracer)
        for i in range(num_stages)
    ]

    # ── Microbatch sharding ───────────────────────────────────────────────────
    assert data.shape[0] % n_microbatches == 0, (
        f"batch {data.shape[0]} not divisible by n_microbatches={n_microbatches}"
    )
    mbs = [mb.to(stages[0].device) for mb in data.chunk(n_microbatches, dim=0)]
    tgts = (
        [t.to(stages[-1].device) for t in targets.chunk(n_microbatches, dim=0)]
        if targets is not None
        else None
    )

    for st in stages:
        st.clear()
        st.zero_grad_acc()

    def _run():
        engine = _Engine(stages, overlap=overlap)
        try:
            if schedule == "gpipe":
                return _step_gpipe(stages, engine, mbs, tgts, loss_fn, tracer)
            elif schedule == "1f1b":
                return _step_1f1b(stages, engine, mbs, tgts, loss_fn, tracer)
            elif schedule == "interleaved":
                p = pp_group_size or len({d for d in devices})
                return _step_interleaved(
                    stages, engine, mbs, tgts, loss_fn, pp_group_size=p, tracer=tracer
                )
            else:
                raise ValueError(f"unknown schedule: {schedule!r}")
        finally:
            engine.close()

    if profile_path:
        from torch.profiler import ProfilerActivity, profile as _torch_profile
        with _torch_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as _prof:
            outputs, losses = _run()
        for d in {st.device for st in stages if st.device.type == "cuda"}:
            torch.cuda.synchronize(d)
        _prof.export_chrome_trace(profile_path)
    else:
        outputs, losses = _run()

    # Make sure all device work is done before reporting/exporting timings.
    for d in {st.device for st in stages if st.device.type == "cuda"}:
        torch.cuda.synchronize(d)

    if tracer is not None:
        tracer.export(trace_path)

    return PipelineResult(outputs, losses, stages)
