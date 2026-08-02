"""
ramtorch.pipeline_relay
-----------------------
Relay-style pipeline-parallel executor.

This is a *redesign of the execution engine only*. The model splitting, the
manual-autograd ``Stage`` contract, and the schedule builders are unchanged
(and reused from :mod:`ramtorch.pipeline`). What changes is *how* the ops are
driven.

Why a redesign
~~~~~~~~~~~~~~
The legacy driver (:func:`ramtorch.pipeline._run_schedule_ops`) built correct
per-stage op lists, then **re-grouped them into "all forwards, then all
backwards"** before dispatch. That collapsed every schedule — including 1F1B —
into GPipe, because scheduling was coupled to a central poller that dispatched
ops only when their input dependency happened to be ready.

The relay design removes the central scheduler entirely:

1.  Each schedule builder produces, **per stage**, a fully-ordered static op
    list: ``("F", mb)``, ``("B", mb)``, ``("W", mb)``.
2.  Each stage gets **one worker thread** that simply walks its own list in
    order. It blocks *only* on cross-stage data arrival — never on a central
    dispatch decision.
3.  Cross-stage handoff is a **relay**: forward activations flow stage ``s ->
    s+1`` and backward gradients flow ``s+1 -> s`` through per-``(stage,
    microbatch)`` thread-safe mailboxes.

Because the per-stage *list order itself* encodes the schedule, the 1F1B
zig-zag is preserved structurally instead of being re-flattened.

One GPU == one stage == one thread. Each thread sets its CUDA device context
once and owns its stage's serial execution; independent stages genuinely
overlap because CUDA kernel launches release the GIL and each device has its
own default stream.

``overlap=False`` runs the same per-stage op lists sequentially in a valid
topological order, for debugging.
"""

from __future__ import annotations

import threading
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from .pipeline import (
    PerfettoTracer,
    PipelineResult,
    Stage,
    _FakeCompute,
    _build_1f1b_rank_ops,
    _build_gpipe_rank_ops,
    _build_staggered_1b1f_rank_ops,
)

__all__ = ["run_pipeline_relay"]


# ── Mailbox ───────────────────────────────────────────────────────────────────

class _Mailbox:
    """
    One-slot thread-safe handoff for a single (stage, microbatch) tensor.

    The producer calls :meth:`put` exactly once; the consumer calls :meth:`get`,
    which blocks until the value arrives. This is the relay baton: a forward
    activation for stage ``s``/mb, or a backward gradient for stage ``s``/mb.

    Cross-thread CUDA safety: a tensor produced on the producer's device/stream
    is only *enqueued* there — the host thread returns before the kernels
    finish. To stop the consumer from reading a half-written tensor on its own
    stream, the producer records a :class:`torch.cuda.Event` on its current
    stream after producing, and the consumer makes its own current stream wait
    on that event before using the tensor. On CPU this is a no-op.
    """

    __slots__ = ("_ready", "_value", "_cuda_event")

    def __init__(self):
        self._ready = threading.Event()
        self._value: Optional[torch.Tensor] = None
        self._cuda_event: Optional[torch.cuda.Event] = None

    def put(self, value: Optional[torch.Tensor]):
        if value is not None and value.is_cuda:
            ev = torch.cuda.Event()
            ev.record()  # on the producer's current stream
            self._cuda_event = ev
        self._value = value
        self._ready.set()

    def get(self) -> Optional[torch.Tensor]:
        self._ready.wait()
        if self._cuda_event is not None:
            # Order the consumer's current stream after the producer's work.
            torch.cuda.current_stream().wait_event(self._cuda_event)
        return self._value


# ── Relay worker ─────────────────────────────────────────────────────────────

class _RelayWorker(threading.Thread):
    """
    One thread per stage. Walks the stage's static op list in order.

    The worker blocks only on:
      * its own forward cache (a ``B`` needs the matching ``F`` first — but the
        op list already guarantees that ordering within the stage), and
      * cross-stage mailboxes (the relay handoff).

    It never consults a central scheduler, so the schedule's interleave is
    exactly the order ops appear in ``ops``.
    """

    def __init__(
        self,
        stage: Stage,
        ops: List[Tuple[str, int]],
        micro_batches: List[torch.Tensor],
        targets: Optional[List[torch.Tensor]],
        loss_fn: Callable,
        fwd_in: List[List[_Mailbox]],   # fwd_in[s][mb]: activation INTO stage s for mb
        fwd_out: List[List[_Mailbox]],  # fwd_out[s][mb]: activation OUT of stage s for mb
        bwd_in: List[List[_Mailbox]],   # bwd_in[s][mb]: grad INTO stage s for mb
        bwd_out: List[List[_Mailbox]],  # bwd_out[s][mb]: grad OUT of stage s for mb
    ):
        super().__init__(daemon=True, name=f"relay-stage-{stage.stage_index}")
        self.stage = stage
        self.ops = ops
        self.micro_batches = micro_batches
        self.targets = targets
        self.loss_fn = loss_fn
        self.fwd_in = fwd_in
        self.fwd_out = fwd_out
        self.bwd_in = bwd_in
        self.bwd_out = bwd_out
        self.error: Optional[BaseException] = None
        self._pending_loss: Dict[int, torch.Tensor] = {}  # mb -> loss (last stage)

    def run(self):
        s = self.stage.stage_index
        dev = self.stage.device
        try:
            if dev.type == "cuda":
                torch.cuda.set_device(dev)
            for kind, mb in self.ops:
                if kind == "F":
                    self._do_forward(s, mb)
                elif kind == "W":
                    self._do_loss_wait(s, mb)
                elif kind == "B":
                    self._do_backward(s, mb)
                else:
                    raise ValueError(f"unknown op kind: {kind!r}")
        except BaseException as e:  # noqa: BLE001 - propagate to orchestrator
            self.error = e

    # ------------------------------------------------------------------
    def _do_forward(self, s: int, mb: int):
        if self.stage.is_first:
            x = self.micro_batches[mb]
        else:
            x = self.fwd_in[s][mb].get()  # relay: wait for activation (s-1, mb)
        out = self.stage.forward_one_chunk(mb, x)
        if not self.stage.is_last:
            self.fwd_out[s][mb].put(out)  # relay: hand activation to (s+1, mb)

    def _do_loss_wait(self, s: int, mb: int):
        # Last stage computes the loss from its own cached forward. The forward
        # for this mb already ran earlier in this stage's list (the list order
        # guarantees it), so the cache is populated. We compute the loss here so
        # the backward can run as a separate op later in the list.
        loss_fn = self.loss_fn
        target = self.targets[mb] if self.targets is not None else None
        with self.stage._lock:
            inp, out = self.stage._cache[mb]
        loss = loss_fn(out, target)
        # Stash for the subsequent B op and for harvesting.
        self.stage.losses[mb] = loss.detach()
        self._pending_loss[mb] = loss

    def _do_backward(self, s: int, mb: int):
        if self.stage.is_last:
            loss = self._pending_loss.pop(mb)
            grad_in = self.stage.backward_one_chunk(mb, loss=loss)
        else:
            grad_output = self.bwd_in[s][mb].get()  # relay: wait for grad (s+1, mb)
            grad_in = self.stage.backward_one_chunk(mb, grad_output=grad_output)
        if not self.stage.is_first:
            self.bwd_out[s][mb].put(grad_in)  # relay: hand grad to (s-1, mb)


# ── Relay engine ──────────────────────────────────────────────────────────────

class _RelayEngine:
    """
    Builds mailboxes + one worker per stage and runs them to completion.

    The engine itself does no scheduling — it only constructs the relay wiring,
    starts the threads, and harvests results.
    """

    def __init__(
        self,
        stages: List[Stage],
        rank_ops: List[List[Tuple[str, int]]],
        micro_batches: List[torch.Tensor],
        targets: Optional[List[torch.Tensor]],
        loss_fn: Callable,
    ):
        self.stages = stages
        self.p = len(stages)
        self.m = len(micro_batches)

        # One mailbox PER (pipeline edge, microbatch). Sharing a single slot
        # across microbatches lets a fast producer overwrite activations/grads
        # before the consumer reads them (the exact bookkeeping bug this
        # redesign set out to avoid).
        #
        # fwd_edge[e][mb]: activation flowing stage e -> e+1 for microbatch mb.
        # bwd_edge[e][mb]: grad flowing stage e+1 -> e for microbatch mb.
        # There are p-1 edges, indexed e = 0 .. p-2.
        fwd_edge = [[_Mailbox() for _ in range(self.m)] for _ in range(self.p - 1)]
        bwd_edge = [[_Mailbox() for _ in range(self.m)] for _ in range(self.p - 1)]

        # Per-stage views (indexed by stage, then mb). Only valid edges are set;
        # unused entries are None and never touched because the op list only
        # reads/writes real edges.
        fwd_in = [None] + [fwd_edge[e] for e in range(self.p - 1)]   # stage s reads edge s-1
        fwd_out = [fwd_edge[e] for e in range(self.p - 1)] + [None]  # stage s writes edge s
        bwd_in = [bwd_edge[e] for e in range(self.p - 1)] + [None]   # stage s reads edge s
        bwd_out = [None] + [bwd_edge[e] for e in range(self.p - 1)]  # stage s writes edge s-1

        self.workers = [
            _RelayWorker(
                stage=stages[s],
                ops=rank_ops[s],
                micro_batches=micro_batches,
                targets=targets,
                loss_fn=loss_fn,
                fwd_in=fwd_in,
                fwd_out=fwd_out,
                bwd_in=bwd_in,
                bwd_out=bwd_out,
            )
            for s in range(self.p)
        ]

    def run(self) -> None:
        for w in self.workers:
            w.start()
        for w in self.workers:
            w.join()
        for w in self.workers:
            if w.error is not None:
                raise RuntimeError(
                    f"relay worker (stage {w.stage.stage_index}) failed"
                ) from w.error
        # Barrier: a worker thread returns as soon as it has *enqueued* its last
        # CUDA kernel, not when that kernel has finished on the device. Without
        # this sync, run() can return while trailing kernels are still in flight
        # — which both risks use-after-return of staged tensors and causes
        # profilers/tracers to cut off the tail of the computation. Block until
        # every stage's device has drained its stream.
        for dev in {st.device for st in self.stages if st.device.type == "cuda"}:
            torch.cuda.synchronize(dev)


# ── Sequential (debug) executor ───────────────────────────────────────────────

def _run_sequential(
    stages: List[Stage],
    rank_ops: List[List[Tuple[str, int]]],
    micro_batches: List[torch.Tensor],
    targets: Optional[List[torch.Tensor]],
    loss_fn: Callable,
):
    """
    Execute the same per-stage op lists without threads. Ops are interleaved in
    dependency order: repeatedly scan stages in a round-robin and run the next
    op whose cross-stage input is already available. Deterministic; for
    debugging only.
    """
    p = len(stages)
    m = len(micro_batches)
    fwd_done: Dict[Tuple[int, int], torch.Tensor] = {}
    bwd_done: Dict[Tuple[int, int], Optional[torch.Tensor]] = {}
    losses: Dict[int, torch.Tensor] = {}

    cursors = [0] * p

    def _ready(s: int) -> bool:
        if cursors[s] >= len(rank_ops[s]):
            return False
        kind, mb = rank_ops[s][cursors[s]]
        if kind == "F":
            return s == 0 or (s - 1, mb) in fwd_done
        if kind == "W":
            with stages[s]._lock:
                return mb in stages[s]._cache
        # B: every backward records its input-grad into bwd_done[(s, mb)] so the
        # stage below (s-1) can consume it. The last stage's grad signal is its
        # locally-computed loss (keyed by mb), not a cross-stage tensor.
        if s == p - 1:
            return mb in losses
        return (s + 1, mb) in bwd_done

    remaining = sum(len(ops) for ops in rank_ops)
    while remaining > 0:
        progressed = False
        for s in range(p):
            while _ready(s):
                kind, mb = rank_ops[s][cursors[s]]
                cursors[s] += 1
                remaining -= 1
                progressed = True
                if kind == "F":
                    x = micro_batches[mb] if s == 0 else fwd_done[(s - 1, mb)]
                    out = stages[s].forward_one_chunk(mb, x)
                    if s < p - 1:
                        fwd_done[(s, mb)] = out
                elif kind == "W":
                    tgt = targets[mb] if targets is not None else None
                    with stages[s]._lock:
                        _, out = stages[s]._cache[mb]
                    loss = loss_fn(out, tgt)
                    stages[s].losses[mb] = loss.detach()
                    losses[mb] = loss
                else:  # B
                    if s == p - 1:
                        g = stages[s].backward_one_chunk(mb, loss=losses[mb])
                    else:
                        g = stages[s].backward_one_chunk(
                            mb, grad_output=bwd_done[(s + 1, mb)]
                        )
                    # Record this stage's input-grad for the stage below.
                    if s > 0:
                        bwd_done[(s, mb)] = g
        if not progressed:
            # Should not happen with a valid schedule; guard against deadlock.
            raise RuntimeError("sequential executor made no progress (dead schedule)")


# ── Public API ────────────────────────────────────────────────────────────────

def run_pipeline_relay(
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
    trace_path: Optional[str] = None,
    profile_path: Optional[str] = None,
) -> PipelineResult:
    """
    Split ``model`` and run one pipeline-parallel step using the relay executor.

    Signature and semantics mirror :func:`ramtorch.pipeline.run_pipeline`, but
    execution uses one thread per stage walking a static per-stage op list, so
    the requested schedule (GPipe / 1F1B) is honored exactly.

    Parameters
    ----------
    model          : full model (traced/split; stage modules moved to devices)
    example_input  : one example microbatch input for the tracer
    split_spec     : ``{layer_name: SplitPoint.BEGINNING|END}`` for pipeline()
    data           : full batch input tensor (dim 0 split into microbatches)
    targets        : optional full-batch targets, split the same way
    schedule       : "gpipe" | "1f1b"
    n_microbatches : number of microbatches
    loss_fn        : callable(output, target) -> scalar loss (default: sum)
    devices        : one device per stage (default: cuda:i round-robin / CPU)
    fake_compute   : None | "replace" | {"fwd": s|[s...], "bwd": s|[s...]}
    overlap        : per-stage worker threads (True) or sequential debug (False)
    trace_path     : if set, write a Chrome-trace JSON of op-level spans
    profile_path   : if set, capture a torch.profiler (kineto) trace of the step

    Returns
    -------
    PipelineResult with per-microbatch outputs/losses and stages holding
    accumulated param grads (``result.flush_grads()`` writes them to ``.grad``).
    """
    from torch.distributed.pipelining import pipeline as _split_pipeline

    if schedule not in ("gpipe", "1f1b", "staggered_1b1f"):
        raise ValueError(
            f"run_pipeline_relay supports 'gpipe', '1f1b' and 'staggered_1b1f', "
            f"got {schedule!r} (interleaved deferred until the orchestrator is proven)"
        )

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
    m = len(mbs)

    for st in stages:
        st.clear()
        st.zero_grad_acc()

    # ── Static per-stage execution order ──────────────────────────────────────
    if schedule == "gpipe":
        rank_ops = _build_gpipe_rank_ops(num_stages, m)
    elif schedule == "1f1b":
        rank_ops = _build_1f1b_rank_ops(num_stages, m)
    else:  # staggered_1b1f
        rank_ops = _build_staggered_1b1f_rank_ops(num_stages, m)

    if profile_path:
        from torch.profiler import ProfilerActivity, profile as _torch_profile
        with _torch_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as _prof:
            outputs, losses = _run_and_capture(stages, m, overlap, rank_ops,
                                               mbs, tgts, loss_fn)
            # Drain all device work BEFORE the profiler stops. CUDA kernels are
            # timestamped when they execute on the device; if the profiler is
            # torn down while trailing kernels are still enqueued, the tail of
            # the computation is lost from the trace. Synchronize inside the
            # profile region so the full timeline is captured.
            for d in {st.device for st in stages if st.device.type == "cuda"}:
                torch.cuda.synchronize(d)
        _prof.export_chrome_trace(profile_path)
    else:
        outputs, losses = _run_and_capture(stages, m, overlap, rank_ops,
                                           mbs, tgts, loss_fn)

    # Make sure all device work is done before reporting/exporting timings.
    for d in {st.device for st in stages if st.device.type == "cuda"}:
        torch.cuda.synchronize(d)

    if tracer is not None:
        tracer.export(trace_path)

    return PipelineResult(outputs, losses, stages)


def _run_and_capture(stages, m, overlap, rank_ops, mbs, tgts, loss_fn):
    """
    Run the step and capture last-stage forward outputs.

    ``forward_one_chunk`` stores ``(input, output)`` in the stage cache and the
    matching backward pops it. To return outputs to the caller we snapshot each
    last-stage forward output as it is produced, by wrapping the last stage's
    forward for the duration of the step.
    """
    last = stages[-1]
    captured: Dict[int, torch.Tensor] = {}
    orig_forward = last.forward_one_chunk

    def _capturing_forward(mb_index, x):
        out = orig_forward(mb_index, x)
        captured[mb_index] = out.detach()
        return out

    last.forward_one_chunk = _capturing_forward
    try:
        if overlap:
            engine = _RelayEngine(stages, rank_ops, mbs, tgts, loss_fn)
            engine.run()
        else:
            _run_sequential(stages, rank_ops, mbs, tgts, loss_fn)
    finally:
        last.forward_one_chunk = orig_forward

    outputs = [captured[i] for i in range(m)]
    losses = [last.losses[i] for i in range(m)]
    return outputs, losses
