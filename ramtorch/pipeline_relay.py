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

__all__ = ["run_pipeline_relay", "Pipeline"]


def _normalize_autocast(autocast) -> Optional[torch.dtype]:
    """Normalize the ``autocast`` option (None | dtype | 'bf16' | 'fp16')."""
    if autocast is None:
        return None
    if isinstance(autocast, str):
        aliases = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }
        key = autocast.lower()
        if key not in aliases:
            raise ValueError(
                f"unknown autocast dtype {autocast!r}; expected 'bf16', 'fp16', "
                f"or a torch.dtype"
            )
        return aliases[key]
    if isinstance(autocast, torch.dtype):
        return autocast
    raise ValueError(
        f"autocast must be None, a torch.dtype, or 'bf16'/'fp16'; got "
        f"{type(autocast).__name__}"
    )


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

    __slots__ = ("_ready", "_value", "_cuda_events")

    def __init__(self):
        self._ready = threading.Event()
        self._value = None
        self._cuda_events: List[torch.cuda.Event] = []

    def put(self, value):
        # ``value`` may be a single tensor, a tuple/list of tensors (multi-arg
        # stage output / grad), or None. Record one CUDA event per CUDA tensor
        # so the consumer's stream waits for every element to finish writing.
        tensors = (
            [t for t in value if isinstance(t, torch.Tensor)]
            if isinstance(value, (tuple, list))
            else [value]
        )
        for t in tensors:
            if t is not None and t.is_cuda:
                ev = torch.cuda.Event()
                ev.record()  # on the producer's current stream
                self._cuda_events.append(ev)
        self._value = value
        self._ready.set()

    def release(self):
        """Wake any blocked :meth:`get` with no value (used on abort)."""
        self._value = None
        self._ready.set()

    def get(self):
        self._ready.wait()
        for ev in self._cuda_events:
            # Order the consumer's current stream after the producer's work.
            torch.cuda.current_stream().wait_event(ev)
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
        abort: threading.Event,          # set by any failing worker to stop the rest
        on_error,                        # callback(worker) invoked on failure
        grad_outputs: Optional[List] = None,  # per-mb bypass grad (tensor/tuple/callable)
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
        self.abort = abort
        self._on_error = on_error
        self.error: Optional[BaseException] = None
        self._pending_loss: Dict[int, torch.Tensor] = {}  # mb -> loss (last stage)
        # Grad-bypass escape hatch: per-microbatch dL/dOutput for the last
        # stage, supplied directly instead of a scalar loss. Each entry is a
        # tensor/tuple (pre-sharded) or a callable (out, target)->grad resolved
        # at the W op once the forward output is cached.
        self.grad_outputs = grad_outputs
        self._pending_grad: Dict[int, object] = {}  # mb -> resolved grad (last stage)

    def _check_abort(self):
        if self.abort.is_set():
            raise RuntimeError(
                f"relay aborted (stage {self.stage.stage_index} stopping because "
                f"another stage failed)"
            )

    def run(self):
        s = self.stage.stage_index
        dev = self.stage.device
        try:
            if dev.type == "cuda":
                torch.cuda.set_device(dev)
            for kind, mb in self.ops:
                self._check_abort()
                if kind == "F":
                    self._do_forward(s, mb)
                elif kind == "W":
                    self._do_loss_wait(s, mb)
                elif kind == "B":
                    self._do_backward(s, mb)
                else:
                    raise ValueError(f"unknown op kind: {kind!r}")
        except BaseException as e:  # noqa: BLE001 - propagate to orchestrator
            # Record the error and signal the engine to abort the other workers,
            # releasing every mailbox so no stage stays blocked on get() forever.
            if self.error is None:
                self.error = e
            self._on_error(self)

    # ------------------------------------------------------------------
    def _do_forward(self, s: int, mb: int):
        if self.stage.is_first:
            x = self.micro_batches[mb]
        else:
            x = self.fwd_in[s][mb].get()  # relay: wait for activation (s-1, mb)
            self._check_abort()           # released mailbox (abort) returns None
        out = self.stage.forward_one_chunk(mb, x)
        if not self.stage.is_last:
            self.fwd_out[s][mb].put(out)  # relay: hand activation to (s+1, mb)

    def _do_loss_wait(self, s: int, mb: int):
        # Last stage resolves its backward seed from its own cached forward. The
        # forward for this mb already ran earlier in this stage's list (the list
        # order guarantees it), so the cache is populated. We resolve here so the
        # backward can run as a separate op later in the list.
        target = self.targets[mb] if self.targets is not None else None
        with self.stage._lock:
            inp, outs, _needs_grad = self.stage._cache[mb]
        # Reconstruct the module's original output form (single tensor or tuple)
        # so loss_fn / a grad callable sees exactly what the module returned.
        out = outs if len(outs) != 1 else outs[0]
        if self.grad_outputs is not None:
            # Grad-bypass: no scalar loss. Resolve the per-mb gradient (a
            # callable is invoked here, on the worker, with the live output;
            # a pre-sharded tensor/tuple is used as-is) and stash it for B.
            g = self.grad_outputs[mb]
            if callable(g):
                g = g(out, target)
            self._pending_grad[mb] = g
            return
        loss_fn = self.loss_fn
        loss = self.stage.compute_loss(out, loss_fn, target)
        # Stash for the subsequent B op and for harvesting.
        self.stage.losses[mb] = loss.detach()
        self._pending_loss[mb] = loss

    def _do_backward(self, s: int, mb: int):
        if self.stage.is_last:
            if self.grad_outputs is not None:
                # Grad-bypass: backprop the user-supplied dL/dOutput directly.
                g = self._pending_grad.pop(mb)
                grad_in = self.stage.backward_one_chunk(mb, grad_output=g)
            else:
                loss = self._pending_loss.pop(mb)
                grad_in = self.stage.backward_one_chunk(mb, loss=loss)
        else:
            grad_output = self.bwd_in[s][mb].get()  # relay: wait for grad (s+1, mb)
            self._check_abort()                   # released mailbox (abort)
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
        grad_outputs: Optional[List] = None,
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

        # Abort wiring: if any worker fails, set `abort` and RELEASE every mailbox
        # so blocked get()s wake up and the other workers exit too (instead of
        # deadlocking the join on mailboxes that will never be filled).
        self.abort = threading.Event()
        all_boxes = [box for row in fwd_edge for box in row] + \
                    [box for row in bwd_edge for box in row]

        def _on_error(_worker):
            self.abort.set()
            for box in all_boxes:
                box.release()

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
                abort=self.abort,
                on_error=_on_error,
                grad_outputs=grad_outputs,
            )
            for s in range(self.p)
        ]

    def run(self) -> None:
        for w in self.workers:
            w.start()
        for w in self.workers:
            w.join()
        # Surface the FIRST real worker error (not the secondary abort errors
        # raised by stages that were unblocked by the abort).
        first_error = None
        for w in self.workers:
            if w.error is not None and not (
                isinstance(w.error, RuntimeError) and "relay aborted" in str(w.error)
            ):
                first_error = w.error
                break
        if first_error is not None:
            raise RuntimeError(
                f"relay worker failed: {first_error}"
            ) from first_error
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
    grad_outputs: Optional[List] = None,
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
    bwd_seeds: Dict[int, object] = {}  # mb -> loss OR bypass grad (last stage)

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
        # locally-resolved backward seed (loss OR bypass grad, keyed by mb).
        if s == p - 1:
            return mb in bwd_seeds
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
                        _, outs, _ng = stages[s]._cache[mb]
                    out = outs if len(outs) != 1 else outs[0]
                    if grad_outputs is not None:
                        # Grad-bypass: resolve the per-mb gradient (callable is
                        # invoked with the live output; tensor/tuple used as-is).
                        g = grad_outputs[mb]
                        bwd_seeds[mb] = g(out, tgt) if callable(g) else g
                    else:
                        loss = stages[s].compute_loss(out, loss_fn, tgt)
                        stages[s].losses[mb] = loss.detach()
                        losses[mb] = loss
                        bwd_seeds[mb] = loss
                else:  # B
                    if s == p - 1:
                        if grad_outputs is not None:
                            g = stages[s].backward_one_chunk(
                                mb, grad_output=bwd_seeds[mb]
                            )
                        else:
                            g = stages[s].backward_one_chunk(mb, loss=bwd_seeds[mb])
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


# ── Forward-only (inference) engine ───────────────────────────────────────────

class _InferWorker(threading.Thread):
    """
    One thread per stage for forward-only pipelined inference.

    Each stage walks microbatches in order: wait for the previous stage's
    activation (or take the local microbatch on stage 0), compute the forward
    under ``no_grad``, and hand the output to the next stage. Stage ``s+1``
    working on microbatch ``k`` overlaps with stage ``s`` on microbatch ``k+1``.
    """

    def __init__(self, stage, micro_batches, fwd_in, fwd_out, abort, on_error,
                 last_outputs, tracer):
        super().__init__(daemon=True, name=f"infer-stage-{stage.stage_index}")
        self.stage = stage
        self.micro_batches = micro_batches
        self.fwd_in = fwd_in
        self.fwd_out = fwd_out
        self.abort = abort
        self._on_error = on_error
        self.last_outputs = last_outputs  # dict mb -> output (last stage writes)
        self.tracer = tracer
        self.error: Optional[BaseException] = None

    def run(self):
        s = self.stage.stage_index
        dev = self.stage.device
        try:
            if dev.type == "cuda":
                torch.cuda.set_device(dev)
            for mb in range(len(self.micro_batches)):
                if self.abort.is_set():
                    return
                x = self.micro_batches[mb] if self.stage.is_first else self.fwd_in[s][mb].get()
                if self.abort.is_set():
                    return
                t0 = self.tracer._ts() if self.tracer else 0.0
                # ``x`` may be a single tensor or a tuple of tensors (multi-input
                # stage). Move each to the device and unpack into the module call.
                with self.stage._autocast_ctx():
                    if isinstance(x, (tuple, list)):
                        out = self.stage.module(*[t.to(dev) for t in x])
                    else:
                        out = self.stage.module(x.to(dev))
                if self.tracer:
                    self.tracer.record_cpu(
                        f"F mb{mb}", "fwd", f"stage {s}", t0, self.tracer._ts()
                    )
                if self.stage.is_last:
                    self.last_outputs[mb] = out
                else:
                    self.fwd_out[s][mb].put(out)
        except BaseException as e:  # noqa: BLE001
            if self.error is None:
                self.error = e
            self._on_error(self)


def _run_inference(
    stages: List[Stage],
    data,
    *,
    n_microbatches: int,
    trace_path: Optional[str],
    profile_path: Optional[str] = None,
):
    """
    Forward-only GPipe-style pipelined inference (no backward, no grad).

    Splits ``data`` into microbatches and relays them through the stages with one
    worker thread per stage, so stages stay busy on different microbatches
    concurrently.

    ``data`` accepts the same forms as :meth:`Pipeline.step` (tensor, flat tuple,
    or nested pre-diced tuple). The OUTPUT mirrors the input shape: a nested
    pre-diced input yields a nested tuple of per-microbatch outputs (so a
    downstream consumer can start on microbatch 0 without waiting for the rest);
    a tensor or flat-tuple input yields a single concatenated tensor (dim 0).

    If the batch size is not divisible by ``n_microbatches``, the input is padded
    up to the next multiple (by repeating rows) before sharding and the padding is
    sliced off the output afterward — so any batch size works.
    """
    p = len(stages)
    # Detect whether the input was pre-diced (nested) so we can mirror it in the
    # output. _shard_input normalizes all forms into per-microbatch inputs.
    was_prediced = (
        isinstance(data, (tuple, list))
        and len(data) == n_microbatches
        and all(isinstance(e, (torch.Tensor, tuple, list)) for e in data)
    )

    # ── Pad to a multiple of n_microbatches (tensor / flat-tuple inputs) ──────
    # Pre-diced (nested) inputs are already per-microbatch and skip this.
    pad_rows = 0
    if not was_prediced:
        def _pad_to_multiple(t: torch.Tensor) -> torch.Tensor:
            nonlocal pad_rows
            bs = t.shape[0]
            rem = bs % n_microbatches
            if rem == 0:
                return t
            pad_rows = n_microbatches - rem
            pad = t[:1].expand(pad_rows, *t.shape[1:])
            return torch.cat([t, pad], dim=0)

        if isinstance(data, torch.Tensor):
            data = _pad_to_multiple(data)
        elif isinstance(data, (tuple, list)) and all(
            isinstance(e, torch.Tensor) for e in data
        ):
            # Flat tuple of full-batch tensors: pad each element the same amount.
            data = type(data)(_pad_to_multiple(e) for e in data)

    mbs, _ = _shard_input(data, n_microbatches, stages[0].device)
    m = len(mbs)

    tracer = PerfettoTracer() if trace_path else None

    # Mailboxes for the forward relay, one per (edge, microbatch).
    fwd_edge = [[_Mailbox() for _ in range(m)] for _ in range(p - 1)]
    fwd_in = [None] + [fwd_edge[e] for e in range(p - 1)]
    fwd_out = [fwd_edge[e] for e in range(p - 1)] + [None]

    abort = threading.Event()
    all_boxes = [box for row in fwd_edge for box in row]

    def _on_error(_w):
        abort.set()
        for box in all_boxes:
            box.release()

    last_outputs: Dict[int, torch.Tensor] = {}
    workers = [
        _InferWorker(stages[s], mbs, fwd_in, fwd_out, abort, _on_error,
                     last_outputs, tracer)
        for s in range(p)
    ]

    def _run():
        for w in workers:
            w.start()
        for w in workers:
            w.join()

    if profile_path:
        from torch.profiler import ProfilerActivity, profile as _torch_profile
        with _torch_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as _prof:
            _run()
            # Drain device work BEFORE the profiler stops so the full kernel
            # timeline is captured.
            for d in {st.device for st in stages if st.device.type == "cuda"}:
                torch.cuda.synchronize(d)
        _prof.export_chrome_trace(profile_path)
    else:
        _run()

    for w in workers:
        if w.error is not None:
            raise RuntimeError(
                f"inference worker (stage {w.stage.stage_index}) failed"
            ) from w.error

    # Drain device work before reading outputs / exporting the trace.
    for d in {st.device for st in stages if st.device.type == "cuda"}:
        torch.cuda.synchronize(d)

    if tracer is not None:
        tracer.export(trace_path)

    if was_prediced:
        # Mirror the pre-diced input: return per-microbatch outputs as a nested
        # tuple (independent tensors), so downstream can start on mb0 eagerly.
        # A stage returning a tuple output yields a per-microbatch tuple here.
        return tuple(last_outputs[i] for i in range(m))

    def _concat_arg(vals):
        # Concatenate one output position across microbatches (dim 0), then
        # slice off any padding rows added for divisibility.
        out = torch.cat(vals, dim=0)
        if pad_rows > 0:
            out = out[: out.shape[0] - pad_rows]
        return out

    first = last_outputs[0]
    if isinstance(first, (tuple, list)):
        # Multi-output stage: concatenate each output arg independently.
        n_args = len(first)
        return tuple(
            _concat_arg([last_outputs[i][j] for i in range(m)])
            for j in range(n_args)
        )
    # Single-tensor output: single concatenated tensor (dim 0).
    return _concat_arg([last_outputs[i] for i in range(m)])


# ── Public API ────────────────────────────────────────────────────────────────

class _ChunkSequential(nn.Module):
    """Resident counterpart of an offloaded chunk list.

    Runs the chunks in order on one device, relaying a tuple output as the
    next chunk's positional args — the exact OffloadModel dicing convention —
    so the same chunk list works with ``offload=True`` or ``offload=False``
    without touching the chunks.
    """

    def __init__(self, chunks):
        super().__init__()
        self.chunks = nn.ModuleList(chunks)

    def forward(self, *args):
        out = args if len(args) != 1 else args[0]
        for c in self.chunks:
            out = c(*out) if isinstance(out, tuple) else c(out)
        return out


class Pipeline:
    """
    A reusable pipeline-parallel wrapper around ``model``.

    Splits the model ONCE (via ``torch.distributed.pipelining.pipeline``) and
    builds the per-device ``Stage`` objects ONCE, then :meth:`step` runs a
    pipeline schedule on each new batch without re-splitting. This is the
    correct entry point for iterative training: calling the one-shot
    :func:`run_pipeline_relay` repeatedly re-traces the model every step, which
    both is wasteful and fails (re-splitting an already-split model nests the
    traced forward and blows the recursion limit).

    Parameters
    ----------
    model          : full model (traced/split once; stage modules moved to devices).
                     Not required if ``stage_modules`` is given.
    example_input  : one example microbatch input for the tracer (must match the
                     model's current device; CPU model -> CPU example).
                     Not required if ``stage_modules`` is given.
    split_spec     : ``{layer_name: SplitPoint.BEGINNING|END}`` for pipeline().
                     Not required if ``stage_modules`` is given.
    stage_modules  : optional list of PRE-PARTITIONED ``nn.Module`` stages, one
                     per pipeline stage, in order (see also ``chunk_modules``
                     for the flat-list convenience form). When provided, the
                     ``torch.export`` tracer is **bypassed entirely** — use this
                     when the model is too complex for ``torch.export`` to trace
                     (graph breaks, data-dependent control flow, custom ops).
                     Each module's forward takes the previous stage's output and
                     returns its own output, exactly like a manually-split model.
                     ``model``/``example_input``/``split_spec`` are ignored.
                     An entry that is a LIST/TUPLE of modules becomes an
                     **offloaded stage** (:class:`ramtorch.pipeline_offload.
                     OffloadStage`): the stage model is chunked, its masters
                     are relocated to CPU pinned memory, and weights stream
                     through a sliding GPU window prefetched in the schedule's
                     exact chunk order (chunk ``i+1`` consumes chunk ``i``'s
                     output — the OffloadModel dicing convention). Mixed
                     pipelines (some stages offloaded, some plain) are fine.
    chunk_modules  : the simplest way to build a chunked pipeline: a FLAT
                     ordered list of chunk modules for the WHOLE model ("here,
                     I have 32 chunks"). The pipeline dices it into one stage
                     per device — evenly by count (earlier stages take the
                     remainder), or per ``chunks_per_stage``. Stages stream
                     their weights when ``offload=True`` (default) or stay
                     fully resident when ``offload=False``.
                     Mutually exclusive with ``stage_modules``/``model``.
    chunks_per_stage :
                     optional explicit per-stage chunk counts for
                     ``chunk_modules``, e.g. ``[8, 8, 8, 8]`` or ``[7, 8, 8, 9]``
                     (must sum to ``len(chunk_modules)``; its length sets the
                     stage count and must match ``devices`` if both are given).
    offload        : whether chunked stages STREAM their weights (default
                     True). With ``offload=False`` the same chunk lists (flat
                     ``chunk_modules`` or nested ``stage_modules`` entries)
                     build plain RESIDENT stages instead — each stage runs its
                     chunks in sequence on its GPU with all weights resident.
                     Same convenient dicing, no streaming machinery.
    offload_window : streaming GPU slots per offloaded stage (default 2;
                     >= 2 overlaps weight loads with compute).
    offload_pin    : chunks per offloaded stage pinned permanently on its GPU
                     (evenly spaced; default 0). GPU weight memory per
                     offloaded stage ~= ``offload_window + offload_pin`` chunks.
    offload_keep_activations :
                     backward strategy for offloaded stages: ``True``
                     (default — keep per-chunk graphs, plain-pipeline-like
                     activation memory) or ``"checkpoint"`` (per-chunk
                     non-reentrant checkpoint: recompute-level memory,
                     dropout-safe). Engine recompute mode is not supported
                     inside the pipeline.
    offload_grad_accum :
                     "stream" (default): grad accumulators live on the GPU in
                     ``offload_acc_slots`` streaming slots — zero CPU math;
                     evicted accs spill over D2H and reload over H2D like
                     weights. "cpu": legacy per-microbatch D2H packet + CPU
                     accumulate (use when even the acc slots don't fit).
    offload_acc_slots :
                     grad-acc residency slots per offloaded stage (default =
                     ``offload_window``). >= streamed chunks per stage keeps
                     every acc GPU-resident for the whole step (one spill per
                     chunk at flush time).
    devices        : one device per stage (default: cuda:i round-robin / CPU)
    fake_compute   : None | "replace" | {"fwd": s|[s...], "bwd": s|[s...]}
    overlap        : per-stage worker threads (True) or sequential debug (False)
    autocast       : mixed precision: None (off, default), a ``torch.dtype``,
                     or the strings ``"bf16"`` / ``"fp16"``. Each stage worker
                     enters ``torch.autocast(device_type, dtype)`` around its
                     forward and loss computation (autocast is thread-local, so
                     wrapping ``step()`` in the caller does nothing). Backward
                     runs outside autocast; params/grads stay fp32. fp16 is
                     INFERENCE-ONLY: ``step()`` raises because fp16 training
                     needs loss scaling (unsupported) — use bf16.

    After construction, ``self.stages`` holds the Stage objects (which own the
    gradient accumulators). With the traced path, the original model's parameters
    are shared with those stages — so a single optimizer over ``model.parameters()``
    sees the accumulated grads after ``step()`` + :meth:`flush_grads`. With
    ``stage_modules``, optimize over the modules' own parameters (e.g.
    ``itertools.chain(*(m.parameters() for m in stage_modules))``).
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        example_input: Optional[torch.Tensor] = None,
        split_spec: Optional[dict] = None,
        *,
        stage_modules: Optional[Sequence[nn.Module]] = None,
        chunk_modules: Optional[Sequence[nn.Module]] = None,
        chunks_per_stage: Optional[Sequence[int]] = None,
        devices: Optional[Sequence[Union[str, torch.device]]] = None,
        fake_compute=None,
        overlap: bool = True,
        autocast=None,
        offload: bool = True,
        offload_window: int = 2,
        offload_pin: int = 0,
        offload_keep_activations: Union[bool, str] = True,
        offload_grad_accum: str = "stream",
        offload_acc_slots: Optional[int] = None,
    ):
        self.model = model
        self.overlap = overlap
        self.autocast_dtype = _normalize_autocast(autocast)

        if chunk_modules is not None:
            # ── Flat-chunk convenience: dice into offloaded stages ────────────
            if stage_modules is not None or model is not None:
                raise ValueError(
                    "chunk_modules is mutually exclusive with stage_modules "
                    "and model — pass ONE flat list of chunks and let the "
                    "pipeline split it"
                )
            stage_modules = self._split_chunks(
                chunk_modules, chunks_per_stage, devices
            )
        elif chunks_per_stage is not None:
            raise ValueError("chunks_per_stage requires chunk_modules")

        # Tuple / nested-tuple inputs are only supported on the manual
        # (stage_modules) path; the traced torch.export path does not support them.
        self._manual = stage_modules is not None

        if stage_modules is not None:
            # ── Manual path: bypass torch.export entirely ─────────────────────
            stage_modules = list(stage_modules)
            if len(stage_modules) < 1:
                raise ValueError("stage_modules must contain at least one module")
            self.num_stages = len(stage_modules)
            self.devices = self._resolve_devices(devices)
            chunked = [isinstance(mod, (list, tuple)) for mod in stage_modules]
            if offload and any(chunked) and fake_compute is not None:
                raise ValueError(
                    "fake_compute is not supported with offloaded (chunked) "
                    "stage_modules entries (pass offload=False for resident "
                    "chunked stages)"
                )
            fake = (
                _FakeCompute(fake_compute, self.num_stages)
                if fake_compute is not None
                else None
            )
            self.stages = []
            for i, mod in enumerate(stage_modules):
                if chunked[i] and offload:
                    from .pipeline_offload import OffloadStage

                    self.stages.append(OffloadStage(
                        mod, i, self.num_stages, device=self.devices[i],
                        tracer=None, autocast_dtype=self.autocast_dtype,
                        window=offload_window, pin=offload_pin,
                        keep_activations=offload_keep_activations,
                        grad_accum=offload_grad_accum,
                        acc_slots=offload_acc_slots,
                    ))
                else:
                    if chunked[i]:
                        # offload=False: same chunk list, fully resident
                        mod = _ChunkSequential(mod)
                    self.stages.append(Stage(
                        mod, i, self.num_stages, device=self.devices[i],
                        fake=fake, tracer=None,
                        autocast_dtype=self.autocast_dtype,
                    ))
            return

        # ── Traced path: split the full model via torch.export ────────────────
        if model is None or example_input is None or split_spec is None:
            raise ValueError(
                "either provide stage_modules (pre-partitioned stages) OR all of "
                "model, example_input, and split_spec (to trace+split)"
            )
        from torch.distributed.pipelining import pipeline as _split_pipeline

        pipe = _split_pipeline(
            module=model, mb_args=(example_input,), split_spec=split_spec
        )
        self.num_stages = pipe.num_stages
        self.devices = self._resolve_devices(devices)

        fake = (
            _FakeCompute(fake_compute, self.num_stages)
            if fake_compute is not None
            else None
        )

        self.stages = [
            Stage(pipe.get_stage_module(i), i, self.num_stages, device=self.devices[i],
                  fake=fake, tracer=None, autocast_dtype=self.autocast_dtype)
            for i in range(self.num_stages)
        ]

    @staticmethod
    def _split_chunks(chunk_modules, chunks_per_stage, devices):
        """Dice a flat chunk list into per-stage chunk lists (offloaded stages).

        Stage count comes from ``chunks_per_stage`` if given, else ``devices``,
        else every visible CUDA device (1 CPU stage without CUDA). An even
        split hands the remainder to the EARLIER stages (the build_stages
        convention).
        """
        chunk_modules = list(chunk_modules)
        if len(chunk_modules) < 1:
            raise ValueError("chunk_modules must contain at least one module")
        for i, m in enumerate(chunk_modules):
            if isinstance(m, (list, tuple)):
                raise ValueError(
                    f"chunk_modules must be a FLAT list of modules; entry {i} "
                    "is a list/tuple (nested per-stage lists belong in "
                    "stage_modules)"
                )

        if chunks_per_stage is not None:
            counts = [int(c) for c in chunks_per_stage]
            if devices is not None and len(devices) != len(counts):
                raise ValueError(
                    f"chunks_per_stage has {len(counts)} entries but "
                    f"{len(devices)} devices were given"
                )
            if any(c < 1 for c in counts):
                raise ValueError("every chunks_per_stage entry must be >= 1")
            if sum(counts) != len(chunk_modules):
                raise ValueError(
                    f"chunks_per_stage sums to {sum(counts)} but there are "
                    f"{len(chunk_modules)} chunk modules"
                )
        else:
            if devices is not None:
                n_stages = len(devices)
            else:
                n_cuda = (torch.cuda.device_count()
                          if torch.cuda.is_available() else 0)
                n_stages = n_cuda if n_cuda > 0 else 1
            if len(chunk_modules) < n_stages:
                raise ValueError(
                    f"{len(chunk_modules)} chunks cannot fill {n_stages} "
                    "stages (need at least one chunk per stage)"
                )
            base, rem = divmod(len(chunk_modules), n_stages)
            counts = [base + (1 if i < rem else 0) for i in range(n_stages)]

        it = iter(chunk_modules)
        return [[next(it) for _ in range(c)] for c in counts]

    def _resolve_devices(self, devices):
        if devices is None:
            n_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if n_cuda > 0:
                devices = [
                    torch.device("cuda", i % n_cuda) for i in range(self.num_stages)
                ]
            else:
                devices = [torch.device("cpu")] * self.num_stages
        else:
            devices = [torch.device(d) for d in devices]
            assert len(devices) == self.num_stages, (
                f"need {self.num_stages} devices, got {len(devices)}"
            )
        return devices

    def step(
        self,
        data: torch.Tensor,
        *,
        targets: Optional[torch.Tensor] = None,
        schedule: str = "1f1b",
        n_microbatches: int = 4,
        loss_fn: Optional[Callable] = None,
        trace_path: Optional[str] = None,
        profile_path: Optional[str] = None,
        grad_outputs=None,
    ) -> PipelineResult:
        """
        Run one pipeline-parallel step on ``data`` (dim 0 split into microbatches).

        Returns a :class:`PipelineResult`; call ``result.flush_grads()`` to write
        accumulated microbatch grads (mean-scaled) into ``.grad`` before an
        optimizer step. See :func:`run_pipeline_relay` for parameter details.

        ``data`` may be a tensor, a flat tuple of tensors (unpacked as positional
        args into the stage-0 module), or a nested pre-diced tuple of microbatches.
        Tuple forms require the manual ``stage_modules`` path.

        ``grad_outputs`` bypasses ``loss_fn``: backprop a user-supplied
        ``dL/dOutput`` directly into the last stage (mutually exclusive with
        ``loss_fn``; no loss is reported). See :func:`run_pipeline_relay`.
        """
        if self.autocast_dtype == torch.float16:
            raise ValueError(
                "fp16 autocast training is not supported: fp16 gradients "
                "underflow without loss scaling (GradScaler), which the "
                "pipeline does not integrate. Use autocast=torch.bfloat16 for "
                "training; fp16 remains available for infer()."
            )
        if isinstance(data, (tuple, list)) and not self._manual:
            raise ValueError(
                "tuple / nested-tuple pipeline inputs are only supported on the "
                "manual stage_modules path (Pipeline(stage_modules=[...])). The "
                "traced PipelineModel path (torch.export) does not support them."
            )
        return _run_step(
            self.stages,
            data,
            targets=targets,
            schedule=schedule,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            overlap=self.overlap,
            trace_path=trace_path,
            profile_path=profile_path,
            grad_outputs=grad_outputs,
        )

    def flush_grads(self, scale: Optional[float] = None, n_microbatches: int = 1):
        """Write accumulated grads into ``.grad`` (default scale 1/n_microbatches)."""
        s = scale if scale is not None else 1.0 / n_microbatches
        for st in self.stages:
            st.flush_grads(scale=s)

    def close(self):
        """Release stage background resources (offloaded stages' loader and
        writeback threads). Idempotent; plain stages no-op."""
        for st in self.stages:
            st.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @torch.no_grad()
    def infer(
        self,
        data,
        *,
        n_microbatches: int = 4,
        trace_path: Optional[str] = None,
        profile_path: Optional[str] = None,
    ):
        """
        Pipelined inference: a forward-only GPipe-style run (no backward).

        The batch is split into ``n_microbatches`` microbatches and relayed
        through the stages with one worker thread per stage, so while stage
        ``s+1`` computes microbatch ``k``, stage ``s`` is already computing
        microbatch ``k+1`` — keeping every GPU busy instead of the whole-batch
        sequential relay (which leaves all but one GPU idle at any moment).

        Runs under ``no_grad`` and does not retain activations.

        ``data`` accepts the same forms as :meth:`step` (tensor, flat tuple, or
        nested pre-diced tuple). The OUTPUT mirrors the input shape: a nested
        pre-diced input yields a nested tuple of per-microbatch outputs (so a
        downstream consumer can start on microbatch 0 without waiting for the
        rest); a tensor or flat-tuple input yields a single concatenated tensor.

        ``trace_path`` optionally writes an op-level Chrome-trace (Perfetto) of
        the forward spans so you can see the overlap. ``profile_path`` captures
        a full torch.profiler (kineto) trace of the run.
        """
        return _run_inference(
            self.stages, data, n_microbatches=n_microbatches,
            trace_path=trace_path, profile_path=profile_path,
        )


def _shard_input(data, n_microbatches: int, device):
    """
    Normalize the pipeline input into a list of per-microbatch inputs.

    Accepted forms (manual ``stage_modules`` path only — see note below):
      * tensor                -> chunked into ``n_microbatches`` along dim 0;
                                 each microbatch is a tensor.
      * (t0, t1, ...)         -> FLAT tuple of tensors, each chunked along dim 0;
                                 each microbatch is a tuple ``(t0_mb, t1_mb, ...)``
                                 passed to the stage-0 module as *positional args*.
      * ((mb0...), (mb1...))  -> NESTED tuple = PRE-DICED microbatches. The outer
                                 tuple has one entry per microbatch; each entry is
                                 itself a tensor or a tuple of tensors. These are
                                 used as-is (independent microbatches, no shared
                                 storage), so downstream consumers can start on
                                 microbatch 0 without waiting for the rest.

    Returns ``(mbs, is_tuple)`` where ``mbs[mb]`` is the input for microbatch mb
    (a tensor, or a tuple of tensors to be unpacked as positional args), and
    ``is_tuple`` tells the caller whether to unpack on the way into the module.

    NOTE: tuple / nested-tuple inputs are only supported on the manual
    ``stage_modules`` path. The traced ``PipelineModel`` path (torch.export) does
    NOT support them — ``Pipeline.__init__`` asserts this.
    """
    def _to_dev(t):
        return t.to(device) if isinstance(t, torch.Tensor) else t

    if isinstance(data, (tuple, list)):
        # NESTED tuple = pre-diced microbatches: outer length == n_microbatches
        # and every entry is itself a tuple/list of tensors. Each entry is one
        # independent microbatch (used as-is, no shared storage).
        if (
            len(data) == n_microbatches
            and all(isinstance(e, (tuple, list)) for e in data)
        ):
            mbs = [tuple(_to_dev(t) for t in e) for e in data]
            return mbs, True

        # NESTED tuple of single-tensor microbatches: outer length ==
        # n_microbatches and every entry is a tensor. Treat as pre-diced.
        if (
            len(data) == n_microbatches
            and all(isinstance(e, torch.Tensor) for e in data)
        ):
            mbs = [_to_dev(e) for e in data]
            return mbs, False

        # FLAT tuple of full-batch tensors: chunk each element per microbatch and
        # pass the per-microbatch tuple as positional args to the stage-0 module.
        if all(isinstance(e, torch.Tensor) for e in data):
            for e in data:
                assert e.shape[0] % n_microbatches == 0, (
                    f"tuple element batch {e.shape[0]} not divisible by "
                    f"n_microbatches={n_microbatches}"
                )
            per_elem = [e.chunk(n_microbatches, dim=0) for e in data]
            mbs = [
                tuple(_to_dev(per_elem[k][mb]) for k in range(len(data)))
                for mb in range(n_microbatches)
            ]
            return mbs, True

    # Single tensor.
    assert isinstance(data, torch.Tensor), (
        f"unsupported pipeline input type: {type(data)} "
        "(expected tensor, flat tuple of tensors, or nested pre-diced tuple)"
    )
    assert data.shape[0] % n_microbatches == 0, (
        f"batch {data.shape[0]} not divisible by n_microbatches={n_microbatches}"
    )
    mbs = [_to_dev(mb) for mb in data.chunk(n_microbatches, dim=0)]
    return mbs, False


def _run_step(
    stages: List[Stage],
    data,
    *,
    targets: Optional[torch.Tensor],
    schedule: str,
    n_microbatches: int,
    loss_fn: Optional[Callable],
    overlap: bool,
    trace_path: Optional[str],
    profile_path: Optional[str],
    grad_outputs=None,
) -> PipelineResult:
    """Run one pipeline step on pre-built stages (no re-split). Shared by
    ``Pipeline.step`` and the one-shot ``run_pipeline_relay``.

    ``data`` may be a tensor, a flat tuple of tensors (unpacked as positional
    args into the stage-0 module), or a nested pre-diced tuple of microbatches.
    See :func:`_shard_input`. Tuple forms require the manual ``stage_modules``
    path (not the traced ``PipelineModel`` path).

    ``grad_outputs`` is an escape hatch that BYPASSES ``loss_fn``: the last
    stage backprops a user-supplied ``dL/dOutput`` directly instead of
    computing a scalar loss. Mutually exclusive with ``loss_fn``. Accepts a
    callable ``(output, target) -> grad`` (resolved per-microbatch on the last
    stage worker, mirroring ``loss_fn``) or a full-batch tensor / flat tuple of
    tensors (chunked along dim 0 like ``targets``; a tuple is aligned to the
    last-stage module outputs, ``None`` at no-grad slots). No loss is reported.
    """
    if schedule not in ("gpipe", "1f1b", "staggered_1b1f"):
        raise ValueError(
            f"supported schedules: 'gpipe', '1f1b', 'staggered_1b1f'; got {schedule!r}"
        )
    if grad_outputs is not None and loss_fn is not None:
        raise ValueError(
            "grad_outputs and loss_fn are mutually exclusive: pass grad_outputs "
            "to bypass the loss and backprop a gradient directly, OR loss_fn to "
            "compute a scalar loss — not both."
        )
    loss_fn = loss_fn or (lambda out, _: out.sum())
    num_stages = len(stages)

    # Microbatch sharding (handles tensor / flat tuple / nested pre-diced tuple).
    mbs, is_tuple_input = _shard_input(data, n_microbatches, stages[0].device)
    last_dev = stages[-1].device
    tgts = (
        [t.to(last_dev) for t in targets.chunk(n_microbatches, dim=0)]
        if targets is not None
        else None
    )
    m = len(mbs)

    # Shard the bypass gradient per-microbatch (tensor / flat-tuple form). A
    # callable is replicated per microbatch and resolved lazily on the worker.
    grads = None
    if grad_outputs is not None:
        if callable(grad_outputs):
            grads = [grad_outputs] * m
        elif isinstance(grad_outputs, torch.Tensor):
            grads = [g.to(last_dev) for g in grad_outputs.chunk(m, dim=0)]
        elif isinstance(grad_outputs, (tuple, list)):
            per_elem = [e.chunk(m, dim=0) for e in grad_outputs]
            grads = [
                tuple(per_elem[k][mb].to(last_dev) for k in range(len(grad_outputs)))
                for mb in range(m)
            ]
        else:
            raise TypeError(
                f"grad_outputs must be a callable, a tensor, or a flat tuple of "
                f"tensors; got {type(grad_outputs).__name__}"
            )

    tracer = PerfettoTracer() if trace_path else None
    # Attach the tracer to stages for this step (stages are built with tracer=None
    # when reused; the one-shot path builds them fresh each call).
    for st in stages:
        st.tracer = tracer
        st.clear()
        st.zero_grad_acc()

    # Static per-stage execution order
    if schedule == "gpipe":
        rank_ops = _build_gpipe_rank_ops(num_stages, m)
    elif schedule == "1f1b":
        rank_ops = _build_1f1b_rank_ops(num_stages, m)
    else:  # staggered_1b1f
        rank_ops = _build_staggered_1b1f_rank_ops(num_stages, m)

    # Announce the step to every stage now that the op order is known.
    # Offloaded stages expand their op list into a chunk prefetch itinerary
    # here (plain stages no-op), so their loaders start prefetching before
    # the workers even launch.
    for s, st in enumerate(stages):
        st.begin_step(rank_ops[s], m)

    if profile_path:
        from torch.profiler import ProfilerActivity, profile as _torch_profile
        with _torch_profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=False,
            with_stack=False,
        ) as _prof:
            outputs, losses = _run_and_capture(stages, m, overlap, rank_ops,
                                               mbs, tgts, loss_fn,
                                               grad_outputs=grads)
            # Drain all device work BEFORE the profiler stops so the full kernel
            # timeline is captured.
            for d in {st.device for st in stages if st.device.type == "cuda"}:
                torch.cuda.synchronize(d)
        _prof.export_chrome_trace(profile_path)
    else:
        outputs, losses = _run_and_capture(stages, m, overlap, rank_ops,
                                           mbs, tgts, loss_fn,
                                           grad_outputs=grads)

    # Make sure all device work is done before reporting/exporting timings.
    for d in {st.device for st in stages if st.device.type == "cuda"}:
        torch.cuda.synchronize(d)

    if tracer is not None:
        tracer.export(trace_path)
        for st in stages:
            st.tracer = None

    return PipelineResult(outputs, losses, stages)


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
    grad_outputs=None,
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
    grad_outputs   : escape hatch that BYPASSES ``loss_fn``: the last stage
                     backprops a user-supplied ``dL/dOutput`` directly instead of
                     computing a scalar loss. Mutually exclusive with ``loss_fn``.
                     Accepts a callable ``(output, target) -> grad`` (resolved
                     per-microbatch on the last-stage worker) or a full-batch
                     tensor / flat tuple of tensors (chunked along dim 0 like
                     ``targets``; a tuple is aligned to the last-stage module
                     outputs, ``None`` at no-grad slots). No loss is reported
                     (``result.loss`` raises).

    Returns
    -------
    PipelineResult with per-microbatch outputs/losses and stages holding
    accumulated param grads (``result.flush_grads()`` writes them to ``.grad``).

    Note
    ----
    This splits the model every call. For iterative training, build a
    :class:`Pipeline` once and call its ``step()`` per batch instead — reusing
    the split avoids re-tracing (which is slow and fails on an already-split
    model).
    """
    pipe = Pipeline(
        model,
        example_input,
        split_spec,
        devices=devices,
        fake_compute=fake_compute,
        overlap=overlap,
    )
    return pipe.step(
        data,
        targets=targets,
        schedule=schedule,
        n_microbatches=n_microbatches,
        loss_fn=loss_fn,
        trace_path=trace_path,
        profile_path=profile_path,
        grad_outputs=grad_outputs,
    )


def _run_and_capture(stages, m, overlap, rank_ops, mbs, tgts, loss_fn,
                     grad_outputs=None):
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
        # The last stage may return a tuple of tensors (multi-output stage);
        # detach each element so the captured snapshot carries no graph.
        captured[mb_index] = (
            tuple(t.detach() for t in out)
            if isinstance(out, (tuple, list))
            else out.detach()
        )
        return out

    last.forward_one_chunk = _capturing_forward
    try:
        if overlap:
            engine = _RelayEngine(stages, rank_ops, mbs, tgts, loss_fn,
                                  grad_outputs=grad_outputs)
            engine.run()
        else:
            _run_sequential(stages, rank_ops, mbs, tgts, loss_fn,
                            grad_outputs=grad_outputs)
    finally:
        last.forward_one_chunk = orig_forward

    outputs = [captured[i] for i in range(m)]
    # Under the grad-bypass escape hatch (grad_outputs), no scalar loss is ever
    # computed, so last.losses is empty; report an empty losses list (the
    # PipelineResult.loss property raises a clear error in that case).
    losses = [last.losses[i] for i in range(m)] if last.losses else []
    return outputs, losses
