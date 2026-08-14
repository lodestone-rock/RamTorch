"""
ramtorch.pipeline_offload
-------------------------
Weight offloading INSIDE pipeline parallelism: an :class:`OffloadStage` is a
pipeline :class:`~ramtorch.pipeline.Stage` whose model is pre-diced into an
ordered list of layer chunks, with the chunk masters (weights, grad
accumulators) living in CPU pinned memory and streamed through a sliding GPU
window by the :class:`~ramtorch.offload.OffloadModel` engine — the same
loader-thread / Belady-window / writeback-thread machinery, driven by the
pipeline schedule instead of ``OffloadModel.step()``.

Why this composes cleanly (validated first in
:mod:`ramtorch.pipeline_offload_simulator`): the relay executor walks a
STATIC per-stage op list, so the entire step's chunk-touch order is known up
front. Each ``("F", mb)`` expands to chunks ``0..L-1`` and each ``("B", mb)``
to ``L-1..0``; :meth:`OffloadStage.begin_step` announces the whole expanded
itinerary to the engine's loader before the step starts, and the prefetcher
overlaps H2D weight loads with compute and with the pipeline bubbles. The
staggered_1b1f steady state (``... B F B F ...``) gives echo reuse for free —
the turnaround chunk is already resident.

Simulated cost (p=4, m=8, L=8, compute-bound): holding a window of 2 of 8
chunks per GPU (~25% of stage weight memory) costs ~0.4% makespan.

Backward strategies (the ``keep_activations`` knob of the engine):

* ``True`` (default): grad-enabled forward keeps each chunk's graph;
  weights still stream via the storage free/refill trick
  (``untyped_storage().resize_(0)`` on eviction, refilled by ``_acquire``
  before that chunk's backward). Activation memory scales with the number of
  in-flight microbatches (which the staggered_1b1f schedule bounds at
  ``p - s``) times all-chunk activations — the same shape as the plain
  pipeline's memory.
* ``"checkpoint"``: keep-mode plumbing but each chunk runs under
  non-reentrant ``torch.utils.checkpoint`` — internal activations dropped at
  forward and recomputed during that chunk's backward, RNG stashed/restored
  (dropout-safe). Recompute-level memory.
* Engine ``recompute`` mode (``keep_activations=False``) is NOT supported
  here: the relay computes the last stage's loss from the cached forward
  output at the ``W`` op, which recompute mode leaves graph-disconnected
  (its forward runs under ``no_grad``). ``"checkpoint"`` dominates it anyway
  (see docs/offload.md).

NO NVMe TIER: an offloaded pipeline stage always keeps its masters in CPU
pinned RAM. Sustained pipeline training from disk would rewrite the on-disk
masters every step across every stage — guaranteed drive thrashing — so this
integration deliberately does not expose ``nvme``/``nvme_path``.

Usage (via :class:`ramtorch.pipeline_relay.Pipeline` — a stage entry that is
a LIST of modules becomes an offloaded stage)::

    stage0 = [Block() for _ in range(8)]      # chunked -> offloaded
    stage1 = nn.Sequential(...)               # plain module -> resident
    pipe = Pipeline(stage_modules=[stage0, stage1],
                    devices=["cuda:1", "cuda:3"],
                    offload_window=2, offload_pin=0)
    res = pipe.step(x, targets=y, schedule="staggered_1b1f",
                    n_microbatches=8, loss_fn=F.cross_entropy)
    res.flush_grads()          # streamed grads -> CPU .grad, residency dropped
    opt.step()                 # AdamW(fused=True): CPU masters + GPU params

Gotchas
-------
* The engine ctor RELOCATES chunk params in place (streamed -> CPU pinned,
  pinned -> GPU). Deepcopy reference copies BEFORE building the pipeline.
* Chunks follow the OffloadModel dicing convention: chunk ``i+1`` consumes
  chunk ``i``'s output (tensor or tuple; non-float elements pass grad-free).
  The stage's own input/output contract is unchanged from a plain stage.
* Buffer mutations (BatchNorm running stats) are not written back — use
  buffer-free norms (LayerNorm is fine).
* After ANY in-place master update that bypasses ``flush_grads`` (which
  invalidates for you), call :meth:`OffloadStage.invalidate_residency` or the
  next step computes with stale resident copies.
"""

from __future__ import annotations

import threading
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.profiler import record_function

from .offload import OffloadModel
from .pipeline import Stage

__all__ = ["OffloadStage"]


class OffloadStage(Stage):
    """
    A pipeline stage whose chunked model streams weights from CPU pinned RAM.

    Fulfils the manual-autograd :class:`Stage` contract (``forward_one_chunk``
    / ``backward_one_chunk`` / ``_cache`` / ``losses`` / ``flush_grads``) so
    the relay executor, mailboxes, and schedules are unchanged — but forward
    walks the chunks ``0..L-1`` and backward walks ``L-1..0``, acquiring each
    chunk's streamed weights from the engine and relaying gradients
    chunk-to-chunk exactly like ``OffloadModel``'s keep-mode step.

    Parameters
    ----------
    chunk_modules    : ordered chunk modules (OffloadModel dicing convention).
    stage_index, num_stages, device, tracer, autocast_dtype : as in Stage.
    window           : streaming GPU slots (>= 1; >= 2 overlaps load/compute).
    pin              : chunks pinned permanently on the GPU (evenly spaced).
    keep_activations : ``True`` (keep per-chunk graphs, default) or
                       ``"checkpoint"`` (recompute-level memory, dropout-safe).
                       Engine recompute mode (``False``) is rejected — see the
                       module docstring.

    Total GPU weight memory per stage ~= ``window + pin`` chunks (plus
    activations per the backward strategy).
    """

    def __init__(
        self,
        chunk_modules: Sequence[nn.Module],
        stage_index: int,
        num_stages: int,
        device: Union[torch.device, str] = "cpu",
        tracer=None,
        autocast_dtype: Optional[torch.dtype] = None,
        *,
        window: int = 2,
        pin: int = 0,
        keep_activations: Union[bool, str] = True,
    ):
        # Deliberately NOT calling Stage.__init__: it moves the module onto
        # the stage device, but an offloaded stage's masters must go to CPU
        # pinned memory (the engine ctor relocates them).
        if keep_activations not in (True, "checkpoint"):
            raise ValueError(
                "OffloadStage supports keep_activations=True or 'checkpoint'; "
                f"got {keep_activations!r} (engine recompute mode is not "
                "supported inside the pipeline — see ramtorch/pipeline_offload.py)"
            )
        self.device = torch.device(device)
        self.stage_index = stage_index
        self.num_stages = num_stages
        self.is_first = stage_index == 0
        self.is_last = stage_index == num_stages - 1
        self.fake = None  # fake-compute is not supported for offloaded stages
        self.tracer = tracer
        self.autocast_dtype = autocast_dtype
        self.dev_idx = self.device.index if self.device.index is not None else 0
        self._track = f"stage {stage_index}"
        self.losses: Dict[int, torch.Tensor] = {}

        self.engine = OffloadModel(
            list(chunk_modules),
            device=self.device,
            window=window,
            pin=pin,
            keep_activations=keep_activations,
        )
        # ``module`` is what Pipeline.infer's worker calls directly; the
        # engine's streamed forward (inference ring) drops in unchanged.
        self.module = self.engine
        self._use_checkpoint = keep_activations == "checkpoint"

        self.params: List[nn.Parameter] = list(self.engine.parameters())
        # Grad accumulators live in the engine (CPU pinned for streamed
        # chunks, GPU for pinned chunks); the base-class dict stays empty.
        self.grad_acc: Dict[nn.Parameter, torch.Tensor] = {}

        self._lock = threading.Lock()
        # Base-contract cache: mb -> (input, last-chunk outs, needs_grad).
        self._cache: Dict[int, tuple] = {}
        # Per-microbatch chunk caches: mb -> [(inps, outs, raw, tensors), ...]
        self._mb_chunks: Dict[int, List[Optional[tuple]]] = {}
        self._inp_tuple: Dict[int, bool] = {}  # mb -> stage input was a tuple

    # ------------------------------------------------------------------
    def begin_step(self, ops: Sequence[Tuple[str, int]], n_microbatches: int):
        """Announce this step's full chunk itinerary to the engine's loader.

        Called by the executor once the schedule's per-stage op list is known
        (before any worker starts). ``("F", mb)`` expands to chunks
        ``0..L-1``, ``("B", mb)`` to ``L-1..0`` (``W`` ops touch no weights),
        which is exactly the order ``forward_one_chunk``/``backward_one_chunk``
        will acquire them in — the prefetcher sees the whole step ahead, so
        Belady eviction and the F<->B echo reuse work across op boundaries.
        """
        eng = self.engine
        n = eng.n
        itin: List[int] = []
        for op in ops:
            kind = op[0]
            if kind == "F":
                itin.extend(range(n))
            elif kind == "B":
                itin.extend(range(n - 1, -1, -1))
        with eng._cv:
            eng._check_error()
            # any previous itinerary is fully consumed by now (or belongs to
            # an aborted step) — start this step's future list fresh
            del eng._future[:]
            eng._fpos = 0
            eng._future.extend(itin)
            eng._cv.notify_all()
        with self._lock:
            self._mb_chunks.clear()
            self._inp_tuple.clear()

    # ------------------------------------------------------------------
    def forward_one_chunk(self, mb_index: int, x):
        t0 = self.tracer._ts() if self.tracer else 0.0
        ev0 = ev1 = None
        if self.tracer and self.device.type == "cuda":
            ev0 = torch.cuda.Event(enable_timing=True)
            ev0.record()

        eng = self.engine
        is_tuple = isinstance(x, (tuple, list))
        xs = tuple(x) if is_tuple else (x,)
        hs = tuple(t.to(self.device) for t in xs)

        per_chunk: List[Optional[tuple]] = []
        raw = hs[0]
        for i in range(eng.n):
            with record_function(f"F{i} mb{mb_index}"):
                tensors = eng._acquire(i)
                inps = tuple(t.detach() for t in hs)
                for t in inps:
                    # chunk boundaries need grads to relay backward; the
                    # stage's own input needs one only on non-first stages
                    t.requires_grad_(
                        t.is_floating_point() and (i > 0 or not self.is_first)
                    )
                with torch.enable_grad(), self._autocast_ctx():
                    if self._use_checkpoint:
                        raw = torch.utils.checkpoint.checkpoint(
                            lambda *a, _i=i, _ts=tensors:
                                eng._call_chunk(_i, _ts, a),
                            *inps, use_reentrant=False,
                        )
                    else:
                        raw = eng._call_chunk(i, tensors, inps)
                outs = eng._as_tuple(raw)
                per_chunk.append((inps, outs, raw, tensors))
                hs = tuple(t.detach() for t in outs)
            eng._release()

        # per-arg needs_grad mask of the stage OUTPUT (the last chunk's),
        # honoring the chunk module's out_no_grad attribute like a plain stage
        no_grad_idx = frozenset(
            getattr(eng.chunks[eng.n - 1], "out_no_grad", ()) or ()
        )
        needs_grad = tuple(
            (j not in no_grad_idx) and o.is_floating_point()
            for j, o in enumerate(outs)
        )
        stage_inp = per_chunk[0][0] if is_tuple else per_chunk[0][0][0]
        with self._lock:
            self._cache[mb_index] = (stage_inp, outs, needs_grad)
            self._mb_chunks[mb_index] = per_chunk
            self._inp_tuple[mb_index] = is_tuple

        if self.tracer:
            name = f"F mb{mb_index}"
            if ev0 is not None:
                ev1 = torch.cuda.Event(enable_timing=True)
                ev1.record()
                self.tracer.record_gpu(name, "fwd", self._track,
                                       self.dev_idx, ev0, ev1)
            self.tracer.record_cpu(name, "fwd", self._track, t0,
                                   self.tracer._ts())
        return raw

    # ------------------------------------------------------------------
    def backward_one_chunk(
        self,
        mb_index: int,
        grad_output=None,
        loss: Optional[torch.Tensor] = None,
        loss_fn=None,
        target=None,
    ):
        with self._lock:
            self._cache.pop(mb_index, None)
            per_chunk = self._mb_chunks.pop(mb_index)
            inp_is_tuple = self._inp_tuple.pop(mb_index)

        t0 = self.tracer._ts() if self.tracer else 0.0
        ev0 = None
        if self.tracer and self.device.type == "cuda":
            ev0 = torch.cuda.Event(enable_timing=True)
            ev0.record()

        eng = self.engine
        n = eng.n
        _, last_outs, last_raw, _ = per_chunk[n - 1]

        # ── resolve the backward seed for the LAST chunk ──────────────────
        seed_loss: Optional[torch.Tensor] = None
        grad_outs: Optional[tuple] = None
        if grad_output is None and self.is_last:
            if loss is None:
                assert loss_fn is not None, (
                    "last stage needs loss_fn+target or loss"
                )
                out = last_outs if len(last_outs) != 1 else last_outs[0]
                loss = self.compute_loss(out, loss_fn, target)
            self.losses[mb_index] = loss.detach()
            seed_loss = loss
        else:
            # grad from the stage above, OR the grad-bypass escape hatch on
            # the last stage — either way a value aligned to the stage output
            assert grad_output is not None, "non-last stage needs grad_output"
            gos = (
                tuple(grad_output)
                if isinstance(grad_output, (tuple, list))
                else (grad_output,)
            )
            if len(gos) != len(last_outs):
                raise ValueError(
                    f"stage {self.stage_index}: grad_output has {len(gos)} "
                    f"element(s) but the stage returned {len(last_outs)} "
                    "(must be aligned, None at no-grad slots)"
                )
            grad_outs = tuple(
                g.to(self.device) if g is not None else None for g in gos
            )

        # ── chunkwise backward: L-1 .. 0, relaying grads like _step_keep ──
        for i in range(n - 1, -1, -1):
            with record_function(f"B{i} mb{mb_index}"):
                eng._acquire(i)  # refills evicted weight storage
                inps, outs, raw_out, tensors = per_chunk[i]
                if i == n - 1 and seed_loss is not None:
                    # seed from the precomputed graph-connected loss; the
                    # engine helper differentiates whatever loss_fn returns
                    _l, grad_outs, named = eng._grads_for(
                        i, tensors, outs, inps, None,
                        lambda _raw, _tgt: seed_loss, None,
                        is_last=True, raw_out=raw_out,
                    )
                else:
                    _l, grad_outs, named = eng._grads_for(
                        i, tensors, outs, inps, grad_outs, None, None,
                        is_last=False, raw_out=raw_out,
                    )
                eng._accumulate(eng._state[i], named)
                per_chunk[i] = None  # free this chunk's graph/activations
            eng._release()

        if self.tracer:
            name = f"B mb{mb_index}"
            if ev0 is not None:
                ev1 = torch.cuda.Event(enable_timing=True)
                ev1.record()
                self.tracer.record_gpu(name, "bwd", self._track,
                                       self.dev_idx, ev0, ev1)
            self.tracer.record_cpu(name, "bwd", self._track, t0,
                                   self.tracer._ts())

        if self.is_first:
            return None
        # grad_outs is now aligned with chunk 0's inputs == the stage input
        return grad_outs if inp_is_tuple else grad_outs[0]

    # ── grads / optimizer interop (delegate to the engine) ────────────────
    def flush_grads(self, scale: float = 1.0):
        """Engine flush: accumulated grads -> ``.grad`` on the masters (CPU
        pinned for streamed chunks, GPU for pinned chunks), then residency is
        invalidated so the next step reloads post-optimizer weights."""
        self.engine.flush_grads(scale=scale)

    def zero_grad_acc(self):
        self.engine.zero_grad_acc()

    def invalidate_residency(self):
        """Drop streamed GPU weight copies (see OffloadModel). Needed after a
        manual in-place master update that bypassed :meth:`flush_grads`."""
        self.engine.invalidate_residency()

    def clear(self):
        with self._lock:
            self._cache.clear()
            self._mb_chunks.clear()
            self._inp_tuple.clear()

    def close(self):
        """Stop the engine's loader/writeback threads (idempotent)."""
        self.engine.close()
