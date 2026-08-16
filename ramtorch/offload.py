"""
ramtorch.offload
----------------
Windowed CPU->GPU layer-streaming executor ("sliding window + pinned layers").

The design was validated first in :mod:`ramtorch.offload_simulator`: weights
live in CPU pinned memory, a **loader thread** prefetches upcoming chunks into
a GPU window of ``window`` slots over a dedicated H2D stream, and ``pin``
evenly spaced chunks stay permanently on the GPU (they never load, never
evict, and ease PCIe traffic at the cost of their memory). Eviction is
farthest-next-use (Belady — optimal, since the itinerary is known). Total GPU
weight memory ~= ``window + pin`` chunks.

Gradient accumulation for streamed chunks (``grad_accum=``):

* ``"stream"`` (default): the accumulator itself streams like a weight. Each
  chunk's accumulated grad lives in exactly one place — "empty" (zeroed,
  nothing since the last flush), on the GPU (``acc_slots`` bounded slots,
  default = ``window``), or evicted to its pinned CPU buffer. Backward adds
  happen **on the GPU** on the compute stream; under slot pressure the
  accumulator is spilled D2H (a plain overwrite copy — zero CPU arithmetic)
  and reloaded H2D by the loader before that chunk's next backward (the
  itinerary marks which visits are backwards, so reloads prefetch like
  weights). ``flush_grads`` spills whatever is still on-GPU in one batch.
  With ``acc_slots >= streamed chunks`` accumulators never leave the GPU
  between flushes: grad PCIe traffic drops from once per microbatch to once
  per step.
* ``"cpu"`` (legacy): every microbatch's grads are copied D2H as a packet and
  a **writeback thread** adds them into the pinned CPU accumulators
  (``grad_acc += staging`` — host math, serialized per stage, and the reason
  compute-bound configs used to stall).

Optional third tier (``nvme=K`` / ``nvme_layers=...`` + ``nvme_path=``):
masters of the selected chunks move from CPU RAM to a file on disk, held as
mmap-backed tensors (:class:`ramtorch.nvme_store.NvmeTensorStore` — pure
PyTorch, no GDS). The loader stages them disk -> shared pinned buffer -> GPU
on its one thread, which is exactly the simulator's empirically validated
"slower H2D" model (the disk->GPU host hop serializes on the H2D copy engine
anyway — measured in ``examples/nvme_h2d_contention_test.py``). Placement
defaults to *interleaved* among the chunks (best in simulation: slow loads
spread out and hide behind neighboring compute). Optimizer steps update the
mapped masters in place; the kernel writes them back to disk lazily. Grad
accumulators for NVMe chunks still live in RAM (transient, one chunk's worth
of overlap at a time is what matters for streaming).

Model dicing follows the pipeline convention (``Pipeline(stage_modules=...)``)
rather than module surgery: you cut your own model into an ordered list of
chunk modules, each taking the previous chunk's output. This supersedes the
CUDA-stream-bouncing pattern in :mod:`ramtorch.modules.linear`
(``CPUBouncingLinear``), which is deprecated.

Concurrency style follows :mod:`ramtorch.pipeline_relay`: plain threads with
condition-variable handshakes; compute blocks only on "chunk not resident
yet", the loader blocks only on "no slot free / nothing to prefetch". CUDA
cross-stream safety uses the same event pattern as the relay mailboxes (the
producer records an event on its stream; the consumer's stream waits on it).

Autograd contract (mirrors the manual-autograd ``Stage``): backward walks
chunks in reverse (the simulator's training "echo": F0..F(n-1), B(n-1)..B0)
calling ``torch.autograd.grad`` per chunk. Two backward strategies:

* ``keep_activations=False`` (default, "recompute"): the training forward
  runs under ``no_grad`` caching only chunk-boundary activations; backward
  reloads each chunk and **recomputes its forward** (per-chunk gradient
  checkpointing). A naively retained graph would keep references to the
  streamed GPU weights and nothing would actually be freed between F_i and
  B_i — recompute sidesteps that entirely.
* ``keep_activations=True`` ("keep"): the forward builds each chunk's graph
  and keeps its internal activations, so backward skips the recompute. The
  graph's weight references point at stable per-chunk tensor objects whose
  **storage** is freed on eviction and refilled by the loader before that
  chunk's backward (``untyped_storage().resize_(0)``, the FSDP resharding
  trick) — so weights still stream despite the live graph. Costs activation
  memory for all chunks at once; also the correct mode for stochastic chunks
  (dropout), which recompute would resample.
* ``keep_activations="checkpoint"``: keep-mode plumbing (grad-enabled
  forward, stable graph tensors, storage free/refill) but each chunk runs
  under ``torch.utils.checkpoint`` (non-reentrant), so its internal
  activations are dropped and autograd recomputes them at backward —
  recompute-mode memory with checkpoint's RNG stash/restore, i.e. the
  dropout-safe recompute. NB: a bare ``torch.utils.checkpoint`` inside the
  user's chunk module breaks — by backward time ``functional_call`` has
  reverted the module's params to the CPU masters, so the recompute would
  read the wrong weights. To checkpoint *selected regions* of your own
  forward instead of whole chunks, use :func:`offload_checkpoint` (with
  ``keep_activations=True``).

Usage::

    chunks = [make_block(i) for i in range(24)]      # dice your own model
    model = OffloadModel(chunks, device="cuda:0", window=2, pin=6)

    out = model(x)                                    # streamed inference

    result = model.step(x, targets=y, loss_fn=F.cross_entropy)
    model.flush_grads()                               # acc -> .grad
    optimizer.step()                                  # CPU+GPU param groups

Notes
-----
* Chunks exchange a single tensor by default; a chunk may also return a
  TUPLE of tensors, whose elements become the next chunk's positional args
  (like the pipeline's tuple outputs). Non-float / no-grad elements (masks,
  position ids) pass through without gradients.
* Buffer mutations inside chunks (e.g. BatchNorm running stats) happen on the
  streamed GPU copy and are NOT written back; use eval-mode/buffer-free norms.
* The optimizer sees mixed devices: streamed params (and their grads) on CPU,
  pinned params on the GPU. torch optimizers handle per-param devices fine.
  For CPU params, a stochastic/fused CPU optimizer (e.g. ramtorch AdamW)
  keeps the step cheap.
* The engine's internal primitives (``_announce``/``_acquire``/``_release``,
  ``_call_chunk``, ``_grads_for``, ``_accumulate``) are also driven externally
  by :class:`ramtorch.pipeline_offload.OffloadStage`, which streams a pipeline
  stage's weights with the itinerary derived from the pipeline schedule
  instead of ``step()``'s fixed F/B echo. Treat their signatures as a shared
  contract when refactoring.
"""

from __future__ import annotations

import contextlib
import grp
import json
import os
import pwd
import queue
import threading
import time
from collections import deque
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.utils.checkpoint
from torch.profiler import record_function

from .nvme_store import NvmeTensorStore
from .offload_simulator import evenly_pinned, interleaved_nvme

__all__ = ["OffloadModel", "OffloadStepResult", "offload_checkpoint"]

_INF = float("inf")

# TRAINING with the NVMe tier is gated behind an explicit opt-in because it
# rewrites the on-disk masters every optimizer step and can wear out a
# consumer SSD (see docs/offload.md, "Drive-endurance caution"). The gate
# fires on the first step() of a model with nvme chunks — NOT at
# construction, and never for inference-only use (forward() writes the
# scratch file once and only reads afterwards, which is harmless).
# Unlocking requires BOTH: (1) the user is a sudoer (root, or a member of
# the sudo/wheel group — the machine's owner class, who can replace the
# drive), and (2) RAMTORCH_NVME_ACKNOWLEDGE=1 is set. The obnoxious warning
# prints unless RAMTORCH_NVME_QUIET=1 is also set.
_NVME_ACK_ENV = "RAMTORCH_NVME_ACKNOWLEDGE"
_NVME_QUIET_ENV = "RAMTORCH_NVME_QUIET"

_NVME_WARNING = f"""
{'!' * 78}
!!  RamTorch NVMe weight tier — SSD WEAR WARNING
!!
!!  You are about to TRAIN with model master weights on an NVMe
!!  scratch file.
!!
!!  * INFERENCE is fine (and never shows this warning): the file is
!!    written once, then only read.
!!  * TRAINING REWRITES EVERY NVMe-RESIDENT MASTER ON EVERY OPTIMIZER
!!    STEP. Sustained training can write TERABYTES PER DAY — a big model
!!    (FLUX-scale, tens of GB on disk) can reach PETABYTES PER DAY —
!!    enough to wear out a consumer SSD (rated only a few hundred TBW)
!!    in months, days, or less.
!!
!!  THIS TIER IS A LAST RESORT. Be wise — it is almost certainly not
!!  worth it. Go back before it's too late. Before you burn write
!!  cycles, try EVERY alternative first:
!!    * buy/borrow more RAM (it is cheaper than the drive you'll kill),
!!    * shrink the model, quantize, or reduce the pinned/window sizes,
!!    * rent a bigger machine for the training run.
!!  If none of those are possible and you still proceed, you are
!!  knowingly trading an SSD's lifespan for this run.
!!
!!  By running this as a sudoer with {_NVME_ACK_ENV}=1 you
!!  acknowledged that YOU are responsible for any drive wear or
!!  failure this causes.
!!
!!  Mitigations: keep the tier small and cold-only, use a high-endurance
!!  (enterprise/DWPD-rated) or sacrificial drive, monitor SMART counters.
!!
!!  This warning prints on every training run. Set {_NVME_QUIET_ENV}=1
!!  to silence it (the tier stays unlocked while {_NVME_ACK_ENV}=1).
{'!' * 78}
"""


def _is_sudoer() -> bool:
    """True if the effective user is root or in the sudo/wheel group.

    Group membership (not ``sudo -v``) so passwordless-sudo and already-
    authenticated users are covered without prompting. Fails CLOSED: any
    lookup error (non-POSIX host, missing groups) counts as not a sudoer.
    """
    try:
        if os.geteuid() == 0:
            return True
        uid, gid = os.geteuid(), os.getegid()
        name = pwd.getpwuid(uid).pw_name
        for gname in ("sudo", "wheel"):
            try:
                g = grp.getgrnam(gname)
            except KeyError:
                continue
            if name in g.gr_mem or g.gr_gid == gid:
                return True
    except Exception:
        pass
    return False


def _check_nvme_unlocked() -> None:
    """Gate TRAINING on the NVMe tier: sudoer-only, env-var acknowledgment.

    Called from the first ``step()`` of a model with nvme chunks. Inference
    (``forward()``) never triggers it — the scratch file is written once at
    construction and only read afterwards, which doesn't wear the drive.
    """
    if not _is_sudoer():
        raise RuntimeError(
            "TRAINING with the OffloadModel NVMe tier is restricted to "
            "sudoers (root or sudo/wheel group members). It rewrites the "
            "on-disk master weights every optimizer step and can physically "
            "wear out the SSD — the decision to risk a drive belongs to "
            "whoever owns the machine. Ask a sudoer to run it, or use the "
            "model for inference only (forward() is not gated). See "
            "docs/offload.md, 'Drive-endurance caution'."
        )
    if os.environ.get(_NVME_ACK_ENV) != "1":
        raise RuntimeError(
            "TRAINING with the OffloadModel NVMe tier is LOCKED. It rewrites "
            "the on-disk master weights every optimizer step and can wear "
            "out an SSD (consumer drives are rated for only a few hundred "
            "TBW; a FLUX-scale model can write PETABYTES per day). "
            "Inference (forward()) is not gated. If you must train this way "
            "and accept responsibility for any drive wear, set the "
            f"environment variable {_NVME_ACK_ENV}=1 and retry. See "
            "docs/offload.md, 'Drive-endurance caution'."
        )
    if os.environ.get(_NVME_QUIET_ENV) != "1":
        print(_NVME_WARNING, flush=True)


def offload_checkpoint(module: nn.Module, *args, **kwargs):
    """Gradient-checkpoint ``module(*args, **kwargs)`` inside a chunk forward.

    Use this INSTEAD of ``torch.utils.checkpoint.checkpoint`` to mark the
    parts of a chunk's forward you want recomputed at backward, keeping the
    rest of the chunk's activations live — the selective middle ground
    between ``keep_activations=True`` (keep everything) and
    ``keep_activations="checkpoint"`` (recompute every chunk wholesale)::

        class Block(nn.Module):
            def forward(self, x):
                x = self.light(x)                        # activations kept
                return offload_checkpoint(self.heavy, x) # recomputed at bwd

        model = OffloadModel(blocks, keep_activations=True)

    Why plain torch checkpoint breaks here: :class:`OffloadModel` invokes
    chunks through ``functional_call``, which reverts the module's params to
    the CPU masters when it exits — so torch checkpoint's backward-time
    recompute would read the wrong (CPU) weights and fail with a device
    mismatch. This wrapper snapshots the module's *effective* tensors at
    forward time (inside the engine's ``functional_call`` these are the
    streamed GPU tensors, whose storage the loader refills before that
    chunk's backward) and re-applies them for the recompute.

    Like torch checkpoint (non-reentrant), the RNG state is stashed and
    restored, so dropout inside the marked region replays the same masks.
    Works transparently outside :class:`OffloadModel` too (the snapshot is
    then just the module's own params), so a model written with this marker
    runs unchanged as a plain module. No-op under ``no_grad`` (inference,
    and the recompute-mode training forward).
    """
    if not torch.is_grad_enabled():
        return module(*args, **kwargs)
    tensors = dict(module.named_parameters()) | dict(module.named_buffers())

    def _rerun(*a, **kw):
        return torch.func.functional_call(module, tensors, a, kw)

    return torch.utils.checkpoint.checkpoint(
        _rerun, *args, use_reentrant=False, **kwargs
    )


class OffloadStepResult:
    """Output + loss of one :meth:`OffloadModel.step`.

    ``output`` mirrors the last chunk's return value (tensor or tuple).
    With ``grad_outputs=`` (grad bypass) no loss is ever computed, so
    accessing ``.loss`` raises instead of silently returning ``None``.
    """

    def __init__(self, output, loss, *, grad_bypassed: bool = False):
        self.output = output
        self._loss = loss
        self._grad_bypassed = grad_bypassed

    @property
    def loss(self):
        if self._grad_bypassed:
            raise RuntimeError(
                "step() ran with grad_outputs= (grad bypass): the loss is "
                "never computed. Compute your loss outside the model."
            )
        return self._loss


class _ChunkState:
    """Per-chunk storage: CPU master tensors, grad accumulators, staging."""

    def __init__(self, module: nn.Module, gpu_pinned: bool,
                 device: torch.device, use_cuda: bool, nvme: bool = False,
                 idx: int = -1, pin_acc: bool = False):
        self.module = module
        self.gpu_pinned = gpu_pinned
        self.idx = idx
        # streamed-accumulator (grad_accum="stream") state, guarded by the
        # engine's _cv: the accumulated value lives in EXACTLY one place —
        #   "empty": zero (nothing accumulated since the last flush/zero)
        #   "gpu"  : acc_gpu holds it (CPU grad_acc is stale)
        #   "cpu"  : grad_acc holds it (evicted / spilled)
        self.acc_where: str = "empty"
        self.acc_gpu: Optional[Dict[str, torch.Tensor]] = None
        # last event that must complete before acc_gpu may be read on another
        # stream (H2D reload or the latest GPU add); acc_fresh marks a reload
        # the compute stream has not synchronized with yet
        self.acc_ev = None
        self.acc_fresh = False
        # nvme masters start as plain CPU tensors here; OffloadModel rehomes
        # them onto the file mapping right after all states are built (the
        # store lays out every nvme chunk in one file, so it needs them all)
        self.nvme = nvme
        self.param_names = frozenset(n for n, _ in module.named_parameters())
        # keep_activations mode: stable GPU tensor objects referenced by the
        # autograd graph; eviction frees their storage, reload refills it
        self.graph_tensors: Optional[Dict[str, torch.Tensor]] = None

        # persistent .grad buffers for flush_grads (CPU ones pinned so a
        # streaming optimizer can H2D them at full PCIe speed)
        self.flush_bufs: Dict[str, torch.Tensor] = {}

        if gpu_pinned:
            module.to(device)
            self.tensors: Dict[str, torch.Tensor] = dict(
                module.named_parameters()
            ) | dict(module.named_buffers())
            # grads accumulate on the GPU next to the pinned weights
            self.grad_acc: Dict[str, torch.Tensor] = {
                n: torch.zeros_like(p) for n, p in module.named_parameters()
            }
            self.staging = None
            return

        # streamed chunk: master weights in CPU pinned memory (nvme chunks
        # skip the pinning — they are about to move onto the file mapping)
        for p in module.parameters():
            p.data = p.data.detach().cpu()
            if use_cuda and not nvme:
                p.data = p.data.pin_memory()
        for b in module.buffers():
            b.data = b.data.detach().cpu()
            if use_cuda and not nvme:
                b.data = b.data.pin_memory()
        self.tensors = dict(module.named_parameters()) | dict(
            module.named_buffers()
        )
        # stream mode transfers accs whole over PCIe: pin them so the copies
        # run at full speed ("cpu" mode keeps them pageable and stages instead)
        self.grad_acc = {
            n: (torch.zeros_like(p, device="cpu").pin_memory() if pin_acc
                else torch.zeros_like(p, device="cpu"))
            for n, p in module.named_parameters()
        }
        # pinned D2H staging buffers, allocated lazily at first backward
        # (grad_accum="cpu" packet path only)
        self.staging: Optional[Dict[str, torch.Tensor]] = None


class _Resident:
    """A streamed chunk's GPU copy + the H2D completion event."""

    __slots__ = ("tensors", "event")

    def __init__(self, tensors: Dict[str, torch.Tensor], event):
        self.tensors = tensors
        self.event = event


class _ActPinnedPool:
    """Reusable pinned staging buffers for activation packets.

    Buckets by power-of-two byte size; training loops repeat shapes, so
    after the first step every take() is a hit. A buffer is returned with
    the event of its last H2D read — take() synchronizes it only if the
    read is somehow still in flight (rare: by reuse time it is long done).
    A fresh pinned alloc per pack is 10-20x slower (probe, §0.11).
    """

    def __init__(self):
        self._free: Dict[int, List[tuple]] = {}

    @staticmethod
    def _bucket(nbytes: int) -> int:
        return max(512, 1 << max(0, nbytes - 1).bit_length())

    def take(self, nbytes: int, pin: bool) -> torch.Tensor:
        bucket = self._bucket(nbytes)
        free = self._free.get(bucket)
        if free:
            buf, ev = free.pop()
            if ev is not None and not ev.query():
                ev.synchronize()
            return buf
        return torch.empty(bucket, dtype=torch.uint8, pin_memory=pin)

    def give(self, buf: torch.Tensor, ev) -> None:
        self._free.setdefault(buf.numel(), []).append((buf, ev))


class _ActEntry:
    """One saved tensor inside an activation packet."""

    __slots__ = ("gpu", "buf", "view", "dtype", "shape", "nbytes")

    def __init__(self, gpu: torch.Tensor):
        self.gpu: Optional[torch.Tensor] = gpu
        self.buf: Optional[torch.Tensor] = None    # pooled pinned uint8
        self.view: Optional[torch.Tensor] = None   # buf viewed as the tensor
        self.dtype = gpu.dtype
        self.shape = tuple(gpu.shape)
        self.nbytes = gpu.numel() * gpu.element_size()


class _ActPacket:
    """The saved tensors of one chunk forward (``saved_tensors_hooks``).

    States mirror the simulator's residency classes:
      * ``"gpu"``       — resident, dirty (no RAM copy yet)
      * ``"gpu_clean"`` — resident with a still-valid RAM copy (a re-drop
                          is free: just release the GPU refs again)
      * ``"cpu"``       — offloaded (GPU refs dropped; reload before B)
    ``bpos`` is the packet's backward position in the announced schedule —
    eviction picks the farthest (Belady, exact: the itinerary is known).
    """

    __slots__ = ("key", "bpos", "entries", "dedup", "where",
                 "d2h_ev", "h2d_ev")

    def __init__(self, key, bpos: int):
        self.key = key
        self.bpos = bpos
        self.entries: List[_ActEntry] = []
        self.dedup: Dict[tuple, int] = {}
        self.where = "gpu"
        self.d2h_ev = None    # offload copies complete (RAM copy valid)
        self.h2d_ev = None    # reload copies complete (GPU refs valid)


class OffloadModel(nn.Module):
    """
    Run an ordered list of chunk modules with weights streamed from CPU
    through a sliding GPU window, with ``pin`` chunks resident permanently.

    Parameters
    ----------
    chunks     : ordered chunk modules (same dicing convention as
                 ``Pipeline(stage_modules=...)``); chunk ``i+1`` consumes
                 chunk ``i``'s output — a single tensor, or a tuple whose
                 elements become the next chunk's positional args (non-float
                 elements pass through without grads).
    device     : the compute device (default: first CUDA device, else CPU;
                 CPU is supported for tests — copies degrade to clones).
    window     : streaming slots on the GPU (>= 1; >= 2 to overlap load with
                 compute — see the simulator's window sweep).
    pin        : number of chunks pinned permanently on the GPU, evenly
                 spaced (``pin_layers`` overrides). Total weight memory is
                 ~``window + pin`` chunks.
    pin_layers : explicit chunk indices to pin (overrides ``pin``).
    nvme       : number of chunks whose masters live on disk instead of CPU
                 RAM, interleaved evenly among the chunks (``nvme_layers``
                 overrides; overlaps with pinned chunks are dropped — pinned
                 wins). Their weights are held as mmap-backed tensors in a
                 single scratch file and stream disk -> pinned staging -> GPU
                 on the loader thread ("slower H2D"). Saves CPU RAM at the
                 cost of load latency for those chunks. Optimizer steps
                 update the mapped masters in place (page-cache write-back).
                 Inference is ungated; TRAINING (the first :meth:`step`)
                 requires the sudoer + RAMTORCH_NVME_ACKNOWLEDGE=1 consent
                 gate because it rewrites the on-disk masters every
                 optimizer step (SSD wear — see docs/offload.md).
    nvme_layers: explicit chunk indices for the NVMe tier (overrides
                 ``nvme``).
    nvme_path  : path of the scratch weights file, REQUIRED when the NVMe
                 tier is used. Put it on a real drive — /tmp is often tmpfs
                 (RAM), which silently defeats the point. Deleted on
                 :meth:`close`.
    keep_activations :
                 backward strategy for :meth:`step`.
                 ``False`` (default, "recompute"): the training forward runs
                 under ``no_grad`` caching only chunk-boundary activations;
                 backward reloads each chunk and recomputes its forward
                 (per-chunk gradient checkpointing). Cheapest memory, but
                 every chunk's forward runs twice, and stochastic chunks
                 (dropout) would resample on recompute.
                 ``True`` ("keep"): the forward builds each chunk's autograd
                 graph and keeps its internal activations on the GPU, so
                 backward skips the recompute. Weights are STILL evicted:
                 the graph references stable weight tensor objects whose
                 storage is freed on eviction (``untyped_storage().resize_(0)``,
                 the FSDP resharding trick) and refilled by the loader
                 before that chunk's backward. Costs activation memory for
                 all chunks at once; also the correct mode for stochastic
                 chunks since nothing is resampled.
                 ``"checkpoint"``: keep-mode plumbing, but each chunk runs
                 under non-reentrant ``torch.utils.checkpoint`` — internal
                 activations are dropped and recomputed by autograd during
                 backward, with RNG stashed/restored. Recompute-cheap
                 memory AND dropout-safe (the mode recompute cannot be).
    offload_activations :
                 stream saved activations to pinned CPU RAM like weights
                 (``saved_tensors_hooks`` packets, one per chunk forward).
                 Policy is the simulator-validated *lazy* one: offload only
                 when more than ``act_slots`` chunk graphs are resident,
                 evict the packet whose backward is farthest, reload one
                 backward ahead. Requires ``keep_activations=True`` or
                 ``"checkpoint"`` (recompute mode already drops them).
                 With keep mode this bounds activation memory to
                 ~``act_slots`` chunks instead of all ``n``. Bit-exact:
                 the same values return at unpack. Never touches NVMe.
    act_slots  : resident chunk-activation packets allowed before the lazy
                 offload kicks in (default 2 — simulator sweet spot: one
                 in compute + one prefetched; raise it to trade GPU memory
                 for less PCIe traffic).
    """

    def __init__(
        self,
        chunks: Sequence[nn.Module],
        *,
        device: Optional[Union[str, torch.device]] = None,
        window: int = 2,
        pin: int = 0,
        pin_layers: Optional[Sequence[int]] = None,
        nvme: int = 0,
        nvme_layers: Optional[Sequence[int]] = None,
        nvme_path: Optional[str] = None,
        keep_activations: Union[bool, str] = False,
        grad_accum: str = "stream",
        acc_slots: Optional[int] = None,
        offload_activations: bool = False,
        act_slots: int = 2,
    ):
        super().__init__()
        if len(chunks) < 1:
            raise ValueError("need at least one chunk")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if keep_activations not in (False, True, "checkpoint"):
            raise ValueError(
                "keep_activations must be False, True, or 'checkpoint', "
                f"got {keep_activations!r}"
            )
        if grad_accum not in ("stream", "cpu"):
            raise ValueError(
                f"grad_accum must be 'stream' or 'cpu', got {grad_accum!r}"
            )
        if acc_slots is not None and acc_slots < 1:
            raise ValueError(f"acc_slots must be >= 1, got {acc_slots}")
        if offload_activations and keep_activations is False:
            raise ValueError(
                "offload_activations requires keep_activations=True or "
                "'checkpoint' — recompute mode has no saved-tensor graph "
                "to offload from (it already drops activations)"
            )
        if act_slots < 1:
            raise ValueError(f"act_slots must be >= 1, got {act_slots}")

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._cuda = self.device.type == "cuda"
        self.window = window
        self.keep_activations = keep_activations
        # keep and checkpoint share the stable-graph-tensor machinery
        # (storage free on eviction / refill on reload)
        self._keep_graph = bool(keep_activations)
        self.n = len(chunks)

        if pin_layers is not None:
            pinned_idx = frozenset(int(i) for i in pin_layers)
            if not all(0 <= i < self.n for i in pinned_idx):
                raise ValueError(f"pin_layers out of range [0, {self.n})")
        else:
            pinned_idx = evenly_pinned(self.n, min(pin, self.n))
        self.pinned_layers = pinned_idx

        if nvme_layers is not None:
            nvme_idx = frozenset(int(i) for i in nvme_layers)
            if not all(0 <= i < self.n for i in nvme_idx):
                raise ValueError(f"nvme_layers out of range [0, {self.n})")
        else:
            nvme_idx = interleaved_nvme(self.n, min(nvme, self.n))
        nvme_idx -= pinned_idx  # pinned wins (as in the simulator)
        if nvme_idx and nvme_path is None:
            raise ValueError(
                "nvme_path is required when nvme chunks are requested; give "
                "a file path on the actual drive (NOT /tmp — often tmpfs)"
            )
        # training-only consent gate: checked at the first step(), not here —
        # inference with nvme masters is read-only after the initial write
        self._nvme_gate_passed = False
        self.nvme_layers = nvme_idx

        self.grad_accum = grad_accum
        self.acc_slots = window if acc_slots is None else acc_slots
        self.offload_activations = bool(offload_activations)
        self.act_slots = act_slots

        # register chunks so .parameters()/.state_dict() work
        self.chunks = nn.ModuleList(chunks)
        pin_acc = grad_accum == "stream" and self._cuda
        self._state = [
            _ChunkState(m, i in pinned_idx, self.device, self._cuda,
                        nvme=i in nvme_idx, idx=i, pin_acc=pin_acc)
            for i, m in enumerate(self.chunks)
        ]

        # rehome NVMe masters onto one file mapping (frees their RAM copies)
        self._nvme_store: Optional[NvmeTensorStore] = None
        self._nvme_staging: Optional[torch.Tensor] = None  # loader-only
        self._staging_ev = None       # H2D-out-of-staging completion event
        if nvme_idx:
            payload = {
                f"{i}.{name}": t
                for i in sorted(nvme_idx)
                for name, t in self._state[i].tensors.items()
            }
            self._nvme_store = NvmeTensorStore(nvme_path)
            mapped = self._nvme_store.write(payload)
            with torch.no_grad():
                for i in sorted(nvme_idx):
                    for name, t in self._state[i].tensors.items():
                        t.data = mapped[f"{i}.{name}"]

        # ── residency state (guarded by one condition variable) ──────────
        self._cv = threading.Condition()
        self._future: List[int] = []      # upcoming chunk uses, itinerary order
        # aligned op kinds ("F"/"B") — the acc prefetcher only cares about Bs
        self._future_kinds: List[str] = []
        self._fpos = 0                    # next compute position in _future
        self._resident: Dict[int, _Resident] = {}
        self._in_flight: Optional[int] = None
        self._in_use: Optional[int] = None
        # streamed grad accumulator residency (grad_accum="stream")
        self._acc_res: set = set()            # chunk idxs with a GPU acc
        self._acc_evicting: set = set()       # eviction D2H queued/running
        self._acc_in_flight: Optional[int] = None  # loader H2D acc reload
        self._acc_adding: Optional[int] = None     # compute mid-add (no evict)
        self._error: Optional[BaseException] = None
        self._closed = False

        # dedicated copy streams (two PCIe engines: H2D prefetch, D2H grads)
        self._h2d_stream = torch.cuda.Stream(self.device) if self._cuda else None
        self._d2h_stream = torch.cuda.Stream(self.device) if self._cuda else None

        # ── activation packet store (compute-thread only, no lock) ───────
        # saved_tensors_hooks packets keyed by the caller's chunk key
        # (int chunk idx for step(); (mb, chunk) for OffloadStage)
        self._act_store: Dict[object, _ActPacket] = {}
        self._act_pool = _ActPinnedPool()
        self._act_in_use: Optional[object] = None  # packet mid-backward

        self.stats = {"loads": 0, "nvme_loads": 0, "acquire_wait_s": 0.0,
                      "acc_loads": 0, "acc_evictions": 0,
                      "act_offloads": 0, "act_reloads": 0,
                      "act_bytes_offloaded": 0}
        # (track, name, start_us, dur_us) spans from the worker threads,
        # collected only while step(profile_path=...) is active — kineto
        # cannot see record_function on threads it did not enter from
        self._span_log: Optional[List[tuple]] = None

        self._loader = threading.Thread(
            target=self._loader_loop, daemon=True, name="offload-loader"
        )
        self._loader.start()
        # bounded: if D2H writeback lags behind compute, step() blocks on put()
        # instead of letting per-chunk GPU grads pile up unboundedly
        self._wb_q: "queue.Queue" = queue.Queue(maxsize=4)
        # D2H copies enqueued but not yet accumulated into grad_acc:
        # (state, names, done_event) in FIFO order. Writeback thread only.
        self._wb_pending: deque = deque()
        self._writeback = threading.Thread(
            target=self._writeback_loop, daemon=True, name="offload-writeback"
        )
        self._writeback.start()

    # ── loader thread ──────────────────────────────────────────────────────
    def _next_use(self, layer: int, from_pos: int) -> float:
        """Position of the next use of ``layer`` in the known future."""
        try:
            return self._future.index(layer, from_pos)
        except ValueError:
            return _INF

    def _next_acc_use(self, layer: int, from_pos: int) -> float:
        """Position of the next BACKWARD use of ``layer`` (acc Belady)."""
        pos = from_pos
        while True:
            try:
                pos = self._future.index(layer, pos)
            except ValueError:
                return _INF
            if self._future_kinds[pos] == "B":
                return pos
            pos += 1

    def _pick_load(self):
        """Under _cv: next (layer, needed_pos) to prefetch, or None."""
        for pos in range(self._fpos, len(self._future)):
            layer = self._future[pos]
            if self._state[layer].gpu_pinned:
                continue
            if layer in self._resident or layer == self._in_flight:
                continue
            return layer, pos
        return None

    def _pick_acc_load(self):
        """Under _cv: next evicted acc needed by an upcoming B, or None."""
        if self.grad_accum != "stream" or self._acc_in_flight is not None:
            return None
        for pos in range(self._fpos, len(self._future)):
            if self._future_kinds[pos] != "B":
                continue
            layer = self._future[pos]
            state = self._state[layer]
            if state.gpu_pinned or layer in self._acc_res:
                continue
            if state.acc_where == "cpu":
                return layer, pos
        return None

    def _acc_pick_victim(self, need_pos: float):
        """Under _cv: farthest-next-B-use resident acc to spill, or None.

        Never picks an acc mid-add or mid-eviction, and never one whose next
        B use is at/before ``need_pos`` (that eviction would thrash)."""
        victim, victim_nu = None, float(need_pos)
        for r in self._acc_res:
            if r == self._acc_adding or r in self._acc_evicting:
                continue
            nu = self._next_acc_use(r, self._fpos)
            if nu > victim_nu:
                victim, victim_nu = r, nu
        return victim

    def _try_start_load(self):
        """Under _cv: pick the loader's next action.

        Returns ``("w", layer)`` weight load, ``("acc", layer)`` grad-acc
        reload, ``("evict", layer)`` acc spill needed to free an acc slot,
        or None (nothing to do / no slot freeable yet — wait for progress).
        """
        cand = self._pick_load()
        if cand is not None:
            layer, pos = cand
            occupancy = (len(self._resident)
                         + (1 if self._in_flight is not None else 0))
            can_start = occupancy < self.window
            if not can_start:
                victim, victim_nu = None, -1.0
                for r in self._resident:
                    if r == self._in_use:
                        continue
                    nu = self._next_use(r, self._fpos)
                    if nu > victim_nu:
                        victim, victim_nu = r, nu
                if victim is not None and victim_nu > pos:
                    del self._resident[victim]
                    if self._keep_graph:
                        # the autograd graph still references these tensor
                        # objects; free the storage behind them (refilled
                        # before backward)
                        vstate = self._state[victim]
                        if vstate.graph_tensors is not None:
                            for t in vstate.graph_tensors.values():
                                t.untyped_storage().resize_(0)
                    can_start = True
            if can_start:
                self._in_flight = layer
                return ("w", layer)
            # weight slot blocked: fall through — an acc reload can still
            # use the idle H2D link meanwhile

        acc = self._pick_acc_load()
        if acc is not None:
            layer, pos = acc
            occupancy = len(self._acc_res)
            if occupancy >= self.acc_slots:
                victim = self._acc_pick_victim(pos)
                if victim is None:
                    return None
                self._acc_evicting.add(victim)
                return ("evict", victim)
            self._acc_in_flight = layer
            return ("acc", layer)
        return None

    def _loader_loop(self):
        try:
            if self._cuda:
                torch.cuda.set_device(self.device)
            while True:
                with self._cv:
                    action = self._try_start_load()
                    while action is None and not self._closed:
                        self._cv.wait()
                        action = self._try_start_load()
                    if self._closed:
                        return
                what, layer = action
                state = self._state[layer]

                if what == "evict":
                    # hand the spill to the writeback thread (D2H channel);
                    # put() may block on a full queue — we hold no lock here
                    self._wb_q.put(("acc_evict", state))
                    continue

                if what == "acc":
                    # H2D reload of an evicted grad accumulator
                    label = f"A{layer} acc reload"
                    t_us = time.monotonic_ns() / 1e3
                    with torch.no_grad(), record_function(label):
                        if self._cuda:
                            with torch.cuda.stream(self._h2d_stream):
                                gpu = {
                                    n: a.to(self.device, non_blocking=True)
                                    for n, a in state.grad_acc.items()
                                }
                                ev = torch.cuda.Event()
                                ev.record()
                        else:  # pragma: no cover — cpu engines add in place
                            gpu, ev = dict(state.grad_acc), None
                    if self._span_log is not None:
                        self._span_log.append((
                            "offload h2d loader", label,
                            t_us, time.monotonic_ns() / 1e3 - t_us,
                        ))
                    with self._cv:
                        state.acc_gpu = gpu
                        state.acc_ev = ev
                        state.acc_fresh = True
                        state.acc_where = "gpu"
                        self._acc_res.add(layer)
                        self._acc_in_flight = None
                        self.stats["acc_loads"] += 1
                        self._cv.notify_all()
                    continue

                # copy outside the lock, on the H2D stream (no_grad so the
                # copies are plain leaves, not autograd-tracked views)
                label = (f"N{layer} nvme load" if state.nvme
                         else f"L{layer} h2d load")
                t_us = time.monotonic_ns() / 1e3
                with torch.no_grad(), record_function(label):
                    if self._cuda:
                        with torch.cuda.stream(self._h2d_stream):
                            gpu = self._materialize(state)
                            ev = torch.cuda.Event()
                            ev.record()
                        if state.nvme:
                            # staging is reused by the next nvme load; it may
                            # be overwritten only after this H2D completes
                            self._staging_ev = ev
                    else:
                        gpu = self._materialize(state)
                        ev = None
                if self._span_log is not None:
                    self._span_log.append((
                        "offload h2d loader", label,
                        t_us, time.monotonic_ns() / 1e3 - t_us,
                    ))
                with self._cv:
                    self._resident[layer] = _Resident(gpu, ev)
                    self._in_flight = None
                    self.stats["loads"] += 1
                    if state.nvme:
                        self.stats["nvme_loads"] += 1
                    self._cv.notify_all()
        except BaseException as e:  # noqa: BLE001 — propagate to compute thread
            with self._cv:
                self._error = e
                self._cv.notify_all()

    def _stage_nvme(self, state: _ChunkState) -> Dict[str, torch.Tensor]:
        """Copy a chunk's file-backed masters into the shared pinned staging
        buffer; returns pinned views (loader thread only).

        The ``copy_`` from the mmap tensors page-faults from disk when cold
        (page-cache speed when hot); the caller's H2D copies out of the
        pinned views then run at full PCIe speed. Both hops run serially on
        the one loader thread — the simulator's "slower H2D" model.
        """
        align = 64  # covers every dtype's element size for .view(dtype)
        need = 0
        for t in state.tensors.values():
            nb = t.numel() * t.element_size()
            need += -(-nb // align) * align
        if self._nvme_staging is None or self._nvme_staging.numel() < need:
            self._nvme_staging = torch.empty(
                need, dtype=torch.uint8, pin_memory=True
            )
        if self._staging_ev is not None:
            # previous nvme chunk's H2D out of this buffer must be done
            self._staging_ev.synchronize()
            self._staging_ev = None
        views: Dict[str, torch.Tensor] = {}
        off = 0
        for n, t in state.tensors.items():
            nb = t.numel() * t.element_size()
            v = self._nvme_staging[off:off + nb].view(t.dtype).view(t.shape)
            v.copy_(t)
            views[n] = v
            off += -(-nb // align) * align
        return views

    def _materialize(self, state: _ChunkState) -> Dict[str, torch.Tensor]:
        """Produce a chunk's device tensors (runs on the loader thread).

        Recompute mode: fresh copies each load (the caching allocator recycles
        the blocks). Keep mode: stable tensor objects — the autograd graph
        points at them across evictions, so a reload refills the same objects'
        storage instead of allocating new ones.

        NVMe chunks route their masters through the shared pinned staging
        buffer first (disk -> pinned), so the device copies below always
        read pinned memory.
        """
        src = state.tensors
        if state.nvme and self._cuda:
            src = self._stage_nvme(state)

        if not self._keep_graph:
            if self._cuda:
                return {
                    n: t.to(self.device, non_blocking=True)
                    for n, t in src.items()
                }
            return {n: t.clone() for n, t in src.items()}

        if state.graph_tensors is None:
            gt = {}
            for n, t in src.items():
                g = (t.to(self.device, non_blocking=True)
                     if self._cuda else t.clone())
                if n in state.param_names:
                    g.requires_grad_(True)  # a graph leaf for autograd.grad
                gt[n] = g
            state.graph_tensors = gt
            return gt

        for n, t in src.items():
            g = state.graph_tensors[n]
            store = g.untyped_storage()
            if store.size() == 0:
                store.resize_(g.numel() * g.element_size())
            # refill through .data: it has its own version counter, so the
            # in-place write is invisible to autograd's saved-tensor check
            # (the values are identical to what the graph saved — that is
            # this mode's whole contract)
            g.data.copy_(t, non_blocking=self._cuda)
        return state.graph_tensors

    # ── writeback thread ───────────────────────────────────────────────────
    def _writeback_loop(self):
        if self._cuda:
            torch.cuda.set_device(self.device)
        while True:
            job = self._wb_q.get()
            if job is None:
                return
            try:
                if self._error is None:
                    t_us = time.monotonic_ns() / 1e3
                    if job == "drain":
                        label = "G wb drain"
                        with record_function(label):
                            self._retire_writebacks(wait_all=True)
                    elif job[0] == "acc_evict":
                        label = f"G acc evict {job[1].idx}"
                        with record_function(label):
                            self._do_acc_evict(job[1])
                    else:
                        label = "G d2h writeback"
                        with record_function(label):
                            self._do_writeback(*job)
                    if self._span_log is not None:
                        self._span_log.append((
                            "offload d2h writeback", label,
                            t_us, time.monotonic_ns() / 1e3 - t_us,
                        ))
            except BaseException as e:  # noqa: BLE001 — surface on next check
                with self._cv:
                    self._error = e
                    self._cv.notify_all()
            finally:
                self._wb_q.task_done()

    def _do_writeback(self, state: _ChunkState, grads: Dict[str, torch.Tensor],
                      ev) -> None:
        if state.staging is None:
            state.staging = {
                n: torch.empty(g.shape, dtype=g.dtype, device="cpu",
                               pin_memory=self._cuda)
                for n, g in grads.items()
            }
        if self._cuda:
            # this state's previous copy (if still in flight) targets the
            # same staging buffers we are about to overwrite — retire it first
            self._retire_writebacks(for_state=state)
            with torch.cuda.stream(self._d2h_stream):
                self._d2h_stream.wait_event(ev)
                for n, g in grads.items():
                    state.staging[n].copy_(g, non_blocking=True)
                    # the grad was allocated on the compute stream; make the
                    # allocator aware it is read on the D2H stream before the
                    # reference is dropped
                    g.record_stream(self._d2h_stream)
                done = torch.cuda.Event()
                done.record()
            # do NOT synchronize here: blocking this thread per packet idles
            # the D2H stream during every CPU accumulate and backpressures
            # compute through the bounded queue. Defer the accumulate until
            # the copy has actually landed (checked via event.query()).
            self._wb_pending.append((state, list(grads), done))
            self._retire_writebacks()  # only copies that already completed
        else:
            for n, g in grads.items():
                state.staging[n].copy_(g)
                state.grad_acc[n] += state.staging[n]

    def _retire_writebacks(self, for_state=None, wait_all: bool = False):
        """Accumulate landed D2H copies into ``grad_acc`` (writeback thread
        only). Entries retire in FIFO order — the D2H stream is FIFO, so the
        head always completes first.

        Default: retire only copies whose event already fired (non-blocking).
        ``for_state``: block until no pending entry targets that state (its
        staging buffers are about to be reused). ``wait_all``: drain
        everything (the flush paths).
        """
        pend = self._wb_pending
        while pend:
            must = wait_all or (
                for_state is not None
                and any(e[0] is for_state for e in pend)
            )
            if not must and not pend[0][2].query():
                break
            state, names, done = pend.popleft()
            done.synchronize()
            for n in names:
                state.grad_acc[n] += state.staging[n]

    def _wb_drain(self):
        """Flush queued writeback jobs AND retire all pending D2H copies.

        ``_wb_q.join()`` alone is not enough since copies retire lazily; the
        sentinel makes the writeback thread itself do the final blocking
        retirement (``_wb_pending`` is single-thread-owned by design).
        """
        self._wb_q.put("drain")
        self._wb_q.join()

    def _do_acc_evict(self, state: _ChunkState) -> None:
        """Spill a GPU grad accumulator home to its pinned CPU tensor.

        Stream-mode replacement for the per-microbatch CPU accumulate: the
        value MOVES (plain overwrite copy, no arithmetic) — after this the
        CPU ``grad_acc`` holds it and the GPU copy is freed. A later B visit
        reloads it via the loader ("A{i} acc reload")."""
        with self._cv:
            gpu, ev = state.acc_gpu, state.acc_ev
        if gpu is None:  # already spilled (a flush raced this job)
            with self._cv:
                self._acc_evicting.discard(state.idx)
                self._cv.notify_all()
            return
        if self._cuda:
            with torch.no_grad(), torch.cuda.stream(self._d2h_stream):
                if ev is not None:
                    # order after the H2D reload / last GPU add of this acc
                    self._d2h_stream.wait_event(ev)
                for n, g in gpu.items():
                    state.grad_acc[n].copy_(g, non_blocking=True)
                    g.record_stream(self._d2h_stream)
                done = torch.cuda.Event()
                done.record()
            done.synchronize()
        else:  # pragma: no cover — cpu engines accumulate in place
            with torch.no_grad():
                for n, g in gpu.items():
                    state.grad_acc[n].copy_(g)
        with self._cv:
            state.acc_gpu = None
            state.acc_where = "cpu"
            self._acc_res.discard(state.idx)
            self._acc_evicting.discard(state.idx)
            self.stats["acc_evictions"] += 1
            self._cv.notify_all()

    # ── compute-side residency handshake ───────────────────────────────────
    def _check_error(self):
        if self._error is not None:
            raise RuntimeError("offload worker thread failed") from self._error

    def _acquire(self, layer: int) -> Dict[str, torch.Tensor]:
        """Block until ``layer`` is on the GPU; returns its tensors by name."""
        state = self._state[layer]
        if state.gpu_pinned:
            return state.tensors
        t0 = time.perf_counter()
        with record_function(f"wait L{layer}"):
            with self._cv:
                while layer not in self._resident and self._error is None:
                    self._cv.wait()
                self._check_error()
                res = self._resident[layer]
                self._in_use = layer
        self.stats["acquire_wait_s"] += time.perf_counter() - t0
        if res.event is not None:
            # order the compute stream after the H2D copy (relay-mailbox style).
            # NB: current_stream(self.device) — the bare current_stream() is
            # the CALLING thread's current device, which may not be ours, and
            # a wait_event on another device's stream orders nothing.
            compute_stream = torch.cuda.current_stream(self.device)
            compute_stream.wait_event(res.event)
            # weights were allocated on the H2D stream but are consumed on the
            # compute stream; record so eviction (dropping the reference) does
            # not let the allocator hand the memory to a new copy while
            # queued compute kernels still read it
            for t in res.tensors.values():
                t.record_stream(compute_stream)
        return res.tensors

    def _release(self):
        """Advance the itinerary position and wake the loader."""
        with self._cv:
            self._in_use = None
            self._fpos += 1
            # compact the consumed prefix now and then
            if self._fpos > 8 * self.n:
                del self._future[: self._fpos]
                del self._future_kinds[: self._fpos]
                self._fpos = 0
            self._cv.notify_all()

    def _announce(self, itinerary: List[int],
                  kinds: Optional[Sequence[str]] = None):
        """Append upcoming chunk uses so the loader can prefetch ahead.

        ``kinds`` marks each entry "F" or "B" (default all "F"); the loader
        prefetches evicted grad accumulators for upcoming "B" entries in
        stream mode. Weight prefetch ignores kinds.
        """
        if kinds is None:
            kinds = ["F"] * len(itinerary)
        elif len(kinds) != len(itinerary):
            raise ValueError("kinds must align with the itinerary")
        with self._cv:
            self._check_error()
            self._future.extend(itinerary)
            self._future_kinds.extend(kinds)
            self._cv.notify_all()

    def _call_chunk(self, layer: int, tensors: Dict[str, torch.Tensor],
                    args: tuple):
        return torch.func.functional_call(self.chunks[layer], tensors, args)

    # ── activation offload (saved_tensors_hooks packets, notes §0.11) ───────
    # Everything here runs on the compute thread only (step() caller or the
    # stage's relay worker) — no lock. Copies ride the existing D2H/H2D
    # streams, ordered purely by events (no host syncs). Policy is the
    # simulator-validated "lazy": offload only under slot pressure, victim =
    # farthest next backward (exact Belady — bpos comes from the announced
    # schedule), clean re-drops are free, reloads prefetched one B ahead.

    @contextlib.contextmanager
    def _act_ctx(self, key, bpos: int, tensors: Dict[str, torch.Tensor]):
        """Install pack/unpack hooks around one chunk's forward.

        ``key`` identifies the packet (chunk idx for :meth:`step`,
        ``(mb, chunk)`` for the pipeline stage); ``bpos`` is its backward's
        position in the schedule (eviction ranks by it). ``tensors`` is the
        mapping passed to ``functional_call`` — their storage pointers are
        the weight filter (weights already stream; §0.11 probe recipe).
        MUST be installed OUTSIDE ``torch.utils.checkpoint`` regions, never
        inside (hooks outside see only region inputs — the desired set).
        """
        if not self.offload_activations:
            yield
            return
        weight_ptrs = {
            t.untyped_storage().data_ptr() for t in tensors.values()
            if t.untyped_storage().size() > 0
        }
        packet = _ActPacket(key, bpos)
        self._act_store[key] = packet
        dev = self.device

        def pack(t: torch.Tensor):
            if t.device != dev:
                return ("raw", t)
            st = t.untyped_storage()
            if st.size() == 0 or st.data_ptr() in weight_ptrs:
                return ("raw", t)
            # dedup: autograd saves the same tensor many times (e.g. an
            # input reused by several ops) — one entry, one copy
            dk = (st.data_ptr(), t.storage_offset(), t.dtype,
                  tuple(t.shape), tuple(t.stride()), t._version)
            idx = packet.dedup.get(dk)
            if idx is None:
                idx = len(packet.entries)
                packet.entries.append(_ActEntry(t))
                packet.dedup[dk] = idx
            return ("act", key, idx)

        def unpack(h):
            if h[0] == "raw":
                return h[1]
            return self._act_unpack(h[1], h[2])

        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            yield

    def _act_offload(self, packet: _ActPacket):
        """Evict a packet to pinned RAM (D2H stream) or re-drop if clean."""
        if packet.where == "gpu_clean":
            # the RAM copy from the previous offload is still valid
            for e in packet.entries:
                e.gpu = None
            packet.where = "cpu"
            packet.h2d_ev = None
            return
        if packet.entries:
            self.stats["act_offloads"] += 1
        for e in packet.entries:
            if e.buf is None:
                e.buf = self._act_pool.take(e.nbytes, pin=self._cuda)
                e.view = e.buf[: e.nbytes].view(e.dtype).view(e.shape)
            self.stats["act_bytes_offloaded"] += e.nbytes
        if self._cuda:
            ready = torch.cuda.Event()
            ready.record(torch.cuda.current_stream(self.device))
            with torch.no_grad(), torch.cuda.stream(self._d2h_stream):
                self._d2h_stream.wait_event(ready)
                for e in packet.entries:
                    src = (e.gpu if e.gpu.is_contiguous()
                           else e.gpu.contiguous())
                    e.view.copy_(src, non_blocking=True)
                    e.gpu.record_stream(self._d2h_stream)
                    e.gpu = None
                ev = torch.cuda.Event()
                ev.record()
            packet.d2h_ev = ev
        else:
            with torch.no_grad():
                for e in packet.entries:
                    e.view.copy_(e.gpu)
                    e.gpu = None
        packet.where = "cpu"
        packet.h2d_ev = None

    def _act_reload(self, key):
        """Bring an offloaded packet back to the GPU (H2D stream)."""
        packet = self._act_store[key]
        if packet.where != "cpu":
            return
        if packet.entries:
            self.stats["act_reloads"] += 1
        if self._cuda:
            with torch.no_grad(), torch.cuda.stream(self._h2d_stream):
                if packet.d2h_ev is not None:
                    self._h2d_stream.wait_event(packet.d2h_ev)
                for e in packet.entries:
                    g = torch.empty(e.shape, dtype=e.dtype, device=self.device)
                    g.copy_(e.view, non_blocking=True)
                    e.gpu = g
                ev = torch.cuda.Event()
                ev.record()
            packet.h2d_ev = ev
        else:
            with torch.no_grad():
                for e in packet.entries:
                    e.gpu = e.view.clone()
        packet.where = "gpu_clean"

    def _act_unpack(self, key, idx: int) -> torch.Tensor:
        """saved_tensors_hooks unpack: hand autograd the (reloaded) tensor."""
        packet = self._act_store[key]
        e = packet.entries[idx]
        if e.gpu is None:
            # prefetch miss — issue the reload now (still event-ordered,
            # the compute stream just waits longer)
            self._act_reload(key)
        if self._cuda and packet.h2d_ev is not None:
            cs = torch.cuda.current_stream(self.device)
            cs.wait_event(packet.h2d_ev)
            e.gpu.record_stream(cs)
        return e.gpu

    def _act_resident(self) -> List[_ActPacket]:
        return [p for p in self._act_store.values() if p.where != "cpu"]

    def _act_settle(self):
        """Lazy pressure check after a forward: evict down to act_slots.

        Victim = resident packet with the farthest backward (largest bpos),
        never the one mid-backward. A packet only exists once its forward
        ran, so occupancy can transiently hit act_slots+1 during a chunk's
        forward — the bound applies between chunks (as in the simulator).
        """
        resident = [p for p in self._act_resident()
                    if p.key != self._act_in_use]
        excess = len(self._act_resident()) - self.act_slots
        while excess > 0 and resident:
            victim = max(resident, key=lambda p: p.bpos)
            self._act_offload(victim)
            resident.remove(victim)
            excess -= 1

    def _act_stage_backward(self, key, prefetch_keys: Sequence[object] = ()):
        """Prepare backward for ``key``: reload it + prefetch one B ahead."""
        keep = {key, *prefetch_keys}
        packet = self._act_store.get(key)
        if packet is not None and packet.where == "cpu":
            self._act_make_room(keep)
            self._act_reload(key)
        self._act_in_use = key
        for nk in prefetch_keys:
            nxt = self._act_store.get(nk)
            if nxt is not None and nxt.where == "cpu":
                self._act_make_room(keep)
                self._act_reload(nk)

    def _act_make_room(self, keep):
        """Evict farthest-backward packets until a slot is free."""
        resident = [p for p in self._act_resident() if p.key not in keep]
        while len(self._act_resident()) >= self.act_slots and resident:
            victim = max(resident, key=lambda p: p.bpos)
            self._act_offload(victim)
            resident.remove(victim)

    def _act_discard(self, key):
        """Free a packet after its backward; pinned buffers go to the pool."""
        packet = self._act_store.pop(key, None)
        if packet is None:
            return
        ev = packet.h2d_ev or packet.d2h_ev  # last op touching the buffers
        for e in packet.entries:
            if e.buf is not None:
                self._act_pool.give(e.buf, ev)
            e.gpu = e.view = e.buf = None
        if self._act_in_use == key:
            self._act_in_use = None

    def _act_assert_drained(self):
        """Every packet must be consumed by its backward within the step."""
        if self._act_store:
            leaked = list(self._act_store)
            self._act_store.clear()
            self._act_in_use = None
            raise RuntimeError(
                f"activation packets leaked past the step: {leaked} — "
                "every announced backward must run (internal bug)"
            )

    # ── tuple intermediates ─────────────────────────────────────────────────
    # A chunk may return a single tensor OR a tuple of tensors; a tuple's
    # elements become the next chunk's positional args. Internally everything
    # is normalized to tuples; the raw (un-normalized) output is preserved
    # for the final result and the loss_fn.
    def _to_device_args(self, x) -> tuple:
        if isinstance(x, (tuple, list)):
            return tuple(t.to(self.device) for t in x)
        return (x.to(self.device),)

    @staticmethod
    def _as_tuple(out) -> tuple:
        return out if isinstance(out, tuple) else (out,)

    @staticmethod
    def _detach_like(raw):
        """Detach preserving the chunk's raw output structure."""
        if isinstance(raw, tuple):
            return tuple(t.detach() for t in raw)
        return raw.detach()

    # ── inference ──────────────────────────────────────────────────────────
    @torch.no_grad()
    def forward(self, x) -> torch.Tensor:
        """Streamed forward pass (the simulator's inference "ring").

        ``x`` may be a single tensor or a tuple; a chunk returning a tuple
        feeds its elements as positional args to the next chunk.
        """
        self._announce(list(range(self.n)))
        hs = self._to_device_args(x)
        raw = hs[0]
        for i in range(self.n):
            with record_function(f"F{i} infer"):
                tensors = self._acquire(i)
                raw = self._call_chunk(i, tensors, hs)
                hs = self._as_tuple(raw)
            self._release()
        return raw

    # ── training ───────────────────────────────────────────────────────────
    def step(
        self,
        x: torch.Tensor,
        *,
        targets: Optional[torch.Tensor] = None,
        loss_fn: Optional[Callable] = None,
        grad_outputs=None,
        profile_path: Optional[str] = None,
    ) -> OffloadStepResult:
        """
        One forward+backward through the streamed chunks (training "echo":
        F0..F(n-1) then B(n-1)..B0, self-warming at the turnaround).

        Backward strategy follows ``keep_activations`` (see the class
        docstring): recompute (default) re-runs each chunk's forward during
        backward; keep retains per-chunk graphs and skips the recompute.
        Param grads accumulate into per-chunk accumulators (CPU for streamed
        chunks, GPU for pinned); call :meth:`flush_grads` before the
        optimizer step. Repeated ``step()`` calls accumulate (microbatch
        gradient accumulation).

        ``grad_outputs``: grad-bypass escape hatch (mirrors the pipeline's).
        Skip the loss entirely and feed a precomputed ``dL/dOutput`` into
        the last chunk's backward — either a tensor (or tuple aligned with
        a tuple output), or a callable ``(output, targets) -> grad`` that is
        resolved with the live output. Mutually exclusive with ``loss_fn``;
        the result's ``.loss`` raises since no loss is computed.

        Mixed precision: wrap the call in ``torch.autocast`` — everything
        (forward, backward recompute, loss) runs on the calling thread, so
        the ambient context applies; no engine support needed (unlike the
        pipeline, whose stage workers are separate threads).

        ``profile_path``: if set, capture a torch.profiler (kineto) trace of
        the whole step — compute ops (``F3`` / ``B3``), H2D loads (``L3 h2d
        load``), D2H grad writebacks, stall waits (``wait L3``), every CUDA
        kernel and memcpy — and write Chrome-trace JSON to this path. Drop
        the file into https://ui.perfetto.dev to inspect the overlap.
        """
        if grad_outputs is not None and loss_fn is not None:
            raise ValueError(
                "grad_outputs and loss_fn are mutually exclusive: grad "
                "bypass skips the loss computation entirely"
            )
        if self.nvme_layers and not self._nvme_gate_passed:
            _check_nvme_unlocked()  # training-only; forward() never gates
            self._nvme_gate_passed = True
        loss_fn = loss_fn or (lambda out, _: out.sum())
        target = targets.to(self.device) if targets is not None else None

        def _run() -> OffloadStepResult:
            self._announce(
                list(range(self.n)) + list(range(self.n - 1, -1, -1)),
                kinds=["F"] * self.n + ["B"] * self.n,
            )
            if self.keep_activations == "checkpoint":
                result = self._step_keep(x, target, loss_fn,
                                         use_checkpoint=True,
                                         grad_outputs=grad_outputs)
            elif self.keep_activations:
                result = self._step_keep(x, target, loss_fn,
                                         grad_outputs=grad_outputs)
            else:
                result = self._step_recompute(x, target, loss_fn,
                                              grad_outputs=grad_outputs)
            # ensure all writebacks landed before the caller touches grad_acc
            self._wb_drain()
            self._check_error()
            return result

        if profile_path is None:
            return _run()
        from torch.profiler import ProfilerActivity, profile as _torch_profile
        activities = [ProfilerActivity.CPU]
        if self._cuda:
            activities.append(ProfilerActivity.CUDA)
        with _torch_profile(activities=activities) as prof:
            self._span_log = []
            # calibration marker: kineto timestamps use its own clock, so
            # pair this span's trace ts with a monotonic reading to place
            # the worker-thread spans (see _inject_thread_spans)
            with record_function("offload_clock_sync"):
                sync_wall_us = time.monotonic_ns() / 1e3
            result = _run()
            # drain device work BEFORE the profiler stops so the trace
            # includes the trailing copies/kernels
            if self._cuda:
                torch.cuda.synchronize(self.device)
        spans, self._span_log = self._span_log, None
        prof.export_chrome_trace(profile_path)
        self._inject_thread_spans(profile_path, spans, sync_wall_us)
        return result

    @staticmethod
    def _inject_thread_spans(path: str, spans: List[tuple],
                             sync_wall_us: float) -> None:
        """Splice loader/writeback spans into an exported chrome trace.

        Kineto only records ``record_function`` spans on the thread that
        entered the profiler, so the H2D-load / D2H-writeback annotations
        from our worker threads never appear. We log them against
        ``time.monotonic_ns`` and shift onto kineto's clock using the
        ``offload_clock_sync`` marker, adding one named track per worker.
        """
        with open(path) as f:
            trace = json.load(f)
        events = trace.get("traceEvents", [])
        sync = next(
            (e for e in events if e.get("name") == "offload_clock_sync"), None
        )
        if sync is None or not spans:
            return
        offset = sync["ts"] - sync_wall_us
        pid = sync.get("pid", 0)
        tids: Dict[str, int] = {}
        for track, name, t_us, dur_us in spans:
            tid = tids.setdefault(track, 900_000 + len(tids))
            events.append({
                "ph": "X", "cat": "user_annotation", "name": name,
                "pid": pid, "tid": tid, "ts": t_us + offset, "dur": dur_us,
            })
        for track, tid in tids.items():
            events.append({
                "ph": "M", "name": "thread_name", "pid": pid, "tid": tid,
                "args": {"name": track},
            })
        with open(path, "w") as f:
            json.dump(trace, f)

    def _resolve_grad_outputs(self, grad_outputs, raw_out, target,
                              outs: tuple) -> tuple:
        """Normalize a grad-bypass ``grad_outputs`` for the last chunk.

        Callable form is resolved with the (detached) live output and the
        targets, mirroring ``loss_fn``; the values are gradients, not graph
        nodes, so resolution runs under ``no_grad``.
        """
        if callable(grad_outputs):
            with torch.no_grad():
                go = grad_outputs(self._detach_like(raw_out), target)
        else:
            go = grad_outputs
        gos = self._as_tuple(go)
        if len(gos) != len(outs):
            raise ValueError(
                f"grad_outputs has {len(gos)} element(s) but the last chunk "
                f"returned {len(outs)}"
            )
        return tuple(
            g.to(self.device) if g is not None else None for g in gos
        )

    def _grads_for(self, i: int, tensors: Dict[str, torch.Tensor],
                   outs: tuple, inps: tuple, grad_outs, loss_fn, target,
                   is_last: bool, raw_out):
        """autograd.grad for chunk ``i`` + the loss on the last chunk.

        ``outs``/``inps`` are the normalized output/input tuples;
        ``grad_outs`` is a tuple aligned with ``outs`` (``None`` entries =
        no gradient flowed into that element); ``raw_out`` is the chunk's
        raw (un-normalized) output for the ``loss_fn`` call.

        Returns ``(loss_or_none, input_grads_tuple, named_param_grads)``
        where the input-grads tuple is aligned with ``inps`` (``None`` for
        elements that need no grad — non-float passthroughs etc.).
        """
        state = self._state[i]
        param_names = [nme for nme in tensors if nme in state.param_names]
        params = [tensors[nme] for nme in param_names]
        need_in = [t.requires_grad for t in inps]
        diff_inputs = [t for t, nd in zip(inps, need_in) if nd]
        loss = None
        with torch.enable_grad():
            if is_last:
                loss = loss_fn(raw_out, target)
                grads = torch.autograd.grad(
                    loss, diff_inputs + params, allow_unused=True,
                )
            else:
                # differentiate only the outputs a gradient actually flowed
                # into (int passthroughs and dead branches contribute None)
                douts, dgrads = [], []
                for o, g in zip(outs, grad_outs):
                    if g is not None and o.requires_grad:
                        douts.append(o)
                        dgrads.append(g)
                if not douts:
                    return None, tuple(None for _ in inps), {}
                grads = torch.autograd.grad(
                    douts, diff_inputs + params,
                    grad_outputs=dgrads, allow_unused=True,
                )
        in_grads = iter(grads[: len(diff_inputs)])
        input_grads = tuple(
            next(in_grads) if nd else None for nd in need_in
        )
        named_grads = {
            nme: g
            for nme, g in zip(param_names, grads[len(diff_inputs):])
            if g is not None
        }
        return loss, input_grads, named_grads

    def _accumulate(self, state: _ChunkState,
                    named_grads: Dict[str, torch.Tensor]):
        """Route param grads.

        Pinned chunks add into their permanent GPU accumulator. Streamed
        chunks: ``grad_accum="stream"`` adds on the GPU into a streamed
        accumulator (zero CPU math); ``"cpu"`` ships a per-microbatch packet
        to the writeback thread (D2H + CPU add — the legacy path).
        """
        if state.gpu_pinned:
            # grads stay on the GPU next to the pinned weights
            with torch.no_grad():
                for nme, g in named_grads.items():
                    state.grad_acc[nme] += g.detach()
        elif not named_grads:
            return
        elif self.grad_accum == "stream":
            self._acc_add(state, named_grads)
        else:
            if self._cuda:
                ev = torch.cuda.Event()
                # grads are ready on OUR device's compute stream (record on it
                # explicitly — the caller's current device may differ)
                ev.record(torch.cuda.current_stream(self.device))
            else:
                ev = None
            self._wb_q.put(
                (state, {nme: g.detach() for nme, g in named_grads.items()}, ev)
            )

    def _acc_add(self, state: _ChunkState,
                 named_grads: Dict[str, torch.Tensor]):
        """Add ``named_grads`` into the chunk's streamed GPU accumulator.

        First touch in an accumulation cycle zero-inits the acc in place (no
        transfer). An evicted acc is reloaded by the loader (it sees this B
        at the head of the announced future) — we only wait. Blocking here
        is the stream-mode analogue of the old bounded-queue backpressure.
        """
        layer = state.idx
        if not self._cuda:
            # device IS the storage tier: add straight into the accumulator
            with torch.no_grad():
                for nme, g in named_grads.items():
                    state.grad_acc[nme] += g.detach()
            state.acc_where = "cpu"
            return

        while True:
            evict_target = None
            ready = False
            with self._cv:
                self._check_error()
                if layer in self._acc_res and layer not in self._acc_evicting:
                    self._acc_adding = layer
                    ready = True
                elif state.acc_where == "empty":
                    if len(self._acc_res) < self.acc_slots:
                        with torch.no_grad():
                            state.acc_gpu = {
                                n: torch.zeros_like(a, device=self.device)
                                for n, a in state.grad_acc.items()
                            }
                        state.acc_ev = None
                        state.acc_fresh = False
                        state.acc_where = "gpu"
                        self._acc_res.add(layer)
                        self._acc_adding = layer
                        ready = True
                    else:
                        # need a slot NOW: any idle resident acc will do
                        victim = self._acc_pick_victim(-1.0)
                        if victim is not None:
                            self._acc_evicting.add(victim)
                            evict_target = self._state[victim]
                # acc_where == "cpu": reload is the loader's job — wait
                if not ready and evict_target is None:
                    self._cv.wait()
                    continue
            if ready:
                break
            # enqueue outside the lock (put() may block on a full queue)
            self._wb_q.put(("acc_evict", evict_target))

        cs = torch.cuda.current_stream(self.device)
        fresh = getattr(state, "acc_fresh", False)
        if fresh and state.acc_ev is not None:
            # the acc landed on the H2D stream: order our adds after it
            cs.wait_event(state.acc_ev)
        with torch.no_grad():
            for nme, g in named_grads.items():
                state.acc_gpu[nme] += g.detach()
            if fresh:
                for t in state.acc_gpu.values():
                    t.record_stream(cs)
        ev = torch.cuda.Event()
        ev.record(cs)
        with self._cv:
            state.acc_ev = ev
            state.acc_fresh = False
            self._acc_adding = None
            self._cv.notify_all()

    def _step_recompute(self, x, target, loss_fn,
                        grad_outputs=None) -> OffloadStepResult:
        """no_grad forward caching boundaries; backward recomputes per chunk."""
        n = self.n
        boundary: List[Optional[tuple]] = []
        hs = self._to_device_args(x)
        raw = hs[0]
        with torch.no_grad():
            for i in range(n):
                boundary.append(hs)
                with record_function(f"F{i}"):
                    tensors = self._acquire(i)
                    raw = self._call_chunk(i, tensors, hs)
                    hs = self._as_tuple(raw)
                self._release()
        output = raw

        loss_out: Optional[torch.Tensor] = None
        grad_outs: Optional[tuple] = None
        for i in range(n - 1, -1, -1):
            with record_function(f"B{i} recompute+grad"):
                tensors = self._acquire(i)
                state = self._state[i]
                params = [tensors[nme] for nme in state.param_names]
                for p in params:
                    p.requires_grad_(True)

                inps = tuple(t.detach() for t in boundary[i])
                for t in inps:
                    t.requires_grad_(i > 0 and t.is_floating_point())
                with torch.enable_grad():
                    raw_out = self._call_chunk(i, tensors, inps)
                outs = self._as_tuple(raw_out)
                is_last = i == n - 1
                if is_last and grad_outputs is not None:
                    grad_outs = self._resolve_grad_outputs(
                        grad_outputs, raw_out, target, outs)
                    is_last = False  # seeded like a middle chunk
                loss, grad_outs, named_grads = self._grads_for(
                    i, tensors, outs, inps, grad_outs, loss_fn, target,
                    is_last=is_last, raw_out=raw_out,
                )
                if loss is not None:
                    loss_out = loss.detach()
                self._accumulate(state, named_grads)
                for p in params:
                    p.requires_grad_(False)
                boundary[i] = None  # free the boundary activations
            self._release()
        return OffloadStepResult(output, loss_out,
                                 grad_bypassed=grad_outputs is not None)

    def _step_keep(self, x, target, loss_fn, use_checkpoint: bool = False,
                   grad_outputs=None) -> OffloadStepResult:
        """Grad-enabled forward keeping per-chunk graphs; no recompute.

        The graphs reference each chunk's stable ``graph_tensors``; the loader
        frees/refills their storage across evictions, so weights still stream
        while every internal activation stays on the GPU. Backward for chunk
        ``i`` runs only after ``_acquire(i)`` guarantees the storage is
        refilled (saved tensors read the same objects).

        ``use_checkpoint``: wrap each chunk in non-reentrant
        ``torch.utils.checkpoint`` — internal activations are dropped at
        forward and recomputed lazily when ``torch.autograd.grad`` unpacks
        them during that chunk's backward (which is after ``_acquire(i)``,
        so the weight storages are refilled by then). Checkpoint stashes and
        restores the RNG state, so stochastic chunks recompute the SAME
        dropout masks — unlike ``_step_recompute``.
        """
        n = self.n
        act_on = self.offload_activations
        # per-chunk (input_leaves, outs, raw_out, tensors)
        cache: List[Optional[tuple]] = [None] * n
        hs = self._to_device_args(x)
        raw_out = hs[0]
        for i in range(n):
            with record_function(f"F{i}"):
                tensors = self._acquire(i)
                inps = tuple(t.detach() for t in hs)
                for t in inps:
                    t.requires_grad_(i > 0 and t.is_floating_point())
                # bpos: B_i runs at echo position 2n-1-i (F0..Fn-1, Bn-1..B0)
                with self._act_ctx(i, 2 * n - 1 - i, tensors), \
                        torch.enable_grad():
                    if use_checkpoint:
                        raw_out = torch.utils.checkpoint.checkpoint(
                            lambda *a, _i=i, _ts=tensors:
                                self._call_chunk(_i, _ts, a),
                            *inps, use_reentrant=False,
                        )
                    else:
                        raw_out = self._call_chunk(i, tensors, inps)
                outs = self._as_tuple(raw_out)
                cache[i] = (inps, outs, raw_out, tensors)
                hs = tuple(t.detach() for t in outs)
                if act_on:
                    self._act_settle()
            self._release()
        output = self._detach_like(raw_out)

        loss_out: Optional[torch.Tensor] = None
        grad_outs: Optional[tuple] = None
        for i in range(n - 1, -1, -1):
            with record_function(f"B{i} grad"):
                self._acquire(i)  # refills evicted weights before backward
                if act_on:
                    self._act_stage_backward(i, (i - 1,) if i > 0 else ())
                inps, outs, raw_out, tensors = cache[i]
                is_last = i == n - 1
                if is_last and grad_outputs is not None:
                    grad_outs = self._resolve_grad_outputs(
                        grad_outputs, raw_out, target, outs)
                    is_last = False  # seeded like a middle chunk
                loss, grad_outs, named_grads = self._grads_for(
                    i, tensors, outs, inps, grad_outs, loss_fn, target,
                    is_last=is_last, raw_out=raw_out,
                )
                if loss is not None:
                    loss_out = loss.detach()
                self._accumulate(self._state[i], named_grads)
                cache[i] = None  # release the chunk's graph + activations
                if act_on:
                    self._act_discard(i)
            self._release()
        if act_on:
            self._act_assert_drained()
        return OffloadStepResult(output, loss_out,
                                 grad_bypassed=grad_outputs is not None)

    # ── grads / optimizer interop ──────────────────────────────────────────
    def flush_grads(self, scale: float = 1.0):
        """Write accumulated grads into ``.grad`` and reset (pipeline-style).

        Streamed chunks get CPU grads on their CPU params; pinned chunks get
        GPU grads on their GPU params. With k accumulated ``step()`` calls,
        pass ``scale=1/k`` for the mean.

        The ``.grad`` tensors are persistent buffers reused across calls
        (CPU ones live in pinned memory). Recommended optimizer:
        ``torch.optim.AdamW(model.parameters(), fused=True)`` — torch groups
        params by device, so the CPU masters get the fused CPU kernel and
        pinned chunks the fused CUDA kernel.

        Also drops the streamed chunks' resident GPU weight copies
        (:meth:`invalidate_residency`): an optimizer step typically follows,
        and a leftover resident copy would silently serve pre-update weights
        to the next forward.
        """
        self._wb_drain()
        self._check_error()
        self._acc_spill_all()
        for state in self._state:
            named = dict(state.module.named_parameters())
            for nme, acc in state.grad_acc.items():
                buf = state.flush_bufs.get(nme)
                if buf is None:
                    buf = torch.empty_like(acc)
                    if self._cuda and acc.device.type == "cpu":
                        buf = buf.pin_memory()
                    state.flush_bufs[nme] = buf
                torch.mul(acc, scale, out=buf)
                named[nme].grad = buf
                acc.zero_()
            if not state.gpu_pinned:
                state.acc_where = "empty"  # next cycle zero-inits on GPU
        self.invalidate_residency()

    def _acc_spill_all(self):
        """Bring every streamed GPU accumulator home to CPU (stream mode).

        Batched on the D2H stream with a single host sync. Runs after
        ``_wb_drain`` (no eviction jobs in flight) and blocks a concurrent
        loader reload from racing the spill."""
        if self.grad_accum != "stream" or not self._cuda:
            return
        with self._cv:
            while ((self._acc_in_flight is not None or self._acc_evicting)
                   and self._error is None):
                self._cv.wait()
            self._check_error()
            spill = [st for st in self._state if st.acc_gpu is not None]
            for st in spill:
                self._acc_evicting.add(st.idx)
        if not spill:
            return
        with torch.no_grad(), torch.cuda.stream(self._d2h_stream):
            for st in spill:
                if st.acc_ev is not None:
                    self._d2h_stream.wait_event(st.acc_ev)
                for n, g in st.acc_gpu.items():
                    st.grad_acc[n].copy_(g, non_blocking=True)
                    g.record_stream(self._d2h_stream)
            done = torch.cuda.Event()
            done.record()
        done.synchronize()
        with self._cv:
            for st in spill:
                st.acc_gpu = None
                st.acc_where = "cpu"
                st.acc_ev = None
                st.acc_fresh = False
                self._acc_res.discard(st.idx)
                self._acc_evicting.discard(st.idx)
            self._cv.notify_all()

    def invalidate_residency(self):
        """Drop streamed GPU weight copies so the next use reloads masters.

        Call after updating params in place (an optimizer step) if you did
        not go through :meth:`flush_grads` (which calls this for you).
        Resident copies are a cache keyed on the CPU masters; they do NOT
        see master updates, so without invalidation the next step would
        compute with stale weights for whichever chunks happened to stay
        resident. Pinned chunks live on the GPU (the optimizer updates them
        directly) and are unaffected.
        """
        with self._cv:
            # a load already in flight would land post-invalidation holding
            # pre-update weights — let it land first, then purge it too
            while self._in_flight is not None and self._error is None:
                self._cv.wait()
            for layer in list(self._resident):
                del self._resident[layer]
                if self._keep_graph:
                    state = self._state[layer]
                    if state.graph_tensors is not None:
                        for t in state.graph_tensors.values():
                            t.untyped_storage().resize_(0)
            self._cv.notify_all()

    def zero_grad_acc(self):
        if self.grad_accum == "stream":
            with self._cv:
                while ((self._acc_in_flight is not None
                        or self._acc_evicting) and self._error is None):
                    self._cv.wait()
                self._check_error()
                for st in self._state:
                    if not st.gpu_pinned:
                        st.acc_gpu = None  # drop — the value is discarded
                        st.acc_where = "empty"
                        st.acc_ev = None
                        st.acc_fresh = False
                        self._acc_res.discard(st.idx)
                self._cv.notify_all()
        for state in self._state:
            for acc in state.grad_acc.values():
                acc.zero_()

    # ── lifecycle ──────────────────────────────────────────────────────────
    def close(self):
        """Stop the loader/writeback threads (idempotent).

        Also unlinks the NVMe scratch file. The mapped master tensors stay
        readable until garbage collected (POSIX unlink semantics), but the
        file is gone — save a checkpoint first if you need the weights.
        """
        if self._closed:
            return
        with self._cv:
            self._closed = True
            self._cv.notify_all()
        self._wb_q.put(None)
        self._loader.join(timeout=5)
        self._writeback.join(timeout=5)
        if self._nvme_store is not None:
            self._nvme_store.close()
            self._nvme_store = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
