"""
ramtorch.offload_optimizer  (PRIVATE / educational — not exported)
------------------------------------------------------------------
Windowed CPU->GPU streaming AdamW ("the optimizer step, offload-style").

.. warning::
   **Kept for educational purposes only** — in practice, use
   ``torch.optim.AdamW(..., fused=True)`` instead. The fused CPU kernel is a
   single multithreaded pass over host RAM (~DDR bandwidth, e.g. ~28 B/param
   at 100+ GB/s), while this streamed design is bounded by PCIe (~28 B/param
   at 20-50 GB/s). No amount of upload/compute/writeback overlap lets PCIe
   beat one pass over DDR, so on any machine whose RAM is faster than its
   PCIe link the fused CPU optimizer wins outright (measured ~33 ms vs
   ~104-141 ms for 128M fp32 params on an RTX PRO 6000 box; see
   ``examples/offload_optimizer_check.py``). This module still handily beats
   the *non-fused* foreach CPU path and the legacy per-param-bounce
   :class:`ramtorch.AdamW`, and remains a clean worked example of the
   window/stream/event pattern applied to an optimizer — hence it stays.

Plain (non-fused) torch optimizers on CPU params run many eager passes over
the state, which is usually the bottleneck once
:class:`ramtorch.OffloadModel` has made the forward/backward stream-bound.
The legacy :class:`ramtorch.AdamW` bounces each parameter to the GPU, but
per-param, on a single stream, with periodic global synchronizes — no
prefetch, no overlap.

:class:`OffloadAdamW` applies the same sliding-window design the offload
executor uses for weights, to the optimizer step itself:

* master params, grads, and Adam state (``exp_avg`` / ``exp_avg_sq``) live in
  CPU **pinned** memory;
* params are packed into transfer **buckets** (~``bucket_mb`` each, never
  splitting a param, never crossing a param group);
* ``window`` staging slot-sets live on the GPU; three CUDA streams — H2D
  prefetch, compute, D2H writeback — are chained per bucket with events, so
  bucket ``i+1`` uploads while bucket ``i`` computes and bucket ``i-1``
  writes back. The step becomes PCIe-bound instead of CPU-bound;
* the math is the exact fp32 ``torch.optim.AdamW`` update (same op order as
  torch's multi-tensor path), so trajectories match ``torch.optim.AdamW``;
* params already on a GPU (e.g. OffloadModel's pinned chunks) skip the
  window entirely and get a direct on-device foreach update, state resident
  next to them.

Unlike the executor there are no worker threads: nothing here blocks on
autograd, so a single Python thread enqueuing onto the three streams (with
event fences for slot reuse) gets full overlap. ``step()`` returns after a
final D2H sync, so the pinned masters are consistent when OffloadModel's
loader next reads them.

Usage with OffloadModel (note the private import)::

    from ramtorch.offload_optimizer import OffloadAdamW

    model = OffloadModel(chunks, device="cuda:0", window=2, pin=4)
    opt = OffloadAdamW(model.parameters(), lr=1e-3)

    res = model.step(x, targets=y, loss_fn=F.cross_entropy)
    model.flush_grads()      # pinned .grad buffers + residency invalidation
    opt.step()
    model.zero_grad_acc()

Also works standalone on any model whose params live on the CPU.

Notes
-----
* ``state_dtype=torch.bfloat16`` halves the state's pinned RAM and PCIe
  traffic; compute still happens in fp32 on the GPU. Pair it with
  ``stochastic_rounding=True`` to avoid the bias of repeated
  round-to-nearest on the tiny state updates.
* fp32 CPU masters take the fast path (pure async pinned copies). Non-fp32
  params still work but their H2D conversion happens on the host — keep
  masters fp32 for speed.
* GPU staging memory is ``window * 4 * max_bucket_bytes`` (+ raw staging
  when ``state_dtype`` is not fp32).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
from torch.optim import Optimizer

from .stochastic_optimizers.adamw import copy_stochastic_

__all__ = ["OffloadAdamW"]


class _Slot:
    """One GPU staging slot: flat fp32 buffers for p/g/m/v + fence events.

    ``mr``/``vr`` are raw staging buffers in the (non-fp32) state dtype so
    state copies stay pure async memcpys; conversion to/from fp32 happens on
    the GPU.
    """

    def __init__(self, numel: int, device: torch.device,
                 state_dtype: torch.dtype):
        self.p = torch.empty(numel, dtype=torch.float32, device=device)
        self.g = torch.empty(numel, dtype=torch.float32, device=device)
        self.m = torch.empty(numel, dtype=torch.float32, device=device)
        self.v = torch.empty(numel, dtype=torch.float32, device=device)
        if state_dtype != torch.float32:
            self.mr = torch.empty(numel, dtype=state_dtype, device=device)
            self.vr = torch.empty(numel, dtype=state_dtype, device=device)
        else:
            self.mr = self.vr = None
        self.h2d_done = torch.cuda.Event()
        self.compute_done = torch.cuda.Event()
        self.d2h_done = torch.cuda.Event()
        self.used = False  # d2h_done has been recorded at least once


class OffloadAdamW(Optimizer):
    """
    AdamW with CPU-resident state, streamed through a GPU window per step.

    Parameters
    ----------
    params       : iterable of params or param-group dicts (torch style).
    lr, betas, eps, weight_decay : AdamW hyperparameters (torch defaults).
    decoupled_weight_decay :
                   ``True`` (default) = AdamW (decoupled decay);
                   ``False`` = classic Adam L2 (decay folded into the grad).
    bucket_mb    : target transfer bucket size in MiB of fp32 param data.
                   Buckets never split a param and never span param groups.
    window       : GPU staging slot-sets (>= 2 overlaps upload / compute /
                   writeback; 1 disables overlap).
    device       : GPU used for the update math (default: current CUDA
                   device). ``"cpu"`` (or no CUDA) falls back to plain
                   foreach CPU math.
    state_dtype  : dtype of the CPU-resident ``exp_avg``/``exp_avg_sq``
                   (fp32 default; bf16 halves state RAM + PCIe traffic,
                   math still fp32 on the GPU).
    stochastic_rounding :
                   use stochastic rounding when writing bf16 state (and
                   bf16 params) back — recommended with bf16 state.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        *,
        decoupled_weight_decay: bool = True,
        bucket_mb: float = 32.0,
        window: int = 2,
        device: Optional[Union[str, torch.device]] = None,
        state_dtype: torch.dtype = torch.float32,
        stochastic_rounding: bool = False,
    ):
        if lr < 0.0:
            raise ValueError(f"invalid lr: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid betas: {betas}")
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

        if device is None:
            device = ("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._cuda = self.device.type == "cuda"
        if not self._cuda:
            state_dtype = torch.float32  # CPU fallback computes in-place fp32
        self._state_dtype = state_dtype
        self._stochastic = stochastic_rounding
        self._decoupled = decoupled_weight_decay
        self._bucket_bytes = int(bucket_mb * 2**20)
        self.window = window

        # built lazily (and rebuilt after add_param_group)
        self._cpu_buckets: Optional[List[Tuple[int, List[Tuple]]]] = None
        self._gpu_params: List[Tuple[int, List[torch.Tensor]]] = []
        self._max_bucket_numel = 0
        self._slots: Optional[List[_Slot]] = None
        self._h2d_stream = None
        self._compute_stream = None
        self._d2h_stream = None

    # ── plan / state management ─────────────────────────────────────────────
    def add_param_group(self, param_group):
        super().add_param_group(param_group)
        self._cpu_buckets = None  # force replan

    def _build_plan(self):
        """Pack CPU params into transfer buckets; collect GPU params."""
        cpu_buckets: List[Tuple[int, List[Tuple]]] = []
        gpu_params: List[Tuple[int, List[torch.Tensor]]] = []
        max_numel = 0
        for gi, group in enumerate(self.param_groups):
            gpu = [p for p in group["params"] if p.device.type != "cpu"]
            if gpu:
                gpu_params.append((gi, gpu))
            items: List[Tuple] = []
            total = 0
            for p in group["params"]:
                if p.device.type != "cpu":
                    continue
                if not p.is_contiguous():
                    raise ValueError(
                        "OffloadAdamW requires contiguous params "
                        f"(got non-contiguous {tuple(p.shape)})"
                    )
                n = p.numel()
                if items and (total + n) * 4 > self._bucket_bytes:
                    cpu_buckets.append((gi, items))
                    max_numel = max(max_numel, total)
                    items, total = [], 0
                items.append((p, total, n))
                total += n
            if items:
                cpu_buckets.append((gi, items))
                max_numel = max(max_numel, total)
        self._cpu_buckets = cpu_buckets
        self._gpu_params = gpu_params
        if max_numel > self._max_bucket_numel:
            self._max_bucket_numel = max_numel
            self._slots = None  # reallocate bigger staging

    def _ensure_slots(self):
        if self._slots is not None:
            return
        n_slots = min(self.window, max(1, len(self._cpu_buckets)))
        self._slots = [
            _Slot(self._max_bucket_numel, self.device, self._state_dtype)
            for _ in range(n_slots)
        ]
        if self._h2d_stream is None:
            self._h2d_stream = torch.cuda.Stream(self.device)
            self._compute_stream = torch.cuda.Stream(self.device)
            self._d2h_stream = torch.cuda.Stream(self.device)

    def _ensure_state(self, p: torch.Tensor) -> dict:
        st = self.state[p]
        if len(st) == 0:
            st["step"] = 0
            if p.device.type == "cpu":
                ea = torch.zeros(p.shape, dtype=self._state_dtype)
                es = torch.zeros(p.shape, dtype=self._state_dtype)
                if self._cuda:
                    ea, es = ea.pin_memory(), es.pin_memory()
            else:
                # GPU-resident param: state lives next to it, torch-style
                ea, es = torch.zeros_like(p), torch.zeros_like(p)
            st["exp_avg"], st["exp_avg_sq"] = ea, es
        return st

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        # the base class casts state to param dtype/device; restore our
        # CPU-state contract (state_dtype + pinned memory)
        if not self._cuda:
            return
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if not st:
                    continue
                for k in ("exp_avg", "exp_avg_sq"):
                    t = st[k]
                    if t.device.type == "cpu":
                        t = t.to(self._state_dtype)
                        if not t.is_pinned():
                            t = t.pin_memory()
                        st[k] = t

    # ── the AdamW math (torch.optim.AdamW multi-tensor op order) ────────────
    @staticmethod
    def _adamw_math(ps, gs, ms, vs, steps, *, lr, beta1, beta2, eps,
                    weight_decay, decoupled):
        if weight_decay != 0:
            if decoupled:
                torch._foreach_mul_(ps, 1 - lr * weight_decay)
            else:
                gs = torch._foreach_add(gs, ps, alpha=weight_decay)
        torch._foreach_lerp_(ms, gs, 1 - beta1)
        torch._foreach_mul_(vs, beta2)
        torch._foreach_addcmul_(vs, gs, gs, 1 - beta2)
        step_size = [-(lr / (1 - beta1 ** t)) for t in steps]
        bc2_sqrt = [math.sqrt(1 - beta2 ** t) for t in steps]
        denom = torch._foreach_sqrt(vs)
        torch._foreach_div_(denom, bc2_sqrt)
        torch._foreach_add_(denom, eps)
        torch._foreach_addcdiv_(ps, ms, denom, step_size)

    def _hp(self, group):
        beta1, beta2 = group["betas"]
        return dict(lr=group["lr"], beta1=beta1, beta2=beta2,
                    eps=group["eps"], weight_decay=group["weight_decay"],
                    decoupled=self._decoupled)

    # ── step ────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._cpu_buckets is None:
            self._build_plan()

        # 1. GPU-resident params: direct on-device foreach update
        for gi, params in self._gpu_params:
            group = self.param_groups[gi]
            by_key: Dict[Tuple, List] = {}
            for p in params:
                if p.grad is None:
                    continue
                st = self._ensure_state(p)
                st["step"] += 1
                by_key.setdefault((p.device, p.dtype), []).append((p, st))
            for (_, _), pairs in by_key.items():
                self._adamw_math(
                    [p for p, _ in pairs],
                    [p.grad for p, _ in pairs],
                    [st["exp_avg"] for _, st in pairs],
                    [st["exp_avg_sq"] for _, st in pairs],
                    [st["step"] for _, st in pairs],
                    **self._hp(group),
                )

        # 2. CPU params: streamed through the GPU window (or CPU fallback)
        if self._cpu_buckets:
            if self._cuda:
                self._streamed_step()
            else:
                self._cpu_step()
        return loss

    # ── streamed CPU-param update ───────────────────────────────────────────
    @staticmethod
    def _flat(t: torch.Tensor) -> torch.Tensor:
        return t.view(-1) if t.is_contiguous() else t.reshape(-1)

    def _writeback(self, dst_flat: torch.Tensor, src_fp32: torch.Tensor,
                   raw: Optional[torch.Tensor], seed: int):
        """D2H one tensor: optional GPU-side (stochastic) down-convert first."""
        if raw is not None:
            if self._stochastic and raw.dtype == torch.bfloat16:
                copy_stochastic_(raw, src_fp32, seed)
            else:
                raw.copy_(src_fp32)
            dst_flat.copy_(raw, non_blocking=True)
        elif self._stochastic and dst_flat.dtype == torch.bfloat16:
            copy_stochastic_(dst_flat, src_fp32, seed)
        else:
            dst_flat.copy_(src_fp32, non_blocking=True)

    def _streamed_step(self):
        self._ensure_slots()
        h2d, comp, d2h = (self._h2d_stream, self._compute_stream,
                          self._d2h_stream)
        bi = 0
        for gi, items in self._cpu_buckets:
            group = self.param_groups[gi]
            active = []
            for p, off, n in items:
                if p.grad is None:
                    continue
                st = self._ensure_state(p)
                st["step"] += 1
                active.append((p, st, off, n))
            if not active:
                continue
            slot = self._slots[bi % len(self._slots)]
            bi += 1

            with torch.cuda.stream(h2d):
                if slot.used:
                    # previous occupant's writeback must have read the slot
                    h2d.wait_event(slot.d2h_done)
                for p, st, off, n in active:
                    slot.p[off:off + n].copy_(self._flat(p),
                                              non_blocking=True)
                    slot.g[off:off + n].copy_(self._flat(p.grad),
                                              non_blocking=True)
                    m, v = st["exp_avg"], st["exp_avg_sq"]
                    if slot.mr is not None:
                        slot.mr[off:off + n].copy_(self._flat(m),
                                                   non_blocking=True)
                        slot.vr[off:off + n].copy_(self._flat(v),
                                                   non_blocking=True)
                    else:
                        slot.m[off:off + n].copy_(self._flat(m),
                                                  non_blocking=True)
                        slot.v[off:off + n].copy_(self._flat(v),
                                                  non_blocking=True)
                slot.h2d_done.record()

            with torch.cuda.stream(comp):
                comp.wait_event(slot.h2d_done)
                end = active[-1][2] + active[-1][3]
                if slot.mr is not None:
                    # up-convert state to fp32 on the GPU
                    slot.m[:end].copy_(slot.mr[:end])
                    slot.v[:end].copy_(slot.vr[:end])
                ps = [slot.p[off:off + n] for _, _, off, n in active]
                gs = [slot.g[off:off + n] for _, _, off, n in active]
                ms = [slot.m[off:off + n] for _, _, off, n in active]
                vs = [slot.v[off:off + n] for _, _, off, n in active]
                steps = [st["step"] for _, st, _, _ in active]
                self._adamw_math(ps, gs, ms, vs, steps, **self._hp(group))
                slot.compute_done.record()

            with torch.cuda.stream(d2h):
                d2h.wait_event(slot.compute_done)
                for p, st, off, n in active:
                    s = st["step"] * 3
                    self._writeback(self._flat(p), slot.p[off:off + n],
                                    None, s + 1)
                    mr = (slot.mr[off:off + n]
                          if slot.mr is not None else None)
                    vr = (slot.vr[off:off + n]
                          if slot.vr is not None else None)
                    self._writeback(self._flat(st["exp_avg"]),
                                    slot.m[off:off + n], mr, s + 2)
                    self._writeback(self._flat(st["exp_avg_sq"]),
                                    slot.v[off:off + n], vr, s + 3)
                slot.d2h_done.record()
                slot.used = True

        # masters must be consistent before the caller (or the OffloadModel
        # loader) reads them from the host side
        self._d2h_stream.synchronize()

    # ── CPU fallback (no CUDA / device="cpu") ───────────────────────────────
    def _cpu_step(self):
        for gi, items in self._cpu_buckets:
            group = self.param_groups[gi]
            ps, gs, ms, vs, steps = [], [], [], [], []
            for p, _, _ in items:
                if p.grad is None:
                    continue
                if p.dtype != torch.float32:
                    raise NotImplementedError(
                        "OffloadAdamW CPU fallback supports fp32 params only"
                    )
                st = self._ensure_state(p)
                st["step"] += 1
                ps.append(p)
                gs.append(p.grad)
                ms.append(st["exp_avg"])
                vs.append(st["exp_avg_sq"])
                steps.append(st["step"])
            if ps:
                self._adamw_math(ps, gs, ms, vs, steps, **self._hp(group))
