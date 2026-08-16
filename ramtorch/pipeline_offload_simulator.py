"""
pipeline_offload_simulator.py
-----------------------------
Combined simulator: pipeline-parallel schedules (``schedule_simulator``) with
per-stage windowed CPU->GPU weight streaming (``offload_simulator``).

The idea being tested: each pipeline stage's model is pre-diced into ``L``
layer-chunks whose masters (weights/grads/optimizer state) live in CPU pinned
memory, exactly like ``OffloadModel``. Because the pipeline schedule fixes each
stage's op order, the chunk-touch order is FULLY deterministic and prefetch
can be scheduled exactly:

  * ``F -> F`` (warmup forwards):   chunks ``0..L-1, 0..L-1``   (inference ring)
  * ``F -> B`` / ``B -> F`` (1b1f): ``0..L-1, L-1..0, 0..L-1``  (echo — the
    turnaround chunk is already resident: free reuse)
  * ``B -> B`` (cooldown drain):    ``L-1..0, L-1..0``          (flipped ring)

Pipeline bubbles + small activation traffic leave each GPU's PCIe link mostly
idle; this simulator quantifies how much weight streaming that idle link can
hide, and where the H2D channel becomes the new critical path.

Model (one instance of the ``offload_simulator`` machinery PER STAGE):
  * Each stage owns three serial channels: gpu compute, H2D loads, D2H grad
    writebacks (each GPU has its own PCIe link).
  * Residency capacity is ``window`` streaming slots (+ pinned chunks).
    Eviction is farthest-next-use (Belady) over the stage's known itinerary;
    the prefetcher greedily loads the earliest non-resident chunk.
  * Compute walks the stage's chunk itinerary serially; the FIRST chunk of a
    stage-op additionally waits on the cross-stage dependency, exactly as in
    ``schedule_simulator``:
      - first chunk of ``F(s, mb)`` waits for the last chunk of
        ``F(s-1, mb)`` (+ ``comm``);
      - first chunk of ``B(s, mb)`` waits for the last chunk of
        ``B(s+1, mb)`` (+ ``comm``); on the last stage it waits on the
        stage's own ``F(s, mb)`` (the loss — same device, no comm).
  * Every backward chunk emits a grad writeback on the D2H channel.

Grad accumulation modes (``grad_mode=``):
  * ``"writeback"`` (current implementation): every backward chunk emits a
    per-microbatch grad packet — D2H copy (``td2h``) then, if ``tcpu`` > 0, a
    serial CPU accumulate (models ``grad_acc += staging``, the measured
    bottleneck on real hardware: ~100-150 ms per 800 MB packet).
  * ``"stream"``: the grad ACCUMULATOR is streamed like the weights. Zero-
    init on first touch (free, occupies an ``acc_slots`` residency slot), GPU
    add at the end of each B (free), H2D reload (``th2d``) when it fell out
    of residency, D2H writeback (``td2h``) to hand a slot back or to flush.
    The CPU does zero math — RAM is storage only. With ``acc_slots >= L``
    the accs never leave the GPU (one writeback per chunk per step); with
    small ``acc_slots`` it degrades to a load->add->writeback round trip per
    microbatch visit.

Activation offloading (``tact`` > 0, same model as ``offload_simulator``):
each stage's ``F(mb, chunk)`` saves one activation packet that the matching
``B(mb, chunk)`` consumes. Packets occupy ``act_slots`` GPU slots per stage (a
pool separate from the weight window). A packet can be offloaded to CPU RAM
(kind "O" on the stage's D2H channel) freeing its slot, and must be reloaded
(kind "R" on the stage's H2D channel) before its B. Eviction is
farthest-next-B-use; packets with a valid RAM copy re-drop for free.
``act_policy="eager"`` offloads at birth whenever D2H is idle; ``"lazy"``
only under slot pressure. This matters most for EARLY stages, which hold
activations for all in-flight microbatches across the pipeline bubble.

Simplifications (deliberate): boundary activation hops between stages cost
``comm`` latency but do NOT occupy the h2d/d2h channels (the premise: boundary
activations are small next to weights; the INTERNAL saved-tensor packets that
``tact`` models are the big ones); host DDR bandwidth shared across GPUs is
not modeled; the optimizer step is out of scope (same as
``offload_simulator``).

Usage:
    python -m ramtorch.pipeline_offload_simulator                  # W sweep
    python -m ramtorch.pipeline_offload_simulator --window 2
    python -m ramtorch.pipeline_offload_simulator --window 2 --plot out.png
    python -m ramtorch.pipeline_offload_simulator --schedule gpipe --th2d 3
    python -m ramtorch.pipeline_offload_simulator --window 2 --tact 0.3 \
        --act-slots 4                          # activation offload on
    # importable:
    from ramtorch.pipeline_offload_simulator import (
        expand_rank_ops, simulate_pipeline_offload, gantt, plot_gantt)
"""

from __future__ import annotations

import argparse
import bisect
from collections import deque
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

try:  # normal package import
    from .schedule_simulator import (
        gpipe_ops,
        onef1b_ops,
        staggered_1f1b_ops,
        simulate as simulate_schedule,
    )
    from .offload_simulator import evenly_pinned
except ImportError:  # run as a bare script (torch-less python): load siblings
    import importlib.util as _ilu
    import os as _os

    def _load_sibling(name: str):
        path = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                             name + ".py")
        spec = _ilu.spec_from_file_location(name, path)
        mod = _ilu.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    _ss = _load_sibling("schedule_simulator")
    _osim = _load_sibling("offload_simulator")
    gpipe_ops = _ss.gpipe_ops
    onef1b_ops = _ss.onef1b_ops
    staggered_1f1b_ops = _ss.staggered_1f1b_ops
    simulate_schedule = _ss.simulate
    evenly_pinned = _osim.evenly_pinned

__all__ = [
    "expand_rank_ops",
    "simulate_pipeline_offload",
    "PipelineOffloadResult",
    "gantt",
    "plot_gantt",
]

StageOp = Tuple[str, int]           # ("F"|"B"|"W", mb) from the schedule builders
ChunkOp = Tuple[str, int, int]      # ("F"|"B", mb, chunk)
Span = Tuple[float, float]

INF = float("inf")
_EPS = 1e-9


# ── Itinerary expansion ───────────────────────────────────────────────────────

def expand_rank_ops(
    rank_ops: Sequence[Sequence[StageOp]], chunks: int
) -> Tuple[List[List[ChunkOp]], List[Dict[Tuple[str, int], int]],
           List[Dict[Tuple[str, int], int]]]:
    """Expand per-stage schedule op lists into per-stage chunk itineraries.

    ``("F", mb)`` becomes chunk ops ``(F, mb, 0..L-1)``; ``("B", mb)`` becomes
    ``(B, mb, L-1..0)`` (backward traverses the stage's chunks in reverse).
    ``W`` markers are dropped (zero-cost bookkeeping in the schedule sim).

    Returns ``(itineraries, firsts, lasts)`` where ``firsts[s][(kind, mb)]`` /
    ``lasts[s][(kind, mb)]`` are the itinerary indices of the stage-op's first
    and last chunk op — the dependency attach points.
    """
    if chunks < 1:
        raise ValueError(f"chunks must be >= 1, got {chunks}")
    itins: List[List[ChunkOp]] = []
    firsts: List[Dict[Tuple[str, int], int]] = []
    lasts: List[Dict[Tuple[str, int], int]] = []
    for ops in rank_ops:
        itin: List[ChunkOp] = []
        first: Dict[Tuple[str, int], int] = {}
        last: Dict[Tuple[str, int], int] = {}
        for op in ops:
            kind, mb = op[0], op[1]
            if kind == "W":
                continue
            if kind not in ("F", "B"):
                raise ValueError(f"bad op kind {kind!r}")
            order = range(chunks) if kind == "F" else range(chunks - 1, -1, -1)
            first[(kind, mb)] = len(itin)
            for c in order:
                itin.append((kind, mb, c))
            last[(kind, mb)] = len(itin) - 1
        itins.append(itin)
        firsts.append(first)
        lasts.append(last)
    return itins, firsts, lasts


# ── Result ────────────────────────────────────────────────────────────────────

class PipelineOffloadResult:
    """Timeline + metrics from one ``simulate_pipeline_offload`` run.

    ``spans[s]`` maps channel -> ordered ``(op, (start, end))``:
      * ``"gpu"``: op = ``(kind, mb, chunk)`` with kind "F"/"B"
      * ``"h2d"``: op = ``("L", chunk)`` weight load, ``("A", chunk)``
        grad-accumulator reload (stream mode), or ``("R", mb, chunk)``
        activation reload (``tact`` > 0)
      * ``"d2h"``: op = ``("G", chunk)`` grad / accumulator writeback or
        ``("O", mb, chunk)`` activation offload (``tact`` > 0)
      * ``"cpu"``: op = ``("C", chunk)`` CPU accumulate (writeback mode
        with ``tcpu`` > 0)
    """

    CHANNELS = ("gpu", "h2d", "d2h", "cpu")

    def __init__(
        self,
        p: int,
        m: int,
        chunks: int,
        window: int,
        tf: float,
        tb: float,
        th2d: float,
        td2h: float,
        comm: float,
        pinned: FrozenSet[int],
        itins: List[List[ChunkOp]],
        spans: List[Dict[str, List[Tuple[Tuple, Span]]]],
        n_loads: List[int],
        peak_resident: List[int],
        ref_makespan: float,
        *,
        grad_mode: str = "writeback",
        tcpu: float = 0.0,
        acc_slots: int = 0,
        n_acc_loads: Optional[List[int]] = None,
        n_acc_wb: Optional[List[int]] = None,
        peak_acc: Optional[List[int]] = None,
        tact: float = 0.0,
        act_slots: Optional[List[int]] = None,
        act_policy: str = "eager",
        act_timelines: Optional[List[List[Tuple[float, int]]]] = None,
        n_act_offloads: Optional[List[int]] = None,
        n_act_reloads: Optional[List[int]] = None,
        baseline_peak_act: Optional[List[int]] = None,
    ):
        self.p, self.m, self.chunks, self.window = p, m, chunks, window
        self.tf, self.tb, self.th2d, self.td2h, self.comm = tf, tb, th2d, td2h, comm
        self.pinned = frozenset(pinned)
        self.itins = itins
        self.spans = spans
        self.n_loads = n_loads
        self.peak_resident = peak_resident
        self.ref_makespan = ref_makespan
        self.grad_mode = grad_mode
        self.tcpu = tcpu
        self.acc_slots = acc_slots
        self.n_acc_loads = n_acc_loads or [0] * p
        self.n_acc_wb = n_acc_wb or [0] * p
        self.peak_acc = peak_acc or [0] * p
        self.tact = tact
        self.act_slots = act_slots or [0] * p
        self.act_policy = act_policy
        self.act_timelines = act_timelines or [[] for _ in range(p)]
        self.n_act_offloads = n_act_offloads or [0] * p
        self.n_act_reloads = n_act_reloads or [0] * p
        self.baseline_peak_act = baseline_peak_act or [0] * p
        # hide the cpu row unless something actually ran there
        if not any(st.get("cpu") for st in spans):
            self.CHANNELS = ("gpu", "h2d", "d2h")
        self.makespan = max(
            (en for st_spans in spans for ch in st_spans.values()
             for _, (_, en) in ch),
            default=0.0,
        )

    # -- per-stage metrics --
    def busy(self, s: int, ch: str) -> float:
        return sum(en - st for _, (st, en) in self.spans[s][ch])

    def util(self, s: int, ch: str) -> float:
        return 0.0 if self.makespan == 0 else self.busy(s, ch) / self.makespan

    def naive_loads(self, s: int) -> int:
        """Loads if nothing were ever retained: one per compute op on an
        unpinned chunk (= ``2*m*chunks`` per stage at pin=0)."""
        return sum(1 for _, _, c in self.itins[s] if c not in self.pinned)

    @property
    def total_mem(self) -> int:
        """GPU weight slots per stage: streaming window + pinned chunks."""
        return self.window + len(self.pinned)

    @property
    def act_on(self) -> bool:
        return self.tact > 0 and any(self.baseline_peak_act)

    def peak_act(self, s: int) -> int:
        """Peak GPU activation slots occupied on stage ``s`` (0 when off)."""
        return max((occ for _, occ in self.act_timelines[s]), default=0)

    @property
    def overhead_vs_ref(self) -> float:
        """(makespan - no-offload pipeline makespan) / no-offload makespan."""
        if self.ref_makespan == 0:
            return 0.0
        return (self.makespan - self.ref_makespan) / self.ref_makespan

    def metrics(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "p": self.p,
            "m": self.m,
            "chunks": self.chunks,
            "window": self.window,
            "pinned": sorted(self.pinned),
            "total_mem": self.total_mem,
            "makespan": self.makespan,
            "ref_makespan": self.ref_makespan,
            "overhead_vs_ref": self.overhead_vs_ref,
            "grad_mode": self.grad_mode,
            "tcpu": self.tcpu,
            "acc_slots": self.acc_slots,
            "gpu_util": [self.util(s, "gpu") for s in range(self.p)],
            "h2d_util": [self.util(s, "h2d") for s in range(self.p)],
            "d2h_util": [self.util(s, "d2h") for s in range(self.p)],
            "cpu_util": [self.util(s, "cpu") if "cpu" in self.CHANNELS
                         else 0.0 for s in range(self.p)],
            "loads": list(self.n_loads),
            "naive_loads": [self.naive_loads(s) for s in range(self.p)],
            "acc_loads": list(self.n_acc_loads),
            "acc_writebacks": list(self.n_acc_wb),
            "peak_resident": list(self.peak_resident),
            "peak_acc": list(self.peak_acc),
        }
        if self.act_on:
            out.update({
                "tact": self.tact,
                "act_slots": list(self.act_slots),
                "act_policy": self.act_policy,
                "peak_act": [self.peak_act(s) for s in range(self.p)],
                "baseline_peak_act": list(self.baseline_peak_act),
                "n_act_offloads": list(self.n_act_offloads),
                "n_act_reloads": list(self.n_act_reloads),
            })
        return out


# ── Simulator ─────────────────────────────────────────────────────────────────

def simulate_pipeline_offload(
    rank_ops: Sequence[Sequence[StageOp]],
    chunks: int,
    window: int,
    tf: float = 1.0,
    tb: float = 2.0,
    th2d: float = 0.5,
    td2h: float = 0.5,
    comm: float = 0.0,
    pinned: Optional[FrozenSet[int]] = None,
    grad_mode: str = "writeback",
    tcpu: float = 0.0,
    acc_slots: Optional[int] = None,
    tact: float = 0.0,
    act_slots: Optional[int] = None,
    act_policy: str = "eager",
) -> PipelineOffloadResult:
    """Discrete-event simulation: p offload state machines + cross-stage deps.

    ``tf``/``tb``/``th2d``/``td2h`` are PER-CHUNK costs (a stage-level forward
    costs ``chunks * tf``). ``pinned`` chunk indices (same set on every stage)
    are permanently GPU-resident on top of the ``window`` streaming slots.

    ``grad_mode="writeback"``: every B chunk emits a per-microbatch grad
    packet: D2H copy (``td2h``), then a serial CPU accumulate (``tcpu``, 0
    disables — the pre-fix simulator behavior). ``grad_mode="stream"``: the
    grad accumulator is streamed like the weights through ``acc_slots``
    residency slots (default = ``window``); zero-init first touch is free,
    reloads cost ``th2d`` on the H2D channel, evictions/flushes cost ``td2h``
    on the D2H channel, and the CPU does nothing. Pinned chunks accumulate
    on-GPU in both modes (no grad traffic), matching ``OffloadModel``.

    ``tact`` > 0 enables ACTIVATION offloading (0 = feature off, behavior
    byte-identical to before): per stage, ``F(mb, chunk)`` creates one packet
    that ``B(mb, chunk)`` consumes, occupying one of ``act_slots`` per-stage
    GPU slots (default: unlimited). Offloads ("O", cost ``tact``) ride the
    stage's D2H channel — needed offloads outrank grad/acc writebacks;
    reloads ("R", cost ``tact``) ride the stage's H2D channel, prefetched in
    itinerary order after weight/acc needs. ``act_policy`` as in
    ``offload_simulator``.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    if grad_mode not in ("writeback", "stream"):
        raise ValueError(f"grad_mode must be writeback|stream, got {grad_mode!r}")
    if act_policy not in ("eager", "lazy"):
        raise ValueError(f"act_policy must be eager|lazy, got {act_policy!r}")
    if act_slots is not None and act_slots < 1:
        raise ValueError(f"act_slots must be >= 1, got {act_slots}")
    pinned = frozenset(pinned or ())
    for c in pinned:
        if not (0 <= c < chunks):
            raise ValueError(f"pinned chunk {c} out of range [0, {chunks})")
    acc_slots = window if acc_slots is None else acc_slots
    if grad_mode == "stream" and acc_slots < 1:
        raise ValueError(f"acc_slots must be >= 1, got {acc_slots}")

    p = len(rank_ops)
    itins, firsts, lasts = expand_rank_ops(rank_ops, chunks)
    m = 1 + max((mb for itin in itins for _, mb, _ in itin), default=-1)
    n_ops = [len(itin) for itin in itins]
    capacity = window + len(pinned)
    stream = grad_mode == "stream" and td2h > 0

    # chunk -> itinerary positions, per stage (Belady next-use queries)
    uses: List[List[List[int]]] = [
        [[] for _ in range(chunks)] for _ in range(p)
    ]
    acc_uses: List[List[List[int]]] = [
        [[] for _ in range(chunks)] for _ in range(p)
    ]
    for s, itin in enumerate(itins):
        for i, (k, _, c) in enumerate(itin):
            uses[s][c].append(i)
            if k == "B":
                acc_uses[s][c].append(i)

    def next_use(s: int, chunk: int, from_pos: int) -> float:
        us = uses[s][chunk]
        j = bisect.bisect_left(us, from_pos)
        return us[j] if j < len(us) else INF

    def next_acc_use(s: int, chunk: int, from_pos: int) -> float:
        us = acc_uses[s][chunk]
        j = bisect.bisect_left(us, from_pos)
        return us[j] if j < len(us) else INF

    # ── activation machinery (tact > 0) ─────────────────────────────────────
    # per stage: packet key (mb, chunk) -> the position of its single B
    act_bpos: List[Dict[Tuple[int, int], int]] = [{} for _ in range(p)]
    for s, itin in enumerate(itins):
        for i, (k, mb, c) in enumerate(itin):
            if k == "B":
                act_bpos[s][(mb, c)] = i
    act_on = tact > 0 and any(act_bpos)
    # per stage baseline peak = concurrently-alive packets with NO offloading
    baseline_peak_act = [0] * p
    for s, itin in enumerate(itins):
        alive: set = set()
        for i, (k, mb, c) in enumerate(itin):
            if k == "F" and act_bpos[s].get((mb, c), -1) > i:
                alive.add((mb, c))
                baseline_peak_act[s] = max(baseline_peak_act[s], len(alive))
            elif k == "B":
                alive.discard((mb, c))
    n_act_slots = [
        baseline_peak_act[s] if act_slots is None else act_slots
        for s in range(p)
    ]

    def next_act_use(s: int, key: Tuple[int, int], from_pos: int) -> float:
        bp = act_bpos[s].get(key)
        return bp if bp is not None and bp >= from_pos else INF

    # per-stage channel state
    resident: List[set] = [set(pinned) for _ in range(p)]
    comp_idx = [0] * p
    # gpu_cur[s] = (itin_index, start); h2d_cur[s] = (kind, payload, start)
    # with kind "L" weight load / "A" acc reload (payload = chunk) or "R"
    # act reload (payload = (mb, chunk)); d2h_cur[s] = (kind, payload, start)
    # with kind "G" grad/acc writeback (chunk) or "O" act offload ((mb,
    # chunk)); cpu_cur[s] = (chunk, start)
    gpu_cur: List[Optional[Tuple[int, float]]] = [None] * p
    gpu_end = [0.0] * p
    h2d_cur: List[Optional[Tuple[str, object, float]]] = [None] * p
    h2d_end = [0.0] * p
    d2h_cur: List[Optional[Tuple[str, object, float]]] = [None] * p
    d2h_end = [0.0] * p
    d2h_q: List[deque] = [deque() for _ in range(p)]  # (chunk, ready_time)
    cpu_cur: List[Optional[Tuple[int, float]]] = [None] * p
    cpu_end = [0.0] * p
    cpu_q: List[deque] = [deque() for _ in range(p)]  # (chunk, ready_time)

    # stream-mode grad accumulator residency (streamed chunks only)
    acc_res: List[set] = [set() for _ in range(p)]
    acc_dirty: List[set] = [set() for _ in range(p)]
    acc_touched: List[set] = [set() for _ in range(p)]

    # activation packet state per stage: key (mb, chunk) -> "gpu" (resident
    # dirty) | "gpu_clean" (resident + valid RAM copy) | "offloading" |
    # "reloading" | "ram"; absent = no live packet
    act_state: List[Dict[Tuple[int, int], str]] = [{} for _ in range(p)]
    act_timelines: List[List[Tuple[float, int]]] = [
        [(0.0, 0)] for _ in range(p)
    ]
    n_act_off = [0] * p
    n_act_rel = [0] * p

    spans: List[Dict[str, List[Tuple[Tuple, Span]]]] = [
        {"gpu": [], "h2d": [], "d2h": [], "cpu": []} for _ in range(p)
    ]
    n_loads = [0] * p
    n_acc_loads = [0] * p
    n_acc_wb = [0] * p
    peak_res = [len(pinned) for _ in range(p)]
    peak_acc = [0] * p
    # (stage, kind, mb) -> end time of that stage-op's LAST chunk
    done: Dict[Tuple[int, str, int], float] = {}
    now = 0.0

    def running_b_chunk(s: int) -> Optional[int]:
        """Chunk of the B op currently on the GPU (its acc is in use)."""
        if gpu_cur[s] is None:
            return None
        k, _, c = itins[s][gpu_cur[s][0]]
        return c if k == "B" else None

    def acc_make_room(s: int, need_pos: int) -> bool:
        """Free an acc slot for a need at ``need_pos``: True if one is free
        or a clean, idle acc with a farther next B use was evicted (its
        value is safe in RAM — a later reload brings it back)."""
        if len(acc_res[s]) < acc_slots:
            return True
        in_use = running_b_chunk(s)
        cleaning = (d2h_cur[s][1] if d2h_cur[s] is not None
                    and d2h_cur[s][0] == "G" else None)
        victim, victim_nu = None, float(need_pos)
        for r in acc_res[s]:
            if r == in_use or r == cleaning or r in acc_dirty[s]:
                continue
            nu = next_acc_use(s, r, comp_idx[s])
            if nu > victim_nu:
                victim, victim_nu = r, nu
        if victim is None:
            return False
        acc_res[s].discard(victim)
        return True

    def act_occupied(s: int) -> int:
        return sum(1 for st in act_state[s].values() if st != "ram")

    def record_act(s: int, t: float) -> None:
        act_timelines[s].append((t, act_occupied(s)))

    def act_in_use(s: int) -> Optional[Tuple[int, int]]:
        """Packet key the current gpu op is writing (F) or reading (B)."""
        if gpu_cur[s] is None:
            return None
        _, mb, c = itins[s][gpu_cur[s][0]]
        return (mb, c)

    def act_make_room(s: int, need_pos: int) -> bool:
        """A slot is free, or a CLEAN packet (valid RAM copy) with next use
        beyond ``need_pos`` can be dropped for free."""
        if act_occupied(s) < n_act_slots[s]:
            return True
        busy = act_in_use(s)
        victim, victim_nu = None, float(need_pos)
        for key, st in act_state[s].items():
            if st != "gpu_clean" or key == busy:
                continue
            nu = next_act_use(s, key, comp_idx[s])
            if nu > victim_nu:
                victim, victim_nu = key, nu
        if victim is None:
            return False
        act_state[s][victim] = "ram"  # free drop: RAM copy already valid
        record_act(s, now)
        return True

    def act_offload_victim(
        s: int, need_pos: float
    ) -> Optional[Tuple[int, int]]:
        """Farthest-next-B-use DIRTY resident packet beyond ``need_pos``."""
        busy = act_in_use(s)
        victim, victim_nu = None, float(need_pos)
        for key, st in act_state[s].items():
            if st != "gpu" or key == busy:
                continue
            nu = next_act_use(s, key, comp_idx[s])
            if nu > victim_nu:
                victim, victim_nu = key, nu
        return victim

    def act_need_pos(s: int) -> Optional[int]:
        """Itinerary position of the earliest op TRULY blocked on an act
        slot: a B whose packet sits in RAM, or an F that must create a
        packet after accounting for slots that intervening Bs free."""
        free = n_act_slots[s] - act_occupied(s)
        if free > 0:
            return None
        if gpu_cur[s] is not None:
            rk, rmb, rc = itins[s][gpu_cur[s][0]]
            rst = act_state[s].get((rmb, rc))
            if rk == "B" and rst is not None and rst != "ram":
                free += 1  # the running B frees its slot at its end
        for pos in range(comp_idx[s], n_ops[s]):
            k, mb, c = itins[s][pos]
            key = (mb, c)
            st = act_state[s].get(key)
            if k == "B":
                if st == "ram":
                    return pos  # its reload needs a slot
                if st is not None:
                    free += 1   # on-GPU packet frees its slot at B end
            elif (key not in act_state[s]
                    and act_bpos[s].get(key, -1) > pos):
                if free <= 0:
                    return pos
                free -= 1
        return None

    def dep_ready(s: int, i: int) -> Optional[float]:
        """Earliest time the cross-stage dependency of itinerary op ``i`` on
        stage ``s`` allows a start; None if the upstream op hasn't finished."""
        kind, mb, _ = itins[s][i]
        if firsts[s][(kind, mb)] != i:
            return 0.0  # non-first chunk: only in-stage serial order applies
        if kind == "F":
            if s == 0:
                return 0.0
            e = done.get((s - 1, "F", mb))
            return None if e is None else e + comm
        if s == p - 1:  # loss: waits on the stage's own forward, same device
            e = done.get((s, "F", mb))
            return None if e is None else e
        e = done.get((s + 1, "B", mb))
        return None if e is None else e + comm

    while True:
        # 1) completions due at `now`
        for s in range(p):
            if gpu_cur[s] is not None and gpu_end[s] <= now + _EPS:
                i, st = gpu_cur[s]
                kind, mb, c = itins[s][i]
                spans[s]["gpu"].append(((kind, mb, c), (st, gpu_end[s])))
                if lasts[s][(kind, mb)] == i:
                    done[(s, kind, mb)] = gpu_end[s]
                if kind == "B" and td2h > 0 and c not in pinned:
                    if stream:
                        acc_dirty[s].add(c)  # GPU add landed in the acc
                    else:
                        d2h_q[s].append((c, gpu_end[s]))
                if kind == "B" and act_on and (mb, c) in act_state[s]:
                    del act_state[s][(mb, c)]  # packet consumed, slot freed
                    record_act(s, now)
                gpu_cur[s] = None
            if h2d_cur[s] is not None and h2d_end[s] <= now + _EPS:
                lk, c, st = h2d_cur[s]
                if lk == "R":
                    spans[s]["h2d"].append(((lk,) + c, (st, h2d_end[s])))
                    act_state[s][c] = "gpu_clean"  # RAM copy stays valid
                    record_act(s, now)
                else:
                    spans[s]["h2d"].append(((lk, c), (st, h2d_end[s])))
                    if lk == "L":
                        resident[s].add(c)
                    else:  # "A": acc reload landed
                        acc_res[s].add(c)
                        peak_acc[s] = max(peak_acc[s], len(acc_res[s]))
                h2d_cur[s] = None
            if d2h_cur[s] is not None and d2h_end[s] <= now + _EPS:
                dk, c, st = d2h_cur[s]
                if dk == "O":
                    spans[s]["d2h"].append(((dk,) + c, (st, d2h_end[s])))
                    act_state[s][c] = "ram"  # slot freed
                    record_act(s, now)
                else:
                    spans[s]["d2h"].append((("G", c), (st, d2h_end[s])))
                    if stream:
                        # value safe in RAM; stays resident
                        acc_dirty[s].discard(c)
                    elif tcpu > 0:
                        cpu_q[s].append((c, d2h_end[s]))
                d2h_cur[s] = None
            if cpu_cur[s] is not None and cpu_end[s] <= now + _EPS:
                c, st = cpu_cur[s]
                spans[s]["cpu"].append((("C", c), (st, cpu_end[s])))
                cpu_cur[s] = None

        # 2) start whatever can start at `now` (fixed point for zero-cost ops)
        progressed = False
        for s in range(p):
            # compute
            if gpu_cur[s] is None and comp_idx[s] < n_ops[s]:
                i = comp_idx[s]
                kind, mb, c = itins[s][i]
                dep = dep_ready(s, i)
                ok = (c in resident[s] and dep is not None
                      and dep <= now + _EPS)
                if ok and stream and kind == "B" and c not in pinned:
                    # the GPU add at the end of B needs the acc on device and
                    # not mid-writeback (the copy must not race the add)
                    if (d2h_cur[s] is not None
                            and d2h_cur[s][:2] == ("G", c)):
                        ok = False
                    elif c in acc_res[s]:
                        pass
                    elif c not in acc_touched[s] and acc_make_room(s, i):
                        acc_res[s].add(c)  # zero-init in place: no transfer
                        acc_touched[s].add(c)
                        peak_acc[s] = max(peak_acc[s], len(acc_res[s]))
                    else:
                        ok = False  # wait for the acc reload / a free slot
                makes_packet = False
                if ok and act_on:
                    key = (mb, c)
                    if kind == "F":
                        if act_bpos[s].get(key, -1) > i:
                            makes_packet = act_make_room(s, i)
                            ok = makes_packet
                    else:  # B needs its packet on the GPU (if one was made)
                        st_a = act_state[s].get(key)
                        ok = st_a is None or st_a in ("gpu", "gpu_clean")
                if ok:
                    gpu_cur[s] = (i, now)
                    gpu_end[s] = now + (tf if kind == "F" else tb)
                    comp_idx[s] += 1
                    if makes_packet:
                        act_state[s][(mb, c)] = "gpu"
                        record_act(s, now)
                    progressed = True

            # cpu accumulate (writeback mode)
            if cpu_cur[s] is None and cpu_q[s] and cpu_q[s][0][1] <= now + _EPS:
                c, _ = cpu_q[s].popleft()
                cpu_cur[s] = (c, now)
                cpu_end[s] = now + tcpu
                progressed = True

            # d2h: a PRESSURE act offload (an op is blocked on an act slot)
            # outranks grads; grads outrank EAGER act offloads
            if act_on and d2h_cur[s] is None:
                need = act_need_pos(s)
                if need is not None:
                    v = act_offload_victim(s, need)
                    if v is not None:
                        act_state[s][v] = "offloading"
                        d2h_cur[s] = ("O", v, now)
                        d2h_end[s] = now + tact
                        n_act_off[s] += 1
                        progressed = True
            # per-microbatch grad packets (writeback mode) ...
            if d2h_cur[s] is None and d2h_q[s] and d2h_q[s][0][1] <= now + _EPS:
                c, _ = d2h_q[s].popleft()
                d2h_cur[s] = ("G", c, now)
                d2h_end[s] = now + td2h
                progressed = True
            # ... or acc writebacks (stream mode): finals (past their last B)
            # must go out anyway; others only under slot pressure, farthest
            # next use first (the acc stays resident and turns clean)
            elif d2h_cur[s] is None and stream and acc_dirty[s]:
                in_use = running_b_chunk(s)
                best, best_nu = None, -1.0
                for c in acc_dirty[s]:
                    if c == in_use:
                        continue
                    nu = next_acc_use(s, c, comp_idx[s])
                    if nu == INF:
                        best, best_nu = c, INF
                        break
                    if len(acc_res[s]) >= acc_slots and nu > best_nu:
                        best, best_nu = c, nu
                if best is not None:
                    d2h_cur[s] = ("G", best, now)
                    d2h_end[s] = now + td2h
                    n_acc_wb[s] += 1
                    progressed = True
            if act_on and d2h_cur[s] is None and act_policy == "eager":
                # never offload a packet whose B is the current or next op
                v = act_offload_victim(s, comp_idx[s] + 1)
                if v is not None:
                    act_state[s][v] = "offloading"
                    d2h_cur[s] = ("O", v, now)
                    d2h_end[s] = now + tact
                    n_act_off[s] += 1
                    progressed = True

            # prefetch: earliest unmet need on this stage — a weight load
            # (any op), an evicted acc's reload (stream mode, B ops), or an
            # offloaded activation's reload (tact > 0, B ops)
            if h2d_cur[s] is None:
                for pos in range(comp_idx[s], n_ops[s]):
                    kind, mb, c = itins[s][pos]
                    if c not in resident[s]:
                        can_start = len(resident[s]) < capacity
                        if not can_start:
                            # Belady: evict the farthest-next-use resident
                            # chunk, never the one compute is running, never
                            # pinned, and only if farther than the need
                            in_use = (
                                itins[s][gpu_cur[s][0]][2]
                                if gpu_cur[s] is not None else None
                            )
                            victim, victim_nu = None, -1.0
                            for r in resident[s]:
                                if r == in_use or r in pinned:
                                    continue
                                nu = next_use(s, r, comp_idx[s])
                                if nu > victim_nu:
                                    victim, victim_nu = r, nu
                            if victim is not None and victim_nu > pos:
                                resident[s].discard(victim)
                                can_start = True
                        if can_start:
                            h2d_cur[s] = ("L", c, now)
                            h2d_end[s] = now + th2d
                            n_loads[s] += 1
                            peak_res[s] = max(peak_res[s],
                                              len(resident[s]) + 1)
                            progressed = True
                        break  # first unmet weight need wins (or waits)
                    if (stream and kind == "B" and c not in pinned
                            and c in acc_touched[s]
                            and c not in acc_res[s]):
                        if acc_make_room(s, pos):
                            h2d_cur[s] = ("A", c, now)
                            h2d_end[s] = now + th2d
                            n_acc_loads[s] += 1
                            progressed = True
                            break
                        # acc slot-blocked: fall through, keep scanning
                    if (act_on and kind == "B"
                            and act_state[s].get((mb, c)) == "ram"):
                        if act_make_room(s, pos):
                            act_state[s][(mb, c)] = "reloading"
                            h2d_cur[s] = ("R", (mb, c), now)
                            h2d_end[s] = now + tact
                            n_act_rel[s] += 1
                            record_act(s, now)
                            progressed = True
                            break
                        # act slot-blocked: scan on for other work

        if progressed:
            continue

        # 3) done, or advance time to the next event
        work_left = any(
            comp_idx[s] < n_ops[s]
            or gpu_cur[s] is not None
            or h2d_cur[s] is not None
            or d2h_cur[s] is not None
            or cpu_cur[s] is not None
            or d2h_q[s]
            or cpu_q[s]
            or acc_dirty[s]
            for s in range(p)
        )
        if not work_left:
            break
        pending: List[float] = []
        for s in range(p):
            for cur, end in ((gpu_cur[s], gpu_end[s]),
                             (h2d_cur[s], h2d_end[s]),
                             (d2h_cur[s], d2h_end[s]),
                             (cpu_cur[s], cpu_end[s])):
                if cur is not None:
                    pending.append(end)
            # a stage whose next op waits only on a known future dep time
            # (comm latency) must wake at that time, not a channel end
            if gpu_cur[s] is None and comp_idx[s] < n_ops[s]:
                dep = dep_ready(s, comp_idx[s])
                if dep is not None and dep > now + _EPS:
                    pending.append(dep)
        if not pending:
            state = {
                s: dict(comp=f"{comp_idx[s]}/{n_ops[s]}",
                        resident=sorted(resident[s]),
                        act=dict(act_state[s]))
                for s in range(p)
            }
            raise RuntimeError(
                f"pipeline-offload schedule deadlocked at t={now}: {state} — bug"
            )
        now = min(pending)

    ref = simulate_schedule(
        rank_ops, tf=chunks * tf, tb=chunks * tb, comm=comm
    ).makespan
    return PipelineOffloadResult(
        p, m, chunks, window, tf, tb, th2d, td2h, comm, pinned,
        itins, spans, n_loads, peak_res, ref,
        grad_mode=grad_mode, tcpu=tcpu, acc_slots=acc_slots,
        n_acc_loads=n_acc_loads, n_acc_wb=n_acc_wb, peak_acc=peak_acc,
        tact=tact,
        act_slots=n_act_slots if act_on else [0] * p,
        act_policy=act_policy, act_timelines=act_timelines,
        n_act_offloads=n_act_off, n_act_reloads=n_act_rel,
        baseline_peak_act=baseline_peak_act if act_on else [0] * p,
    )


# ── ASCII Gantt ───────────────────────────────────────────────────────────────

_ROW_CHAR = {"F": "F", "B": "b", "L": "L", "G": "G", "A": "A", "C": "C",
             "O": "O", "R": "R"}


def gantt(res: PipelineOffloadResult, width: int = 120) -> str:
    """Three rows per stage (gpu / h2d / d2h); '.' = idle."""
    ms = res.makespan
    if ms <= 0:
        return "(empty)"
    cols = max(10, width)
    lines = []
    for s in range(res.p):
        for ch in res.CHANNELS:
            row = ["."] * cols
            for op, (st, en) in res.spans[s][ch]:
                a = int(st / ms * cols)
                b = max(a + 1, int(en / ms * cols))
                for col in range(a, min(b, cols)):
                    row[col] = _ROW_CHAR[op[0]]
            lines.append(f"  s{s} {ch:>3} |{''.join(row)}|")
    scale = "".join(
        str(int(t / 10 % 10)) if t % 10 == 0 else ("|" if t % 5 == 0 else " ")
        for t in range(cols)
    )
    lines.append(f"         +{scale}+  0..{ms:.1f}")
    return "\n".join(lines)


# ── Matplotlib Gantt ──────────────────────────────────────────────────────────

def plot_gantt(
    results: List[Tuple[str, PipelineOffloadResult]],
    out_path: Optional[str] = None,
    show: bool = True,
    scale: float = 1.0,
    dpi: int = 130,
):
    """One subplot per result; 3 rows per stage (gpu / h2d / d2h).

    Compute bars are labeled ``F{mb}c{chunk}`` / ``B{mb}c{chunk}``, loads
    ``L{chunk}``, grad writebacks ``G{chunk}``. Idle gaps on the gpu rows are
    pipeline bubble + offload stall; idle on h2d rows is spare PCIe headroom.

    ``scale`` multiplies the figure dimensions (fonts keep their point
    size, so bars/labels get more room — use ~1.5-2 for busy schedules);
    ``dpi`` controls the saved image resolution.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {"F": "#0072B2", "B": "#E69F00", "L": "#009E73", "G": "#CC79A7",
              "A": "#56B4E9", "C": "#D55E00",
              "O": "#56B4E9", "R": "#F0E442"}
    hatches = {"F": "", "B": "//", "L": "", "G": "", "A": "//", "C": "",
               "O": "\\\\", "R": "\\\\"}
    kind_name = {"F": "forward", "B": "backward", "L": "H2D weight load",
                 "G": "D2H grad/acc writeback", "A": "H2D acc reload",
                 "C": "CPU accumulate", "O": "act offload (D2H)",
                 "R": "act reload (H2D)"}

    n = len(results)
    max_ms = max((r.makespan for _, r in results), default=1.0)
    heights = [3 * r.p * 0.42 + 0.9 for _, r in results]
    fig, axes = plt.subplots(
        n, 1,
        figsize=(scale * min(48, max(12, max_ms * 0.22)),
                 scale * sum(heights)),
        sharex=True, squeeze=False,
        gridspec_kw={"height_ratios": heights},
    )

    for ax, (name, res) in zip(axes[:, 0], results):
        nrows = 3 * res.p
        yticks, ylabels = [], []
        for s in range(res.p):
            for ci, ch in enumerate(res.CHANNELS):
                y = nrows - 1 - (3 * s + ci)  # stage 0 gpu on top
                yticks.append(y)
                ylabels.append(f"s{s} {ch}")
                for op, (st, en) in res.spans[s][ch]:
                    kind = op[0]
                    ax.barh(y, en - st, left=st, height=0.72,
                            color=colors[kind], hatch=hatches[kind],
                            edgecolor="white", linewidth=0.3)
                    if en - st >= res.makespan * 0.004:
                        label = (f"{kind}{op[1]}c{op[2]}" if len(op) == 3
                                 else f"{kind}{op[1]}")
                        bbox = (
                            dict(facecolor=colors[kind], edgecolor="none",
                                 pad=0.4)
                            if hatches[kind] else None
                        )
                        txt = "black" if kind == "R" else "white"
                        ax.text((st + en) / 2, y, label, ha="center",
                                va="center", fontsize=5.5, color=txt,
                                weight="bold", bbox=bbox)
            if s < res.p - 1:  # separator between stages
                ax.axhline(nrows - 1 - (3 * s + 2) - 0.5, color="#999",
                           linewidth=0.8, linestyle=":")
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_ylim(-0.6, nrows - 0.4)
        ax.grid(axis="x", alpha=0.25)
        mt = res.metrics()
        ax.set_title(
            f"{name}   makespan={res.makespan:.1f}"
            f"  vs_no-offload={res.ref_makespan:.1f}"
            f" (+{100 * res.overhead_vs_ref:.0f}%)"
            f"  loads={sum(res.n_loads)}/{sum(mt['naive_loads'])} naive"
            f"  W={res.window}+{len(res.pinned)}pin"
            f"  peak_res={max(res.peak_resident)}"
            + (f"  act={max(res.peak_act(s) for s in range(res.p))}"
               f"/{max(res.act_slots)} slots"
               f" (base {max(res.baseline_peak_act)},"
               f" O={sum(res.n_act_offloads)} R={sum(res.n_act_reloads)})"
               if res.act_on else ""),
            loc="left", fontsize=10,
        )
    axes[-1, 0].set_xlabel("time")
    handles = [
        Patch(facecolor=c, hatch=hatches[k], edgecolor="white",
              label=kind_name[k])
        for k, c in colors.items()
    ]
    axes[0, 0].legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle(
        "Pipeline schedule + per-stage windowed weight offload", y=0.998
    )
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"wrote {out_path}")
    if show:
        plt.show()
    return fig


# ── CLI ───────────────────────────────────────────────────────────────────────

_BUILDERS = {
    "1b1f": staggered_1f1b_ops,
    "gpipe": gpipe_ops,
    "1f1b": onef1b_ops,
}


def report(name: str, res: PipelineOffloadResult, width: int = 120) -> str:
    mt = res.metrics()
    out = [
        f"  {name:<16} makespan={res.makespan:7.1f}"
        f"  no-offload={res.ref_makespan:7.1f}"
        f"  overhead={100 * res.overhead_vs_ref:5.1f}%"
        f"  mem={res.total_mem}/{res.chunks} chunks/GPU"
    ]
    for s in range(res.p):
        line = (
            f"    s{s}: gpu={100 * res.util(s, 'gpu'):3.0f}%"
            f"  h2d={100 * res.util(s, 'h2d'):3.0f}%"
            f"  d2h={100 * res.util(s, 'd2h'):3.0f}%"
        )
        if "cpu" in res.CHANNELS:
            line += f"  cpu={100 * res.util(s, 'cpu'):3.0f}%"
        line += (
            f"  loads={res.n_loads[s]:4d}/{mt['naive_loads'][s]:4d} naive"
            f"  peak_res={res.peak_resident[s]}"
        )
        if res.grad_mode == "stream":
            line += (
                f"  accL={res.n_acc_loads[s]:3d}"
                f"  accW={res.n_acc_wb[s]:3d}"
                f"  peak_acc={res.peak_acc[s]}"
            )
        if res.act_on:
            line += (
                f"  actO={res.n_act_offloads[s]:3d}"
                f"  actR={res.n_act_reloads[s]:3d}"
                f"  peak_act={res.peak_act(s)}/{res.act_slots[s]}"
                f" (base {res.baseline_peak_act[s]})"
            )
        out.append(line)
    out.append(gantt(res, width))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p", type=int, default=4, help="pipeline stages (GPUs)")
    ap.add_argument("--m", type=int, default=8, help="microbatches")
    ap.add_argument("--chunks", type=int, default=8,
                    help="layer chunks per stage (L)")
    ap.add_argument("--window", type=int, default=0,
                    help="streaming slots per GPU (W); 0 = sweep W=1..L")
    ap.add_argument("--tf", type=float, default=1.0,
                    help="forward cost per chunk")
    ap.add_argument("--tb", type=float, default=2.0,
                    help="backward cost per chunk")
    ap.add_argument("--th2d", type=float, default=0.5,
                    help="H2D weight load cost per chunk")
    ap.add_argument("--td2h", type=float, default=0.5,
                    help="D2H grad writeback cost per chunk (0 disables)")
    ap.add_argument("--comm", type=float, default=0.0,
                    help="cross-stage activation/grad hop latency")
    ap.add_argument("--pin", type=int, default=0,
                    help="pin this many evenly spaced chunks per stage")
    ap.add_argument("--grad-mode", choices=("writeback", "stream"),
                    default="writeback",
                    help="grad accumulation: per-microbatch D2H+CPU add "
                         "(writeback) or streamed GPU accumulator (stream)")
    ap.add_argument("--tcpu", type=float, default=0.0,
                    help="CPU accumulate cost per grad packet "
                         "(writeback mode; 0 disables)")
    ap.add_argument("--acc-slots", type=int, default=None,
                    help="grad accumulator residency slots per stage "
                         "(stream mode; default = window)")
    ap.add_argument("--tact", type=float, default=0.0,
                    help="activation packet transfer cost each way "
                         "(0 = activation offload disabled)")
    ap.add_argument("--act-slots", type=int, default=0,
                    help="GPU activation slots per stage; 0 with --window = "
                         "sweep over slot counts, 0 in the window sweep = "
                         "unlimited")
    ap.add_argument("--act-policy", choices=("eager", "lazy"),
                    default="eager",
                    help="eager = offload packets at birth when D2H is idle; "
                         "lazy = only under slot pressure")
    ap.add_argument("--schedule", choices=sorted(_BUILDERS), default="1b1f")
    ap.add_argument("--width", type=int, default=120, help="ascii gantt width")
    ap.add_argument("--plot", type=str, nargs="?", const="", default=None,
                    help="render a matplotlib Gantt; optionally a .png path")
    ap.add_argument("--plot-scale", type=float, default=1.0,
                    help="multiply the Gantt figure size (fonts keep their "
                         "point size; ~1.5-2 declutters busy schedules)")
    ap.add_argument("--plot-dpi", type=int, default=130,
                    help="resolution of the saved Gantt image")
    args = ap.parse_args()

    p, m, L = args.p, args.m, args.chunks
    rank_ops = _BUILDERS[args.schedule](p, m)
    pinned = evenly_pinned(L, args.pin)
    kw = dict(tf=args.tf, tb=args.tb, th2d=args.th2d, td2h=args.td2h,
              comm=args.comm, pinned=pinned, grad_mode=args.grad_mode,
              tcpu=args.tcpu, acc_slots=args.acc_slots)
    if args.tact > 0:
        kw.update(tact=args.tact, act_policy=args.act_policy,
                  act_slots=(None if args.act_slots <= 0
                             else args.act_slots))
    print(
        f"schedule={args.schedule} p={p} m={m} chunks={L} "
        f"tf={args.tf} tb={args.tb} th2d={args.th2d} td2h={args.td2h} "
        f"comm={args.comm} pin={args.pin} grad_mode={args.grad_mode}"
        f" tcpu={args.tcpu} acc_slots={args.acc_slots}"
        + (f" tact={args.tact} act_policy={args.act_policy}"
           if args.tact > 0 else "")
        + "\n"
    )

    # activation-slot sweep: --window given, --tact > 0, --act-slots 0
    if args.window > 0 and args.tact > 0 and args.act_slots <= 0:
        base = simulate_pipeline_offload(
            rank_ops, L, args.window, **{**kw, "act_slots": None})
        peak0 = max(base.baseline_peak_act)
        print(f"  act-slot sweep (W={args.window}, no-offload peak = "
              f"{peak0} packets on the worst stage):")
        print(f"  {'slots':>5}  {'makespan':>9}  {'overhead':>8}  "
              f"{'gpu% (per stage)':<20}  {'offl':>5}  {'reld':>5}  "
              f"{'peakA':>5}")
        rows_a = []
        for slots in range(1, max(2, peak0 + 1)):
            res = simulate_pipeline_offload(
                rank_ops, L, args.window, **{**kw, "act_slots": slots})
            rows_a.append((slots, res))
            gpu_utils = "/".join(f"{100 * res.util(s, 'gpu'):.0f}"
                                 for s in range(res.p))
            print(f"  {slots:>5}  {res.makespan:>9.1f}  "
                  f"{100 * res.overhead_vs_ref:>7.1f}%  {gpu_utils:<20}  "
                  f"{sum(res.n_act_offloads):>5}  "
                  f"{sum(res.n_act_reloads):>5}  "
                  f"{max(res.peak_act(s) for s in range(res.p)):>5}")
        if args.plot is not None:
            out = args.plot or None
            picks = sorted({1, max(1, peak0 // 2), peak0})
            plot_gantt(
                [(f"{args.schedule} p={p} m={m} L={L} W={args.window} "
                  f"act_slots={sl}", r)
                 for sl, r in rows_a if sl in picks],
                out_path=out, show=(out is None),
                scale=args.plot_scale, dpi=args.plot_dpi,
            )
        return 0

    if args.window > 0:
        res = simulate_pipeline_offload(rank_ops, L, args.window, **kw)
        print(report(f"W={args.window}", res, args.width))
        if args.plot is not None:
            out = args.plot or None
            plot_gantt(
                [(f"{args.schedule} p={p} m={m} L={L} W={args.window}", res)],
                out_path=out, show=(out is None),
                scale=args.plot_scale, dpi=args.plot_dpi,
            )
        return 0

    # sweep table over window sizes
    rows = []
    for w in range(1, L + 1):
        rows.append((w, simulate_pipeline_offload(rank_ops, L, w, **kw)))
    act_hdr = (f"  {'offl':>5}  {'reld':>5}  {'peakA':>5}"
               if args.tact > 0 else "")
    print(f"  {'W':>3}  {'makespan':>9}  {'overhead':>8}  {'gpu% (per stage)':<20}"
          f"  {'h2d%max':>7}  {'loads':>6}  {'naive':>6}  {'peak':>4}{act_hdr}")
    for w, res in rows:
        gpu_utils = "/".join(f"{100 * res.util(s, 'gpu'):.0f}"
                             for s in range(res.p))
        act = (f"  {sum(res.n_act_offloads):>5}  {sum(res.n_act_reloads):>5}"
               f"  {max(res.peak_act(s) for s in range(res.p)):>5}"
               if args.tact > 0 else "")
        print(f"  {w:>3}  {res.makespan:>9.1f}  "
              f"{100 * res.overhead_vs_ref:>7.1f}%  {gpu_utils:<20}  "
              f"{100 * max(res.util(s, 'h2d') for s in range(res.p)):>7.0f}"
              f"  {sum(res.n_loads):>6}  "
              f"{sum(res.naive_loads(s) for s in range(res.p)):>6}  "
              f"{max(res.peak_resident):>4}{act}")
    if args.plot is not None:
        out = args.plot or None
        picks = sorted({1, max(1, L // 2), L})
        plot_gantt(
            [(f"{args.schedule} p={p} m={m} L={L} W={w}", res)
             for w, res in rows if w in picks],
            out_path=out, show=(out is None),
            scale=args.plot_scale, dpi=args.plot_dpi,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
