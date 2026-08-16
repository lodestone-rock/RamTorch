"""
offload_simulator.py
--------------------
Fast, executor-free simulator for windowed CPU->GPU layer streaming
(offload/prefetch schedules), the offloading analog of
``schedule_simulator.py``.

The model: N layer-chunks live in CPU pinned memory; the GPU holds at most W
of them (the *window*). Compute walks an itinerary of ``("F", i)`` /
``("B", i)`` ops serially; a prefetcher streams upcoming layers over the H2D
link while compute runs, and backward grad writebacks go out over the D2H
link. Three serial resources overlap each other: GPU compute, the H2D copy
engine, and the D2H copy engine (matching the two PCIe copy engines that
``CPUBouncingLinear`` exploits with its two streams).

Dependencies modeled:
  * F_i / B_i need layer i resident on the GPU; compute is serial and runs in
    itinerary order (which already encodes F_(i-1) -> F_i and B_(i+1) -> B_i).
  * B_i emits a grad writeback G_i on the D2H channel (td2h = 0 disables).
  * Residency capacity is W chunks. Eviction is free (weights on the GPU are
    read-only copies) and uses farthest-next-use in the itinerary — Belady's
    algorithm, optimal for a known itinerary. A load reserves its slot when
    it starts.
  * Prefetcher: whenever the H2D link is idle, load the earliest itinerary
    layer that is not resident, if a slot is free or a resident layer with a
    strictly farther next use can be evicted (never evicts the layer compute
    is currently running).
  * Optional NVMe tier: layers in ``nvme`` load on the *same* H2D channel
    but cost ``tnvme`` (> ``th2d``) per load — "slower H2D". Empirically the
    disk->GPU path is disk -> RAM -> GPU and its second hop serializes on
    the H2D copy engine with all other host->device traffic (measured in
    ``examples/nvme_h2d_contention_test.py``: concurrent H2D throughputs sum
    to the single-engine limit), so a separate parallel channel would be
    wrong for anything but true GPUDirect Storage. Placement helpers:
    ``interleaved_nvme`` (spread NVMe layers between CPU ones) vs
    ``tail_nvme`` (CPU first, NVMe tail).

Itineraries:
  * Training (echo): F0..F(N-1), B(N-1)..B0, repeatable for multiple steps.
    The turnaround is self-warming — the window holds the last W layers
    exactly when backward starts, and the first W layers when the next step's
    forward starts.
  * Inference (ring): F0..F(N-1) repeated; after F_i the layer's next use is
    the next pass, so the prefetcher wraps and streams 0, 1, 2, ... while the
    tail layers compute.

  * Optional ACTIVATION offloading (``tact`` > 0): each F(i) produces one
    activation packet that backward B(i) consumes. Packets occupy
    ``act_slots`` GPU slots (a separate pool from the weight window — the
    unit is "one chunk's saved activations", a different byte size). A
    packet may be *offloaded* to CPU RAM (kind "O" on the D2H channel,
    sharing it with grad writebacks) which frees its slot, and must then be
    *reloaded* (kind "R" on the H2D channel, sharing it with weight loads)
    before its B runs. Eviction is farthest-next-B-use (Belady); once a
    packet has a RAM copy, re-dropping it is free ("clean drop").
    ``act_policy="eager"`` offloads at birth whenever D2H is idle (spreads
    traffic into the idle forward phase); ``"lazy"`` offloads only under
    slot pressure (bursts, but never moves bytes it doesn't have to).
    Inference ops never create packets (no_grad saves nothing).

Out of scope: optimizer step, multi-GPU composition (see
``pipeline_offload_simulator`` for the pipeline version).

Usage:
    python -m ramtorch.offload_simulator                     # window sweep table
    python -m ramtorch.offload_simulator --layers 10 --window 5 --mode train
    python -m ramtorch.offload_simulator --layers 10 --window 5 --plot out.png
    # NVMe tier: 6 of 12 layers on NVMe, interleaved vs tail placement
    python -m ramtorch.offload_simulator --layers 12 --window 4 --nvme 6
    # activation offload: packet cost 0.4, 3 GPU activation slots
    python -m ramtorch.offload_simulator --layers 10 --window 3 --tact 0.4 \
        --act-slots 3
    # importable:
    from ramtorch.offload_simulator import (
        train_itinerary, infer_itinerary, simulate_offload, gantt, plot_gantt,
        interleaved_nvme, tail_nvme)

Follow-up phase (not built here): once this simulator validates the design,
implement a portable streaming executor at layer-chunk granularity — a loader
thread with pinned-memory copies and event/queue handshakes like
``pipeline_relay``'s mailboxes, instead of raw CUDA stream choreography as in
``ramtorch/modules/linear.py`` — reusing the pipeline ``Stage`` splitting
conventions so offloading composes with ``stage_modules``.
"""

from __future__ import annotations

import argparse
import bisect
import json
from collections import deque
from typing import Dict, FrozenSet, List, Optional, Tuple

Op = Tuple[str, int]           # ("F"|"B", layer)
Itinerary = List[Op]
Span = Tuple[float, float]     # (start, end)

INF = float("inf")
_EPS = 1e-9


# ── Itinerary builders ────────────────────────────────────────────────────────

def train_itinerary(n: int, steps: int = 1) -> Itinerary:
    """Echo itinerary: F0..F(n-1), B(n-1)..B0, repeated ``steps`` times."""
    ops: Itinerary = []
    for _ in range(steps):
        ops.extend(("F", i) for i in range(n))
        ops.extend(("B", i) for i in reversed(range(n)))
    return ops


def infer_itinerary(n: int, passes: int = 1) -> Itinerary:
    """Ring itinerary: F0..F(n-1), repeated ``passes`` times."""
    return [("F", i) for _ in range(passes) for i in range(n)]


def evenly_pinned(n: int, k: int) -> FrozenSet[int]:
    """k evenly spaced layer indices out of n (e.g. 6/12 -> {0,2,4,6,8,10})."""
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k} n={n}")
    return frozenset(i * n // k for i in range(k)) if k else frozenset()


def interleaved_nvme(n: int, k: int) -> FrozenSet[int]:
    """k evenly spaced NVMe-resident layers out of n (e.g. 6/12 ->
    {1,3,5,7,9,11}).

    Uses the *last* index of each stride so layer 0 stays CPU-resident when
    k < n — the first load (the one compute unavoidably waits on) goes over
    the faster H2D link.
    """
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k} n={n}")
    return frozenset((i + 1) * n // k - 1 for i in range(k)) if k else frozenset()


def tail_nvme(n: int, k: int) -> FrozenSet[int]:
    """Contiguous placement: the last k of n layers on NVMe ("CPU then NVMe")."""
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k} n={n}")
    return frozenset(range(n - k, n))


# ── Result ────────────────────────────────────────────────────────────────────

class OffloadResult:
    """Timeline + metrics from one ``simulate_offload`` run.

    ``spans`` maps channel name -> ordered list of ``((kind, layer), (st, en))``:
      * ``"gpu"``: kind "F" or "B" (compute)
      * ``"h2d"``: kind "L" (load from CPU RAM), "N" (slower load staged
        from NVMe — same channel, higher cost), or "R" (activation reload)
      * ``"d2h"``: kind "G" (grad writeback) or "O" (activation offload)

    ``resident_at[i]`` is the frozenset of resident layers at the instant
    compute op ``i`` starts (snapshot taken before any same-instant eviction).
    ``occupancy_timeline`` is a list of ``(time, occupied_slots)`` events where
    occupied slots include in-flight load reservations.
    """

    def __init__(
        self,
        itinerary: Itinerary,
        n_layers: int,
        window: int,
        tf: float,
        tb: float,
        th2d: float,
        td2h: float,
        spans: Dict[str, List[Tuple[Tuple[str, int], Span]]],
        resident_at: List[FrozenSet[int]],
        occupancy_timeline: List[Tuple[float, int]],
        n_loads: int,
        warmup: int = 0,
        pinned: FrozenSet[int] = frozenset(),
        nvme: FrozenSet[int] = frozenset(),
        tnvme: float = 0.0,
        *,
        tact: float = 0.0,
        act_slots: int = 0,
        act_policy: str = "eager",
        act_timeline: Optional[List[Tuple[float, int]]] = None,
        n_act_offloads: int = 0,
        n_act_reloads: int = 0,
        baseline_peak_act: int = 0,
    ):
        self.itinerary = itinerary
        self.n_layers = n_layers
        self.window = window
        self.warmup = warmup
        self.pinned = frozenset(pinned)
        self.nvme = frozenset(nvme)
        self.tf, self.tb, self.th2d, self.td2h = tf, tb, th2d, td2h
        self.tnvme = tnvme
        self.tact = tact
        self.act_slots = act_slots
        self.act_policy = act_policy
        self.act_timeline = act_timeline or []
        self.n_act_offloads = n_act_offloads
        self.n_act_reloads = n_act_reloads
        self.baseline_peak_act = baseline_peak_act
        self.spans = spans
        self.resident_at = resident_at
        self.occupancy_timeline = occupancy_timeline
        self.n_loads = n_loads
        self.makespan = max(
            (en for ch in spans.values() for _, (_, en) in ch), default=0.0
        )

    # -- per-channel metrics --
    def busy(self, ch: str) -> float:
        return sum(en - st for _, (st, en) in self.spans[ch])

    def util(self, ch: str) -> float:
        return 0.0 if self.makespan == 0 else self.busy(ch) / self.makespan

    @property
    def compute_end(self) -> float:
        return max((en for _, (_, en) in self.spans["gpu"]), default=0.0)

    @property
    def stall(self) -> float:
        """Compute idle time in [0, last compute end] — waiting for loads.

        Includes the unavoidable initial wait for the first layer (>= th2d).
        """
        return self.compute_end - self.busy("gpu")

    @property
    def peak_resident(self) -> int:
        return max((occ for _, occ in self.occupancy_timeline), default=0)

    @property
    def act_on(self) -> bool:
        return self.tact > 0 and self.baseline_peak_act > 0

    @property
    def peak_act(self) -> int:
        """Peak GPU activation slots actually occupied (0 when disabled)."""
        return max((occ for _, occ in self.act_timeline), default=0)

    # -- analytic lower bounds --
    @property
    def total_compute(self) -> float:
        nf = sum(1 for k, _ in self.itinerary if k == "F")
        nb = sum(1 for k, _ in self.itinerary if k == "B")
        return nf * self.tf + nb * self.tb

    @property
    def compute_bound(self) -> float:
        """Serial compute + the first layer's load latency (0 if pinned,
        ``tnvme`` if NVMe-resident, else ``th2d``)."""
        if not self.itinerary:
            return 0.0
        l0 = self.itinerary[0][1]
        if l0 in self.pinned:
            first = 0.0
        elif l0 in self.nvme:
            first = self.tnvme
        else:
            first = self.th2d
        return first + self.total_compute

    @property
    def total_mem(self) -> int:
        """GPU slots this config consumes: streaming window + pinned layers."""
        return self.window + len(self.pinned)

    @property
    def min_act_roundtrips(self) -> int:
        """Certain lower bound on forced activation round trips: at the
        aliveness peak at most ``act_slots`` packets fit on the GPU, so at
        least ``peak - act_slots`` must be in RAM at that instant — each got
        there via one offload and needs one reload."""
        if not self.act_on:
            return 0
        return max(0, self.baseline_peak_act - self.act_slots)

    @property
    def transfer_bound(self) -> float:
        """Each distinct unpinned layer loads at least once, all on the one
        H2D channel (NVMe loads just cost more); writebacks serialize on
        D2H. Forced activation round trips add ``tact`` on both channels."""
        used = {l for _, l in self.itinerary} - self.pinned
        n_nvme = len(used & self.nvme)
        n_cpu = len(used) - n_nvme
        nb = sum(1 for k, _ in self.itinerary if k == "B")
        d2h_total = nb * self.td2h if self.td2h > 0 else 0.0
        act_min = self.min_act_roundtrips * self.tact
        return max(n_cpu * self.th2d + n_nvme * self.tnvme + act_min,
                   d2h_total + act_min)

    @property
    def bound(self) -> float:
        return max(self.compute_bound, self.transfer_bound)

    @property
    def regime(self) -> str:
        return (
            "compute-bound"
            if self.compute_bound >= self.transfer_bound
            else "transfer-bound"
        )

    @property
    def bound_gap(self) -> float:
        """(makespan - bound) / bound; 0 means the schedule hit its bound."""
        return 0.0 if self.bound == 0 else (self.makespan - self.bound) / self.bound

    def metrics(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "n_layers": self.n_layers,
            "window": self.window,
            "warmup": self.warmup,
            "pinned": sorted(self.pinned),
            "total_mem": self.total_mem,
            "ops": len(self.itinerary),
            "nvme": sorted(self.nvme),
            "makespan": self.makespan,
            "stall": self.stall,
            "gpu_util": self.util("gpu"),
            "h2d_util": self.util("h2d"),
            "d2h_util": self.util("d2h"),
            "n_loads": self.n_loads,
            "n_nvme_loads": sum(
                1 for (k, _), _ in self.spans["h2d"] if k == "N"
            ),
            "peak_resident": self.peak_resident,
            "compute_bound": self.compute_bound,
            "transfer_bound": self.transfer_bound,
            "regime": self.regime,
            "bound_gap": self.bound_gap,
        }
        if self.act_on:
            out.update({
                "tact": self.tact,
                "act_slots": self.act_slots,
                "act_policy": self.act_policy,
                "peak_act": self.peak_act,
                "baseline_peak_act": self.baseline_peak_act,
                "n_act_offloads": self.n_act_offloads,
                "n_act_reloads": self.n_act_reloads,
                "min_act_roundtrips": self.min_act_roundtrips,
            })
        return out


# ── Simulator ─────────────────────────────────────────────────────────────────

def simulate_offload(
    itinerary: Itinerary,
    n_layers: int,
    window: int,
    tf: float = 1.0,
    tb: float = 2.0,
    th2d: float = 0.5,
    td2h: float = 0.5,
    warmup: int = 0,
    pinned: Optional[FrozenSet[int]] = None,
    nvme: Optional[FrozenSet[int]] = None,
    tnvme: float = 2.5,
    tact: float = 0.0,
    act_slots: Optional[int] = None,
    act_policy: str = "eager",
) -> OffloadResult:
    """Discrete-event simulation of the windowed streaming schedule.

    Three serial channels (gpu compute, H2D loads, D2H writebacks) run
    concurrently. Zero-cost ops are handled exactly (starts and completions
    at the same instant are processed to a fixed point before time advances).

    ``warmup``: hold compute until the first ``warmup`` loads have completed
    (a preload phase). Capped at ``window`` (the GPU cannot hold more) and at
    the number of distinct layers in the itinerary. 0 = greedy start, i.e.
    compute begins as soon as its first layer is resident. Since the
    prefetcher is asynchronous and never waits for compute (except for
    Belady eviction stalls, which only ever *resolve* as compute advances),
    warmup can never finish earlier than greedy — it only moves stall time
    into one upfront block.

    ``pinned``: layers permanently resident on the GPU — never loaded, never
    evicted, occupying their own slots *in addition to* the ``window``
    streaming slots (total memory = ``window + len(pinned)``).

    ``nvme``: layers resident on NVMe instead of CPU RAM. They load on the
    same H2D channel but cost ``tnvme`` per load ("slower H2D") — the
    disk->GPU path's host->device hop serializes on the H2D copy engine
    (see ``examples/nvme_h2d_contention_test.py``). Pinned layers are
    dropped from the set (they never load, so residency class is moot).

    ``tact`` > 0 enables ACTIVATION offloading (0 = feature off, behavior
    byte-identical to before). F(i) creates one activation packet (only if
    a matching B(i) appears later — inference passes save nothing) that
    occupies one of ``act_slots`` GPU slots (default: unlimited) until its
    B(i) completes. Offloads ("O", cost ``tact``) ride the D2H channel and
    free the slot; reloads ("R", cost ``tact``) ride H2D and are prefetched
    in itinerary order. Under slot pressure the farthest-next-B-use dirty
    packet is offloaded (needed offloads outrank grad writebacks); packets
    with a valid RAM copy re-drop for free. ``act_policy="eager"`` also
    offloads at birth whenever D2H is otherwise idle; ``"lazy"`` moves
    bytes only under pressure.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    pinned = frozenset(pinned or ())
    for l in pinned:
        if not (0 <= l < n_layers):
            raise ValueError(f"pinned layer {l} out of range [0, {n_layers})")
    nvme = frozenset(nvme or ())
    for l in nvme:
        if not (0 <= l < n_layers):
            raise ValueError(f"nvme layer {l} out of range [0, {n_layers})")
    nvme -= pinned
    for k, l in itinerary:
        if k not in ("F", "B"):
            raise ValueError(f"bad op kind {k!r}")
        if not (0 <= l < n_layers):
            raise ValueError(f"layer {l} out of range [0, {n_layers})")
    if act_policy not in ("eager", "lazy"):
        raise ValueError(f"act_policy must be eager|lazy, got {act_policy!r}")
    if act_slots is not None and act_slots < 1:
        raise ValueError(f"act_slots must be >= 1, got {act_slots}")
    distinct_unpinned = len({l for _, l in itinerary} - pinned)
    warmup = max(0, min(warmup, window, distinct_unpinned))
    capacity = window + len(pinned)

    n_ops = len(itinerary)
    # positions where each layer is used, for next-use (Belady) queries
    uses: List[List[int]] = [[] for _ in range(n_layers)]
    for i, (_, l) in enumerate(itinerary):
        uses[l].append(i)

    def next_use(layer: int, from_pos: int) -> float:
        us = uses[layer]
        j = bisect.bisect_left(us, from_pos)
        return us[j] if j < len(us) else INF

    # ── activation machinery (tact > 0 and at least one B op) ──────────────
    act_buses: List[List[int]] = [[] for _ in range(n_layers)]
    for i, (k, l) in enumerate(itinerary):
        if k == "B":
            act_buses[l].append(i)
    act_on = tact > 0 and any(act_buses)
    # baseline peak = concurrently-alive packets with NO offloading (what
    # the feature saves); also the "unlimited" slot count
    _alive: set = set()
    baseline_peak_act = 0
    for i, (k, l) in enumerate(itinerary):
        if k == "F" and any(b > i for b in act_buses[l]):
            _alive.add(l)
            baseline_peak_act = max(baseline_peak_act, len(_alive))
        elif k == "B":
            _alive.discard(l)
    n_act_slots = baseline_peak_act if act_slots is None else act_slots

    def next_act_use(layer: int, from_pos: int) -> float:
        us = act_buses[layer]
        j = bisect.bisect_left(us, from_pos)
        return us[j] if j < len(us) else INF

    # layer -> "gpu" (resident dirty) | "gpu_clean" (resident + RAM copy) |
    # "offloading" | "reloading" | "ram"; absent = no live packet
    act_state: Dict[int, str] = {}
    act_timeline: List[Tuple[float, int]] = [(0.0, 0)]
    n_act_off = 0
    n_act_rel = 0

    def act_occupied() -> int:
        return sum(1 for s in act_state.values() if s != "ram")

    resident: set = set(pinned)
    comp_idx = 0                       # next itinerary op to run
    gpu_cur: Optional[Tuple[str, int, float]] = None   # (kind, layer, start)
    gpu_end = 0.0
    # h2d_cur = (kind, layer, start), kind "L"/"N" weight load or "R" act
    # reload; d2h_cur = (kind, layer, start), kind "G" grad or "O" offload
    h2d_cur: Optional[Tuple[str, int, float]] = None
    h2d_end = 0.0
    d2h_cur: Optional[Tuple[str, int, float]] = None
    d2h_end = 0.0
    d2h_q: deque = deque()             # (layer, ready_time), FIFO

    def load_cost(layer: int) -> float:
        return tnvme if layer in nvme else th2d

    def load_kind(layer: int) -> str:
        return "N" if layer in nvme else "L"

    spans: Dict[str, List[Tuple[Tuple[str, int], Span]]] = {
        "gpu": [], "h2d": [], "d2h": []
    }
    resident_at: List[FrozenSet[int]] = []
    occ_timeline: List[Tuple[float, int]] = [(0.0, len(pinned))]
    n_loads = 0
    now = 0.0

    def occupancy() -> int:
        return len(resident) + (1 if h2d_cur is not None
                                and h2d_cur[0] != "R" else 0)

    def record_occ(t: float) -> None:
        occ_timeline.append((t, occupancy()))

    def record_act(t: float) -> None:
        act_timeline.append((t, act_occupied()))

    def act_in_use() -> Optional[int]:
        """Layer whose packet the current gpu op is writing/reading."""
        return gpu_cur[1] if gpu_cur is not None else None

    def act_make_room(need_pos: int) -> bool:
        """A slot is free, or a CLEAN packet (valid RAM copy) with next use
        beyond ``need_pos`` can be dropped for free. True if a slot is now
        available."""
        if act_occupied() < n_act_slots:
            return True
        busy = act_in_use()
        victim, victim_nu = None, float(need_pos)
        for l, s in act_state.items():
            if s != "gpu_clean" or l == busy:
                continue
            nu = next_act_use(l, comp_idx)
            if nu > victim_nu:
                victim, victim_nu = l, nu
        if victim is None:
            return False
        act_state[victim] = "ram"     # free drop: RAM copy already valid
        record_act(now)
        return True

    def act_offload_victim(need_pos: float) -> Optional[int]:
        """Farthest-next-B-use DIRTY resident packet beyond ``need_pos``."""
        busy = act_in_use()
        victim, victim_nu = None, float(need_pos)
        for l, s in act_state.items():
            if s != "gpu" or l == busy:
                continue
            nu = next_act_use(l, comp_idx)
            if nu > victim_nu:
                victim, victim_nu = l, nu
        return victim

    def act_need_pos() -> Optional[int]:
        """Itinerary position of the earliest op TRULY blocked on an act
        slot: a B whose packet sits in RAM, or an F that must create a
        packet after accounting for slots that intervening Bs free."""
        free = n_act_slots - act_occupied()
        if free > 0:
            return None
        if (gpu_cur is not None and gpu_cur[0] == "B"
                and act_state.get(gpu_cur[1]) is not None
                and act_state.get(gpu_cur[1]) != "ram"):
            free += 1  # the running B frees its packet's slot at its end
        for pos in range(comp_idx, n_ops):
            k, l = itinerary[pos]
            st = act_state.get(l)
            if k == "B":
                if st == "ram":
                    return pos  # its reload needs a slot
                if st is not None:
                    free += 1   # on-GPU packet frees its slot at B end
            elif l not in act_state and next_act_use(l, pos) < INF:
                if free <= 0:
                    return pos
                free -= 1
        return None

    while True:
        # 1) process completions due at `now`
        if gpu_cur is not None and gpu_end <= now + _EPS:
            kind, layer, st = gpu_cur
            spans["gpu"].append(((kind, layer), (st, gpu_end)))
            if kind == "B" and td2h > 0:
                d2h_q.append((layer, gpu_end))
            if kind == "B" and act_on and layer in act_state:
                del act_state[layer]        # packet consumed, slot freed
                record_act(now)
            gpu_cur = None
        if h2d_cur is not None and h2d_end <= now + _EPS:
            hkind, layer, st = h2d_cur
            spans["h2d"].append(((hkind, layer), (st, h2d_end)))
            h2d_cur = None
            if hkind == "R":
                act_state[layer] = "gpu_clean"  # RAM copy stays valid
                record_act(now)
            else:
                resident.add(layer)
                record_occ(now)
        if d2h_cur is not None and d2h_end <= now + _EPS:
            dkind, layer, st = d2h_cur
            spans["d2h"].append(((dkind, layer), (st, d2h_end)))
            d2h_cur = None
            if dkind == "O":
                act_state[layer] = "ram"        # slot freed
                record_act(now)

        # 2) start whatever can start at `now` (loop back for zero-cost chains)
        progressed = False

        # compute (checked before H2D so resident_at snapshots precede
        # any same-instant eviction)
        if gpu_cur is None and comp_idx < n_ops:
            kind, layer = itinerary[comp_idx]
            n_weight_loads = sum(
                1 for (k, _), _ in spans["h2d"] if k != "R"
            ) if act_on else len(spans["h2d"])
            warmed = comp_idx > 0 or n_weight_loads >= warmup
            ok = layer in resident and warmed
            makes_packet = False
            if ok and act_on:
                if kind == "F":
                    # only layers with a future B save activations
                    if next_act_use(layer, comp_idx) < INF:
                        makes_packet = act_make_room(comp_idx)
                        ok = makes_packet
                else:  # B needs its packet on the GPU (if one was made)
                    s = act_state.get(layer)
                    ok = s is None or s in ("gpu", "gpu_clean")
            if ok:
                resident_at.append(frozenset(resident))
                dur = tf if kind == "F" else tb
                gpu_cur = (kind, layer, now)
                gpu_end = now + dur
                comp_idx += 1
                if makes_packet:
                    act_state[layer] = "gpu"
                    record_act(now)
                progressed = True

        # d2h: a PRESSURE offload (an op is blocked on an act slot) outranks
        # grads; grads outrank EAGER offloads (policy "eager" spreads the
        # traffic into otherwise-idle link time)
        if act_on and d2h_cur is None:
            need = act_need_pos()
            if need is not None:
                v = act_offload_victim(need)
                if v is not None:
                    act_state[v] = "offloading"
                    d2h_cur = ("O", v, now)
                    d2h_end = now + tact
                    n_act_off += 1
                    progressed = True
        if d2h_cur is None and d2h_q and d2h_q[0][1] <= now + _EPS:
            layer, _ = d2h_q.popleft()
            d2h_cur = ("G", layer, now)
            d2h_end = now + td2h
            progressed = True
        if (act_on and d2h_cur is None and act_policy == "eager"):
            # never offload a packet whose B is the current or next op
            v = act_offload_victim(comp_idx + 1)
            if v is not None:
                act_state[v] = "offloading"
                d2h_cur = ("O", v, now)
                d2h_end = now + tact
                n_act_off += 1
                progressed = True

        # prefetch: earliest unmet need in itinerary order — a weight load
        # (NVMe layers ride the same channel at their higher cost) or an
        # offloaded activation's reload before its B
        if h2d_cur is None:
            for pos in range(comp_idx, n_ops):
                pk, l = itinerary[pos]
                if l not in resident:
                    can_start = occupancy() < capacity
                    if not can_start:
                        # Belady eviction: farthest next use, never the layer
                        # compute is running and never a pinned layer, only
                        # if farther than the candidate
                        in_use = gpu_cur[1] if gpu_cur is not None else None
                        victim, victim_nu = None, -1.0
                        for r in resident:
                            if r == in_use or r in pinned:
                                continue
                            nu = next_use(r, comp_idx)
                            if nu > victim_nu:
                                victim, victim_nu = r, nu
                        if victim is not None and victim_nu > pos:
                            resident.discard(victim)
                            record_occ(now)
                            can_start = True
                    if can_start:
                        h2d_cur = (load_kind(l), l, now)
                        h2d_end = now + load_cost(l)
                        n_loads += 1
                        record_occ(now)
                        progressed = True
                    break  # first unmet weight need wins (or waits)
                if (act_on and pk == "B" and act_state.get(l) == "ram"):
                    if act_make_room(pos):
                        act_state[l] = "reloading"
                        h2d_cur = ("R", l, now)
                        h2d_end = now + tact
                        n_act_rel += 1
                        record_act(now)
                        progressed = True
                        break
                    continue  # slot-blocked: scan on for a weight load

        if progressed:
            continue

        # 3) done, or advance time to the next completion
        work_left = (
            comp_idx < n_ops
            or gpu_cur is not None
            or h2d_cur is not None
            or d2h_cur is not None
            or d2h_q
        )
        if not work_left:
            break
        pending = [
            end
            for cur, end in (
                (gpu_cur, gpu_end), (h2d_cur, h2d_end), (d2h_cur, d2h_end)
            )
            if cur is not None
        ]
        if not pending:
            raise RuntimeError(
                f"offload schedule deadlocked at t={now} "
                f"(comp_idx={comp_idx}/{n_ops}, resident={sorted(resident)}, "
                f"window={window}, act={dict(act_state)}) — bug"
            )
        now = min(pending)

    return OffloadResult(
        itinerary, n_layers, window, tf, tb, th2d, td2h,
        spans, resident_at, occ_timeline, n_loads, warmup=warmup,
        pinned=pinned, nvme=nvme, tnvme=tnvme,
        tact=tact, act_slots=n_act_slots if act_on else 0,
        act_policy=act_policy, act_timeline=act_timeline,
        n_act_offloads=n_act_off, n_act_reloads=n_act_rel,
        baseline_peak_act=baseline_peak_act if act_on else 0,
    )


# ── ASCII Gantt ───────────────────────────────────────────────────────────────

_ROW_CHAR = {"F": "F", "B": "b", "L": "L", "N": "N", "G": "G",
             "O": "O", "R": "R"}


def gantt(res: OffloadResult, width: int = 100) -> str:
    """Three rows (gpu / h2d / d2h); '.' = idle. NVMe loads show as 'N' on
    the h2d row (same channel, slower); activation offloads as 'O' on d2h,
    reloads as 'R' on h2d."""
    ms = res.makespan
    if ms <= 0:
        return "(empty)"
    cols = max(10, width)
    lines = []
    for ch in ("gpu", "h2d", "d2h"):
        row = ["."] * cols
        for (kind, _), (st, en) in res.spans[ch]:
            a = int(st / ms * cols)
            b = max(a + 1, int(en / ms * cols))
            for c in range(a, min(b, cols)):
                row[c] = _ROW_CHAR[kind]
        lines.append(f"  {ch:>4} |{''.join(row)}|")
    scale = "".join(
        str(int(t / 10 % 10)) if t % 10 == 0 else ("|" if t % 5 == 0 else " ")
        for t in range(cols)
    )
    lines.append(f"       +{scale}+  0..{ms:.1f}")
    return "\n".join(lines)


# ── Matplotlib Gantt ──────────────────────────────────────────────────────────

def plot_gantt(
    results: List[Tuple[str, OffloadResult]],
    out_path: Optional[str] = None,
    show: bool = True,
    scale: float = 1.0,
    dpi: int = 130,
):
    """One Gantt (gpu/h2d/d2h rows) + residency strip per result.

    Bars are labeled ``F3 / B3 / L3 / N3 / G3`` (op kind + layer index);
    NVMe loads ("N") share the h2d row in a distinct color. The strip below
    each Gantt steps through occupied window slots over time.

    ``scale`` multiplies the figure dimensions (fonts keep their point
    size, so bars/labels get more room — use ~1.5-2 for busy schedules);
    ``dpi`` controls the saved image resolution.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Okabe-Ito colorblind-safe palette; the backward hatch adds a
    # color-independent cue vs the H2D loads (the confusable pair).
    colors = {"F": "#0072B2", "B": "#E69F00", "L": "#009E73",
              "N": "#D55E00", "G": "#CC79A7", "O": "#56B4E9",
              "R": "#F0E442"}
    hatches = {"F": "", "B": "//", "L": "", "N": "", "G": "", "O": "\\\\",
               "R": "\\\\"}
    kind_name = {"F": "forward", "B": "backward", "L": "H2D load",
                 "N": "NVMe load", "G": "D2H grad",
                 "O": "act offload (D2H)", "R": "act reload (H2D)"}
    n = len(results)
    max_ms = max((r.makespan for _, r in results), default=1.0)
    fig, axes = plt.subplots(
        2 * n, 1,
        figsize=(scale * max(10, max_ms * 0.45), scale * 3.2 * n),
        sharex=True, squeeze=False,
        gridspec_kw={"height_ratios": [3, 1] * n},
    )

    for k, (name, res) in enumerate(results):
        ax, axr = axes[2 * k, 0], axes[2 * k + 1, 0]
        rows = ("gpu", "h2d", "d2h")
        for ri, ch in enumerate(rows):
            y = len(rows) - 1 - ri  # gpu on top
            for (kind, layer), (st, en) in res.spans[ch]:
                ax.barh(y, en - st, left=st, height=0.7, color=colors[kind],
                        hatch=hatches[kind], edgecolor="white", linewidth=0.4)
                if en - st >= res.makespan * 0.015:
                    bbox = (
                        dict(facecolor=colors[kind], edgecolor="none", pad=0.6)
                        if hatches[kind] else None
                    )
                    txt = "black" if kind == "R" else "white"
                    ax.text((st + en) / 2, y, f"{kind}{layer}", ha="center",
                            va="center", fontsize=7, color=txt,
                            weight="bold", bbox=bbox)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([rows[len(rows) - 1 - i] for i in range(len(rows))])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(
            f"{name}   makespan={res.makespan:.1f}  stall={res.stall:.1f}  "
            f"loads={res.n_loads}  peak_res={res.peak_resident}/{res.total_mem}"
            + (f" (pinned {len(res.pinned)})" if res.pinned else "")
            + (f"  act={res.peak_act}/{res.act_slots} slots "
               f"(base {res.baseline_peak_act}, O={res.n_act_offloads} "
               f"R={res.n_act_reloads})" if res.act_on else "")
            + f"  {res.regime} (gap {100 * res.bound_gap:.0f}%)",
            loc="left", fontsize=10,
        )
        # residency strip (weight slots; activation slots overlaid when on)
        ts = [t for t, _ in res.occupancy_timeline] + [res.makespan]
        os_ = [o for _, o in res.occupancy_timeline]
        os_ = os_ + [os_[-1] if os_ else 0]
        axr.step(ts, os_, where="post", color="#555", linewidth=1.2,
                 label="weight slots")
        axr.fill_between(ts, os_, step="post", alpha=0.25, color="#888")
        axr.axhline(res.total_mem, color="#C0392B", linestyle="--", linewidth=0.8)
        cap = res.total_mem
        if res.act_on:
            ta = [t for t, _ in res.act_timeline] + [res.makespan]
            oa = [o for _, o in res.act_timeline]
            oa = oa + [oa[-1] if oa else 0]
            axr.step(ta, oa, where="post", color="#56B4E9", linewidth=1.2,
                     label="act slots")
            axr.axhline(res.act_slots, color="#56B4E9", linestyle=":",
                        linewidth=0.8)
            axr.legend(loc="upper right", fontsize=6)
            cap = max(cap, res.act_slots)
        axr.set_ylim(0, cap + 1)
        axr.set_ylabel("slots", fontsize=8)
        axr.grid(axis="x", alpha=0.25)

    axes[-1, 0].set_xlabel("time")
    kinds_present = {
        kind
        for _, res in results
        for ch in res.spans.values()
        for (kind, _), _ in ch
    }
    handles = [
        Patch(facecolor=c, hatch=hatches[k], edgecolor="white",
              label=kind_name[k])
        for k, c in colors.items()
        if k in kinds_present
    ]
    axes[0, 0].legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle("Windowed CPU->GPU offload streaming schedule", y=0.995)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
        print(f"wrote {out_path}")
    if show:
        plt.show()
    return fig


# ── CLI / demo ────────────────────────────────────────────────────────────────

def _build(mode: str, n: int, steps: int) -> Itinerary:
    if mode == "train":
        return train_itinerary(n, steps=steps)
    if mode == "infer":
        return infer_itinerary(n, passes=steps)
    raise ValueError(f"mode must be 'train' or 'infer', got {mode!r}")


def report(name: str, res: OffloadResult, width: int = 100) -> str:
    out = [
        f"  {name:<14} makespan={res.makespan:7.1f}  stall={res.stall:6.1f}  "
        f"gpu={100 * res.util('gpu'):3.0f}%  h2d={100 * res.util('h2d'):3.0f}%  "
        f"d2h={100 * res.util('d2h'):3.0f}%  loads={res.n_loads:3d}  "
        f"peak={res.peak_resident}/{res.window}  "
        f"{res.regime} gap={100 * res.bound_gap:4.1f}%"
    ]
    if res.act_on:
        out.append(
            f"  {'':<14} act[{res.act_policy}]: peak={res.peak_act}"
            f"/{res.act_slots} slots (no-offload base "
            f"{res.baseline_peak_act})  offloads={res.n_act_offloads}  "
            f"reloads={res.n_act_reloads}  min_rt={res.min_act_roundtrips}"
        )
    out.append(gantt(res, width))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", type=int, default=10, help="layer chunks (N)")
    ap.add_argument("--window", type=int, default=0,
                    help="GPU window slots (W); 0 = sweep table over W=1..N")
    ap.add_argument("--tf", type=float, default=1.0, help="forward cost per layer")
    ap.add_argument("--tb", type=float, default=2.0, help="backward cost per layer")
    ap.add_argument("--th2d", type=float, default=0.5, help="H2D load cost per layer")
    ap.add_argument("--td2h", type=float, default=0.5,
                    help="D2H grad writeback cost per layer (0 disables)")
    ap.add_argument("--mode", choices=("train", "infer"), default="train")
    ap.add_argument("--steps", type=int, default=1,
                    help="training steps (or inference passes)")
    ap.add_argument("--warmup", type=int, default=0,
                    help="preload this many layers before compute starts "
                         "(-1 = full window; 0 = greedy start)")
    ap.add_argument("--pin", type=int, default=0,
                    help="pin this many evenly spaced layers permanently on "
                         "the GPU (adds to memory on top of the window)")
    ap.add_argument("--nvme", type=int, default=0,
                    help="place this many layers on NVMe; they load on the "
                         "same H2D channel but cost --tnvme each")
    ap.add_argument("--tnvme", type=float, default=2.5,
                    help="NVMe load cost per layer (default 2.5 = 5x th2d)")
    ap.add_argument("--nvme-placement",
                    choices=("interleave", "tail", "both"), default="both",
                    help="NVMe layer layout: interleave = evenly spread among "
                         "CPU layers; tail = CPU first, NVMe last; "
                         "both = compare (default)")
    ap.add_argument("--tact", type=float, default=0.0,
                    help="activation packet transfer cost each way "
                         "(0 = activation offload disabled)")
    ap.add_argument("--act-slots", type=int, default=0,
                    help="GPU activation slots; 0 with --window = sweep over "
                         "slot counts, 0 in the window sweep = unlimited")
    ap.add_argument("--act-policy", choices=("eager", "lazy"),
                    default="eager",
                    help="eager = offload packets at birth when D2H is idle; "
                         "lazy = only under slot pressure")
    ap.add_argument("--width", type=int, default=100, help="ascii gantt width")
    ap.add_argument("--plot", type=str, nargs="?", const="", default=None,
                    help="render a matplotlib Gantt; optionally give a .png path")
    ap.add_argument("--plot-scale", type=float, default=1.0,
                    help="multiply the Gantt figure size (fonts keep their "
                         "point size; ~1.5-2 declutters busy schedules)")
    ap.add_argument("--plot-dpi", type=int, default=130,
                    help="resolution of the saved Gantt image")
    ap.add_argument("--json", action="store_true", help="print metrics as JSON")
    args = ap.parse_args()

    n = args.layers
    itin = _build(args.mode, n, args.steps)
    warmup = args.warmup
    kw = dict(tf=args.tf, tb=args.tb, th2d=args.th2d, td2h=args.td2h,
              pinned=evenly_pinned(n, args.pin))
    if args.tact > 0:
        kw.update(tact=args.tact, act_policy=args.act_policy,
                  act_slots=(None if args.act_slots <= 0
                             else args.act_slots))
    print(
        f"n={n} layers, mode={args.mode}, steps={args.steps}, "
        f"tf={args.tf} tb={args.tb} th2d={args.th2d} td2h={args.td2h}"
        + (f" tnvme={args.tnvme} nvme={args.nvme}" if args.nvme else "")
        + (f" tact={args.tact} act_policy={args.act_policy}"
           if args.tact > 0 else "")
        + "\n"
    )

    # NVMe placements to simulate: [(label, nvme_set)]
    if args.nvme > 0:
        kw["tnvme"] = args.tnvme
        placements = []
        if args.nvme_placement in ("interleave", "both"):
            placements.append(("interleave", interleaved_nvme(n, args.nvme)))
        if args.nvme_placement in ("tail", "both"):
            placements.append(("tail", tail_nvme(n, args.nvme)))
    else:
        placements = [("", frozenset())]

    # activation-slot sweep: --window given, --tact > 0, --act-slots 0
    if args.window > 0 and args.tact > 0 and args.act_slots <= 0:
        wu = args.window if warmup < 0 else warmup
        _, nv0 = placements[0]
        base = simulate_offload(itin, n, args.window, warmup=wu, nvme=nv0,
                                **{**kw, "act_slots": None})
        print(f"  act-slot sweep (W={args.window}, no-offload peak = "
              f"{base.baseline_peak_act} packets):")
        print(f"  {'slots':>5}  {'makespan':>9}  {'stall':>7}  {'gpu%':>5}  "
              f"{'h2d%':>5}  {'d2h%':>5}  {'offl':>5}  {'reld':>5}  "
              f"{'peakA':>5}  {'gap%':>5}")
        rows_a = []
        for slots in range(1, max(2, base.baseline_peak_act + 1)):
            res = simulate_offload(itin, n, args.window, warmup=wu, nvme=nv0,
                                   **{**kw, "act_slots": slots})
            rows_a.append((slots, res))
            print(f"  {slots:>5}  {res.makespan:>9.1f}  {res.stall:>7.1f}  "
                  f"{100 * res.util('gpu'):>5.0f}  "
                  f"{100 * res.util('h2d'):>5.0f}  "
                  f"{100 * res.util('d2h'):>5.0f}  {res.n_act_offloads:>5}  "
                  f"{res.n_act_reloads:>5}  {res.peak_act:>5}  "
                  f"{100 * res.bound_gap:>5.1f}")
        if args.plot is not None:
            out = args.plot or None
            picks = sorted({1, max(1, base.baseline_peak_act // 2),
                            base.baseline_peak_act})
            plot_gantt(
                [(f"{args.mode} n={n} W={args.window} act_slots={s}", r)
                 for s, r in rows_a if s in picks],
                out_path=out, show=(out is None),
                scale=args.plot_scale, dpi=args.plot_dpi,
            )
        return 0

    if args.window > 0:
        wu = args.window if warmup < 0 else warmup
        results = [
            (label, simulate_offload(itin, n, args.window, warmup=wu,
                                     nvme=nv, **kw))
            for label, nv in placements
        ]
        if args.json:
            payload = [r.metrics() if not lbl else {"placement": lbl,
                                                    **r.metrics()}
                       for lbl, r in results]
            print(json.dumps(payload[0] if len(payload) == 1 else payload,
                             indent=2))
        else:
            for lbl, res in results:
                tag = f"W={args.window}" + (f" {lbl}" if lbl else "")
                print(report(tag, res, args.width))
                if len(results) > 1:
                    print()
        if args.plot is not None:
            out = args.plot or None
            plot_gantt(
                [(f"{args.mode} n={n} W={args.window}"
                  + (f" nvme={args.nvme} {lbl}" if lbl else ""), res)
                 for lbl, res in results],
                out_path=out, show=(out is None),
                scale=args.plot_scale, dpi=args.plot_dpi,
            )
        return 0

    # default: comparison table across window sizes (x placements if --nvme)
    rows = []
    for w in range(1, n + 1):
        wu = w if warmup < 0 else warmup
        for label, nv in placements:
            res = simulate_offload(itin, n, w, warmup=wu, nvme=nv, **kw)
            rows.append((w, label, res))
    if args.json:
        payload = [r.metrics() if not lbl else {"placement": lbl,
                                                **r.metrics()}
                   for _, lbl, r in rows]
        print(json.dumps(payload, indent=2))
    else:
        plc_hdr = f"  {'placement':<11}" if args.nvme else ""
        act_hdr = (f"  {'offl':>5}  {'reld':>5}  {'peakA':>5}"
                   if args.tact > 0 else "")
        print(f"  {'W':>3}{plc_hdr}  {'makespan':>9}  {'stall':>7}  {'gpu%':>5}  "
              f"{'h2d%':>5}  {'d2h%':>5}  {'loads':>5}  {'peak':>4}  "
              f"{'regime':<15}  {'gap%':>5}{act_hdr}")
        for w, lbl, res in rows:
            plc = f"  {lbl:<11}" if args.nvme else ""
            act = (f"  {res.n_act_offloads:>5}  {res.n_act_reloads:>5}  "
                   f"{res.peak_act:>5}" if args.tact > 0 else "")
            print(f"  {w:>3}{plc}  {res.makespan:>9.1f}  {res.stall:>7.1f}  "
                  f"{100 * res.util('gpu'):>5.0f}  {100 * res.util('h2d'):>5.0f}  "
                  f"{100 * res.util('d2h'):>5.0f}  {res.n_loads:>5}  "
                  f"{res.peak_resident:>4}  {res.regime:<15}  "
                  f"{100 * res.bound_gap:>5.1f}{act}")
    if args.plot is not None:
        out = args.plot or None
        picks = sorted({1, max(1, n // 2), n})
        plot_gantt(
            [(f"{args.mode} n={n} W={w}" + (f" {lbl}" if lbl else ""), res)
             for w, lbl, res in rows if w in picks],
            out_path=out, show=(out is None),
            scale=args.plot_scale, dpi=args.plot_dpi,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
