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

Itineraries:
  * Training (echo): F0..F(N-1), B(N-1)..B0, repeatable for multiple steps.
    The turnaround is self-warming — the window holds the last W layers
    exactly when backward starts, and the first W layers when the next step's
    forward starts.
  * Inference (ring): F0..F(N-1) repeated; after F_i the layer's next use is
    the next pass, so the prefetcher wraps and streams 0, 1, 2, ... while the
    tail layers compute.

Out of scope: activation memory (orthogonal — checkpointing), optimizer step,
multi-GPU composition.

Usage:
    python -m ramtorch.offload_simulator                     # window sweep table
    python -m ramtorch.offload_simulator --layers 10 --window 5 --mode train
    python -m ramtorch.offload_simulator --layers 10 --window 5 --plot out.png
    # importable:
    from ramtorch.offload_simulator import (
        train_itinerary, infer_itinerary, simulate_offload, gantt, plot_gantt)

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


# ── Result ────────────────────────────────────────────────────────────────────

class OffloadResult:
    """Timeline + metrics from one ``simulate_offload`` run.

    ``spans`` maps channel name -> ordered list of ``((kind, layer), (st, en))``:
      * ``"gpu"``: kind "F" or "B" (compute)
      * ``"h2d"``: kind "L" (weight load)
      * ``"d2h"``: kind "G" (grad writeback)

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
    ):
        self.itinerary = itinerary
        self.n_layers = n_layers
        self.window = window
        self.warmup = warmup
        self.pinned = frozenset(pinned)
        self.tf, self.tb, self.th2d, self.td2h = tf, tb, th2d, td2h
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

    # -- analytic lower bounds --
    @property
    def total_compute(self) -> float:
        nf = sum(1 for k, _ in self.itinerary if k == "F")
        nb = sum(1 for k, _ in self.itinerary if k == "B")
        return nf * self.tf + nb * self.tb

    @property
    def compute_bound(self) -> float:
        """Serial compute + the first layer's load latency (0 if pinned)."""
        if not self.itinerary:
            return 0.0
        first = 0.0 if self.itinerary[0][1] in self.pinned else self.th2d
        return first + self.total_compute

    @property
    def total_mem(self) -> int:
        """GPU slots this config consumes: streaming window + pinned layers."""
        return self.window + len(self.pinned)

    @property
    def transfer_bound(self) -> float:
        """Each distinct unpinned layer loads at least once; writebacks serialize."""
        distinct = len({l for _, l in self.itinerary} - self.pinned)
        nb = sum(1 for k, _ in self.itinerary if k == "B")
        d2h_total = nb * self.td2h if self.td2h > 0 else 0.0
        return max(distinct * self.th2d, d2h_total)

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
        return {
            "n_layers": self.n_layers,
            "window": self.window,
            "warmup": self.warmup,
            "pinned": sorted(self.pinned),
            "total_mem": self.total_mem,
            "ops": len(self.itinerary),
            "makespan": self.makespan,
            "stall": self.stall,
            "gpu_util": self.util("gpu"),
            "h2d_util": self.util("h2d"),
            "d2h_util": self.util("d2h"),
            "n_loads": self.n_loads,
            "peak_resident": self.peak_resident,
            "compute_bound": self.compute_bound,
            "transfer_bound": self.transfer_bound,
            "regime": self.regime,
            "bound_gap": self.bound_gap,
        }


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
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    pinned = frozenset(pinned or ())
    for l in pinned:
        if not (0 <= l < n_layers):
            raise ValueError(f"pinned layer {l} out of range [0, {n_layers})")
    for k, l in itinerary:
        if k not in ("F", "B"):
            raise ValueError(f"bad op kind {k!r}")
        if not (0 <= l < n_layers):
            raise ValueError(f"layer {l} out of range [0, {n_layers})")
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

    resident: set = set(pinned)
    comp_idx = 0                       # next itinerary op to run
    gpu_cur: Optional[Tuple[str, int, float]] = None   # (kind, layer, start)
    gpu_end = 0.0
    h2d_cur: Optional[Tuple[int, float]] = None        # (layer, start)
    h2d_end = 0.0
    d2h_cur: Optional[Tuple[int, float]] = None        # (layer, start)
    d2h_end = 0.0
    d2h_q: deque = deque()             # (layer, ready_time), FIFO

    spans: Dict[str, List[Tuple[Tuple[str, int], Span]]] = {
        "gpu": [], "h2d": [], "d2h": []
    }
    resident_at: List[FrozenSet[int]] = []
    occ_timeline: List[Tuple[float, int]] = [(0.0, len(pinned))]
    n_loads = 0
    now = 0.0

    def occupancy() -> int:
        return len(resident) + (1 if h2d_cur is not None else 0)

    def record_occ(t: float) -> None:
        occ_timeline.append((t, occupancy()))

    while True:
        # 1) process completions due at `now`
        if gpu_cur is not None and gpu_end <= now + _EPS:
            kind, layer, st = gpu_cur
            spans["gpu"].append(((kind, layer), (st, gpu_end)))
            if kind == "B" and td2h > 0:
                d2h_q.append((layer, gpu_end))
            gpu_cur = None
        if h2d_cur is not None and h2d_end <= now + _EPS:
            layer, st = h2d_cur
            spans["h2d"].append((("L", layer), (st, h2d_end)))
            h2d_cur = None
            resident.add(layer)
            record_occ(now)
        if d2h_cur is not None and d2h_end <= now + _EPS:
            layer, st = d2h_cur
            spans["d2h"].append((("G", layer), (st, d2h_end)))
            d2h_cur = None

        # 2) start whatever can start at `now` (loop back for zero-cost chains)
        progressed = False

        # compute (checked before H2D so resident_at snapshots precede
        # any same-instant eviction)
        if gpu_cur is None and comp_idx < n_ops:
            kind, layer = itinerary[comp_idx]
            warmed = comp_idx > 0 or len(spans["h2d"]) >= warmup
            if layer in resident and warmed:
                resident_at.append(frozenset(resident))
                dur = tf if kind == "F" else tb
                gpu_cur = (kind, layer, now)
                gpu_end = now + dur
                comp_idx += 1
                progressed = True

        # grad writeback
        if d2h_cur is None and d2h_q and d2h_q[0][1] <= now + _EPS:
            layer, _ = d2h_q.popleft()
            d2h_cur = (layer, now)
            d2h_end = now + td2h
            progressed = True

        # prefetch: earliest itinerary layer not yet resident
        if h2d_cur is None:
            cand: Optional[Tuple[int, int]] = None  # (layer, needed_pos)
            for pos in range(comp_idx, n_ops):
                l = itinerary[pos][1]
                if l not in resident:
                    cand = (l, pos)
                    break
            if cand is not None:
                l, pos = cand
                can_start = occupancy() < capacity
                if not can_start:
                    # Belady eviction: farthest next use, never the layer
                    # compute is running and never a pinned layer, only if
                    # farther than the candidate
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
                    h2d_cur = (l, now)
                    h2d_end = now + th2d
                    n_loads += 1
                    record_occ(now)
                    progressed = True

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
                f"window={window}) — bug"
            )
        now = min(pending)

    return OffloadResult(
        itinerary, n_layers, window, tf, tb, th2d, td2h,
        spans, resident_at, occ_timeline, n_loads, warmup=warmup,
        pinned=pinned,
    )


# ── ASCII Gantt ───────────────────────────────────────────────────────────────

_ROW_CHAR = {"F": "F", "B": "b", "L": "L", "G": "G"}


def gantt(res: OffloadResult, width: int = 100) -> str:
    """Three rows (gpu / h2d / d2h); '.' = idle."""
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
):
    """One Gantt (gpu/h2d/d2h rows) + residency strip per result.

    Bars are labeled ``F3 / B3 / L3 / G3`` (op kind + layer index); the strip
    below each Gantt steps through occupied window slots over time.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Okabe-Ito colorblind-safe palette; the backward hatch adds a
    # color-independent cue vs the H2D loads (the confusable pair).
    colors = {"F": "#0072B2", "B": "#E69F00", "L": "#009E73", "G": "#CC79A7"}
    hatches = {"F": "", "B": "//", "L": "", "G": ""}
    kind_name = {"F": "forward", "B": "backward", "L": "H2D load", "G": "D2H grad"}
    n = len(results)
    max_ms = max((r.makespan for _, r in results), default=1.0)
    fig, axes = plt.subplots(
        2 * n, 1,
        figsize=(max(10, max_ms * 0.45), 3.2 * n),
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
                    ax.text((st + en) / 2, y, f"{kind}{layer}", ha="center",
                            va="center", fontsize=7, color="white",
                            weight="bold", bbox=bbox)
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([rows[len(rows) - 1 - i] for i in range(len(rows))])
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(
            f"{name}   makespan={res.makespan:.1f}  stall={res.stall:.1f}  "
            f"loads={res.n_loads}  peak_res={res.peak_resident}/{res.total_mem}"
            + (f" (pinned {len(res.pinned)})" if res.pinned else "")
            + f"  {res.regime} (gap {100 * res.bound_gap:.0f}%)",
            loc="left", fontsize=10,
        )
        # residency strip
        ts = [t for t, _ in res.occupancy_timeline] + [res.makespan]
        os_ = [o for _, o in res.occupancy_timeline]
        os_ = os_ + [os_[-1] if os_ else 0]
        axr.step(ts, os_, where="post", color="#555", linewidth=1.2)
        axr.fill_between(ts, os_, step="post", alpha=0.25, color="#888")
        axr.axhline(res.total_mem, color="#C0392B", linestyle="--", linewidth=0.8)
        axr.set_ylim(0, res.total_mem + 1)
        axr.set_ylabel("slots", fontsize=8)
        axr.grid(axis="x", alpha=0.25)

    axes[-1, 0].set_xlabel("time")
    handles = [
        Patch(facecolor=c, hatch=hatches[k], edgecolor="white",
              label=kind_name[k])
        for k, c in colors.items()
    ]
    axes[0, 0].legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle("Windowed CPU->GPU offload streaming schedule", y=0.995)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
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
    ap.add_argument("--width", type=int, default=100, help="ascii gantt width")
    ap.add_argument("--plot", type=str, nargs="?", const="", default=None,
                    help="render a matplotlib Gantt; optionally give a .png path")
    ap.add_argument("--json", action="store_true", help="print metrics as JSON")
    args = ap.parse_args()

    n = args.layers
    itin = _build(args.mode, n, args.steps)
    warmup = args.warmup
    kw = dict(tf=args.tf, tb=args.tb, th2d=args.th2d, td2h=args.td2h,
              pinned=evenly_pinned(n, args.pin))
    print(
        f"n={n} layers, mode={args.mode}, steps={args.steps}, "
        f"tf={args.tf} tb={args.tb} th2d={args.th2d} td2h={args.td2h}\n"
    )

    if args.window > 0:
        wu = args.window if warmup < 0 else warmup
        res = simulate_offload(itin, n, args.window, warmup=wu, **kw)
        if args.json:
            print(json.dumps(res.metrics(), indent=2))
        else:
            print(report(f"W={args.window}", res, args.width))
        if args.plot is not None:
            out = args.plot or None
            plot_gantt([(f"{args.mode} n={n} W={args.window}", res)],
                       out_path=out, show=(out is None))
        return 0

    # default: comparison table across window sizes
    rows = []
    for w in range(1, n + 1):
        wu = w if warmup < 0 else warmup
        res = simulate_offload(itin, n, w, warmup=wu, **kw)
        rows.append((w, res))
    if args.json:
        print(json.dumps([r.metrics() for _, r in rows], indent=2))
    else:
        print(f"  {'W':>3}  {'makespan':>9}  {'stall':>7}  {'gpu%':>5}  "
              f"{'h2d%':>5}  {'d2h%':>5}  {'loads':>5}  {'peak':>4}  "
              f"{'regime':<15}  {'gap%':>5}")
        for w, res in rows:
            print(f"  {w:>3}  {res.makespan:>9.1f}  {res.stall:>7.1f}  "
                  f"{100 * res.util('gpu'):>5.0f}  {100 * res.util('h2d'):>5.0f}  "
                  f"{100 * res.util('d2h'):>5.0f}  {res.n_loads:>5}  "
                  f"{res.peak_resident:>4}  {res.regime:<15}  "
                  f"{100 * res.bound_gap:>5.1f}")
    if args.plot is not None:
        out = args.plot or None
        picks = sorted({1, max(1, n // 2), n})
        plot_gantt([(f"{args.mode} n={n} W={w}", res)
                    for w, res in rows if w in picks],
                   out_path=out, show=(out is None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
