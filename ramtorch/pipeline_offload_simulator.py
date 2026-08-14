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

Simplifications (deliberate): activation hops cost ``comm`` latency but do NOT
occupy the h2d/d2h channels (the premise: boundary activations are small next
to weights); host DDR bandwidth shared across GPUs is not modeled; the
optimizer step is out of scope (same as ``offload_simulator``).

Usage:
    python -m ramtorch.pipeline_offload_simulator                  # W sweep
    python -m ramtorch.pipeline_offload_simulator --window 2
    python -m ramtorch.pipeline_offload_simulator --window 2 --plot out.png
    python -m ramtorch.pipeline_offload_simulator --schedule gpipe --th2d 3
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
      * ``"h2d"``: op = ``("L", chunk)`` weight load
      * ``"d2h"``: op = ``("G", chunk)`` grad writeback
    """

    CHANNELS = ("gpu", "h2d", "d2h")

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
    ):
        self.p, self.m, self.chunks, self.window = p, m, chunks, window
        self.tf, self.tb, self.th2d, self.td2h, self.comm = tf, tb, th2d, td2h, comm
        self.pinned = frozenset(pinned)
        self.itins = itins
        self.spans = spans
        self.n_loads = n_loads
        self.peak_resident = peak_resident
        self.ref_makespan = ref_makespan
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
    def overhead_vs_ref(self) -> float:
        """(makespan - no-offload pipeline makespan) / no-offload makespan."""
        if self.ref_makespan == 0:
            return 0.0
        return (self.makespan - self.ref_makespan) / self.ref_makespan

    def metrics(self) -> Dict[str, object]:
        return {
            "p": self.p,
            "m": self.m,
            "chunks": self.chunks,
            "window": self.window,
            "pinned": sorted(self.pinned),
            "total_mem": self.total_mem,
            "makespan": self.makespan,
            "ref_makespan": self.ref_makespan,
            "overhead_vs_ref": self.overhead_vs_ref,
            "gpu_util": [self.util(s, "gpu") for s in range(self.p)],
            "h2d_util": [self.util(s, "h2d") for s in range(self.p)],
            "d2h_util": [self.util(s, "d2h") for s in range(self.p)],
            "loads": list(self.n_loads),
            "naive_loads": [self.naive_loads(s) for s in range(self.p)],
            "peak_resident": list(self.peak_resident),
        }


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
) -> PipelineOffloadResult:
    """Discrete-event simulation: p offload state machines + cross-stage deps.

    ``tf``/``tb``/``th2d``/``td2h`` are PER-CHUNK costs (a stage-level forward
    costs ``chunks * tf``). ``pinned`` chunk indices (same set on every stage)
    are permanently GPU-resident on top of the ``window`` streaming slots.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    pinned = frozenset(pinned or ())
    for c in pinned:
        if not (0 <= c < chunks):
            raise ValueError(f"pinned chunk {c} out of range [0, {chunks})")

    p = len(rank_ops)
    itins, firsts, lasts = expand_rank_ops(rank_ops, chunks)
    m = 1 + max((mb for itin in itins for _, mb, _ in itin), default=-1)
    n_ops = [len(itin) for itin in itins]
    capacity = window + len(pinned)

    # chunk -> itinerary positions, per stage (Belady next-use queries)
    uses: List[List[List[int]]] = [
        [[] for _ in range(chunks)] for _ in range(p)
    ]
    for s, itin in enumerate(itins):
        for i, (_, _, c) in enumerate(itin):
            uses[s][c].append(i)

    def next_use(s: int, chunk: int, from_pos: int) -> float:
        us = uses[s][chunk]
        j = bisect.bisect_left(us, from_pos)
        return us[j] if j < len(us) else INF

    # per-stage channel state
    resident: List[set] = [set(pinned) for _ in range(p)]
    comp_idx = [0] * p
    # gpu_cur[s] = (itin_index, start); h2d_cur[s] = (chunk, start);
    # d2h_cur[s] = (chunk, start)
    gpu_cur: List[Optional[Tuple[int, float]]] = [None] * p
    gpu_end = [0.0] * p
    h2d_cur: List[Optional[Tuple[int, float]]] = [None] * p
    h2d_end = [0.0] * p
    d2h_cur: List[Optional[Tuple[int, float]]] = [None] * p
    d2h_end = [0.0] * p
    d2h_q: List[deque] = [deque() for _ in range(p)]  # (chunk, ready_time)

    spans: List[Dict[str, List[Tuple[Tuple, Span]]]] = [
        {"gpu": [], "h2d": [], "d2h": []} for _ in range(p)
    ]
    n_loads = [0] * p
    peak_res = [len(pinned) for _ in range(p)]
    # (stage, kind, mb) -> end time of that stage-op's LAST chunk
    done: Dict[Tuple[int, str, int], float] = {}
    now = 0.0

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
                if kind == "B" and td2h > 0:
                    d2h_q[s].append((c, gpu_end[s]))
                gpu_cur[s] = None
            if h2d_cur[s] is not None and h2d_end[s] <= now + _EPS:
                c, st = h2d_cur[s]
                spans[s]["h2d"].append((("L", c), (st, h2d_end[s])))
                h2d_cur[s] = None
                resident[s].add(c)
            if d2h_cur[s] is not None and d2h_end[s] <= now + _EPS:
                c, st = d2h_cur[s]
                spans[s]["d2h"].append((("G", c), (st, d2h_end[s])))
                d2h_cur[s] = None

        # 2) start whatever can start at `now` (fixed point for zero-cost ops)
        progressed = False
        for s in range(p):
            # compute
            if gpu_cur[s] is None and comp_idx[s] < n_ops[s]:
                i = comp_idx[s]
                kind, _, c = itins[s][i]
                dep = dep_ready(s, i)
                if (c in resident[s] and dep is not None
                        and dep <= now + _EPS):
                    gpu_cur[s] = (i, now)
                    gpu_end[s] = now + (tf if kind == "F" else tb)
                    comp_idx[s] += 1
                    progressed = True

            # grad writeback
            if d2h_cur[s] is None and d2h_q[s] and d2h_q[s][0][1] <= now + _EPS:
                c, _ = d2h_q[s].popleft()
                d2h_cur[s] = (c, now)
                d2h_end[s] = now + td2h
                progressed = True

            # prefetch: earliest itinerary chunk not resident on this stage
            if h2d_cur[s] is None:
                cand: Optional[Tuple[int, int]] = None  # (chunk, needed_pos)
                for pos in range(comp_idx[s], n_ops[s]):
                    c = itins[s][pos][2]
                    if c not in resident[s]:
                        cand = (c, pos)
                        break
                if cand is not None:
                    c, pos = cand
                    can_start = len(resident[s]) < capacity
                    if not can_start:
                        # Belady: evict the farthest-next-use resident chunk,
                        # never the one compute is running, never pinned, and
                        # only if farther than the candidate's need
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
                        h2d_cur[s] = (c, now)
                        h2d_end[s] = now + th2d
                        n_loads[s] += 1
                        peak_res[s] = max(peak_res[s], len(resident[s]) + 1)
                        progressed = True

        if progressed:
            continue

        # 3) done, or advance time to the next event
        work_left = any(
            comp_idx[s] < n_ops[s]
            or gpu_cur[s] is not None
            or h2d_cur[s] is not None
            or d2h_cur[s] is not None
            or d2h_q[s]
            for s in range(p)
        )
        if not work_left:
            break
        pending: List[float] = []
        for s in range(p):
            for cur, end in ((gpu_cur[s], gpu_end[s]),
                             (h2d_cur[s], h2d_end[s]),
                             (d2h_cur[s], d2h_end[s])):
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
                        resident=sorted(resident[s]))
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
    )


# ── ASCII Gantt ───────────────────────────────────────────────────────────────

_ROW_CHAR = {"F": "F", "B": "b", "L": "L", "G": "G"}


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
):
    """One subplot per result; 3 rows per stage (gpu / h2d / d2h).

    Compute bars are labeled ``F{mb}c{chunk}`` / ``B{mb}c{chunk}``, loads
    ``L{chunk}``, grad writebacks ``G{chunk}``. Idle gaps on the gpu rows are
    pipeline bubble + offload stall; idle on h2d rows is spare PCIe headroom.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colors = {"F": "#0072B2", "B": "#E69F00", "L": "#009E73", "G": "#CC79A7"}
    hatches = {"F": "", "B": "//", "L": "", "G": ""}
    kind_name = {"F": "forward", "B": "backward", "L": "H2D weight load",
                 "G": "D2H grad writeback"}

    n = len(results)
    max_ms = max((r.makespan for _, r in results), default=1.0)
    heights = [3 * r.p * 0.42 + 0.9 for _, r in results]
    fig, axes = plt.subplots(
        n, 1,
        figsize=(min(48, max(12, max_ms * 0.22)), sum(heights)),
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
                        label = (f"{kind}{op[1]}c{op[2]}" if ch == "gpu"
                                 else f"{kind}{op[1]}")
                        bbox = (
                            dict(facecolor=colors[kind], edgecolor="none",
                                 pad=0.4)
                            if hatches[kind] else None
                        )
                        ax.text((st + en) / 2, y, label, ha="center",
                                va="center", fontsize=5.5, color="white",
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
            f"  peak_res={max(res.peak_resident)}",
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
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
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
        out.append(
            f"    s{s}: gpu={100 * res.util(s, 'gpu'):3.0f}%"
            f"  h2d={100 * res.util(s, 'h2d'):3.0f}%"
            f"  d2h={100 * res.util(s, 'd2h'):3.0f}%"
            f"  loads={res.n_loads[s]:4d}/{mt['naive_loads'][s]:4d} naive"
            f"  peak_res={res.peak_resident[s]}"
        )
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
    ap.add_argument("--schedule", choices=sorted(_BUILDERS), default="1b1f")
    ap.add_argument("--width", type=int, default=120, help="ascii gantt width")
    ap.add_argument("--plot", type=str, nargs="?", const="", default=None,
                    help="render a matplotlib Gantt; optionally a .png path")
    args = ap.parse_args()

    p, m, L = args.p, args.m, args.chunks
    rank_ops = _BUILDERS[args.schedule](p, m)
    pinned = evenly_pinned(L, args.pin)
    kw = dict(tf=args.tf, tb=args.tb, th2d=args.th2d, td2h=args.td2h,
              comm=args.comm, pinned=pinned)
    print(
        f"schedule={args.schedule} p={p} m={m} chunks={L} "
        f"tf={args.tf} tb={args.tb} th2d={args.th2d} td2h={args.td2h} "
        f"comm={args.comm} pin={args.pin}\n"
    )

    if args.window > 0:
        res = simulate_pipeline_offload(rank_ops, L, args.window, **kw)
        print(report(f"W={args.window}", res, args.width))
        if args.plot is not None:
            out = args.plot or None
            plot_gantt(
                [(f"{args.schedule} p={p} m={m} L={L} W={args.window}", res)],
                out_path=out, show=(out is None),
            )
        return 0

    # sweep table over window sizes
    rows = []
    for w in range(1, L + 1):
        rows.append((w, simulate_pipeline_offload(rank_ops, L, w, **kw)))
    print(f"  {'W':>3}  {'makespan':>9}  {'overhead':>8}  {'gpu% (per stage)':<20}"
          f"  {'h2d%max':>7}  {'loads':>6}  {'naive':>6}  {'peak':>4}")
    for w, res in rows:
        gpu_utils = "/".join(f"{100 * res.util(s, 'gpu'):.0f}"
                             for s in range(res.p))
        print(f"  {w:>3}  {res.makespan:>9.1f}  "
              f"{100 * res.overhead_vs_ref:>7.1f}%  {gpu_utils:<20}  "
              f"{100 * max(res.util(s, 'h2d') for s in range(res.p)):>7.0f}"
              f"  {sum(res.n_loads):>6}  "
              f"{sum(res.naive_loads(s) for s in range(res.p)):>6}  "
              f"{max(res.peak_resident):>4}")
    if args.plot is not None:
        out = args.plot or None
        picks = sorted({1, max(1, L // 2), L})
        plot_gantt(
            [(f"{args.schedule} p={p} m={m} L={L} W={w}", res)
             for w, res in rows if w in picks],
            out_path=out, show=(out is None),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
