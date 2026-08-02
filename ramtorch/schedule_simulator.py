"""
schedule_simulator.py
---------------------
Fast, executor-free simulator for pipeline schedules.

Given a per-stage op list (the same ``[("F"|"B"|"W", mb), ...]`` lists the
schedule builders produce), this computes the *ideal* timeline each op would
get under the true data dependencies, then reports:

  * makespan             — when the last op finishes
  * per-stage busy/idle  — how much of the makespan each stage is computing
  * bubble               — idle time (the thing 1F1B tries to shrink)
  * peak in-flight mbs   — max microbatches with a stored activation per stage
                           (the memory 1F1B bounds; GPipe does not)

Dependencies modeled (identical to the relay executor):
  F(s, mb) : needs F(s-1, mb) done   [activation from the previous stage]
             (stage 0 needs nothing — its input is the local microbatch)
  B(s, mb) : needs B(s+1, mb) done   [grad from the next stage]
             (last stage needs its own F(s, mb) done — the loss)
  W(s, mb) : zero-cost marker on the last stage; needs F(s, mb) done
  Plus: ops on the same stage run serially, in list order.

Unlike the earlier memoized recurrence (which double-counted the serial chain
and gave wrong makespans), this is a discrete-event simulation: it walks each
stage's list in order and, because deps point both down (F) and up (B) the
pipeline, iterates to a fixed point. Deterministic and cheap.

Usage:
    python -m ramtorch.schedule_simulator                 # built-in comparisons
    python -m ramtorch.schedule_simulator --p 4 --m 8 --tf 1 --tb 2
    # importable:
    from ramtorch.schedule_simulator import simulate, gpipe_ops, onef1b_ops, \
        staggered_1f1b_ops, fwd_grouped_ops, plot_gantt
"""

from __future__ import annotations

import argparse
from typing import Callable, Dict, List, Optional, Tuple

Op = Tuple[str, int]           # ("F"|"B"|"W", microbatch)
RankOps = List[List[Op]]       # per-stage ordered op list
Span = Tuple[float, float]     # (start, end)


# ── Schedule builders (self-contained; mirror ramtorch.pipeline) ──────────────

def gpipe_ops(p: int, m: int) -> RankOps:
    """All forwards, then all backwards, on every stage."""
    ops: RankOps = [[("F", mb) for mb in range(m)] for _ in range(p)]
    ops[p - 1].extend(("W", mb) for mb in range(m))
    for s in range(p):
        ops[s].extend(("B", mb) for mb in range(m))
    return ops


def onef1b_ops(p: int, m: int) -> RankOps:
    """1F1B: warmup forwards, steady F/B, cooldown backwards (per stage)."""
    warmup = min(p - 1, m)
    steady = m - warmup
    ops: RankOps = []
    for s in range(p):
        ro: List[Op] = [("F", mb) for mb in range(warmup)]
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


def staggered_1f1b_ops(p: int, m: int) -> RankOps:
    """
    Staggered-warmup, backward-eager schedule ("flipped 1F1B" / 1B1F).

    Two differences from textbook 1F1B:

    1. **Staggered warmup**: stage ``s`` runs ``(p - 1 - s)`` warmup forwards
       before alternating, so the first stage crams the most microbatches and
       the LAST stage does *zero* warmup. The last stage therefore backwards
       each microbatch immediately after its forward — pulling the gradient
       back up the pipeline as early as possible, which compacts the whole
       sequence.

    2. **Backward-eager steady state**: after warmup, a stage runs its backward
       as soon as the microbatch's forward (and the downstream grad, for
       non-last stages) is available, rather than always doing forward first.

    The op *list* encodes the intent; the executor/simulator resolves the true
    cross-stage timing. Pattern for p=2, m=4:

        s0: F0 F1 .  B0 F2 B1 F3 B2 .  B3
        s1: .  F0 B0 F1 B1 F2 B2 F3 B3
    """
    ops: RankOps = []
    for s in range(p):
        warmup = max(0, min(p - 1 - s, m))
        ro: List[Op] = []
        fwd = 0
        bwd = 0
        # warmup: cram `warmup` forwards
        for _ in range(warmup):
            ro.append(("F", fwd))
            fwd += 1
        # steady + cooldown: emit ops so that a backward is issued as early as
        # its microbatch is ready. We interleave remaining forwards with
        # backwards, backward-eager: after the warmup, each stage does
        #   F(fwd) [if any left] then B(bwd) for the oldest outstanding mb,
        # but crucially the LAST stage (warmup=0) does B right after each F.
        while bwd < m:
            if fwd < m:
                ro.append(("F", fwd))
                fwd += 1
            # backward the oldest mb whose forward we've issued on this stage
            if bwd < fwd:
                if s == p - 1:
                    ro.append(("W", bwd))
                ro.append(("B", bwd))
                bwd += 1
        ops.append(ro)
    return ops


def fwd_grouped_ops(p: int, m: int, warmup: int, group: int) -> RankOps:
    """
    Tunable middle ground between GPipe and 1F1B.

    Run ``warmup`` forwards, then alternate ``group`` forwards with one
    backward, then drain the remaining backwards. Larger ``group`` crams more
    forwards together (fewer F/B alternations) at the cost of more in-flight
    activation memory. ``group=1`` ≈ 1F1B (after warmup); large ``group`` →
    GPipe-like.

    This is a *valid* schedule: every B(mb) is preceded on its stage by F(mb),
    and the executor resolves cross-stage deps by microbatch, so any per-stage
    F/B ordering that keeps F(mb) before B(mb) is runnable.
    """
    ops: RankOps = []
    for s in range(p):
        ro: List[Op] = []
        fwd_done = 0
        bwd_done = 0
        # warmup forwards
        for _ in range(min(warmup, m)):
            ro.append(("F", fwd_done))
            fwd_done += 1
        # steady: `group` forwards, then one backward
        while fwd_done < m:
            for _ in range(group):
                if fwd_done >= m:
                    break
                ro.append(("F", fwd_done))
                fwd_done += 1
            if bwd_done < fwd_done - (0 if s == p - 1 else 0):
                if s == p - 1:
                    ro.append(("W", bwd_done))
                ro.append(("B", bwd_done))
                bwd_done += 1
        # drain remaining backwards
        while bwd_done < m:
            if s == p - 1:
                ro.append(("W", bwd_done))
            ro.append(("B", bwd_done))
            bwd_done += 1
        ops.append(ro)
    return ops


# ── Simulator ─────────────────────────────────────────────────────────────────

class Result:
    def __init__(self, p, m, tf, tb, spans: List[List[Tuple[Op, Span]]]):
        self.p, self.m, self.tf, self.tb = p, m, tf, tb
        # spans[s] = list of ((kind, mb), (start, end)) for non-W ops, in order
        self.spans = spans
        self.makespan = max((en for s in spans for _, (_, en) in s), default=0.0)

    # -- metrics --
    def busy(self, s: int) -> float:
        return sum(en - st for _, (st, en) in self.spans[s])

    def idle(self, s: int) -> float:
        return self.makespan - self.busy(s)

    def bubble_frac(self) -> float:
        total = self.makespan * self.p
        busy = sum(self.busy(s) for s in range(self.p))
        return 0.0 if total == 0 else (total - busy) / total

    def peak_inflight(self, s: int) -> int:
        """Max microbatches whose activation is stored but not yet backwarded.

        An activation for mb is "live" on stage s from F(s,mb).end until
        B(s,mb).start. This is the memory 1F1B bounds.
        """
        f_end: Dict[int, float] = {}
        b_start: Dict[int, float] = {}
        for (kind, mb), (st, en) in self.spans[s]:
            if kind == "F":
                f_end[mb] = en
            elif kind == "B":
                b_start[mb] = st
        # sweep events
        events = []
        for mb in f_end:
            events.append((f_end[mb], +1))
            if mb in b_start:
                events.append((b_start[mb], -1))
        events.sort(key=lambda x: (x[0], -x[1]))  # ends before starts at same t
        cur = peak = 0
        for _, d in events:
            cur += d
            peak = max(peak, cur)
        return peak


def simulate(
    rank_ops: RankOps,
    tf: float = 1.0,
    tb: float = 1.0,
) -> Result:
    """
    Compute the ideal (start, end) span of every op via fixed-point iteration.

    start(s, i) = max(end of op i-1 on stage s, deps_ready)
      deps_ready(F) = end(F, s-1, mb)            (0 for stage 0)
      deps_ready(B) = end(B, s+1, mb)            (end(F, s, mb) for last stage)
      deps_ready(W) = end(F, s, mb)
    end = start + (tf if F else tb if B else 0)

    Because F deps flow down and B deps flow up, iterate sweeps until stable.
    """
    p = len(rank_ops)
    m = 1 + max((mb for ops in rank_ops for _, mb in ops), default=-1)

    # index of each (kind, mb) within a stage's list
    pos: List[Dict[Tuple[str, int], int]] = []
    for ops in rank_ops:
        d = {}
        for i, (k, mb) in enumerate(ops):
            d[(k, mb)] = i
        pos.append(d)

    # end times; start times derived. None = not yet computed.
    end: List[List[Optional[float]]] = [
        [None] * len(ops) for ops in rank_ops
    ]
    start: List[List[Optional[float]]] = [
        [None] * len(ops) for ops in rank_ops
    ]

    def dep_ready(s: int, i: int) -> float:
        kind, mb = rank_ops[s][i]
        if kind == "F":
            if s == 0:
                return 0.0
            j = pos[s - 1].get(("F", mb))
            return 0.0 if j is None or end[s - 1][j] is None else end[s - 1][j]
        if kind == "W":
            j = pos[s].get(("F", mb))
            return 0.0 if j is None or end[s][j] is None else end[s][j]
        # B
        if s == p - 1:
            j = pos[s].get(("F", mb))
            return 0.0 if j is None or end[s][j] is None else end[s][j]
        j = pos[s + 1].get(("B", mb))
        return 0.0 if j is None or end[s + 1][j] is None else end[s + 1][j]

    dur = {"F": tf, "B": tb, "W": 0.0}

    # Fixed-point: an op is "computable" when all the end-times it reads are set.
    def computable(s: int, i: int) -> bool:
        kind, mb = rank_ops[s][i]
        if i > 0 and end[s][i - 1] is None:
            return False
        if kind == "F":
            return s == 0 or end[s - 1][pos[s - 1][("F", mb)]] is not None
        if kind == "W":
            return end[s][pos[s][("F", mb)]] is not None
        if s == p - 1:
            return end[s][pos[s][("F", mb)]] is not None
        return end[s + 1][pos[s + 1][("B", mb)]] is not None

    remaining = sum(len(ops) for ops in rank_ops)
    guard = remaining * (remaining + 1)  # generous iteration cap
    while remaining and guard > 0:
        progressed = False
        for s in range(p):
            for i in range(len(rank_ops[s])):
                if end[s][i] is None and computable(s, i):
                    prev_end = end[s][i - 1] if i > 0 else 0.0
                    st = max(prev_end, dep_ready(s, i))
                    kind = rank_ops[s][i][0]
                    start[s][i] = st
                    end[s][i] = st + dur[kind]
                    remaining -= 1
                    progressed = True
        if not progressed:
            raise RuntimeError("schedule has a cyclic dependency (dead schedule)")
        guard -= 1

    spans: List[List[Tuple[Op, Span]]] = []
    for s in range(p):
        lst = []
        for i, (k, mb) in enumerate(rank_ops[s]):
            if k == "W":
                continue
            lst.append(((k, mb), (start[s][i], end[s][i])))
        spans.append(lst)
    return Result(p, m, tf, tb, spans)


# ── ASCII Gantt ───────────────────────────────────────────────────────────────

def gantt(res: Result, width: int = 100) -> str:
    """Render one row per stage; F = forward, b = backward, . = idle bubble."""
    ms = res.makespan
    if ms <= 0:
        return "(empty)"
    cols = max(10, width)
    lines = []
    for s in range(res.p):
        row = ["."] * cols
        for (kind, mb), (st, en) in res.spans[s]:
            a = int(st / ms * cols)
            b = max(a + 1, int(en / ms * cols))
            for c in range(a, min(b, cols)):
                row[c] = "F" if kind == "F" else "b"
        lines.append(f"  s{s} |{''.join(row)}|")
    scale = "".join(
        str(int(t / 10 % 10)) if t % 10 == 0 else ("|" if t % 5 == 0 else " ")
        for t in range(cols)
    )
    lines.append(f"     +{scale}+  0..{ms:.1f}")
    return "\n".join(lines)


# ── Matplotlib Gantt ──────────────────────────────────────────────────────────

def plot_gantt(
    results: List[Tuple[str, "Result"]],
    serial: float,
    out_path: Optional[str] = None,
    show: bool = True,
):
    """
    Draw one Gantt subplot per schedule. Rows = stages; each op is a labeled bar
    (F = forward, B = backward), idle gaps are the bubble. Bar labels show the
    op kind + microbatch so the execution order is readable directly.
    """
    import matplotlib
    if not show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n = len(results)
    maxp = max(r.p for _, r in results)
    fig, axes = plt.subplots(
        n, 1, figsize=(max(10, max(r.makespan for _, r in results) * 0.5), 2.2 * n),
        sharex=True, squeeze=False,
    )
    colors = {"F": "#4C9BE0", "B": "#E0764C"}  # fwd blue, bwd orange

    for ax, (name, res) in zip(axes[:, 0], results):
        for s in range(res.p):
            y = res.p - 1 - s  # stage 0 on top
            for (kind, mb), (st, en) in res.spans[s]:
                ax.barh(y, en - st, left=st, height=0.7,
                        color=colors[kind], edgecolor="white", linewidth=0.4)
                label = f"{kind}{mb}"
                # only label if the bar is wide enough to read
                if en - st >= res.makespan * 0.02:
                    ax.text((st + en) / 2, y, label, ha="center", va="center",
                            fontsize=7, color="white", weight="bold")
        ax.set_yticks(range(res.p))
        ax.set_yticklabels([f"s{res.p - 1 - i}" for i in range(res.p)])
        ax.set_ylim(-0.6, res.p - 0.4)
        ax.grid(axis="x", alpha=0.25)
        ax.set_title(
            f"{name}   makespan={res.makespan:.0f}  "
            f"bubble={100*res.bubble_frac():.0f}%  "
            f"speedup={serial/res.makespan:.2f}x  "
            f"peak_mem={max(res.peak_inflight(s) for s in range(res.p))}",
            loc="left", fontsize=10,
        )
    axes[-1, 0].set_xlabel("time")
    handles = [Patch(facecolor=colors["F"], label="forward"),
               Patch(facecolor=colors["B"], label="backward")]
    axes[0, 0].legend(handles=handles, loc="upper right", fontsize=8)
    fig.suptitle("Pipeline schedule execution order", y=0.995)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        print(f"wrote {out_path}")
    if show:
        plt.show()
    return fig


# ── CLI / demo ────────────────────────────────────────────────────────────────

def report(name: str, res: Result, serial: float) -> str:
    out = [f"  {name:<10} makespan={res.makespan:5.1f}  "
           f"bubble={100*res.bubble_frac():4.1f}%  "
           f"speedup_vs_serial={serial/res.makespan:4.2f}x  "
           f"peak_inflight={[res.peak_inflight(s) for s in range(res.p)]}"]
    out.append(gantt(res))
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--p", type=int, default=4, help="pipeline depth (stages)")
    ap.add_argument("--m", type=int, default=8, help="microbatches")
    ap.add_argument("--tf", type=float, default=1.0, help="forward cost")
    ap.add_argument("--tb", type=float, default=1.0, help="backward cost")
    ap.add_argument("--width", type=int, default=100, help="gantt width (cols)")
    ap.add_argument("--group", type=int, default=0,
                    help="if >0, also simulate fwd_grouped with this fwd-per-bwd group")
    ap.add_argument("--warmup", type=int, default=0,
                    help="warmup forwards for fwd_grouped (default: p-1)")
    ap.add_argument("--plot", type=str, nargs="?", const="", default=None,
                    help="render a matplotlib Gantt; optionally give an output .png path")
    args = ap.parse_args()
    p, m, tf, tb = args.p, args.m, args.tf, args.tb

    serial = m * p * (tf + tb)
    print(f"p={p} stages, m={m} microbatches, tf={tf}, tb={tb}  (serial={serial:.0f})\n")

    schedules: List[Tuple[str, RankOps]] = [
        ("gpipe", gpipe_ops(p, m)),
        ("1f1b", onef1b_ops(p, m)),
    ]
    if args.group > 0:
        w = args.warmup if args.warmup > 0 else p - 1
        schedules.append((f"grp{args.group}/w{w}", fwd_grouped_ops(p, m, w, args.group)))

    results: List[Tuple[str, Result]] = []
    for name, ops in schedules:
        res = simulate(ops, tf=tf, tb=tb)
        results.append((name, res))
        print(report(name, res, serial))
        print()

    if args.plot is not None:
        out = args.plot or None
        plot_gantt(results, serial, out_path=out, show=(out is None))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
