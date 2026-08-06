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
    python -m ramtorch.schedule_simulator --p 4 --m 8 --v 4 --comm 0.1  # interleaved
    # importable:
    from ramtorch.schedule_simulator import simulate, gpipe_ops, onef1b_ops, \
        staggered_1f1b_ops, interleaved_1b1f_ops, fwd_grouped_ops, plot_gantt
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


def _interleaved_schedule_table(m: int, v: int, group: int) -> List[Tuple[int, int]]:
    """virtual_microbatch_id -> (microbatch_id, model_chunk_id), forward direction.

    Built in groups of ``group`` microbatches; within a group, for each chunk
    (ascending) emit every microbatch in the group. Mirrors Megatron's
    ``get_schedule_table``. Backwards use the same table with chunk reversed
    (``v - 1 - chunk``).
    """
    table: List[Tuple[int, int]] = []
    for g0 in range(0, m, group):
        g1 = min(g0 + group, m)
        for c in range(v):
            for mb in range(g0, g1):
                table.append((mb, c))
    return table


def interleaved_1b1f_ops(p: int, m: int, v: int, group: Optional[int] = None) -> RankOps:
    """
    Interleaved 1F1B (Megatron-style virtual pipeline), faithful port.

    Each physical stage ``s`` owns ``v`` virtual chunks; chunk ``c`` sits at
    virtual rank ``vr = c*p + s``, so the logical pipeline is ``p*v`` deep.
    Interleaving shrinks the bubble (~1/v) at the cost of more cross-device
    activation/grad hops (``comm`` in ``simulate``) and more in-flight
    activation memory (each chunk's microbatches stay live independently).

    Structure (per stage ``s``, ``n = m*v`` virtual units):

    * **warmup**: ``w_s = (p-1-s)*2 + (v-1)*group`` forwards (earlier stages
      warm up more; later stages start 1F1B sooner).
    * **steady**: ``F(k + w_s)`` then ``B(k)`` — one forward, one backward.
    * **cooldown**: drain the remaining ``w_s`` backwards.

    The forward chunk for virtual id ``k`` comes from the schedule table;
    the backward chunk is reversed (``v-1 - chunk``) because gradients flow
    from high to low virtual rank. ``W`` markers only on the final virtual
    rank (``chunk == v-1, s == p-1``).

    ``group`` is the microbatch group size per virtual stage (Megatron's
    ``microbatch_group_size_per_vp_stage``); defaults to ``p`` when ``m`` is a
    multiple of ``p``, else ``m`` (a single group — required for feasibility
    when the grouping would leave a misaligned partial group).

    NOTE: each virtual F/B costs ``tf/v`` / ``tb/v`` (a chunk holds ``1/v`` of
    the stage's layers). ``simulate`` applies this scaling automatically when
    the op list carries chunks.
    """
    if v < 1:
        raise ValueError("v must be >= 1")
    if group is None:
        group = p if m % p == 0 else m
    n = m * v
    table = _interleaved_schedule_table(m, v, group)
    assert len(table) == n
    ops: RankOps = []
    for s in range(p):
        w = min(n, (p - 1 - s) * 2 + (v - 1) * group)
        ro: List[Op] = []
        # warmup forwards
        for k in range(w):
            mb, c = table[k]
            ro.append(("F", mb, c))
        # steady: F(k+w) then B(k), backward chunk reversed
        for k in range(n - w):
            mb, c = table[k + w]
            ro.append(("F", mb, c))
            bmb, bc = table[k]
            bc = v - 1 - bc
            if bc == v - 1 and s == p - 1:
                ro.append(("W", bmb, bc))
            ro.append(("B", bmb, bc))
        # cooldown backwards
        for k in range(n - w, n):
            bmb, bc = table[k]
            bc = v - 1 - bc
            if bc == v - 1 and s == p - 1:
                ro.append(("W", bmb, bc))
            ro.append(("B", bmb, bc))
        ops.append(ro)
    return ops


# ── Simulator ─────────────────────────────────────────────────────────────────

class Result:
    def __init__(self, p, m, tf, tb, spans: List[List[Tuple[Op, Span]]], v: int = 1):
        self.p, self.m, self.tf, self.tb, self.v = p, m, tf, tb, v
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
        """Max units whose activation is stored but not yet backwarded on stage s.

        A unit is keyed by (mb, chunk) so interleaved chunks count separately.
        Live from F(s,mb,chunk).end until B(s,mb,chunk).start.
        """
        f_end: Dict[Tuple[int, int], float] = {}
        b_start: Dict[Tuple[int, int], float] = {}
        for op, (st, en) in self.spans[s]:
            kind, mb, chunk = _norm_op(op)
            if kind == "F":
                f_end[(mb, chunk)] = en
            elif kind == "B":
                b_start[(mb, chunk)] = st
        # sweep events
        events = []
        for key in f_end:
            events.append((f_end[key], +1))
            if key in b_start:
                events.append((b_start[key], -1))
        events.sort(key=lambda x: (x[0], -x[1]))  # ends before starts at same t
        cur = peak = 0
        for _, d in events:
            cur += d
            peak = max(peak, cur)
        return peak


def _norm_op(op: Tuple) -> Tuple[str, int, int]:
    """Normalize an op tuple to (kind, mb, chunk); chunk defaults to 0."""
    if len(op) == 3:
        return op[0], op[1], op[2]
    return op[0], op[1], 0


def simulate(
    rank_ops: RankOps,
    tf: float = 1.0,
    tb: float = 1.0,
    comm: float = 0.0,
) -> Result:
    """
    Compute the ideal (start, end) span of every op via fixed-point iteration.

    Each op may be ("F"|"B"|"W", mb) or ("F"|"B"|"W", mb, chunk). The virtual
    rank of an op is vr = chunk*p + s; the logical pipeline is a chain of
    virtual ranks 0..p*v-1 (v = number of chunks). Deps flow along vr:

    start(s, i) = max(end of op i-1 on stage s, deps_ready)
      deps_ready(F) = end(F at vr-1)            (0 for vr 0)
      deps_ready(B) = end(B at vr+1)            (end(F at vr) for the last vr)
      deps_ready(W) = end(F at vr)
    end = start + (tf if F else tb if B else 0)

    A dep crossing between two virtual ranks always crosses a physical device
    boundary (consecutive virtual ranks live on different devices), so `comm`
    is added to dep_ready for every cross-vr hop. This models the extra
    communication interleaving pays for its smaller bubble.
    """
    p = len(rank_ops)
    # normalize every op to (kind, mb, chunk) up front
    norm_ops: List[List[Tuple[str, int, int]]] = [
        [_norm_op(op) for op in ops] for ops in rank_ops
    ]
    m = 1 + max((mb for ops in norm_ops for _, mb, _ in ops), default=-1)
    max_vr = -1
    for s, ops in enumerate(norm_ops):
        for _, _, chunk in ops:
            max_vr = max(max_vr, chunk * p + s)
    v = (max_vr + p) // p if max_vr >= 0 else 1  # chunks (>=1)

    # index of each (kind, vr, mb) within its physical stage's list
    pos: List[Dict[Tuple[str, int, int], int]] = []
    for s, ops in enumerate(norm_ops):
        d = {}
        for i, (k, mb, chunk) in enumerate(ops):
            d[(k, chunk * p + s, mb)] = i
        pos.append(d)

    # locate an op on any stage by (kind, vr, mb) -> (stage, index)
    owner: Dict[Tuple[str, int, int], Tuple[int, int]] = {}
    for s in range(p):
        for key, i in pos[s].items():
            owner[key] = (s, i)

    end: List[List[Optional[float]]] = [[None] * len(ops) for ops in norm_ops]
    start: List[List[Optional[float]]] = [[None] * len(ops) for ops in norm_ops]
    last_vr = p * v - 1

    def _end_of(kind: str, vr: int, mb: int) -> Optional[float]:
        loc = owner.get((kind, vr, mb))
        if loc is None:
            return None
        return end[loc[0]][loc[1]]

    def dep_ready(s: int, i: int) -> float:
        kind, mb, chunk = norm_ops[s][i]
        vr = chunk * p + s
        if kind == "F":
            if vr == 0:
                return 0.0
            e = _end_of("F", vr - 1, mb)
            return (e + comm) if e is not None else 0.0
        if kind == "W":
            e = _end_of("F", vr, mb)
            return e if e is not None else 0.0
        # B
        if vr == last_vr:
            e = _end_of("F", vr, mb)
            return e if e is not None else 0.0
        e = _end_of("B", vr + 1, mb)
        return (e + comm) if e is not None else 0.0

    dur = {"F": tf / v, "B": tb / v, "W": 0.0}

    # Fixed-point: an op is "computable" when all the end-times it reads are set.
    def computable(s: int, i: int) -> bool:
        kind, mb, chunk = norm_ops[s][i]
        vr = chunk * p + s
        if i > 0 and end[s][i - 1] is None:
            return False
        if kind == "F":
            return vr == 0 or _end_of("F", vr - 1, mb) is not None
        if kind == "W":
            return _end_of("F", vr, mb) is not None
        if vr == last_vr:
            return _end_of("F", vr, mb) is not None
        return _end_of("B", vr + 1, mb) is not None

    remaining = sum(len(ops) for ops in norm_ops)
    guard = remaining * (remaining + 1)  # generous iteration cap
    while remaining and guard > 0:
        progressed = False
        for s in range(p):
            for i in range(len(norm_ops[s])):
                if end[s][i] is None and computable(s, i):
                    prev_end = end[s][i - 1] if i > 0 else 0.0
                    st = max(prev_end, dep_ready(s, i))
                    kind = norm_ops[s][i][0]
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
        for i, (k, mb, chunk) in enumerate(norm_ops[s]):
            if k == "W":
                continue
            lst.append(((k, mb, chunk), (start[s][i], end[s][i])))
        spans.append(lst)
    return Result(p, m, tf, tb, spans, v=v)


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
        for op, (st, en) in res.spans[s]:
            kind, mb, chunk = _norm_op(op)
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
    gpipe_makespan: Optional[float] = None,
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

    def _shade(hex_color: str, chunk: int, v: int) -> str:
        """Lighten a base color per virtual chunk so chunks are distinguishable.

        chunk 0 keeps the base color; higher chunks get progressively lighter
        (up to ~55% toward white) so the eye can separate interleaved chunks
        within one stage's row while F stays blue-ish and B stays orange-ish.
        """
        if v <= 1:
            return hex_color
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        frac = 0.55 * (chunk / max(1, v - 1))  # 0 -> base, 1 -> 55% toward white
        r = int(r + (255 - r) * frac)
        g = int(g + (255 - g) * frac)
        b = int(b + (255 - b) * frac)
        return f"#{r:02x}{g:02x}{b:02x}"

    for ax, (name, res) in zip(axes[:, 0], results):
        for s in range(res.p):
            y = res.p - 1 - s  # stage 0 on top
            for op, (st, en) in res.spans[s]:
                kind, mb, chunk = _norm_op(op)
                ax.barh(y, en - st, left=st, height=0.7,
                        color=_shade(colors[kind], chunk, res.v),
                        edgecolor="white", linewidth=0.4)
                label = f"{kind}{mb}" if res.v == 1 else f"{kind}{mb}c{chunk}"
                # only label if the bar is wide enough to read
                if en - st >= res.makespan * 0.02:
                    ax.text((st + en) / 2, y, label, ha="center", va="center",
                            fontsize=7, color="white", weight="bold")
        ax.set_yticks(range(res.p))
        ax.set_yticklabels([f"s{res.p - 1 - i}" for i in range(res.p)])
        ax.set_ylim(-0.6, res.p - 0.4)
        ax.grid(axis="x", alpha=0.25)
        gpipe_part = (
            f"  vs_gpipe={gpipe_makespan/res.makespan:.2f}x"
            if gpipe_makespan else ""
        )
        ax.set_title(
            f"{name}   makespan={res.makespan:.0f}  "
            f"bubble={100*res.bubble_frac():.0f}%  "
            f"speedup={serial/res.makespan:.2f}x"
            f"{gpipe_part}  "
            f"peak_mem={max(res.peak_inflight(s) for s in range(res.p))}",
            loc="left", fontsize=10,
        )
    axes[-1, 0].set_xlabel("time")
    handles = [Patch(facecolor=colors["F"], label="forward"),
               Patch(facecolor=colors["B"], label="backward")]
    # if any schedule is interleaved, show the chunk shading in the legend
    max_v = max((r.v for _, r in results), default=1)
    if max_v > 1:
        for c in range(max_v):
            handles.append(Patch(facecolor=_shade(colors["F"], c, max_v),
                                 label=f"F chunk{c}"))
            handles.append(Patch(facecolor=_shade(colors["B"], c, max_v),
                                 label=f"B chunk{c}"))
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

def report(name: str, res: Result, serial: float, gpipe_makespan: Optional[float] = None) -> str:
    gpipe_part = (
        f"  speedup_vs_gpipe={gpipe_makespan/res.makespan:4.2f}x"
        if gpipe_makespan else ""
    )
    out = [f"  {name:<10} makespan={res.makespan:5.1f}  "
           f"bubble={100*res.bubble_frac():4.1f}%  "
           f"speedup_vs_serial={serial/res.makespan:4.2f}x"
           f"{gpipe_part}  "
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
    ap.add_argument("--v", type=int, default=1,
                    help="virtual chunks per stage for interleaved_1b1f (>1 enables it)")
    ap.add_argument("--comm", type=float, default=0.0,
                    help="per-hop comm cost added to every cross-virtual-rank dependency")
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
        ("1b1f", staggered_1f1b_ops(p, m)),
    ]
    if args.v > 1:
        schedules.append((f"ilv{args.v}", interleaved_1b1f_ops(p, m, args.v)))
    if args.group > 0:
        w = args.warmup if args.warmup > 0 else p - 1
        schedules.append((f"grp{args.group}/w{w}", fwd_grouped_ops(p, m, w, args.group)))

    results: List[Tuple[str, Result]] = []
    for name, ops in schedules:
        res = simulate(ops, tf=tf, tb=tb, comm=args.comm)
        results.append((name, res))

    # gpipe is always the first schedule; use it as the relative-speedup baseline
    gpipe_makespan = results[0][1].makespan if results and results[0][0] == "gpipe" else None
    for name, res in results:
        print(report(name, res, serial, gpipe_makespan))
        print()

    if args.plot is not None:
        out = args.plot or None
        plot_gantt(results, serial, out_path=out, show=(out is None),
                   gpipe_makespan=gpipe_makespan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
