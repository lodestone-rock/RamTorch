"""
offload_sim_check.py
--------------------
Validate ramtorch.offload_simulator: invariants under fuzz, corner-case
exactness, the n=10 W=5 training-echo scenario, and a minimum-window sweep.

Checks:
  1. Fuzz grid (n, W, cost ratios, train/infer): residency never exceeds W,
     compute never runs a non-resident layer, every op is scheduled (no
     deadlock), channels are serial, makespan >= both analytic bounds.
  2. Corner-case exactness:
       th2d=td2h=0        -> makespan == total compute
       tf=tb=0, td2h=0    -> makespan == n_loads*th2d, H2D gapless
       W >= n             -> zero stalls after the first pass, forever
  3. User scenario: n=10, W=5 training echo — residents at the start of B8
     are exactly {5,6,7,8,9}; the only stall is the initial load (th2d).
  4. Sweep report: minimum window W for zero steady-state stall as a function
     of th2d/tf, forward-only and training — the table that sizes real
     deployments.

Run:  PYTHONPATH=. python examples/offload_sim_check.py
"""

import sys

from ramtorch.offload_simulator import (
    evenly_pinned,
    infer_itinerary,
    simulate_offload,
    train_itinerary,
)

EPS = 1e-9


def steady_stall(res, first_steady_op: int) -> float:
    """Sum of gaps between consecutive compute spans from op index onward."""
    spans = res.spans["gpu"]
    gaps = 0.0
    for i in range(max(1, first_steady_op), len(spans)):
        gaps += max(0.0, spans[i][1][0] - spans[i - 1][1][1])
    return gaps


# ── 1. invariant fuzz ─────────────────────────────────────────────────────────

def check_invariants(res, itin, mem) -> list:
    errs = []
    if res.peak_resident > mem:
        errs.append(f"peak_resident {res.peak_resident} > mem {mem}")
    if len(res.spans["gpu"]) != len(itin):
        errs.append(f"{len(res.spans['gpu'])}/{len(itin)} compute ops scheduled")
    if len(res.resident_at) != len(itin):
        errs.append("resident_at length mismatch")
    for i, ((kind, layer), _) in enumerate(res.spans["gpu"]):
        if (kind, layer) != itin[i]:
            errs.append(f"op {i} is {(kind, layer)}, itinerary has {itin[i]}")
            break
        if layer not in res.resident_at[i]:
            errs.append(f"op {i} ({kind}{layer}) ran while layer not resident")
            break
    for ch in ("gpu", "h2d", "d2h"):
        spans = res.spans[ch]
        for i in range(1, len(spans)):
            if spans[i][1][0] < spans[i - 1][1][1] - EPS:
                errs.append(f"{ch} spans overlap at index {i}")
                break
    nb = sum(1 for k, _ in itin if k == "B")
    if res.td2h > 0 and len(res.spans["d2h"]) != nb:
        errs.append(f"{len(res.spans['d2h'])} writebacks, expected {nb}")
    if res.makespan < res.compute_bound - EPS:
        errs.append(f"makespan {res.makespan} < compute bound {res.compute_bound}")
    if res.makespan < res.transfer_bound - EPS:
        errs.append(f"makespan {res.makespan} < transfer bound {res.transfer_bound}")
    return errs


def fuzz() -> bool:
    costs = [
        (1.0, 2.0, 0.5, 0.5),
        (1.0, 1.0, 1.0, 1.0),
        (1.0, 2.0, 3.0, 1.5),   # transfer-bound
        (1.0, 2.0, 0.0, 0.0),   # free transfers
        (0.0, 0.0, 1.0, 0.0),   # free compute
        (1.0, 3.5, 0.7, 1.3),   # d2h slower than h2d
    ]
    total = bad = 0
    for n in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        for w in range(1, n + 3):
            for tf, tb, th2d, td2h in costs:
                for name, itin in (
                    ("train", train_itinerary(n, steps=2)),
                    ("infer", infer_itinerary(n, passes=2)),
                ):
                    for npin in {0, n // 2}:
                        pinned = evenly_pinned(n, npin)
                        total += 1
                        try:
                            res = simulate_offload(itin, n, w, tf=tf, tb=tb,
                                                   th2d=th2d, td2h=td2h,
                                                   pinned=pinned)
                            errs = check_invariants(res, itin, w + npin)
                        except Exception as e:  # noqa: BLE001 — report any crash
                            errs = [f"exception: {e}"]
                        if errs:
                            bad += 1
                            print(f"  [{name} n={n} W={w} pin={npin} "
                                  f"costs={tf},{tb},{th2d},{td2h}] "
                                  + "; ".join(errs))
    print(f"[fuzz] {total - bad}/{total} configs pass all invariants")
    return bad == 0


# ── 2. corner cases ───────────────────────────────────────────────────────────

def corner_cases() -> bool:
    ok = True

    # free transfers -> makespan == total compute, zero stall
    for n, w in ((1, 1), (5, 2), (10, 5), (8, 8)):
        itin = train_itinerary(n)
        res = simulate_offload(itin, n, w, tf=1.0, tb=2.0, th2d=0.0, td2h=0.0)
        if abs(res.makespan - res.total_compute) > EPS or res.stall > EPS:
            ok = False
            print(f"  [free-transfer n={n} W={w}] makespan={res.makespan} "
                  f"!= compute {res.total_compute} (stall={res.stall})")

    # free compute -> makespan == serialized H2D time, gapless link
    for n, w in ((5, 2), (10, 5), (6, 1)):
        itin = train_itinerary(n)
        res = simulate_offload(itin, n, w, tf=0.0, tb=0.0, th2d=1.0, td2h=0.0)
        expect = res.n_loads * 1.0
        h2d = res.spans["h2d"]
        gapless = all(
            abs(h2d[i][1][0] - h2d[i - 1][1][1]) < EPS for i in range(1, len(h2d))
        )
        if abs(res.makespan - expect) > EPS or not gapless:
            ok = False
            print(f"  [free-compute n={n} W={w}] makespan={res.makespan} "
                  f"!= {expect} loads*th2d (gapless={gapless})")

    # all layers pinned -> no loads, no stalls, makespan == total compute
    for n in (1, 4, 10):
        itin = train_itinerary(n, steps=2)
        res = simulate_offload(itin, n, 1, tf=1.0, tb=2.0, th2d=5.0,
                               td2h=0.0, pinned=evenly_pinned(n, n))
        if (res.n_loads != 0 or res.stall > EPS
                or abs(res.makespan - res.total_compute) > EPS):
            ok = False
            print(f"  [all-pinned n={n}] loads={res.n_loads} "
                  f"stall={res.stall} makespan={res.makespan}")

    # W >= n: after the first pass, zero stalls forever (loads happen once)
    for mode, itin_fn, per_pass in (
        ("infer", lambda n: infer_itinerary(n, passes=3), lambda n: n),
        ("train", lambda n: train_itinerary(n, steps=3), lambda n: 2 * n),
    ):
        for n in (4, 10):
            for th2d in (0.5, 3.0):  # both regimes
                res = simulate_offload(itin_fn(n), n, n, tf=1.0, tb=2.0,
                                       th2d=th2d, td2h=0.5)
                s = steady_stall(res, per_pass(n))
                if s > EPS:
                    ok = False
                    print(f"  [W>=n {mode} n={n} th2d={th2d}] "
                          f"stall {s} after first pass")
                if res.n_loads != n:
                    ok = False
                    print(f"  [W>=n {mode} n={n} th2d={th2d}] "
                          f"{res.n_loads} loads, expected {n}")

    print(f"[corner cases] {'OK' if ok else 'MISMATCH'}")
    return ok


# ── 3. user scenario: n=10 W=5 training echo ──────────────────────────────────

def user_scenario() -> bool:
    n, w = 10, 5
    itin = train_itinerary(n)
    res = simulate_offload(itin, n, w, tf=1.0, tb=2.0, th2d=0.5, td2h=0.5)
    ok = True

    idx_b8 = itin.index(("B", 8))  # F0..F9 then B9 -> index 11
    got = set(res.resident_at[idx_b8])
    if got != {5, 6, 7, 8, 9}:
        ok = False
        print(f"  residents at start of B8: {sorted(got)} != [5..9]")

    # the echo turnaround is self-warming: the only stall is the initial load
    if abs(res.stall - 0.5) > EPS:
        ok = False
        print(f"  stall {res.stall} != th2d 0.5 (turnaround not stall-free)")

    # F9 -> B9 back-to-back (turnaround pays no wait)
    spans = res.spans["gpu"]
    f9_end = spans[9][1][1]
    b9_start = spans[10][1][0]
    if abs(b9_start - f9_end) > EPS:
        ok = False
        print(f"  turnaround gap: B9 starts {b9_start}, F9 ends {f9_end}")

    print(f"[user scenario n=10 W=5] {'OK' if ok else 'MISMATCH'} "
          f"(residents@B8={sorted(got)}, stall={res.stall})")
    return ok


# ── 4. minimum-window sweep ───────────────────────────────────────────────────

def min_window_sweep(n: int = 10) -> bool:
    """Min W with zero steady-state stall vs th2d/tf — sizes real deployments."""
    ratios = (0.25, 0.5, 1.0, 2.0, 4.0)
    print(f"[sweep] min W for zero steady-state stall (n={n}, tf=1, tb=2):")
    print(f"  {'th2d/tf':>8}  {'infer':>6}  {'train':>6}")
    ok = True
    for r in ratios:
        row = []
        for mode, itin, first_steady in (
            ("infer", infer_itinerary(n, passes=3), 2 * n),
            ("train", train_itinerary(n, steps=3), 4 * n),
        ):
            best = None
            for w in range(1, n + 1):
                res = simulate_offload(itin, n, w, tf=1.0, tb=2.0,
                                       th2d=r, td2h=r)
                if steady_stall(res, first_steady) <= EPS:
                    best = w
                    break
            if best is None:
                ok = False  # W=n must always reach steady state with no loads
                row.append("??")
            else:
                row.append(str(best))
        print(f"  {r:>8}  {row[0]:>6}  {row[1]:>6}")
    return ok


def main() -> int:
    ok = fuzz()
    ok &= corner_cases()
    ok &= user_scenario()
    ok &= min_window_sweep()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
