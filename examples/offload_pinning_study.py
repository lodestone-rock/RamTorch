"""
offload_pinning_study.py
------------------------
Does pinning some layers permanently on the GPU beat spending the same memory
on a bigger streaming window?

Setup: n=12 layers, training echo, base streaming window W=2. Pinned layers
are evenly spaced, never loaded, never evicted, and cost one slot each — so
the apples-to-apples comparisons at equal total memory are:

    pin 6/12 + W=2  (mem 8)   vs   pure W=8
    pin 3/12 + W=2  (mem 5)   vs   pure W=5
    baseline: pure W=2 (mem 2)

Both sides of each pair also do the same steady-state H2D traffic: pinning k
layers shrinks the streamed set to n-k with window 2 (~2*(n-k-2) loads/step),
while the pure window keeps ~2*(n-W) = same count. What differs is *timing*:
a big window gives the prefetcher deep lookahead, while pinning gives the
tiny window free time whenever compute runs a pinned layer.

Sweep th2d/tf both ways (loads faster and slower than compute) over several
steps and report makespan, steady per-step time, stall and loads.

Run:  PYTHONPATH=. python examples/offload_pinning_study.py
"""

import sys

from ramtorch.offload_simulator import (
    evenly_pinned,
    simulate_offload,
    train_itinerary,
)

EPS = 1e-9


def per_step_steady(res, ops_per_step: int) -> float:
    """Duration of the last step (steady state), from compute span ends."""
    ends = [en for _, (_, en) in res.spans["gpu"]]
    if len(ends) < 2 * ops_per_step:
        return ends[-1] if ends else 0.0
    return ends[-1] - ends[-1 - ops_per_step]


def study(n: int = 12, steps: int = 4) -> bool:
    ratios = (0.5, 1.0, 2.0, 4.0, 8.0)
    configs = [
        ("W=2            (mem 2)", 2, 0),
        ("pin 3 + W=2    (mem 5)", 2, 3),
        ("W=5            (mem 5)", 5, 0),
        ("pin 6 + W=2    (mem 8)", 2, 6),
        ("W=8            (mem 8)", 8, 0),
    ]
    itin = train_itinerary(n, steps=steps)
    ops_per_step = 2 * n
    ok = True

    print(f"train echo, n={n}, steps={steps}, tf=1 tb=2, td2h=th2d, "
          f"pinned = evenly spaced")
    for r in ratios:
        print(f"\n  th2d/tf = {r}")
        print(f"    {'config':<24} {'makespan':>9} {'step_time':>9} "
              f"{'stall':>7} {'loads':>6} {'h2d%':>5}  {'regime':<15}")
        for name, w, k in configs:
            pinned = evenly_pinned(n, k)
            res = simulate_offload(itin, n, w, tf=1.0, tb=2.0, th2d=r,
                                   td2h=r, pinned=pinned)
            if res.peak_resident > w + k:
                ok = False
                print(f"    !! {name}: peak {res.peak_resident} > mem {w + k}")
            print(f"    {name:<24} {res.makespan:>9.1f} "
                  f"{per_step_steady(res, ops_per_step):>9.1f} "
                  f"{res.stall:>7.1f} {res.n_loads:>6} "
                  f"{100 * res.util('h2d'):>5.0f}  {res.regime:<15}")
    return ok


def main() -> int:
    ok = study()
    print("\nPASS" if ok else "\nFAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
