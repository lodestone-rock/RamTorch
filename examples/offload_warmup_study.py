"""
offload_warmup_study.py
-----------------------
Does preloading the window ("warmup") before the first forward speed up
windowed CPU->GPU streaming? Sweep the load/compute speed ratio both ways
(loads slower AND faster than compute) and multiple training steps, and
compare greedy start (compute begins the moment layer 0 lands) against a
full-window warmup (compute held until W layers are resident).

Expectation from the model: the prefetcher is asynchronous and never waits
for compute — H2D streams back-to-back either way, and its only stalls
(Belady eviction waits) resolve *sooner* when compute is further ahead. So
warmup can never reduce the makespan; it can only relocate compute stall
into one upfront block. The study verifies this quantitatively and also
asserts it over a fuzz grid.

Run:  PYTHONPATH=. python examples/offload_warmup_study.py
"""

import sys

from ramtorch.offload_simulator import simulate_offload, train_itinerary

EPS = 1e-9


def study(n: int = 10, w: int = 5) -> bool:
    ratios = (0.25, 0.5, 1.0, 2.0, 4.0)   # th2d/tf: <1 loads faster, >1 slower
    steps_list = (1, 4, 16)
    ok = True

    print(f"train echo, n={n} W={w}, tf=1 tb=2, td2h=th2d")
    print(f"  {'th2d/tf':>8} {'steps':>6}  {'greedy':>9} {'warmup':>9} "
          f"{'diff':>7}  {'greedy stall':>12} {'warmup stall':>12}  {'regime':<15}")
    for r in ratios:
        for steps in steps_list:
            itin = train_itinerary(n, steps=steps)
            kw = dict(tf=1.0, tb=2.0, th2d=r, td2h=r)
            greedy = simulate_offload(itin, n, w, warmup=0, **kw)
            warm = simulate_offload(itin, n, w, warmup=w, **kw)
            diff = warm.makespan - greedy.makespan
            if diff < -EPS:
                ok = False
                flag = "  <-- warmup WON (unexpected)"
            else:
                flag = ""
            print(f"  {r:>8} {steps:>6}  {greedy.makespan:>9.1f} "
                  f"{warm.makespan:>9.1f} {diff:>+7.2f}  "
                  f"{greedy.stall:>12.2f} {warm.stall:>12.2f}  "
                  f"{greedy.regime:<15}{flag}")
        print()
    return ok


def fuzz_warmup_never_wins() -> bool:
    """Assert warmup >= greedy makespan across a broad grid."""
    costs = [
        (1.0, 2.0, 0.25, 0.25),
        (1.0, 2.0, 1.0, 1.0),
        (1.0, 2.0, 4.0, 2.0),
        (1.0, 1.0, 2.0, 0.0),
        (0.5, 3.0, 1.5, 1.5),
    ]
    total = bad = 0
    for n in (1, 2, 4, 8, 16):
        for w in range(1, n + 2):
            for steps in (1, 3):
                for tf, tb, th2d, td2h in costs:
                    for wu in range(1, w + 1):
                        total += 1
                        itin = train_itinerary(n, steps=steps)
                        kw = dict(tf=tf, tb=tb, th2d=th2d, td2h=td2h)
                        g = simulate_offload(itin, n, w, warmup=0, **kw)
                        m = simulate_offload(itin, n, w, warmup=wu, **kw)
                        if m.makespan < g.makespan - EPS:
                            bad += 1
                            print(f"  warmup beat greedy: n={n} W={w} wu={wu} "
                                  f"steps={steps} costs={tf},{tb},{th2d},{td2h}: "
                                  f"{m.makespan} < {g.makespan}")
    print(f"[fuzz] warmup never beats greedy: {total - bad}/{total} configs")
    return bad == 0


def main() -> int:
    ok = study()
    ok &= fuzz_warmup_never_wins()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
