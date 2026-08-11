"""
vshape_schedule_check.py
------------------------
Validate the V-shape (echo) interleaved 1F1B schedule builder against the
hand-drawn reference timelines, then fuzz it and compare against the other
schedules.

The reference notation (one row per stage, one column per unit time slot):

    f{k}   forward of chunk 0 for microbatch k, descending s0 -> s(p-1)
    fi{k}  forward of the echo chunk, ascending s(p-1) -> s0
    b{k}   backward of the echo chunk, descending s0 -> s(p-1)
    bi{k}  backward of chunk 0, ascending s(p-1) -> s0
    .      idle slot

Checks:
  1. `vshape_1f1b_ops` reproduces both hand-drawn op orders token-for-token.
  2. `simulate(..., placement="vshape")` at unit per-chunk cost reproduces the
     hand-drawn timing slot-for-slot (start == column index), makespan 38 / 16.
  3. Validity fuzz across p=1..8, m=1..32 (simulate raises on dead schedules).

Run:  PYTHONPATH=. python examples/vshape_schedule_check.py
"""

import sys

from ramtorch.schedule_simulator import simulate, vshape_1f1b_ops

# ── Hand-drawn references (p=4) ───────────────────────────────────────────────

TABLE_M8 = """
f0 f1 f2 f3 f4 f5 f6 fi0 b0 fi1 b1 fi2 b2 fi3 b3 f7 bi0 fi4 b4 . bi1 fi5 b5 . bi2 fi6 b6 . bi3 fi7 b7 bi4 . bi5 . bi6 . bi7
. f0 f1 f2 f3 f4 fi0 f5 fi1 b0 fi2 b1 fi3 b2 f6 bi0 fi4 b3 f7 bi1 fi5 b4 . bi2 fi6 b5 . bi3 fi7 b6 bi4 b7 bi5 . bi6 . bi7 .
. . f0 f1 f2 fi0 f3 fi1 f4 fi2 b0 fi3 b1 f5 bi0 fi4 b2 f6 bi1 fi5 b3 f7 bi2 fi6 b4 . bi3 fi7 b5 bi4 b6 bi5 b7 bi6 . bi7 . .
. . . f0 fi0 f1 fi1 f2 fi2 f3 fi3 b0 f4 bi0 fi4 b1 f5 bi1 fi5 b2 f6 bi2 fi6 b3 f7 bi3 fi7 b4 bi4 b5 bi5 b6 bi6 b7 bi7 . . .
"""

TABLE_M1 = """
f0 . . . . . . fi0 b0 . . . . . . bi0
. f0 . . . . fi0 . . b0 . . . . bi0 .
. . f0 . . fi0 . . . . b0 . . bi0 . .
. . . f0 fi0 . . . . . . b0 bi0 . . .
"""


def parse_table(text):
    """-> (ops_per_stage [(kind, mb, chunk), ...], slot_per_stage [col, ...])"""
    ops, slots = [], []
    for line in text.strip().splitlines():
        row_ops, row_slots = [], []
        for col, tok in enumerate(line.split()):
            if tok == ".":
                continue
            if tok.startswith("fi"):
                op = ("F", int(tok[2:]), 1)
            elif tok.startswith("bi"):
                op = ("B", int(tok[2:]), 0)
            elif tok.startswith("f"):
                op = ("F", int(tok[1:]), 0)
            elif tok.startswith("b"):
                op = ("B", int(tok[1:]), 1)
            else:
                raise ValueError(f"bad token {tok!r}")
            row_ops.append(op)
            row_slots.append(col)
        ops.append(row_ops)
        slots.append(row_slots)
    return ops, slots


def fmt(op):
    kind, mb, chunk = op
    return {("F", 0): "f", ("F", 1): "fi", ("B", 1): "b", ("B", 0): "bi"}[
        (kind, chunk)
    ] + str(mb)


def check_table(name, table, p, m):
    ref_ops, ref_slots = parse_table(table)
    built = vshape_1f1b_ops(p, m)
    ok = True

    # 1. exact op order (builder emits zero-cost W markers on stage 0 — skip them)
    for s in range(p):
        got = [op for op in built[s] if op[0] != "W"]
        if got != ref_ops[s]:
            ok = False
            for i, (g, r) in enumerate(zip(got, ref_ops[s])):
                if g != r:
                    print(f"  [{name}] s{s}: first diff at op {i}: "
                          f"built={fmt(g)} ref={fmt(r)}")
                    break
            else:
                print(f"  [{name}] s{s}: length {len(got)} vs ref {len(ref_ops[s])}")

    # 2. slot-exact timing: tf=tb=2 with v=2 -> every chunk op costs 1 slot
    res = simulate(built, tf=2.0, tb=2.0, placement="vshape")
    expected_makespan = max(sl[-1] for sl in ref_slots) + 1
    if res.makespan != expected_makespan:
        ok = False
        print(f"  [{name}] makespan {res.makespan} != hand-drawn {expected_makespan}")
    for s in range(p):
        got_starts = [(op, st) for (op, (st, _)) in res.spans[s]]
        for (op, st), ref_op, col in zip(got_starts, ref_ops[s], ref_slots[s]):
            if op != ref_op or st != col:
                ok = False
                print(f"  [{name}] s{s}: {fmt(op)} starts at {st}, "
                      f"hand-drawn has {fmt(ref_op)} at col {col}")
                break

    status = "OK" if ok else "MISMATCH"
    print(f"[{name}] builder order + slot timing vs hand-drawn table: {status} "
          f"(makespan {res.makespan:.0f})")
    return ok


def fuzz():
    bad = 0
    for p in range(1, 9):
        for m in range(1, 33):
            ops = vshape_1f1b_ops(p, m)
            counts = {}
            for s in range(p):
                for kind, _, chunk in ops[s]:
                    if kind != "W":
                        counts[(kind, chunk)] = counts.get((kind, chunk), 0) + 1
            if any(counts.get(k, 0) != p * m
                   for k in (("F", 0), ("F", 1), ("B", 0), ("B", 1))):
                print(f"  bad op counts at p={p} m={m}: {counts}")
                bad += 1
                continue
            try:
                simulate(ops, tf=1.0, tb=1.7, comm=0.13, placement="vshape")
            except Exception as e:
                print(f"  simulate failed at p={p} m={m}: {e}")
                bad += 1
    print(f"[fuzz] p=1..8 x m=1..32: {256 - bad}/256 valid")
    return bad == 0


def main():
    ok = check_table("p4 m8", TABLE_M8, p=4, m=8)
    ok &= check_table("p4 m1", TABLE_M1, p=4, m=1)
    ok &= fuzz()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
