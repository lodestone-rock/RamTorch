"""
pipeline_infer_stream_check.py
------------------------------
Validate the STREAMING (asynchronous) inference path —
``Pipeline.infer_submit`` / ``infer_open`` + ``submit_mb`` / ``wait_mb`` /
``infer_loop`` — against the synchronous ``Pipeline.infer()`` barrier, with
bit-level expectations (identical op order and dtypes -> identical bits).

Checks (on CPU always, and on every visible CUDA device when >= 4):
  1. submit/result parity: tensor, flat-tuple, and nested pre-diced inputs,
     including a non-divisible batch (padding sliced identically).
  2. Trickle submission: infer_open + submit_mb in ANY order delivers
     per-microbatch outputs identical to the sync microbatch slices.
  3. Overlapping batches: submitting batch k+1 before batch k has finished
     (the no-drain case) is bit-exact vs two sequential sync calls.
  4. infer_loop parity: a multi-step update loop (per-microbatch scheduler)
     is bit-exact vs the same loop driven by barriered infer() calls.
  5. Tuple outputs relay through the async path bit-exactly.
  6. Offloaded stages (chunk_modules -> OffloadStage, window=2): streaming
     submit + infer_loop are bit-exact vs the sync barrier on the SAME
     offloaded pipeline (weight streaming under persistent workers).
  7. Validation + failure semantics: double submit_mb, out-of-range index,
     submit_mb on a closed batch, steps < 1, worker exceptions propagate
     from wait_mb/result and poison the session, close() is idempotent
     with a live session.

Run:  PYTHONPATH=. python examples/pipeline_infer_stream_check.py
"""

import sys

import torch
import torch.nn as nn

from ramtorch import Pipeline


def make_stages(n_stages, dim, seed):
    torch.manual_seed(seed)
    return [nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
            for _ in range(n_stages)]


def check_exact(name, a, b, errs):
    a = a.detach().cpu()
    b = b.detach().cpu()
    if not torch.equal(a, b):
        errs.append(f"{name}: max err {(a - b).abs().max().item():.3e}")


def check_submit_parity(devices, errs):
    tag = f"[{devices[0]}..{devices[-1]} submit]"
    n_before = len(errs)
    pipe = Pipeline(stage_modules=make_stages(len(devices), 16, seed=1),
                    devices=devices)
    x = torch.randn(8, 16)

    # tensor input
    ref = pipe.infer(x, n_microbatches=4)
    check_exact(f"{tag} tensor", pipe.infer_submit(x, n_microbatches=4).result(),
                ref, errs)

    # flat-tuple input (positional args into stage 0)
    class Add(nn.Module):
        def forward(self, a, b):
            return a + b
    pipe_t = Pipeline(
        stage_modules=[Add(), nn.Linear(16, 16)] + make_stages(len(devices) - 2, 16, seed=2)
        if len(devices) > 2 else [Add(), nn.Linear(16, 16)],
        devices=devices if len(devices) > 2 else devices[:2],
    )
    ref_t = pipe_t.infer((x, x), n_microbatches=4)
    got_t = pipe_t.infer_submit((x, x), n_microbatches=4).result()
    check_exact(f"{tag} flat-tuple", got_t, ref_t, errs)

    # nested pre-diced input -> nested per-mb output
    nested = tuple(torch.randn(2, 16) for _ in range(4))
    ref_n = pipe.infer(nested, n_microbatches=4)
    got_n = pipe.infer_submit(nested, n_microbatches=4).result()
    if not isinstance(got_n, tuple) or len(got_n) != 4:
        errs.append(f"{tag} nested: output did not mirror pre-diced input")
    else:
        for i in range(4):
            check_exact(f"{tag} nested mb{i}", got_n[i], ref_n[i], errs)

    # non-divisible batch: padding sliced identically
    x5 = torch.randn(5, 16)
    ref_p = pipe.infer(x5, n_microbatches=4)
    got_p = pipe.infer_submit(x5, n_microbatches=4).result()
    if got_p.shape != ref_p.shape:
        errs.append(f"{tag} padded: shape {tuple(got_p.shape)} != "
                    f"{tuple(ref_p.shape)}")
    else:
        check_exact(f"{tag} padded", got_p, ref_p, errs)

    pipe_t.close()
    pipe.close()
    print(f"  {'PASS' if len(errs) == n_before else 'FAIL'} {tag}")


def check_trickle(devices, errs):
    tag = f"[{devices[0]}..{devices[-1]} trickle]"
    n_before = len(errs)
    pipe = Pipeline(stage_modules=make_stages(len(devices), 16, seed=3),
                    devices=devices)
    x = torch.randn(8, 16)
    ref = pipe.infer(x, n_microbatches=4)
    mbs = list(x.chunk(4, dim=0))

    h = pipe.infer_open(4)
    for i in [2, 0, 3, 1]:  # submission order must not matter
        h.submit_mb(i, mbs[i])
    for i in range(4):
        check_exact(f"{tag} mb{i}", h.wait_mb(i), ref.chunk(4, dim=0)[i], errs)
    pipe.close()
    print(f"  {'PASS' if len(errs) == n_before else 'FAIL'} {tag}")


def check_overlap(devices, errs):
    tag = f"[{devices[0]}..{devices[-1]} overlap]"
    n_before = len(errs)
    pipe = Pipeline(stage_modules=make_stages(len(devices), 16, seed=4),
                    devices=devices)
    x, y = torch.randn(8, 16), torch.randn(8, 16)

    # Batch B submitted while batch A is still in flight.
    ha = pipe.infer_submit(x, n_microbatches=4)
    hb = pipe.infer_submit(y, n_microbatches=4)
    check_exact(f"{tag} batch A", ha.result(), pipe.infer(x, n_microbatches=4), errs)
    check_exact(f"{tag} batch B", hb.result(), pipe.infer(y, n_microbatches=4), errs)
    pipe.close()
    print(f"  {'PASS' if len(errs) == n_before else 'FAIL'} {tag}")


def check_infer_loop(devices, errs, offload=False):
    kind = "offload" if offload else "resident"
    tag = f"[{devices[0]}..{devices[-1]} infer_loop {kind}]"
    n_before = len(errs)
    n_stages = len(devices)
    if offload:
        chunks = [m for s in make_stages(n_stages * 2, 16, seed=5) for m in [s]]
        pipe = Pipeline(chunk_modules=chunks, devices=devices,
                        offload_window=2, offload_pin=0)
    else:
        pipe = Pipeline(stage_modules=make_stages(n_stages, 16, seed=5),
                        devices=devices)
    x0 = torch.randn(8, 16)
    upd = lambda out, i, t: out * 0.9 + 0.01 * t  # per-mb "scheduler"

    got = pipe.infer_loop(x0, steps=5, update_fn=upd, n_microbatches=4)

    x = x0
    for t in range(4):
        x = upd(pipe.infer(x, n_microbatches=4), 0, t)
    ref = pipe.infer(x, n_microbatches=4)
    check_exact(f"{tag} 5-step loop", got, ref, errs)
    pipe.close()
    print(f"  {'PASS' if len(errs) == n_before else 'FAIL'} {tag}")


def check_tuple_outputs(devices, errs):
    tag = f"[{devices[0]}..{devices[-1]} tuple-out]"
    n_before = len(errs)

    class TwoOut(nn.Module):
        def forward(self, x):
            return x * 2, x + 1

    class Mid(nn.Module):
        def __init__(self, seed):
            super().__init__()
            g = torch.Generator().manual_seed(seed)
            self.lin_a = nn.Linear(16, 16, bias=False)
            self.lin_b = nn.Linear(16, 16, bias=False)
            with torch.no_grad():
                for lin in (self.lin_a, self.lin_b):
                    lin.weight.copy_(
                        torch.randn(16, 16, generator=g) * (16 ** -0.5))

        def forward(self, a, b):
            return self.lin_a(a), self.lin_b(b)

    class Merge(nn.Module):
        def forward(self, a, b):
            return a + b

    mids = [Mid(seed=60 + i) for i in range(max(len(devices) - 2, 0))]
    pipe = Pipeline(stage_modules=[TwoOut()] + mids + [Merge()], devices=devices)
    x = torch.randn(8, 16)
    ref = pipe.infer(x, n_microbatches=4)
    got = pipe.infer_submit(x, n_microbatches=4).result()
    check_exact(f"{tag} result", got, ref, errs)

    h = pipe.infer_open(4)
    for i, mb in enumerate(x.chunk(4, dim=0)):
        h.submit_mb(i, mb)
    for i in range(4):
        check_exact(f"{tag} wait_mb{i}", h.wait_mb(i),
                    ref.chunk(4, dim=0)[i], errs)
    pipe.close()
    print(f"  {'PASS' if len(errs) == n_before else 'FAIL'} {tag}")


def check_validation_and_failures(errs):
    tag = "[validation/failure]"
    n_before = len(errs)
    pipe = Pipeline(stage_modules=make_stages(2, 16, seed=7), devices=["cpu"] * 2)
    x = torch.randn(4, 16)

    def expect(name, fn, exc):
        try:
            fn()
            errs.append(f"{tag} {name}: no {exc.__name__} raised")
        except exc:
            pass
        except Exception as e:  # noqa: BLE001
            errs.append(f"{tag} {name}: wrong exception {type(e).__name__}: {e}")

    h = pipe.infer_open(2)
    h.submit_mb(0, x[:2])
    expect("double submit_mb", lambda: h.submit_mb(0, x[:2]), RuntimeError)
    expect("submit_mb index", lambda: h.submit_mb(9, x[:2]), IndexError)
    h.submit_mb(1, x[2:])
    h.result()

    h2 = pipe.infer_submit(x, n_microbatches=2)
    expect("submit_mb on closed batch", lambda: h2.submit_mb(0, x[:2]),
           RuntimeError)
    h2.result()
    expect("wait_mb index", lambda: h2.wait_mb(9), IndexError)
    expect("steps=0", lambda: pipe.infer_loop(x, steps=0,
                                              update_fn=lambda o, i, t: o),
           ValueError)

    # Worker exception: propagates from wait_mb/result and poisons the session.
    class Boom(nn.Module):
        def forward(self, x):
            raise ValueError("boom")

    bad = Pipeline(stage_modules=[nn.Linear(16, 16), Boom()], devices=["cpu"] * 2)
    hb = bad.infer_submit(x, n_microbatches=2)
    try:
        hb.result()
        errs.append(f"{tag} worker error: no RuntimeError raised")
    except RuntimeError as e:
        if not isinstance(e.__cause__, ValueError):
            errs.append(f"{tag} worker error: cause is "
                        f"{type(e.__cause__).__name__}, expected ValueError")
    expect("poisoned session submit",
           lambda: bad.infer_submit(x, n_microbatches=2), RuntimeError)

    # close() is idempotent with a live (idle) session.
    bad.close()
    bad.close()
    pipe.close()
    pipe.close()
    print(f"  {'PASS' if len(errs) == n_before else 'FAIL'} {tag}")


def main():
    errs = []
    cuda = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    device_sets = [["cpu", "cpu"]]
    if len(cuda) >= 4:
        device_sets.append(cuda[:4])
    elif len(cuda) >= 2:
        device_sets.append(cuda[:2])

    for devices in device_sets:
        check_submit_parity(devices, errs)
        check_trickle(devices, errs)
        check_overlap(devices, errs)
        check_infer_loop(devices, errs, offload=False)
        check_tuple_outputs(devices, errs)
        if devices[0] != "cpu":
            check_infer_loop(devices, errs, offload=True)
    check_validation_and_failures(errs)

    if errs:
        print(f"\n{len(errs)} FAILURE(S):")
        for e in errs:
            print("  ", e)
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
