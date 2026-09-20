"""
pipeline_infer_stream_check.py
------------------------------
Validate the STREAMING (asynchronous) inference path —
``Pipeline.infer_submit`` / ``infer_open`` + ``submit_mb`` / ``wait_mb`` /
``infer_loop`` — against the synchronous ``Pipeline.infer()`` barrier, with
bit-level expectations (identical op order and dtypes -> identical bits).

Checks (default: CPU plus up to four visible CUDA devices, minimum two GPUs):
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
     with a live session. Closing with queued work must drain every stage.
  8. CUDA custom caller streams: inputs produced on first/last GPU, consumed
     by wait_mb/result on a nondefault last-GPU stream with another GPU current;
     three continued loops, no autograd graphs or gradient accumulators.
     Resident and streamed windows 1/2 use three chunks per stage (eviction),
     compared to an independent resident oracle. Optional BF16 uses autocast.

Run:  PYTHONPATH=. python examples/pipeline_infer_stream_check.py
      PYTHONPATH=. python examples/pipeline_infer_stream_check.py \\
          --devices cuda:0,cuda:1,cuda:2,cuda:3 --bf16
"""

import argparse
import sys
import threading
from contextlib import contextmanager

import torch
import torch.nn as nn

from ramtorch import Pipeline


def make_stages(n_stages, dim, seed):
    torch.manual_seed(seed)
    return [nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Linear(dim * 2, dim))
            for _ in range(n_stages)]


def check_exact(name, a, b, errs):
    a = a.detach().cpu().contiguous()
    b = b.detach().cpu().contiguous()
    if a.shape != b.shape or a.dtype != b.dtype:
        errs.append(f"{name}: shape/dtype {a.shape}/{a.dtype} != {b.shape}/{b.dtype}")
    elif not torch.isfinite(a).all() or not torch.isfinite(b).all():
        errs.append(f"{name}: non-finite output (including matching NaNs/Infs)")
    elif not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
        # torch.equal alone misses signed-zero differences and allows differing
        # dtypes. Raw bytes make the claimed bit-identity literal.
        error = (a.float() - b.float()).abs().max().item() if a.numel() else 0.0
        errs.append(f"{name}: different bytes, max err {error:.3e}")


def bounded(call, timeout):
    """Run a whole check in one thread, preserving its CUDA stream context.

    A broken wait/close must fail rather than hang the test process. Abort the
    suite on timeout: do not run more checks alongside an abandoned worker.
    """
    errors, results = [], []

    def run():
        try:
            results.append(call())
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=run, daemon=True, name="infer-check")
    thread.start()
    thread.join(timeout)
    if thread.is_alive():
        raise AssertionError(f"inference check hung for {timeout}s")
    if errors:
        raise errors[0]
    return results[0]


def synchronize(devices):
    for device in dict.fromkeys(map(torch.device, devices)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)


def check_no_grad(name, out, pipe, errs):
    if out.requires_grad or out.grad_fn is not None:
        errs.append(f"{name}: inference output retained an autograd graph")
    for i, stage in enumerate(pipe.stages):
        if stage.grad_acc or stage._cache or any(p.grad is not None for p in stage.params):
            errs.append(f"{name}: stage {i} allocated gradient/activation state")
        if hasattr(stage, "engine"):
            for j, state in enumerate(stage.engine._state):
                if (state.grad_acc or state.acc_gpu is not None or state.flush_bufs
                        or state.staging is not None):
                    errs.append(f"{name}: stage {i} chunk {j} allocated gradient state")


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


@contextmanager
def caller_stream(stream, current_device):
    """Select a tensor-device stream but deliberately leave another GPU current."""
    with torch.cuda.stream(stream), torch.cuda.device(current_device):
        assert torch.cuda.current_stream(stream.device) == stream
        assert torch.cuda.current_device() == torch.device(current_device).index
        yield


def check_cuda_caller_streams(devices, errs, bf16=False, window=None):
    precision = "bf16" if bf16 else "fp32"
    kind = "resident" if window is None else f"offload window={window}"
    tag = f"[{devices[0]}..{devices[-1]} caller streams {kind} {precision}]"
    n_before = len(errs)
    dim, m, steps = 32, 4, 3
    kwargs = dict(devices=devices, chunks_per_stage=[3] * len(devices),
                  autocast="bf16" if bf16 else None)

    class ResidualCheckBlock(nn.Module):
        def __init__(self, block):
            super().__init__()
            self.block = block

        def forward(self, x):
            assert not torch.is_grad_enabled(), "inference worker enabled gradients"
            out = self.block(x)
            assert out.dtype == (torch.bfloat16 if bf16 else torch.float32)
            # Retain input sensitivity through a deep pipeline, even in BF16;
            # a contracting MLP stack can round stale-input differences away.
            return x + out * 0.125

    def chunks():
        return [ResidualCheckBlock(block)
                for block in make_stages(3 * len(devices), dim, seed=31)]

    # Three chunks PER STAGE forces repeated eviction for both windows. The
    # independent resident oracle has exactly the same parameters/op ordering.
    pipe = Pipeline(chunk_modules=chunks(),
                    offload=window is not None, offload_window=window or 2,
                    offload_pin=0, **kwargs)
    ref_pipe = None
    try:
        ref_pipe = Pipeline(chunk_modules=chunks(), offload=False, **kwargs)
        first, last = devices[0], devices[-1]
        consumer = torch.cuda.Stream(device=last)
        source_streams = {d: torch.cuda.Stream(device=d) for d in (first, last)}
        seed = torch.arange(8 * dim, dtype=torch.float32).reshape(8, dim) / 128
        seed_gpu = {d: seed.to(d) for d in source_streams}
        synchronize(devices)  # setup only, never between handoff and consumption
        pending = []
        # Keep input/output owners until the observer drain; this check focuses
        # on event ordering, not allocator reuse or user-owned input mutation.
        keepalive = []
        for source in (first, last):
            producer = source_streams[source]
            other = last if source == first else first
            with caller_stream(producer, other), torch.no_grad():
                x = torch.empty_like(seed_gpu[source], requires_grad=True)
                # A short GPU delay makes a missing producer dependency visible
                # without imposing a CPU/device synchronization.
                with torch.cuda.device(source):
                    torch.cuda._sleep(2_000_000)
                x.copy_(seed_gpu[source])
                handle = pipe.infer_submit(x, n_microbatches=m)
                opened = pipe.infer_open(m)
                for i in (2, 0, 3, 1):
                    opened.submit_mb(i, x.chunk(m)[i])
            keepalive.extend((x, handle, opened))
            with caller_stream(consumer, first):
                # Actually enqueue dependent kernels BEFORE the final drain.
                # Merely comparing raw handles after synchronize hides races.
                out = handle.result()
                check_no_grad(tag, out, pipe, errs)
                pending.append((f"{source} result", (out + 0.25).clone(), 0))
                for i in range(m):
                    mb = opened.wait_mb(i)
                    check_no_grad(tag, mb, pipe, errs)
                    pending.append((f"{source} wait_mb{i}", (mb + 0.25).clone(), i + 1))
                keepalive.extend((out, mb))
        synchronize(devices)
        ref = ref_pipe.infer(seed, n_microbatches=m)
        for name, actual, index in pending:
            expected = ref if index == 0 else ref.chunk(m)[index - 1]
            check_exact(f"{tag} {name}", actual, expected + 0.25, errs)

        scheduler_weight = nn.Parameter(torch.tensor(0.875, device=last))
        synchronize(devices)

        def update(out, i, t):
            assert not torch.is_grad_enabled(), "infer_loop scheduler enabled gradients"
            assert out.grad_fn is None and not out.requires_grad
            # Reading a trainable scalar exposes a missing no_grad even if the
            # worker returned an already-detached output.
            return out * scheduler_weight + (i + t) / 128

        with caller_stream(source_streams[last], first), torch.no_grad():
            x = seed_gpu[last].clone().requires_grad_()
        # Switch source->consumer explicitly for our own x construction; API
        # handoffs themselves must provide all subsequent ordering.
        consumer.wait_stream(source_streams[last])
        continued = []
        with caller_stream(consumer, first):
            for _ in range(3):
                x = pipe.infer_loop(x, steps=steps, update_fn=update, n_microbatches=m)
                check_no_grad(tag, x, pipe, errs)
                continued.append(x.clone())
        synchronize(devices)
        x_ref = seed
        with torch.no_grad():
            for run, actual in enumerate(continued):
                for t in range(steps):
                    x_ref = ref_pipe.infer(x_ref, n_microbatches=m)
                    if t != steps - 1:
                        x_ref = torch.cat([update(mb, i, t)
                                           for i, mb in enumerate(x_ref.chunk(m))])
                check_exact(f"{tag} continued loop {run}", actual, x_ref, errs)
        assert scheduler_weight.grad is None
    finally:
        pipe.close()
        if ref_pipe is not None:
            ref_pipe.close()
    print(f"  {'PASS' if len(errs) == n_before else 'FAIL'} {tag}")


def check_validation_and_failures(devices, errs):
    tag = f"[{devices[0]}..{devices[-1]} validation/failure]"
    n_before = len(errs)
    pipe = Pipeline(stage_modules=make_stages(len(devices), 16, seed=7), devices=devices)
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

    bad = Pipeline(stage_modules=make_stages(len(devices) - 1, 16, seed=8) + [Boom()],
                   devices=devices)
    try:
        hb = bad.infer_open(2)
        hb.submit_mb(0, x[:2])
        # Leave mb1 unsubmitted: failure must also release its result mailbox.
        # A whole-batch submit can race the deliberately immediate exception.
        for name, read in (("wait_mb", lambda: hb.wait_mb(0)),
                           ("missing wait_mb", lambda: hb.wait_mb(1)),
                           ("result", hb.result)):
            try:
                read()
                errs.append(f"{tag} worker {name}: no RuntimeError raised")
            except RuntimeError as e:
                if not isinstance(e.__cause__, ValueError):
                    errs.append(f"{tag} worker {name}: cause is "
                                f"{type(e.__cause__).__name__}, expected ValueError")
        expect("poisoned session submit",
               lambda: bad.infer_submit(x, n_microbatches=2), RuntimeError)

        # A partially submitted open batch cannot ever finish after close.
        # Its completed output survives; missing slots must wake with an error.
        partial = pipe.infer_open(2)
        partial.submit_mb(0, x[:2])
        completed = partial.wait_mb(0)
        pipe.close()
        check_exact(f"{tag} partial completed", partial.wait_mb(0), completed, errs)
        expect("partial close wait_mb", lambda: partial.wait_mb(1), RuntimeError)
        expect("partial close result", partial.result, RuntimeError)
        expect("partial close submit_mb", lambda: partial.submit_mb(1, x[2:]), RuntimeError)
    finally:
        # close() is idempotent with failed and live (idle) sessions.
        bad.close()
        bad.close()
        pipe.close()
        pipe.close()
    print(f"  {'PASS' if len(errs) == n_before else 'FAIL'} {tag}")


def check_close_in_flight(devices, errs):
    tag = f"[{devices[0]}..{devices[-1]} close in flight]"
    n_before = len(errs)
    entered, release = threading.Event(), threading.Event()

    class Gate(nn.Module):
        def forward(self, x):
            entered.set()
            if not release.wait(10):
                raise RuntimeError("close check gate timed out")
            return x + 1

    pipe = Pipeline(stage_modules=[Gate()] + [nn.Identity() for _ in devices[1:]],
                    devices=devices)
    close_errors = []
    closer = None
    x = torch.arange(64, dtype=torch.float32).reshape(4, 16)
    try:
        handle = pipe.infer_submit(x, n_microbatches=2)
        assert entered.wait(10), "first stage never entered close check gate"
        workers = list(pipe._infer_sess._workers)

        def close():
            try:
                pipe.close()
            except BaseException as error:
                close_errors.append(error)

        closer = threading.Thread(target=close, daemon=True)
        closer.start()
        # Give downstream workers a chance to consume an incorrectly early
        # shutdown sentinel while the first stage is deliberately blocked.
        for worker in workers[1:]:
            worker.join(0.1)
        release.set()
        closer.join(10)
        assert not closer.is_alive(), "close() did not drain submitted work"
        if close_errors:
            raise close_errors[0]
        assert not any(w.is_alive() for w in workers), "close left a worker alive"
        # All workers have stopped: inspect readiness before wait_mb to make a
        # dropped queued output an immediate diagnostic instead of a hang.
        if not all(box._ready.is_set() for box in handle._boxes):
            errs.append(f"{tag}: close dropped submitted microbatches")
        else:
            check_exact(tag, handle.result(), x + 1, errs)
    finally:
        release.set()
        if closer is not None:
            closer.join(10)
        pipe.close()
    print(f"  {'PASS' if len(errs) == n_before else 'FAIL'} {tag}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--devices", help="comma-separated devices (at least two); "
                    "default: CPU plus up to four visible CUDA devices")
    ap.add_argument("--bf16", action="store_true",
                    help="also check BF16 caller streams when every requested GPU supports it")
    ap.add_argument("--timeout", type=float, default=120,
                    help="per-check timeout in seconds (aborts suite on a hang)")
    args = ap.parse_args()
    if args.timeout <= 0:
        ap.error("--timeout must be positive")
    errs = []
    cuda = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if args.devices:
        devices = [str(torch.device(d.strip())) for d in args.devices.split(",")]
        if len(devices) < 2:
            ap.error("--devices needs at least two devices")
        if not all(d == "cpu" for d in devices) and not all(d in cuda for d in devices):
            ap.error("use only CPU devices or explicit visible cuda:N devices")
        if devices[0] != "cpu" and len(set(devices)) != len(devices):
            ap.error("CUDA stream checks need distinct GPUs")
        device_sets = [devices]
    else:
        device_sets = [["cpu", "cpu"]]
        if len(cuda) >= 2:
            device_sets.append(cuda[:4])

    def run(fn, devices, **kwargs):
        print(f"  RUN {fn.__name__} on {','.join(devices)} {kwargs}", flush=True)
        bounded(lambda: fn(devices, errs, **kwargs), args.timeout)

    for devices in device_sets:
        for fn in (check_submit_parity, check_trickle, check_overlap,
                   check_infer_loop, check_tuple_outputs,
                   check_validation_and_failures, check_close_in_flight):
            run(fn, devices)
        if devices[0] != "cpu":
            run(check_infer_loop, devices, offload=True)
            precisions = [False]
            if args.bf16:
                supported = []
                for device in devices:
                    with torch.cuda.device(device):
                        supported.append(torch.cuda.is_bf16_supported())
                if all(supported):
                    precisions.append(True)
                else:
                    print(f"  SKIP bf16: unsupported on {devices}")
            for bf16 in precisions:
                for window in (None, 1, 2):
                    run(check_cuda_caller_streams, devices, bf16=bf16, window=window)

    if errs:
        print(f"\n{len(errs)} FAILURE(S):")
        for e in errs:
            print("  ", e)
        sys.exit(1)
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
