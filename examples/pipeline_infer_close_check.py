"""Regression checks for closing streaming inference with outstanding work.

Resident checks hold stage 0 behind an Event until close() has started joining
that worker. This deterministically catches sending every stage its stop marker
before upstream stages finish forwarding, without relying on sleeps or workload
size. Offload checks exercise ordinary live close with two chunks per stage and
windows 1 and 2; they intentionally do not gate or mutate streamed modules.

Also check partially submitted handles: submitted slots survive close, missing
slots wake with an error, further submission fails, and repeated close is safe.
All checks run in daemon threads with deadlines so a broken wait/close cannot
hang this test process. CUDA comparisons synchronize at observation boundaries
and require finite, dtype/shape-identical, byte-identical outputs.

Run from the repository root:
    PYTHONPATH=. env/bin/python examples/pipeline_infer_close_check.py
    PYTHONPATH=. env/bin/python examples/pipeline_infer_close_check.py --devices cpu,cpu
    PYTHONPATH=. env/bin/python examples/pipeline_infer_close_check.py --devices cuda:0,cuda:1

The default covers CPU and all visible CUDA devices (two stages on cuda:0 when
only one GPU is visible). Same-GPU stages exercise CUDA ordering, not PCIe relay.
"""

import argparse
import math
import threading
import time

import torch
import torch.nn as nn

from ramtorch import Pipeline


def bounded(call, timeout):
    """Preserve one caller's CUDA context; abandon the suite on any timeout."""
    results, errors = [], []

    def run():
        try:
            results.append(call())
        except BaseException as error:
            errors.append(error)

    worker = threading.Thread(target=run, daemon=True, name="infer-close-check")
    worker.start()
    worker.join(timeout)
    if worker.is_alive():
        raise AssertionError(f"shutdown check timed out after {timeout:g}s")
    if errors:
        raise errors[0]
    return results[0]


def synchronize(devices):
    for device in dict.fromkeys(map(torch.device, devices)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)


def check_exact(got, expected):
    got = got.detach().cpu().contiguous()
    expected = expected.detach().cpu().contiguous()
    assert got.shape == expected.shape, (got.shape, expected.shape)
    assert got.dtype == expected.dtype, (got.dtype, expected.dtype)
    assert torch.isfinite(got).all(), "non-finite output"
    assert torch.isfinite(expected).all(), "non-finite reference"
    assert torch.equal(got.view(torch.uint8), expected.view(torch.uint8)), (
        f"output bytes differ; max error {(got - expected).abs().max().item():.3e}"
    )


def expect_runtime_error(call, message):
    try:
        call()
    except RuntimeError as error:
        assert message in str(error), f"unexpected error: {error}"
    else:
        raise AssertionError(f"expected RuntimeError containing {message!r}")


class StageZeroGate(nn.Module):
    """A host-side gate used only on resident modules, never offloaded chunks."""

    def __init__(self, module, timeout):
        super().__init__()
        self.module = module
        self.timeout = timeout
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()

    def arm(self):
        self.entered.clear()
        self.release.clear()

    def forward(self, x):
        self.entered.set()
        if not self.release.wait(self.timeout):
            raise RuntimeError("stage-0 test gate timed out")
        return self.module(x)


def close_in_flight(pipe, handle, gate, timeout):
    session = handle._session
    workers = session._workers
    started = threading.Event()
    joining_upstream = threading.Event()
    errors = []
    original_join = workers[0].join
    closer = None

    def observed_join(*args, **kwargs):
        # Both the old and repaired implementations first join worker 0.
        # At this point the old implementation has put downstream sentinels
        # ahead of all gated work, whereas the repair has stopped only stage 0.
        # Observing this boundary makes the regression deterministic even if
        # the main thread sees _closed before close has enqueued any sentinel.
        joining_upstream.set()
        return original_join(*args, **kwargs)

    def close():
        started.set()
        try:
            pipe.close()
        except BaseException as error:
            errors.append(error)

    try:
        if gate is not None:
            assert gate.entered.wait(timeout), "stage 0 never entered the gate"
            assert not any(box._ready.is_set() for box in handle._boxes), (
                "a result escaped the stage-0 gate"
            )
            workers[0].join = observed_join
        closer = threading.Thread(target=close, daemon=True, name="infer-closer")
        closer.start()
        assert started.wait(timeout), "close thread never started"
        deadline = time.monotonic() + timeout
        while not session._closed:
            remaining = deadline - time.monotonic()
            assert remaining > 0, "close never marked the session closed"
            # A bounded Event wait yields without using a sleep as race proof.
            joining_upstream.wait(min(0.01, remaining))
        if gate is not None:
            assert joining_upstream.wait(timeout), "close never joined stage 0"
            assert closer.is_alive(), "close returned before gated work drained"
            gate.release.set()
        closer.join(timeout)
        assert not closer.is_alive(), "close did not drain submitted work"
        if errors:
            raise errors[0]
        assert not any(worker.is_alive() for worker in workers), (
            "close left an inference worker alive"
        )
        assert session.error is None, f"unexpected worker error: {session.error}"
        # Inspect before wait_mb/result: a lost output must be an immediate
        # failure, not a blocked read. Partial handles' missing slots also wake.
        missing = [i for i, box in enumerate(handle._boxes)
                   if not box._ready.is_set()]
        assert not missing, f"close left result mailboxes blocked: {missing}"
    finally:
        if gate is not None:
            gate.release.set()
        if closer is not None:
            closer.join(timeout)
        workers[0].join = original_join


def check_shutdown(devices, window, partial, timeout):
    torch.manual_seed(2026)
    dim = 128
    chunks = [[nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                             nn.Linear(dim * 2, dim))
               for _ in range(2)] for _ in devices]
    gate = None
    if window is None:
        stages = [nn.Sequential(*stage_chunks) for stage_chunks in chunks]
        gate = StageZeroGate(stages[0], timeout)
        stages[0] = gate
        pipe = Pipeline(stage_modules=stages, devices=devices)
    else:
        pipe = Pipeline(stage_modules=chunks, devices=devices,
                        offload_window=window, offload_pin=0)

    try:
        n_microbatches = 4 if partial else 32
        x = torch.randn(n_microbatches * 2, dim)
        # Same pipeline, dtype, shapes and operation order; the synchronous
        # path provides a fully drained reference before the gate is armed.
        reference = pipe.infer(x, n_microbatches=n_microbatches)
        synchronize(devices)
        reference = reference.detach().cpu()
        if gate is not None:
            gate.arm()
        if partial:
            handle = pipe.infer_open(4)
            inputs = x.chunk(4)
            for mb in (0, 2):
                handle.submit_mb(mb, inputs[mb])
        else:
            handle = pipe.infer_submit(x, n_microbatches=n_microbatches)

        close_in_flight(pipe, handle, gate, timeout)
        # Exercise both session idempotence and Pipeline's cleared-session path
        # before reading outputs; neither may invalidate delivered values.
        handle._session.close()
        pipe.close()
        synchronize(devices)
        if partial:
            expected = reference.chunk(4)
            for mb in (0, 2):
                check_exact(handle.wait_mb(mb), expected[mb])
            for mb in (1, 3):
                expect_runtime_error(lambda mb=mb: handle.wait_mb(mb), "aborted")
            expect_runtime_error(handle.result, "aborted")
            expect_runtime_error(lambda: handle.submit_mb(1, inputs[1]), "closed")
        else:
            check_exact(handle.result(), reference)
        synchronize(devices)
    finally:
        if gate is not None:
            gate.release.set()
        pipe.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", help="comma-separated CPU or visible cuda:N "
                        "devices, at least two stages; default: CPU plus CUDA")
    parser.add_argument("--timeout", type=float, default=120,
                        help="whole-check deadline in seconds; abort on a hang")
    args = parser.parse_args()
    if not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("--timeout must be finite and positive")
    cuda = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    if args.devices:
        try:
            devices = [str(torch.device(d.strip())) for d in args.devices.split(",")]
        except (RuntimeError, ValueError) as error:
            parser.error(str(error))
        if len(devices) < 2:
            parser.error("--devices requires at least two stages")
        if not (all(d == "cpu" for d in devices) or all(d in cuda for d in devices)):
            parser.error("use only CPU devices or explicit visible cuda:N devices")
        device_sets = [devices]
    else:
        device_sets = [["cpu", "cpu"]]
        if cuda:
            device_sets.append(cuda if len(cuda) >= 2 else cuda * 2)

    torch.set_num_threads(1)
    for devices in device_sets:
        for window in (None, 1, 2):
            kind = "resident gated" if window is None else f"offload window={window}"
            for partial in (False, True):
                case = "partial 0+2/4" if partial else "full 32"
                label = f"{','.join(devices)} {kind} {case}"
                print(f"CHECK {label}", flush=True)
                bounded(lambda: check_shutdown(devices, window, partial,
                                               args.timeout / 4), args.timeout)
                print(f"  PASS {label}", flush=True)
    print("ALL SHUTDOWN CHECKS PASSED")


if __name__ == "__main__":
    main()
