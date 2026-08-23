"""
offload_inference_memory.py
---------------------------
Gauge what an INFERENCE-ONLY deployment costs across the whole offload
spectrum, sweeping the pinned/streamed split at runtime with
``OffloadModel.set_pinned`` (no rebuild per config — one model, one set of
weights, retiered in place between passes).

Inference is the friendliest regime for offload: no backward, so no gradient
accumulators (they are allocated on a chunk's first backward, so a model that
only ever runs forwards holds nothing but masters + the window), no D2H
gradient traffic, and no activation retention. The only trade is weights
crossing PCIe per touch, which is what this sweep measures.

Reads as: 100% offloaded = every chunk streams from pinned host RAM through
the ``window`` GPU slots; 0% offloaded = every chunk pinned resident (the
full-resident model, still going through the engine). Between them you buy
latency back with GPU memory, one chunk at a time.

Reported per config:

* ``GPU rest``  — allocated bytes with the model idle (masters of the pinned
  chunks; the streamed ones live in host RAM)
* ``GPU peak``  — peak during a forward (adds the window + live activations)
* ``host pin``  — pinned (non-pageable) host RAM holding streamed masters
* ``H2D/pass``  — weight bytes streamed per forward (from the engine's own
  load counter, so it includes window reuse)
* ``ms/pass``   — median wall time, and the speedup vs the same model fully
  streamed

GPU numbers exclude a measured floor (the CUDA context's fixed scratch —
cuBLAS workspaces, ~8 MiB per stream, which ``memory_allocated`` counts),
taken with the model holding no weights on the device, so ``GPU rest`` is
exactly the pinned masters and ``GPU rest + host pin`` is constant across the
sweep.

Every config's output is checked bit-identical (``torch.equal``) against the
first: the tier is a memory placement, never a numerical choice.

The interesting non-linearity is that ``H2D/pass`` hits zero as soon as
``window + pinned >= n``: the streamed chunks all fit in the window at once
and are never evicted, so the second pass onward is fully resident at a
window's worth of extra GPU memory. Sample run, 8 x 32 MiB chunks, batch 256,
W=2 on an H100 (256 MiB of weights, plain resident peak 267 MiB, 1.9 ms):

    offload   GPU rest   GPU peak   host pin   H2D/pass   ms/pass
       100%        0.0       66.0      256.2      256.2      12.6
        75%       64.0      138.1      192.1      192.1      10.8
        50%      128.1      202.1      128.1      128.1       6.1
        25%      192.1      234.1       64.0        0.0       2.2
         0%      256.2      266.2        0.0        0.0       2.2

Pipeline analogue (``--pipeline cuda:1,cuda:3``): the same sweep driven by
``Pipeline.set_offload_pinned(k)`` + ``Pipeline.infer``, measured per stage
GPU.

Run:  PYTHONPATH=. python examples/offload_inference_memory.py
      PYTHONPATH=. python examples/offload_inference_memory.py --device cuda:1 \
          --chunks 12 --dim 2048 --hidden 8192 --batch 512
      PYTHONPATH=. python examples/offload_inference_memory.py \
          --pipeline cuda:1,cuda:3
"""

import argparse
import copy
import time

import torch
import torch.nn as nn

from ramtorch import Pipeline
from ramtorch.offload import OffloadModel


def make_chunks(n, dim, hidden, seed=0):
    """A transformer-block-ish MLP stack: uniform, easy to reason about."""
    torch.manual_seed(seed)
    return [nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                          nn.Linear(hidden, dim)) for _ in range(n)]


def pin_levels(n, points):
    """``points`` pinned-chunk counts spanning 0..n, endpoints included."""
    if points >= n + 1:
        return list(range(n + 1))
    return sorted({round(i * n / (points - 1)) for i in range(points)})


class Meter:
    """CUDA allocation / timing probe that degrades to timing-only on CPU."""

    def __init__(self, devices):
        self.devices = [torch.device(d) for d in devices]
        self.cuda = all(d.type == "cuda" for d in self.devices)

    def sync(self):
        for d in self.devices:
            if d.type == "cuda":
                torch.cuda.synchronize(d)

    def allocated(self):
        self.sync()
        return {str(d): (torch.cuda.memory_allocated(d) if d.type == "cuda"
                         else 0) for d in self.devices}

    def reset_peak(self):
        self.sync()
        for d in self.devices:
            if d.type == "cuda":
                torch.cuda.reset_peak_memory_stats(d)

    def peak(self):
        self.sync()
        return {str(d): (torch.cuda.max_memory_allocated(d) if d.type == "cuda"
                         else 0) for d in self.devices}

    def time(self, fn, iters, warmup=True):
        """Median wall time of ``fn``, plus its last return value."""
        if warmup:
            fn()                        # allocator + post-retier residency
            self.sync()
        times, out = [], None
        for _ in range(iters):
            t0 = time.perf_counter()
            out = fn()
            self.sync()
            times.append(time.perf_counter() - t0)
        times.sort()
        return times[len(times) // 2], out


MIB = 2 ** 20


def purge_residency(model):
    """Drop every streamed chunk's GPU copy (window slots included).

    A retier wipes residency, but only if something actually moves — a no-op
    ``set_pinned`` returns early — so bounce one chunk across the tier line
    and back. Only needed to get a weights-free measurement baseline; normal
    inference never has to do this.
    """
    if 0 in model.pinned_layers:
        model.unpin_chunks([0])
        model.pin_chunks([0])
    else:
        model.pin_chunks([0])
        model.unpin_chunks([0])


def sweep_single(device, n, dim, hidden, batch, window, iters, points):
    """Retier one OffloadModel across the offload spectrum, inference only."""
    meter = Meter([device])
    chunks = make_chunks(n, dim, hidden)
    ref_chunks = copy.deepcopy(chunks)
    model = OffloadModel(chunks, device=device, window=window, pin=0)
    torch.manual_seed(1)
    x = torch.randn(batch, dim)
    chunk_bytes = model._state[0].nbytes()

    print(f"model: {n} x [Linear({dim}->{hidden}) GELU Linear({hidden}->{dim})]"
          f" = {n * chunk_bytes / MIB:.1f} MiB of weights "
          f"({chunk_bytes / MIB:.1f} MiB/chunk), batch {batch}, W={window}, "
          f"{device}")
    print(f"inference only: median of {iters} forwards after a warmup pass"
          + ("" if meter.cuda else " (GPU columns read 0 on cpu)") + "\n")

    # A pass before the sweep so the CUDA context's fixed overheads (cuBLAS
    # workspaces, ~8 MiB per stream, which memory_allocated counts) are in
    # place for every row instead of landing on row 1; then purge residency so
    # the floor is exactly "context scratch, no weights on the GPU".
    key = str(torch.device(device))
    with torch.no_grad():
        model(x)
    purge_residency(model)
    floor = meter.allocated()[key]
    print(f"excluded floor (CUDA context scratch, no weights): "
          f"{floor / MIB:.1f} MiB\n")

    print(f"{'offload':>8s} {'pinned':>7s} {'GPU rest':>9s} {'GPU peak':>9s} "
          f"{'host pin':>9s} {'H2D/pass':>9s} {'ms/pass':>8s} {'vs 100%':>8s}")
    rows, base_out, base_ms = [], None, None
    mismatch = False
    for k in pin_levels(n, points):
        model.set_pinned(k)
        streamed = sorted(model.streamed_layers)
        host_pin = sum(model._state[i].nbytes() for i in streamed)
        rest = meter.allocated()[key] - floor
        with torch.no_grad():
            model(x)                    # warm the window into steady state
            meter.reset_peak()
            loads0 = model.stats["loads"]
            ms, out = meter.time(lambda: model(x), iters, warmup=False)
        peak = meter.peak()[key] - floor
        h2d = (model.stats["loads"] - loads0) / iters * chunk_bytes

        out = out.cpu()                 # keep the reference off the GPU budget
        if base_out is None:
            base_out, base_ms = out, ms
        elif not torch.equal(base_out, out):
            mismatch = True
        pct = 100.0 * len(streamed) / n
        rows.append((pct, k, rest, peak, host_pin, h2d, ms))
        print(f"{pct:7.0f}% {k:7d} {rest / MIB:9.1f} {peak / MIB:9.1f} "
              f"{host_pin / MIB:9.1f} {h2d / MIB:9.1f} {ms * 1e3:8.1f} "
              f"{base_ms / ms:7.2f}x")

    accs = sum(len(st.grad_acc) for st in model._state)
    print(f"\ngrad accumulators after the whole inference sweep: {accs} "
          f"(expected 0 — they are born on a chunk's first backward)")
    print(f"outputs across the sweep: "
          f"{'BIT-IDENTICAL' if not mismatch else 'MISMATCH (bug!)'}")
    model.close()
    del model, chunks               # `chunks` also holds the pinned masters
    if meter.cuda:
        torch.cuda.empty_cache()

    # honest baseline: the same weights fully resident, no engine at all.
    # Re-floored, because tearing the engine down freed part of the sweep's
    # floor (its staging buffers) along with it.
    rfloor = meter.allocated()[key]
    resident = nn.Sequential(*ref_chunks).to(device).eval()
    xd = x.to(device)
    meter.reset_peak()
    with torch.no_grad():
        ms, _ = meter.time(lambda: resident(xd), iters)
    rpeak = meter.peak()[key] - rfloor
    print(f"\nplain nn.Sequential on {device} (no engine, no window): "
          f"peak {rpeak / MIB:.1f} MiB, {ms * 1e3:.1f} ms/pass")
    cheap = min(rows, key=lambda r: r[3])
    saving = (f"{rpeak / cheap[3]:.1f}x less GPU memory than resident, "
              if meter.cuda and cheap[3] > 0 else "")
    print(f"cheapest config: {cheap[0]:.0f}% offloaded -> {saving}"
          f"{cheap[6] / ms:.1f}x the latency")
    del resident
    if meter.cuda:
        torch.cuda.empty_cache()


def sweep_pipeline(devices, n, dim, hidden, batch, window, iters, points):
    """Same sweep, driven stage-wise by Pipeline.set_offload_pinned."""
    meter = Meter(devices)
    per_stage = n // len(devices)
    chunks = make_chunks(per_stage * len(devices), dim, hidden, seed=3)
    pipe = Pipeline(chunk_modules=chunks, devices=list(devices),
                    offload_window=window, offload_pin=0)
    torch.manual_seed(2)
    x = torch.randn(batch, dim)
    chunk_bytes = pipe.stages[0].engine._state[0].nbytes()

    print(f"\npipeline: {len(devices)} stages x {per_stage} chunks of "
          f"{chunk_bytes / MIB:.1f} MiB on {', '.join(devices)}, "
          f"batch {batch}, W={window}")
    pipe.infer(x, n_microbatches=len(devices))       # context scratch warmup
    for st in pipe.stages:
        purge_residency(st)                          # weights-free baseline
    floor = meter.allocated()

    print(f"{'offload':>8s} {'pinned/stage':>13s} "
          + " ".join(f"{d + ' rest':>12s} {d + ' peak':>12s}"
                     for d in devices)
          + f" {'ms/pass':>8s}")
    base_out, mismatch = None, False
    for k in pin_levels(per_stage, points):
        pipe.set_offload_pinned(k)
        rest = meter.allocated()
        meter.reset_peak()
        ms, out = meter.time(
            lambda: pipe.infer(x, n_microbatches=len(devices)), iters)
        peak = meter.peak()
        out = out.cpu()                 # keep the reference off the GPU budget
        if base_out is None:
            base_out = out
        elif not torch.equal(base_out, out):
            mismatch = True
        cells = " ".join(
            f"{(rest[d] - floor[d]) / MIB:12.1f} "
            f"{(peak[d] - floor[d]) / MIB:12.1f}" for d in map(str, devices))
        pct = 100.0 * (per_stage - k) / per_stage
        print(f"{pct:7.0f}% {k:13d} {cells} {ms * 1e3:8.1f}")

    accs = sum(len(s.grad_acc) for st in pipe.stages for s in st.engine._state)
    print(f"grad accumulators after the sweep: {accs} (expected 0)")
    print(f"outputs across the sweep: "
          f"{'BIT-IDENTICAL' if not mismatch else 'MISMATCH (bug!)'}")
    pipe.close()


def main():
    ap = argparse.ArgumentParser(
        description="inference-only memory/latency sweep over the offload "
                    "percentage, retiered in place")
    ap.add_argument("--device", default="cuda:0",
                    help="single-GPU device (cpu works, minus the GPU columns)")
    ap.add_argument("--pipeline", default=None, metavar="DEV,DEV",
                    help="also sweep a pipeline across these devices")
    ap.add_argument("--chunks", type=int, default=8)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--points", type=int, default=5,
                    help="pinned levels to sample between 0 and --chunks")
    args = ap.parse_args()

    if args.points < 2:
        raise SystemExit("--points must be >= 2 (both ends of the sweep)")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("no CUDA available — falling back to --device cpu "
              "(memory columns will read 0)")
        args.device = "cpu"
    sweep_single(args.device, args.chunks, args.dim, args.hidden, args.batch,
                 args.window, args.iters, args.points)
    if args.pipeline:
        devs = [d.strip() for d in args.pipeline.split(",") if d.strip()]
        if len(devs) < 2:
            raise SystemExit("--pipeline needs at least two devices")
        if args.chunks < len(devs):
            raise SystemExit("--chunks must be >= the number of stages")
        sweep_pipeline(devs, args.chunks, args.dim, args.hidden, args.batch,
                       args.window, args.iters, args.points)


if __name__ == "__main__":
    main()
