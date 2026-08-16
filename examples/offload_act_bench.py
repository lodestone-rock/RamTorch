"""
offload_act_bench.py
--------------------
Measure what engine activation offload (offload_activations=True) buys and
costs on real hardware — the probe's section-F measurement (notes §0.11)
rerun against the shipped implementation.

Model: 12 x MLP block [Linear(2048->8192) + GELU + Linear(8192->2048)]
streamed chunks, batch 2048, W=2, cuda:1 — a transformer-block-like shape
where the saved INTERNALS per chunk (GELU input + second linear's input,
2 x 64 MiB) dominate the 16 MiB chunk boundary. That matters because the
engine's backward cache must keep boundary tensors alive regardless (they
are the autograd roots/leaves that _grads_for differentiates), so hook
packets can only evacuate what the graph alone holds: internals in keep
mode, boundary copies in checkpoint mode.

For keep + checkpoint modes x {off, act_slots=2, act_slots=1}: peak GPU MiB
(steady state, after a warmup step) and median ms/step over 5 steps, plus
the engine's activation traffic counters.

Run:  PYTHONPATH=. python examples/offload_act_bench.py [--device cuda:1]
"""

import argparse
import time

import torch
import torch.nn as nn

from ramtorch.offload import OffloadModel

N, DIM, HIDDEN, BATCH, WINDOW, STEPS = 12, 2048, 8192, 2048, 2, 5


def make_chunks(seed=0):
    torch.manual_seed(seed)
    return [nn.Sequential(nn.Linear(DIM, HIDDEN), nn.GELU(),
                          nn.Linear(HIDDEN, DIM)) for _ in range(N)]


def bench(device, keep, act, act_slots):
    model = OffloadModel(make_chunks(), device=device, window=WINDOW,
                         keep_activations=keep,
                         offload_activations=act, act_slots=act_slots)
    torch.manual_seed(1)
    x = torch.randn(BATCH, DIM)
    y = torch.randn(BATCH, DIM)
    loss_fn = nn.functional.mse_loss

    model.step(x, targets=y, loss_fn=loss_fn)  # warmup (pools, allocator)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(STEPS):
        t0 = time.perf_counter()
        model.step(x, targets=y, loss_fn=loss_fn)
        torch.cuda.synchronize(device)
        times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated(device)
    stats = dict(model.stats)
    model.flush_grads()
    model.close()
    torch.cuda.empty_cache()
    times.sort()
    return peak, times[len(times) // 2], stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()
    dev = args.device

    print(f"n={N} dim={DIM} hidden={HIDDEN} batch={BATCH} W={WINDOW} on "
          f"{dev} (median of {STEPS} steps after warmup)")
    print(f"{'mode':12s} {'act':10s} {'peak MiB':>9s} {'ms/step':>8s} "
          f"{'offloads/step':>13s} {'MiB moved/step':>14s}")
    for keep in (True, "checkpoint"):
        label = "keep" if keep is True else "checkpoint"
        for act, slots in ((False, 2), (True, 2), (True, 1)):
            peak, med, stats = bench(dev, keep, act, slots)
            tag = f"slots={slots}" if act else "off"
            per_step = stats["act_offloads"] / (STEPS + 1)
            mib = stats["act_bytes_offloaded"] / (STEPS + 1) / 2**20
            print(f"{label:12s} {tag:10s} {peak / 2**20:9.0f} "
                  f"{med * 1e3:8.1f} {per_step:13.1f} {mib:14.0f}")


if __name__ == "__main__":
    main()
