"""
gds_vs_staged_bench.py
----------------------
Benchmark disk -> GPU weight loading: RamTorch's staged path (read into
pinned RAM, async H2D) vs PyTorch's ``torch.cuda.gds`` (cuFile).

True GPUDirect Storage needs the ``nvidia-fs`` kernel module and a supported
filesystem (ext4/xfs directly on NVMe). Without it, cuFile silently runs in
*compatibility mode* — POSIX reads into an internal bounce buffer + H2D —
which is architecturally the same staged path RamTorch implements, so the
comparison is implementation vs implementation. The script reports which
mode cuFile is in.

Methods (each loads the whole file to a GPU tensor, N iterations):
  staged O_DIRECT   chunked O_DIRECT pread -> pinned ping-pong -> async H2D
                    (always cold: bypasses the page cache by construction)
  staged buffered   same pipeline, buffered pread (cold = fadvise-dropped,
                    warm = page cache) — what cuFile compat does internally
  gds               GdsFile.load_storage (cold / warm variants)

Usage:
    python examples/gds_vs_staged_bench.py [--size-gb 1] [--iters 5]
        [--dir DIR] [--device cuda:N]
"""

from __future__ import annotations

import argparse
import os
import time

import torch

GiB = 1 << 30
CHUNK = 64 << 20
ALIGN = 4096


def make_file(path: str, size: int) -> None:
    if os.path.exists(path) and os.path.getsize(path) == size:
        return
    block = os.urandom(64 << 20)
    with open(path, "wb") as f:
        w = 0
        while w < size:
            n = min(len(block), size - w)
            f.write(block[:n])
            w += n
        f.flush()
        os.fsync(f.fileno())


def drop_cache(path: str) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)


def staged_load(path: str, size: int, gpu: torch.Tensor,
                staging: list, stream: torch.cuda.Stream,
                direct: bool) -> None:
    """Chunked read -> pinned ping-pong -> async H2D (the RamTorch path)."""
    mvs = [memoryview(s.numpy()) for s in staging]
    events = [torch.cuda.Event(), torch.cuda.Event()]
    for ev in events:
        ev.record()
    flags = os.O_RDONLY | (os.O_DIRECT if direct else 0)
    fd = os.open(path, flags)
    try:
        for k, off in enumerate(range(0, size, CHUNK)):
            b = k % 2
            n = min(CHUNK, size - off)
            events[b].synchronize()
            done = 0
            while done < n:
                r = os.preadv(fd, [mvs[b][done:n]], off + done)
                if r <= 0:
                    raise IOError(f"short read at {off + done}")
                done += r
            with torch.cuda.stream(stream):
                gpu[off:off + n].copy_(staging[b][:n], non_blocking=True)
                events[b].record()
        stream.synchronize()
    finally:
        os.close(fd)


def bench(label: str, fn, iters: int, size: int, results: list) -> None:
    rates = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        rates.append(size / GiB / (time.perf_counter() - t0))
    per = " ".join(f"{r:5.2f}" for r in rates)
    results.append((label, sum(rates) / len(rates)))
    print(f"  {label:<28} mean {results[-1][1]:5.2f} GB/s   [{per}]")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size-gb", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--dir", type=str, default=os.path.dirname(__file__) or ".")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    size = int(args.size_gb * GiB) // ALIGN * ALIGN
    if args.device:
        device = torch.device(args.device)
    else:
        best = max(range(torch.cuda.device_count()),
                   key=lambda i: torch.cuda.mem_get_info(i)[0])
        device = torch.device(f"cuda:{best}")
    torch.cuda.set_device(device)

    nvidia_fs = os.path.isdir("/proc/driver/nvidia-fs")
    print(f"device: {device} ({torch.cuda.get_device_name(device)})")
    print(f"nvidia-fs kernel module: {'LOADED — true GDS possible' if nvidia_fs else 'absent — cuFile will use compat mode (POSIX + bounce buffer)'}")
    print(f"size {size / GiB:.2f} GiB, {args.iters} iters\n")

    path = os.path.join(args.dir, "_gds_bench_file.bin")
    make_file(path, size)

    stream = torch.cuda.Stream(device)
    staging = [torch.empty(CHUNK, dtype=torch.uint8, pin_memory=True)
               for _ in range(2)]
    for s in staging:
        assert s.data_ptr() % ALIGN == 0
    gpu = torch.empty(size, dtype=torch.uint8, device=device)
    results: list = []

    # warm up CUDA
    gpu.fill_(0)
    torch.cuda.synchronize(device)

    print("-- staged (RamTorch path) --")
    bench("staged O_DIRECT (cold)",
          lambda: staged_load(path, size, gpu, staging, stream, direct=True),
          args.iters, size, results)
    bench("staged buffered (cold)",
          lambda: (drop_cache(path),
                   staged_load(path, size, gpu, staging, stream,
                               direct=False))[-1],
          args.iters, size, results)
    staged_load(path, size, gpu, staging, stream, direct=False)  # prime cache
    bench("staged buffered (warm)",
          lambda: staged_load(path, size, gpu, staging, stream, direct=False),
          args.iters, size, results)

    print("\n-- torch.cuda.gds (cuFile) --")
    try:
        gds_file = torch.cuda.gds.GdsFile(path, os.O_RDONLY)
        storage = gpu.untyped_storage()

        def gds_cold():
            drop_cache(path)
            gds_file.load_storage(storage, offset=0)

        drop_cache(path)
        bench("gds load_storage (cold)", gds_cold, args.iters, size, results)
        gds_file.load_storage(storage, offset=0)  # prime cache
        bench("gds load_storage (warm)",
              lambda: gds_file.load_storage(storage, offset=0),
              args.iters, size, results)
        del gds_file
    except Exception as e:  # noqa: BLE001 — report and continue
        print(f"  torch.cuda.gds unavailable: {type(e).__name__}: {e}")

    print("\n== summary (mean GB/s) ==")
    for label, mean in results:
        print(f"  {label:<28} {mean:6.2f}")

    os.unlink(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
