"""
nvme_h2d_contention_test.py
---------------------------
Empirically test whether streaming weights from NVMe to the GPU is
independent of ordinary pinned-CPU->GPU (H2D) copies, or whether the two
contend — the modeling assumption behind the ``nvme`` channel in
``ramtorch/offload_simulator.py``.

Without GPUDirect Storage the "NVMe -> GPU" path is really two hops:

    disk --(DMA)--> host RAM --(H2D copy engine, PCIe)--> GPU

so the second hop shares the H2D copy engine and the GPU's PCIe link with
any concurrent CPU->GPU traffic. This script measures how much that hurts:

  phase 1  solo disk -> pinned RAM        (O_DIRECT, page cache bypassed)
  phase 2  solo pinned RAM -> GPU (H2D)
  phase 3  solo disk -> GPU pipeline      (load 1 GB, free it, x iters)
  phase 4  CONCURRENT: phase-3 pipeline + a pure H2D stream
  phase 5  control: two concurrent pure H2D streams (same-engine sharing)
  phase 6  phase 4 again under torch.profiler (chrome trace + memcpy table)

If NVMe->GPU were truly independent, phase 4 would show both sides at their
solo bandwidth. Full contention would look like phase 5.

Usage:
    python examples/nvme_h2d_contention_test.py [--size-gb 1] [--iters 10]
        [--dir DIR] [--device cuda:N] [--trace /tmp/trace.json]
"""

from __future__ import annotations

import argparse
import os
import threading
import time
from dataclasses import dataclass, field
from typing import List

import torch

GiB = 1 << 30
CHUNK = 64 << 20          # 64 MiB chunks: big enough for full disk bw,
                          # small enough to pipeline read/copy hops
O_DIRECT_ALIGN = 4096


# ── helpers ───────────────────────────────────────────────────────────────────

def make_test_file(path: str, size: int) -> None:
    if os.path.exists(path) and os.path.getsize(path) == size:
        return
    print(f"writing {size / GiB:.1f} GiB test file at {path} ...")
    block = os.urandom(64 << 20)
    with open(path, "wb") as f:
        written = 0
        while written < size:
            n = min(len(block), size - written)
            f.write(block[:n])
            written += n
        f.flush()
        os.fsync(f.fileno())


def pinned_bytes(size: int) -> torch.Tensor:
    t = torch.empty(size, dtype=torch.uint8, pin_memory=True)
    assert t.data_ptr() % O_DIRECT_ALIGN == 0, "pinned buffer not 4K-aligned"
    return t


def read_odirect(fd: int, buf_mv: memoryview, offset: int, length: int) -> None:
    """Read [offset, offset+length) from fd into buf_mv (O_DIRECT-safe)."""
    done = 0
    while done < length:
        n = os.preadv(fd, [buf_mv[done:length]], offset + done)
        if n <= 0:
            raise IOError(f"short read at {offset + done}")
        done += n


@dataclass
class Rates:
    """Per-iteration achieved bandwidth in GB/s."""
    label: str
    gbps: List[float] = field(default_factory=list)

    def add(self, nbytes: int, seconds: float) -> None:
        self.gbps.append(nbytes / GiB / seconds)

    @property
    def mean(self) -> float:
        return sum(self.gbps) / len(self.gbps) if self.gbps else 0.0

    def show(self) -> str:
        per = " ".join(f"{g:5.2f}" for g in self.gbps)
        return f"{self.label:<34} mean {self.mean:5.2f} GB/s   [{per}]"


# ── phases ────────────────────────────────────────────────────────────────────

def solo_disk_read(path: str, staging: torch.Tensor, size: int,
                   iters: int) -> Rates:
    r = Rates("disk -> pinned RAM (O_DIRECT)")
    mv = memoryview(staging.numpy())
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        for _ in range(iters):
            t0 = time.perf_counter()
            read_odirect(fd, mv, 0, size)
            r.add(size, time.perf_counter() - t0)
    finally:
        os.close(fd)
    return r


def h2d_loop(src: torch.Tensor, dst: torch.Tensor, stream: torch.cuda.Stream,
             iters: int, rates: Rates, stop: threading.Event = None) -> None:
    """Pure pinned->GPU copies on `stream`. If `stop` is given, loop until it
    is set (concurrent mode) instead of a fixed count."""
    n = src.numel()
    i = 0
    while (stop is None and i < iters) or (stop is not None and not stop.is_set()):
        t0 = time.perf_counter()
        with torch.cuda.stream(stream):
            dst.copy_(src, non_blocking=True)
        stream.synchronize()
        rates.add(n, time.perf_counter() - t0)
        i += 1


def disk_to_gpu_pipeline(path: str, staging: List[torch.Tensor], size: int,
                         iters: int, stream: torch.cuda.Stream,
                         device: torch.device, rates: Rates) -> None:
    """Load `size` bytes disk->GPU (double-buffered chunks), free, repeat.

    Chunk k: O_DIRECT read into staging[k % 2], then async H2D on `stream`
    while the next chunk reads into the other staging buffer.
    """
    mvs = [memoryview(s.numpy()) for s in staging]
    events = [torch.cuda.Event(), torch.cuda.Event()]
    for ev in events:
        ev.record()  # make first queries valid
    fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
    try:
        for _ in range(iters):
            t0 = time.perf_counter()
            gpu = torch.empty(size, dtype=torch.uint8, device=device)
            for k, off in enumerate(range(0, size, CHUNK)):
                b = k % 2
                n = min(CHUNK, size - off)
                events[b].synchronize()      # staging[b] free to overwrite?
                read_odirect(fd, mvs[b], off, n)
                with torch.cuda.stream(stream):
                    gpu[off:off + n].copy_(staging[b][:n], non_blocking=True)
                    events[b].record()
            stream.synchronize()
            rates.add(size, time.perf_counter() - t0)
            del gpu                          # "load 1 GB then delete it"
    finally:
        os.close(fd)


# ── main ──────────────────────────────────────────────────────────────────────

def pick_device() -> torch.device:
    best, best_free = 0, -1
    for i in range(torch.cuda.device_count()):
        free, _ = torch.cuda.mem_get_info(i)
        if free > best_free:
            best, best_free = i, free
    return torch.device(f"cuda:{best}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size-gb", type=float, default=1.0)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--dir", type=str, default=os.path.dirname(__file__) or ".")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--trace", type=str, default="/tmp/nvme_h2d_trace.json")
    args = ap.parse_args()

    size = int(args.size_gb * GiB) // O_DIRECT_ALIGN * O_DIRECT_ALIGN
    device = torch.device(args.device) if args.device else pick_device()
    props = torch.cuda.get_device_properties(device)
    print(f"device: {device} ({props.name}), test size {size / GiB:.2f} GiB, "
          f"{args.iters} iters")

    path = os.path.join(args.dir, "_nvme_h2d_testfile.bin")
    make_test_file(path, size)

    torch.cuda.set_device(device)
    stream_a = torch.cuda.Stream(device)     # disk->GPU pipeline
    stream_b = torch.cuda.Stream(device)     # competing pure H2D
    staging = [pinned_bytes(CHUNK), pinned_bytes(CHUNK)]
    full_pinned = pinned_bytes(size)         # source for the pure H2D loop
    h2d_dst = torch.empty(size, dtype=torch.uint8, device=device)
    big_staging = pinned_bytes(size)         # for solo disk phase

    # warm up CUDA / allocator
    h2d_dst.copy_(full_pinned, non_blocking=True)
    torch.cuda.synchronize()

    print("\n-- phase 1: solo disk read --")
    r_disk = solo_disk_read(path, big_staging, size, args.iters)
    print(r_disk.show())

    print("\n-- phase 2: solo H2D --")
    r_h2d = Rates("pinned -> GPU (H2D)")
    h2d_loop(full_pinned, h2d_dst, stream_b, args.iters, r_h2d)
    print(r_h2d.show())

    print("\n-- phase 3: solo disk -> GPU pipeline --")
    r_pipe = Rates("disk -> GPU (load 1GB, free, x N)")
    disk_to_gpu_pipeline(path, staging, size, args.iters, stream_a, device,
                         r_pipe)
    print(r_pipe.show())

    print("\n-- phase 4: CONCURRENT disk->GPU pipeline + pure H2D --")
    r_pipe_c = Rates("disk -> GPU  (concurrent)")
    r_h2d_c = Rates("pinned -> GPU (concurrent)")
    stop = threading.Event()
    tb = threading.Thread(target=h2d_loop, args=(full_pinned, h2d_dst,
                                                 stream_b, 0, r_h2d_c, stop))
    tb.start()
    disk_to_gpu_pipeline(path, staging, size, args.iters, stream_a, device,
                         r_pipe_c)
    stop.set()
    tb.join()
    print(r_pipe_c.show())
    print(r_h2d_c.show())

    print("\n-- phase 5: control, two concurrent pure H2D streams --")
    r_b1 = Rates("H2D stream 1 (concurrent)")
    r_b2 = Rates("H2D stream 2 (concurrent)")
    dst2 = torch.empty(size, dtype=torch.uint8, device=device)
    stop = threading.Event()
    t2 = threading.Thread(target=h2d_loop, args=(big_staging, dst2,
                                                 stream_a, 0, r_b2, stop))
    t2.start()
    h2d_loop(full_pinned, h2d_dst, stream_b, args.iters, r_b1)
    stop.set()
    t2.join()
    print(r_b1.show())
    print(r_b2.show())

    print("\n-- phase 6: concurrent run under torch.profiler --")
    prof_iters = max(2, args.iters // 4)
    r_pp = Rates("disk -> GPU  (profiled)")
    r_hp = Rates("pinned -> GPU (profiled)")
    stop = threading.Event()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA],
    ) as prof:
        tb = threading.Thread(target=h2d_loop, args=(full_pinned, h2d_dst,
                                                     stream_b, 0, r_hp, stop))
        tb.start()
        disk_to_gpu_pipeline(path, staging, size, prof_iters, stream_a,
                             device, r_pp)
        stop.set()
        tb.join()
    prof.export_chrome_trace(args.trace)
    print(f"chrome trace written to {args.trace}")
    memcpy_rows = [
        e for e in prof.key_averages() if "Memcpy" in e.key or "copy_" in e.key
    ]
    for e in memcpy_rows:
        cuda_t = getattr(e, "self_device_time_total",
                         getattr(e, "self_cuda_time_total", 0.0))
        print(f"  {e.key:<40} count={e.count:<5} "
              f"cuda_total={cuda_t / 1e3:9.1f} ms")

    # ── verdict ──────────────────────────────────────────────────────────
    print("\n== summary (mean GB/s) ==")
    for r in (r_disk, r_h2d, r_pipe, r_pipe_c, r_h2d_c, r_b1, r_b2):
        print(f"  {r.show()}")
    keep_h2d = 100 * r_h2d_c.mean / r_h2d.mean if r_h2d.mean else 0
    keep_pipe = 100 * r_pipe_c.mean / r_pipe.mean if r_pipe.mean else 0
    ctrl_keep = 100 * (r_b1.mean + r_b2.mean) / r_h2d.mean if r_h2d.mean else 0
    print(f"\n  concurrent H2D keeps  {keep_h2d:5.1f}% of its solo bandwidth")
    print(f"  concurrent disk->GPU keeps {keep_pipe:5.1f}% of its solo bandwidth")
    print(f"  control (2x H2D) aggregate = {ctrl_keep:5.1f}% of solo H2D "
          f"(100% = one shared engine, fully serialized)")

    os.unlink(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
