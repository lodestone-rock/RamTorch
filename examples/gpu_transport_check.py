"""Bounded all-pairs CUDA transport diagnostic, independent of RamTorch.

Run on an explicitly available multi-GPU machine before pipeline tests::

    python examples/gpu_transport_check.py --devices 0,1,2,3

Checks exact finite FP32/BF16 patterns, blocking copies and nondefault producer/
copy/consumer streams with event ordering. At most a few MiB per tensor; this
is a correctness probe, not a stress test or a bandwidth measurement. Capability
flags do not prove which physical route CUDA used. No NCCL/process groups.
"""
import argparse
import itertools

import torch


def pattern(size, trial, dtype):
    # Dyadic bounded values are exact even in BF16. Include signs and zeros;
    # periodic-but-shifted pattern detects zeros, stale buffers and scrambling.
    index = torch.arange(size, dtype=torch.int64)
    return (((index * 37 + trial * 19) % 251 - 125).float() / 8).to(dtype)


def check_pair(source, destination, sizes, trials):
    producer = torch.cuda.Stream(device=source)
    copier = torch.cuda.Stream(device=source)
    consumer = torch.cuda.Stream(device=destination)
    checks = 0
    for dtype, size, trial in itertools.product((torch.float32, torch.bfloat16), sizes, range(trials)):
        host = pattern(size, trial, dtype)
        x = host.to(source)
        torch.cuda.synchronize(source)
        copied = x.to(destination, non_blocking=False)
        if not torch.equal(copied.cpu(), host):
            raise AssertionError(f"blocking corruption: {source}->{destination} {dtype} {size} trial={trial}")
        checks += 1
        # A distinct result must be produced on a nondefault stream; no device
        # synchronization until observing output. Keep all source/dest refs alive.
        with torch.cuda.device(source), torch.cuda.stream(producer):
            x = host.to(source, non_blocking=True)
            x = -x
            ready = producer.record_event()
        with torch.cuda.device(source), torch.cuda.stream(copier):
            copier.wait_event(ready)
            copied = x.to(destination, non_blocking=True)
            sent = copier.record_event()
        with torch.cuda.device(destination), torch.cuda.stream(consumer):
            consumer.wait_event(sent)
            result = -copied
            done = consumer.record_event()
        done.synchronize()
        if not torch.equal(result.cpu(), host):
            raise AssertionError(f"stream corruption: {source}->{destination} {dtype} {size} trial={trial}")
        checks += 1
    peer = torch.cuda.can_device_access_peer(source, destination)
    print(f"PASS {source}->{destination}: {checks} exact copies; peer_access={peer}", flush=True)
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--devices", default="0,1,2,3")
    parser.add_argument("--sizes", default="32,4096,262144,1048576", help="elements per tensor")
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    devices = [torch.device("cuda", int(index)) for index in args.devices.split(",")]
    sizes = [int(size) for size in args.sizes.split(",")]
    if len(devices) < 2 or len(set(devices)) != len(devices) or args.trials < 1 or min(sizes) < 1:
        parser.error("need at least two distinct CUDA devices and positive sizes/trials")
    if not torch.cuda.is_available() or any(device.index >= torch.cuda.device_count() for device in devices):
        parser.error("requested GPUs are unavailable")
    torch.set_num_threads(1)
    print(f"torch={torch.__version__} CUDA={torch.version.cuda}", flush=True)
    for device in devices:
        with torch.cuda.device(device):
            if not torch.cuda.is_bf16_supported():
                parser.error(f"{device} does not support BF16")
        print(f"{device}: {torch.cuda.get_device_name(device)}", flush=True)
    total = sum(check_pair(source, destination, sizes, args.trials)
                for source, destination in itertools.permutations(devices, 2))
    print(f"ALL {total} TRANSPORT CHECKS PASSED ({len(devices)*(len(devices)-1)} directed pairs)")


if __name__ == "__main__":
    main()
