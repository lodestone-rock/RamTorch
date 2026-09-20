"""
pipeline_infer_stream_demo.py
-----------------------------
Why streaming inference exists: iterative INFERENCE loops (diffusion
denoising, autoregressive-ish refinement, any per-sample recurrence) call the
model once per step, and plain ``Pipeline.infer()`` is a full barrier per
call — worker threads are joined and every device synchronized before it
returns. Between steps the pipeline DRAINS (later stages finish the tail
while stage 0 idles) and then REFILLS, costing ~(p-1) microbatch-forwards of
bubble per step on every stage.

The streaming API (``infer_submit`` / ``infer_open`` + ``submit_mb`` /
``wait_mb`` / ``infer_loop``) keeps one persistent worker per stage, so step
t+1's first microbatch can enter stage 0 once its step-t output is ready —
avoiding a mandatory full-batch drain (the diagrams are idealized schedules):

    sync infer() per step (p=4, m=4, 4 steps — dots are bubbles):
        s0: f0 f1 f2 f3 . . . f0 f1 f2 f3 . . . ...
        s1: . f0 f1 f2 f3 . . . f0 f1 f2 f3 . . ...
        s2: . . f0 f1 f2 f3 . . . f0 f1 f2 f3 . ...
        s3: . . . f0 f1 f2 f3 . . . f0 f1 f2 f3 ...

    streaming (same work, no drain):
        s0: f0 f1 f2 f3 f0 f1 f2 f3 f0 f1 f2 f3 f0 f1 f2 f3
        s1: . f0 f1 f2 f3 f0 f1 f2 f3 f0 f1 f2 f3 f0 f1 f2 f3
        s2: . . f0 f1 f2 f3 f0 f1 f2 f3 f0 f1 f2 f3 f0 f1 f2 f3
        s3: . . . f0 f1 f2 f3 f0 f1 f2 f3 f0 f1 f2 f3 f0 f1 f2 f3

This demo runs a toy refinement loop — out = model(x), then
x <- out - lr * out between forwards (equivalently (1-lr)*out) — applied
PER MICROBATCH. The scheduler must be per-sample independent for the overlap
to be valid. It runs three ways and requires finite, byte-identical results:

  sync     : for t: x = sched(pipe.infer(x))          (barrier per step)
  handles  : infer_open/submit_mb/wait_mb driven by hand (fully streamed)
  loop     : pipe.infer_loop(x, steps, update_fn)      (same, one call)

Run:  PYTHONPATH=. python examples/pipeline_infer_stream_demo.py
      PYTHONPATH=. python examples/pipeline_infer_stream_demo.py --offload
      PYTHONPATH=. python examples/pipeline_infer_stream_demo.py \
          --devices cuda:0,cuda:1 --steps 12 --mbs 8 --dim 2048
      PYTHONPATH=. python examples/pipeline_infer_stream_demo.py \
          --profile loop --profile-path trace_loop.json.gz   # Perfetto trace
"""

import argparse
import time

import torch
import torch.nn as nn

from ramtorch import Pipeline


class VectorBlock(nn.Module):
    """One matmul pair plus `vec_ops` rounds of elementwise spam (SiLU,
    RMSNorm-style normalize, tanh) on the hidden activation.

    The point is timeline visibility: GEMMs run on tensor cores and finish in
    microseconds, which makes the pipeline schedule hard to see in a trace.
    Vector kernels are launch/memory-bound, so each microbatch occupies a
    clearly visible band of many small kernels on the GPU stream — and the
    drain bubbles show as clean gaps. All ops are per-row deterministic, so
    bit-identity across the three modes still holds.
    """

    def __init__(self, dim, hidden, vec_ops):
        super().__init__()
        self.up = nn.Linear(dim, hidden)
        self.down = nn.Linear(hidden, dim)
        self.vec_ops = vec_ops
        self.eps = 1e-6

    def forward(self, x):
        h = self.up(x)
        for _ in range(self.vec_ops):
            h = h * torch.sigmoid(h)            # silu
            # rmsnorm (per-row; keeps the spam numerically bounded)
            h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
            h = h + 0.5 * torch.tanh(h) * h
        return self.down(h)


def make_chunks(n, dim, hidden, vec_ops):
    """n identical-shape vector-heavy blocks (residual wrap added by caller)."""
    torch.manual_seed(0)
    return [VectorBlock(dim, hidden, vec_ops) for _ in range(n)]


class Residual(nn.Module):
    """Wrap a block so the toy loop doesn't blow up: x + 0.1 * block(x)."""

    def __init__(self, block):
        super().__init__()
        self.block = block

    def forward(self, x):
        return x + 0.1 * self.block(x)


def build_pipe(devices, chunks_per_stage, dim, hidden, vec_ops,
               offload, window, bf16=False):
    n = len(devices) * chunks_per_stage
    chunks = [Residual(b) for b in make_chunks(n, dim, hidden, vec_ops)]
    if offload:
        return Pipeline(
            chunk_modules=chunks,
            devices=devices,
            offload_window=window,
            offload_pin=0,
            autocast="bf16" if bf16 else None,
        )
    per = [
        nn.Sequential(*chunks[i * chunks_per_stage:(i + 1) * chunks_per_stage])
        for i in range(len(devices))
    ]
    return Pipeline(stage_modules=per, devices=devices,
                    autocast="bf16" if bf16 else None)


@torch.no_grad()
def sched_step(out, lr):
    """Toy per-sample scheduler: shrink the model output between forwards."""
    return out - lr * out


@torch.no_grad()
def run_sync(pipe, x0, steps, m, lr):
    # Same semantics as infer_loop: `steps` forwards, `steps - 1` scheduler
    # updates between them (no update after the final forward).
    x = x0
    for _ in range(steps - 1):
        x = sched_step(pipe.infer(x, n_microbatches=m), lr)
    return pipe.infer(x, n_microbatches=m)


@torch.no_grad()
def run_handles(pipe, x0, steps, m, lr):
    """The streaming pattern, written out by hand: feed step t+1's first
    microbatch the moment its step-t output lands, while later stages are
    still working on step t's tail."""
    h = pipe.infer_submit(x0, n_microbatches=m)
    for t in range(steps - 1):
        nxt = pipe.infer_open(m)
        for i in range(m):
            nxt.submit_mb(i, sched_step(h.wait_mb(i), lr))
        h = nxt
    outs = [h.wait_mb(i) for i in range(m)]
    return torch.cat(outs, dim=0)


@torch.no_grad()
def run_loop(pipe, x0, steps, m, lr):
    return pipe.infer_loop(
        x0, steps=steps, n_microbatches=m,
        update_fn=lambda out, i, t: sched_step(out, lr),
    )


def synchronize(devices):
    """Timing/profile boundary drain, NOT part of the streaming hot path."""
    for device in dict.fromkeys(map(torch.device, devices)):
        if device.type == "cuda":
            torch.cuda.synchronize(device)


def require_exact(name, actual, expected):
    if actual.requires_grad or actual.grad_fn is not None:
        raise AssertionError(f"{name}: inference retained an autograd graph")
    actual = actual.detach().cpu().contiguous()
    expected = expected.detach().cpu().contiguous()
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        raise AssertionError(f"{name}: output shape/dtype mismatch")
    if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
        raise AssertionError(f"{name}: non-finite inference output")
    if not torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)):
        error = (actual.float() - expected.float()).abs().max().item()
        raise AssertionError(f"{name}: byte mismatch (max error {error:.3e})")


def main():
    ap = argparse.ArgumentParser(
        description="streaming vs barriered inference in an iterative loop")
    ap.add_argument("--devices", default="cuda:0,cuda:1,cuda:2,cuda:3",
                    help="comma-separated, one per stage")
    ap.add_argument("--chunks-per-stage", type=int, default=2)
    ap.add_argument("--dim", type=int, default=1024)
    ap.add_argument("--hidden", type=int, default=16384)
    ap.add_argument("--vec-ops", type=int, default=8,
                    help="rounds of elementwise spam per block (SiLU + "
                         "RMSNorm + tanh) — crank this to make each "
                         "microbatch's compute band clearly visible")
    ap.add_argument("--batch", type=int, default=8192)
    ap.add_argument("--mbs", type=int, default=8, help="microbatches per step")
    ap.add_argument("--steps", type=int, default=8, help="denoising steps")
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--iters", type=int, default=3, help="timed repetitions")
    ap.add_argument("--offload", action="store_true",
                    help="stream stage weights from CPU RAM (OffloadStage)")
    ap.add_argument("--window", type=int, default=2)
    ap.add_argument("--bf16", action="store_true", help="BF16 autocast in stage workers")
    ap.add_argument("--profile", default=None,
                    choices=["sync", "handles", "loop"],
                    help="capture a kineto (Perfetto) trace of one loop in "
                         "this mode (runs AFTER the timing table)")
    ap.add_argument("--profile-path", default=None,
                    help="trace path; .gz enables compression "
                         "(default: infer_stream_<mode>.json.gz)")
    args = ap.parse_args()
    for name in ("chunks_per_stage", "dim", "hidden", "batch", "mbs", "steps", "iters", "window"):
        if getattr(args, name) < 1:
            ap.error(f"--{name.replace('_', '-')} must be positive")
    if args.vec_ops < 0:
        ap.error("--vec-ops must be nonnegative")
    # The manual-handle recipe concatenates padded microbatches itself. Keep
    # this timing workload divisible rather than silently comparing extra rows.
    if args.batch % args.mbs:
        ap.error("--batch must be divisible by --mbs for the manual-handle demo")
    devices = []
    for value in args.devices.split(","):
        device = torch.device(value.strip())
        if device.type not in ("cpu", "cuda"):
            ap.error("--devices supports CPU or CUDA only")
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        devices.append(str(device))
    if args.bf16:
        for device in map(torch.device, devices):
            if device.type == "cuda":
                with torch.cuda.device(device):
                    if not torch.cuda.is_bf16_supported():
                        ap.error(f"BF16 is unsupported on {device}")
    pipe = build_pipe(devices, args.chunks_per_stage, args.dim, args.hidden,
                      args.vec_ops, args.offload, args.window, args.bf16)
    try:
        torch.manual_seed(1)
        x0 = torch.randn(args.batch, args.dim)
        print(f"{len(devices)} stages x {args.chunks_per_stage} blocks "
              f"[Linear({args.dim}->{args.hidden}) + {args.vec_ops}x vector spam "
              f"+ Linear] on {', '.join(devices)}"
              + (f" (OFFLOADED, window={args.window})" if args.offload else ""))
        print(f"torch={torch.__version__}, precision={'bf16 autocast' if args.bf16 else 'fp32'}, "
              "optimizer=none (inference)")
        for device in dict.fromkeys(map(torch.device, devices)):
            if device.type == "cuda":
                print(f"  {device}: {torch.cuda.get_device_name(device)}")
        print(f"toy refinement: {args.steps} forwards x {args.mbs} microbatches of "
              f"{args.batch // args.mbs} rows; between forwards, "
              f"out=model(x), x <- out - {args.lr}*out")
        print("Timing: unprofiled wall time, every participating CUDA device "
              "drained before/after each repetition block.\n")

        results, times = {}, {}
        modes = {"sync": run_sync, "handles": run_handles, "loop": run_loop}
        for name, fn in modes.items():
            fn(pipe, x0, 2, args.mbs, args.lr)  # contexts, cuBLAS, workers
            synchronize(devices)
            t0 = time.perf_counter()
            for _ in range(args.iters):
                out = fn(pipe, x0, args.steps, args.mbs, args.lr)
            synchronize(devices)
            times[name] = (time.perf_counter() - t0) / args.iters
            results[name] = out

        base = results["sync"]
        # Fail the process BEFORE printing successful timings if correctness
        # fails; finite raw-byte equality is stronger than torch.equal.
        for name, out in results.items():
            require_exact(name, out, base)
        print(f"{'mode':>8s} {'s/loop':>9s} {'sync/time':>9s}   vs sync")
        for name in modes:
            print(f"{name:>8s} {times[name]:9.3f} "
                  f"{times['sync'] / times[name]:8.2f}x   FINITE / BIT-IDENTICAL")
        print("\nRatios describe this run only; they are not GPU utilization "
              "or a guarantee of an overlap speedup.")

        if args.profile is not None:
            from ramtorch.pipeline_2bw_trace import TraceCapture, inspect_trace

            path = args.profile_path or f"infer_stream_{args.profile}.json.gz"
            # Fresh workers start inside the capture; existing persistent
            # threads can be invisible to some profiler implementations.
            profile_pipe = build_pipe(
                devices, args.chunks_per_stage, args.dim, args.hidden,
                args.vec_ops, args.offload, args.window, args.bf16)
            try:
                synchronize(devices)
                with TraceCapture(path):
                    try:
                        profiled = modes[args.profile](profile_pipe, x0, args.steps,
                                                       args.mbs, args.lr)
                    finally:
                        try:
                            profile_pipe.close()
                        finally:
                            # Real completion on ALL GPUs before Kineto stops;
                            # this cost is outside the clean timing block.
                            synchronize(devices)
            finally:
                # Also clean up if profiler initialization/export failed.
                profile_pipe.close()
            require_exact(f"profile {args.profile}", profiled, base)
            stats = inspect_trace(path)
            counts = stats["gpu_kernels_per_device"]
            missing = [str(d) for d in map(torch.device, devices)
                       if d.type == "cuda" and not counts.get(str(d.index), 0)]
            print(f"profile ({args.profile}) written to {path}; "
                  f"real CUDA kernels by device: {counts}")
            if missing:
                raise RuntimeError(f"trace has no real CUDA kernels for {missing}; "
                                   "check CUPTI/profiler support before judging overlap")
            print("Open at https://ui.perfetto.dev; host spans are not GPU utilization.")
    finally:
        pipe.close()


if __name__ == "__main__":
    main()
