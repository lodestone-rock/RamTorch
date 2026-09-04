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
t+1's first microbatch enters stage 0 the moment its step-t output is ready —
the pipeline never drains:

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

This demo runs a toy "denoising" loop — x <- x - lr * model(x), applied
PER MICROBATCH (the scheduler must be per-sample independent for the overlap
to be valid) — three ways and checks all three are BIT-IDENTICAL:

  sync     : for t: x = sched(pipe.infer(x))          (barrier per step)
  handles  : infer_open/submit_mb/wait_mb driven by hand (fully streamed)
  loop     : pipe.infer_loop(x, steps, update_fn)      (same, one call)

Run:  PYTHONPATH=. python examples/pipeline_infer_stream_demo.py
      PYTHONPATH=. python examples/pipeline_infer_stream_demo.py --offload
      PYTHONPATH=. python examples/pipeline_infer_stream_demo.py \
          --devices cuda:0,cuda:1 --steps 12 --mbs 8 --dim 2048
      PYTHONPATH=. python examples/pipeline_infer_stream_demo.py \
          --profile loop --profile-path trace_loop.json   # Perfetto trace
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
               offload, window):
    n = len(devices) * chunks_per_stage
    chunks = [Residual(b) for b in make_chunks(n, dim, hidden, vec_ops)]
    if offload:
        return Pipeline(
            chunk_modules=chunks,
            devices=devices,
            offload_window=window,
            offload_pin=0,
        )
    per = [
        nn.Sequential(*chunks[i * chunks_per_stage:(i + 1) * chunks_per_stage])
        for i in range(len(devices))
    ]
    return Pipeline(stage_modules=per, devices=devices)


def sched_step(out, lr):
    """Toy per-sample scheduler: one Euler-ish denoising update."""
    return out - lr * out


def run_sync(pipe, x0, steps, m, lr):
    # Same semantics as infer_loop: `steps` forwards, `steps - 1` scheduler
    # updates between them (no update after the final forward).
    x = x0
    for _ in range(steps - 1):
        x = sched_step(pipe.infer(x, n_microbatches=m), lr)
    return pipe.infer(x, n_microbatches=m)


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


def run_loop(pipe, x0, steps, m, lr):
    return pipe.infer_loop(
        x0, steps=steps, n_microbatches=m,
        update_fn=lambda out, i, t: sched_step(out, lr),
    )


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
    ap.add_argument("--profile", default=None,
                    choices=["sync", "handles", "loop"],
                    help="capture a kineto (Perfetto) trace of one loop in "
                         "this mode (runs AFTER the timing table)")
    ap.add_argument("--profile-path", default=None,
                    help="trace output path (default: infer_stream_<mode>.json)")
    args = ap.parse_args()

    devices = [d.strip() for d in args.devices.split(",")]
    pipe = build_pipe(devices, args.chunks_per_stage, args.dim, args.hidden,
                      args.vec_ops, args.offload, args.window)
    p = len(devices)
    torch.manual_seed(1)
    x0 = torch.randn(args.batch, args.dim)

    print(f"{p} stages x {args.chunks_per_stage} blocks "
          f"[Linear({args.dim}->{args.hidden}) + {args.vec_ops}x vector spam "
          f"+ Linear] on {', '.join(devices)}"
          + (" (OFFLOADED, window=%d)" % args.window if args.offload else ""))
    print(f"toy denoising: {args.steps} steps x {args.mbs} microbatches of "
          f"{args.batch // args.mbs} rows, x <- x - {args.lr}*model(x)\n")

    results, times = {}, {}
    for name, fn in [("sync", run_sync), ("handles", run_handles),
                     ("loop", run_loop)]:
        fn(pipe, x0, 2, args.mbs, args.lr)  # warmup (CUDA ctx, cuBLAS, workers)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.iters):
            out = fn(pipe, x0, args.steps, args.mbs, args.lr)
        torch.cuda.synchronize()
        times[name] = (time.perf_counter() - t0) / args.iters
        results[name] = out

    base = results["sync"]
    print(f"{'mode':>8s} {'s/loop':>9s} {'speedup':>8s}   vs sync")
    for name in ("sync", "handles", "loop"):
        same = torch.equal(base, results[name])
        print(f"{name:>8s} {times[name]:9.3f} "
              f"{times['sync'] / times[name]:7.2f}x   "
              f"{'BIT-IDENTICAL' if same else 'MISMATCH (bug!)'}")

    drain = (p - 1)  # refill bubble per step, in microbatch-forwards
    saved = times["sync"] - times["loop"]
    print(f"\nstreaming saved {saved * 1e3:.0f} ms per {args.steps}-step loop "
          f"(~{drain} microbatch-forwards of drain bubble per step with "
          f"barriered infer())")

    if args.profile is not None:
        from torch.profiler import ProfilerActivity, profile as torch_profile

        path = args.profile_path or f"infer_stream_{args.profile}.json"
        fn = {"sync": run_sync, "handles": run_handles,
              "loop": run_loop}[args.profile]
        with torch_profile(activities=[ProfilerActivity.CPU,
                                       ProfilerActivity.CUDA]) as prof:
            fn(pipe, x0, args.steps, args.mbs, args.lr)
            # Drain device work BEFORE the profiler stops so the full kernel
            # timeline is captured.
            torch.cuda.synchronize()
        prof.export_chrome_trace(path)
        print(f"profile ({args.profile} mode) written to {path} "
              f"— open at https://ui.perfetto.dev")
    pipe.close()


if __name__ == "__main__":
    main()
