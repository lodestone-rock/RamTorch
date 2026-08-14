"""
mnist_pipeline_big_transformer_manual.py
-----------------------------------------
MNIST training with a ~100M-param transformer, pipeline-parallel across GPUs —
using **manual pre-partitioned stages** instead of the torch.export tracer.

WHY MANUAL?
-----------
The convenient `PipelineModel` API auto-splits your model by tracing it with
`torch.export`. That works for simple models (linear stacks, basic CNNs), but
complex architectures — `nn.MultiheadAttention`, dynamic reshapes, control flow,
custom ops — routinely break the tracer or produce subtly-wrong graphs. This
example's transformer is exactly such a case: the exported graph mangles the
attention reshapes across the stage boundary (`view size is not compatible`).

The robust answer: **partition the model yourself** into a list of stage
modules, and hand them to `Pipeline(stage_modules=[...])`. This bypasses the
tracer entirely — each stage is a plain `nn.Module` whose forward takes the
previous stage's output. It's a trivial change to make in your own model class,
and it's the recommended approach for anything non-trivial.

`PipelineModel` is the convenient API for simple models; `Pipeline(stage_modules=...)`
is the recommended API for real architectures. Make the informed choice.

WEIGHT OFFLOADING (--offload)
-----------------------------
With `--offload`, each stage is passed to `Pipeline` as a LIST of chunk modules
(embed / one transformer block / norm+head each) instead of one module, which
makes it an `OffloadStage`: the chunk masters live in CPU pinned RAM and stream
through a `--window`-slot GPU window, prefetched in the schedule's exact chunk
order. GPU weight memory per stage drops to ~(window + pin) chunks. The
transformer blocks have real arithmetic intensity, so unlike the wide-MLP
example (`mnist_pipeline_offload.py`) the traffic largely hides behind compute
and pipeline bubbles — check the `--profile` trace in Perfetto to see the H2D
loads overlapping the F/B spans.

Run:
    python examples/mnist_pipeline_big_transformer_manual.py
    python examples/mnist_pipeline_big_transformer_manual.py --steps 200 --devices cuda:1 cuda:3
    python examples/mnist_pipeline_big_transformer_manual.py \
        --devices cuda:0 cuda:1 cuda:2 cuda:3 --offload --window 2 --profile
"""

from __future__ import annotations

import argparse
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ramtorch import Pipeline, PipelineOptimizer


# ── Building blocks ───────────────────────────────────────────────────────────

class Block(nn.Module):
    """A standard pre-norm transformer block."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + a
        x = x + self.mlp(self.norm2(x))
        return x


# ── The model, PRE-PARTITIONED into stages ────────────────────────────────────
# Instead of one monolithic model that a tracer must split, we define the model
# as a list of stage modules up front. Each stage's forward consumes the
# previous stage's output. This is the whole "manual split" — trivial, and the
# tracer is never involved.

class EmbedStage(nn.Module):
    """Stage 0: patch embed + positional embedding + the first K blocks."""

    def __init__(self, dim: int, heads: int, patch: int, n_blocks: int):
        super().__init__()
        n_patches = (28 // patch) ** 2
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        # init the embed params BEFORE creating the blocks so this consumes
        # the RNG in the same order as the chunked form (EmbedChunk, then
        # Blocks) — a fixed seed then gives IDENTICAL weights in both modes
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(n_blocks)])

    def forward(self, x):
        # x: (B, 1, 28, 28) image
        x = self.patch_embed(x)                  # (B, dim, 7, 7)
        x = x.flatten(2).transpose(1, 2)         # (B, 49, dim)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)           # (B, 50, dim)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return x                                  # token sequence -> next stage


class HeadStage(nn.Module):
    """Stage 1: the remaining blocks + final norm + classifier head."""

    def __init__(self, dim: int, heads: int, n_blocks: int, num_classes: int):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x: token sequence from the previous stage
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])                 # logits from the cls token


def build_stages(dim: int, depth: int, heads: int, patch: int,
                 num_classes: int, n_stages: int):
    """
    Split `depth` blocks across `n_stages` stages as evenly as possible.
    Stage 0 = embed + first chunk of blocks; stages 1..n-1 = block chunks;
    the LAST stage additionally owns the norm + head.
    Returns a list of stage modules in order.
    """
    # Distribute blocks as evenly as possible (earlier stages get the remainder).
    base = depth // n_stages
    rem = depth % n_stages
    counts = [base + (1 if i < rem else 0) for i in range(n_stages)]

    stages = []
    for i, cnt in enumerate(counts):
        if i == 0:
            stages.append(EmbedStage(dim, heads, patch, cnt))
        elif i == n_stages - 1:
            stages.append(HeadStage(dim, heads, cnt, num_classes))
        else:
            # A middle stage is just a stack of blocks.
            stages.append(nn.Sequential(*[Block(dim, heads) for _ in range(cnt)]))
    return stages


# ── Offloaded variant: the SAME model diced one level finer ──────────────────
# For --offload each stage becomes a LIST of chunk modules (the OffloadModel
# dicing convention: chunk i+1 consumes chunk i's output). Same math, but now
# the Pipeline streams each stage's chunks from CPU pinned RAM.

class EmbedChunk(nn.Module):
    """Patch embed + cls token + positional embedding (no blocks)."""

    def __init__(self, dim: int, patch: int):
        super().__init__()
        n_patches = (28 // patch) ** 2
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)                  # (B, dim, 7, 7)
        x = x.flatten(2).transpose(1, 2)         # (B, 49, dim)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        return torch.cat([cls, x], dim=1) + self.pos_embed


class HeadChunk(nn.Module):
    """Final norm + classifier head on the cls token."""

    def __init__(self, dim: int, num_classes: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        return self.head(self.norm(x)[:, 0])


def build_stage_chunks(dim: int, depth: int, heads: int, patch: int,
                       num_classes: int, n_stages: int):
    """Same split as build_stages, but each stage is a list of chunks."""
    base = depth // n_stages
    rem = depth % n_stages
    counts = [base + (1 if i < rem else 0) for i in range(n_stages)]

    stage_chunks = []
    for i, cnt in enumerate(counts):
        # construct in pipeline order (embed BEFORE its blocks) so the RNG
        # stream matches build_stages — same seed, same weights, both modes
        chunks = []
        if i == 0:
            chunks.append(EmbedChunk(dim, patch))
        chunks.extend(Block(dim, heads) for _ in range(cnt))
        if i == n_stages - 1:
            chunks.append(HeadChunk(dim, num_classes))
        stage_chunks.append(chunks)
    return stage_chunks


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--micro", type=int, default=4, help="microbatches per step")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=12)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--schedule", default="staggered_1b1f",
                    choices=["gpipe", "staggered_1b1f", "1f1b"])
    ap.add_argument("--devices", nargs="+", default=["cuda:1", "cuda:3"],
                    help="one device per pipeline stage")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save", type=str, default="", help="optional checkpoint path")
    ap.add_argument("--parallel-opt", action="store_true",
                    help="use PipelineOptimizer (per-stage parallel optimizer) "
                         "instead of a single sequential Adam")
    # ── Weight offloading (stream each stage's chunks from CPU pinned RAM) ──
    ap.add_argument("--offload", action="store_true",
                    help="dice the model into a flat chunk list "
                         "(Pipeline(chunk_modules=...)) and STREAM each "
                         "stage's chunks: masters in CPU pinned RAM, "
                         "sliding GPU window")
    ap.add_argument("--chunks", action="store_true",
                    help="same flat chunk_modules dicing but RESIDENT stages "
                         "(offload=False): the chunking convenience without "
                         "the streaming machinery")
    ap.add_argument("--window", type=int, default=2,
                    help="streamed GPU chunk slots per offloaded stage")
    ap.add_argument("--pin", type=int, default=0,
                    help="chunks pinned permanently on each stage's GPU")
    ap.add_argument("--offload-mode", default="keep",
                    choices=["keep", "checkpoint"],
                    help="offloaded backward: keep per-chunk graphs, or "
                         "per-chunk non-reentrant checkpoint (less memory)")
    # ── Profiling (bounded to a small step window so files stay small) ──
    ap.add_argument("--profile", action="store_true",
                    help="capture a kineto profile + op-level Perfetto trace "
                         "during the profile window")
    ap.add_argument("--profile-start", type=int, default=20,
                    help="first step to profile (after this many warm steps)")
    ap.add_argument("--profile-steps", type=int, default=3,
                    help="how many steps to profile (keeps the trace small)")
    args = ap.parse_args()

    n_stages = len(args.devices)
    if n_stages < 1:
        raise SystemExit("need at least one device")
    if n_stages > 1 and torch.cuda.device_count() < 2:
        raise SystemExit("multiple GPUs required; pass --devices cuda:0 for single")

    torch.manual_seed(args.seed)

    # ── Data ──────────────────────────────────────────────────────────────────
    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=tf),
        batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True,
    )
    test_loader = DataLoader(
        datasets.MNIST("./data", train=False, download=True, transform=tf),
        batch_size=512, shuffle=False, num_workers=2,
    )

    # ── Build the pre-partitioned stages (NO torch.export) ────────────────────
    if args.offload and args.chunks:
        raise SystemExit("--offload already implies chunked stages; "
                         "pass only one of --offload / --chunks")
    if args.offload or args.chunks:
        # Flat chunk_modules path: dice the whole model into ONE ordered chunk
        # list; the Pipeline splits it across the devices (chunks_per_stage
        # keeps the embed/head placement of build_stage_chunks). offload=True
        # streams each stage's chunks; offload=False keeps them resident.
        stage_chunks = build_stage_chunks(args.dim, args.depth, args.heads,
                                          args.patch, num_classes=10,
                                          n_stages=n_stages)
        chunks = [m for st in stage_chunks for m in st]
        counts = [len(st) for st in stage_chunks]
        n_params = sum(p.numel() for m in chunks for p in m.parameters())
        kind = "offloaded" if args.offload else "resident chunked"
        extra = (f" (window={args.window} pin={args.pin} "
                 f"mode={args.offload_mode})" if args.offload else "")
        print(f"model: {n_params/1e6:.1f}M params, {len(chunks)} chunks -> "
              f"{n_stages} {kind} stages {counts}{extra}")
        pipe = Pipeline(
            chunk_modules=chunks, chunks_per_stage=counts,
            devices=args.devices,
            offload=args.offload,
            offload_window=args.window, offload_pin=args.pin,
            offload_keep_activations=(True if args.offload_mode == "keep"
                                      else "checkpoint"),
        )
    else:
        stages = build_stages(args.dim, args.depth, args.heads, args.patch,
                              num_classes=10, n_stages=n_stages)
        n_params = sum(p.numel() for st in stages for p in st.parameters())
        print(f"model: {n_params/1e6:.1f}M params, manually split into {n_stages} stages")
        for i, st in enumerate(stages):
            sp = sum(p.numel() for p in st.parameters())
            print(f"  stage {i} ({type(st).__name__}): {sp/1e6:.1f}M params -> {args.devices[i]}")
        pipe = Pipeline(stage_modules=stages, devices=args.devices)
    # Optimize over ALL stage params (they're separate modules, not one model).
    # pipe.stages[i].params works for plain and offloaded stages alike (for
    # offloaded ones the masters live in CPU pinned RAM).
    if args.parallel_opt:
        if args.offload:
            raise SystemExit("--parallel-opt is not supported with --offload; "
                             "use the default single optimizer (fused AdamW)")
        opt = PipelineOptimizer(
            pipe.stages, lambda p: torch.optim.Adam(p, lr=args.lr)
        )
        print("optimizer: PipelineOptimizer (per-stage parallel)")
    elif args.offload:
        # fused=True groups params by device: CPU masters get the fused CPU
        # kernel, any GPU-resident params the fused CUDA kernel.
        opt = torch.optim.AdamW(
            itertools.chain(*(st.params for st in pipe.stages)),
            lr=args.lr, fused=True,
        )
        print("optimizer: single AdamW(fused=True) over CPU masters")
    else:
        # Works for plain module stages and resident chunked (--chunks) stages
        # alike: every Stage exposes its own .params list.
        opt = torch.optim.Adam(
            itertools.chain(*(st.params for st in pipe.stages)), lr=args.lr
        )
        print("optimizer: single sequential Adam")
    loss_fn = nn.CrossEntropyLoss()
    print(f"schedule={args.schedule}  microbatches={args.micro}  Adam lr={args.lr}")

    # ── Train ─────────────────────────────────────────────────────────────────
    if args.offload and torch.cuda.is_available():
        for d in args.devices:
            torch.cuda.reset_peak_memory_stats(d)
    print(f"\ntraining ({args.steps} steps) ...")
    if args.profile:
        stop = args.profile_start + args.profile_steps
        print(f"  profiling steps [{args.profile_start}, {stop}) "
              f"-> profile_win.json + trace_win.json")
    import time
    step = 0
    running = 0.0
    t_fwd_bwd = 0.0   # pipeline step time
    t_opt = 0.0       # optimizer step time
    done = False
    while not done:
        for x, y in train_loader:
            # Profile only inside the small window so the trace files stay small.
            # Each profiled step writes its own file (trace_win_<step>.json), so
            # you can open any single step's full F/B relay in Perfetto.
            in_window = (
                args.profile
                and args.profile_start <= step < args.profile_start + args.profile_steps
            )
            t0 = time.perf_counter()
            result = pipe.step(
                x, targets=y, schedule=args.schedule,
                n_microbatches=args.micro, loss_fn=loss_fn,
                trace_path=(f"trace_win_{step}.json" if in_window else None),
                profile_path=(f"profile_win_{step}.json" if in_window else None),
            )
            result.flush_grads()
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            opt.step()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            t2 = time.perf_counter()
            t_fwd_bwd += t1 - t0
            t_opt += t2 - t1

            running += result.loss.item()
            step += 1
            if step % 10 == 0:
                print(f"  step {step:>4}: loss={running/10:.4f}", flush=True)
                running = 0.0
            if step >= args.steps:
                done = True
                break

    print(f"\ntiming breakdown over {step} steps:")
    print(f"  pipeline fwd+bwd : {t_fwd_bwd*1e3:8.1f}ms  ({t_fwd_bwd/step*1e3:6.2f} ms/step)")
    print(f"  optimizer step   : {t_opt*1e3:8.1f}ms  ({t_opt/step*1e3:6.2f} ms/step)")
    print(f"  optimizer share  : {100*t_opt/(t_fwd_bwd+t_opt):.1f}% of step time")
    if args.offload:
        print("  offload engine stats per stage:")
        for i, st in enumerate(pipe.stages):
            s = st.engine.stats
            print(f"    stage {i}: H2D loads={s['loads']} "
                  f"({s['loads']/step:.1f}/step), acquire stall="
                  f"{s['acquire_wait_s']:.2f}s ({s['acquire_wait_s']/step*1e3:.1f} ms/step)")
        for d in args.devices:
            peak = torch.cuda.max_memory_allocated(d) / 2**20
            print(f"    {d}: peak GPU memory {peak:.0f} MiB")

    # ── Evaluate: pipelined inference (GPipe-forward, keeps all GPUs busy) ────
    # st.module is the stage's module (for offloaded stages, the streaming
    # engine, whose eval() propagates to the chunks).
    for st in pipe.stages:
        st.module.eval()
    correct = total = 0
    for bi, (x, y) in enumerate(test_loader):
        # pipe.infer() microbatches the batch through the stages concurrently
        # (forward-only GPipe) instead of a sequential whole-batch relay, so the
        # GPUs stay busy. no_grad + no activation retention.
        # Profile just the first eval batch so the inference-overlap trace stays
        # small (mirrors the bounded training profiler).
        profile_inf = args.profile and bi == 0
        logits = pipe.infer(
            x,
            n_microbatches=args.micro,
            trace_path=("trace_infer.json" if profile_inf else None),
            profile_path=("profile_infer.json" if profile_inf else None),
        )
        pred = logits.argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += y.size(0)
    acc = correct / total
    print(f"\ntest accuracy after {step} steps: {acc:.4f}")
    if args.profile:
        print("  inference profile -> profile_infer.json + trace_infer.json (first eval batch)")

    # ── Checkpoint: save each stage's state_dict ──────────────────────────────
    if args.save:
        # st.module covers both cases: the plain stage module, or the offload
        # engine (whose state_dict holds the CPU masters under chunks.N.*).
        ckpt = {f"stage_{i}": st.module.state_dict()
                for i, st in enumerate(pipe.stages)}
        torch.save(ckpt, args.save)
        print(f"saved checkpoint -> {args.save}")

    # Shut down the parallel optimizer's worker threads, if used.
    if args.parallel_opt:
        opt.close()
    pipe.close()   # stops offloaded stages' loader/writeback threads (no-op otherwise)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
