"""
mnist_frozen_encoder_overlap.py
--------------------------------
Diffusion-style two-model scenario: a FROZEN, randomly-initialized "text
encoder" transformer produces a context token via PIPELINE INFERENCE, which is
then fed as a context token to a TRAINED transformer doing pipeline
forward+backward.

The point of the test: does the frozen encoder's forward-only inference overlap
cleanly with the trained model's training compute? In real diffusion training
(e.g. a frozen CLIP/T5 encoder feeding a UNet), the encoder inference should
hide under the training forward/backward instead of serializing before it.

Layout
------
  * Frozen encoder  : random-init transformer, its OWN Pipeline, runs .infer()
                      (GPipe-forward, no backward, no grad). Produces a context
                      token embedding from a pseudo "caption" (a small int
                      sequence derived from the MNIST label — stands in for text).
  * Trained model   : a ViT-style image classifier, its OWN Pipeline, runs
                      .step() (staggered_1b1f). Takes the image patches PLUS the
                      encoder's context token as an extra sequence token.

Both pipelines run on the SAME two GPUs (interleaved stages), so the profiler
shows whether the frozen inference overlaps the training or serializes.

The overlap is achieved by running the frozen encoder's infer() on a background
thread for batch k+1 while the main thread trains on batch k (double-buffered
prefetch) — the same way a dataloader prefetches. We then profile the steady
state to see inference and training compute interleaved on the GPUs.

Run:
    python examples/mnist_frozen_encoder_overlap.py --steps 60
    python examples/mnist_frozen_encoder_overlap.py --steps 60 --profile \
        --profile-start 30 --profile-steps 2
"""

from __future__ import annotations

import argparse
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ramtorch import Pipeline


# ── Shared transformer block ──────────────────────────────────────────────────

class Block(nn.Module):
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


# ── Frozen "text encoder" (random init, inference only) ───────────────────────
# Stands in for a frozen CLIP/T5 text encoder in diffusion training. It maps a
# pseudo-caption (a short int sequence) to a single context embedding. Frozen:
# no_grad, no optimizer, pipeline INFERENCE only.

class FrozenEncoderEmbed(nn.Module):
    """Stage 0 of the frozen encoder: token embed + first blocks."""

    def __init__(self, vocab: int, dim: int, heads: int, n_blocks: int, ctx_len: int):
        super().__init__()
        self.tok_embed = nn.Embedding(vocab, dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, ctx_len, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(n_blocks)])
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, idx):  # idx: (B, ctx_len) int tokens
        x = self.tok_embed(idx) + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return x


class FrozenEncoderHead(nn.Module):
    """Stage 1 of the frozen encoder: remaining blocks + pool to a context vector."""

    def __init__(self, dim: int, heads: int, n_blocks: int):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x.mean(dim=1)  # (B, dim) context embedding


# ── Trained image model (consumes the context token) ─────────────────────────

class TrainEmbed(nn.Module):
    """Stage 0 of the trained model: patch embed + context token + first blocks.

    Takes TWO inputs as positional args — ``(ctx, x)`` — which the pipeline
    dices into microbatches automatically (flat-tuple input). ``ctx`` is the
    frozen encoder's context embedding (B, dim); ``x`` is the image (B,1,28,28).
    """

    def __init__(self, dim: int, heads: int, patch: int, n_blocks: int):
        super().__init__()
        n_patches = (28 // patch) ** 2
        self.patch_embed = nn.Conv2d(1, dim, kernel_size=patch, stride=patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        # +1 for cls, +1 for the context token from the frozen encoder
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 2, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(n_blocks)])
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, ctx, x):  # ctx: (mb, dim), x: (mb, 1, 28, 28)
        B = x.size(0)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)              # (mb, 49, dim)
        cls = self.cls_token.expand(B, -1, -1)
        ctx_tok = ctx.unsqueeze(1).to(x.device)        # (mb, 1, dim)
        x = torch.cat([cls, x, ctx_tok], dim=1)        # (mb, 51, dim)
        x = x + self.pos_embed
        for blk in self.blocks:
            x = blk(x)
        return x


class TrainHead(nn.Module):
    """Stage 1 of the trained model: remaining blocks + norm + classifier head."""

    def __init__(self, dim: int, heads: int, n_blocks: int, num_classes: int):
        super().__init__()
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(n_blocks)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def build_even_blocks(dim, heads, n_blocks):
    return nn.Sequential(*[Block(dim, heads) for _ in range(n_blocks)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dim", type=int, default=512)
    ap.add_argument("--depth", type=int, default=6, help="blocks per model")
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--patch", type=int, default=4)
    ap.add_argument("--ctx-len", type=int, default=8, help="frozen-encoder pseudo-caption length")
    ap.add_argument("--schedule", default="staggered_1b1f",
                    choices=["gpipe", "staggered_1b1f", "1f1b"])
    ap.add_argument("--devices", nargs=2, default=["cuda:1", "cuda:3"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--profile-start", type=int, default=30)
    ap.add_argument("--profile-steps", type=int, default=2)
    args = ap.parse_args()

    if torch.cuda.device_count() < 2:
        raise SystemExit("need 2 GPUs")
    torch.manual_seed(args.seed)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_loader = DataLoader(
        datasets.MNIST("./data", train=True, download=True, transform=tf),
        batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=True,
    )

    # ── Frozen encoder pipeline (random init, inference only) ─────────────────
    # Two stages: embed+blocks on device0, blocks+pool on device1.
    enc0 = FrozenEncoderEmbed(vocab=10, dim=args.dim, heads=args.heads,
                              n_blocks=args.depth // 2, ctx_len=args.ctx_len)
    enc1 = FrozenEncoderHead(dim=args.dim, heads=args.heads,
                             n_blocks=args.depth - args.depth // 2)
    for p in itertools.chain(enc0.parameters(), enc1.parameters()):
        p.requires_grad_(False)  # frozen
    enc_pipe = Pipeline(stage_modules=[enc0, enc1], devices=args.devices)
    n_frozen = sum(p.numel() for st in (enc0, enc1) for p in st.parameters())
    print(f"frozen encoder: {n_frozen/1e6:.1f}M params (random init, INFERENCE only)")

    # ── Trained model factory (fresh init + optimizer per run, so the two modes
    #    start from IDENTICAL weights and the comparison is valid) ─────────────
    loss_fn = nn.CrossEntropyLoss()

    def build_trained():
        torch.manual_seed(args.seed)  # identical init every time
        tr0 = TrainEmbed(dim=args.dim, heads=args.heads, patch=args.patch,
                         n_blocks=args.depth // 2)
        tr1 = TrainHead(dim=args.dim, heads=args.heads,
                        n_blocks=args.depth - args.depth // 2, num_classes=10)
        pipe = Pipeline(stage_modules=[tr0, tr1], devices=args.devices)
        opt = torch.optim.Adam(
            itertools.chain(*(st.parameters() for st in (tr0, tr1))), lr=args.lr
        )
        return tr0, tr1, pipe, opt

    n_train = sum(p.numel() for p in TrainEmbed(dim=args.dim, heads=args.heads,
                  patch=args.patch, n_blocks=1).parameters())
    print(f"trained model: ~{n_train/1e6:.1f}M params (built fresh per run)")

    # ── Pseudo-caption: derive a short int "text" sequence from the label ─────
    # In real diffusion this is the caption tokens; here we fabricate a
    # deterministic ctx_len-token sequence from the label so the encoder has
    # something to process. (Content doesn't matter — we're testing overlap.)
    def make_captions(labels: torch.Tensor) -> torch.Tensor:
        g = torch.Generator().manual_seed(1234)
        base = torch.randint(0, 10, (args.ctx_len,), generator=g)
        return (labels.unsqueeze(1) + base.unsqueeze(0)) % 10  # (B, ctx_len)

    print(f"schedule={args.schedule}  micro={args.micro}  dim={args.dim} depth={args.depth}")

    # SERIAL flow (the simple, recommended pattern): run the frozen encoder's
    # pipeline inference for the whole batch first, then feed (ctx, x) as a
    # flat-tuple input to the trained model's pipeline step. The pipeline dices
    # BOTH ctx and x into microbatches automatically and unpacks them as
    # positional args into the trained stage-0 module.
    import time

    def run_loop():
        tr0, tr1, pipe, opt = build_trained()
        step = 0
        running = 0.0
        losses = []
        # Bounded profiler over a few steps (captures encoder inference + training).
        prof = None
        if args.profile:
            from torch.profiler import (
                ProfilerActivity, profile as _tp, schedule as _sched,
            )
            start, nrec = args.profile_start, args.profile_steps
            prof = _tp(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=_sched(wait=start, warmup=1, active=nrec, repeat=1),
                on_trace_ready=lambda p: p.export_chrome_trace("profile_serial.json"),
            )
            prof.__enter__()
            print(f"  profiling steps [{start}, {start + nrec}) -> profile_serial.json")
        t0 = time.perf_counter()
        for x, y in train_loader:
            if step >= args.steps:
                break
            # 1. Frozen encoder inference on PRE-DICED caption microbatches.
            #    Feeding a nested pre-diced input makes infer() return a nested
            #    pre-diced output too (output mirrors input shape), so each
            #    context microbatch is independent.
            caps = make_captions(y)
            caps_mbs = tuple(caps.chunk(args.micro, dim=0))  # nested pre-diced
            ctx_mbs = enc_pipe.infer(caps_mbs, n_microbatches=args.micro)
            # 2. Build the NESTED pre-diced training input: one (ctx_mb, x_mb)
            #    tuple per microbatch. The training pipeline consumes each
            #    microbatch independently — it can start on microbatch 0 as soon
            #    as its inference is done, without waiting for the rest.
            x_mbs = x.chunk(args.micro, dim=0)
            nested = tuple((ctx_mbs[k], x_mbs[k]) for k in range(args.micro))
            result = pipe.step(nested, targets=y, schedule=args.schedule,
                               n_microbatches=args.micro, loss_fn=loss_fn)
            result.flush_grads(); opt.step(); opt.zero_grad(set_to_none=True)
            running += result.loss.item(); losses.append(result.loss.item()); step += 1
            if prof is not None:
                prof.step()
            if step % 10 == 0:
                print(f"  step {step:>4}: loss={running/10:.4f}", flush=True); running = 0.0
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if prof is not None:
            prof.__exit__(None, None, None)
        return dt, losses

    dt, losses = run_loop()
    print(f"\n{args.steps} steps in {dt*1e3:.0f}ms  ({dt/args.steps*1e3:.1f} ms/step)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
