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

Run:
    python examples/mnist_pipeline_big_transformer_manual.py
    python examples/mnist_pipeline_big_transformer_manual.py --steps 200 --devices cuda:1 cuda:3
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
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(n_blocks)])
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

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
    stages = build_stages(args.dim, args.depth, args.heads, args.patch,
                          num_classes=10, n_stages=n_stages)
    n_params = sum(p.numel() for st in stages for p in st.parameters())
    print(f"model: {n_params/1e6:.1f}M params, manually split into {n_stages} stages")
    for i, st in enumerate(stages):
        sp = sum(p.numel() for p in st.parameters())
        print(f"  stage {i} ({type(st).__name__}): {sp/1e6:.1f}M params -> {args.devices[i]}")

    # ── Build the pipeline from the manual stages ─────────────────────────────
    pipe = Pipeline(stage_modules=stages, devices=args.devices)
    # Optimize over ALL stage params (they're separate modules, not one model).
    opt = torch.optim.Adam(
        itertools.chain(*(st.parameters() for st in stages)), lr=args.lr
    )
    loss_fn = nn.CrossEntropyLoss()
    print(f"schedule={args.schedule}  microbatches={args.micro}  Adam lr={args.lr}")

    # ── Train ─────────────────────────────────────────────────────────────────
    print(f"\ntraining ({args.steps} steps) ...")
    step = 0
    running = 0.0
    done = False
    while not done:
        for x, y in train_loader:
            result = pipe.step(
                x, targets=y, schedule=args.schedule,
                n_microbatches=args.micro, loss_fn=loss_fn,
            )
            result.flush_grads()
            opt.step()
            opt.zero_grad(set_to_none=True)

            running += result.loss.item()
            step += 1
            if step % 10 == 0:
                print(f"  step {step:>4}: loss={running/10:.4f}", flush=True)
                running = 0.0
            if step >= args.steps:
                done = True
                break

    # ── Evaluate: run inference by relaying through the stages manually ───────
    for st in stages:
        st.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            h = x
            for st, dev in zip(stages, args.devices):
                h = st(h.to(dev))          # relay activations stage -> stage
            pred = h.argmax(1).cpu()
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = correct / total
    print(f"\ntest accuracy after {step} steps: {acc:.4f}")

    # ── Checkpoint: save each stage's state_dict ──────────────────────────────
    if args.save:
        ckpt = {f"stage_{i}": st.state_dict() for i, st in enumerate(stages)}
        torch.save(ckpt, args.save)
        print(f"saved checkpoint -> {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
