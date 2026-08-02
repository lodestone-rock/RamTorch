"""
pipeline_transformer_perfetto.py
---------------------------------
Real-compute pipeline-parallel demo on an actual transformer with QK-norm,
slow enough to visualize clearly in Perfetto.

Block (decoder-style, pre-norm):
    x = x + MHA(RMSNorm(x))      # causal MHA with per-head QK RMSNorm + RoPE-free
    x = x + SwiGLU(RMSNorm(x))   # gated MLP, ~4x expansion

12 layers, dim 4096, 32 heads. Split 6+6 across 2 GPUs. Input is a token
sequence (B, T, D) so attention does real (T x T) work. One Chrome-trace JSON
per schedule; open at https://ui.perfetto.dev.

Usage:
    python examples/pipeline_transformer_perfetto.py
"""

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.pipelining import SplitPoint

from ramtorch import run_pipeline

# ── Transformer with QK-norm ─────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class CausalMHA(nn.Module):
    """Multi-head causal self-attention with per-head QK RMSNorm."""

    def __init__(self, d: int, n_heads: int):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.n_heads = n_heads
        self.head_dim = d // n_heads
        self.qkv = nn.Linear(d, 3 * d)
        self.o_proj = nn.Linear(d, d)
        # QK-norm: normalize each head's q and k (removes logit scale drift)
        self.q_norm = RMSNorm(self.head_dim)
        self.k_norm = RMSNorm(self.head_dim)

    def forward(self, x):
        B, T, D = x.shape
        qkv = self.qkv(x)  # (B, T, 3D)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, T, hd)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        q = self.q_norm(q)
        k = self.k_norm(k)

        # causal attention; is_causal=True applies the mask internally
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # (B, H, T, hd)
        out = out.transpose(1, 2).reshape(B, T, D)
        return self.o_proj(out)


class SwiGLU(nn.Module):
    """Gated MLP: down(silu(gate(x)) * up(x)), ~4x hidden."""

    def __init__(self, d: int, mult: int = 4):
        super().__init__()
        h = d * mult
        self.gate = nn.Linear(d, h)
        self.up = nn.Linear(d, h)
        self.down = nn.Linear(h, d)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int):
        super().__init__()
        self.norm1 = RMSNorm(d)
        self.attn = CausalMHA(d, n_heads)
        self.norm2 = RMSNorm(d)
        self.mlp = SwiGLU(d)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, d: int = 4096, n_layers: int = 12, n_heads: int = 32, out_dim: int = 4096):
        super().__init__()
        self.embed = nn.Linear(d, d)
        self.layers = nn.ModuleList([Block(d, n_heads) for _ in range(n_layers)])
        self.norm_f = RMSNorm(d)
        self.head = nn.Linear(d, out_dim)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(self.norm_f(x))


# ── Minimal matmul-residual (tiny trace: 1 GEMM + add per layer) ─────────────

class MatmulResidLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return x + self.lin(x)


class MatmulResidNet(nn.Module):
    def __init__(self, d: int, n_layers: int):
        super().__init__()
        self.embed = nn.Linear(d, d)
        self.layers = nn.ModuleList([MatmulResidLayer(d) for _ in range(n_layers)])
        self.head = nn.Linear(d, d)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


# ── Simple FFN (clean trace: few matmuls + norm per layer) ───────────────────

class FFNLayer(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, d * 4)
        self.fc2 = nn.Linear(d * 4, d)

    def forward(self, x):
        h = self.norm(x)
        return x + self.fc2(F.gelu(self.fc1(h)))


class FFNNet(nn.Module):
    def __init__(self, d: int, n_layers: int):
        super().__init__()
        self.embed = nn.Linear(d, d)
        self.layers = nn.ModuleList([FFNLayer(d) for _ in range(n_layers)])
        self.head = nn.Linear(d, d)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


# ── Config ────────────────────────────────────────────────────────────────────

# Toggle model: "matmul" (tiny: 1 GEMM+add/layer), "ffn" (clean),
#               or "transformer" (heavy, realistic)
MODEL = "ffn"

# Toggle: also capture a full torch.profiler trace (CPU dispatch + CUDA kernels
# + memcpies) per schedule, written to profile_<sched>.json
PROFILE = True

DEVICES = ["cuda:1", "cuda:3"]
N_MICRO = 8
loss_fn = nn.MSELoss()

if MODEL == "transformer":
    DIM, N_LAYERS, N_HEADS = 4096, 12, 32
    SEQ, BATCH = 1024, 16          # (B, T, D) input, causal attention
    SPLIT_SPEC = {"layers.6": SplitPoint.BEGINNING}
    def build_model():
        return Transformer(DIM, N_LAYERS, N_HEADS)
    INPUT_SHAPE = (BATCH, SEQ, DIM)
elif MODEL == "ffn":
    DIM, N_LAYERS = 4096, 10
    BATCH = 64                     # (B, D) input, no sequence dim
    SPLIT_SPEC = {"layers.5": SplitPoint.BEGINNING}
    def build_model():
        return FFNNet(DIM, N_LAYERS)
    INPUT_SHAPE = (BATCH, DIM)
else:  # matmul
    DIM, N_LAYERS = 4096, 10
    BATCH = 64                     # (B, D) input
    SPLIT_SPEC = {"layers.5": SplitPoint.BEGINNING}
    def build_model():
        return MatmulResidNet(DIM, N_LAYERS)
    INPUT_SHAPE = (BATCH, DIM)


def main():
    if torch.cuda.device_count() < 2:
        raise SystemExit("need at least 2 GPUs")

    n_params = sum(p.numel() for p in build_model().parameters())
    print(f"model: {MODEL}  params: {n_params/1e6:.1f}M ({n_params*4/1e9:.2f} GB fp32)", flush=True)
    print(f"input: {INPUT_SHAPE}, {N_MICRO} microbatches", flush=True)

    torch.manual_seed(0)
    data = torch.randn(*INPUT_SHAPE)
    targets = torch.randn(*INPUT_SHAPE)

    def fresh_model():
        torch.manual_seed(1234)
        return build_model()

    ref_loss = None
    for sched in ["gpipe", "1f1b"]:
        m = fresh_model()
        trace = f"trace_{sched}.json"
        prof = f"profile_{sched}.json" if PROFILE else None
        print(f"running {sched} ...", flush=True)
        t0 = time.perf_counter()
        result = run_pipeline(
            m,
            example_input=data[: BATCH // N_MICRO],  # one microbatch: (B/m, T, D)
            split_spec=SPLIT_SPEC,
            data=data,
            targets=targets,
            schedule=sched,
            n_microbatches=N_MICRO,
            loss_fn=loss_fn,
            devices=DEVICES,
            overlap=True,
            trace_path=trace,
            profile_path=prof,
        )
        dt = time.perf_counter() - t0
        loss = result.loss.item()
        if ref_loss is None:
            ref_loss = loss
        ok = "OK " if abs(loss - ref_loss) < 1e-4 * abs(ref_loss) else "DIFF"
        size = os.path.getsize(trace) / 1e3
        msg = f"[{ok}] {sched:<8} loss={loss:.6f}  wall={dt*1e3:7.1f}ms  -> {trace} ({size:.0f} KB)"
        if prof:
            msg += f"  + {prof} ({os.path.getsize(prof)/1e6:.1f} MB)"
        print(msg, flush=True)

    print("\nOpen traces at https://ui.perfetto.dev", flush=True)


if __name__ == "__main__":
    main()
