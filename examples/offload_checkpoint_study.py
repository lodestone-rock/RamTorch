"""
offload_checkpoint_study.py
---------------------------
Compare OffloadModel's three backward strategies head to head:

  recompute   keep_activations=False        engine reloads + re-runs each
                                            chunk's forward during backward
  keep        keep_activations=True         forward keeps every chunk's graph
                                            and internal activations
  checkpoint  keep_activations="checkpoint" keep-mode plumbing, but each chunk
                                            runs under non-reentrant
                                            torch.utils.checkpoint — autograd
                                            recomputes internals at backward,
                                            with RNG stashed/restored

The question this study answers: does torch-native gradient checkpointing
(instead of the engine's own recompute) buy us anything? Expected: yes —
recompute-mode activation memory AND dropout safety in one mode, because
checkpoint replays the SAME rng (engine recompute resamples dropout masks,
which is why recompute + dropout is forbidden).

Plus the SELECTIVE middle ground: `ramtorch.offload_checkpoint(module, *args)`
lets the user mark, inside their own forward, exactly which sub-parts of a
chunk to checkpoint (run with keep_activations=True; unmarked parts keep
their activations). A bare torch.utils.checkpoint would break there — the
engine calls chunks via functional_call, which reverts params to the CPU
masters before the backward-time recompute runs — so the helper snapshots
the effective (streamed GPU) tensors at forward time and re-applies them
for the recompute.

Four sections:
  1. Deterministic parity: all three modes vs a full-resident plain model,
     bit-identical loss trajectories + final weights over SGD steps.
  2. Dropout parity: keep and checkpoint stay bit-identical to plain;
     recompute drifts (masks resampled at backward — shown, not hidden).
  3. Memory/time: an activation-heavy config where keep's cost shows up;
     checkpoint should sit near recompute's peak memory at similar speed.
  4. Selective marks: user-marked chunks (light part kept, heavy 4x
     expansion checkpointed) — bit-identical to an ordinary unmarked plain
     model (incl. dropout), peak memory between keep and checkpoint-all.

Run:
    python examples/offload_checkpoint_study.py --device cuda:1
"""

from __future__ import annotations

import argparse
import copy
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ramtorch import OffloadModel, offload_checkpoint

MODES = [("recompute", False), ("keep", True), ("checkpoint", "checkpoint")]


def build_chunks(d: int, blocks: int, dropout: float = 0.0) -> list[nn.Module]:
    def block() -> nn.Module:
        layers: list[nn.Module] = [nn.Linear(d, 4 * d), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(4 * d, d))
        return nn.Sequential(*layers)

    return [block() for _ in range(blocks)]


class SelectiveBlock(nn.Module):
    """A chunk whose forward marks only the heavy 4x expansion for recompute.

    ``mark=False`` gives the identical model with an ordinary forward, for
    apples-to-apples references and benchmarks.
    """

    def __init__(self, d: int, dropout: float = 0.0, mark: bool = True):
        super().__init__()
        self.mark = mark
        self.light = nn.Linear(d, d)
        layers: list[nn.Module] = [nn.Linear(d, 4 * d), nn.GELU()]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(4 * d, d))
        self.heavy = nn.Sequential(*layers)

    def forward(self, x):
        h = self.light(x)          # activations kept either way
        if self.mark:
            return offload_checkpoint(self.heavy, h)
        return self.heavy(h)


def build_selective(d: int, blocks: int, dropout: float = 0.0,
                    mark: bool = True, seed: int = 42) -> list[nn.Module]:
    torch.manual_seed(seed)  # same seed => same init for mark=True/False twins
    return [SelectiveBlock(d, dropout, mark) for _ in range(blocks)]


def make_data(steps: int, batch: int, d: int) -> list[tuple]:
    torch.manual_seed(0)
    return [(torch.randn(batch, d), torch.randn(batch, d))
            for _ in range(steps)]


def run_plain(template, data, lr, device):
    torch.manual_seed(1234)
    model = nn.Sequential(*[copy.deepcopy(c) for c in template]).to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for x, y in data:
        out = model(x.to(device))
        loss = F.mse_loss(out, y.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    params = [p.detach().cpu().clone() for p in model.parameters()]
    return losses, params


def run_offload(template, mode, data, lr, device, window=2, pin=0):
    torch.manual_seed(1234)
    model = OffloadModel(
        [copy.deepcopy(c) for c in template],
        device=device, window=window, pin=pin, keep_activations=mode,
    )
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    losses = []
    for x, y in data:
        res = model.step(x, targets=y, loss_fn=F.mse_loss)
        model.flush_grads()
        opt.step()
        losses.append(res.loss.item())
    params = [p.detach().cpu().clone() for p in model.parameters()]
    model.close()
    return losses, params


def compare(name, ref_losses, ref_params, losses, params):
    dl = max(abs(a - b) for a, b in zip(ref_losses, losses))
    dw = max((a - b).abs().max().item() for a, b in zip(ref_params, params))
    flag = "BIT-IDENTICAL" if dl == 0.0 and dw == 0.0 else ""
    print(f"  {name:<12} max|dloss|={dl:.3e}  max|dweight|={dw:.3e}  {flag}")
    return dl == 0.0 and dw == 0.0


def bench_mode(template, mode, device, batch, d, warmup=2, iters=8):
    dev = torch.device(device)
    model = OffloadModel([copy.deepcopy(c) for c in template],
                         device=dev, window=2, pin=0, keep_activations=mode)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    x = torch.randn(batch, d)
    y = torch.randn(batch, d)

    def one_step():
        model.step(x, targets=y, loss_fn=F.mse_loss)
        model.flush_grads()
        opt.step()

    for _ in range(warmup):
        one_step()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.perf_counter()
    for _ in range(iters):
        one_step()
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    dt = (time.perf_counter() - t0) / iters
    peak = (torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else 0)
    stall = model.stats["acquire_wait_s"]
    model.close()
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return dt, peak, stall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device",
                    default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    # parity config (small, fast)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    # perf config (activation-heavy so keep-mode's memory cost is visible)
    ap.add_argument("--perf-d", type=int, default=1024)
    ap.add_argument("--perf-blocks", type=int, default=12)
    ap.add_argument("--perf-batch", type=int, default=4096)
    args = ap.parse_args()
    dev = torch.device(args.device)
    ok = True

    # ── 1. Deterministic parity ───────────────────────────────────────────────
    print(f"== 1. deterministic parity vs plain full-resident ({args.steps} "
          f"SGD steps, d={args.d}, {args.blocks} blocks) ==")
    template = build_chunks(args.d, args.blocks)
    data = make_data(args.steps, args.batch, args.d)
    ref_losses, ref_params = run_plain(template, data, args.lr, dev)
    for name, mode in MODES:
        losses, params = run_offload(template, mode, data, args.lr, dev)
        ok &= compare(name, ref_losses, ref_params, losses, params)

    # ── 2. Dropout parity ─────────────────────────────────────────────────────
    # keep and checkpoint must match plain bit-exactly (same masks, forward
    # RNG consumption identical; checkpoint stashes/restores RNG for its
    # backward recompute). Engine recompute RESAMPLES masks at backward:
    # shown here as real drift, which is why recompute+dropout is forbidden.
    print(f"\n== 2. dropout parity (p=0.1) ==")
    template = build_chunks(args.d, args.blocks, dropout=0.1)
    ref_losses, ref_params = run_plain(template, data, args.lr, dev)
    for name, mode in MODES:
        losses, params = run_offload(template, mode, data, args.lr, dev)
        exact = compare(name, ref_losses, ref_params, losses, params)
        if name == "recompute":
            print("               ^ expected drift: recompute resamples "
                  "dropout masks at backward")
        else:
            ok &= exact

    # ── 3. Memory / time ──────────────────────────────────────────────────────
    print(f"\n== 3. memory/time (d={args.perf_d}, {args.perf_blocks} blocks, "
          f"batch={args.perf_batch}, window=2 pin=0) ==")
    template = build_chunks(args.perf_d, args.perf_blocks)
    n_params = sum(p.numel() for c in template for p in c.parameters())
    print(f"   {n_params / 1e6:.0f}M params "
          f"({n_params * 4 / 2**20:.0f} MiB fp32)")
    print(f"   {'mode':<12} {'step ms':>9} {'peak GPU MiB':>13} "
          f"{'stall ms/step':>14}")
    for name, mode in MODES:
        dt, peak, stall = bench_mode(template, mode, str(dev),
                                     args.perf_batch, args.perf_d)
        print(f"   {name:<12} {dt * 1e3:>9.1f} {peak / 2**20:>13.0f} "
              f"{stall * 1e3 / 10:>14.1f}")

    # ── 4. Selective user-marked checkpointing (offload_checkpoint) ───────────
    # The reference is an ORDINARY unmarked plain model — proving the marks
    # change memory, not math. keep_activations=True on the offload side;
    # only the marked heavy submodules recompute.
    print(f"\n== 4. selective marks: offload_checkpoint(self.heavy, x) inside "
          f"the chunk forward, keep_activations=True ==")
    for label, p_drop in (("deterministic", 0.0), ("dropout p=0.1", 0.1)):
        marked = build_selective(args.d, args.blocks, p_drop, mark=True)
        unmarked = build_selective(args.d, args.blocks, p_drop, mark=False)
        ref_losses, ref_params = run_plain(unmarked, data, args.lr, dev)
        losses, params = run_offload(marked, True, data, args.lr, dev)
        print(f"  [{label}] offload keep+marks vs plain unmarked:")
        ok &= compare("selective", ref_losses, ref_params, losses, params)

    if dev.type == "cuda":
        print(f"\n   memory/time with marks (same perf config, "
              f"SelectiveBlock = light d\u00d7d kept + heavy 4x marked):")
        print(f"   {'variant':<18} {'step ms':>9} {'peak GPU MiB':>13}")
        variants = [
            ("keep, no marks",
             build_selective(args.perf_d, args.perf_blocks, mark=False), True),
            ("keep + marks",
             build_selective(args.perf_d, args.perf_blocks, mark=True), True),
            ("checkpoint (all)",
             build_selective(args.perf_d, args.perf_blocks, mark=False),
             "checkpoint"),
        ]
        for vname, template, mode in variants:
            dt, peak, _ = bench_mode(template, mode, str(dev),
                                     args.perf_batch, args.perf_d)
            print(f"   {vname:<18} {dt * 1e3:>9.1f} {peak / 2**20:>13.0f}")

    print(f"\n{'ALL PARITY CHECKS PASSED' if ok else 'PARITY FAILURES — see above'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
