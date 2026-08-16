"""
activation_hooks_probe.py
-------------------------
Empirical probe for the planned ACTIVATION-OFFLOAD feature: can
``torch.autograd.graph.saved_tensors_hooks`` grab a chunk's activations,
move them to CPU, and restore them bit-exactly at backward — and how do the
hooks interact with non-reentrant ``torch.utils.checkpoint``, the offload
engine's weight streaming (``graph_tensors`` + the ``resize_(0)`` evict/refill
trick), worker threads, and autocast?

Sections (each prints PASS/FAIL evidence):
  A. Inventory — what pack actually sees: plain call vs ``functional_call``
     (the engine's calling convention) vs non-reentrant checkpoint.
  B. Round-trip parity — offload-to-CPU hook variants (naive sync, pinned +
     event on the current stream, dedicated D2H stream + events) vs a
     no-hook baseline: grads must be BIT-EXACT.  Plus the built-in
     ``save_on_cpu(pin_memory=True)`` and a data_ptr-keyed dedup cache.
  C. Checkpoint interplay — hooks outside the checkpoint (should see only
     the region INPUTS = the chunk-boundary activation), hooks inside the
     checkpointed fn (recompute behavior), and the exact
     hooks + checkpoint + functional_call nesting of
     ``OffloadStage.forward_one_chunk``.
  D. Engine interplay — hooks wrapped around a real ``OffloadModel.step()``
     in keep / "checkpoint" modes: identify the streamed ``graph_tensors``
     weights and pass them through untouched; evict/refill must stay
     bit-exact with hooks active (version-counter safety).
  E. Threads + autocast — hooks are thread-local: forward+backward on a
     worker thread (relay pattern) works; hooks on another thread see
     nothing; a bf16 autocast run stays bit-exact.
  F. Measurement (GPU only) — peak memory + ms/step for
     {no hooks, sync, async} x {keep, checkpoint} on a 12-chunk MLP.

Run:  PYTHONPATH=. env/bin/python examples/activation_hooks_probe.py
      [--device cuda:1] [--skip-f]
"""

from __future__ import annotations

import argparse
import copy
import threading
import time
from collections import defaultdict
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as torch_ckpt
from torch.autograd.graph import save_on_cpu, saved_tensors_hooks
from torch.func import functional_call

FAILURES: list = []
OBSERVATIONS: list = []


def banner(title: str):
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def check(ok: bool, name: str, detail: str = ""):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")


def note(msg: str):
    print(f"  · {msg}")
    OBSERVATIONS.append(msg)


def grads_bitexact(a: list, b: list):
    """(all bit-equal?, max abs diff) over two grad lists."""
    worst = 0.0
    exact = True
    for ga, gb in zip(a, b):
        if ga is None and gb is None:
            continue
        if (ga is None) != (gb is None):
            return False, float("inf")
        if not torch.equal(ga, gb):
            exact = False
            worst = max(worst, (ga - gb).abs().max().item())
    return exact, worst


# ──────────────────────────────────────────────────────────────────────────
# Hook factories.  Every pack payload is a tuple whose first element tags
# the strategy; unpack dispatches on it.
# ──────────────────────────────────────────────────────────────────────────

def _should_offload(t: torch.Tensor, skip_fn) -> bool:
    if not t.is_floating_point():
        return False
    if skip_fn is not None and skip_fn(t):
        return False
    return True


def offload_hooks_sync(stats, skip_fn=None, phase=None):
    """Naive synchronous offload: ``.to('cpu', copy=True)`` at pack,
    blocking copy back at unpack.  On a CPU device the 'offload' degrades to
    a clone so the mechanics are still exercised."""

    def pack(t):
        stats["pack"] += 1
        if phase is not None:
            stats[f"pack_{phase[0]}"] += 1
        if not _should_offload(t, skip_fn):
            stats["passthrough"] += 1
            return ("raw", t)
        stats["offloaded"] += 1
        stats["bytes"] += t.numel() * t.element_size()
        return ("cpu", t.detach().to("cpu", copy=True), t.device)

    def unpack(p):
        stats["unpack"] += 1
        if p[0] == "raw":
            return p[1]
        _, cpu_t, dev = p
        return cpu_t.to(dev)

    return saved_tensors_hooks(pack, unpack)


def offload_hooks_pinned(stats, skip_fn=None):
    """CUDA: non_blocking D2H into a fresh pinned buffer on the CURRENT
    stream + a completion event; unpack makes the consumer stream wait the
    event before the H2D copy back."""

    def pack(t):
        stats["pack"] += 1
        if not t.is_cuda or not _should_offload(t, skip_fn):
            stats["passthrough"] += 1
            return ("raw", t)
        tt = t if t.is_contiguous() else t.contiguous()
        buf = torch.empty(tt.shape, dtype=tt.dtype, pin_memory=True)
        buf.copy_(tt, non_blocking=True)  # D2H, ordered on the current stream
        done = torch.cuda.Event()
        done.record(torch.cuda.current_stream(t.device))
        stats["offloaded"] += 1
        stats["bytes"] += t.numel() * t.element_size()
        return ("pinned", buf, done, t.device)

    def unpack(p):
        stats["unpack"] += 1
        if p[0] == "raw":
            return p[1]
        _, buf, done, dev = p
        # GPU-side ordering: H2D issued after this waits until the D2H landed
        torch.cuda.current_stream(dev).wait_event(done)
        return buf.to(dev, non_blocking=True)

    return saved_tensors_hooks(pack, unpack)


class PinnedPool:
    """Reusable pinned staging buffers keyed by (shape, dtype).  Call
    ``reset()`` only after a full device synchronize (end of step)."""

    def __init__(self):
        self._free = defaultdict(list)
        self._all = []

    def take(self, t: torch.Tensor) -> torch.Tensor:
        key = (tuple(t.shape), t.dtype)
        lst = self._free[key]
        if lst:
            return lst.pop()
        buf = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        self._all.append((key, buf))
        return buf

    def reset(self):
        self._free.clear()
        for key, buf in self._all:
            self._free[key].append(buf)


def offload_hooks_stream(d2h_stream, stats, skip_fn=None, pool=None):
    """CUDA: D2H on a DEDICATED stream with full event discipline —
    the async variant a real feature would use.  ``record_stream`` keeps the
    caching allocator from recycling the source before the copy lands."""

    def pack(t):
        stats["pack"] += 1
        if not t.is_cuda or not _should_offload(t, skip_fn):
            stats["passthrough"] += 1
            return ("raw", t)
        cur = torch.cuda.current_stream(t.device)
        ready = torch.cuda.Event()
        ready.record(cur)
        d2h_stream.wait_event(ready)  # producer finished writing t
        with torch.cuda.stream(d2h_stream):
            tt = t if t.is_contiguous() else t.contiguous()
            buf = pool.take(tt) if pool is not None else torch.empty(
                tt.shape, dtype=tt.dtype, pin_memory=True)
            buf.copy_(tt, non_blocking=True)
            t.record_stream(d2h_stream)
            if tt is not t:
                tt.record_stream(d2h_stream)
        done = torch.cuda.Event()
        done.record(d2h_stream)
        stats["offloaded"] += 1
        stats["bytes"] += t.numel() * t.element_size()
        return ("stream", buf, done, t.device)

    def unpack(p):
        stats["unpack"] += 1
        if p[0] == "raw":
            return p[1]
        _, buf, done, dev = p
        torch.cuda.current_stream(dev).wait_event(done)
        return buf.to(dev, non_blocking=True)

    return saved_tensors_hooks(pack, unpack)


def offload_hooks_dedup(stats, skip_fn=None):
    """Sync offload behind a cache keyed on (data_ptr, dtype, shape, stride,
    version): a tensor saved by N ops crosses PCIe once."""
    cache = {}

    def pack(t):
        stats["pack"] += 1
        if not _should_offload(t, skip_fn):
            stats["passthrough"] += 1
            return ("raw", t)
        key = (t.data_ptr(), t.dtype, tuple(t.shape),
               tuple(t.stride()), t._version)
        hit = cache.get(key)
        if hit is not None:
            stats["dedup_hits"] += 1
            return hit
        stats["copies"] += 1
        payload = ("cpu", t.detach().to("cpu", copy=True), t.device)
        cache[key] = payload
        return payload

    def unpack(p):
        stats["unpack"] += 1
        if p[0] == "raw":
            return p[1]
        return p[1].to(p[2])

    return saved_tensors_hooks(pack, unpack)


# ──────────────────────────────────────────────────────────────────────────
class TinyAttn(nn.Module):
    def __init__(self, d: int, heads: int = 4):
        super().__init__()
        self.h = heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, T, d = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)

        def sp(t):
            return t.view(B, T, self.h, d // self.h).transpose(1, 2)

        q, k, v = sp(q), sp(k), sp(v)
        a = torch.softmax(q @ k.transpose(-2, -1) / (d // self.h) ** 0.5, -1)
        o = (a @ v).transpose(1, 2).reshape(B, T, d)
        return self.proj(o)


# ══════════════════════════════════════════════════════════════════════════
# A. Inventory
# ══════════════════════════════════════════════════════════════════════════
def section_a(dev: torch.device):
    banner("A. Inventory — what does pack actually see?")
    d = 32
    torch.manual_seed(0)
    mlp = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)).to(dev)
    attn = TinyAttn(d).to(dev)
    x = torch.randn(4, d, device=dev)
    xt = torch.randn(2, 6, d, device=dev)

    def inventory(label, fn, inp, named_params, max_rows=10):
        param_ptrs = {p.data_ptr(): n for n, p in named_params}
        recs = []

        def pack(t):
            recs.append((tuple(t.shape),
                         str(t.dtype).replace("torch.", ""),
                         param_ptrs.get(t.data_ptr()),
                         t.is_leaf))
            return t

        with saved_tensors_hooks(pack, lambda t: t):
            out = fn(inp)
        out.sum().backward()
        n_param = sum(1 for r in recs if r[2] is not None)
        n_act = len(recs) - n_param
        print(f"  -- {label}: {len(recs)} saved "
              f"({n_param} weight/param, {n_act} activation)")
        for shape, dt, pname, leaf in recs[:max_rows]:
            kind = (f"param:{pname}" if pname
                    else ("leaf" if leaf else "activation"))
            print(f"       {str(shape):>20} {dt:>9}  {kind}")
        if len(recs) > max_rows:
            print(f"       ... {len(recs) - max_rows} more")
        return recs

    # 1. plain call: saved tensors include the module's own weights.
    # NB: a Linear's weight is saved only when its INPUT requires grad
    # (the weight is needed for grad-wrt-input) — the first layer's weight
    # never shows up because x has requires_grad=False.
    recs = inventory("plain call (MLP)", mlp, x, mlp.named_parameters())
    n_param = sum(1 for r in recs if r[2] is not None)
    check(n_param >= 1, "plain call saves weights too",
          f"{n_param} of {len(recs)} packed tensors are module params")
    note("a layer's weight is saved only if its input requires grad — the "
         "first chunk of a first stage saves fewer weights than the rest")

    # 2. functional_call with swapped-in tensors (the engine's convention):
    #    the SAVED weights are the swapped tensors, NOT the module params
    td = {n: p.detach().clone().requires_grad_(True)
          for n, p in mlp.named_parameters()}
    recs = inventory("functional_call (engine convention)",
                     lambda i: functional_call(mlp, td, (i,)), x, td.items())
    module_ptrs = {p.data_ptr() for p in mlp.parameters()}
    saved_ptrs = set()

    def pack2(t):
        saved_ptrs.add(t.data_ptr())
        return t

    with saved_tensors_hooks(pack2, lambda t: t):
        functional_call(mlp, td, (x,)).sum().backward()
    check(not (saved_ptrs & module_ptrs),
          "functional_call saves the swapped-in tensors, not module params",
          "0 module-param data_ptrs in the pack log")
    swapped_hits = sum(1 for r in recs if r[2] is not None)
    check(swapped_hits >= 1, "swapped tensors identified by data_ptr",
          f"{swapped_hits} match(es) — an engine-weight filter by data_ptr "
          "works")

    # 3. non-reentrant checkpoint: outer hooks see only the region INPUTS
    xg = x.clone().requires_grad_(True)
    inventory(
        "non-reentrant checkpoint (hooks OUTSIDE)",
        lambda i: torch_ckpt.checkpoint(mlp, i, use_reentrant=False),
        xg, mlp.named_parameters())
    packed_ptrs = []

    def pack3(t):
        packed_ptrs.append(t.data_ptr())
        return t

    xg2 = x.clone().requires_grad_(True)
    with saved_tensors_hooks(pack3, lambda t: t):
        out = torch_ckpt.checkpoint(mlp, xg2, use_reentrant=False)
    out.sum().backward()
    check(len(packed_ptrs) <= 2 and xg2.data_ptr() in packed_ptrs,
          "outer hooks see only checkpoint-region inputs",
          f"packed {len(packed_ptrs)} tensor(s), incl. the region input — "
          "internals were dropped by checkpoint's own inner hooks")

    # 4. attention block for reference (softmax output etc. get saved)
    inventory("plain call (tiny attention)", attn, xt,
              attn.named_parameters())
    note("pack fires for EVERY autograd-saved tensor: weights, op outputs "
         "(GELU/softmax), matmul operands — a real feature MUST filter")


# ══════════════════════════════════════════════════════════════════════════
# B. Round-trip parity
# ══════════════════════════════════════════════════════════════════════════
def section_b(dev: torch.device):
    banner("B. Round-trip parity — offload to CPU and back, bit-exact?")
    d, B = 128, 16
    torch.manual_seed(1)
    master = nn.Sequential(
        *[nn.Sequential(nn.Linear(d, d), nn.GELU()) for _ in range(3)]
    ).to(dev)
    x = torch.randn(B, d, device=dev)
    y = torch.randn(B, d, device=dev)

    def run(mk_ctx):
        """mk_ctx(model) -> context manager or None."""
        model = copy.deepcopy(master)
        ctx = mk_ctx(model) if mk_ctx is not None else None
        if ctx is None:
            out = model(x)
        else:
            with ctx:
                out = model(x)
        loss = F.mse_loss(out, y)
        loss.backward()  # NB: backward OUTSIDE the hook ctx — unpack fires anyway
        return (loss.detach().clone(),
                [p.grad.detach().clone() for p in model.parameters()])

    base_loss, base_grads = run(None)

    def act_filter(model):
        """The FEATURE configuration: weights pass through, only
        activations offload (the engine already streams weights)."""
        ptrs = {p.data_ptr() for p in model.parameters()}
        return lambda t: t.data_ptr() in ptrs

    variants = [("sync .cpu()",
                 lambda m, s: offload_hooks_sync(s, skip_fn=act_filter(m)))]
    if dev.type == "cuda":
        variants += [
            ("pinned + event (current stream)",
             lambda m, s: offload_hooks_pinned(s, skip_fn=act_filter(m))),
            ("dedicated D2H stream + events",
             lambda m, s: offload_hooks_stream(torch.cuda.Stream(dev), s,
                                               skip_fn=act_filter(m))),
        ]

    for name, mk in variants:
        stats = defaultdict(int)
        loss, grads = run(lambda m: mk(m, stats))
        exact, worst = grads_bitexact(base_grads, grads)
        check(exact and torch.equal(loss, base_loss),
              f"{name} [activations only]: grads bit-exact vs baseline",
              f"offloaded {stats['offloaded']} tensors "
              f"({stats['bytes'] / 1e6:.1f} MB), max grad diff {worst:.1e}")
        check(stats["unpack"] >= stats["offloaded"],
              f"{name}: every offloaded tensor was unpacked",
              f"pack {stats['pack']} / unpack {stats['unpack']}")

    # Offloading EVERYTHING (weights included) is NOT the feature config;
    # on CUDA a pinned round trip of the WEIGHTS wobbles grads by 1-2 ULP
    # (~3e-10) even though the restored bits are identical — and torch's own
    # save_on_cpu(pin_memory=True) shows the IDENTICAL wobble.  Verify our
    # implementation matches the upstream reference exactly.
    pin = dev.type == "cuda"
    _, soc_grads = run(lambda m: save_on_cpu(pin_memory=pin))
    exact, worst = grads_bitexact(base_grads, soc_grads)
    check(worst < 1e-8, f"save_on_cpu(pin_memory={pin}) [everything]",
          f"max grad diff vs baseline {worst:.1e} "
          f"({'bit-exact' if exact else '1-2 ULP wobble, weights round-trip'})")
    if dev.type == "cuda":
        stats = defaultdict(int)
        _, all_grads = run(lambda m: offload_hooks_pinned(stats))
        exact2, worst2 = grads_bitexact(soc_grads, all_grads)
        check(exact2,
              "pinned hooks [everything] bit-identical to save_on_cpu",
              f"max diff {worst2:.1e} — our hooks == torch's reference impl")
        note("offloading WEIGHTS through pinned memory costs 1-2 ULP on "
             "grads (identical to torch's save_on_cpu); activations-only "
             "offload — the feature config — is bit-exact")

    # dedup: one tensor consumed (and saved) by two Linears
    class TwoBranch(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = nn.Linear(d, d)
            self.b = nn.Linear(d, d)

        def forward(self, inp):
            h = torch.relu(inp)
            return self.a(h) + self.b(h)  # h saved twice

    torch.manual_seed(2)
    tb_master = TwoBranch().to(dev)

    def run_tb(ctx):
        m = copy.deepcopy(tb_master)
        if ctx is None:
            out = m(x)
        else:
            with ctx:
                out = m(x)
        F.mse_loss(out, y).backward()
        return [p.grad.detach().clone() for p in m.parameters()]

    tb_base = run_tb(None)
    stats = defaultdict(int)
    tb_grads = run_tb(offload_hooks_dedup(stats))
    exact, worst = grads_bitexact(tb_base, tb_grads)
    check(stats["dedup_hits"] >= 1,
          "dedup: same tensor saved by several ops packs once",
          f"{stats['copies']} copies for {stats['copies'] + stats['dedup_hits']}"
          f" saves ({stats['dedup_hits']} cache hits)")
    check(exact, "dedup: grads still bit-exact", f"max diff {worst:.1e}")


# ══════════════════════════════════════════════════════════════════════════
# C. Non-reentrant checkpoint interplay
# ══════════════════════════════════════════════════════════════════════════
def section_c(dev: torch.device):
    banner("C. Interplay with non-reentrant torch.utils.checkpoint")
    d = 64
    torch.manual_seed(3)
    mod = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d)).to(dev)
    x0 = torch.randn(8, d, device=dev)

    def clear():
        for p in mod.parameters():
            p.grad = None

    n_calls = [0]

    def fn(inp):
        n_calls[0] += 1
        return mod(inp)

    def snap():
        return [p.grad.detach().clone() for p in mod.parameters()]

    # baseline: checkpoint, no hooks
    clear()
    n_calls[0] = 0
    xb = x0.clone().requires_grad_(True)
    torch_ckpt.checkpoint(fn, xb, use_reentrant=False).pow(2).mean().backward()
    base = ([xb.grad.detach().clone()], snap())
    check(n_calls[0] == 2, "baseline recomputes once at backward",
          f"fn ran {n_calls[0]}x (fwd + recompute)")

    # C1: hooks OUTSIDE the checkpoint call (feature nesting)
    clear()
    n_calls[0] = 0
    x1 = x0.clone().requires_grad_(True)
    stats = defaultdict(int)
    phase = ["fwd"]
    with offload_hooks_sync(stats, phase=phase):
        out = torch_ckpt.checkpoint(fn, x1, use_reentrant=False)
    phase[0] = "bwd"
    out.pow(2).mean().backward()
    exact, worst = grads_bitexact(base[1], snap())
    xexact = torch.equal(base[0][0], x1.grad)
    check(exact and xexact, "C1 hooks outside ckpt: grads bit-exact",
          f"max diff {worst:.1e}")
    check(stats["offloaded"] <= 2 and stats["pack_bwd"] == 0,
          "C1 outer hooks packed only the region input at forward",
          f"offloaded {stats['offloaded']} tensor(s), "
          f"{stats['pack_bwd']} packs during backward")
    check(n_calls[0] == 2, "C1 recompute still happens exactly once",
          f"fn ran {n_calls[0]}x")

    # C1b: hook ctx STILL OPEN during backward — does the recompute leak
    # its internal saves into our hooks?
    clear()
    n_calls[0] = 0
    x1b = x0.clone().requires_grad_(True)
    stats = defaultdict(int)
    phase = ["fwd"]
    with offload_hooks_sync(stats, phase=phase):
        out = torch_ckpt.checkpoint(fn, x1b, use_reentrant=False)
        phase[0] = "bwd"
        out.pow(2).mean().backward()
    exact, worst = grads_bitexact(base[1], snap())
    check(exact and torch.equal(base[0][0], x1b.grad),
          "C1b ctx open across backward: grads bit-exact",
          f"max diff {worst:.1e}")
    note(f"C1b: {stats['pack_bwd']} pack(s) fired during backward/recompute "
         "(checkpoint's own recompute hook is innermost, so region internals "
         "do not reach outer hooks)")

    # C2: hooks INSIDE the checkpointed fn — expected to be a bad idea;
    # observe what actually happens
    clear()
    stats2 = defaultdict(int)
    calls2 = [0]

    def fn_inner(inp):
        calls2[0] += 1
        with offload_hooks_sync(stats2):
            return mod(inp)

    x2 = x0.clone().requires_grad_(True)
    try:
        out = torch_ckpt.checkpoint(fn_inner, x2, use_reentrant=False)
        packs_fwd = stats2["pack"]
        out.pow(2).mean().backward()
        exact, worst = grads_bitexact(base[1], snap())
        xex = torch.equal(base[0][0], x2.grad)
        check(exact and xex, "C2 hooks inside ckpt region: grads bit-exact",
              f"max diff {worst:.1e}")
        recomputes = calls2[0] - 1
        msg = (f"C2: hooks inside the region packed {packs_fwd} at forward, "
               f"{stats2['pack'] - packs_fwd} during recompute (fn ran "
               f"{calls2[0]}x) — inner hooks OVERRIDE checkpoint's, so those "
               "tensors are offloaded instead of dropped")
        if recomputes == 0:
            msg += ("; with every internal save captured, checkpoint had "
                    "NOTHING left to recompute — checkpointing fully defeated")
        note(msg)
    except Exception as e:  # noqa: BLE001 — probe reports whatever happens
        note(f"C2: hooks inside a checkpointed fn RAISED: {type(e).__name__}: "
             f"{e}")
        check(False, "C2 hooks inside ckpt region completed", repr(e))

    # C3: the exact OffloadStage.forward_one_chunk nesting —
    # hooks OUTSIDE, checkpoint(functional_call), weights via closure
    td = {n: p.detach().clone().requires_grad_(True)
          for n, p in mod.named_parameters()}

    def fn3(inp):
        return functional_call(mod, td, (inp,))

    def run3(ctx):
        x3 = x0.clone().requires_grad_(True)
        if ctx is None:
            out = torch_ckpt.checkpoint(fn3, x3, use_reentrant=False)
        else:
            with ctx:
                out = torch_ckpt.checkpoint(fn3, x3, use_reentrant=False)
        loss = out.pow(2).mean()
        return torch.autograd.grad(loss, [x3] + list(td.values()))

    g_base = run3(None)
    stats3 = defaultdict(int)
    g_hook = run3(offload_hooks_sync(stats3))
    exact, worst = grads_bitexact(list(g_base), list(g_hook))
    check(exact, "C3 hooks + checkpoint + functional_call: grads bit-exact",
          f"max diff {worst:.1e}")
    check(stats3["offloaded"] <= 2,
          "C3 outer hooks never saw the closure-captured weights",
          f"offloaded {stats3['offloaded']} tensor(s) — only the chunk "
          "boundary input, exactly what activation offload wants")


# ══════════════════════════════════════════════════════════════════════════
# D. Real OffloadModel with hooks
# ══════════════════════════════════════════════════════════════════════════
def section_d(dev: torch.device):
    banner("D. Hooks around a real OffloadModel.step() (weight streaming)")
    from ramtorch.offload import OffloadModel

    n, d, B = 6, 64, 8
    torch.manual_seed(4)
    chunks = [nn.Sequential(nn.Linear(d, d), nn.GELU()) for _ in range(n)]
    # deepcopy BEFORE any OffloadModel ctor (it relocates params in place)
    ref = nn.Sequential(*copy.deepcopy(chunks)).to(dev)
    x = torch.randn(B, d)
    y = torch.randn(B, d)

    ref.zero_grad()
    ref_loss = F.mse_loss(ref(x.to(dev)), y.to(dev))
    ref_loss.backward()
    ref_grads = {f"{i}.{nme}": p.grad.detach().clone()
                 for i in range(n)
                 for nme, p in ref[i].named_parameters()}

    for keep in (True, "checkpoint"):
        model = OffloadModel(copy.deepcopy(chunks), device=dev, window=2,
                             pin=1, keep_activations=keep)
        stats = defaultdict(int)

        def is_engine_weight(t, _m=model, _s=stats):
            """Streamed graph_tensors + GPU-pinned params, by live data_ptr.
            (Storage ptrs of size-0 evicted chunks are stale — excluded.)"""
            ptr = t.untyped_storage().data_ptr()
            for st in _m._state:
                if st.gpu_pinned:
                    for p in st.tensors.values():
                        if p.untyped_storage().data_ptr() == ptr:
                            _s["engine_weight_passthrough"] += 1
                            return True
                gt = st.graph_tensors
                if gt:
                    for g in gt.values():
                        if (g.untyped_storage().size() > 0
                                and g.untyped_storage().data_ptr() == ptr):
                            _s["engine_weight_passthrough"] += 1
                            return True
            return False

        label = "keep" if keep is True else keep
        with offload_hooks_sync(stats, skip_fn=is_engine_weight):
            res = model.step(x, targets=y, loss_fn=F.mse_loss)
        model.flush_grads()

        errs = []
        worst = 0.0
        for i in range(n):
            for nme, p in model.chunks[i].named_parameters():
                if p.grad is None:
                    errs.append(f"chunk{i}.{nme}: no grad")
                    continue
                dcheck = (p.grad.detach().cpu()
                          - ref_grads[f"{i}.{nme}"].cpu()).abs().max().item()
                worst = max(worst, dcheck)
                if dcheck > 1e-6:
                    errs.append(f"chunk{i}.{nme}: {dcheck:.2e}")
        check(not errs and abs(res.loss.item() - ref_loss.item()) < 1e-7,
              f"[{label}] loss + every grad match reference with hooks active",
              f"max grad err {worst:.2e}")
        note(f"[{label}] pack saw {stats['engine_weight_passthrough']} "
             f"engine-weight saves (passed through), offloaded "
             f"{stats['offloaded']} activation tensors "
             f"({stats['bytes'] / 1e3:.0f} KB), loads={model.stats['loads']}")

        if keep is True:
            check(stats["engine_weight_passthrough"] > 0,
                  "[keep] streamed graph_tensors weights identified in pack",
                  "data_ptr filter against state.graph_tensors works")
        else:
            check(stats["engine_weight_passthrough"] == 0,
                  "[checkpoint] weights hidden inside per-chunk checkpoint",
                  "outer hooks only ever saw boundary activations")

        # evict/refill invariant: streamed storages actually freed
        alive = sum(
            1 for st in model._state
            if st.graph_tensors is not None
            and any(t.untyped_storage().size() > 0
                    for t in st.graph_tensors.values())
        )
        check(alive <= model.window,
              f"[{label}] evict/refill (resize_(0)) intact with hooks",
              f"{alive} streamed chunks hold weight storage, window=2 — "
              "and no version-counter error was raised")
        model.close()


# ══════════════════════════════════════════════════════════════════════════
# E. Threads + autocast
# ══════════════════════════════════════════════════════════════════════════
def section_e(dev: torch.device):
    banner("E. Worker threads (relay pattern) + bf16 autocast")
    d, B = 64, 8
    torch.manual_seed(5)
    master = nn.Sequential(nn.Linear(d, d), nn.GELU(),
                           nn.Linear(d, d)).to(dev)
    x = torch.randn(B, d, device=dev)
    y = torch.randn(B, d, device=dev)

    def run_plain(model):
        F.mse_loss(model(x), y).backward()
        return [p.grad.detach().clone() for p in model.parameters()]

    base = run_plain(copy.deepcopy(master))

    # E1: hooks installed ON the worker thread (forward + backward there)
    box = {}

    def worker():
        try:
            m = copy.deepcopy(master)
            stats = defaultdict(int)
            with offload_hooks_sync(stats):
                out = m(x)
            F.mse_loss(out, y).backward()
            box["grads"] = [p.grad.detach().clone() for p in m.parameters()]
            box["stats"] = stats
        except Exception as e:  # noqa: BLE001
            box["err"] = e

    t = threading.Thread(target=worker)
    t.start()
    t.join()
    if "err" in box:
        check(False, "E1 hooks on worker thread", repr(box["err"]))
    else:
        exact, worst = grads_bitexact(base, box["grads"])
        check(exact, "E1 worker-thread fwd+bwd with hooks: bit-exact",
              f"offloaded {box['stats']['offloaded']}, max diff {worst:.1e}")

    # E2: hooks ctx open on MAIN thread; worker computes WITHOUT hooks —
    # thread-locality means main's hooks must see nothing
    stats2 = defaultdict(int)
    box2 = {}

    def worker2():
        m = copy.deepcopy(master)
        F.mse_loss(m(x), y).backward()
        box2["grads"] = [p.grad.detach().clone() for p in m.parameters()]

    with offload_hooks_sync(stats2):
        t2 = threading.Thread(target=worker2)
        t2.start()
        t2.join()
    check(stats2["pack"] == 0,
          "E2 hooks are thread-local: main-thread ctx saw 0 worker saves",
          "each pipeline stage worker must install its own hooks")
    exact, _ = grads_bitexact(base, box2["grads"])
    check(exact, "E2 worker unaffected by main-thread hooks", "")

    # E3: bf16 autocast + hooks vs bf16 autocast baseline
    ac_dev = dev.type

    def run_ac(ctx):
        m = copy.deepcopy(master)
        with torch.autocast(ac_dev, torch.bfloat16):
            if ctx is None:
                out = m(x)
            else:
                with ctx:
                    out = m(x)
            loss = F.mse_loss(out.float(), y)
        loss.backward()
        return [p.grad.detach().clone() for p in m.parameters()]

    ac_base = run_ac(None)
    stats3 = defaultdict(int)
    ac_hook = run_ac(offload_hooks_sync(stats3))
    exact, worst = grads_bitexact(ac_base, ac_hook)
    bf16_saves = stats3["offloaded"]
    check(exact, "E3 bf16 autocast with hooks: bit-exact vs autocast baseline",
          f"offloaded {bf16_saves} tensors (bf16 payloads), "
          f"max diff {worst:.1e}")


# ══════════════════════════════════════════════════════════════════════════
# F. Measurement
# ══════════════════════════════════════════════════════════════════════════
def section_f(dev: torch.device):
    banner("F. Cost/benefit on GPU (12-chunk MLP, boundary+internal saves)")
    if dev.type != "cuda":
        note("F skipped: needs CUDA")
        return
    d, n, B = 2048, 12, 4096
    torch.manual_seed(6)
    chunks = nn.ModuleList(
        [nn.Sequential(nn.Linear(d, d), nn.GELU()) for _ in range(n)]
    ).to(dev)
    params = list(chunks.parameters())
    param_ptrs = {p.data_ptr() for p in params}

    def skip_params(t):
        return t.data_ptr() in param_ptrs  # weights stay resident

    d2h = torch.cuda.Stream(dev)
    pool = PinnedPool()
    warmup, iters = 2, 5
    rows = []

    def one_step(mode, ctx):
        x = torch.randn(B, d, device=dev).requires_grad_(True)
        h = x
        with (ctx if ctx is not None else nullcontext()):
            for m in chunks:
                if mode == "checkpoint":
                    h = torch_ckpt.checkpoint(m, h, use_reentrant=False)
                else:
                    h = m(h)
        h.pow(2).mean().backward()
        for p in params:
            p.grad = None
        torch.cuda.synchronize(dev)
        pool.reset()

    for mode in ("keep", "checkpoint"):
        variants = [("none", None), ("sync", "sync"), ("async", "async")]
        if mode == "keep":
            variants.insert(1, ("save_on_cpu", "soc"))
        for vname, vkind in variants:
            stats = defaultdict(int)

            def mk_ctx():
                if vkind is None:
                    return None
                if vkind == "soc":
                    return save_on_cpu(pin_memory=True)
                if vkind == "sync":
                    return offload_hooks_sync(stats, skip_fn=skip_params)
                return offload_hooks_stream(d2h, stats,
                                            skip_fn=skip_params, pool=pool)

            for _ in range(warmup):
                one_step(mode, mk_ctx())
            torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.perf_counter()
            for _ in range(iters):
                one_step(mode, mk_ctx())
            ms = (time.perf_counter() - t0) * 1e3 / iters
            peak = torch.cuda.max_memory_allocated(dev) / 2**20
            mb = stats["bytes"] / 2**20 / max(1, warmup + iters)
            rows.append((mode, vname, ms, peak,
                         stats["offloaded"] // max(1, warmup + iters), mb))

    print(f"\n  {'mode':<11} {'hooks':<12} {'ms/step':>9} "
          f"{'peak MiB':>9} {'packs':>6} {'MiB moved':>10}")
    for mode, vname, ms, peak, packs, mb in rows:
        print(f"  {mode:<11} {vname:<12} {ms:9.1f} {peak:9.0f} "
              f"{packs:6d} {mb:10.1f}")
    by = {(m, v): (ms, pk) for m, v, ms, pk, *_ in rows}
    k_none, k_async = by[("keep", "none")], by[("keep", "async")]
    note(f"keep: async offload peak {k_none[1]:.0f} -> {k_async[1]:.0f} MiB "
         f"({k_none[1] / max(k_async[1], 1):.1f}x), "
         f"{k_none[0]:.0f} -> {k_async[0]:.0f} ms/step")
    c_none, c_async = by[("checkpoint", "none")], by[("checkpoint", "async")]
    note(f"checkpoint: async offload (boundaries only) peak "
         f"{c_none[1]:.0f} -> {c_async[1]:.0f} MiB, "
         f"{c_none[0]:.0f} -> {c_async[0]:.0f} ms/step")
    note("sync variant allocates fresh pinned buffers per pack (worst case); "
         "async uses a pooled pinned staging + dedicated D2H stream")


# ══════════════════════════════════════════════════════════════════════════
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None,
                    help="e.g. cuda:1 (default: cuda:1 if >1 GPUs, else "
                         "cuda:0 if available, else cpu)")
    ap.add_argument("--skip-f", action="store_true",
                    help="skip the GPU measurement section")
    args = ap.parse_args()

    if args.device is not None:
        dev = torch.device(args.device)
    elif torch.cuda.is_available():
        dev = torch.device("cuda:1" if torch.cuda.device_count() > 1
                           else "cuda:0")
    else:
        dev = torch.device("cpu")
    if dev.type == "cuda":
        torch.cuda.set_device(dev)
    print(f"torch {torch.__version__}  device={dev}")

    section_a(dev)
    section_b(dev)
    section_c(dev)
    section_d(dev)
    section_e(dev)
    if not args.skip_f:
        section_f(dev)

    banner("SUMMARY")
    for msg in OBSERVATIONS:
        print(f"  · {msg}")
    if FAILURES:
        print(f"\n  {len(FAILURES)} FAILURE(S):")
        for f_ in FAILURES:
            print(f"    - {f_}")
        raise SystemExit(1)
    print("\n  ALL CHECKS PASSED — saved_tensors_hooks can grab, offload and "
          "restore\n  chunk activations bit-exactly, incl. under non-reentrant"
          " checkpoint and\n  the streaming engine.  GO for the feature.")


if __name__ == "__main__":
    main()
