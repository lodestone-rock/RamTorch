"""
offload_streaming_check.py
--------------------------
Validate the OffloadModel streaming executor (sliding window + pinned layers)
against a plain full-resident reference model.

Checks (on CPU always, and on CUDA when available):
  1. Inference: streamed forward == reference forward, twice (ring reuse).
  2. Training: step() loss and every param grad == reference backward, across
     window/pin configurations including W=1, W=n, all-pinned.
  3. Gradient accumulation: two step() calls + flush_grads(1/2) == mean grads.
  4. Load accounting: with window >= n, step 2 issues zero new loads.

On CUDA, also reports peak GPU memory of a streamed step vs the full-resident
reference (the point of the whole exercise).

Run:  PYTHONPATH=. python examples/offload_streaming_check.py
"""

import copy
import sys

import torch
import torch.nn as nn

from ramtorch.offload import OffloadModel

ATOL, RTOL = 1e-5, 1e-4


def make_chunks(n_chunks: int, dim: int, seed: int = 0):
    torch.manual_seed(seed)
    return [
        nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        for _ in range(n_chunks)
    ]


def max_err(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.detach().cpu() - b.detach().cpu()).abs().max().item()


def check_close(name: str, a, b, errs: list):
    if not torch.allclose(a.detach().cpu(), b.detach().cpu(),
                          atol=ATOL, rtol=RTOL):
        errs.append(f"{name}: max err {max_err(a, b):.3e}")


def run_config(device: str, n: int, dim: int, window: int, pin: int,
               keep=False) -> list:
    """Compare one OffloadModel config against the full-resident reference."""
    errs: list = []
    chunks = make_chunks(n, dim)
    ref = nn.Sequential(*copy.deepcopy(chunks)).to(device)
    model = OffloadModel(chunks, device=device, window=window, pin=pin,
                         keep_activations=keep)

    torch.manual_seed(1)
    x = torch.randn(8, dim)
    y = torch.randn(8, dim)
    loss_fn = nn.functional.mse_loss

    # 1. inference, twice (second pass exercises the ring / re-streaming)
    with torch.no_grad():
        want = ref(x.to(device))
    for it in range(2):
        got = model(x)
        check_close(f"forward pass {it}", got, want, errs)

    # 2. one training step vs reference backward
    res = model.step(x, targets=y, loss_fn=loss_fn)
    ref.zero_grad()
    ref_loss = loss_fn(ref(x.to(device)), y.to(device))
    ref_loss.backward()
    check_close("loss", res.loss, ref_loss, errs)
    check_close("output", res.output, want, errs)

    model.flush_grads()
    for i in range(n):
        got_p = dict(model.chunks[i].named_parameters())
        want_p = dict(ref[i].named_parameters())
        for nme, p in got_p.items():
            if p.grad is None:
                errs.append(f"chunk{i}.{nme}: no grad")
                continue
            check_close(f"chunk{i}.{nme}.grad", p.grad, want_p[nme].grad, errs)

    # 3. gradient accumulation: 2 steps + flush(1/2) == same grads (same batch)
    model.step(x, targets=y, loss_fn=loss_fn)
    model.step(x, targets=y, loss_fn=loss_fn)
    model.flush_grads(scale=0.5)
    for i in range(n):
        got_p = dict(model.chunks[i].named_parameters())
        want_p = dict(ref[i].named_parameters())
        for nme, p in got_p.items():
            check_close(f"accum chunk{i}.{nme}.grad", p.grad,
                        want_p[nme].grad, errs)

    # keep mode: streamed weight storages must actually be freed (only the
    # window + pinned chunks may hold memory once the step is done)
    if keep:
        alive = sum(
            1 for st in model._state
            if st.graph_tensors is not None
            and any(t.untyped_storage().size() > 0
                    for t in st.graph_tensors.values())
        )
        if alive > window:
            errs.append(f"keep mode: {alive} streamed chunks still hold "
                        f"weight storage, window is {window}")

    model.close()
    return errs


def check_dropout_keep(device: str) -> list:
    """keep/checkpoint modes must not resample stochastic layers between fwd
    and bwd: the reported loss must exactly equal loss_fn(reported output)."""
    errs: list = []
    for mode in (True, "checkpoint"):
        torch.manual_seed(3)
        n, dim = 6, 32
        chunks = [
            nn.Sequential(nn.Linear(dim, dim), nn.Dropout(p=0.5), nn.GELU())
            for _ in range(n)
        ]
        model = OffloadModel(chunks, device=device, window=2,
                             keep_activations=mode)
        model.train()
        x, y = torch.randn(8, dim), torch.randn(8, dim)
        res = model.step(x, targets=y, loss_fn=nn.functional.mse_loss)
        recomputed = nn.functional.mse_loss(res.output, y.to(model.device))
        if max_err(res.loss, recomputed) > 1e-7:
            errs.append(f"{mode}-mode dropout: loss {res.loss.item():.6f} != "
                        f"loss_fn(output) {recomputed.item():.6f} — "
                        f"mask resampled")
        model.close()
    return errs


def check_load_accounting(device: str) -> list:
    errs: list = []
    n, dim = 6, 32
    model = OffloadModel(make_chunks(n, dim), device=device, window=n, pin=0)
    x = torch.randn(4, dim)
    model.step(x)
    loads_1 = model.stats["loads"]
    model.step(x)
    loads_2 = model.stats["loads"]
    if loads_1 != n or loads_2 != n:
        errs.append(f"W>=n loads: step1={loads_1} step2={loads_2}, expected "
                    f"{n} then no more (window holds everything)")
    model.close()
    return errs


def memory_report(device: str):
    """Peak GPU memory: streamed (W=2, pin=0) vs full-resident reference."""
    n, dim, batch = 24, 2048, 32
    chunks = make_chunks(n, dim, seed=7)
    x = torch.randn(batch, dim)
    y = torch.randn(batch, dim)
    loss_fn = nn.functional.mse_loss

    torch.cuda.reset_peak_memory_stats(device)
    ref = nn.Sequential(*copy.deepcopy(chunks)).to(device)
    loss = loss_fn(ref(x.to(device)), y.to(device))
    loss.backward()
    torch.cuda.synchronize(device)
    ref_peak = torch.cuda.max_memory_allocated(device)
    del ref, loss
    torch.cuda.empty_cache()

    peaks = {}
    for keep in (False, True):
        torch.cuda.reset_peak_memory_stats(device)
        model = OffloadModel(copy.deepcopy(chunks), device=device, window=2,
                             pin=0, keep_activations=keep)
        model.step(x, targets=y, loss_fn=loss_fn)
        torch.cuda.synchronize(device)
        peaks[keep] = (torch.cuda.max_memory_allocated(device),
                       dict(model.stats))
        model.close()
        torch.cuda.empty_cache()

    for keep, (off_peak, stats) in peaks.items():
        mode = "keep      " if keep else "recompute "
        print(f"[memory n={n} dim={dim}] {mode} W=2 peak "
              f"{off_peak / 2**20:.0f} MiB vs full-resident "
              f"{ref_peak / 2**20:.0f} MiB "
              f"({ref_peak / max(off_peak, 1):.1f}x smaller), "
              f"loads={stats['loads']}, "
              f"acquire_wait={stats['acquire_wait_s'] * 1e3:.1f} ms")


def main() -> int:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda:0")

    n, dim = 8, 64
    configs = [
        (2, 0),          # plain sliding window
        (2, 4),          # window + evenly pinned (the recommended design)
        (1, 0),          # minimal window
        (n, 0),          # everything windowed
        (1, n),          # everything pinned
    ]
    ok = True
    for device in devices:
        for keep in (False, True, "checkpoint"):
            for window, pin in configs:
                errs = run_config(device, n, dim, window, pin, keep=keep)
                mode = {False: "recompute", True: "keep",
                        "checkpoint": "checkpoint"}[keep]
                status = "OK" if not errs else "MISMATCH"
                print(f"[{device} W={window} pin={pin} {mode}] {status}")
                for e in errs[:5]:
                    print(f"    {e}")
                ok &= not errs
        errs = check_load_accounting(device) + check_dropout_keep(device)
        for e in errs:
            print(f"    {e}")
        ok &= not errs

    if torch.cuda.is_available():
        memory_report("cuda:0")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
