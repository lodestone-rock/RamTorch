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
  5. Tuple intermediates: chunks exchanging (a, b, int-passthrough) tuples —
     forward + loss + grad parity in all three backward modes.
  6. Mixed precision: step() wrapped in torch.autocast(bf16) vs a plain model
     trained with the same recipe. Needs NO engine support — offload compute
     runs on the calling thread (autocast is thread-local), unlike pipeline
     stage workers.
  7. Grad bypass (grad_outputs=): callable + tensor forms vs a reference
     loss backward; .loss raises; loss_fn+grad_outputs raises ValueError.

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


class _SplitHead(nn.Module):
    """x -> (a, b, k): two float streams + an int passthrough (no grad)."""

    def __init__(self, dim):
        super().__init__()
        self.la = nn.Linear(dim, dim)
        self.lb = nn.Linear(dim, dim)

    def forward(self, x):
        k = torch.arange(x.size(0), device=x.device)
        return self.la(x), self.lb(x), k


class _TwoStream(nn.Module):
    """(a, b, k) -> (a', b', k): streams interact, k threads through."""

    def __init__(self, dim):
        super().__init__()
        self.la = nn.Linear(dim, dim)
        self.lb = nn.Linear(dim, dim)

    def forward(self, a, b, k):
        return (nn.functional.gelu(self.la(a)) + 0.1 * b,
                nn.functional.gelu(self.lb(b)),
                k)


class _MergeTail(nn.Module):
    """(a, b, k) -> y: merges the streams, consumes k grad-free."""

    def __init__(self, dim):
        super().__init__()
        self.lo = nn.Linear(dim, dim)

    def forward(self, a, b, k):
        return self.lo(a + b) + 0.01 * k.float().unsqueeze(1)


class _RefMulti(nn.Module):
    """Full-resident reference that relays tuples exactly like OffloadModel."""

    def __init__(self, chunks):
        super().__init__()
        self.chunks = nn.ModuleList(chunks)

    def forward(self, x):
        out = x
        for c in self.chunks:
            out = c(*out) if isinstance(out, tuple) else c(out)
        return out


def check_multi_arg(device: str) -> list:
    """Tuple intermediates: parity vs reference in all three backward modes."""
    errs: list = []
    torch.manual_seed(5)
    dim, n_mid = 32, 4
    chunks = ([_SplitHead(dim)] + [_TwoStream(dim) for _ in range(n_mid)]
              + [_MergeTail(dim)])
    ref = _RefMulti(copy.deepcopy(chunks)).to(device)
    torch.manual_seed(1)
    x = torch.randn(8, dim)
    y = torch.randn(8, dim)
    loss_fn = nn.functional.mse_loss

    with torch.no_grad():
        want = ref(x.to(device))
    ref.zero_grad()
    ref_loss = loss_fn(ref(x.to(device)), y.to(device))
    ref_loss.backward()

    for mode in (False, True, "checkpoint"):
        n_before = len(errs)
        model = OffloadModel(copy.deepcopy(chunks), device=device, window=2,
                             keep_activations=mode)
        got = model(x)
        check_close(f"multiarg[{mode}] forward", got, want, errs)
        res = model.step(x, targets=y, loss_fn=loss_fn)
        check_close(f"multiarg[{mode}] loss", res.loss, ref_loss, errs)
        check_close(f"multiarg[{mode}] output", res.output, want, errs)
        model.flush_grads()
        for (nme, p), (rnme, rp) in zip(model.chunks.named_parameters(),
                                        ref.chunks.named_parameters()):
            if p.grad is None:
                errs.append(f"multiarg[{mode}] {nme}: no grad")
                continue
            check_close(f"multiarg[{mode}] {nme}.grad", p.grad, rp.grad, errs)
        model.close()
        status = "OK" if len(errs) == n_before else "MISMATCH"
        label = {False: "recompute", True: "keep", "checkpoint": "checkpoint"}
        print(f"[{device} multi-arg {label[mode]}] {status}")
    return errs


_MODE_LABEL = {False: "recompute", True: "keep", "checkpoint": "checkpoint"}


def check_autocast(device: str) -> list:
    """step() under ambient torch.autocast(bf16) == plain model, same recipe.

    The whole step (forward, backward recompute, loss) runs on the calling
    thread, so the user's autocast context simply applies — verifying the
    engine needs no autocast plumbing of its own.
    """
    errs: list = []
    ac_dev = "cuda" if device.startswith("cuda") else "cpu"
    n, dim, steps, lr = 6, 64, 3, 1e-2
    chunks = make_chunks(n, dim, seed=11)
    torch.manual_seed(2)
    x, y = torch.randn(8, dim), torch.randn(8, dim)
    loss_fn = nn.functional.mse_loss

    ref = nn.Sequential(*copy.deepcopy(chunks)).to(device)
    ropt = torch.optim.SGD(ref.parameters(), lr=lr)
    ref_losses = []
    for _ in range(steps):
        with torch.autocast(ac_dev, dtype=torch.bfloat16):
            loss = loss_fn(ref(x.to(device)), y.to(device))
        ropt.zero_grad(set_to_none=True)
        loss.backward()
        ropt.step()
        ref_losses.append(loss.detach())

    for mode in (False, True, "checkpoint"):
        n_before = len(errs)
        model = OffloadModel(copy.deepcopy(chunks), device=device, window=2,
                             keep_activations=mode)
        opt = torch.optim.SGD(model.parameters(), lr=lr)
        for s in range(steps):
            with torch.autocast(ac_dev, dtype=torch.bfloat16):
                res = model.step(x, targets=y, loss_fn=loss_fn)
            model.flush_grads()
            opt.step()
            check_close(f"autocast[{mode}] loss step{s}", res.loss,
                        ref_losses[s], errs)
        for (nme, p), (_, rp) in zip(model.chunks.named_parameters(),
                                     ref.named_parameters()):
            check_close(f"autocast[{mode}] weight {nme}", p, rp, errs)
        model.close()
        status = "OK" if len(errs) == n_before else "MISMATCH"
        print(f"[{device} autocast-bf16 {_MODE_LABEL[mode]}] {status}")
    return errs


def check_grad_bypass(device: str) -> list:
    """grad_outputs=: seeded backward == reference loss backward."""
    errs: list = []
    n, dim = 6, 32
    chunks = make_chunks(n, dim, seed=13)
    torch.manual_seed(4)
    x, y = torch.randn(8, dim), torch.randn(8, dim)
    loss_fn = nn.functional.mse_loss

    ref = nn.Sequential(*copy.deepcopy(chunks)).to(device)
    ref.zero_grad()
    ref_loss = loss_fn(ref(x.to(device)), y.to(device))
    ref_loss.backward()

    def compare_grads(mode, tag):
        model.flush_grads()
        for (nme, p), (_, rp) in zip(model.chunks.named_parameters(),
                                     ref.named_parameters()):
            if p.grad is None:
                errs.append(f"bypass[{mode}] {tag} {nme}: no grad")
                continue
            check_close(f"bypass[{mode}] {tag} {nme}.grad", p.grad,
                        rp.grad, errs)

    for mode in (False, True, "checkpoint"):
        n_before = len(errs)
        model = OffloadModel(copy.deepcopy(chunks), device=device, window=2,
                             keep_activations=mode)

        # callable form: dL/dOut of mean-MSE, resolved with the live output
        res = model.step(
            x, targets=y,
            grad_outputs=lambda out, tgt: 2.0 * (out - tgt) / out.numel(),
        )
        try:
            _ = res.loss
            errs.append(f"bypass[{mode}]: .loss did not raise")
        except RuntimeError:
            pass
        compare_grads(mode, "callable")

        # tensor form: dL/dOut precomputed from a no_grad forward
        model.zero_grad_acc()
        with torch.no_grad():
            out0 = model(x)
        go = 2.0 * (out0 - y.to(model.device)) / out0.numel()
        model.step(x, grad_outputs=go)
        compare_grads(mode, "tensor")

        # loss_fn + grad_outputs must refuse loudly
        try:
            model.step(x, loss_fn=loss_fn, grad_outputs=go)
            errs.append(f"bypass[{mode}]: loss_fn+grad_outputs did not raise")
        except ValueError:
            pass

        model.close()
        status = "OK" if len(errs) == n_before else "MISMATCH"
        print(f"[{device} grad-bypass {_MODE_LABEL[mode]}] {status}")
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
        errs = (check_load_accounting(device) + check_dropout_keep(device)
                + check_multi_arg(device) + check_autocast(device)
                + check_grad_bypass(device))
        for e in errs:
            print(f"    {e}")
        ok &= not errs

    if torch.cuda.is_available():
        memory_report("cuda:0")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
