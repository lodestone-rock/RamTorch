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
  8. Activation offload (offload_activations=True): BIT-exact vs the same
     config with it off — keep + checkpoint x act_slots 1/2, 3-step
     optimizer cycling, dropout mask preservation, tuple intermediates,
     window=1 + acc_slots=1 + act_slots=1 combined thrash. Also byte-exact
     traffic accounting on a known architecture, which doubles as the
     probe's "hooks never pack weights" counter (one leaked weight would
     inflate act_bytes_offloaded detectably). Recompute mode + activation
     offload must be rejected.
  9. Runtime tier reassignment (set_pinned / pin_chunks / unpin_chunks): a
     model that retiers is BIT-exact vs one built with the final tier from
     the start (before step 0 and mid-trajectory, all backward modes, both
     grad_accum paths, with activation offload, extremes and round trips);
     the hard reset is observable (pre-retier accumulation is discarded) and
     complete; optimizer state follows its params (fused + non-fused);
     rejections (bad indices, pending itinerary, closed model, NVMe chunks).

On CUDA, also probes what retiering costs in GPU memory (resident bytes vs
the analytic per-chunk cost, free-on-unpin, a 5-cycle leak check and step-peak
monotonicity) and reports peak GPU memory of a streamed step vs the
full-resident reference (the point of the whole exercise).

Run:  PYTHONPATH=. python examples/offload_streaming_check.py
"""

import copy
import os
import sys
import tempfile

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


def _train_run(device, chunks, *, keep, act, act_slots=2, window=2,
               acc_slots=None, steps=3, dim=32, lr=0.05):
    """3 SGD steps on an OffloadModel; returns (losses, grads, weights,
    infer output, stats) — everything an exact-parity comparison needs."""
    model = OffloadModel(copy.deepcopy(chunks), device=device, window=window,
                         keep_activations=keep, acc_slots=acc_slots,
                         offload_activations=act, act_slots=act_slots)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    torch.manual_seed(2)
    x, y = torch.randn(8, dim), torch.randn(8, dim)
    losses, grads = [], {}
    for s in range(steps):
        torch.manual_seed(20 + s)  # identical dropout draws across runs
        res = model.step(x, targets=y, loss_fn=nn.functional.mse_loss)
        losses.append(res.loss.detach().cpu().clone())
        model.flush_grads()
        if s == 0:
            grads = {nme: p.grad.detach().cpu().clone()
                     for nme, p in model.chunks.named_parameters()}
        opt.step()
        opt.zero_grad(set_to_none=False)
    weights = {nme: p.detach().cpu().clone()
               for nme, p in model.chunks.named_parameters()}
    with torch.no_grad():
        infer = model(x).detach().cpu().clone()
    stats = dict(model.stats)
    model.close()
    return losses, grads, weights, infer, stats


def _exact(name, a, b, errs):
    if not torch.equal(a, b):
        errs.append(f"{name}: NOT bit-exact (max err "
                    f"{(a - b).abs().max().item():.3e})")


def check_act_offload(device: str) -> list:
    """Activation offload: bit-exact parity vs the same config with it off."""
    errs: list = []
    n, dim = 6, 32

    def compare(tag, base, test):
        for s, (bl, tl) in enumerate(zip(base[0], test[0])):
            _exact(f"{tag} loss step{s}", bl, tl, errs)
        for nme in base[1]:
            _exact(f"{tag} grad {nme}", base[1][nme], test[1][nme], errs)
        for nme in base[2]:
            _exact(f"{tag} weight {nme}", base[2][nme], test[2][nme], errs)
        _exact(f"{tag} infer", base[3], test[3], errs)

    # keep/checkpoint x act_slots 1 (thrash) / 2, 3-step optimizer cycling
    for keep in (True, "checkpoint"):
        chunks = make_chunks(n, dim, seed=17)
        base = _train_run(device, chunks, keep=keep, act=False)
        for slots in (1, 2):
            nb = len(errs)
            test = _train_run(device, chunks, keep=keep, act=True,
                              act_slots=slots)
            compare(f"act[{_MODE_LABEL[keep]} slots={slots}]", base, test)
            # traffic accounting doubles as the no-weights-packed counter:
            # Linear+GELU saves exactly 2 activations per chunk in keep mode
            # (linear input + gelu input) and 1 in checkpoint mode (the
            # boundary input); lazy policy offloads (n - slots) packets per
            # step. A single packed weight would add dim*dim*4 bytes.
            saves = 2 if keep is True else 1
            steps = 3
            want_moves = steps * (n - slots)
            want_bytes = want_moves * saves * 8 * dim * 4
            if test[4]["act_offloads"] != want_moves:
                errs.append(f"act[{_MODE_LABEL[keep]} slots={slots}]: "
                            f"{test[4]['act_offloads']} offloads, expected "
                            f"{want_moves} (lazy policy)")
            if test[4]["act_bytes_offloaded"] != want_bytes:
                errs.append(f"act[{_MODE_LABEL[keep]} slots={slots}]: "
                            f"{test[4]['act_bytes_offloaded']} bytes moved, "
                            f"expected {want_bytes} — a non-activation "
                            f"tensor (weight?) got packed")
            status = "OK" if len(errs) == nb else "MISMATCH"
            print(f"[{device} act-offload {_MODE_LABEL[keep]} "
                  f"slots={slots}] {status}")

    # combined thrash: minimal weight window + acc slot + act slot
    for keep in (True, "checkpoint"):
        nb = len(errs)
        chunks = make_chunks(n, dim, seed=18)
        base = _train_run(device, chunks, keep=keep, act=False,
                          window=1, acc_slots=1)
        test = _train_run(device, chunks, keep=keep, act=True, act_slots=1,
                          window=1, acc_slots=1)
        compare(f"act-thrash[{_MODE_LABEL[keep]}]", base, test)
        status = "OK" if len(errs) == nb else "MISMATCH"
        print(f"[{device} act-thrash {_MODE_LABEL[keep]} W=1 acc=1 act=1] "
              f"{status}")

    # dropout masks must survive the offload/reload round trip
    for keep in (True, "checkpoint"):
        nb = len(errs)
        torch.manual_seed(19)
        dchunks = [
            nn.Sequential(nn.Linear(dim, dim), nn.Dropout(p=0.5), nn.GELU())
            for _ in range(n)
        ]
        base = _train_run(device, dchunks, keep=keep, act=False)
        test = _train_run(device, dchunks, keep=keep, act=True, act_slots=1)
        compare(f"act-dropout[{_MODE_LABEL[keep]}]", base, test)
        status = "OK" if len(errs) == nb else "MISMATCH"
        print(f"[{device} act-dropout {_MODE_LABEL[keep]}] {status}")

    # tuple intermediates (multi-save chunks, int passthrough)
    for keep in (True, "checkpoint"):
        nb = len(errs)
        torch.manual_seed(23)
        tchunks = ([_SplitHead(dim)] + [_TwoStream(dim) for _ in range(3)]
                   + [_MergeTail(dim)])
        base = _train_run(device, tchunks, keep=keep, act=False)
        test = _train_run(device, tchunks, keep=keep, act=True, act_slots=1)
        compare(f"act-tuple[{_MODE_LABEL[keep]}]", base, test)
        status = "OK" if len(errs) == nb else "MISMATCH"
        print(f"[{device} act-tuple {_MODE_LABEL[keep]}] {status}")

    # recompute mode has nothing to offload — must refuse loudly
    try:
        OffloadModel(make_chunks(2, dim, seed=25), device=device,
                     keep_activations=False, offload_activations=True)
        errs.append("recompute + offload_activations was not rejected")
    except ValueError:
        pass
    return errs


def _retier_run(device, chunks, *, pin, retier=None, retier_at=0, keep=True,
                grad_accum="stream", act=False, window=2, steps=3, dim=32,
                lr=0.05):
    """Like :func:`_train_run`, with an optional tier change mid-trajectory.

    The tier is a memory placement, not a math change, so a model that
    retiers into ``retier`` must follow the SAME trajectory as one built with
    that tier from the start — whether the move happens before step 0 or
    between later steps. SGD keeps that comparison bit-exact (elementwise
    fp32 ops are IEEE-identical on CPU and CUDA; fused AdamW would not be —
    see docs/offload.md on fused CPU vs fused CUDA rounding).
    """
    model = OffloadModel(copy.deepcopy(chunks), device=device, window=window,
                         pin_layers=sorted(pin), keep_activations=keep,
                         grad_accum=grad_accum, offload_activations=act)
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    torch.manual_seed(2)
    x, y = torch.randn(8, dim), torch.randn(8, dim)
    losses, grads = [], {}
    for s in range(steps):
        if retier is not None and s == retier_at:
            model.set_pinned(sorted(retier), optimizers=[opt])
        torch.manual_seed(20 + s)
        res = model.step(x, targets=y, loss_fn=nn.functional.mse_loss)
        losses.append(res.loss.detach().cpu().clone())
        model.flush_grads()
        if s == 0:
            grads = {nme: p.grad.detach().cpu().clone()
                     for nme, p in model.chunks.named_parameters()}
        opt.step()
        opt.zero_grad(set_to_none=False)
    weights = {nme: p.detach().cpu().clone()
               for nme, p in model.chunks.named_parameters()}
    with torch.no_grad():
        infer = model(x).detach().cpu().clone()
    model.close()
    return losses, grads, weights, infer


def _assert_wiped(model, moved, tag, errs):
    """A retier is a hard reset: nothing transient may survive it."""
    for st in model._state:
        for nme, acc in st.grad_acc.items():
            if acc.count_nonzero().item():
                errs.append(f"{tag}: chunk{st.idx}.{nme} accumulator not "
                            "zeroed by the retier")
        if st.acc_where != "empty" or st.acc_gpu is not None:
            errs.append(f"{tag}: chunk{st.idx} acc_where={st.acc_where} "
                        f"acc_gpu={st.acc_gpu is not None} after the retier")
        if st.acc_touched:
            errs.append(f"{tag}: chunk{st.idx} still marked as accumulated")
        if st.graph_tensors is not None:
            errs.append(f"{tag}: chunk{st.idx} kept graph_tensors")
    # buffers whose device/pinning depends on the tier are only stale for the
    # chunks that actually moved; unmoved chunks keep theirs on purpose
    for i in sorted(moved):
        st = model._state[i]
        if st.staging is not None or st.flush_bufs:
            errs.append(f"{tag}: moved chunk{i} kept staging/flush buffers")
        for nme, p in st.module.named_parameters():
            if p.grad is not None:
                errs.append(f"{tag}: chunk{i}.{nme}.grad survived the retier")
    if model._act_store:
        errs.append(f"{tag}: {len(model._act_store)} activation packet(s) "
                    "survived the retier")
    if model._resident:
        errs.append(f"{tag}: {len(model._resident)} resident weight copies "
                    "survived the retier")
    if model._future or model._fpos:
        errs.append(f"{tag}: itinerary not cleared "
                    f"({len(model._future)} entries, fpos={model._fpos})")


def _assert_tiers(model, tag, errs):
    """Masters must physically live where their tier says."""
    for st in model._state:
        for nme, p in st.module.named_parameters():
            if st.gpu_pinned:
                if p.device != model.device:
                    errs.append(f"{tag}: pinned chunk{st.idx}.{nme} on "
                                f"{p.device}, expected {model.device}")
            else:
                if p.device.type != "cpu":
                    errs.append(f"{tag}: streamed chunk{st.idx}.{nme} on "
                                f"{p.device}, expected cpu")
                elif model._cuda and not p.data.is_pinned():
                    errs.append(f"{tag}: streamed chunk{st.idx}.{nme} master "
                                "is not in pinned memory")


def check_retier(device: str) -> list:
    """Runtime tier reassignment: set_pinned / pin_chunks / unpin_chunks."""
    errs: list = []
    n, dim = 6, 32
    P1, P2 = {0, 3}, {1, 4}
    mse = nn.functional.mse_loss

    def compare(tag, base, test):
        for s, (bl, tl) in enumerate(zip(base[0], test[0])):
            _exact(f"{tag} loss step{s}", bl, tl, errs)
        for nme in base[1]:
            _exact(f"{tag} grad {nme}", base[1][nme], test[1][nme], errs)
        for nme in base[2]:
            _exact(f"{tag} weight {nme}", base[2][nme], test[2][nme], errs)
        _exact(f"{tag} infer", base[3], test[3], errs)

    # 1. trajectory parity vs a model built directly at the final tier —
    #    retiering before the first step and between later steps
    for keep in (False, True, "checkpoint"):
        for ga in ("stream", "cpu"):
            nb = len(errs)
            chunks = make_chunks(n, dim, seed=31)
            base = _retier_run(device, chunks, pin=P2, keep=keep,
                               grad_accum=ga)
            for at in (0, 1):
                test = _retier_run(device, chunks, pin=P1, retier=P2,
                                   retier_at=at, keep=keep, grad_accum=ga)
                compare(f"retier[{_MODE_LABEL[keep]} ga={ga} at={at}]",
                        base, test)
            status = "OK" if len(errs) == nb else "MISMATCH"
            print(f"[{device} retier {_MODE_LABEL[keep]} ga={ga}] {status}")

    # 2. extremes (offload everything / pin everything) + round trip
    nb = len(errs)
    chunks = make_chunks(n, dim, seed=32)
    allp = set(range(n))
    for target in (set(), allp):
        base = _retier_run(device, chunks, pin=target)
        test = _retier_run(device, chunks, pin=P1, retier=target)
        compare(f"retier[-> {'all' if target else 'none'} pinned]", base, test)
    base = _retier_run(device, chunks, pin=P1)
    model = OffloadModel(copy.deepcopy(chunks), device=device, window=2,
                         pin_layers=sorted(P1), keep_activations=True)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    model.set_pinned(sorted(P2), optimizers=[opt])
    model.set_pinned(sorted(P1), optimizers=[opt])
    if set(model.pinned_layers) != P1:
        errs.append(f"round trip landed on {sorted(model.pinned_layers)}, "
                    f"expected {sorted(P1)}")
    if set(model.streamed_layers) != allp - P1:
        errs.append("streamed_layers is not the complement of pinned_layers")
    _assert_tiers(model, "retier[round trip]", errs)
    torch.manual_seed(2)
    x, y = torch.randn(8, dim), torch.randn(8, dim)
    losses = []
    for s in range(3):
        torch.manual_seed(20 + s)
        losses.append(model.step(x, targets=y, loss_fn=mse)
                      .loss.detach().cpu().clone())
        model.flush_grads()
        opt.step()
        opt.zero_grad(set_to_none=False)
    for s, (bl, tl) in enumerate(zip(base[0], losses)):
        _exact(f"retier[round trip] loss step{s}", bl, tl, errs)
    for nme, p in model.chunks.named_parameters():
        _exact(f"retier[round trip] weight {nme}", base[2][nme],
               p.detach().cpu(), errs)
    model.close()
    print(f"[{device} retier extremes+round-trip] "
          f"{'OK' if len(errs) == nb else 'MISMATCH'}")

    # 3. with activation offload on (packets must be drained, not stranded)
    nb = len(errs)
    chunks = make_chunks(n, dim, seed=34)
    for keep in (True, "checkpoint"):
        base = _retier_run(device, chunks, pin=P2, keep=keep, act=True)
        test = _retier_run(device, chunks, pin=P1, retier=P2, retier_at=1,
                           keep=keep, act=True)
        compare(f"retier[act {_MODE_LABEL[keep]}]", base, test)
    print(f"[{device} retier act-offload] "
          f"{'OK' if len(errs) == nb else 'MISMATCH'}")

    # 4. the wipe is observable: pre-retier accumulation is discarded, so one
    #    step after the retier == exactly one step's grads
    nb = len(errs)
    for ga in ("stream", "cpu"):
        chunks = make_chunks(n, dim, seed=35)
        a = OffloadModel(copy.deepcopy(chunks), device=device, window=2,
                         pin_layers=sorted(P1), keep_activations=True,
                         grad_accum=ga, offload_activations=True)
        torch.manual_seed(2)
        x, y = torch.randn(8, dim), torch.randn(8, dim)
        torch.manual_seed(21)
        a.step(x, targets=y, loss_fn=mse)      # accumulation about to be lost
        info = a.set_pinned(sorted(P2))
        if not info["grads_discarded"]:
            errs.append(f"retier[wipe ga={ga}]: grads_discarded was False "
                        "after retiering mid-accumulation")
        if info["pinned"] != sorted(P2) or info["unpinned"] != sorted(P1):
            errs.append(f"retier[wipe ga={ga}]: summary {info} does not "
                        "describe the move")
        _assert_wiped(a, P1 | P2, f"retier[wipe ga={ga}]", errs)
        _assert_tiers(a, f"retier[wipe ga={ga}]", errs)
        torch.manual_seed(21)
        a.step(x, targets=y, loss_fn=mse)
        a.flush_grads()
        got = {nme: p.grad.detach().cpu().clone()
               for nme, p in a.chunks.named_parameters()}
        a.close()

        b = OffloadModel(copy.deepcopy(chunks), device=device, window=2,
                         pin_layers=sorted(P2), keep_activations=True,
                         grad_accum=ga, offload_activations=True)
        torch.manual_seed(21)
        b.step(x, targets=y, loss_fn=mse)
        b.flush_grads()
        for nme, p in b.chunks.named_parameters():
            _exact(f"retier[wipe ga={ga}] grad {nme}", p.grad.detach().cpu(),
                   got[nme], errs)
        b.close()
    print(f"[{device} retier wipe] "
          f"{'OK' if len(errs) == nb else 'MISMATCH'}")

    # 5. optimizer continuity: state follows the params across the move
    nb = len(errs)
    chunks = make_chunks(n, dim, seed=36)
    for fused in ((False, True) if device.startswith("cuda") else (False,)):
        model = OffloadModel(copy.deepcopy(chunks), device=device, window=2,
                             pin_layers=sorted(P1), keep_activations=True)
        before = [id(p) for p in model.parameters()]
        keys_before = set(model.state_dict())
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=fused)
        torch.manual_seed(2)
        x, y = torch.randn(8, dim), torch.randn(8, dim)
        for _ in range(2):
            model.step(x, targets=y, loss_fn=mse)
            model.flush_grads()
            opt.step()
            opt.zero_grad(set_to_none=False)
        model.set_pinned(sorted(P2), optimizers=opt)   # bare optimizer too
        if [id(p) for p in model.parameters()] != before:
            errs.append(f"retier[opt fused={fused}]: parameter objects were "
                        "replaced (param groups would go stale)")
        if set(model.state_dict()) != keys_before:
            errs.append(f"retier[opt fused={fused}]: state_dict keys changed")
        for p, st in opt.state.items():
            for key, val in st.items():
                if not isinstance(val, torch.Tensor):
                    continue
                want = p.device
                if key == "step" and not fused:
                    want = torch.device("cpu")
                if val.device != want:
                    errs.append(f"retier[opt fused={fused}]: state['{key}'] "
                                f"on {val.device}, param on {p.device}")
        for _ in range(2):      # must keep training without a device error
            model.step(x, targets=y, loss_fn=mse)
            model.flush_grads()
            opt.step()
            opt.zero_grad(set_to_none=False)
        model.close()
    print(f"[{device} retier optimizer-state] "
          f"{'OK' if len(errs) == nb else 'MISMATCH'}")

    # 6. rejections
    nb = len(errs)
    model = OffloadModel(make_chunks(n, dim, seed=37), device=device,
                         window=2, pin=2, keep_activations=True)
    for bad in ([n], -1, [0, n + 1]):
        try:
            model.pin_chunks(bad)
            errs.append(f"out-of-range chunk index {bad!r} was not rejected")
        except ValueError:
            pass
    for bad in (True, [True]):
        try:
            model.set_pinned(bad)
            errs.append(f"bool chunk spec {bad!r} was not rejected")
        except TypeError:
            pass
    try:
        model.set_pinned(-1)
        errs.append("negative pin count was not rejected")
    except ValueError:
        pass
    # mid-step guard: an announced-but-unconsumed itinerary must refuse
    model._announce([0, 1], kinds=["F", "F"])
    try:
        model.set_pinned(0)
        errs.append("retier with a pending itinerary was not rejected")
    except RuntimeError:
        pass
    model.set_pinned(0, force=True)   # the documented recovery path
    if model.pinned_layers:
        errs.append("force=True retier did not apply")
    model.step(torch.randn(4, dim), targets=torch.randn(4, dim), loss_fn=mse)
    model.close()
    try:
        model.pin_chunks([0])
        errs.append("retier on a closed model was not rejected")
    except RuntimeError:
        pass
    # nvme chunks share one scratch-file layout -> not retierable
    with tempfile.TemporaryDirectory() as tmp:
        nv = OffloadModel(make_chunks(4, dim, seed=38), device=device,
                          window=2, nvme=2,
                          nvme_path=os.path.join(tmp, "w.bin"))
        on_disk = sorted(nv.nvme_layers)
        try:
            nv.pin_chunks(on_disk[:1])
            errs.append("retiering an NVMe chunk was not rejected")
        except ValueError:
            pass
        free = [i for i in range(4) if i not in nv.nvme_layers]
        nv.pin_chunks(free[:1])       # non-NVMe chunks stay retierable
        if free[0] not in nv.pinned_layers:
            errs.append("pinning a non-NVMe chunk of an NVMe model failed")
        nv.close()
    print(f"[{device} retier rejections] "
          f"{'OK' if len(errs) == nb else 'MISMATCH'}")
    return errs


def probe_retier_memory(device: str):
    """Measure what retiering does to GPU memory (CUDA only).

    Pinning k chunks must cost k chunks of weights + accumulators and nothing
    else, unpinning must give it all back, and repeated cycles must not leak
    (a stranded accumulator / graph_tensors / resident copy would show up).
    """
    errs: list = []
    n, dim, batch = 8, 512, 32
    chunks = make_chunks(n, dim, seed=41)
    x, y = torch.randn(batch, dim), torch.randn(batch, dim)
    mse = nn.functional.mse_loss

    model = OffloadModel(copy.deepcopy(chunks), device=device, window=2,
                         pin=0, keep_activations=True)
    # weights + buffers + the GPU grad accumulator of one chunk
    per_chunk = model._state[0].nbytes() + sum(
        p.numel() * p.element_size()
        for p in model._state[0].module.parameters()
    )
    slack = per_chunk // 4

    def allocated():
        torch.cuda.synchronize(device)
        return torch.cuda.memory_allocated(device)

    def step_peak():
        torch.cuda.reset_peak_memory_stats(device)
        model.step(x, targets=y, loss_fn=mse)
        model.flush_grads()
        torch.cuda.synchronize(device)
        return torch.cuda.max_memory_allocated(device)

    levels = (0, n // 2, n)
    # phase 1 — resident cost, measured BEFORE any step so the only GPU
    # tenants are the pinned masters and their accumulators (a flush would
    # add a persistent .grad buffer per pinned chunk, measured in phase 2)
    resident = []
    base_alloc = allocated()
    for pin in levels:
        model.set_pinned(pin)
        want = len(model.pinned_layers) * per_chunk
        got = allocated() - base_alloc
        resident.append(got)
        if abs(got - want) > slack:
            errs.append(f"pin={pin}: {got / 2**20:.1f} MiB resident after the "
                        f"retier, expected ~{want / 2**20:.1f} MiB")
    # unpinning gives it all back
    model.set_pinned(0)
    back = allocated() - base_alloc
    if abs(back) > slack:
        errs.append(f"unpinning left {back / 2**20:.1f} MiB behind")

    # phase 2 — step peaks must grow with the pin count and come back down
    peaks = []
    for pin in levels:
        model.set_pinned(pin)
        peaks.append(step_peak())
    model.set_pinned(0)
    if not peaks[0] < peaks[1] < peaks[2]:
        errs.append(f"step peak is not monotonic in the pin count: {peaks}")
    peak_back = step_peak()
    if abs(peak_back - peaks[0]) > per_chunk:
        errs.append(f"step peak after unpinning back is {peak_back / 2**20:.1f}"
                    f" MiB vs {peaks[0] / 2**20:.1f} MiB originally")
    # leak check: five full cycles must return to where they started. The
    # baseline is taken HERE, after the warmup steps above, because cuBLAS
    # allocates a per-stream workspace on first use (~8 MiB each) through the
    # caching allocator, and memory_allocated counts it.
    cycle_base = allocated()
    for _ in range(5):
        model.set_pinned(n // 2)
        model.step(x, targets=y, loss_fn=mse)
        model.flush_grads()
        model.set_pinned(0)
    leaked = allocated() - cycle_base
    if abs(leaked) > slack:
        errs.append(f"5 pin/unpin cycles leaked {leaked / 2**20:.2f} MiB")
    model.close()

    print(f"[retier memory n={n} dim={dim}] chunk={per_chunk / 2**20:.2f} MiB "
          f"(weights+acc), baseline {base_alloc / 2**20:.1f} MiB")
    for pin, res, peak in zip(levels, resident, peaks):
        print(f"    pin={pin}: +{res / 2**20:6.2f} MiB resident, "
              f"step peak {peak / 2**20:6.1f} MiB")
    print(f"    after unpinning back: +{back / 2**20:.2f} MiB resident, "
          f"step peak {peak_back / 2**20:.1f} MiB, "
          f"5-cycle leak {leaked / 2**20:.2f} MiB")
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
                + check_grad_bypass(device) + check_act_offload(device)
                + check_retier(device))
        for e in errs:
            print(f"    {e}")
        ok &= not errs

    if torch.cuda.is_available():
        errs = probe_retier_memory("cuda:0")
        for e in errs:
            print(f"    {e}")
        ok &= not errs
        memory_report("cuda:0")

    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
