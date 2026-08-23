"""
pipeline_offload_check.py
-------------------------
Validate weight offloading INSIDE pipeline parallelism (OffloadStage) against
two independent references, with bit-level expectations:

  A. a plain full-resident Pipeline built from the SAME stage modules
     (identical op order and dtypes -> should match bit-for-bit), and
  B. a sequential single-model microbatch grad-accumulation loop
     (the ground-truth semantics of a pipeline step).

Checks (on CPU always, and on two CUDA devices when available):
  1. Core matrix: schedules {staggered_1b1f, gpipe} x backward modes
     {keep, checkpoint} x (window, pin) in {(1,0), (2,0), (L,0), (2,2)} —
     loss + streamed infer() output + every parameter grad.
  2. Multi-step training: 3 SGD(momentum) steps -> final weights identical
     (exercises flush_grads -> optimizer -> residency invalidation cycling).
  3. Mixed pipeline: offloaded stage 0 + plain resident stage 1.
  4. Tuple intermediates: chunks exchange (a, b, int-passthrough) tuples both
     ACROSS chunk boundaries inside a stage and ACROSS the stage boundary.
  5. Autocast bf16: Pipeline(autocast=torch.bfloat16), offloaded vs plain.
  6. Grad bypass: step(grad_outputs=...) on the last (offloaded) stage vs the
     equivalent loss backward; .loss must raise.
  7. Engine recompute mode (offload_keep_activations=False) must be rejected,
     and fake_compute with chunked stages must be rejected.
  8. Flat chunk_modules= convenience: even + weighted (chunks_per_stage=)
     splits match the nested stage_modules spec bit-for-bit; malformed specs
     (sum/length mismatches, nesting, too few chunks) are rejected.
  9. Activation offload (offload_activations=True): bit-exact vs the plain
     pipeline across {staggered_1b1f, gpipe, 1f1b} x {keep, checkpoint} x
     act_slots {1, 2}, plus W=1 + acc_slots=1 + act_slots=1 combined thrash
     and the mixed offloaded+plain topology.
 10. Runtime tier reassignment (Pipeline.set_offload_pinned): retiering a
     stage mid-training is BIT-exact vs a pipeline built with the final pin
     from the start ({staggered_1b1f, gpipe} x {keep, checkpoint}, before
     step 0 and between steps, with activation offload, pin-all/pin-none);
     the hard reset is complete per stage; spec forms (dict / per-stage list
     / broadcast set) and a pre-built optimizer keep working; per-stage GPU
     memory grows only on the retiered stage's device and 5 cycles do not
     leak; mixed and flat-chunk pipelines behave, resident ones refuse.
 11. infer() allocates no gradient state on either stage kind: a plain stage's
     accumulators (a full copy of the shard on its GPU) and an offloaded
     stage's (one buffer per chunk) appear on the first BACKWARD, so
     construction and any number of forwards allocate none.

Run:  PYTHONPATH=. python examples/pipeline_offload_check.py
"""

import copy
import itertools
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from ramtorch import Pipeline

ATOL, RTOL = 1e-5, 1e-4
MODE_LABEL = {True: "keep", "checkpoint": "checkpoint"}


def make_chunks(n_chunks: int, dim: int, seed: int):
    torch.manual_seed(seed)
    return [nn.Sequential(nn.Linear(dim, dim), nn.GELU())
            for _ in range(n_chunks)]


def check_close(name, a, b, errs, exact=False):
    a = a.detach().cpu()
    b = b.detach().cpu()
    tol = dict(atol=0.0, rtol=0.0) if exact else dict(atol=ATOL, rtol=RTOL)
    if not torch.allclose(a, b, **tol):
        errs.append(f"{name}: max err {(a - b).abs().max().item():.3e}")


def named_params(pipe_stage_modules):
    """Flat (name, param) list across a stage_modules spec (lists or modules)."""
    out = []
    for s, entry in enumerate(pipe_stage_modules):
        mods = entry if isinstance(entry, (list, tuple)) else [entry]
        for c, m in enumerate(mods):
            out.extend(((f"s{s}c{c}.{n}", p) for n, p in m.named_parameters()))
    return out


def seq_reference_grads(chunks_flat, x, y, m, loss_fn):
    """Ground truth: sequential microbatch grad accumulation on one model."""
    model = nn.Sequential(*chunks_flat)
    model.zero_grad()
    losses = []
    for xb, yb in zip(x.chunk(m), y.chunk(m)):
        loss = loss_fn(model(xb), yb)
        loss.backward()
        losses.append(loss.detach())
    grads = {n: p.grad / m for n, p in model.named_parameters()}
    return torch.stack(losses).mean(), grads


def run_config(devices, schedule, mode, window, pin, errs,
               m=4, dim=16, chunks_per_stage=4, mixed=False,
               grad_accum="stream", acc_slots=None,
               act=False, act_slots=2):
    """One offloaded-pipeline config vs plain pipeline + sequential ref."""
    acc_tag = "" if grad_accum == "stream" else f" ga={grad_accum}"
    if acc_slots is not None:
        acc_tag += f" slots={acc_slots}"
    if act:
        acc_tag += f" act={act_slots}"
    tag = (f"[{devices[0]}/{devices[-1]} {schedule} {MODE_LABEL[mode]} "
           f"W={window} pin={pin}{' mixed' if mixed else ''}{acc_tag}]")
    n_before = len(errs)

    src = [make_chunks(chunks_per_stage, dim, seed=s) for s in range(2)]
    torch.manual_seed(100)
    head = nn.Linear(dim, 4)
    src[1].append(head)

    off_spec = [copy.deepcopy(src[0]), copy.deepcopy(src[1])]
    if mixed:
        off_spec[1] = nn.Sequential(*off_spec[1])
    plain_spec = [nn.Sequential(*copy.deepcopy(s)) for s in src]
    seq_chunks = copy.deepcopy(src[0]) + copy.deepcopy(src[1])

    pipe = Pipeline(stage_modules=off_spec, devices=devices,
                    offload_window=window, offload_pin=pin,
                    offload_keep_activations=mode,
                    offload_grad_accum=grad_accum,
                    offload_acc_slots=acc_slots,
                    offload_activations=act,
                    offload_act_slots=act_slots)
    ref = Pipeline(stage_modules=plain_spec, devices=devices)

    torch.manual_seed(1)
    x = torch.randn(8, dim)
    y = torch.randint(0, 4, (8,))
    loss_fn = F.cross_entropy

    # multi-step: flush -> optimizer -> residency invalidation must cycle
    op = [p for _, p in named_params(off_spec)]
    rp = [p for _, p in named_params(plain_spec)]
    o_opt = torch.optim.SGD(op, lr=0.05, momentum=0.9)
    r_opt = torch.optim.SGD(rp, lr=0.05, momentum=0.9)

    for step in range(3):
        r1 = pipe.step(x, targets=y, schedule=schedule, n_microbatches=m,
                       loss_fn=loss_fn)
        r2 = ref.step(x, targets=y, schedule=schedule, n_microbatches=m,
                      loss_fn=loss_fn)
        check_close(f"{tag} step{step} loss (vs plain)", r1.loss, r2.loss,
                    errs, exact=True)
        r1.flush_grads()
        r2.flush_grads()

        if step == 0:
            # grads vs plain pipeline (bit-exact) and sequential ref (close)
            seq_loss, seq_grads = seq_reference_grads(
                seq_chunks, x, y, m, loss_fn)
            check_close(f"{tag} loss (vs seq)", r1.loss, seq_loss, errs)
            seq_named = list(seq_grads.items())
            for (on, opar), (_, rpar), (sn, sg) in zip(
                    named_params(off_spec), named_params(plain_spec),
                    seq_named):
                if opar.grad is None:
                    errs.append(f"{tag} {on}: no grad")
                    continue
                check_close(f"{tag} {on}.grad (vs plain)", opar.grad,
                            rpar.grad, errs, exact=True)
                check_close(f"{tag} {on}.grad (vs seq {sn})", opar.grad,
                            sg, errs)

        o_opt.step()
        r_opt.step()
        o_opt.zero_grad(set_to_none=False)
        r_opt.zero_grad(set_to_none=False)

    for (on, opar), (_, rpar) in zip(named_params(off_spec),
                                     named_params(plain_spec)):
        check_close(f"{tag} 3-step weight {on}", opar, rpar, errs, exact=True)

    # streamed inference through the offloaded stage's engine forward
    o_out = pipe.infer(x, n_microbatches=m)
    r_out = ref.infer(x, n_microbatches=m)
    check_close(f"{tag} infer", o_out, r_out, errs, exact=True)

    pipe.close()
    ref.close()
    print(f"{tag} {'OK' if len(errs) == n_before else 'MISMATCH'}")


# ── tuple intermediates ────────────────────────────────────────────────────
class SplitHead(nn.Module):
    """x -> (a, b, k): two float streams + an int passthrough (no grad)."""

    def __init__(self, dim):
        super().__init__()
        self.la = nn.Linear(dim, dim)
        self.lb = nn.Linear(dim, dim)

    def forward(self, x):
        return self.la(x), self.lb(x), torch.arange(x.size(0), device=x.device)


class TwoStream(nn.Module):
    """(a, b, k) -> (a', b', k): streams interact, k threads through."""

    def __init__(self, dim):
        super().__init__()
        self.la = nn.Linear(dim, dim)
        self.lb = nn.Linear(dim, dim)

    def forward(self, a, b, k):
        return F.gelu(self.la(a)) + 0.1 * b, F.gelu(self.lb(b)), k


class MergeTail(nn.Module):
    """(a, b, k) -> y: merges the streams, consumes k grad-free."""

    def __init__(self, dim):
        super().__init__()
        self.lo = nn.Linear(dim, dim)

    def forward(self, a, b, k):
        return self.lo(a + b) + 0.01 * k.float().unsqueeze(1)


class SeqTuple(nn.Module):
    """Sequential reference that relays tuples like the chunk convention."""

    def __init__(self, mods):
        super().__init__()
        self.mods = nn.ModuleList(mods)

    def forward(self, *args):
        out = args if len(args) != 1 else args[0]
        for mod in self.mods:
            out = mod(*out) if isinstance(out, tuple) else mod(out)
        return out


def check_tuple(devices, mode, errs, m=4, dim=16):
    """Tuples across chunk boundaries AND across the stage boundary."""
    tag = f"[{devices[0]}/{devices[-1]} tuple {MODE_LABEL[mode]}]"
    n_before = len(errs)
    torch.manual_seed(7)
    mods = [SplitHead(dim), TwoStream(dim), TwoStream(dim), MergeTail(dim)]

    # stage boundary lands between the TwoStreams -> the mailbox carries
    # the (a, b, k) tuple between the GPUs
    off_spec = [copy.deepcopy(mods[:2]), copy.deepcopy(mods[2:])]
    plain_spec = [SeqTuple(copy.deepcopy(mods[:2])),
                  SeqTuple(copy.deepcopy(mods[2:]))]

    pipe = Pipeline(stage_modules=off_spec, devices=devices, offload_window=1,
                    offload_keep_activations=mode)
    ref = Pipeline(stage_modules=plain_spec, devices=devices)

    torch.manual_seed(2)
    x = torch.randn(8, dim)
    y = torch.randn(8, dim)
    r1 = pipe.step(x, targets=y, schedule="staggered_1b1f", n_microbatches=m,
                   loss_fn=F.mse_loss)
    r2 = ref.step(x, targets=y, schedule="staggered_1b1f", n_microbatches=m,
                  loss_fn=F.mse_loss)
    check_close(f"{tag} loss", r1.loss, r2.loss, errs, exact=True)
    r1.flush_grads()
    r2.flush_grads()
    for (on, opar), (_, rpar) in zip(named_params(off_spec),
                                     named_params(plain_spec)):
        if opar.grad is None:
            errs.append(f"{tag} {on}: no grad")
            continue
        check_close(f"{tag} {on}.grad", opar.grad, rpar.grad, errs,
                    exact=True)

    # sequential ground truth too
    seq = SeqTuple(copy.deepcopy(mods))
    losses = []
    for xb, yb in zip(x.chunk(m), y.chunk(m)):
        loss = F.mse_loss(seq(xb), yb)
        loss.backward()
        losses.append(loss.detach())
    check_close(f"{tag} loss (vs seq)", r1.loss, torch.stack(losses).mean(),
                errs)
    for (on, opar), (_, spar) in zip(named_params(off_spec),
                                     seq.mods.named_parameters()):
        check_close(f"{tag} {on}.grad (vs seq)", opar.grad, spar.grad / m,
                    errs)

    pipe.close()
    ref.close()
    print(f"{tag} {'OK' if len(errs) == n_before else 'MISMATCH'}")


def check_autocast(devices, errs, m=4, dim=32):
    """bf16 autocast: offloaded vs plain pipeline, same recipe."""
    for mode in (True, "checkpoint"):
        tag = f"[{devices[0]}/{devices[-1]} autocast-bf16 {MODE_LABEL[mode]}]"
        n_before = len(errs)
        src = [make_chunks(3, dim, seed=21), make_chunks(3, dim, seed=22)]
        off_spec = [copy.deepcopy(s) for s in src]
        plain_spec = [nn.Sequential(*copy.deepcopy(s)) for s in src]

        pipe = Pipeline(stage_modules=off_spec, devices=devices,
                        autocast=torch.bfloat16, offload_window=2,
                        offload_keep_activations=mode)
        ref = Pipeline(stage_modules=plain_spec, devices=devices,
                       autocast=torch.bfloat16)

        torch.manual_seed(3)
        x, y = torch.randn(8, dim), torch.randn(8, dim)
        r1 = pipe.step(x, targets=y, schedule="staggered_1b1f",
                       n_microbatches=m, loss_fn=F.mse_loss)
        r2 = ref.step(x, targets=y, schedule="staggered_1b1f",
                      n_microbatches=m, loss_fn=F.mse_loss)
        check_close(f"{tag} loss", r1.loss, r2.loss, errs, exact=True)
        r1.flush_grads()
        r2.flush_grads()
        for (on, opar), (_, rpar) in zip(named_params(off_spec),
                                         named_params(plain_spec)):
            check_close(f"{tag} {on}.grad", opar.grad, rpar.grad, errs,
                        exact=True)
        pipe.close()
        ref.close()
        print(f"{tag} {'OK' if len(errs) == n_before else 'MISMATCH'}")


def check_grad_bypass(devices, errs, m=4, dim=16):
    """grad_outputs= seeded into an offloaded LAST stage == loss backward."""
    for mode in (True, "checkpoint"):
        tag = f"[{devices[0]}/{devices[-1]} grad-bypass {MODE_LABEL[mode]}]"
        n_before = len(errs)
        src = [make_chunks(3, dim, seed=31), make_chunks(3, dim, seed=32)]
        off_spec = [copy.deepcopy(s) for s in src]
        plain_spec = [nn.Sequential(*copy.deepcopy(s)) for s in src]

        pipe = Pipeline(stage_modules=off_spec, devices=devices,
                        offload_window=2, offload_keep_activations=mode)
        ref = Pipeline(stage_modules=plain_spec, devices=devices)

        torch.manual_seed(4)
        x, y = torch.randn(8, dim), torch.randn(8, dim)
        # dL/dOut of mean-MSE over the FULL batch: per-microbatch grads must
        # be scaled to the full-batch reduction (m microbatches, mean loss)
        r1 = pipe.step(
            x, schedule="staggered_1b1f", n_microbatches=m,
            grad_outputs=lambda out, tgt: 2.0 * (out - tgt) / out.numel(),
            targets=y,
        )
        try:
            _ = r1.loss
            errs.append(f"{tag}: .loss did not raise under grad bypass")
        except RuntimeError:
            pass
        r2 = ref.step(x, targets=y, schedule="staggered_1b1f",
                      n_microbatches=m, loss_fn=F.mse_loss)
        r1.flush_grads()
        r2.flush_grads()
        for (on, opar), (_, rpar) in zip(named_params(off_spec),
                                         named_params(plain_spec)):
            if opar.grad is None:
                errs.append(f"{tag} {on}: no grad")
                continue
            check_close(f"{tag} {on}.grad", opar.grad, rpar.grad, errs,
                        exact=True)
        pipe.close()
        ref.close()
        print(f"{tag} {'OK' if len(errs) == n_before else 'MISMATCH'}")


def check_flat_chunks(devices, errs, m=4, dim=16):
    """chunk_modules= (flat list + auto/weighted split) == nested spec."""
    tag = f"[{devices[0]}/{devices[-1]} flat-chunks]"
    n_before = len(errs)
    torch.manual_seed(51)
    chunks = make_chunks(8, dim, seed=51) + [nn.Linear(dim, 4)]  # 9 chunks
    ref_chunks = copy.deepcopy(chunks)

    # weighted split [4, 5] vs the equivalent nested stage_modules spec
    pipe = Pipeline(chunk_modules=chunks, chunks_per_stage=[4, 5],
                    devices=devices, offload_window=2)
    ref = Pipeline(stage_modules=[ref_chunks[:4], ref_chunks[4:]],
                   devices=devices, offload_window=2)
    if [len(s.engine.chunks) for s in pipe.stages] != [4, 5]:
        errs.append(f"{tag} wrong weighted split")

    torch.manual_seed(5)
    x = torch.randn(8, dim)
    y = torch.randint(0, 4, (8,))
    r1 = pipe.step(x, targets=y, schedule="staggered_1b1f", n_microbatches=m,
                   loss_fn=F.cross_entropy)
    r2 = ref.step(x, targets=y, schedule="staggered_1b1f", n_microbatches=m,
                  loss_fn=F.cross_entropy)
    check_close(f"{tag} loss", r1.loss, r2.loss, errs, exact=True)
    r1.flush_grads()
    r2.flush_grads()
    for i, (c, rc) in enumerate(zip(chunks, ref_chunks)):
        for (n, p), (_, rp) in zip(c.named_parameters(),
                                   rc.named_parameters()):
            check_close(f"{tag} chunk{i}.{n}.grad", p.grad, rp.grad, errs,
                        exact=True)
    check_close(f"{tag} infer", pipe.infer(x, n_microbatches=m),
                ref.infer(x, n_microbatches=m), errs, exact=True)
    pipe.close()
    ref.close()

    # even split hands the remainder to the earlier stages: 9 -> [5, 4]
    p2 = Pipeline(chunk_modules=make_chunks(9, dim, seed=52), devices=devices)
    if [len(s.engine.chunks) for s in p2.stages] != [5, 4]:
        errs.append(f"{tag} wrong even split: "
                    f"{[len(s.engine.chunks) for s in p2.stages]}")
    p2.close()
    print(f"{tag} {'OK' if len(errs) == n_before else 'MISMATCH'}")


def _pipe_traj(spec, devices, errs, *, pin, retier=None, retier_at=0,
               mode=True, schedule="staggered_1b1f", window=2, m=4, dim=16,
               steps=3, act=False):
    """Train a pipeline for a few steps, optionally retiering mid-flight.

    Returns (losses, first-step grads, final weights, infer output) — the
    tier is a memory placement, so a pipeline that retiers into ``retier``
    must produce exactly this, bit for bit, whether it started there or moved.
    """
    pipe = Pipeline(stage_modules=spec, devices=devices,
                    offload_window=window, offload_pin=pin,
                    offload_keep_activations=mode,
                    offload_activations=act, offload_act_slots=1)
    params = [p for _, p in named_params(spec)]
    opt = torch.optim.SGD(params, lr=0.05, momentum=0.9)
    torch.manual_seed(5)
    x = torch.randn(8, dim)
    y = torch.randint(0, 4, (8,))
    losses, grads = [], {}
    for s in range(steps):
        if retier is not None and s == retier_at:
            info = pipe.set_offload_pinned(retier, optimizers=[opt])
            for st in pipe.stages:
                if hasattr(st, "engine"):
                    _assert_stage_wiped(st, info, errs)
        res = pipe.step(x, targets=y, schedule=schedule, n_microbatches=m,
                        loss_fn=F.cross_entropy)
        losses.append(res.loss.detach().cpu().clone())
        res.flush_grads()
        if s == 0:
            grads = {n: p.grad.detach().cpu().clone()
                     for n, p in named_params(spec)}
        opt.step()
        opt.zero_grad(set_to_none=False)
    weights = {n: p.detach().cpu().clone() for n, p in named_params(spec)}
    infer = pipe.infer(x, n_microbatches=m).detach().cpu().clone()
    pipe.close()
    return losses, grads, weights, infer


def _assert_stage_wiped(stage, info, errs):
    """A stage retier is a hard reset, like the engine's."""
    tag = f"stage{stage.stage_index}"
    eng = stage.engine
    summary = info.get(stage.stage_index, {})
    moved = set(summary.get("pinned", ())) | set(summary.get("unpinned", ()))
    for st in eng._state:
        if st.acc_touched or st.acc_gpu is not None or st.acc_where != "empty":
            errs.append(f"{tag} chunk{st.idx}: accumulator survived the "
                        "retier")
        if st.idx in moved and st.grad_acc:
            errs.append(f"{tag} chunk{st.idx}: moved chunk kept its "
                        "accumulator instead of freeing it")
        if st.graph_tensors is not None:
            errs.append(f"{tag} chunk{st.idx}: graph tensors survived")
    if eng._act_store or eng._resident or eng._future:
        errs.append(f"{tag}: packets/residency/itinerary survived the retier")
    if list(stage.params) != list(eng.parameters()):
        errs.append(f"{tag}: st.params is stale after the retier")
    if stage.pinned_layers != eng.pinned_layers:
        errs.append(f"{tag}: pinned_layers passthrough disagrees")


def check_retier(devices, errs, m=4, dim=16, L=4):
    """Runtime tier reassignment on offloaded stages (set_offload_pinned)."""
    tag = f"[{devices[0]}/{devices[-1]} retier]"
    n_before = len(errs)

    def spec():
        src = [make_chunks(L, dim, seed=s) for s in range(2)]
        torch.manual_seed(100)
        src[1].append(nn.Linear(dim, 4))
        return [copy.deepcopy(src[0]), copy.deepcopy(src[1])]

    def compare(label, base, test):
        for s, (bl, tl) in enumerate(zip(base[0], test[0])):
            check_close(f"{tag} {label} loss step{s}", bl, tl, errs, exact=True)
        for nme in base[1]:
            check_close(f"{tag} {label} grad {nme}", base[1][nme],
                        test[1][nme], errs, exact=True)
        for nme in base[2]:
            check_close(f"{tag} {label} weight {nme}", base[2][nme],
                        test[2][nme], errs, exact=True)
        check_close(f"{tag} {label} infer", base[3], test[3], errs, exact=True)

    # 1. schedules x backward modes: retiering into pin=2 == starting there,
    #    both before the first step and between steps
    for schedule, mode in itertools.product(
            ("staggered_1b1f", "gpipe"), (True, "checkpoint")):
        base = _pipe_traj(spec(), devices, errs, pin=2, mode=mode,
                          schedule=schedule)
        for at in (0, 1):
            test = _pipe_traj(spec(), devices, errs, pin=1, retier=2,
                              retier_at=at, mode=mode, schedule=schedule)
            compare(f"{schedule}/{MODE_LABEL[mode]} at={at}", base, test)

    # 2. with activation offload on, and pinning everything / nothing
    base = _pipe_traj(spec(), devices, errs, pin=2, act=True)
    test = _pipe_traj(spec(), devices, errs, pin=1, retier=2, retier_at=1,
                      act=True)
    compare("act-offload", base, test)
    for target in (0, L):
        base = _pipe_traj(spec(), devices, errs, pin=target)
        test = _pipe_traj(spec(), devices, errs, pin=2, retier=target)
        compare(f"-> pin={target}", base, test)

    # 3. spec forms: dict, per-stage list, broadcast set; a pre-built
    #    optimizer must keep working (params are the same objects)
    off = spec()
    pipe = Pipeline(stage_modules=off, devices=devices, offload_window=2,
                    offload_pin=1)
    opt = torch.optim.SGD([p for _, p in named_params(off)], lr=0.05,
                          momentum=0.9)
    torch.manual_seed(5)
    x = torch.randn(8, dim)
    y = torch.randint(0, 4, (8,))
    # NB a count is spread evenly, so 2 of stage 0's 4 chunks is {0, 2}
    for form, want in (({0: [0, 2]}, ([0, 2], [0])),
                       ([2, [1]], ([0, 2], [1])),
                       ({0, 3}, ([0, 3], [0, 3]))):
        pipe.set_offload_pinned(form, optimizers=[opt])
        got = [sorted(st.pinned_layers) for st in pipe.stages]
        if got != [list(want[0]), list(want[1])]:
            errs.append(f"{tag} spec {form!r} -> pinned {got}, expected "
                        f"{[list(want[0]), list(want[1])]}")
        res = pipe.step(x, targets=y, schedule="staggered_1b1f",
                        n_microbatches=m, loss_fn=F.cross_entropy)
        res.flush_grads()
        opt.step()
        opt.zero_grad(set_to_none=False)
    # per-stage memory: pinning on stage 0 must not touch stage 1's GPU
    if devices[0] != devices[-1] and devices[0].startswith("cuda"):
        pipe.set_offload_pinned(0)
        for d in set(devices):
            torch.cuda.synchronize(d)
        before = {d: torch.cuda.memory_allocated(d) for d in set(devices)}
        pipe.set_offload_pinned({0: L})
        for d in set(devices):
            torch.cuda.synchronize(d)
        after = {d: torch.cuda.memory_allocated(d) for d in set(devices)}
        if after[devices[0]] <= before[devices[0]]:
            errs.append(f"{tag} pinning stage 0 did not grow {devices[0]}")
        if after[devices[-1]] != before[devices[-1]]:
            errs.append(f"{tag} pinning stage 0 changed {devices[-1]} by "
                        f"{after[devices[-1]] - before[devices[-1]]} bytes")
        # 5 cycles must not leak on either device
        pipe.set_offload_pinned(0)
        for d in set(devices):
            torch.cuda.synchronize(d)
        cycle_base = {d: torch.cuda.memory_allocated(d) for d in set(devices)}
        for _ in range(5):
            pipe.set_offload_pinned({0: 2, 1: 2})
            res = pipe.step(x, targets=y, schedule="staggered_1b1f",
                            n_microbatches=m, loss_fn=F.cross_entropy)
            res.flush_grads()
            pipe.set_offload_pinned(0)
        for d in set(devices):
            torch.cuda.synchronize(d)
        for d in set(devices):
            leaked = torch.cuda.memory_allocated(d) - cycle_base[d]
            if leaked > 2**20:
                errs.append(f"{tag} 5 cycles leaked {leaked / 2**20:.2f} MiB "
                            f"on {d}")
    pipe.close()

    # 4. mixed pipeline: the broadcast form skips the plain stage
    mixed = spec()
    mixed[1] = nn.Sequential(*mixed[1])
    mpipe = Pipeline(stage_modules=mixed, devices=devices, offload_window=2)
    info = mpipe.set_offload_pinned(2)
    if list(info) != [0]:
        errs.append(f"{tag} mixed: broadcast touched stages {list(info)}, "
                    "expected only the offloaded one")
    for bad in ({1: 2}, [2, 2]):
        try:
            mpipe.set_offload_pinned(bad)
            errs.append(f"{tag} mixed: spec {bad!r} naming a plain stage was "
                        "not rejected")
        except ValueError:
            pass
    res = mpipe.step(x, targets=y, schedule="staggered_1b1f",
                     n_microbatches=m, loss_fn=F.cross_entropy)
    res.flush_grads()
    mpipe.close()

    # 5. flat chunk_modules pipelines retier the same way; resident
    #    (offload=False) ones have no engine to retier
    flat = Pipeline(chunk_modules=make_chunks(8, dim, seed=61),
                    devices=devices, offload_window=2)
    flat.set_offload_pinned([1, 2])
    if [sorted(st.pinned_layers) for st in flat.stages] != [[0], [0, 2]]:
        errs.append(f"{tag} flat-chunks retier landed on "
                    f"{[sorted(st.pinned_layers) for st in flat.stages]}")
    flat.close()
    resident = Pipeline(chunk_modules=make_chunks(4, dim, seed=62),
                        devices=devices, offload=False)
    try:
        resident.set_offload_pinned(1)
        errs.append(f"{tag} resident pipeline accepted set_offload_pinned")
    except ValueError:
        pass
    resident.close()
    print(f"{tag} {'OK' if len(errs) == n_before else 'MISMATCH'}")


def check_inference_alloc(devices, errs, m=4, dim=64, L=4):
    """infer() must not allocate gradient state, on either stage kind.

    Both stage kinds keep an explicit accumulator per param — a full copy of
    the shard on a plain stage's GPU, one param-sized buffer per chunk (GPU
    for pinned chunks, pinned host RAM for streamed ones) on an offloaded
    stage. They are allocated on the stage's first backward, so an
    inference-only pipeline holds none of it.

    The assertions are structural (accumulator counts), deliberately: at this
    file's toy dimensions GPU byte counts are dominated by the per-stream
    cuBLAS workspaces (~9 MiB, allocated on the first GEMM of each pass) and
    an offloaded stage's accumulators mostly live on the host anyway. The
    byte-level measurement lives in ``offload_streaming_check.py``, where the
    baseline is controllable.
    """
    tag = f"[{devices[0]}/{devices[-1]} inference-alloc]"
    n_before = len(errs)
    x = torch.randn(8, dim)
    y = torch.randn(8, dim)

    def accs(pipe):
        return [sum(len(s.grad_acc) for s in st.engine._state)
                if hasattr(st, "engine") else len(st.grad_acc)
                for st in pipe.stages]

    # plain resident, offloaded, and mixed topologies
    specs = {
        "plain": lambda s: [nn.Sequential(*s[0]), nn.Sequential(*s[1])],
        "offloaded": lambda s: [s[0], s[1]],
        "mixed": lambda s: [s[0], nn.Sequential(*s[1])],
    }
    for label, build in specs.items():
        src = [make_chunks(L, dim, seed=81), make_chunks(L, dim, seed=82)]
        pipe = Pipeline(stage_modules=build(src), devices=devices,
                        offload_window=2, offload_pin=1)
        if any(accs(pipe)):
            errs.append(f"{tag} {label}: accumulators allocated at "
                        f"construction ({accs(pipe)})")
        pipe.infer(x, n_microbatches=m)
        if any(accs(pipe)):
            errs.append(f"{tag} {label}: infer() allocated accumulators "
                        f"({accs(pipe)})")
        pipe.flush_grads()          # no backward yet -> must stay a no-op
        if any(accs(pipe)):
            errs.append(f"{tag} {label}: flush_grads() allocated accumulators")
        if any(p.grad is not None for st in pipe.stages for p in st.params):
            errs.append(f"{tag} {label}: flush without a backward set .grad")
        res = pipe.step(x, targets=y, schedule="staggered_1b1f",
                        n_microbatches=m, loss_fn=F.mse_loss)
        res.flush_grads()
        want = [len(st.params) if not hasattr(st, "engine")
                else sum(len(list(s.module.parameters()))
                         for s in st.engine._state)
                for st in pipe.stages]
        if accs(pipe) != want:
            errs.append(f"{tag} {label}: {accs(pipe)} accumulators after a "
                        f"step, expected {want}")
        if any(p.grad is None for st in pipe.stages for p in st.params):
            errs.append(f"{tag} {label}: a param has no .grad after "
                        "step+flush")
        pipe.close()
    print(f"{tag} {'OK' if len(errs) == n_before else 'MISMATCH'}")


def check_rejections(errs):
    """Unsupported combos must refuse loudly."""
    n_before = len(errs)
    chunks = make_chunks(2, 8, seed=41)
    try:
        Pipeline(stage_modules=[chunks, nn.Linear(8, 8)],
                 devices=["cpu", "cpu"], offload_keep_activations=False)
        errs.append("recompute mode (False) was not rejected")
    except ValueError:
        pass
    try:
        Pipeline(stage_modules=[make_chunks(2, 8, seed=42), nn.Linear(8, 8)],
                 devices=["cpu", "cpu"],
                 fake_compute={"fwd": [0.01, 0.01], "bwd": [0.02, 0.02]})
        errs.append("fake_compute with chunked stages was not rejected")
    except ValueError:
        pass
    flat_bad = [
        ("chunk_modules + stage_modules",
         lambda: Pipeline(chunk_modules=make_chunks(2, 8, seed=43),
                          stage_modules=[nn.Linear(8, 8)], devices=["cpu"])),
        ("chunks_per_stage without chunk_modules",
         lambda: Pipeline(stage_modules=[nn.Linear(8, 8)], devices=["cpu"],
                          chunks_per_stage=[1])),
        ("chunks_per_stage sum mismatch",
         lambda: Pipeline(chunk_modules=make_chunks(4, 8, seed=44),
                          chunks_per_stage=[2, 3], devices=["cpu", "cpu"])),
        ("chunks_per_stage/devices length mismatch",
         lambda: Pipeline(chunk_modules=make_chunks(4, 8, seed=45),
                          chunks_per_stage=[2, 2], devices=["cpu"] * 3)),
        ("nested entry inside chunk_modules",
         lambda: Pipeline(chunk_modules=[make_chunks(2, 8, seed=46)],
                          devices=["cpu"])),
        ("fewer chunks than stages",
         lambda: Pipeline(chunk_modules=make_chunks(1, 8, seed=47),
                          devices=["cpu", "cpu"])),
    ]
    for msg, fn in flat_bad:
        try:
            fn()
            errs.append(f"{msg} was not rejected")
        except ValueError:
            pass
    print(f"[rejections] {'OK' if len(errs) == n_before else 'MISMATCH'}")


def pick_device_pairs():
    pairs = [("cpu", "cpu")]
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        if n >= 4:
            pairs.append(("cuda:1", "cuda:3"))
        elif n >= 2:
            pairs.append(("cuda:0", "cuda:1"))
        else:
            pairs.append(("cuda:0", "cuda:0"))
    return pairs


def main() -> int:
    errs: list = []
    L = 4
    for devices in pick_device_pairs():
        for schedule, mode in itertools.product(
                ("staggered_1b1f", "gpipe"), (True, "checkpoint")):
            for window, pin in ((1, 0), (2, 0), (L, 0), (2, 2)):
                run_config(devices, schedule, mode, window, pin, errs,
                           chunks_per_stage=L)
        run_config(devices, "staggered_1b1f", True, 2, 0, errs,
                   chunks_per_stage=L, mixed=True)
        # legacy CPU packet accumulation stays supported ...
        run_config(devices, "staggered_1b1f", True, 2, 0, errs,
                   chunks_per_stage=L, grad_accum="cpu")
        run_config(devices, "gpipe", "checkpoint", 2, 0, errs,
                   chunks_per_stage=L, grad_accum="cpu")
        # ... and acc_slots=1 forces constant evict/reload round trips —
        # accumulated values must survive the D2H/H2D moves bit-exactly
        run_config(devices, "staggered_1b1f", True, 2, 0, errs,
                   chunks_per_stage=L, acc_slots=1)
        run_config(devices, "gpipe", True, 1, 0, errs,
                   chunks_per_stage=L, acc_slots=1)
        # activation offload: bit-exact vs the plain pipeline across
        # schedules x modes x act_slots {1 (thrash), 2}; gpipe holds all m
        # microbatch packets in flight — the heaviest pressure case
        for schedule, mode in itertools.product(
                ("staggered_1b1f", "gpipe", "1f1b"), (True, "checkpoint")):
            for aslots in (1, 2):
                run_config(devices, schedule, mode, 2, 0, errs,
                           chunks_per_stage=L, act=True, act_slots=aslots)
        # combined thrash: W=1 weights + 1 acc slot + 1 act slot, and the
        # mixed offloaded+plain topology with act offload on
        run_config(devices, "staggered_1b1f", True, 1, 0, errs,
                   chunks_per_stage=L, acc_slots=1, act=True, act_slots=1)
        run_config(devices, "gpipe", "checkpoint", 1, 0, errs,
                   chunks_per_stage=L, acc_slots=1, act=True, act_slots=1)
        run_config(devices, "staggered_1b1f", True, 2, 0, errs,
                   chunks_per_stage=L, mixed=True, act=True, act_slots=1)
        for mode in (True, "checkpoint"):
            check_tuple(devices, mode, errs)
        check_autocast(devices, errs)
        check_grad_bypass(devices, errs)
        check_flat_chunks(devices, errs)
        check_retier(devices, errs, L=L)
        check_inference_alloc(devices, errs, L=L)
    check_rejections(errs)

    for e in errs[:30]:
        print(f"    {e}")
    print("PASS" if not errs else f"FAIL ({len(errs)} mismatches)")
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
