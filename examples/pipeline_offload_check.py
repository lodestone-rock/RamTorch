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
    check_rejections(errs)

    for e in errs[:30]:
        print(f"    {e}")
    print("PASS" if not errs else f"FAIL ({len(errs)} mismatches)")
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
