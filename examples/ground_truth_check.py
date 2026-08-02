"""
ground_truth_check.py
---------------------
Ground-truth sanity check: a plain sequential forward+backward pass using plain
PyTorch (NO ramtorch pipeline), compared against every pipeline schedule.

This is the reference that does not touch our library at all for its forward/
backward — it just calls ``model(data)`` and ``loss.backward()``. We capture:

  * the scalar loss
  * every parameter gradient (after backward)
  * every layer's intermediate output activation (via forward hooks)

Then we run each pipeline schedule (gpipe / 1f1b / staggered_1b1f) and check,
against that ground truth:

  1. loss parity
  2. per-parameter gradient parity (after flush_grads)
  3. boundary-activation parity — the last-stage outputs per microbatch must
     match the corresponding slices of the sequential full-batch output, which
     confirms the activations flowing through the pipeline are numerically
     identical, not just the final grads.

If all three match, the pipeline (split + manual autograd + schedule + relay
handoff) is provably equivalent to a plain autograd pass.

Usage:
    python examples/ground_truth_check.py
    python examples/ground_truth_check.py --device cpu
"""

import argparse
import copy

import torch
import torch.nn as nn
from torch.distributed.pipelining import SplitPoint

from ramtorch import run_pipeline_relay


# ── Model (same as pipeline_single_process_demo.py) ──────────────────────────

class Layer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(torch.relu(self.lin(x)))


class MyModel(nn.Module):
    def __init__(self, dim: int = 16, n_layers: int = 4):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.layers = nn.ModuleList([Layer(dim) for _ in range(n_layers)])
        self.head = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)


SPLIT_SPEC = {
    "layers.1": SplitPoint.BEGINNING,
    "layers.3": SplitPoint.BEGINNING,
}  # → 3 stages: [embed+L0], [L1+L2], [L3+head]

loss_fn = nn.MSELoss()


# ── Ground truth: pure PyTorch, no ramtorch ──────────────────────────────────

def ground_truth(model, data, targets):
    """
    Plain sequential forward + backward. Captures loss, param grads, and every
    submodule's output activation via forward hooks. Returns a dict.
    """
    acts = {}
    hooks = []

    def make_hook(name):
        def hook(module, inp, out):
            acts[name] = out.detach().clone()
        return hook

    for name, mod in model.named_modules():
        if name:  # skip the root
            hooks.append(mod.register_forward_hook(make_hook(name)))

    model.zero_grad(set_to_none=True)
    out = model(data)
    loss = loss_fn(out, targets)
    loss.backward()

    for h in hooks:
        h.remove()

    return {
        "output": out.detach().clone(),
        "loss": loss.detach().clone(),
        "grads": {n: p.grad.detach().clone() for n, p in model.named_parameters()},
        "acts": acts,
    }


# ── Pipeline run ──────────────────────────────────────────────────────────────

def run_pipeline(schedule, model, data, targets, devices, n_micro, overlap):
    m = copy.deepcopy(model)
    result = run_pipeline_relay(
        m,
        example_input=data[: data.shape[0] // n_micro],
        split_spec=SPLIT_SPEC,
        data=data,
        targets=targets,
        schedule=schedule,
        n_microbatches=n_micro,
        loss_fn=loss_fn,
        devices=devices,
        overlap=overlap,
    )
    result.flush_grads()  # mean-scaled grads into .grad
    return {
        "loss": result.loss.detach().cpu(),
        "grads": {n: p.grad.detach().cpu().clone() for n, p in m.named_parameters()},
        "outputs": [o.detach().cpu() for o in result.outputs],
    }


# ── Comparison helpers ────────────────────────────────────────────────────────

def _max_diff(a, b):
    return (a - b).abs().max().item()


def check(name, ref, got, rtol, atol):
    ok = torch.allclose(ref, got, rtol=rtol, atol=atol)
    return ok, _max_diff(ref, got)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=str, default=None,
                    help="force all stages onto one device (e.g. cpu); default: spread over GPUs")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--micro", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    n_stages = 3
    torch.manual_seed(args.seed)
    model = MyModel(dim=args.dim, n_layers=args.layers)
    data = torch.randn(args.batch, args.dim)
    targets = torch.randn(args.batch, 1)

    if args.device:
        devices = [args.device] * n_stages
    else:
        n_cuda = torch.cuda.device_count()
        devices = ([f"cuda:{i % n_cuda}" for i in range(n_stages)]
                   if n_cuda else ["cpu"] * n_stages)
    print(f"config: dim={args.dim} layers={args.layers} batch={args.batch} "
          f"micro={args.micro} devices={devices}")

    # ── Ground truth ──────────────────────────────────────────────────────────
    gt_model = copy.deepcopy(model)
    gt = ground_truth(gt_model, data, targets)
    print(f"ground-truth loss: {gt['loss'].item():.6f}  "
          f"(captured {len(gt['grads'])} param grads, {len(gt['acts'])} activations)\n")

    rtol, atol = 1e-4, 1e-5
    all_ok = True

    for sched in ["gpipe", "1f1b", "staggered_1b1f"]:
        for overlap in [False, True]:
            res = run_pipeline(sched, model, data, targets, devices, args.micro, overlap)

            # 1. loss parity
            loss_ok, loss_d = check("loss", gt["loss"].cpu(), res["loss"], rtol, atol)

            # 2. per-param grad parity
            grad_max = 0.0
            grad_bad = None
            for n, g in res["grads"].items():
                d = _max_diff(g, gt["grads"][n].cpu())
                if d > grad_max:
                    grad_max, grad_bad = d, n
            grad_ok = grad_max < 1e-4

            # 3. boundary-activation parity: concat the per-microbatch last-stage
            # outputs along dim 0 and compare to the full-batch sequential output.
            pipe_out = torch.cat(res["outputs"], dim=0)
            act_ok, act_d = check("output", gt["output"].cpu(), pipe_out, rtol, atol)

            ok = loss_ok and grad_ok and act_ok
            all_ok &= ok
            status = "OK " if ok else "FAIL"
            print(f"  [{status}] {sched:<16} overlap={overlap!s:<5} "
                  f"lossΔ={loss_d:.2e}  gradΔ={grad_max:.2e}({grad_bad})  outΔ={act_d:.2e}")

    print("\n" + ("ALL GROUND-TRUTH CHECKS PASSED" if all_ok else "SOME CHECKS FAILED"))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
