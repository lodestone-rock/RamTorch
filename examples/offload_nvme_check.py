"""
offload_nvme_check.py
---------------------
Correctness checks for the OffloadModel NVMe tier (masters on disk via
``NvmeTensorStore`` mmap, streamed disk -> pinned staging -> GPU).

Verifies against a full-resident reference:
  1. masters actually live on the file mapping (storage filename check)
  2. streamed inference is bit-identical
  3. training grads are bit-identical in all three backward modes
     (recompute / keep / "checkpoint")
  4. an optimizer step updates the mapped masters in place and the next
     forward matches the reference stepped the same way
  5. close() removes the scratch file

Runs on CUDA when available, else CPU (slow but exercises the same paths).

Usage:
    python examples/offload_nvme_check.py
"""

import copy
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ramtorch import OffloadModel  # noqa: E402

N_CHUNKS, DIM, BATCH = 12, 64, 8
NVME, PIN = 5, 2
NVME_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "_offload_nvme_masters.bin")


def make_chunks(seed: int):
    torch.manual_seed(seed)
    return [
        nn.Sequential(nn.Linear(DIM, 4 * DIM), nn.GELU(),
                      nn.Linear(4 * DIM, DIM))
        for _ in range(N_CHUNKS)
    ]


def ref_forward(chunks: nn.ModuleList, x):
    h = x
    for m in chunks:
        h = m(h)
    return h


def cmp(name: str, a, b, failures: list):
    a, b = a.detach().cpu(), b.detach().cpu()
    if torch.equal(a, b):
        return
    err = (a - b).abs().max().item()
    failures.append(f"{name}: max abs err {err:.3e}")


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  chunks={N_CHUNKS}  nvme={NVME}  pin={PIN}")
    torch.manual_seed(0)
    x = torch.randn(BATCH, DIM)
    y = torch.randn(BATCH, DIM)
    failures: list = []

    # full-resident reference (fresh copy per backward mode, same seed)
    ref = nn.ModuleList(make_chunks(seed=1)).to(device)
    out_ref = ref_forward(ref, x.to(device))
    loss_ref = F.mse_loss(out_ref, y.to(device))
    for p in ref.parameters():
        p.grad = None
    loss_ref.backward()

    for mode in (False, True, "checkpoint"):
        model = OffloadModel(
            make_chunks(seed=1), device=device, window=2, pin=PIN,
            nvme=NVME, nvme_path=NVME_PATH, keep_activations=mode,
        )
        tag = {False: "recompute", True: "keep"}.get(mode, mode)

        # 1. masters really live on the file mapping
        assert model.nvme_layers and model.nvme_layers.isdisjoint(
            model.pinned_layers
        ), "nvme placement overlaps pinned"
        for i in sorted(model.nvme_layers):
            p = next(model.chunks[i].parameters())
            fn = p.data.untyped_storage().filename
            assert fn and os.path.samefile(fn, NVME_PATH), \
                f"chunk {i} master not file-backed (storage file: {fn})"
        assert os.path.getsize(NVME_PATH) > 0

        # 2. inference parity
        with torch.no_grad():
            cmp(f"[{tag}] inference", model(x), out_ref, failures)

        # 3. training grad parity
        res = model.step(x, targets=y, loss_fn=F.mse_loss)
        model.flush_grads()
        cmp(f"[{tag}] loss", res.loss, loss_ref, failures)
        for (n, p), (n_ref, p_ref) in zip(
            model.chunks.named_parameters(), ref.named_parameters()
        ):
            assert n == n_ref
            cmp(f"[{tag}] grad {n}", p.grad, p_ref.grad, failures)
        assert model.stats["nvme_loads"] > 0, "no loads hit the nvme tier"

        if mode is False:
            # 4. optimizer step updates the mapped masters in place
            opt = torch.optim.SGD(model.parameters(), lr=0.05)
            opt.step()
            ref_step = copy.deepcopy(ref)
            opt_ref = torch.optim.SGD(ref_step.parameters(), lr=0.05)
            for p, p_ref in zip(model.parameters(), ref_step.parameters()):
                p_ref.grad = p.grad.to(device)
            opt_ref.step()
            with torch.no_grad():
                cmp(f"[{tag}] post-step inference", model(x),
                    ref_forward(ref_step, x.to(device)), failures)
            # the mmap masters must reflect the update
            i = min(model.nvme_layers)
            cmp(f"[{tag}] mmap master updated",
                next(model.chunks[i].parameters()),
                next(iter(ref_step[i].parameters())), failures)

        model.close()
        assert not os.path.exists(NVME_PATH), "close() left the scratch file"
        print(f"  [{tag}] ok (nvme_loads={model.stats['nvme_loads']})")

    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  {f}")
        return 1
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
