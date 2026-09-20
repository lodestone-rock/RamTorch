# Contributing to RamTorch

This is the shared, repository-wide entrypoint for coding agents and human
contributors. Read it before editing. Keep durable guidance here and in `docs/`,
not in local chat transcripts or `agents_notepad/`. `CLAUDE.md` imports this file
for Claude Code; tools without automatic `AGENTS.md` discovery should be pointed
here explicitly. Do not duplicate these instructions in tool-specific files.

## Start here

- Read `README.md` for the project and public API overview.
- Read `docs/pipeline_parallel.md` for pipeline execution, scheduling, ownership,
  precision, profiling, and examples.
- Read `docs/offload.md` before changing weight/activation streaming or NVMe.
- Inspect `git status` and the relevant implementation/check scripts first.
  Preserve unrelated local changes; do not assume the checkout is clean.
- `agents_notepad/` and `scratchpad/` are optional, untracked local research.
  They may not exist in a fresh clone. No library code or required check should
  depend on them. Example output paths under `scratchpad/` are generated artifacts.

## Project shape

RamTorch targets a single workstation/node with ordinary PCIe GPUs and plentiful
RAM, not an NVLink cluster. Modern pipeline execution is one process with one
worker thread per stage; do not introduce process groups, NCCL, or `torchrun` as
an incidental dependency. Legacy ZeRO utilities are separate and still supported.

- `ramtorch/pipeline.py`: resident `Stage`, gradient accumulation, results/tracing.
- `ramtorch/pipeline_relay.py`: `Pipeline`, mailboxes, synchronous schedules,
  streaming inference, and training-session ownership.
- `ramtorch/pipeline_2bw.py`: persistent resident PipeDream-2BW workers, two weight
  banks, bounded admission, and optional CPU optimizer masters/state.
- `ramtorch/pipeline_2bw_trace.py`: bounded Kineto traces and worker annotations.
- `ramtorch/delayed_optim.py`: AdamEF/Lion and update-level error feedback.
- `ramtorch/offload.py`, `pipeline_offload.py`, `nvme_store.py`: streaming engines
  and backing storage. `pipeline_optimizer.py` provides parallel per-stage
  optimizer updates for synchronous pipelines.
- `ramtorch/pipeline_easy.py`: traced/automatic `PipelineModel` convenience API.
- `ramtorch/*simulator.py`: schedule/memory models, not proof of runtime overlap.
- `examples/`: runnable demonstrations, numerical oracles, and `*_check.py`
  regression scripts. Most correctness checks are standalone scripts, not pytest.
- `ramtorch/__init__.py`: public exports. Preserve legacy `ramtorch.AdamW`; it is
  distinct from `torch.optim.AdamW` and the newer `AdamEF`.

## Environment

Use an isolated environment and a PyTorch build appropriate for the machine.
The recent pipeline/2BW work was validated with PyTorch 2.8.0 / CUDA 12.8; this is
a tested configuration, not a guarantee that every older PyTorch works. Pipeline
imports require `torch.distributed.pipelining`. Do not work around an incompatible
PyTorch installation by deleting imports or weakening checks.

```bash
python -m venv .venv
source .venv/bin/activate
# Install a suitable torch/torchvision build for your CPU or CUDA first.
python -m pip install -r requirements.txt
python -m pip install -e .
# Optional, for convergence plots:
python -m pip install matplotlib
```

Run commands from the repository root with the chosen environment's `python`.
`PYTHONPATH=.` also supports running the scripts without an editable install.
Do not commit environments, datasets, traces, checkpoints, machine-specific
paths, remote-machine credentials, or external-repository symlinks.

## Non-negotiable correctness and safety

- Preserve the NVMe **training** consent gate in `offload.py`: root or sudo/wheel
  membership AND `RAMTORCH_NVME_ACKNOWLEDGE=1`; the wear warning remains unless
  the user separately sets `RAMTORCH_NVME_QUIET=1`. Never auto-set these variables,
  bypass the gate, or silently quiet the warning to make a test pass. Inference
  is deliberately ungated after its initial storage write. Do not add NVMe to
  pipeline offloading without an explicit design decision and user consent.
- CUDA enqueue completion is not device completion. Preserve producer/consumer
  event ordering, allocator/tensor lifetimes, D2H readiness, and pinned-host
  buffer ownership until asynchronous copies finish. Grad/autocast contexts
  are thread-local: configure them in workers, not just the caller.
- Do not insert device-wide synchronizations into steady-state hot paths as a
  quick race fix. Boundary drains and correctness observers are different from
  asynchronous training; document synchronization costs explicitly.
- Mean-loss gradients are scaled exactly once across equal-sized microbatches.
  Preserve `None` versus zero-gradient semantics and frozen/unused parameters.
- Inference-only execution must not allocate parameter-sized gradient state.
  Retiering is a documented between-step hard reset; preserve its optimizer
  state migration and lifetime rules.
- 2BW is a separate `Pipeline.train_session()` API, not a schedule string to
  silently substitute into `Pipeline.step()`. Maintain two compute-weight banks,
  one evolving optimizer state, consistent forward/backward versions, and
  `n_microbatches >= pipeline depth`. The first two groups use initial weights.
  Do not overwrite a bank with live graphs or reset history between `run()` calls.
- Keep current 2BW restrictions explicit: resident deterministic buffer-free
  modules, no stochastic/tied/forward-mutating weights, and the exact optimizer
  allowlist. CPU optimizer placement does not mean compute-weight offloading.
- EF acts on complete optimizer **updates**, not extrapolated raw gradients.
  Advance moments once, save the uncorrected update including its LR/decay, and
  preserve startup/missing-gradient/serialization semantics in `delayed_optim.py`.
  Optimizer `state_dict()` support is not a full 2BW-session checkpoint API.

## Validation workflow

Select checks for the changed subsystem; inspect their arguments before running.
Do not run a GPU/NVMe stress test or commandeer a remote machine without approval.
CPU checks exercise semantics, not real multi-GPU transfers or overlap.

```bash
# Small 2BW and EF CPU checks:
PYTHONPATH=. python examples/pipedream_2bw_reference.py --device cpu --self-test
PYTHONPATH=. python examples/pipedream_2bw_check.py --devices cpu,cpu
PYTHONPATH=. python examples/pipedream_2bw_optimizer_check.py --devices cpu,cpu --bf16

# Mailbox/streaming-inference regression, CPU only:
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python examples/pipeline_infer_stream_check.py

# On an explicitly available two-GPU machine:
PYTHONPATH=. python examples/pipedream_2bw_check.py --devices cuda:0,cuda:1
PYTHONPATH=. python examples/pipedream_2bw_optimizer_check.py --devices cuda:0,cuda:1 --bf16

git diff --check
```

For offload changes, inspect `examples/offload_streaming_check.py`,
`pipeline_offload_check.py`, `offload_optimizer_check.py`, and
`offload_checkpoint_study.py`. Their device needs vary; do not blindly run all
examples. Other focused checks cover AMP, tuple/no-grad outputs, grad bypass,
gradient accumulation, and simulators. NVMe checks require special care above.

- Numerical changes need a small independent reference. Preserve exact checks
  where arithmetic is intended to match; do not loosen tolerances to hide a bug.
  CPU vs GPU or fused vs nonfused optimizer arithmetic can legitimately differ.
- Scheduling changes need version/lifetime/order and failure-cleanup checks,
  not just a falling loss or a faster timer. Test continued runs and close/error
  behavior. Separate CUDA transport problems from optimizer/schedule defects.
- Performance claims need clean timings separate from profiling. Capture a
  bounded window, compress traces, verify real CUDA kernels on every requested
  GPU, and report hardware, precision, shapes, optimizer, and synchronization.
  Host spans and simulator predictions are not measured GPU utilization.
- For convergence, use matched initialization/data/hyperparameters and distinguish
  fresh from delayed gradients. `examples/mnist_pipedream_2bw.py` provides Adam/
  Lion controls and EF with a short separate profiler replay. Do not claim EF
  universally improves quality from one seed or select recipes on the test set.

## Change and handoff conventions

Prefer focused edits consistent with surrounding Python style. Keep optional
example dependencies out of core imports. Update public docs and relevant checks
when changing an API, invariant, supported configuration, or optimizer semantics.
Record exact commands/results and what could not be tested; do not imply that
prior machine-specific experiments ran in the current checkout.

Keep this entrypoint concise; put detailed design explanations in `docs/`.
Inspect staged files before committing. Commit/merge/push only as requested;
do not perform an incidental release or version bump. Never overwrite another
contributor's changes or publish local experiment artifacts/credentials.
