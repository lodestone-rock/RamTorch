# Pipeline Parallelism

RamTorch's pipeline parallelism splits a model *vertically* (by layers) across
multiple GPUs and trains/infers pipeline-parallel **in a single process** — no
`torchrun`, no process groups, no NCCL. Each stage runs on its own GPU driven by
one worker thread; activations flow forward and gradients flow backward through
lightweight thread-safe handoffs (a "relay").

This document is the complete reference. For a quick start, see the
`PipelineModel` section in the README.

---

## When to use which API

There are two entry points. **Choose based on your model's complexity:**

| API | How it splits | Use when |
|---|---|---|
| `PipelineModel` | **Auto-split** via `torch.export` tracing | Simple models (linear stacks, basic CNNs/MLPs) |
| `Pipeline(stage_modules=[...])` | **You pre-partition** into stage modules | Real/complex architectures (attention, control flow, custom ops) |
| `Pipeline(chunk_modules=[...])` | **You dice into a flat chunk list**; the pipeline splits it across devices (evenly or per `chunks_per_stage`) | Models that are naturally a stack of blocks; also unlocks per-stage weight streaming (`offload=`) |

**Rule of thumb:** start with `PipelineModel` for convenience. If `torch.export`
fails or misbehaves on your model (see the gotchas below), switch to
`Pipeline(stage_modules=...)` — it's the robust, recommended path for anything
non-trivial. If your model is a plain stack of blocks anyway, `chunk_modules`
is the least code of all.

---

## The `torch.export` gotchas (why manual splitting exists)

`PipelineModel` auto-splits your model by tracing it with `torch.export`. This
is convenient but **brittle**. The tracer:

1. **Fails on dynamic control flow & custom ops.** `torch.export` needs a full
   static graph. Data-dependent `if`/`for`, Python-side branching on tensor
   values, and unsupported custom ops cause graph breaks or hard errors.

2. **Mangles some ops across stage boundaries.** We hit this directly: an
   `nn.MultiheadAttention` transformer failed with
   `view size is not compatible ... spans across two contiguous subspaces` when
   the exported graph was partitioned — the attention's internal reshape
   assumptions broke at the split point.

3. **Specializes on the example batch size.** The traced graph bakes
   `x.size(0)` as a constant, so the stages are shape-specialized to your
   `example_input`'s batch size. (This is why `PipelineModel.forward` chunks and
   pads inference batches to the traced microbatch size.)

4. **Is not idempotent across re-tracing.** Re-splitting an already-split model
   nests the traced forward and blows the recursion limit. (`PipelineModel`
   splits once at construction and reuses the stages, so this only bites if you
   call the one-shot `run_pipeline_relay` in a loop — don't; use `Pipeline` /
   `PipelineModel`.)

**When you hit any of these, pre-partition the model yourself** (next section).

---

## Manual splitting with `Pipeline(stage_modules=[...])`

This bypasses `torch.export` entirely. You define your model as a **list of
stage modules**, each a plain `nn.Module` whose `forward` consumes the previous
stage's output. It's a trivial change to make in your own model class and is the
recommended approach for real architectures.

### Pattern

```python
import itertools, torch
from ramtorch import Pipeline

# You define the split in your model code — no torch.export involved.
stage0 = EmbedAndFirstBlocks(...)     # nn.Module: input -> intermediate
stage1 = RemainingBlocksAndHead(...)  # nn.Module: intermediate -> output

pipe = Pipeline(stage_modules=[stage0, stage1],
                devices=["cuda:0", "cuda:1"])

# Optimize over the stages' own parameters (they're separate modules, so a
# single optimizer must see all of them).
opt = torch.optim.Adam(
    itertools.chain(*(s.parameters() for s in (stage0, stage1))), lr=1e-3)

for x, y in train_loader:
    result = pipe.step(x, targets=y, schedule="staggered_1b1f",
                       n_microbatches=4, loss_fn=loss_fn)
    result.flush_grads()   # mean-scale microbatch grads into .grad
    opt.step()
    opt.zero_grad()
```

### What a stage module must satisfy

- `forward(prev_output) -> output` — takes the previous stage's output, returns
  its own. The first stage takes your pipeline input; the last stage returns the
  final output (logits).
- That's it. The manual autograd (`torch.autograd.grad`) works on any module's
  forward — no graph tracing required.

### Reference

`examples/mnist_pipeline_big_transformer_manual.py` — a ~85M-param ViT-style
transformer (embed stage + head stage) trained this way. The auto-traced
equivalent fails on its attention reshapes; the manual version trains cleanly.

---

## Manual splitting with chunks: `Pipeline(chunk_modules=[...])`

Often the natural way to write a model is not "two big stage modules" but a
**flat ordered list of small chunks** (embed, block, block, ..., head). Hand
that list to the pipeline and let it do the stage splitting for you:

```python
from ramtorch import Pipeline

chunks = [Embed(...)] + [Block(...) for _ in range(30)] + [Head(...)]

# even split: one stage per device, earlier stages take the remainder
pipe = Pipeline(chunk_modules=chunks,
                devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])

# or weight the split yourself (must sum to len(chunk_modules)):
pipe = Pipeline(chunk_modules=chunks, chunks_per_stage=[7, 8, 8, 9],
                devices=["cuda:0", "cuda:1", "cuda:2", "cuda:3"])

opt = torch.optim.Adam(
    itertools.chain(*(st.params for st in pipe.stages)), lr=1e-3)
```

Chunk contract (the same dicing convention as `OffloadModel`): chunk `i+1`
consumes chunk `i`'s output — a single tensor, or a **tuple** whose elements
become the next chunk's positional args (non-float extras like masks pass
through grad-free). The first chunk takes the pipeline input; the last returns
the final output.

Everything else (`step`, schedules, `infer`, autocast, grad bypass, tuple
inputs) works exactly as with `stage_modules`. Notes:

- With neither `chunks_per_stage` nor `devices`, every visible CUDA device
  gets a stage (one CPU stage without CUDA).
- The split weights by **chunk count** only — for unevenly sized chunks, pass
  `chunks_per_stage` yourself.
- Optimize over `pipe.stages[i].params` (as above); `Pipeline` itself has no
  `.parameters()`.
- By default chunked stages **stream their weights from CPU RAM**
  (`offload=True` — see "Streaming stage weights from CPU RAM" below). Pass
  **`offload=False`** for ordinary fully-resident stages: each stage then just
  runs its chunks in sequence on its GPU, bit-identical to hand-building
  `nn.Sequential` stages. Dice once, flip `offload` when a stage stops
  fitting.

Reference: `examples/mnist_pipeline_big_transformer_manual.py --chunks`
(resident) / `--offload` (streamed) — the same ViT diced one level finer, and
`examples/mnist_pipeline_offload.py`, which builds both its pipelines from one
flat chunk list. Parity vs hand-built stages is asserted in
`examples/pipeline_offload_check.py`.

---

## Input features: tensor, tuple, and pre-diced microbatches

`Pipeline.step` and `Pipeline.infer` accept three input forms. **Tuple forms
are only supported on the manual paths (`stage_modules` / `chunk_modules`)** —
the traced `PipelineModel` path raises a clear `ValueError` (torch.export
can't trace them).

### 1. Single tensor (default)

```python
pipe.step(images, targets=y, n_microbatches=4, loss_fn=loss_fn)
```
The tensor is chunked along dim 0 into `n_microbatches` microbatches.

### 2. Flat tuple — multi-input models

```python
pipe.step((ctx, x), targets=y, n_microbatches=4, loss_fn=loss_fn)
```
Each tuple element is chunked along dim 0, and each microbatch is unpacked as
**positional args** into the stage-0 module:

```python
class Stage0(nn.Module):
    def forward(self, ctx, x):   # receives (ctx_mb, x_mb)
        ...
```
Use this for models that take multiple inputs (e.g. a context embedding plus an
image). Every tuple element must have a batch dim divisible by `n_microbatches`.

### 3. Nested pre-diced tuple — independent microbatches

```python
nested = tuple((ctx_mb, x_mb) for ctx_mb, x_mb in zip(ctx.chunk(4), x.chunk(4)))
pipe.step(nested, targets=y, n_microbatches=4, loss_fn=loss_fn)
```
The outer tuple has one entry per microbatch; each entry is a tensor or a tuple
of tensors, **used as-is** (no re-chunking, no shared storage). Because each
microbatch is independent, a downstream consumer can start on microbatch 0 as
soon as it's ready, without waiting for the rest of the batch.

### Output mirrors input (inference)

`Pipeline.infer` returns output in the **same shape as its input**:

```python
logits = pipe.infer(images, n_microbatches=4)         # tensor in  -> tensor out
outs   = pipe.infer(nested, n_microbatches=4)         # nested in  -> nested tuple out
```
Feed pre-diced microbatches, get pre-diced microbatches back; feed a tensor, get
a concatenated tensor. This symmetry makes it easy to chain a frozen encoder's
inference into a training pipeline (see the frozen-encoder example).

---

## Tuple outputs & no-grad args (masks, aux outputs)

A stage can return a **tuple of tensors**, and you can mark some of them as
**forward-only** so they are relayed to the next stage but excluded from the
backward pass. This is the right tool for auxiliary outputs that carry no
gradient — padding/attention masks, routing decisions, detached feature taps —
and it removes the need to pack/unpack non-differentiable values into a single
tensor.

```python
class Stage0(nn.Module):
    # Output arg index 1 is forward-only (no backward through it).
    out_no_grad = (1,)

    def forward(self, x):
        feat = self.blocks(x)          # differentiable
        mask = feat.abs().mean(-1) > 0  # bool padding mask
        return feat, mask              # (tensor, bool)

class Stage1(nn.Module):
    def forward(self, feat, mask):     # receives both as positional args
        feat = feat * mask.unsqueeze(-1).to(feat.dtype)  # masked
        return self.head(feat)
```

**How grad eligibility is decided** for each output arg `i`:

```
needs_grad[i] = (i not in module.out_no_grad) and output[i].is_floating_point()
```

- **Non-float outputs (bool/int/long) are always forward-only**, even without
  `out_no_grad`. A bool mask is auto-skipped, so returning one no longer crashes
  the backward (previously `torch.autograd.grad` would reject a bool grad
  target). This alone fixes the common "pack the mask into a float tensor"
  workaround.
- **`out_no_grad`** (a tuple of output indices on the module) additionally marks
  *floating-point* outputs as forward-only — e.g. a frozen auxiliary head or a
  detached feature tap you don't want grads flowing through.

**What happens under the hood:** the *whole* tuple is relayed forward to the
next stage (one CUDA event per tensor). At the boundary, only the grad-needing
args receive a backward gradient; no-grad args get `None`. The downstream stage
receives every arg as a positional input (see the flat-tuple input form), and
non-requiring inputs are simply excluded from its autograd graph. Everything is
handled inside the relay — no changes to `Pipeline` calls.

Works identically in `step` (training) and `infer` (forward-only). In `infer`,
a tuple-output last stage returns a tuple of per-arg concatenated tensors.
Verified in `examples/tuple_no_grad_check.py` (loss + grad parity vs a
sequential masked baseline, all schedules) and exercised in
`examples/mnist_frozen_encoder_overlap.py` (a bool padding mask crossing a stage
boundary).

---

## Mixed precision (autocast)

**A plain `with torch.autocast(...)` around `pipe.step(...)` does nothing.**
`torch.autocast` state is *thread-local*, and each stage's forward runs in its
own worker thread — the context you enter in the caller thread never reaches
them. (Worse, `overlap=False` runs in the caller thread, so it *would* apply
there — a silent numerics divergence between the two modes.)

Mixed precision is therefore a pipeline option:

```python
pipe = Pipeline(stage_modules=[s0, s1], devices=devices,
                autocast=torch.bfloat16)          # or "bf16"
pipe = PipelineModel(model, example_input, devices=devices, autocast="bf16")
```

`autocast` accepts `None` (off, default), a `torch.dtype`, or `"bf16"` /
`"fp16"`. Each stage then enters `torch.autocast(device_type, dtype)` itself,
around exactly the right ops:

- **Forward and loss computation** run under autocast (standard AMP practice).
- **Backward (`torch.autograd.grad`) runs outside** — autograd replays with the
  dtypes recorded during the forward, as PyTorch recommends.
- **Parameters stay fp32 masters**; autocast casts weights per-op, so gradients
  come back fp32 and the accumulators / `flush_grads()` / your optimizer are
  unchanged. Boundary activations relayed between stages are simply bf16
  tensors (the tuple/no-grad machinery handles them like any float tensor).

Applies identically to `step()`, `infer()`, and `PipelineModel.forward()`.

**fp16 is inference-only.** fp16 training requires gradient (loss) scaling,
which the pipeline does not integrate — `step()` raises a `ValueError` when
`autocast` is fp16. Use **bf16** for training (no scaler needed, and the
recommended dtype on Ampere+ GPUs); fp16 remains available for `infer()`.

Verified in `examples/amp_check.py`: bf16 pipeline training is **bit-identical
(0.0)** to sequential microbatch grad-accum under the same autocast — losses,
gradients, and final weights after multi-step SGD — for all schedules, and the
fp32 (`autocast=None`) path is unchanged.

---

## Bypassing `loss_fn`: backprop a gradient directly (`grad_outputs=`)

Normally the last stage computes a scalar loss via `loss_fn(output, target)` and
backprops it. The `grad_outputs` escape hatch lets you **skip the loss entirely**
and feed a precomputed `dL/dOutput` straight into the last stage's backward.
This is useful when the gradient comes from somewhere the pipeline can't see —
a downstream model, a custom differentiator, RL advantages, or a loss the last
stage doesn't own.

```python
# Callable form: resolved per-microbatch on the last-stage worker (mirrors
# loss_fn). Receives the live output so the grad can depend on it.
res = pipe.step(x, targets=y, n_microbatches=4,
                grad_outputs=lambda out, tgt: 2.0 * (out - tgt))

# Tensor form: a full-batch gradient, chunked along dim 0 exactly like
# `targets`. Use this when you already have dL/dOut computed elsewhere.
res = pipe.step(x, targets=y, n_microbatches=4, grad_outputs=full_batch_grad)

res.flush_grads(); opt.step(); opt.zero_grad()
```

- **Mutually exclusive with `loss_fn`** — passing both raises `ValueError`.
- **No loss is reported**: bypassing means no scalar loss is ever computed, so
  `result.loss` raises a clear `RuntimeError` and `result.losses` is empty.
  `flush_grads()` still mean-scales correctly (it infers the microbatch count
  from the outputs, not the losses).
- For a **tuple-output last stage**, pass a tuple of grads aligned to the module
  outputs (`None` at no-grad slots), matching the `out_no_grad` mask — only the
  grad-needing outputs enter autograd.
- Works with every schedule and `overlap` on/off.

Verified in `examples/grad_bypass_check.py`: pipeline bypass training is
**bit-identical (0.0)** to a sequential manual `out.backward(grad)` baseline —
final weights after multi-step SGD — for both the callable and tensor forms,
across all schedules and overlap modes.

---

## Schedules

| Schedule | Bubble | Peak in-flight activations | Notes |
|---|---|---|---|
| `staggered_1b1f` (default) | **lowest** | ~`num_stages` | Backward-eager + staggered warmup. **Recommended.** |
| `gpipe` | fill + drain | all `n_microbatches` | Simple, highest memory; correctness baseline |
| `1f1b` | steady-state | ~`num_stages` | Textbook forward-first. **Educational only** — see below |

`1f1b` (forward-first) is kept purely for comparison: it computes the *same*
math as `staggered_1b1f` but forwards before backwarding, leaving a large
steady-state bubble (~50% GPU util vs ~92-98% on a 10.9 GB model). It exists to
make the importance of execution order concrete. Explore schedules with the
simulator:

```bash
python -m ramtorch.schedule_simulator --p 8 --m 16 --plot gantt.png
```

---

## Resident PipeDream-2BW training

`Pipeline.train_session(...)` creates a `PipeDream2BWTrainer` with persistent
stage workers and stage-local updates. It is a **separate, one-update-stale
training algorithm**, not another `Pipeline.step` schedule. Existing synchronous
`step` / `flush_grads` / external-optimizer training and inference are unchanged;
2BW weights are **not** expected to match synchronous training.

### Update semantics and two weight banks

Let \(w_0\) be the initial weights, \(g\) the zero-based update group,
\(s_g\) the current optimizer state, and \(p\) the pipeline depth. Set
\(m =\) `n_microbatches`, with **\(m \ge p\)**. All microbatches in group \(g\)
evaluate at \(w_{\max(g-1,0)}\), using that same version for forward and backward.
Accumulate gradients in ascending microbatch order and scale **once** by \(1/m\):

\[
G_g = \frac{1}{m}\sum_{j=0}^{m-1}
\nabla_w \ell_{g,j}\!\left(w_{\max(g-1,0)}\right),\qquad
(w_{g+1}, s_{g+1}) = \operatorname{Optimizer}(w_g, s_g, G_g).
\]

The bootstrap is intentional: **groups 0 and 1 both evaluate \(w_0\)**, but
apply their gradients to \(w_0\) and \(w_1\), respectively. The optimizer always
uses the latest weights and its current state, never a rewound momentum or
Adam state. Each stage holds **two weight banks but only one optimizer/state
set**. A bank remains intact until its live backward graphs retire.

### API and ownership

```python
import torch
from torch import nn
from torch.nn import functional as F
from ramtorch import Pipeline

# Fresh, deterministic, buffer-free resident stages; use ["cpu", "cpu"] on CPU.
pipe = Pipeline(
    stage_modules=[nn.Sequential(nn.Linear(16, 32), nn.GELU()), nn.Linear(32, 16)],
    devices=["cuda:0", "cuda:1"], offload=False,
    # autocast=torch.bfloat16,  # optional; FP32 parameters, no GradScaler
)
factory = lambda params: torch.optim.AdamW(
    params, lr=1e-3, foreach=False, fused=False)

# train_loader yields at least 10 full (inputs, targets) batches, e.g. [32, 16].
loader = iter(train_loader)
with pipe.train_session(optimizer_factory=factory, n_microbatches=4,
                        loss_fn=F.mse_loss, schedule="pipedream_2bw",
                        max_inflight=10) as trainer:
    result = trainer.run(loader, updates=8)
    result = trainer.run(loader, updates=2)  # continues history; total_updates == 10
# close() publishes the latest weights to the original stage modules.
pipe.close()
```

- `optimizer_factory(params)` must create a fresh **standard `torch.optim.SGD`,
  `torch.optim.Adam`, `torch.optim.AdamW`, or RamTorch `AdamEF`/`Lion`**, owning
  exactly those stage parameters. Pass one
  factory for all stages or a list of one factory per stage. For example, use
  `lambda params: torch.optim.SGD(params, lr=0.01, momentum=0.9, foreach=False,
  fused=False)` for momentum SGD. Custom subclasses and differentiable/capturable
  optimizer modes are unsupported. The trainer owns stepping and gradient
  clearing: **no external optimizer, `flush_grads()`, or parameter mutation**
  while the session owns the pipeline.
- `run(loader, updates=N)` consumes exactly `N` full batches and performs `N`
  updates per stage. Each finite run has **one fill and one drain**, not one
  per update group; repeated one-update runs give up cross-group overlap.
  Further runs preserve weight versions, optimizer state, and global group /
  microbatch counters. Each call uses `iter(loader)`; retain an iterator as
  above to continue consuming data instead of restarting a reiterable loader.
  The result contains `updates`, `total_updates`, and `peak_inflight`, not
  retained outputs or a loss history.
- **Caller-serialize the entire lifecycle**, including creation, `run`, `close`,
  and other pipeline entry calls. Close any asynchronous inference session with
  `pipe.close()` first, even if idle. Do not call `step`, inference, or mutate
  modules while a trainer is open. Context exit / `trainer.close()` joins its
  workers and exposes the latest weights. A failed run poisons the session:
  close it; partial updates are **not rolled back**.
- Start with **clean stages**: no cached graphs, parameter `.grad`, or prior
  `stage.grad_acc`. Ordinary `flush_grads()` / `zero_grad()` can leave allocated,
  zero-filled stage accumulators, which still fail this check; prefer a fresh
  `Pipeline` rather than reusing one previously trained through `step()`.
- Initial support is **manual, fully resident, deterministic, buffer-free**
  `stage_modules`, or `chunk_modules` with **`offload=False`** (chunks otherwise
  default to offload). Every stage needs trainable parameters. No auto-traced
  `PipelineModel`, weight/activation offload, shared/tied parameters, dropout /
  stochastic or stateful forwards, activation checkpointing, or fp16 training.
  Use FP32 or BF16 autocast. Forward-mutating embedding `max_norm` and internal
  attention/RNN dropout are rejected; custom forwards must obey the same
  no-mutation/no-randomness restrictions. Stages must all be CUDA-resident
  (or all CPU for tests); mixed CPU/CUDA stages are not supported.

### CPU optimizer state and fused updates (experimental)

`optimizer_device="cpu"` leaves **both compute weight banks on each GPU** but
creates a separate CPU master parameter set for the optimizer. Forward/backward
remain GPU-resident; this is optimizer offload, not weight/activation offload.
The factory receives these CPU masters instead of the module's GPU Parameters:

```python
factory = lambda params: torch.optim.AdamW(
    params, lr=1e-3, foreach=False, fused=True)
with pipe.train_session(optimizer_factory=factory, optimizer_device="cpu",
                        n_microbatches=4, loss_fn=F.mse_loss) as trainer:
    trainer.run(train_loader, updates=12)
```

- Masters, gradient transfer buffers, Adam moments, and step counters are on
  CPU; the **update math executes on CPU** using PyTorch's native fused kernel.
  CUDA stages use pinned CPU masters and gradient buffers. Moments do not move
  between devices. `optimizer_device=None` preserves the GPU optimizer default.
- Each local update scales the accumulated CUDA gradient once, copies it D2H,
  waits for a **stage-stream event**, runs the CPU optimizer, then uploads the
  updated master into the retired GPU bank. The live old bank is not modified.
  Uploads and subsequent compute use the same stage stream; the next update's
  D2H event also fences CPU-master reuse after the preceding H2D read.
- There is no global update barrier, but a stage's worker waits for its own
  transfer and CPU update. Other stages may proceed. This implementation does
  **not** bucket or overlap a stage's CPU update with that same stage's compute.
- CPU storage adds a master copy and persistent gradient buffers alongside Adam
  state. GPU storage no longer contains Adam moments. Choose CPU thread counts
  deliberately: multiple stage workers can enter multithreaded CPU kernels at
  once. The library does not change the application's thread settings.
- CPU fused versus GPU non-fused optimizer rounding is not guaranteed identical.
  Correctness comparisons must use the same optimizer implementation/device in
  the independent oracle. Profiling spans separate `optimizer_d2h`,
  `optimizer_d2h_wait`, `optimizer_cpu`, and `optimizer_h2d` from F/B operations.

For the larger/deeper paired experiment, both variants use **the same 2BW
schedule** (GPU non-fused AdamW versus CPU fused AdamW), not synchronous 1F1B:

```bash
PYTHONPATH=. python -u examples/pipedream_2bw_demo.py --compare-cpu-optimizer \
    --devices cuda:0,cuda:1,cuda:2,cuda:3 --dim 4096 --layers 16 \
    --microbatches 4 --batch-size 128 --updates 12 --optimizer adamw \
    --lr 0.001 --bf16 --cpu-threads 6 \
    --output-dir scratchpad/pipedream_2bw/runs/cpu_fused_4096_16layers
```

Replace `--compare-cpu-optimizer` with **`--compare-all`** to also run synchronous
`staggered_1b1f` with GPU AdamW and CPU-fused AdamW. The synchronous CPU baseline
uses a single resident GPU weight copy, drains each batch's forward/backward,
then runs per-stage CPU updates concurrently. Each stage explicitly fences
D2H before CPU access and completes H2D before the next full batch. It uses
fresh gradients, not the delayed-gradient 2BW rule. This adapter is a demo
comparison path, not a change to the `Pipeline.step()` API.

The demo captures separate fresh-model clean timings and profiles, writes gzip
traces and a ZIP bundle, and records CPU optimizer placement/storage separately
from CUDA allocator usage. This compares a placement **and kernel** change;
it does not isolate CPU-vs-GPU hardware with an identical optimizer kernel.

Focused CPU-fused correctness checks (both FP32 and BF16, frozen/unused/zero
gradients, continuation, no-observer runs, and transfer-storage lifetimes):

```bash
PYTHONPATH=. python examples/pipedream_2bw_check.py --cpu-optimizer-only \
    --devices cuda:0,cuda:1,cuda:2,cuda:3
```

The reference also accepts `--optimizer-device cpu --fused`; the checker's
`--long-run` mode accepts the same flags for custom longer comparisons.

**Measured 2026-09-20:** four RTX PRO 4000 Blackwell GPUs, EPYC 7352 CPU,
PyTorch 2.8.0/CUDA 12.8, six intra-op CPU threads. At width 4096, 16 blocks,
BF16, four microbatches of 128, and 12 updates, clean elapsed time was **3.683 s
GPU optimizer vs 12.280 s CPU fused** (0.300x throughput). Peak stage-0 CUDA
allocation fell from **15.45 GiB to 11.52 GiB**. CPU storage for the whole model
was about **8 GiB masters + 8 GiB retained gradients + 16 GiB Adam state**;
these are logical tensor bytes, not process RSS. Each update transfers about
8 GiB of gradients D2H and 8 GiB of updated weights H2D across all four stages.

The paired gzip traces and ZIP are in
`scratchpad/pipedream_2bw/runs/cpu_fused_4096_16layers/`. Both traces have real
CUDA kernels on all four devices; the CPU trace has 768 D2H and 864 H2D copies
(the latter includes input/target loading). Each stage has 48 F, 48 B, 12 U and
12 of each CPU-optimizer span; weight-version ordering and next-group forwards
before the previous update were verified. Small-model FP32/BF16 checks matched
the independent CPU-fused oracle byte-for-byte, including no-observer runs.
This is an initial stage-local blocking implementation, not a claim about an
optimized bucketed/overlapped CPU optimizer. Timings include first-step state
allocation; profiler spans are not clean isolated kernel timings.

A subsequent matched **four-way `--compare-all` run** on the same workload
measured 2BW / synchronous 1F1B at **3.687 / 4.447 s with GPU AdamW**, and
**12.439 / 14.371 s with CPU-fused AdamW**. That is 1.206x and 1.155x 2BW
throughput respectively. Each is one clean timing sample; all four modes were
rerun together rather than mixing measurements from separate experiments.
All four profiles and metadata are in
`scratchpad/pipedream_2bw/runs/four_way_4096_16layers/`. The synchronous CPU path
also passed small-model FP32/BF16 bytewise checks against an independent
**fresh-gradient** CPU-fused oracle (`--sync-cpu-optimizer-only`), including
continued calls and single-bank storage. 1F1B versus 2BW is not weight parity:
the former uses fresh gradients, the latter one-update-delayed gradients.

### MNIST convergence and update-level error feedback

`examples/mnist_pipedream_2bw.py` compares **Adam and Lion**, each with
synchronous `staggered_1b1f`, ordinary delayed 2BW, and delayed 2BW + EF:

```bash
python examples/mnist_pipedream_2bw.py \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 --epochs 5 \
  --dim 256 --blocks 4 --batch-size 256 --microbatches 4 \
  --profile-warmup 3 --profile-steps 3
```

The example requires `torchvision` (MNIST); `matplotlib` is optional for the PNG.
Outputs include `metrics.json`, `curves.csv`, `convergence.png` when available,
one gzip trace per variant, and `mnist_convergence_bundle.zip`. Use
`--seeds 0,1,2` for repeated convergence runs, `--bf16` for BF16 autocast with
FP32 parameters, or `--profile-steps 0` to disable profiling. Traces use only the
first seed and a **separate fresh-model replay** after warmup, not the full
training run. These short captures include their own fill/drain and are not
steady-state throughput measurements.

The fixed split holds 5,000 training-set examples out for validation, trains on
55,000 with `drop_last=True`, and evaluates the untouched 10,000-example test
set at the end. All variants share initialization, epoch permutations, effective
batch size, and optimizer-specific hyperparameters. Parameters and optimizer
state remain on stage devices. The session persists between epochs; validation
uses a separate latest-weight replica after a drain, without resetting the 2BW
history. No clipping or LR schedule is used; this is not a reproduction of the
paper's LLM recipes. A single seed is a demo, not statistical evidence EF helps.

`ramtorch.AdamEF` and `ramtorch.Lion` implement **update-level**, not raw-gradient,
error feedback. For the complete uncorrected optimizer update `u`, they apply
`x <- x - u - c * (u - previous_update)`. The first completed optimizer call uses
an ordinary update; correction begins on the second. This maps the paper's
initial no-op and bootstrap onto the runtime's count of actual updates: no extra
no-op is inserted into the 2BW schedule. Each gradient advances moments exactly
once. The saved buffer holds the **uncorrected update**, not the corrected
parameter displacement or a subtraction of rounded before/after weights.

Both classes accept `ef_coefficient` (`AdamEF` defaults to 1, `Lion` to 0).
The example uses the same class with `c=0` for plain controls and `c=1` for EF.
`AdamEF` defaults to coupled L2 decay, i.e. **Adam, not AdamW**; setting
`decoupled_weight_decay=True` gives AdamW-style decay. Lion uses decoupled decay.
Decoupled decay is computed at the current/latest weights and included in `u`.
The saved update includes that step's learning rate; changing LR does not
rescale historical updates. Missing gradients skip parameter/moment updates and
zero any existing update history for that logical call, rather than replaying an
older update. Explicit zero gradients still perform normal optimizer updates.

EF adds **one persistent parameter-sized history tensor** when `c>0`; `c=0`
allocates none. Thus relative to ordinary synchronous training, 2BW+EF adds one
compute-weight bank and one update-history state, besides existing moments,
gradients, activation storage and transient update tensors. `metrics.json`
reports logical state bytes rather than implying a peak-memory benchmark.
The new optimizers accept dense real FP32/FP64 parameters; mixed precision here
means BF16 autocast, not BF16 optimizer parameters. They do not integrate fused,
capturable or GradScaler skip semantics. Optimizer `state_dict()` restoration
includes history and startup counters; this is **not** a full 2BW-session
checkpoint API (which would also need both weight banks and version metadata).

Validate the equations and pipeline/reference parity with:

```bash
PYTHONPATH=. python examples/pipedream_2bw_optimizer_check.py --devices cpu,cpu --bf16
PYTHONPATH=. python examples/pipedream_2bw_optimizer_check.py --devices cuda:0,cuda:1,cuda:2,cuda:3 --bf16
```

A five-epoch seed-0 BF16 run on four RTX PRO 4000 Blackwell GPUs (width 256,
four blocks, batch 256, four microbatches) completed 1,070 updates per variant.
Final test accuracy, in synchronous / 2BW / 2BW+EF order: **Adam 97.78% /
97.87% / 97.48%**, **Lion 97.51% / 97.48% / 97.61%**. Adam EF improved the
early validation loss (epoch 1: 0.1762 delayed -> 0.1249 EF, versus 0.1145 sync)
but not final test accuracy. Do not generalize these single-seed differences.
Results and six three-update gzip traces are in
`scratchpad/pipedream_2bw/runs/mnist_ef_bf16/`. Logical EF history is 9,223,208
bytes across this model; plain variants allocate zero history bytes.

Method source: [One-Step Gradient Delay is Not a Barrier for Large-Scale
Asynchronous Pipeline Parallel LLM Pretraining](https://arxiv.org/abs/2606.30634v1),
using the supplied Algorithm 1/2 description. No claim of paper-code reproduction
or universal convergence improvement is made.

### Loader, loss, and bounded overlap

The loader must yield `(inputs, targets)`; use `(inputs, None)` for a loss that
needs no targets. Inputs and non-`None` targets are batched tensors or flat
tuples of batched tensors. Batch dimensions must agree, be at least `m`, and
split evenly into `m` equal-size microbatches. There is **no padding, partial
update group, or nested pre-diced input** in this API. An exhausted loader
before `updates` groups is an error. Do not reuse/mutate yielded storage while
asynchronous consumers may still be copying it.

`loss_fn(output, target)` runs on the last stage, receiving `None` when targets
are absent, and must return a **scalar microbatch-mean loss**. The trainer
averages accumulated gradients across the `m` microbatches; do not pre-divide
the loss by `m`. Tuple stage outputs and forward-only `out_no_grad` masks are
supported, but the session has no `grad_outputs=` bypass.

`max_inflight` counts admitted microbatches and must be at least the pipeline
depth (default `2 * m + depth`). Credits return only after stage-0 backward
retirement, including its CUDA completion event, not merely host enqueueing.
This bounds the runtime's live microbatch window, not memory held by the loader
or debug snapshots. Each **producing stage** transfers its outgoing activation /
input-gradient boundary tensors to the next consumer device, with event-ordered
handoffs. Workers update locally: **no global synchronization per group**.
Backpressure may wait on an individual retirement event; each run drains its
participating streams before returning.

### Correctness checks and profiling

`trainer.run(..., observer=callback)` calls `callback(stage, group, snapshot)`
with detached CPU copies of named `grads` (mean-scaled, before optimizer weight
decay), post-update `weights`, and post-update `optimizer` state. These copies
**synchronize CUDA and are a correctness/debug path, never a benchmark path**.
The independent reference uses copied historical models, not the pipeline
scheduler; exact comparisons are for the same device, dtype, and environment,
not CPU-versus-CUDA bit identity.

`trainer.run(..., profile_path="run.json.gz")` captures a bounded run as a
compressed Kineto trace; create the parent directory first. CPU spans include
stage/group/version/slot annotations, loading, transfers, waits, weight copies,
and optimizer work. They describe host execution/enqueueing, not GPU execution:
real GPU kernel events require working CUDA/Kineto/CUPTI.

Run from the repository root (substitute your environment's Python):

```bash
PYTHONPATH=. python examples/pipedream_2bw_reference.py --device cpu --self-test
PYTHONPATH=. python examples/pipedream_2bw_reference.py --device cuda:0 --optimizer adamw --bf16 --updates 8
PYTHONPATH=. python examples/pipedream_2bw_check.py --devices cpu,cpu
PYTHONPATH=. python examples/pipedream_2bw_check.py --devices cuda:0,cuda:1
PYTHONPATH=. python examples/pipedream_2bw_demo.py --devices cuda:0,cuda:1 \
    --dim 512 --layers 8 --microbatches 8 --updates 16 --batch-size 32 \
    --optimizer adamw --bf16 --output-dir scratchpad/pipedream_2bw/demo
```

For a longer **four-GPU vs single-GPU delayed-gradient** comparison at the
original workload (width 1024, eight residual MLP blocks, four microbatches of
128 samples), use the bounded-snapshot mode:

```bash
PYTHONPATH=. python -u examples/pipedream_2bw_check.py --long-run \
    --devices cuda:0,cuda:1,cuda:2,cuda:3 --dim 1024 --layers 8 \
    --microbatches 4 --batch-size 128 --updates 100 --optimizer adamw \
    --lr 0.001 --bf16 --report scratchpad/pipedream_2bw/runs/long_exact/bf16_100.json
```

Omit `--bf16` for FP32. The oracle executes the complete unsplit model on the
first listed device, while the pipeline performs one continuous `run()` across
all listed devices. Every update compares named scaled gradients, post-update
weights, and optimizer state directly with zero tolerance **and byte equality**
(including signed zero), rejecting nonfinite tensors. Snapshots are checked and
released through bounded per-stage queues instead of retaining all steps. This
adds debug synchronization/backpressure and shares the oracle's GPU with stage
0: **do not use this run's wall time as throughput**. Input fixtures are still
pre-generated on CPU; their memory grows with the requested update count.

To run just the independent single-GPU reference at the same workload:

```bash
PYTHONPATH=. python -u examples/pipedream_2bw_reference.py --device cuda:0 \
    --dim 1024 --layers 8 --microbatches 4 --batch-size 128 --updates 100 \
    --optimizer adamw --lr 0.001 --bf16 --stream
```

`--stream` prints each reference update without retaining the tensor trajectory;
it is not itself a pipeline parity test. The reference uses two ordinary models
and a bounded deque of copied historical states, without runtime bank swapping.

The CUDA-only demo compares 2BW with synchronous `staggered_1b1f`, with separate
warmups, fresh clean-timing models, and separate fresh-model profiles; no
observer or profiling overhead enters reported speedup. Loading, transfers,
optimizer updates, and the final drain count toward timing. `--batch-size` is
**per microbatch**. It writes `pipedream_2bw.json.gz`, `staggered_1b1f.json.gz`,
`metadata.json`, and a compressed `pipedream_2bw_bundle.zip`; missing real GPU
kernel events on any requested device make the demo fail after saving the
diagnostic bundle. This is a throughput comparison of **different update
semantics**, not a final-weight parity test.

**Validation (2026-09-20):** the complete per-update exact suite passed on four
RTX PRO 4000 Blackwell GPUs with PyTorch 2.8.0 / CUDA 12.8, including BF16,
SGD/momentum, AdamW, tuple relays, continuation, minimum-capacity admission,
and failure cleanup. A separate no-observer run also matched final weights and
optimizer state exactly; debug snapshot synchronization is not required for
correctness. Existing resident, tuple/autocast, and streaming-inference checks
passed. The longer base-workload comparison (width 1024, eight blocks, four
microbatches of 128 samples, AdamW at lr=0.001) also passed **100 updates / 400
microbatches separately in BF16 and FP32**, comparing all gradients, weights,
and optimizer state byte-for-byte at every update against the unsplit single-GPU
oracle. Reports are in `scratchpad/pipedream_2bw/runs/long_exact/` as
`bf16_100.json` and `fp32_100.json`. These results are not a claim of bit identity
across precisions or software releases.

The example capture (8 residual blocks, width 1024, 128 samples per microbatch,
4 microbatches/update, 12 updates, BF16/AdamW) contains real CUDA kernels on all
four devices and 48 F / 48 B / 12 U operations per stage. Version ordering and
next-group forwards before the preceding update were checked in the trace.
Clean timing in the final capture run was 0.556 s for 2BW versus 0.864 s for
synchronous 1F1B (1.55x throughput); this is a workload-specific observation,
not a guarantee.

---

## Streaming stage weights from CPU RAM (pipeline + offload)

When a stage's shard doesn't fit in its GPU's memory, combine the pipeline
with the weight-streaming engine: make that stage's `stage_modules` entry a
**list of chunk modules** instead of a single module, and it becomes an
`OffloadStage` — masters (weights, grad accumulators, optimizer state) live in
CPU pinned RAM and stream through a small GPU window:

```python
stage0 = [Block() for _ in range(12)]        # list  -> offloaded stage
stage1 = nn.Sequential(*[Block() for _ in range(12)])  # module -> resident
pipe = Pipeline(
    stage_modules=[stage0, stage1], devices=["cuda:0", "cuda:1"],
    offload_window=2,                # streamed GPU slots per offloaded stage
    offload_pin=0,                   # chunks pinned permanently on the GPU
    offload_keep_activations=True,   # or "checkpoint" (recompute memory)
    offload_grad_accum="stream",     # default: accumulate grads ON the GPU
    offload_acc_slots=None,          # GPU accumulator slots (default: window)
    offload_activations=False,       # stream saved activations to CPU RAM
    offload_act_slots=2,             # resident activation packets per stage
)
res = pipe.step(x, targets=y, schedule="staggered_1b1f",
                n_microbatches=8, loss_fn=F.cross_entropy)
res.flush_grads()   # streamed grads -> CPU .grad; residency invalidated
opt.step()          # AdamW(fused=True) over all stages' params
pipe.close()        # stops loader/writeback threads (also runs on __del__)
```

Or skip the per-stage nesting entirely — hand the pipeline ONE flat list of
chunks via `chunk_modules=` and let it split them across the devices (see
"Manual splitting with chunks" above; `offload=True` is the default there, so
`Pipeline(chunk_modules=chunks, devices=[...], offload_window=2)` is the whole
streamed setup).

**GPU weight memory per offloaded stage ≈ `(window + pin)` chunks** instead of
the whole shard (plus `acc_slots` chunk-sized grad accumulators during
training). With `offload_grad_accum="stream"` (default) each streamed chunk's
grad accumulator lives on the GPU and spills/reloads over the copy streams
like a weight — zero CPU arithmetic; when `acc_slots` covers the streamed
chunks, grads cross PCIe once per step at `flush_grads()`. The legacy
`offload_grad_accum="cpu"` ships every microbatch's grads D2H and adds them
into pinned CPU buffers on the writeback thread — per-microbatch PCIe traffic
plus serial host math, which stalls compute-bound configs. Chunks follow the `OffloadModel` dicing convention (chunk
`i+1` consumes chunk `i`'s output, tuples fine); the stage's own input/output
contract is unchanged, so mixing offloaded and resident stages, tuple stage
boundaries, `autocast=`, `grad_outputs=`, and `infer()` all work as usual.

### Changing `offload_pin` at runtime

The pinned/streamed split is reassignable between steps, per stage or across
the whole pipeline:

```python
pipe.set_offload_pinned(3, optimizers=[opt])                 # every offloaded stage
pipe.set_offload_pinned([3, 2, None, 0], optimizers=[opt])   # aligned with pipe.stages
pipe.set_offload_pinned({1: [0, 4]}, optimizers=[opt])       # by stage index
pipe.stages[0].set_pinned(2, optimizers=[opt])               # or drive one stage
```

An `int` is a count spread evenly (like `offload_pin=`); a `set` of indices
broadcasts; a `list`/`tuple` is **always** per stage (so `[3, 2]` means "stage
0 pins 3, stage 1 pins 2" — use `{0, 4}` or `[[0, 4], [0, 4]]` to broadcast
explicit indices). Plain resident stages are skipped by the broadcast forms
and rejected when named explicitly. Returns `{stage_index: summary}`.

Same contract as the single-GPU engine (see `offload.md`, "Changing the tier
at runtime"): it is a **hard reset** — grad accumulators, `.grad` and
activation packets are discarded, only masters and optimizer state carry over.
Two pipeline-specific notes:

- **Only between clean steps.** `step()` joins its stage workers before
  returning, so right after `flush_grads()` + `opt.step()` is the moment. After
  an *aborted* step the stale itinerary makes the retier refuse; recover with
  `OffloadStage.clear()` (or `force=True`, discarding that step's state).
- **Pass `optimizers=`.** The documented pipeline recipe builds the optimizer
  from `itertools.chain(*(st.params for st in pipe.stages))` once, long before
  any retier. Those `Parameter` objects survive the move (only `.data`
  relocates), but their optimizer state has to be migrated with them or
  `AdamW(fused=True)` will hit a device mismatch.

### Why this composes well

The relay executor walks a **static per-stage op list**, so the entire step's
chunk-touch order is known before it runs: each `F` op touches chunks
`0..L-1`, each `B` op `L-1..0`. `Pipeline.step` announces the whole expanded
itinerary to each stage's loader up front, and the prefetcher overlaps H2D
weight copies with compute *and with the pipeline bubbles*. The
`staggered_1b1f` steady state (`... B F B F ...`) gets echo reuse for free —
the chunk where a backward ends is the chunk where the next forward starts.

### When it pays off

Streaming adds PCIe traffic (weights H2D per touch, grads D2H once per
microbatch), so it only comes for free when compute hides it. Explore your
configuration first with the combined simulator:

```bash
python -m ramtorch.pipeline_offload_simulator --p 4 --m 8 --chunks 8 \
    --window 2 --tf 1 --tb 2 --th2d 0.5 --plot gantt.png
```

Compute-bound (real transformers at useful batch sizes): a window of 2 of 8
chunks measured **+0.4%** makespan in the simulator. Transfer-bound (small
compute per weight byte — e.g. the deliberately-wide MNIST MLP in
`examples/mnist_pipeline_offload.py`): several times slower than
full-resident, at 4.5x less GPU memory; `offload_pin` trades memory back for
traffic when the shard *almost* fits.

### Rules and gotchas

- **Backward strategy**: `offload_keep_activations=True` (keep per-chunk
  graphs; plain-pipeline-like activation memory) or `"checkpoint"`
  (non-reentrant per-chunk checkpoint; recompute-level memory, dropout-safe).
  The engine's own recompute mode (`False`) is rejected — its no-grad forward
  would leave the last stage's loss graph-disconnected at the relay's W op.
- **Activation offload** (`offload_activations=True`): each (microbatch,
  chunk) forward's saved activations become a `saved_tensors_hooks` packet
  that streams to pinned CPU RAM under slot pressure and reloads one backward
  ahead of its use (lazy policy, Belady eviction by the stage's announced op
  schedule — the same signals the weight window uses). This matters more here
  than on a single GPU: a pipeline keeps up to `p` microbatches' activations
  in flight per stage, and the packet cap replaces that `m x chunks` residency
  with `offload_act_slots` packets. Bit-exact (asserted across schedules x
  modes x slot counts in `examples/pipeline_offload_check.py`); packets never
  touch NVMe. Pair it with `offload_keep_activations=True` — checkpoint-mode
  packets hold only chunk boundaries, which the stage's backward cache keeps
  resident anyway.
- **No NVMe tier**, deliberately: sustained pipeline training from disk would
  rewrite every stage's masters every step — guaranteed drive thrashing. Use
  `offload.md`'s single-GPU engine if you truly need it (it is consent-gated).
- The engine ctor **relocates params in place** (streamed → CPU pinned,
  pinned → GPU). Deepcopy any reference copies *before* building the
  `Pipeline`.
- Buffer mutations (BatchNorm running stats) are not written back — use
  buffer-free norms (LayerNorm).
- `fake_compute` is not supported with chunked (offloaded) stage entries.
- Bit-parity with a full-resident pipeline (same op order) is asserted across
  schedules × modes × windows × tuple boundaries × bf16 × grad-bypass ×
  runtime retiering in `examples/pipeline_offload_check.py`.

---

## Inference

`Pipeline.infer` runs a **forward-only GPipe** (no backward, no grad, no
activation retention) with one worker thread per stage, so stage s+1 computes
microbatch k while stage s computes k+1 — every GPU stays busy:

```python
logits = pipe.infer(images, n_microbatches=4)
```

**Inference costs no gradient state.** Every stage keeps explicit gradient
accumulators (a plain stage: one full copy of its shard on its GPU; an
offloaded stage: one param-sized buffer per chunk, GPU for pinned chunks and
pinned host RAM for streamed ones), and they are allocated on that stage's
**first backward** rather than at construction. So an inference-only pipeline
holds its weights and nothing else — for a plain pipeline that halves the
per-GPU footprint versus eager allocation. Consequence: `flush_grads()` on a
pipeline that never ran `step()` leaves `.grad` as `None` instead of writing
zeros.

To size an inference deployment, `examples/offload_inference_memory.py
--pipeline cuda:0,cuda:1` sweeps `offload_pin` across the whole range with
[`set_offload_pinned`](#changing-offload_pin-at-runtime) and reports per-stage
GPU bytes at rest and at peak plus the latency at each point.

### Streaming inference across loop iterations

`infer()` is a **full barrier**: it joins its worker threads and synchronizes
every device before returning. In an iterative inference loop where each call
feeds the next — diffusion denoising is the canonical case — that drains the
pipeline between iterations: stage 0 idles while the later stages finish the
tail, then the pipeline refills. The bubble costs ~(p−1) microbatch-forwards
per iteration on every stage.

When microbatches are **independent** (per-sample denoising: microbatch *i*'s
next step needs only microbatch *i*'s last output), the streaming API keeps
one persistent worker per stage and flows the next iteration in right behind
the previous one — no drain, no refill:

```python
# Fully automatic: update_fn(out_mb, mb_index, step_index) -> next_input_mb
# runs per microbatch the moment that microbatch finishes each round.
x_final = pipe.infer_loop(x0, steps=50, n_microbatches=8,
                          update_fn=lambda out, i, t: scheduler_step(out, i, t))

# Or drive it by hand with handles:
h = pipe.infer_submit(x0, n_microbatches=8)       # non-blocking launch
for t in range(steps - 1):
    nxt = pipe.infer_open(n_microbatches=8)        # trickle-fed batch
    for i in range(8):
        out_i = h.wait_mb(i)                       # block until mb i is done
        nxt.submit_mb(i, scheduler_step(out_i, i, t))
    h = nxt
x_final = h.result()
```

- `infer_submit(data)` takes the same input forms as `infer()` and returns an
  `InferBatch` immediately; `wait_mb(i)` streams per-microbatch outputs back
  (they complete in order — the inter-stage handoffs are FIFO), `result()`
  blocks for the whole batch and applies `infer()`'s shape convention.
- `infer_open(m)` + `submit_mb(i, value)` is the fully-streaming form: a
  microbatch enters stage 0 the instant it's submitted, without waiting for
  the rest of its batch.
- **The update must be per-microbatch independent.** Anything batch-global
  (e.g. normalizing across microbatches) is incompatible with the overlap.
  Keep `update_fn` cheap: it runs serially on the caller thread and the
  pipeline idles while it runs.
- Works with offloaded stages too — each stage's worker drives the engine's
  streamed forward serially, the same access pattern as `infer()`.
- `infer()` is unchanged and remains the right call for one-shot batches;
  the streaming path exists for loops. `Pipeline.close()` stops the
  persistent workers.

Measured on `examples/pipeline_infer_stream_demo.py` (4 GPUs, 8 denoising
steps × 8 microbatches, toy per-sample scheduler): ~1.2–1.3× wall-clock over
barriered `infer()` per loop, bit-identical outputs. The win grows with
pipeline depth and step count (the bubble is per-iteration).

---

## Profiling & debugging

Both `step` and `infer` accept `trace_path` (op-level Perfetto spans) and
`profile_path` (full `torch.profiler` / kineto trace). Bound profiling to a few
steps so files stay small — see `examples/mnist_pipeline_big_transformer_manual.py`
(`--profile --profile-start N --profile-steps K`).

Open traces at <https://ui.perfetto.dev>.

---

## Numerics

Microbatch gradient accumulation is **mean-of-microbatch-means**, bit-identical
to sequential gradient accumulation. It differs from a single full-batch
backward only by normal fp32 reduction-order noise (the same as any
gradient-accumulation setup). With `n_microbatches=1` the pipeline is
bit-identical to a plain full-batch backward.

Validated in `examples/mnist_seq_vs_gradaccum.py`: the pipeline reproduces
sequential grad-accum final weights to **0.0** (bit-exact).

---

## Examples map

| File | What it shows |
|---|---|
| `mnist_pipeline_example.py` | `PipelineModel` quickstart (auto-split MLP) |
| `mnist_pipeline_big_transformer_manual.py` | Manual pre-partitioned transformer (the robust path); `--offload` streams each stage's chunks from CPU RAM (`--window`, `--pin`, `--offload-mode`) |
| `mnist_frozen_encoder_overlap.py` | Frozen "text encoder" inference feeding a trained model (tuple + pre-diced inputs + a bool padding mask crossing a stage boundary) |
| `tuple_no_grad_check.py` | Tuple outputs + per-arg no-grad flags — parity vs sequential masked baseline |
| `amp_check.py` | Mixed precision (`autocast=`) — bf16 bit-identity vs sequential accum, fp16 guard |
| `grad_bypass_check.py` | Grad-bypass (`grad_outputs=`) — backprop a supplied dL/dOutput; bit-identity vs sequential manual backward |
| `mnist_pipeline_vs_single.py` | Pipeline vs single-GPU loss/grad/weight parity |
| `mnist_seq_vs_gradaccum.py` | Pipeline vs sequential grad-accum (liability check) |
| `pipeline_easy_demo.py` | `PipelineModel` forward + train + eval |
| `pipedream_2bw_reference.py` | Independent copied-history stale-gradient oracle, scalar recurrence self-tests, SGD/AdamW, CPU/CUDA and optional BF16 |
| `pipedream_2bw_check.py` | Per-update 2BW gradient/weight/optimizer-state checks against the oracle; bank lifetime, schedule/bootstrap, continuation, bounded admission, and failure cleanup |
| `pipedream_2bw_demo.py` | CUDA 2BW vs synchronous throughput with separate clean timings/profiles, per-device kernel checks, metadata and compressed trace ZIP; not weight parity |
| `mnist_pipeline_offload.py` | Pipeline + weight streaming end to end: memory/traffic/stall report vs full-resident |
| `pipeline_offload_check.py` | Offloaded-stage bit-parity vs plain pipeline + sequential ref (schedules × modes × windows × tuples × bf16 × bypass) |
| `pipeline_infer_stream_demo.py` | Streaming inference in a toy denoising loop: `infer_loop` / handle API vs barriered `infer()` — timing + bit-identity |
| `pipeline_infer_stream_check.py` | Streaming-inference bit-parity vs sync `infer()` (submit/trickle/overlap/loop × tensor/tuple/nested × resident/offloaded) |
