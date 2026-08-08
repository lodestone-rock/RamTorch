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

**Rule of thumb:** start with `PipelineModel` for convenience. If `torch.export`
fails or misbehaves on your model (see the gotchas below), switch to
`Pipeline(stage_modules=...)` — it's the robust, recommended path for anything
non-trivial.

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

## Input features: tensor, tuple, and pre-diced microbatches

`Pipeline.step` and `Pipeline.infer` accept three input forms. **Tuple forms are
only supported on the manual `stage_modules` path** — the traced `PipelineModel`
path raises a clear `ValueError` (torch.export can't trace them).

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

## Inference

`Pipeline.infer` runs a **forward-only GPipe** (no backward, no grad, no
activation retention) with one worker thread per stage, so stage s+1 computes
microbatch k while stage s computes k+1 — every GPU stays busy:

```python
logits = pipe.infer(images, n_microbatches=4)
```

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
| `mnist_pipeline_big_transformer_manual.py` | Manual pre-partitioned transformer (the robust path) |
| `mnist_frozen_encoder_overlap.py` | Frozen "text encoder" inference feeding a trained model (tuple + pre-diced inputs + a bool padding mask crossing a stage boundary) |
| `tuple_no_grad_check.py` | Tuple outputs + per-arg no-grad flags — parity vs sequential masked baseline |
| `amp_check.py` | Mixed precision (`autocast=`) — bf16 bit-identity vs sequential accum, fp16 guard |
| `grad_bypass_check.py` | Grad-bypass (`grad_outputs=`) — backprop a supplied dL/dOutput; bit-identity vs sequential manual backward |
| `mnist_pipeline_vs_single.py` | Pipeline vs single-GPU loss/grad/weight parity |
| `mnist_seq_vs_gradaccum.py` | Pipeline vs sequential grad-accum (liability check) |
| `pipeline_easy_demo.py` | `PipelineModel` forward + train + eval |
