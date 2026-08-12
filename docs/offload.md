# Single-GPU CPU→GPU weight streaming (`OffloadModel`)

Run a model that is **too big for one GPU** by keeping its weights in CPU
pinned memory and streaming them through a small sliding GPU window — with a
few chunks optionally pinned resident. One GPU, one process, no `torchrun`.

This supersedes the per-layer stream-bouncing `ramtorch.Linear` /
`CPUBouncingLinear`, which is **deprecated**. `OffloadModel` dices by *chunk*
(you decide the granularity) instead of by individual `nn.Linear`, and adds a
real loader/writeback overlap engine.

> **Quickstart:** [`examples/offload_quickstart.py`](../examples/offload_quickstart.py)
> **End-to-end training:** [`examples/mnist_offload_example.py`](../examples/mnist_offload_example.py)
> **Full study:** [`examples/offload_vs_plain_demo.py`](../examples/offload_vs_plain_demo.py)
> **Design simulator:** `python -m ramtorch.offload_simulator --help`

---

## The idea in one paragraph

Weights live in CPU **pinned** memory. A **loader thread** prefetches upcoming
chunks into a GPU window of `window` slots over a dedicated H2D stream, so the
copy for chunk `i+1` overlaps the compute of chunk `i`. `pin` evenly-spaced
chunks stay on the GPU permanently (they never load, never evict, easing PCIe
traffic at the cost of their memory). Eviction is **farthest-next-use**
(Belady — optimal, because the chunk itinerary is known in advance). Backward
gradient writebacks return to pinned CPU accumulators over a D2H stream via a
separate **writeback thread**.

**Peak GPU weight memory ≈ `(window + pin)` chunks** — independent of how many
chunks the model has.

---

## Quickstart

You dice the model yourself into an ordered list of chunk modules — the same
convention as `Pipeline(stage_modules=...)`. Chunk `i+1` consumes chunk `i`'s
output — a single tensor, or a tuple whose elements become the next chunk's
positional args (see "Notes & gotchas"). No tracing, no module surgery.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from ramtorch import OffloadModel

# 1. Dice your model into chunks (you choose the granularity).
def make_block(d):
    return nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

chunks = [make_block(1024) for _ in range(24)]   # 24 transformer-ish blocks

# 2. Wrap them. Weights stream through a 2-slot GPU window; 4 stay pinned.
model = OffloadModel(chunks, device="cuda:0",
                     window=2,        # streaming slots (>=2 overlaps load/compute)
                     pin=4,           # chunks pinned resident on the GPU
                     keep_activations=True)

# 3. Inference: one streamed forward.
out = model(x)

# 4. Training: step + flush + your own optimizer.
opt = torch.optim.SGD(model.parameters(), lr=1e-3)
for x, y in loader:
    res = model.step(x, targets=y, loss_fn=F.cross_entropy)
    model.flush_grads()      # accumulated grads -> .grad
    opt.step()
    opt.zero_grad()
    # (or model.zero_grad_acc() to drop the accumulators)

model.close()                # stop the loader/writeback threads
```

`OffloadModel` is an `nn.Module`, so `.parameters()` / `.state_dict()` /
`.train()` / `.eval()` work as usual.

---

## Inference

`model(x)` is a plain streamed forward — that's all there is to it:

```python
model.eval()
logits = model(x)          # weights stream through the window as usual
```

* **Always `no_grad`.** `forward()` is decorated with `@torch.no_grad()`, so
  inference builds no autograd graph and keeps no activations, regardless of
  the `keep_activations` mode you trained with. Peak memory during inference
  is just `(window + pin)` chunks of weights plus the live chunk's
  activations.
* **Same streaming path as training.** The loader thread prefetches upcoming
  chunks over the H2D stream exactly as in a training forward, so the
  transfer- vs compute-bound trade-off (and the `offload_simulator`
  predictions) apply unchanged. There is no backward, so no D2H writeback
  traffic — inference is the friendliest regime for offload.
* **Tuple inputs/outputs work.** `x` may be a tensor or a tuple of tensors
  (elements become the first chunk's positional args), and a tuple-returning
  final chunk gives you a tuple back.
* **Batched / repeated calls are fine.** The window persists between calls;
  chunks still resident from the previous pass are reused (subject to
  eviction), so back-to-back batches over the same weights skip some loads.
  Note `flush_grads()` / `invalidate_residency()` drop that residency — call
  them after optimizer steps, not between inference batches.
* **Autocast works ambiently** — wrap the call in
  `torch.autocast("cuda", dtype=torch.bfloat16)` for bf16 inference; no engine
  support needed (all compute runs on the calling thread).
* **`offload_checkpoint` marks are a no-op** under `no_grad`, so a model
  annotated for selective checkpointing runs unchanged at inference.

For a runnable streamed-eval loop with accuracy + peak-memory reporting, see
`examples/mnist_offload_example.py` (its eval phase) and
`examples/offload_quickstart.py`.

---

## The three knobs

| Knob | Default | What it does |
|---|---|---|
| `window` | `2` | Streaming slots on the GPU. `window=1` never overlaps a load with compute; `window≥2` lets the loader run ahead. Total weight memory ≈ `window + pin` chunks. |
| `pin` | `0` | Number of **evenly-spaced** chunks pinned resident (never loaded/evicted). Eases PCIe pressure and steadies the schedule at the cost of their memory. `pin_layers=[...]` overrides with explicit indices. |
| `nvme` | `0` | Number of chunks whose **masters live on disk** instead of CPU RAM (interleaved placement; `nvme_layers=[...]` overrides, `nvme_path=` required). Saves host RAM at the cost of slower loads for those chunks — see "The NVMe tier" below. |
| `keep_activations` | `False` | Backward strategy: `False` (recompute), `True` (keep), or `"checkpoint"` (see below). |

### `keep_activations` — the three backward strategies

* **`False` — "recompute" (default).** The training forward runs under
  `no_grad`, caching only chunk-*boundary* activations. Backward reloads each
  chunk and **recomputes its forward** (engine-managed per-chunk gradient
  checkpointing). Cheapest activation memory, but every chunk's forward runs
  twice — and stochastic chunks (dropout) would **resample** on recompute, so
  don't use it with dropout.

* **`True` — "keep".** The forward builds each chunk's autograd graph and keeps
  its internal activations, so backward skips the recompute. **Weights still
  stream**: the graph references stable per-chunk weight tensors whose *storage*
  is freed on eviction (`untyped_storage().resize_(0)`, the FSDP resharding
  trick) and refilled by the loader before that chunk's backward. Costs
  activation memory for all chunks at once, and works with stochastic chunks.

* **`"checkpoint"`.** Keep-mode plumbing, but each chunk runs under
  non-reentrant `torch.utils.checkpoint`: internal activations are dropped at
  forward and recomputed by *autograd* during that chunk's backward — by which
  point the loader has refilled the weight storages. Because checkpoint
  stashes/restores the RNG state, the recompute replays the **same** dropout
  masks: recompute-level memory that is also dropout-safe.

Measured (101M params, 12 chunks, batch 4096, window=2, RTX PRO 6000 — see
`examples/offload_checkpoint_study.py`):

| mode | step | peak GPU | dropout-safe |
|---|---|---|---|
| recompute | 206 ms | 674 MiB | no (resamples) |
| keep | 139 ms | 2018 MiB | yes |
| `"checkpoint"` | 166 ms | 674 MiB | **yes** |

All three are bit-identical to a full-resident reference on deterministic
models; with dropout, keep and `"checkpoint"` stay bit-identical while
recompute drifts. Rule of thumb: **keep** when activations fit (fastest —
one forward); **`"checkpoint"`** when they don't (it beat the engine's own
recompute in the study, so it is the better tight-memory default);
**recompute** remains the zero-graph-overhead baseline.

### Selective checkpointing — mark regions in your own forward

`keep_activations="checkpoint"` is all-or-nothing. To choose *which parts* of
a chunk recompute, mark them in your forward with **`offload_checkpoint`** and
run with `keep_activations=True` — unmarked parts keep their activations,
marked parts are recomputed by autograd at backward:

```python
from ramtorch import OffloadModel, offload_checkpoint

class Block(nn.Module):
    def forward(self, x):
        x = self.light(x)                         # activations kept
        return offload_checkpoint(self.heavy, x)  # recomputed at backward

model = OffloadModel(blocks, device="cuda:0", keep_activations=True)
```

Do **not** use a bare `torch.utils.checkpoint` for this — it fails with a
device mismatch. The engine invokes chunks through `functional_call`, which
reverts the module's params to the CPU masters when it exits, so torch
checkpoint's backward-time recompute would read the wrong (CPU) weights.
`offload_checkpoint` snapshots the module's *effective* tensors at forward
time (the streamed GPU tensors, refilled by the loader before backward) and
re-applies them for the recompute. It stashes/restores RNG like torch
checkpoint (dropout-safe), works unchanged in a plain non-offload model, and
is a no-op under `no_grad`.

Measured on the same study config (light `d×d` kept + heavy 4x expansion
marked): keep with no marks 2222 MiB → keep with marks **814 MiB** →
`"checkpoint"` (everything) 702 MiB — and the marked run stays bit-identical
to an ordinary unmarked full-resident model, dropout included.

---

## The NVMe tier — masters on disk (`nvme=`, `nvme_path=`)

When even CPU RAM can't hold all the masters, push some chunks down one more
level:

```python
model = OffloadModel(chunks, device="cuda:0", window=2, pin=4,
                     nvme=8,                              # 8 chunks on disk
                     nvme_path="/mnt/nvme/scratch/weights.bin")
```

> **🔒 This tier is locked behind an environment variable.** Requesting NVMe
> chunks raises `RuntimeError` unless you first set
> `RAMTORCH_NVME_ACKNOWLEDGE=1` — an explicit consent that you understand the
> drive-wear risk described below. Unlocked, it also prints a loud wear
> warning on every construction; set `RAMTORCH_NVME_QUIET=1` as well to
> silence it. (Yes, two variables — that's deliberate.)

* **Pure PyTorch, no GDS.** The selected chunks' weights move into one
  page-aligned scratch file, held as **mmap-backed tensors**
  (`torch.UntypedStorage.from_file`, exposed via `ramtorch.NvmeTensorStore`).
  The loader streams them disk → shared pinned staging buffer → GPU on its
  one thread. We measured the "real" GDS-style alternative and the disk→GPU
  host hop serializes on the H2D copy engine anyway
  (`examples/nvme_h2d_contention_test.py`), so this staged path is the honest
  architecture — it is exactly the **"slower H2D"** model the simulator uses
  (`--nvme K --tnvme ...`).
* **Interleaved placement by default** — NVMe chunks spread evenly among the
  CPU chunks, which simulation shows hides the slow loads behind neighboring
  compute best (`python -m ramtorch.offload_simulator --nvme K` compares
  `interleave` vs `tail` yourself). `nvme_layers=[...]` overrides; overlap
  with pinned chunks resolves in pinned's favor.
* **Training works unchanged.** The mapped masters are ordinary CPU tensors:
  grads accumulate in RAM as usual, and the optimizer updates the masters in
  place — writes land in the page cache and the kernel flushes them to disk
  lazily. `state_dict()` reads through the mapping transparently.
* **Page-cache semantics.** Cold loads run at disk speed; if the OS has free
  RAM, recently-used chunks are served from cache (and evicted under memory
  pressure — which is the scenario this tier exists for). Don't put
  `nvme_path` on tmpfs (`/tmp` often is), that's just RAM with extra steps.
* **Scratch, not a checkpoint.** `close()` deletes the file. Save checkpoints
  with `torch.save(model.state_dict(), ...)` as usual.
* `model.stats["nvme_loads"]` counts disk-tier loads; profile traces label
  them `N3 nvme load` next to the `L3 h2d load` spans.

### ⚠️ Drive-endurance caution: prefer this tier for inference, not training

**This is why the tier requires `RAMTORCH_NVME_ACKNOWLEDGE=1` (see above).**
NAND flash wears out — consumer NVMe drives are typically rated for only a
few hundred TB written (TBW). How the two workloads differ:

* **Inference is fine.** The scratch file is written **once** at construction
  and only ever *read* afterwards. Reads don't wear NAND in any meaningful
  way, so even serving a model from this tier all day is easy on the drive.
* **Training can trash a drive.** Every optimizer step rewrites every NVMe
  master in place, and the kernel's page-cache writeback turns that into real
  device writes. A 10 GiB NVMe tier at 2 steps/s is ~70 GB written per hour,
  **~1.7 TB/day**; a big model (FLUX-scale, tens of GB on the tier) at
  training throughput reaches **petabytes per day** — enough to burn through
  a consumer drive's TBW rating in months, days, or less, and sustained
  write pressure also hurts the drive's read latency (which your loads
  depend on).

If you do train with the NVMe tier:

* Put **cold chunks only** on disk — layers whose masters rarely matter for
  stalls — and keep the tier small (`nvme=` a minority of chunks).
* Prefer a **high-endurance drive** (enterprise/datacenter-class, high
  DWPD/TBW) or a dedicated scratch SSD you consider consumable.
* Watch `nvme_loads` and your drive's SMART "percentage used" / "data units
  written" counters (`smartctl -a /dev/nvmeX`) during long runs.
* Gradient accumulation (fewer optimizer steps per sample) reduces write
  volume proportionally.

---

## The optimizer sees mixed devices — use `fused=True`

Streamed chunks keep their master weights (and grads) **on CPU**; pinned chunks
keep theirs **on the GPU**. After `flush_grads()`, `.grad` lives on the same
device as each param (in a persistent **pinned** buffer for the CPU ones).
One `torch.optim.AdamW(fused=True)` covers both: torch groups params by
device, so the CPU masters get the **fused CPU kernel** — a single
multithreaded pass over host RAM at DDR bandwidth — and the pinned chunks get
the fused CUDA kernel.

```python
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, fused=True)
for x, y in loader:
    res = model.step(x, targets=y, loss_fn=F.cross_entropy)
    model.flush_grads()   # pinned .grad buffers + residency invalidation
    opt.step()            # fused CPU kernel for masters, fused CUDA for pinned
    model.zero_grad_acc()
```

Why fused specifically: the default (non-fused) CPU path makes ~7 eager
passes over params/grads/state and is ~5x slower. The fused kernel touches
everything once (~28 B/param), so it runs at your DDR bandwidth — which also
means **no PCIe-streaming scheme can beat it** (PCIe is 20–50 GB/s vs 100+
GB/s DDR on workstation boards). We built and benchmarked exactly that
scheme — a windowed GPU-streaming AdamW mirroring the OffloadModel design —
and the fused CPU kernel won ~4x (33 ms vs 104–141 ms at 128M params). It
survives as the **private, educational**
`ramtorch.offload_optimizer.OffloadAdamW` (not exported): bit-identical to
`torch.optim.AdamW`, a clean worked example of the window/stream/event
pattern, and still the best option only in narrow cases (very low DDR
bandwidth, CPU saturated by dataloading, or bf16 optimizer state to halve
state RAM). `examples/offload_optimizer_check.py` runs its parity check and
the full benchmark (streamed vs CPU foreach/fused vs legacy vs GPU-resident).

Numerics footnote: the fused CPU and fused CUDA kernels round differently
(~1e-7/step), so a streamed model and a full-resident reference updated by
"the same" `AdamW(fused=True)` drift apart slowly — not a bug, just two
different kernels. `offload_quickstart.py` shows how to compare bit-exactly
anyway (mirror the per-layer optimizer device placement in the reference).

With `k` accumulated `step()` calls between flushes, pass `model.flush_grads(scale=1/k)`
for the mean (gradient accumulation).

**In-place updates and residency.** Resident GPU weight copies are a cache
over the CPU masters; they do **not** see master updates. `flush_grads()`
invalidates the streamed chunks' residency automatically (an optimizer step
usually follows). If you update params in place *without* going through
`flush_grads()`, call `model.invalidate_residency()` yourself — otherwise the
next step computes with stale weights for whichever chunks stayed resident.
(With SGD-sized deltas this hides below fp32 noise, which is why it can go
unnoticed; with Adam-sized steps it breaks parity immediately.)

---

## Mixed precision — just wrap `step()` in autocast

Unlike the pipeline (whose stages run on worker threads, needing the
`autocast=` parameter because autocast state is thread-local), **all offload
compute runs on the calling thread** — the loader/writeback threads only do
copies. So the ambient context applies to the whole step (forward, backward
recompute, loss) and no engine support is needed:

```python
for x, y in loader:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        res = model.step(x, targets=y, loss_fn=F.cross_entropy)
    model.flush_grads()   # grads are fp32 as usual
    opt.step()
    model.zero_grad_acc()
```

Verified in `offload_streaming_check.py`: bf16-autocast training is
**bit-identical** to a full-resident model trained with the same recipe, in
all three backward modes (the recompute modes re-run forward ops *inside*
`step()`, i.e. still under your autocast; `"checkpoint"` additionally stashes
and restores the autocast state itself). Params and grads stay fp32, so the
fused-AdamW recommendation is unchanged.

---

## Bypassing `loss_fn`: backprop a gradient directly (`grad_outputs=`)

The same escape hatch as the pipeline's: skip the loss entirely and feed a
precomputed `dL/dOutput` into the last chunk's backward — useful when the
gradient comes from somewhere the model can't see (a downstream model, a
custom differentiator, RL advantages).

```python
# Callable form: resolved with the live output (mirrors loss_fn).
res = model.step(x, targets=y,
                 grad_outputs=lambda out, tgt: 2.0 * (out - tgt))

# Tensor form: dL/dOut you computed elsewhere (tuple for tuple outputs).
res = model.step(x, grad_outputs=dl_dout)
```

- **Mutually exclusive with `loss_fn`** — passing both raises `ValueError`.
- **No loss is reported** — `res.loss` raises a clear `RuntimeError`.
- Works in all three backward modes; parity vs a reference loss-backward is
  checked in `offload_streaming_check.py`.

---

## When offload helps (and when it doesn't)

Offload trades **PCIe bandwidth** for **GPU memory**. It wins when there's
enough per-chunk compute to hide the H2D copy behind.

* **Transfer-bound regime** (tiny/fast chunks): PCIe can't keep up, compute
  stalls on `wait L*`, and offload is *slower* than full-resident. Not worth it.
* **Compute-bound regime** (enough FLOPs per chunk): loads hide behind compute
  and a streamed step approaches full-resident speed **at a fraction of the
  memory**. This is the target regime.

The **`offload_simulator`** predicts which regime you're in *before* you build
anything. Give it per-chunk costs (`tf` forward, `tb` backward, `th2d` load,
`td2h` writeback) and sweep the window:

```bash
# sweep window 1..N, print makespan / GPU% / stall / peak-resident table
python -m ramtorch.offload_simulator --layers 24 --tf 1.0 --tb 2.0 --th2d 0.5

# one window, ASCII Gantt
python -m ramtorch.offload_simulator --layers 24 --window 2 --pin 4 --plot gantt.png
```

`gpu%` near 100 with small `peak` means the window is hiding the loads — good.
Large `stall` / low `gpu%` means you're transfer-bound.

---

## Profiling a real step

`step(profile_path=...)` writes a Chrome-trace (Perfetto) JSON with compute
spans (`F3`/`B3`), H2D loads (`L3 h2d load`), D2H writebacks, and stall waits
(`wait L3`) — including the loader/writeback *worker-thread* spans, which stock
kineto can't see on its own:

```python
model.step(x, targets=y, loss_fn=F.cross_entropy,
           profile_path="offload.trace.json")
# open at https://ui.perfetto.dev
```

`model.stats` also accumulates `{"loads": int, "acquire_wait_s": float}` — the
total stall time is the single best signal of whether you're transfer-bound.

---

## Notes & gotchas

* **Tuple intermediates are supported.** A chunk may return a single tensor or
  a *tuple* of tensors; a tuple's elements become the next chunk's positional
  args (`def forward(self, a, b, mask): ...`). Non-float / no-grad elements
  (attention masks, position ids) thread through without gradients. The first
  chunk's input and the last chunk's output can be tuples too (`step()`'s
  `loss_fn` receives the raw output).
* **Buffer mutations don't write back.** Buffer edits inside a chunk (e.g.
  BatchNorm running stats) happen on the streamed GPU copy and are discarded.
  Use eval-mode / buffer-free norms (LayerNorm is fine).
* **Dropout**: use `keep_activations=True` or `"checkpoint"` (engine recompute
  would resample the mask).
* **CPU is supported** (`device="cpu"`) for tests — copies degrade to clones,
  so it's slow but correct.
* **`close()`** when done to join the loader/writeback threads (also called from
  `__del__`).

---

## Deprecation

`ramtorch.Linear` / `CPUBouncingLinear` is deprecated (`DeprecationWarning` on
construction). It bounced *individual* linear layers on CUDA streams; the
windowed chunk-streaming model here is strictly more general (you control the
chunk granularity), overlaps better (dedicated H2D/D2H streams + loader/writeback
threads), and pins hot chunks. Migrate by dicing your model into chunk modules
and wrapping them in `OffloadModel`.

---

## Examples

| File | What it shows |
|---|---|
| `offload_quickstart.py` | Minimal train + inference walkthrough (the snippet above, runnable) |
| `mnist_offload_example.py` | Canonical end-to-end training run: dice an MLP into chunks, train on MNIST with `step()` + `flush_grads()` + `AdamW(fused=True)`, streamed eval, checkpoint, peak-memory + stall report. `--backward {recompute,keep,checkpoint}` picks the strategy; `--selective` marks each block's FF with `offload_checkpoint`; `--nvme K --nvme-path FILE` moves K chunks' masters onto the disk tier |
| `offload_vs_plain_demo.py` | Side-by-side vs a full-resident model: 100-step SGD **bit-identity**, wall-time + peak-memory table, Perfetto traces, transfer- vs compute-bound regimes |
| `offload_streaming_check.py` | Streaming executor vs full-resident reference: inference + training loss/grad parity across all three backward modes, tuple intermediates, bf16-autocast bit-identity, grad bypass (CPU always, CUDA when available) |
| `offload_checkpoint_study.py` | The three backward strategies head to head: bit-parity (incl. dropout, where recompute's drift is shown) + the memory/time table above + selective `offload_checkpoint` marks |
| `offload_optimizer_check.py` | The private/educational streamed `OffloadAdamW` vs `torch.optim.AdamW`: trajectory bit-identity + the optimizer-step benchmark showing why `fused=True` CPU AdamW is the recommendation |
| `offload_pinning_study.py` | `pin` vs bigger `window` at equal memory — which reduces stalls more |
| `offload_warmup_study.py` | Preload ("warmup") phase vs greedy start across load/compute ratios |
| `offload_sim_check.py` | `offload_simulator` invariants + agreement with the real executor |

(`vshape_schedule_check.py` belongs to the *pipeline* schedules, not offload — see `docs/pipeline_parallel.md`.)
