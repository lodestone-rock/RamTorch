"""
ramtorch.pipeline_easy
----------------------
A high-level, single-GPU-feel wrapper around the relay pipeline executor.

Design goals:
  * **Auto-split** the model into N balanced stages (no manual split_spec), with
    an optional manual override.
  * **``forward(x)``** for inference — one method call, returns logits, just
    like a normal model (but explicitly a pipeline call, so the parallelism is
    visible, not hidden).
  * **``step()`` + ``flush_grads()``** for training — the same manual,
    explicit grad flow as before (no hidden optimizer magic).

The wrapper is an ``nn.Module`` whose parameters are shared with the pipeline
stages, so a single optimizer over ``wrapper.parameters()`` (or the original
model's) sees the accumulated gradients.

Example — inference:
    pipe = PipelineModel(MyModel(), example_input, devices=["cuda:0","cuda:1"])
    logits = pipe.forward(images)          # -> last-stage output

Example — training:
    pipe = PipelineModel(MyModel(), example_input, devices=["cuda:0","cuda:1"])
    opt = torch.optim.Adam(pipe.parameters(), lr=1e-3)
    for x, y in loader:
        result = pipe.step(x, targets=y, schedule="staggered_1b1f",
                           n_microbatches=4, loss_fn=loss_fn)
        result.flush_grads()               # mean-scale microbatch grads -> .grad
        opt.step(); opt.zero_grad()
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import warnings

import torch
import torch.nn as nn

from .pipeline import PipelineResult, Stage
from .pipeline_relay import Pipeline

if True:  # TYPE_CHECKING-friendly import for the SplitPoint annotation
    from torch.distributed.pipelining import SplitPoint


class PipelinePaddingWarning(UserWarning):
    """Emitted when ``PipelineModel.forward`` pads a partial microbatch.

    Silence it with::

        import warnings
        from ramtorch.pipeline_easy import PipelinePaddingWarning
        warnings.filterwarnings("ignore", category=PipelinePaddingWarning)
    """


# ── Auto split-spec generation ────────────────────────────────────────────────

def auto_split_spec(
    model: nn.Module,
    num_stages: int,
    *,
    balance: str = "params",
    device_weights: Optional[Sequence[float]] = None,
) -> Dict[str, "SplitPoint"]:
    """
    Generate a balanced ``split_spec`` for ``pipeline()`` automatically.

    Splits on the model's TOP-LEVEL children (``named_children``), grouping them
    into ``num_stages`` contiguous blocks with roughly equal total size.

    balance:
      "params" — balance by parameter count (default; good for most nets)
      "children" — balance by number of child modules (good when blocks are
                   uniform, e.g. transformer layers)

    device_weights:
      Optional per-stage throughput weights for HETEROGENEOUS machines, e.g.
      ``[2.0, 1.0]`` gives stage 0 twice as much work as stage 1 (a faster GPU
      takes more layers). Normalized internally; defaults to uniform (homogeneous).
      Length must equal ``num_stages``.

    Returns a dict like ``{"layers.3": SplitPoint.BEGINNING, ...}`` suitable for
    ``torch.distributed.pipelining.pipeline(split_spec=...)``.

    Raises if the model has fewer top-level children than ``num_stages``.
    """
    children = list(model.named_children())
    n = len(children)
    if num_stages < 1:
        raise ValueError("num_stages must be >= 1")
    if num_stages > n:
        raise ValueError(
            f"cannot split into {num_stages} stages: model only has {n} "
            f"top-level children ({[name for name, _ in children]})"
        )
    if num_stages == 1:
        return {}

    # Per-stage throughput weights (heterogeneous machines). Stage s should get a
    # share of work proportional to device_weights[s].
    if device_weights is None:
        device_weights = [1.0] * num_stages
    device_weights = [float(w) for w in device_weights]
    if len(device_weights) != num_stages:
        raise ValueError(
            f"device_weights length {len(device_weights)} != num_stages {num_stages}"
        )
    if any(w <= 0 for w in device_weights):
        raise ValueError("device_weights must all be positive")

    # Weight of each child for balancing.
    if balance == "params":
        weights = [sum(p.numel() for p in c.parameters()) for _, c in children]
    elif balance == "children":
        weights = [1] * n
    else:
        raise ValueError(f"unknown balance mode: {balance!r}")

    total = sum(weights)
    weight_sum = sum(device_weights)

    # Greedy contiguous grouping. Each stage gets a share of the total work
    # proportional to its device weight. We accumulate children into the current
    # stage until they reach that stage's share, then split at the next child.
    split_spec: Dict[str, SplitPoint] = {}
    acc = 0.0
    stages_placed = 1
    # Work share for the current stage (recomputed as we advance).
    stage_share = total * device_weights[0] / weight_sum
    for i, (name, _) in enumerate(children):
        acc += weights[i]
        # Place a split if we've reached this stage's share and still have
        # stages to place and children to distribute.
        remaining_children = n - (i + 1)
        remaining_stages = num_stages - stages_placed
        if (
            stages_placed < num_stages
            and acc >= stage_share
            and remaining_children >= remaining_stages
        ):
            next_name = children[i + 1][0]
            split_spec[next_name] = SplitPoint.BEGINNING
            acc = 0.0
            stages_placed += 1
            stage_share = total * device_weights[stages_placed - 1] / weight_sum

    return split_spec


# ── High-level wrapper ────────────────────────────────────────────────────────

class PipelineModel(nn.Module):
    """
    An ``nn.Module`` that runs ``model`` pipeline-parallel across ``devices``.

    Feels like a single model: ``.forward(x)`` for inference, ``.step()`` +
    ``.flush_grads()`` for training, and ``.parameters()``/``.state_dict()``
    delegate to the wrapped model so optimizers and checkpoints work unchanged.

    Parameters
    ----------
    model         : the full model to parallelize (traced/split once)
    example_input : one example microbatch input for the tracer (must be on the
                    same device as ``model``; a CPU model wants a CPU tensor)
    devices       : one device per stage. If None and ``num_stages`` is set,
                    uses cuda:i round-robin (or CPU). Length == number of stages.
    split_spec    : optional manual split spec; if None, auto-generated to be
                    balanced across ``len(devices)`` stages.
    balance       : "params" | "children" for the auto-split heuristic.
    device_weights: optional per-stage throughput weights for heterogeneous
                    machines (e.g. ``[2.0, 1.0]`` gives stage 0 twice the work).
                    Only used when ``split_spec`` is None (auto-split).
    overlap       : per-stage worker threads (True) or sequential debug (False).
    fake_compute  : None | "replace" | {"fwd": s|[s...], "bwd": s|[s...]}.
    warn_on_padding : if True (default), emit a PipelinePaddingWarning when
                    forward() pads a partial microbatch. Set False to silence.
    """

    def __init__(
        self,
        model: nn.Module,
        example_input: torch.Tensor,
        *,
        devices: Optional[Sequence[Union[str, torch.device]]] = None,
        num_stages: Optional[int] = None,
        split_spec: Optional[dict] = None,
        balance: str = "params",
        device_weights: Optional[Sequence[float]] = None,
        overlap: bool = True,
        fake_compute=None,
        warn_on_padding: bool = True,
    ):
        super().__init__()
        self._model = model
        self._warn_on_padding = warn_on_padding

        # Resolve devices / num_stages.
        if devices is None:
            n_cuda = torch.cuda.device_count() if torch.cuda.is_available() else 0
            if num_stages is None:
                num_stages = max(1, min(n_cuda, 2)) if n_cuda else 1
            devices = (
                [torch.device("cuda", i % n_cuda) for i in range(num_stages)]
                if n_cuda
                else [torch.device("cpu")] * num_stages
            )
        devices = [torch.device(d) for d in devices]
        n_stages = len(devices)

        # Resolve split spec (auto if not provided).
        if split_spec is None:
            split_spec = auto_split_spec(
                model, n_stages, balance=balance, device_weights=device_weights
            )
        self.split_spec = split_spec

        # Build the reusable pipeline (splits once, builds stages once).
        self._pipe = Pipeline(
            model,
            example_input,
            split_spec,
            devices=devices,
            fake_compute=fake_compute,
            overlap=overlap,
        )
        self.devices = self._pipe.devices
        self.num_stages = self._pipe.num_stages
        # The traced graph is shape-specialized to the example's batch size
        # (torch.export bakes x.size(0) as a constant). Remember it so forward()
        # can chunk arbitrary inference batches down to the traced shape.
        self._example_batch = int(example_input.shape[0])

    # ── nn.Module delegation: params / buffers / state_dict come from the model ──
    def parameters(self, recurse: bool = True):
        return self._model.parameters(recurse=recurse)

    def named_parameters(self, prefix: str = "", recurse: bool = True):
        return self._model.named_parameters(prefix=prefix, recurse=recurse)

    def buffers(self, recurse: bool = True):
        return self._model.buffers(recurse=recurse)

    def state_dict(self, *args, **kwargs):
        return self._model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        return self._model.load_state_dict(*args, **kwargs)

    def train(self, mode: bool = True):
        self._model.train(mode)
        for st in self._pipe.stages:
            st.module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    # ── Inference ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def forward(self, data: torch.Tensor) -> torch.Tensor:
        """
        Run inference through the pipeline and return the last stage's output.

        A simple sequential relay: each stage computes its slice and hands the
        activation to the next. No autograd graph is built (``no_grad``), so this
        is cheap and does not retain activations. Explicitly a *pipeline* call —
        the parallelism is visible, not hidden behind a transparent ``model(x)``.

        The traced stages are shape-specialized to the example microbatch's batch
        size, so an arbitrary input batch is chunked into example-sized pieces,
        relayed, and the outputs concatenated along dim 0.
        """
        x = data
        if x.shape[0] == self._example_batch:
            return self._relay(x)
        chunks = []
        for piece in x.split(self._example_batch, dim=0):
            if piece.shape[0] != self._example_batch:
                # Final partial chunk: the traced graph needs the exact example
                # batch size, so pad up to it and slice the padding off after.
                pad = self._example_batch - piece.shape[0]
                if self._warn_on_padding:
                    warnings.warn(
                        f"PipelineModel.forward: input batch ({x.shape[0]}) is not "
                        f"divisible by the traced microbatch size "
                        f"({self._example_batch}); padding the last chunk with "
                        f"{pad} row(s) (discarded after compute). Silence with "
                        f"warnings.filterwarnings('ignore', category=PipelinePaddingWarning) "
                        f"or PipelineModel(..., warn_on_padding=False).",
                        PipelinePaddingWarning,
                        stacklevel=2,
                    )
                pad_rows = piece[:1].expand(pad, *piece.shape[1:])
                out = self._relay(torch.cat([piece, pad_rows], dim=0))
                chunks.append(out[: piece.shape[0]])
            else:
                chunks.append(self._relay(piece))
        return torch.cat(chunks, dim=0)

    @torch.no_grad()
    def _relay(self, x: torch.Tensor) -> torch.Tensor:
        """Single example-sized batch through the stage relay."""
        for st in self._pipe.stages:
            x = x.to(st.device)
            x = st.module(x)
        return x

    # ── Training ──────────────────────────────────────────────────────────────
    def step(
        self,
        data: torch.Tensor,
        *,
        targets: Optional[torch.Tensor] = None,
        schedule: str = "staggered_1b1f",
        n_microbatches: int = 4,
        loss_fn: Optional[Callable] = None,
        trace_path: Optional[str] = None,
        profile_path: Optional[str] = None,
    ) -> PipelineResult:
        """
        One pipeline forward+backward training step. Returns a PipelineResult;
        call ``result.flush_grads()`` (mean-scaled) or ``self.flush_grads()``
        before your optimizer step.

        schedule: "staggered_1b1f" (default, recommended) or "gpipe" (baseline).
        "1f1b" (textbook forward-first) is accepted but is educational-only —
        see the README for why execution order matters.
        """
        return self._pipe.step(
            data,
            targets=targets,
            schedule=schedule,
            n_microbatches=n_microbatches,
            loss_fn=loss_fn,
            trace_path=trace_path,
            profile_path=profile_path,
        )

    def flush_grads(self, scale: Optional[float] = None, n_microbatches: int = 1):
        """Write accumulated microbatch grads into ``.grad`` (mean-scaled)."""
        self._pipe.flush_grads(scale=scale, n_microbatches=n_microbatches)


__all__ = ["PipelineModel", "auto_split_spec", "PipelinePaddingWarning"]
