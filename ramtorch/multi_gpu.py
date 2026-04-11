"""multi_gpu.py — Transparent multi-GPU wrapper for the ThreadPoolExecutor + NCCL pattern.

Design goals
------------
* Write your training loop exactly as you would for a single GPU.
* Hand ``MultiGPUWrapper`` one model (on CPU / meta device), one optimizer
  factory, and a ``forward_backward`` callable.  The wrapper replicates
  everything, runs forward/backward in parallel threads, reduces gradients
  via NCCL (ZeRO-1), and steps every optimizer — all in one call.
* Every internal list (``models``, ``optimizers``, ``schedulers``,
  ``executor``) is a public attribute so you can reach in and tinker
  whenever you need to.

ZeRO-1 optimizer sharding
--------------------------
By default the wrapper uses **ZeRO-1** sharding: trainable parameters are
assigned to GPUs in round-robin order, and each GPU only holds the optimizer
state (momentum buffers, second moments, etc.) for its own slice.  This cuts
optimizer VRAM by ~1/n_gpus with no change to the training loop.

The gradient sync path is therefore *reduce* (not all-reduce): each grad is
summed only onto the owner GPU, the owner steps its optimizer, and the fresh
weights are then *broadcast* back to all replicas.

Minimal usage
-------------
::

    def my_forward_backward(gpu_id, model, chunk) -> float:
        x = chunk.to(f"cuda:{gpu_id}")
        loss = model(x).mean()
        loss.backward()
        return loss.item()

    wrapper = MultiGPUWrapper(
        model_factory=lambda: MyModel(),
        optimizer_factory=lambda params: AdamW(params, lr=1e-4),
        forward_backward_fn=my_forward_backward,
    )
    wrapper.setup()

    for batch in dataloader:
        chunks = wrapper.split_batch(batch)
        loss = wrapper.step(chunks)

Advanced usage — gradient accumulation
---------------------------------------
::

    for i, batch in enumerate(dataloader):
        chunks = wrapper.split_batch(batch)
        loss = wrapper.forward_backward_only(chunks)   # no optimizer step

        if (i + 1) % accum_steps == 0:
            wrapper.reduce_grads()          # ZeRO-1: reduce each grad to its owner
            wrapper.clip_grads(max_norm=1.0)
            wrapper.optimizer_step()        # step owned params + broadcast weights

Split forward / backward
------------------------
Useful when you need the outputs between the forward and backward passes
(e.g. logging predictions, computing auxiliary losses, mixed-precision
scaling, etc.).  Both calls run concurrently across all GPUs::

    # forward_fn: (gpu_id, model, *chunk_args) -> any output
    outputs = wrapper.forward(chunks, forward_fn=my_forward_fn)

    # outputs is a list[any], one entry per GPU
    # inspect / post-process outputs here ...

    # backward_fn: (gpu_id, model, output) -> float
    loss = wrapper.backward(outputs, backward_fn=my_backward_fn)

    wrapper.reduce_grads()   # ZeRO-1: reduce each grad to its owner GPU
    wrapper.clip_grads()
    wrapper.optimizer_step() # step owned params + broadcast weights

Inference (forward-only, eval mode, no_grad)
--------------------------------------------
Same ``forward()`` call — just pass ``eval_mode=True``::

    outputs = wrapper.forward(chunks, forward_fn=my_forward_fn, eval_mode=True)
    # all replicas run concurrently in eval + no_grad, then restored to train

Accessing internals
-------------------
::

    wrapper.models[2]          # model replica on cuda:2
    wrapper.optimizers[0]      # optimizer for GPU 0
    wrapper.executor           # the ThreadPoolExecutor itself
    wrapper.model              # alias for models[0]
    wrapper.optimizer          # alias for optimizers[0]
    wrapper.scheduler          # alias for schedulers[0] (if any)
"""

from __future__ import annotations

import copy
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, TypeVar

import torch
import torch.cuda.nccl as nccl
import torch.nn as nn
from safetensors.torch import save_file

__all__ = ["MultiGPUWrapper"]

T = TypeVar("T")

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# A forward_backward_fn receives (gpu_id, model, *chunk_args) and returns a
# scalar loss value.  The exact signature is flexible — see ForwardBackwardFn.
ForwardBackwardFn = Callable[..., float]


class MultiGPUWrapper:
    """Wraps a model + optimizer for transparent multi-GPU training.

    Parameters
    ----------
    model_factory : callable
        Called once per GPU with no arguments; must return an ``nn.Module``
        on CPU (or meta device).  The wrapper moves it to the right device.
    optimizer_factory : callable
        Called once per GPU with the model's trainable parameter list.
        Must return a ``torch.optim.Optimizer``.
    forward_backward_fn : callable, optional
        Signature: ``fn(gpu_id: int, model: nn.Module, *args, **kwargs) -> float``
        Should run the forward pass, compute the loss, call ``loss.backward()``,
        and return the scalar loss value.  The wrapper calls this in parallel
        threads — make sure it is thread-safe (no shared mutable state outside
        the per-GPU model/optimizer).  Required only when using the combined
        ``forward_backward_only()`` / ``step()`` fast path; omit it if you
        prefer the split ``forward()`` + ``backward()`` pattern.
    scheduler_factory : callable, optional
        Called once per GPU with the optimizer; must return an LR scheduler.
    n_gpus : int, optional
        Number of GPUs to use.  Defaults to ``torch.cuda.device_count()``.
    checkpoint_path : str, optional
        Path to a ``.safetensors`` or ``.pth`` checkpoint to load on startup.
    dtype : torch.dtype, optional
        dtype to cast the model to after moving to device.  Defaults to
        ``torch.float32`` (master weights).  Cast individual sub-modules
        yourself if you need mixed precision.
    gradient_accumulation_steps : int
        How many ``forward_backward_only`` calls before a full
        ``optimizer_step``.  Only used by the convenience ``step()`` method.
    max_grad_norm : float
        Gradient clipping norm.  Set to 0 or ``None`` to disable.
    """

    def __init__(
        self,
        *,
        model_factory: Callable[[], nn.Module],
        optimizer_factory: Callable[[List[nn.Parameter]], torch.optim.Optimizer],
        forward_backward_fn: Optional[ForwardBackwardFn] = None,
        scheduler_factory: Optional[Callable[[torch.optim.Optimizer], Any]] = None,
        n_gpus: Optional[int] = None,
        checkpoint_path: str = "",
        dtype: torch.dtype = torch.float32,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ):
        self.model_factory = model_factory
        self.optimizer_factory = optimizer_factory
        self.forward_backward_fn = forward_backward_fn
        self.scheduler_factory = scheduler_factory
        self.n_gpus: int = n_gpus if n_gpus is not None else torch.cuda.device_count()
        self.checkpoint_path = checkpoint_path
        self.dtype = dtype
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm

        # Public internals — populated by setup()
        self.models: List[nn.Module] = []
        self.optimizers: List[torch.optim.Optimizer] = []
        self.schedulers: List[Any] = []
        self.executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)

        # Convenience aliases (set after setup)
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.scheduler: Optional[Any] = None

        # Internal accumulation counter
        self._accum_count: int = 0

        # ZeRO-1: populated by setup()
        # _param_owner[i]  -> gpu_id that owns the i-th trainable param
        # _owner_params[g] -> list of (param_idx, replica-g param) pairs owned by GPU g
        self._param_owner: List[int] = []
        self._owner_params: List[List[Tuple[int, nn.Parameter]]] = []

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def setup(self):
        """Instantiate and replicate models, optimizers, and schedulers.

        Call this once before the training loop.
        """
        if self.n_gpus == 0:
            raise RuntimeError("No CUDA devices found.")

        print(f"[MultiGPUWrapper] Setting up on {self.n_gpus} GPU(s)...")

        # Load checkpoint state dict once on CPU (shared across GPU copies)
        checkpoint_sd: Optional[Dict[str, torch.Tensor]] = None
        if self.checkpoint_path:
            print(f"[MultiGPUWrapper] Loading checkpoint: {self.checkpoint_path}")
            checkpoint_sd = _load_state_dict(self.checkpoint_path)

        self.models = []
        self.optimizers = []
        self.schedulers = []

        for gpu_id in range(self.n_gpus):
            device = f"cuda:{gpu_id}"
            print(f"[MultiGPUWrapper]   Initialising replica on {device}...")

            model = self.model_factory()

            if checkpoint_sd is not None:
                loaded, mismatched, new = _apply_state_dict(
                    model, checkpoint_sd, device
                )
                if gpu_id == 0:
                    print(
                        f"    Loaded {len(loaded)} keys | "
                        f"{len(mismatched)} mismatched | {len(new)} new"
                    )
                    for m in mismatched[:5]:
                        print(f"      shape mismatch: {m}")

            model = model.to(device=device, dtype=self.dtype)
            model.train()
            self.models.append(model)

        # ------------------------------------------------------------------
        # ZeRO-1: build round-robin ownership map across all trainable params.
        # Each param is owned by exactly one GPU; that GPU holds the optimizer
        # state for it, reducing peak optimizer VRAM by ~1/n_gpus.
        # ------------------------------------------------------------------
        all_trainable: List[nn.Parameter] = [
            p for p in self.models[0].parameters() if p.requires_grad
        ]

        # param_owner[i] = gpu_id that owns the i-th trainable param.
        # Assignment uses a greedy largest-first bin-packing: sort params by
        # element count descending, then always assign the next param to the
        # GPU currently holding the fewest elements.  This keeps the total
        # optimizer-state footprint roughly equal across GPUs even when param
        # sizes vary wildly (e.g. large weight matrices vs tiny bias vectors).
        param_numel = [p.numel() for p in all_trainable]
        sorted_indices = sorted(range(len(all_trainable)), key=lambda i: param_numel[i], reverse=True)
        gpu_loads = [0] * self.n_gpus  # running element count per GPU
        self._param_owner = [0] * len(all_trainable)
        for param_idx in sorted_indices:
            owner = gpu_loads.index(min(gpu_loads))
            self._param_owner[param_idx] = owner
            gpu_loads[owner] += param_numel[param_idx]

        # _owner_params[g] = list of (param_idx, replica-g param) pairs owned by GPU g.
        # param_idx is the flat index into the trainable param list of any replica
        # (all replicas share the same topology), used by reduce_grads() and
        # broadcast_params() to address the right tensor on each replica.
        self._owner_params = [[] for _ in range(self.n_gpus)]
        trainable_per_gpu: List[List[nn.Parameter]] = [
            [p for p in m.parameters() if p.requires_grad] for m in self.models
        ]
        for param_idx, owner in enumerate(self._param_owner):
            self._owner_params[owner].append(
                (param_idx, trainable_per_gpu[owner][param_idx])
            )

        # Build one optimizer per GPU over only its owned params.
        for gpu_id in range(self.n_gpus):
            owned_params = [p for _, p in self._owner_params[gpu_id]]
            optimizer = self.optimizer_factory(owned_params)
            self.optimizers.append(optimizer)

            if self.scheduler_factory is not None:
                self.schedulers.append(self.scheduler_factory(optimizer))

        # Convenience aliases
        self.model = self.models[0]
        self.optimizer = self.optimizers[0]
        self.scheduler = self.schedulers[0] if self.schedulers else None

        # Rebuild executor with the right worker count
        self.executor.shutdown(wait=False)
        self.executor = ThreadPoolExecutor(max_workers=self.n_gpus)

        total = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        owned_elems = [gpu_loads[g] for g in range(self.n_gpus)]
        print(
            f"[MultiGPUWrapper] Ready — {self.n_gpus} replica(s) | "
            f"total={total:,} | trainable={n_trainable:,} | "
            f"ZeRO-1 elements per GPU: {owned_elems}"
        )

    # ------------------------------------------------------------------
    # Batch splitting
    # ------------------------------------------------------------------

    def split_batch(self, *tensors: torch.Tensor) -> List[Tuple[torch.Tensor, ...]]:
        """Split a batch of tensors evenly across GPUs.

        Returns a list of ``n_gpus`` tuples, one per GPU.  Trailing samples
        that don't divide evenly are dropped (same behaviour as the existing
        trainers).

        Example::

            images, labels, weights = batch
            chunks = wrapper.split_batch(images, labels, weights)
            # chunks[i] == (images[s:e], labels[s:e], weights[s:e])
        """
        n = tensors[0].shape[0]
        per_gpu = n // self.n_gpus
        chunks = []
        for g in range(self.n_gpus):
            s, e = g * per_gpu, (g + 1) * per_gpu
            chunks.append(tuple(t[s:e] for t in tensors))
        return chunks

    # ------------------------------------------------------------------
    # Core primitives
    # ------------------------------------------------------------------

    def forward_backward_only(
        self,
        chunks: List[Any],
        **kwargs,
    ) -> float:
        """Run forward + backward on all GPUs in parallel; do NOT step.

        ``chunks[gpu_id]`` is passed as positional args after ``(gpu_id, model)``
        to ``forward_backward_fn``.  If a chunk is a tuple it is unpacked;
        otherwise it is passed as a single argument.

        Returns the sum of per-GPU scalar losses.
        """

        if self.forward_backward_fn is None:
            raise RuntimeError(
                "forward_backward_only() requires forward_backward_fn to be set. "
                "Either pass it to MultiGPUWrapper() or use forward() + backward() instead."
            )
        _fn = self.forward_backward_fn  # narrowed: not None

        def _work(gpu_id: int) -> float:
            chunk = chunks[gpu_id]
            args = chunk if isinstance(chunk, tuple) else (chunk,)
            return _fn(gpu_id, self.models[gpu_id], *args, **kwargs)

        return sum(self.executor.map(_work, range(self.n_gpus)))

    def reduce_grads(self):
        """ZeRO-1 gradient sync: NCCL reduce each grad to its owner GPU only.

        Unlike ``all_reduce`` (which sums grads on *every* GPU), this sends
        each gradient only to the single GPU that owns the corresponding
        optimizer state.  After this call only the owner GPU has the correct
        summed gradient; all other replicas' copies of that grad are stale and
        will be zeroed by ``optimizer_step()``.

        This is the correct primitive to call before ``optimizer_step()`` when
        using ZeRO-1 sharding.
        """
        if self.n_gpus < 2:
            return

        # Build aligned trainable-param lists for every replica once.
        trainable_per_gpu: List[List[nn.Parameter]] = [
            [p for p in m.parameters() if p.requires_grad] for m in self.models
        ]

        for param_idx, owner in enumerate(self._param_owner):
            # Collect the grad tensor from every replica for this param.
            grads = [
                trainable_per_gpu[g][param_idx].grad
                for g in range(self.n_gpus)
            ]
            # Skip if no replica produced a grad (e.g. frozen during this step).
            if all(g is None for g in grads):
                continue
            # Fill missing grads with zeros so NCCL sees a full tensor list.
            ref = next(g for g in grads if g is not None)
            grads = [
                g if g is not None else torch.zeros_like(ref) for g in grads
            ]
            # nccl.reduce(inputs, root=owner) reduces in-place into inputs[owner],
            # i.e. the owner GPU's own grad tensor receives the summed result.
            nccl.reduce(grads, root=owner, op=nccl.SUM)
            # Non-owner replicas no longer hold a valid grad — clear them so
            # they don't accidentally influence clip_grads or a future step.
            for g in range(self.n_gpus):
                if g != owner:
                    trainable_per_gpu[g][param_idx].grad = None

    def broadcast_params(self):
        """Broadcast updated param data from each owner GPU to all replicas.

        Call this *after* ``optimizer_step()`` has updated the owned params on
        each GPU.  Every non-owner replica will receive the fresh weights so
        all replicas stay in sync for the next forward pass.
        """
        if self.n_gpus < 2:
            return

        trainable_per_gpu: List[List[nn.Parameter]] = [
            [p for p in m.parameters() if p.requires_grad] for m in self.models
        ]

        with torch.no_grad():
            for param_idx, owner in enumerate(self._param_owner):
                # nccl.broadcast root is a *device index*, not a list index,
                # and its behaviour with non-zero root is unreliable across
                # communicator cache hits.  The safe approach: rotate the
                # tensor list so the owner is always at position 0, then
                # broadcast with root=0 (unambiguous).
                gpu_order = [owner] + [g for g in range(self.n_gpus) if g != owner]
                tensors = [trainable_per_gpu[g][param_idx].data for g in gpu_order]
                nccl.broadcast(tensors, root=0)

    def all_reduce(self):
        """NCCL all-reduce (SUM) gradients across all GPU replicas.

        .. note::
            This is the *non-ZeRO* path — every GPU accumulates the full
            summed gradient and keeps a full copy of the optimizer state.
            Prefer ``reduce_grads()`` + ``broadcast_params()`` for ZeRO-1
            memory savings.
        """
        if self.n_gpus < 2:
            return
        param_lists = [
            [p for p in m.parameters() if p.requires_grad and p.grad is not None]
            for m in self.models
        ]
        n_params = len(param_lists[0])
        for i in range(n_params):
            grads = [param_lists[g][i].grad for g in range(self.n_gpus)]
            nccl.all_reduce(grads, op=nccl.SUM)

    def clip_grads(self, max_norm: Optional[float] = None):
        """Clip gradients on all GPUs in parallel.

        In ZeRO-1 mode (after ``reduce_grads()``) only the owner GPU holds a
        valid grad for each param, so clipping is applied per-GPU over each
        GPU's owned params only.  The norm is therefore a *local* norm, not a
        global one — acceptable for most training runs.
        """
        norm = max_norm if max_norm is not None else self.max_grad_norm
        if not norm:
            return

        def _clip(gpu_id: int):
            # Clip only the params this GPU owns (the only ones with valid grads
            # after reduce_grads()).
            owned = [p for _, p in self._owner_params[gpu_id]]
            if owned:
                nn.utils.clip_grad_norm_(owned, norm)

        list(self.executor.map(_clip, range(self.n_gpus)))

    def optimizer_step(self):
        """ZeRO-1 optimizer step: each GPU steps its owned params, then
        ``broadcast_params()`` syncs the updated weights to all replicas.

        Also steps schedulers and zeros gradients on all GPUs.
        """

        def _step(gpu_id: int):
            self.optimizers[gpu_id].step()
            if self.schedulers:
                self.schedulers[gpu_id].step()
            self.optimizers[gpu_id].zero_grad()
            # Also zero grads on all other replicas for this GPU's owned params
            # (non-owner grads were already cleared by reduce_grads, but zero
            # the owner's too so the next accumulation starts clean).
            for _, p in self._owner_params[gpu_id]:
                if p.grad is not None:
                    p.grad = None

        list(self.executor.map(_step, range(self.n_gpus)))
        # Broadcast updated weights from each owner to all other replicas.
        self.broadcast_params()
        self._accum_count = 0

    def run_concurrent(self, fn: Callable[[int], T], *args, **kwargs) -> List[T]:
        """Run ``fn`` concurrently on all GPUs and return the results as a list.

        A lightweight wrapper around the internal ``ThreadPoolExecutor``.
        Useful for one-off per-GPU operations that don't fit the standard
        forward/backward pattern (e.g. custom metric collection, weight
        surgery, device-specific I/O, etc.).

        Parameters
        ----------
        fn :
            Callable with signature ``fn(gpu_id: int, *args, **kwargs) -> T``.
            It is invoked once per GPU, with ``gpu_id`` in ``[0, n_gpus)``.
        *args, **kwargs :
            Forwarded verbatim to every ``fn`` call after ``gpu_id``.

        Returns
        -------
        list
            ``[fn(0, ...), fn(1, ...), ..., fn(n_gpus-1, ...)]``

        Example::

            # Collect the L2 norm of each replica's first layer weights
            norms = wrapper.run_concurrent(
                lambda gpu_id: wrapper.models[gpu_id].layer0.weight.norm().item()
            )
        """
        return list(self.executor.map(lambda g: fn(g, *args, **kwargs), range(self.n_gpus)))

    # ------------------------------------------------------------------
    # Convenience: full step with built-in accumulation
    # ------------------------------------------------------------------

    def step(self, chunks: List[Any], **kwargs) -> float:
        """Forward + backward [+ all-reduce + clip + optimizer step].

        Handles gradient accumulation automatically: the optimizer is only
        stepped every ``gradient_accumulation_steps`` calls.

        Returns the scalar loss for this micro-step.
        """
        loss = self.forward_backward_only(chunks, **kwargs)
        self._accum_count += 1

        if self._accum_count >= self.gradient_accumulation_steps:
            self.reduce_grads()    # ZeRO-1: reduce each grad to its owner GPU
            self.clip_grads()
            self.optimizer_step()  # step owned params + broadcast + zero_grad

        return loss

    # ------------------------------------------------------------------
    # Split forward / backward
    # ------------------------------------------------------------------

    def forward(
        self,
        chunks: List[Any],
        forward_fn: Callable[..., Any],
        eval_mode: bool = False,
        **kwargs,
    ) -> List[Any]:
        """Run a forward pass on all GPUs concurrently; return one output per GPU.

        ``forward_fn`` signature::

            def forward_fn(gpu_id: int, model: nn.Module, *chunk_args, **kwargs) -> Any:
                ...

        The return value can be anything — a tensor, a tuple of tensors, a
        dataclass, etc.  It is collected into a list and returned as-is so
        you can inspect or post-process it before calling ``backward()``.

        Parameters
        ----------
        chunks :
            List of per-GPU data chunks, typically from ``split_batch()``.
        forward_fn :
            Callable with signature ``(gpu_id, model, *chunk_args, **kwargs) -> output``.
        eval_mode :
            If ``True``, each replica is switched to ``eval()`` and wrapped in
            ``torch.no_grad()`` for the duration of the call, then restored to
            ``train()``.  Use this for inference / validation.
        **kwargs :
            Forwarded verbatim to ``forward_fn``.

        Returns
        -------
        list
            One output per GPU, in GPU-index order.
        """

        def _work(gpu_id: int) -> Any:
            model = self.models[gpu_id]
            chunk = chunks[gpu_id]
            args = chunk if isinstance(chunk, tuple) else (chunk,)
            if eval_mode:
                model.eval()
                try:
                    with torch.no_grad():
                        return forward_fn(gpu_id, model, *args, **kwargs)
                finally:
                    model.train()
            else:
                return forward_fn(gpu_id, model, *args, **kwargs)

        return list(self.executor.map(_work, range(self.n_gpus)))

    def backward(
        self,
        outputs: List[Any],
        backward_fn: Callable[..., float],
        **kwargs,
    ) -> float:
        """Run a backward pass on all GPUs concurrently given prior ``forward()`` outputs.

        ``backward_fn`` signature::

            def backward_fn(gpu_id: int, model: nn.Module, output: Any, **kwargs) -> float:
                # compute loss from output, call loss.backward(), return scalar
                ...

        Parameters
        ----------
        outputs :
            The list returned by a preceding ``forward()`` call.
        backward_fn :
            Callable with signature ``(gpu_id, model, output, **kwargs) -> float``.
            Must call ``.backward()`` internally and return a scalar loss value.
        **kwargs :
            Forwarded verbatim to ``backward_fn``.

        Returns
        -------
        float
            Sum of per-GPU scalar losses.
        """

        def _work(gpu_id: int) -> float:
            return backward_fn(gpu_id, self.models[gpu_id], outputs[gpu_id], **kwargs)

        return sum(self.executor.map(_work, range(self.n_gpus)))

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str):
        """Save ``models[0]`` state dict to ``.safetensors`` or ``.pth``."""
        sd = self.models[0].state_dict()
        if path.endswith((".safetensors", ".sft")):
            save_file(sd, path)
        else:
            torch.save(sd, path)
        print(f"[MultiGPUWrapper] Saved: {path}")

    def load_checkpoint(self, path: str):
        """Hot-reload a checkpoint into all replicas (e.g. resume mid-run)."""
        print(f"[MultiGPUWrapper] Hot-loading checkpoint: {path}")
        sd = _load_state_dict(path)
        for gpu_id, model in enumerate(self.models):
            device = f"cuda:{gpu_id}"
            _apply_state_dict(model, sd, device)

    # ------------------------------------------------------------------
    # LR helpers
    # ------------------------------------------------------------------

    @property
    def current_lr(self) -> float:
        """Current learning rate from the first optimizer's first param group."""
        return self.optimizers[0].param_groups[0]["lr"]

    @property
    def last_lr(self) -> float:
        """Last LR from the first scheduler (if present), else ``current_lr``."""
        if self.schedulers:
            lrs = self.schedulers[0].get_last_lr()
            return lrs[0] if lrs else self.current_lr
        return self.current_lr

    # ------------------------------------------------------------------
    # Sync weights from GPU 0 → all other replicas
    # ------------------------------------------------------------------

    def sync_weights(self):
        """Broadcast weights from ``models[0]`` to all other replicas.

        Useful after manually modifying ``models[0]`` (e.g. EMA update,
        weight averaging) to keep all replicas in sync.
        """
        if self.n_gpus < 2:
            return
        sd0 = self.models[0].state_dict()
        for gpu_id in range(1, self.n_gpus):
            device = f"cuda:{gpu_id}"
            sd = {k: v.to(device) for k, v in sd0.items()}
            self.models[gpu_id].load_state_dict(sd)

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.cleanup()

    def cleanup(self):
        """Shut down the thread pool."""
        self.executor.shutdown(wait=True)

    def __repr__(self) -> str:
        return (
            f"MultiGPUWrapper("
            f"n_gpus={self.n_gpus}, "
            f"accum={self.gradient_accumulation_steps}, "
            f"max_grad_norm={self.max_grad_norm})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_state_dict(path: str) -> Dict[str, torch.Tensor]:
    """Load a state dict from a ``.safetensors`` or ``.pth`` file."""
    if path.endswith((".safetensors", ".sft")):
        from safetensors.torch import safe_open

        sd: Dict[str, torch.Tensor] = {}
        with safe_open(path, framework="pt", device="cpu") as f:
            for key in f.keys():
                sd[key] = f.get_tensor(key)
        return sd
    return torch.load(path, map_location="cpu")


def _apply_state_dict(
    model: nn.Module,
    checkpoint_sd: Dict[str, torch.Tensor],
    device: str,
) -> Tuple[List[str], List[str], List[str]]:
    """Apply a checkpoint state dict to a model, handling shape mismatches.

    Returns ``(loaded_keys, mismatched_keys, new_keys)``.
    """
    model_sd = model.state_dict()
    loaded, mismatched, new = [], [], []

    for key, model_tensor in model_sd.items():
        if key in checkpoint_sd:
            ckpt_tensor = checkpoint_sd[key]
            if model_tensor.shape == ckpt_tensor.shape:
                model_sd[key] = ckpt_tensor.to(device=device, dtype=model_tensor.dtype)
                loaded.append(key)
            else:
                mismatched.append(
                    f"{key}: model={list(model_tensor.shape)} vs ckpt={list(ckpt_tensor.shape)}"
                )
        else:
            new.append(key)

    model.load_state_dict(model_sd, strict=False)
    return loaded, mismatched, new
