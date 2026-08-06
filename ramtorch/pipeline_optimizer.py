"""
ramtorch.pipeline_optimizer
---------------------------
Parallel per-stage optimizer for pipeline-parallel training.

Why
~~~
In a pipeline, each stage's parameters live on their own GPU. A normal optimizer
steps *all* parameters sequentially on the host — but the per-stage updates are
completely independent (different params, different devices, no shared state).
So we can step each stage's optimizer **in parallel**, one worker thread per
stage, each pinned to its stage's CUDA device. This overlaps the optimizer
compute across GPUs the same way the pipeline overlaps forward/backward.

Design
~~~~~~
``PipelineOptimizer`` holds one ``torch.optim.Optimizer`` per pipeline stage
(each constructed over only that stage's parameters) and a **persistent worker
thread per stage**. Threads are created once at construction and reused every
step: ``step()``/``zero_grad()`` just release an event, the workers run their
stage's optimizer concurrently, and the caller waits for all to finish. This
avoids the per-step thread-spawn/join overhead that would otherwise dominate
small optimizer steps.

Thread safety: each stage's optimizer only touches its own stage's parameters
and device, so concurrent steps are safe. The GIL is released during CUDA kernel
launches, so the per-stage optimizer kernels genuinely overlap across GPUs.

Usage
~~~~~
    from ramtorch import Pipeline, PipelineOptimizer

    pipe = Pipeline(stage_modules=[s0, s1], devices=["cuda:0", "cuda:1"])
    opt = PipelineOptimizer(
        pipe.stages,
        lambda params: torch.optim.Adam(params, lr=1e-3),
    )

    for x, y in loader:
        result = pipe.step(x, targets=y, n_microbatches=4, loss_fn=loss_fn)
        result.flush_grads()
        opt.step()                 # parallel across stages
        opt.zero_grad()            # parallel across stages

    opt.close()                    # shut down the worker threads when done

Pass a list of factories for per-stage hyperparameters::

    opt = PipelineOptimizer(
        pipe.stages,
        [lambda p: torch.optim.Adam(p, lr=1e-3),
         lambda p: torch.optim.Adam(p, lr=5e-4)],
    )
"""

from __future__ import annotations

import threading
from typing import Callable, List, Optional, Sequence, Union

import torch

from .pipeline import Stage

__all__ = ["PipelineOptimizer"]

_OptFactory = Callable[[Sequence[torch.nn.Parameter]], torch.optim.Optimizer]


class _StageWorker(threading.Thread):
    """Persistent per-stage worker: waits for a command, runs it, signals done."""

    def __init__(self, stage: Stage, optimizer: torch.optim.Optimizer):
        super().__init__(daemon=True, name=f"opt-stage-{stage.stage_index}")
        self.stage = stage
        self.optimizer = optimizer
        self._cmd = threading.Event()      # set -> run the pending command
        self._done = threading.Event()     # set -> command finished
        self._done.set()
        self._shutdown = False
        self._fn: Optional[Callable] = None
        self.error: Optional[BaseException] = None

    def run(self):
        dev = self.stage.device
        if dev.type == "cuda":
            torch.cuda.set_device(dev)  # pin this thread to its stage's device
        while True:
            self._cmd.wait()
            self._cmd.clear()
            if self._shutdown:
                self._done.set()
                return
            try:
                self._fn()
            except BaseException as e:  # noqa: BLE001
                self.error = e
            self._done.set()

    def dispatch(self, fn: Callable):
        """Run ``fn`` on this worker (assumes the previous command finished)."""
        self._done.clear()
        self._fn = fn
        self._cmd.set()

    def wait_done(self):
        self._done.wait()

    def shutdown(self):
        self._done.wait()
        self._shutdown = True
        self._cmd.set()


class PipelineOptimizer:
    """
    One optimizer + one persistent worker thread per pipeline stage, stepped in
    parallel.

    Parameters
    ----------
    stages : list[Stage]
        The pipeline's ``Stage`` objects (``pipe.stages``). Each owns its
        parameters on its own device.
    optimizer_factory : callable | list[callable]
        Either a single factory ``params -> Optimizer`` applied to every stage,
        or a list of factories, one per stage (for per-stage hyperparameters).
    """

    def __init__(
        self,
        stages: List[Stage],
        optimizer_factory: Union[_OptFactory, Sequence[_OptFactory]],
    ):
        self.stages = list(stages)
        n = len(self.stages)

        if callable(optimizer_factory):
            factories = [optimizer_factory] * n
        else:
            factories = list(optimizer_factory)
            if len(factories) != n:
                raise ValueError(
                    f"need {n} optimizer factories (one per stage), "
                    f"got {len(factories)}"
                )

        self.optimizers: List[torch.optim.Optimizer] = [
            factories[i](st.params) for i, st in enumerate(self.stages)
        ]
        self._workers = [
            _StageWorker(st, opt) for st, opt in zip(self.stages, self.optimizers)
        ]
        for w in self._workers:
            w.start()
        self._closed = False

    # ── Parallel dispatch ─────────────────────────────────────────────────────
    def _run_all(self, make_fn):
        """Dispatch ``make_fn(i)`` to every worker and wait for all to finish."""
        for i, w in enumerate(self._workers):
            w.dispatch(make_fn(i))
        for w in self._workers:
            w.wait_done()
        for w in self._workers:
            if w.error is not None:
                err = w.error
                w.error = None
                raise RuntimeError("optimizer worker failed") from err
        # Drain device work so the caller sees completed updates.
        for dev in {st.device for st in self.stages if st.device.type == "cuda"}:
            torch.cuda.synchronize(dev)

    # ── Public API (mirrors torch.optim.Optimizer) ────────────────────────────
    def step(self, closure: Optional[Callable] = None):
        """Step every stage's optimizer in parallel."""
        if closure is not None:
            # Closures are a torch.optim.Optimizer feature we don't parallelize;
            # run sequentially to preserve semantics.
            for opt in self.optimizers:
                opt.step(closure)
            return
        self._run_all(lambda i: self.optimizers[i].step)

    def zero_grad(self, set_to_none: bool = True):
        """Zero every stage's grads in parallel."""
        self._run_all(
            lambda i: (lambda: self.optimizers[i].zero_grad(set_to_none=set_to_none))
        )

    # ── Convenience passthroughs ──────────────────────────────────────────────
    @property
    def param_groups(self):
        """All param groups across stages (flattened)."""
        return [g for opt in self.optimizers for g in opt.param_groups]

    def state_dict(self):
        """Combined state dict: {stage_index: optimizer.state_dict()}."""
        return {i: opt.state_dict() for i, opt in enumerate(self.optimizers)}

    def load_state_dict(self, state_dict):
        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(state_dict[i])

    def set_lr(self, lr: float):
        """Set the learning rate on every stage's optimizer."""
        for opt in self.optimizers:
            for g in opt.param_groups:
                g["lr"] = lr

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    def close(self):
        """Shut down the persistent worker threads."""
        if self._closed:
            return
        for w in self._workers:
            w.shutdown()
        for w in self._workers:
            w.join()
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
