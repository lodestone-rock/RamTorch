"""
Utility helpers to provide a unified CUDA/HIP accelerator interface.

PyTorch exposes HIP functionality through the CUDA API surface. On ROCm builds
`torch.version.hip` is set, streams/events map to HIP primitives, but the
runtime still expects CUDA-style device identifiers. This module normalizes
device handling so RamTorch can work across both backends.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch

DeviceLike = Union[str, int, torch.device]


def _coerce_device(device: Optional[DeviceLike]) -> torch.device:
    """
    Convert user-supplied device specifiers into a torch.device with an index.
    """
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "A CUDA/HIP device is required but no GPU is available."
            )
        index = torch.cuda.current_device()
        return torch.device("cuda", index)

    if isinstance(device, torch.device):
        resolved = device
    elif isinstance(device, str):
        spec = device.strip()
        lower_spec = spec.lower()
        if lower_spec.startswith("hip"):
            spec = "cuda" + spec[3:]
        resolved = torch.device(spec)
    elif isinstance(device, int):
        resolved = torch.device("cuda", device)
    else:
        raise TypeError(f"Unsupported device specifier: {device!r}")

    if resolved.type not in {"cuda", "hip"}:
        raise ValueError(f"Expected a CUDA/HIP device, got {resolved}")

    if resolved.index is None:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "Unable to infer GPU index because no CUDA/HIP device is available."
            )
        resolved = torch.device(resolved.type, torch.cuda.current_device())

    return resolved


def _normalize_runtime_device(device: torch.device) -> torch.device:
    """
    Map logical HIP devices to the CUDA runtime device identifier expected by
    the Stream/Event APIs. CUDA devices pass through unchanged.
    """
    if device.type == "hip":
        index = 0 if device.index is None else device.index
        return torch.device("cuda", index)
    return device


def _detect_backend() -> str:
    """
    Determine whether we're running on CUDA or HIP.
    """
    return "hip" if torch.version.hip is not None else "cuda"


@dataclass(frozen=True)
class Accelerator:
    """
    Lightweight adapter exposing CUDA-style stream/event helpers that work for
    both CUDA and HIP backends.
    """

    logical_device: torch.device
    runtime_device: torch.device
    backend: str

    @classmethod
    def create(cls, device: Optional[DeviceLike] = None) -> "Accelerator":
        logical = _coerce_device(device)
        backend = _detect_backend()
        runtime = _normalize_runtime_device(logical if backend == "hip" else logical)
        # torch.cuda APIs guard against invalid indices; no extra validation needed.
        return cls(logical_device=logical, runtime_device=runtime, backend=backend)

    @property
    def key(self) -> Tuple[str, int]:
        """
        Key used for per-device state caches.
        """
        index = self.runtime_device.index
        if index is None:
            raise RuntimeError("Runtime device must have an index for caching.")
        return (self.backend, index)

    @property
    def tensor_device(self) -> torch.device:
        """
        Device specifier suitable for tensor .to(...) calls.
        """
        return self.runtime_device

    @contextmanager
    def device_context(self):
        """
        Ensure CUDA APIs operate on this accelerator's device.
        """
        with torch.cuda.device(self.runtime_device):
            yield

    def new_stream(self) -> torch.cuda.Stream:
        """
        Create a new stream bound to this accelerator's device.
        """
        return torch.cuda.Stream(device=self.runtime_device)

    def new_event(self, **kwargs) -> torch.cuda.Event:
        """
        Create a new event. Disable timing on HIP where it is unsupported.
        """
        if self.backend == "hip":
            kwargs.setdefault("enable_timing", False)
        return torch.cuda.Event(**kwargs)

    def current_stream(self) -> torch.cuda.Stream:
        """
        Retrieve the current stream for this device.
        """
        return torch.cuda.current_stream(self.runtime_device)

    @contextmanager
    def use_stream(self, stream: torch.cuda.Stream):
        """
        Context manager mirroring torch.cuda.stream with explicit scoping.
        """
        with torch.cuda.stream(stream):
            yield

    def synchronize(self):
        """
        Synchronize this accelerator's device.
        """
        torch.cuda.synchronize(self.runtime_device)
