"""
ramtorch.nvme_store
-------------------
File-backed master weights for the NVMe offload tier — pure PyTorch, no
cuFile/GDS or third-party wrappers.

:class:`NvmeTensorStore` reserves one contiguous region on disk (a single
file, page-aligned tensor slots) and exposes each tensor as a *view over an
mmap of that file* via ``torch.UntypedStorage.from_file(shared=True)``:

* Reads page-fault through the OS page cache: cold reads stream from the
  drive at disk speed, hot ones are served from RAM if the kernel has room —
  and get evicted under memory pressure, which is the scenario this tier
  exists for.
* In-place writes (an optimizer step on the mapped params) land in the page
  cache and are written back to the file lazily by the kernel — training
  works unchanged, no explicit read-modify-write cycle needed.
* The mapped tensors are ordinary CPU tensors to the rest of torch:
  ``state_dict()``, ``copy_``, optimizers, everything just works.

This is deliberately NOT GPUDirect Storage: the empirical contention test
(``examples/nvme_h2d_contention_test.py``) showed the disk->GPU path's
host->device hop serializes on the H2D copy engine anyway, so the honest
model — and the simple implementation — is disk -> pinned staging -> GPU on
the loader thread ("slower H2D", see ``ramtorch.offload_simulator``).
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import torch

__all__ = ["NvmeTensorStore"]


class NvmeTensorStore:
    """Reserve space in a file and rehome tensors onto mmap-backed storage.

    Usage::

        store = NvmeTensorStore("/mnt/nvme/scratch/weights.bin")
        mapped = store.write({"0.weight": w, "0.bias": b, ...})
        # mapped["0.weight"] is a CPU tensor viewing the file; assign it to
        # param.data and the original RAM copy is freed
        ...
        store.close()   # unlink the scratch file (mapping stays valid)

    Notes
    -----
    * One-shot: ``write`` lays out and fills the whole file once. The layout
      aligns every tensor to 4096 bytes (page-aligned; also O_DIRECT-friendly
      if the file is ever read by external tooling).
    * Durability is page-cache semantics: the kernel flushes dirty pages on
      its own schedule. This is scratch space for streaming masters, not a
      checkpoint format — keep using ``torch.save`` for checkpoints.
    """

    ALIGN = 4096

    def __init__(self, path: str):
        self.path = str(path)
        self._base: Optional[torch.Tensor] = None  # uint8 view of the file
        self._layout: Dict[str, tuple] = {}        # name -> (offset, nbytes)

    @property
    def nbytes(self) -> int:
        """Reserved file size in bytes (0 before ``write``)."""
        return 0 if self._base is None else self._base.numel()

    def write(self, tensors: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Reserve the file, copy ``tensors`` in, return mmap-backed views.

        The returned tensors have the same name, shape and dtype as the
        inputs but their storage is the file mapping (MAP_SHARED).
        """
        if self._base is not None:
            raise RuntimeError("NvmeTensorStore.write() may only run once")
        if not tensors:
            raise ValueError("no tensors to write")

        off = 0
        for name, t in tensors.items():
            nb = t.numel() * t.element_size()
            self._layout[name] = (off, nb)
            off += -(-nb // self.ALIGN) * self.ALIGN
        total = max(off, self.ALIGN)

        # reserve the space on disk, then map it
        parent = os.path.dirname(os.path.abspath(self.path))
        os.makedirs(parent, exist_ok=True)
        with open(self.path, "wb") as f:
            f.truncate(total)
        storage = torch.UntypedStorage.from_file(
            self.path, shared=True, nbytes=total
        )
        base = torch.empty(0, dtype=torch.uint8)
        base.set_(storage, 0, (total,))
        self._base = base

        out: Dict[str, torch.Tensor] = {}
        with torch.no_grad():
            for name, t in tensors.items():
                out[name] = view = self.tensor(name, t.dtype, t.shape)
                view.copy_(t.detach().to("cpu"))
        return out

    def tensor(self, name: str, dtype: torch.dtype,
               shape: torch.Size) -> torch.Tensor:
        """The mmap-backed tensor for a written slot (a view, zero-copy)."""
        off, nb = self._layout[name]
        v = self._base[off:off + nb].view(dtype)
        return v.view(shape)

    def close(self, unlink: bool = True) -> None:
        """Drop the mapping reference and (by default) delete the file.

        POSIX keeps an unlinked-but-mapped file readable until the last
        tensor viewing it is garbage collected, so existing views stay
        valid; the disk space is reclaimed when they go.
        """
        self._base = None
        self._layout = {}
        if unlink:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass

    def __del__(self):
        # do NOT unlink on GC: the store owner decides scratch-file fate
        self._base = None
