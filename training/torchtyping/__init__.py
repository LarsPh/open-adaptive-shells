"""Runtime shim for `torchtyping`.

This repo historically used `torchtyping` for type annotations (e.g. `TensorType[..., 3]`).
Newer PyTorch versions (including nightly builds needed for RTX 5090 / sm_120) break the
upstream `torchtyping` package at import time with:

  RuntimeError: Cannot subclass _TensorBase directly

For SDFStudio/Nerfstudio runtime, `TensorType` is only used for annotations, so we provide
this minimal, import-safe shim that supports subscription syntax and evaluates to
`torch.Tensor`.
"""

from __future__ import annotations

import torch


class TensorType:
    """Annotation-only placeholder.

    Allows forms like `TensorType[..., 3]` and returns `torch.Tensor` so annotations
    remain valid without requiring the upstream `torchtyping` runtime behavior.
    """

    def __class_getitem__(cls, _item):  # noqa: D401
        return torch.Tensor


__all__ = ["TensorType"]

