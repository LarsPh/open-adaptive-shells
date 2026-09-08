from __future__ import annotations

from dataclasses import dataclass
from typing import Any


def require_drjit() -> Any:
    try:
        import drjit as dr  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Dr.Jit backend requires `drjit`. Install with `uv add drjit` and retry."
        ) from exc
    return dr


@dataclass(slots=True)
class DrJitInterop:
    """Load Dr.Jit only when a GPU renderer object is constructed."""

    dr: Any

    @classmethod
    def create(cls) -> "DrJitInterop":
        return cls(dr=require_drjit())
