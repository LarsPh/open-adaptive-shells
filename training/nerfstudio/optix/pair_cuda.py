from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cupy as cp
import numpy as np
import torch


def _torch_to_cupy_view(tensor: torch.Tensor) -> cp.ndarray:
    if not tensor.is_cuda:
        raise ValueError("Expected CUDA tensor.")
    if not tensor.is_contiguous():
        raise ValueError("Expected contiguous CUDA tensor.")
    return cp.from_dlpack(tensor.detach())


def _prepare_tensor(tensor: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if tensor.dtype != dtype:
        tensor = tensor.to(dtype=dtype)
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    return tensor


class PairIntervalsCudaLauncher:
    """CuPy RawModule launcher for the pair-interval CUDA kernel.

    This wrapper handles:
    - lazy compilation
    - output tensor allocation
    - zero-copy Torch <-> CuPy views via DLPack

    The kernel body in `pair_kernels.cu` is intentionally left as a scaffold.
    """

    def __init__(self, source_path: Path | None = None, threads_per_block: int = 128) -> None:
        self.source_path = source_path or Path(__file__).with_name("pair_kernels.cu")
        self.threads_per_block = int(threads_per_block)
        self._module: cp.RawModule | None = None
        self._kernel = None

    def _load(self):
        if self._kernel is not None:
            return self._kernel
        source = self.source_path.read_text(encoding="utf-8")
        self._module = cp.RawModule(
            code=source,
            options=("-std=c++14",),
            name_expressions=("pair_shell_hit_intervals_kernel",),
        )
        self._kernel = self._module.get_function("pair_shell_hit_intervals_kernel")
        return self._kernel

    def launch(
        self,
        outer_hit_t: torch.Tensor,
        outer_hit_flag: torch.Tensor,
        outer_hit_count: torch.Tensor,
        inner_first_t: torch.Tensor,
        ray_tmin: torch.Tensor,
        min_interval_size: float = 1e-4,
        compress_same_flag_runs: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if outer_hit_t.ndim != 2:
            raise ValueError(f"Expected outer_hit_t with shape (R, max_hits), got {tuple(outer_hit_t.shape)}.")
        if not all(torch.is_tensor(t) and t.is_cuda for t in (outer_hit_t, outer_hit_flag, outer_hit_count, inner_first_t, ray_tmin)):
            raise ValueError("All pair-kernel inputs must be CUDA torch tensors.")

        n_rays, max_hits = [int(v) for v in outer_hit_t.shape]
        max_intervals = (max_hits + 1) // 2
        device = outer_hit_t.device

        outer_hit_t_i = _prepare_tensor(outer_hit_t, torch.float32)
        outer_hit_flag_i = _prepare_tensor(outer_hit_flag, torch.int32)
        outer_hit_count_i = _prepare_tensor(outer_hit_count, torch.int32)
        inner_first_t_i = _prepare_tensor(inner_first_t, torch.float32)
        ray_tmin_i = _prepare_tensor(ray_tmin, torch.float32)

        interval_starts = torch.full((n_rays, max_intervals), -1.0, dtype=torch.float32, device=device)
        interval_ends = torch.full((n_rays, max_intervals), -1.0, dtype=torch.float32, device=device)
        interval_mask_u8 = torch.zeros((n_rays, max_intervals), dtype=torch.uint8, device=device)
        interval_count = torch.zeros((n_rays,), dtype=torch.int32, device=device)

        kernel = self._load()
        blocks = ((n_rays + self.threads_per_block - 1) // self.threads_per_block,)
        threads = (self.threads_per_block,)

        torch_stream = torch.cuda.current_stream(device=device)
        with cp.cuda.ExternalStream(torch_stream.cuda_stream):
            kernel(
                blocks,
                threads,
                (
                    _torch_to_cupy_view(outer_hit_t_i),
                    _torch_to_cupy_view(outer_hit_flag_i),
                    _torch_to_cupy_view(outer_hit_count_i),
                    _torch_to_cupy_view(inner_first_t_i),
                    _torch_to_cupy_view(ray_tmin_i),
                    np.int32(n_rays),
                    np.int32(max_hits),
                    np.int32(max_intervals),
                    np.float32(min_interval_size),
                    np.int32(1 if compress_same_flag_runs else 0),
                    _torch_to_cupy_view(interval_starts),
                    _torch_to_cupy_view(interval_ends),
                    _torch_to_cupy_view(interval_mask_u8),
                    _torch_to_cupy_view(interval_count),
                ),
            )

        # Convert to the same logical types as the current Torch implementation.
        interval_mask = interval_mask_u8 > 0
        return interval_starts, interval_ends, interval_mask, interval_count
