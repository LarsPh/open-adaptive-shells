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


class PackIntervalsCudaLauncher:
    """CuPy RawModule launcher for pack-interval CUDA scaffold."""

    def __init__(self, source_path: Path | None = None, threads_per_block: int = 128) -> None:
        self.source_path = source_path or Path(__file__).with_name("pack_kernels.cu")
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
            name_expressions=("pack_interval_to_sample_bins_kernel",),
        )
        self._kernel = self._module.get_function("pack_interval_to_sample_bins_kernel")
        return self._kernel

    def launch(
        self,
        interval_starts: torch.Tensor,
        interval_ends: torch.Tensor,
        interval_mask: torch.Tensor,
        interval_count: torch.Tensor,
        ray_fars: torch.Tensor,
        n_sample_max: int,
        single_sample_threshold: float,
        sample_spacing: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if interval_starts.ndim != 3:
            raise ValueError(f"Expected interval_starts with shape (H, W, I), got {tuple(interval_starts.shape)}.")
        if not all(
            torch.is_tensor(t) and t.is_cuda for t in (interval_starts, interval_ends, interval_mask, interval_count, ray_fars)
        ):
            raise ValueError("All pack-kernel inputs must be CUDA torch tensors.")

        h, w, max_intervals = [int(v) for v in interval_starts.shape]
        n_rays = int(h * w)
        n_total = int(n_sample_max * max_intervals)
        device = interval_starts.device

        interval_starts_i = _prepare_tensor(interval_starts.reshape(n_rays, max_intervals), torch.float32)
        interval_ends_i = _prepare_tensor(interval_ends.reshape(n_rays, max_intervals), torch.float32)
        interval_mask_u8_i = _prepare_tensor(interval_mask.to(torch.uint8).reshape(n_rays, max_intervals), torch.uint8)
        interval_count_i = _prepare_tensor(interval_count.reshape(n_rays), torch.int32)
        ray_fars_i = _prepare_tensor(ray_fars.reshape(n_rays), torch.float32)

        far_prefill = ray_fars_i[:, None].expand(n_rays, n_total)
        packed_starts = far_prefill.clone()
        packed_ends = far_prefill.clone()
        packed_eval_t = far_prefill.clone()
        packed_mask_u8 = torch.zeros((n_rays, n_total), dtype=torch.uint8, device=device)

        kernel = self._load()
        blocks = ((n_rays + self.threads_per_block - 1) // self.threads_per_block,)
        threads = (self.threads_per_block,)

        torch_stream = torch.cuda.current_stream(device=device)
        with cp.cuda.ExternalStream(torch_stream.cuda_stream):
            kernel(
                blocks,
                threads,
                (
                    _torch_to_cupy_view(interval_starts_i),
                    _torch_to_cupy_view(interval_ends_i),
                    _torch_to_cupy_view(interval_mask_u8_i),
                    _torch_to_cupy_view(interval_count_i),
                    np.int32(n_rays),
                    np.int32(max_intervals),
                    np.int32(n_sample_max),
                    np.int32(n_total),
                    np.float32(single_sample_threshold),
                    np.float32(sample_spacing),
                    _torch_to_cupy_view(packed_starts),
                    _torch_to_cupy_view(packed_ends),
                    _torch_to_cupy_view(packed_mask_u8),
                    _torch_to_cupy_view(packed_eval_t),
                ),
            )

        packed_mask = (packed_mask_u8 > 0).reshape(h, w, n_total)
        return (
            packed_starts.reshape(h, w, n_total),
            packed_ends.reshape(h, w, n_total),
            packed_mask,
            packed_eval_t.reshape(h, w, n_total),
        )
