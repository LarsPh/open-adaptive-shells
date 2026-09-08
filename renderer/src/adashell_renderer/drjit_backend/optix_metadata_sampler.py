from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from adashell_renderer.config import TraceConfig
from adashell_renderer.drjit_backend.interop import DrJitInterop
from adashell_renderer.types import IntervalPack, RayBatch, TraceHitBuffers


_TRACE_METADATA_FLAT_FILL_CUDA = r"""
extern "C" __global__ void adashell_trace_metadata_flat_fill(
    const float* __restrict__ interval_starts,
    const float* __restrict__ interval_ends,
    const int* __restrict__ interval_counts,
    const int* __restrict__ sample_counts,
    const int* __restrict__ ray_offsets,
    float* __restrict__ flat_starts,
    float* __restrict__ flat_ends,
    float* __restrict__ flat_eval_t,
    int* __restrict__ flat_ray_indices,
    const int n_rays,
    const int max_intervals,
    const int n_sample_max,
    const float single_sample_threshold,
    const float sample_spacing
) {
    const int ray = blockIdx.x * blockDim.x + threadIdx.x;
    if (ray >= n_rays) {
        return;
    }

    const int interval_count = interval_counts[ray];
    const int ray_offset = ray_offsets[ray];
    const int expected_samples = sample_counts[ray];
    int cursor = 0;

    for (int interval = 0; interval < interval_count && interval < max_intervals; ++interval) {
        const int slot = ray * max_intervals + interval;
        const float enter = interval_starts[slot];
        const float exit = interval_ends[slot];
        const float interval_len = exit - enter;
        int n_samples = (int)ceilf(fmaxf(interval_len - single_sample_threshold, 0.0f) / sample_spacing) + 1;
        if (n_samples < 1) {
            n_samples = 1;
        }
        if (n_samples > n_sample_max) {
            n_samples = n_sample_max;
        }

        const float denom = (float)(n_samples + 1);
        const float step = (exit - enter) / denom;
        for (int local = 0; local < n_samples; ++local) {
            if (cursor >= expected_samples) {
                return;
            }
            const float local_f = (float)local;
            const float tau = enter + (local_f + 1.0f) * step;
            const float start = local == 0 ? enter : 0.5f * (tau + (enter + local_f * step));
            const float end = local == n_samples - 1 ? exit : 0.5f * (tau + (enter + (local_f + 2.0f) * step));
            const int out = ray_offset + cursor;
            flat_starts[out] = start;
            flat_ends[out] = end;
            flat_eval_t[out] = tau;
            flat_ray_indices[out] = ray;
            ++cursor;
        }
    }
}
"""


def _require_cupy() -> Any:
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("optix_metadata flat fill requires cupy.") from exc
    return cp


def _to_cupy(cp: Any, array: Any, dtype: Any) -> Any:
    if not isinstance(array, cp.ndarray):
        raise TypeError("OptiX metadata sampling requires CuPy arrays.")
    return cp.ascontiguousarray(array.astype(dtype, copy=False))


_TRACE_METADATA_FLAT_FILL_KERNEL: Any | None = None


def _trace_metadata_flat_fill_cuda(
    *,
    interval_starts: Any,
    interval_ends: Any,
    interval_counts: Any,
    sample_counts: Any,
    ray_offsets: Any,
    n_sample_max: int,
    single_sample_threshold: float,
    sample_spacing: float,
) -> tuple[Any, Any, Any, Any, dict[str, float]]:
    global _TRACE_METADATA_FLAT_FILL_KERNEL
    cp = _require_cupy()
    profile: dict[str, float] = {}
    interval_starts_cp = _to_cupy(cp, interval_starts, cp.float32)
    interval_ends_cp = _to_cupy(cp, interval_ends, cp.float32)
    interval_counts_cp = _to_cupy(cp, interval_counts, cp.int32)
    sample_counts_cp = _to_cupy(cp, sample_counts, cp.int32)
    ray_offsets_cp = _to_cupy(cp, ray_offsets, cp.int32)

    total_start = time.perf_counter()
    if int(sample_counts_cp.size) > 0:
        total_samples = int(cp.asnumpy(ray_offsets_cp[-1] + sample_counts_cp[-1]).item())
    else:
        total_samples = 0
    profile["trace_metadata_total_samples_ms"] = (time.perf_counter() - total_start) * 1000.0
    flat_starts = cp.empty((total_samples,), dtype=cp.float32)
    flat_ends = cp.empty((total_samples,), dtype=cp.float32)
    flat_eval_t = cp.empty((total_samples,), dtype=cp.float32)
    flat_ray_indices = cp.empty((total_samples,), dtype=cp.int32)
    if total_samples <= 0:
        return flat_starts, flat_ends, flat_eval_t, flat_ray_indices, profile

    compile_start = time.perf_counter()
    if _TRACE_METADATA_FLAT_FILL_KERNEL is None:
        _TRACE_METADATA_FLAT_FILL_KERNEL = cp.RawKernel(
            _TRACE_METADATA_FLAT_FILL_CUDA,
            "adashell_trace_metadata_flat_fill",
        )
    profile["trace_metadata_cuda_compile_ms"] = (time.perf_counter() - compile_start) * 1000.0

    launch_start = time.perf_counter()
    stream = cp.cuda.get_current_stream()
    n_rays = int(interval_starts_cp.shape[0])
    max_intervals = int(interval_starts_cp.shape[1])
    block_size = 256
    grid_size = (n_rays + block_size - 1) // block_size
    with stream:
        _TRACE_METADATA_FLAT_FILL_KERNEL(
            (grid_size,),
            (block_size,),
            (
                interval_starts_cp,
                interval_ends_cp,
                interval_counts_cp,
                sample_counts_cp,
                ray_offsets_cp,
                flat_starts,
                flat_ends,
                flat_eval_t,
                flat_ray_indices,
                np.int32(n_rays),
                np.int32(max_intervals),
                np.int32(n_sample_max),
                np.float32(single_sample_threshold),
                np.float32(sample_spacing),
            ),
        )
    profile["trace_metadata_cuda_launch_ms"] = (time.perf_counter() - launch_start) * 1000.0
    sync_start = time.perf_counter()
    stream.synchronize()
    profile["trace_metadata_cuda_sync_ms"] = (time.perf_counter() - sync_start) * 1000.0
    return flat_starts, flat_ends, flat_eval_t, flat_ray_indices, profile


@dataclass(slots=True)
class OptixMetadataSamplerCore:
    """Expand OptiX interval metadata into valid-only ray-major sample buffers."""

    trace_config: TraceConfig
    capture_kernel_history: bool = False
    _interop: DrJitInterop = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._interop = DrJitInterop.create()

    def _summarize_kernel_history(self, history: list[dict[str, Any]]) -> dict[str, float | str]:
        execution_time = 0.0
        codegen_time = 0.0
        backend_time = 0.0
        uses_optix = 0.0
        jit_kernels = 0.0
        top_exec_type = ""
        top_codegen_type = ""
        top_exec_time = -1.0
        top_codegen_time = -1.0
        for item in history:
            item_exec = float(item.get("execution_time", 0.0))
            item_codegen = float(item.get("codegen_time", 0.0))
            item_backend = float(item.get("backend_time", 0.0))
            item_type = str(item.get("type", ""))
            execution_time += item_exec
            codegen_time += item_codegen
            backend_time += item_backend
            uses_optix += float(item.get("uses_optix", 0))
            jit_kernels += 1.0 if item_type.endswith("JIT") else 0.0
            if item_exec > top_exec_time:
                top_exec_time = item_exec
                top_exec_type = item_type
            if item_codegen > top_codegen_time:
                top_codegen_time = item_codegen
                top_codegen_type = item_type
        return {
            "kernel_history_count": float(len(history)),
            "kernel_history_execution_ms": execution_time,
            "kernel_history_codegen_ms": codegen_time,
            "kernel_history_backend_ms": backend_time,
            "kernel_history_optix_count": uses_optix,
            "kernel_history_jit_count": jit_kernels,
            "kernel_history_top_exec_ms": max(top_exec_time, 0.0),
            "kernel_history_top_exec_type": top_exec_type,
            "kernel_history_top_codegen_ms": max(top_codegen_time, 0.0),
            "kernel_history_top_codegen_type": top_codegen_type,
        }

    def sample(self, rays: RayBatch, trace_hits: TraceHitBuffers) -> IntervalPack:
        """Return CuPy float32 samples and int32 ray-major offsets/counts.

        The pack owns its CuPy arrays and the Dr.Jit views in `_drjit_cache`.
        CUDA fill completes before Dr.Jit imports them; Dr.Jit evaluation is
        synchronized before returning. No padded samples reach the field.
        """
        required = (
            "optix_interval_starts",
            "optix_interval_ends",
            "optix_interval_counts",
            "optix_sample_counts",
        )
        missing = [key for key in required if key not in trace_hits.metadata]
        if missing:
            raise ValueError(f"OptiX metadata sampling requires trace_hits.metadata keys: {missing}")

        cp = _require_cupy()
        interval_starts = _to_cupy(cp, trace_hits.metadata["optix_interval_starts"], cp.float32)
        interval_ends = _to_cupy(cp, trace_hits.metadata["optix_interval_ends"], cp.float32)
        interval_counts = _to_cupy(cp, trace_hits.metadata["optix_interval_counts"], cp.int32)
        optix_sample_counts = _to_cupy(cp, trace_hits.metadata["optix_sample_counts"], cp.int32)

        n_rays, max_intervals = [int(v) for v in interval_starts.shape]
        if n_rays != rays.num_rays or interval_ends.shape != interval_starts.shape:
            raise ValueError("OptiX interval buffers must match the input ray batch.")
        if interval_counts.shape != (n_rays,) or optix_sample_counts.shape != (n_rays,):
            raise ValueError("OptiX count buffers must contain one count per input ray.")
        n_sample_max = int(self.trace_config.n_sample_max)
        n_total = int(max_intervals * n_sample_max)

        offsets_start = time.perf_counter()
        ray_offsets = cp.cumsum(optix_sample_counts, dtype=cp.int32)
        ray_offsets -= optix_sample_counts
        offsets_ms = (time.perf_counter() - offsets_start) * 1000.0

        fill_start = time.perf_counter()
        flat_starts, flat_ends, flat_eval_t, flat_ray_indices_i32, fill_profile = _trace_metadata_flat_fill_cuda(
            interval_starts=interval_starts,
            interval_ends=interval_ends,
            interval_counts=interval_counts,
            sample_counts=optix_sample_counts,
            ray_offsets=ray_offsets,
            n_sample_max=n_sample_max,
            single_sample_threshold=float(self.trace_config.single_sample_threshold),
            sample_spacing=float(self.trace_config.sample_spacing),
        )
        flat_fill_ms = (time.perf_counter() - fill_start) * 1000.0
        packed_valid_samples = int(flat_starts.shape[0])
        packed_info = cp.stack((ray_offsets, optix_sample_counts), axis=-1).astype(cp.int32, copy=False)

        dr = self._interop.dr
        history_start = len(dr.kernel_history()) if self.capture_kernel_history else 0
        kernel_summary: dict[str, float | str] | None = None
        with dr.scoped_set_flag(dr.JitFlag.KernelHistory, self.capture_kernel_history):
            build_dr_start = time.perf_counter()
            sample_cursor_dr = dr.cuda.Int(optix_sample_counts)
            flat_starts_dr = dr.cuda.Float(flat_starts)
            flat_ends_dr = dr.cuda.Float(flat_ends)
            flat_eval_t_dr = dr.cuda.Float(flat_eval_t)
            flat_ray_indices_dr = dr.cuda.UInt(flat_ray_indices_i32)
            ray_offsets_dr = dr.cuda.UInt(ray_offsets)
            build_dr_ms = (time.perf_counter() - build_dr_start) * 1000.0
            eval_start = time.perf_counter()
            dr.eval(
                sample_cursor_dr,
                flat_starts_dr,
                flat_ends_dr,
                flat_eval_t_dr,
                flat_ray_indices_dr,
                ray_offsets_dr,
            )
            eval_ms = (time.perf_counter() - eval_start) * 1000.0
            sync_start = time.perf_counter()
            dr.sync_thread()
            sync_ms = (time.perf_counter() - sync_start) * 1000.0
            if self.capture_kernel_history:
                kernel_summary = self._summarize_kernel_history(dr.kernel_history()[history_start:])

        packed_meta: dict[str, Any] = {
            "backend": "optix_metadata_cuda_flat",
            "trace_pack_backend": "optix_metadata",
            "trace_pack_backend_optix_metadata": 1.0,
            "trace_metadata_flat_fill": 1.0,
            "trace_metadata_offsets_ms": float(offsets_ms),
            "trace_metadata_fill_ms": float(flat_fill_ms),
            "trace_metadata_build_dr_ms": float(build_dr_ms),
            "trace_metadata_eval_ms": float(eval_ms),
            "trace_metadata_sync_ms": float(sync_ms),
            "valid_samples": float(packed_valid_samples),
            "sample_budget": float(n_rays * n_total),
            "active_sample_budget": float(n_rays * n_total),
            "budget_per_ray": float(n_total),
            "materialized_from_dense_mask": 0.0,
            "metadata_direct_cupy_pack": 1.0,
            "sample_counts": optix_sample_counts,
            **fill_profile,
        }
        if kernel_summary is not None:
            packed_meta.update(kernel_summary)

        packed = IntervalPack(
            starts=flat_starts,
            ends=flat_ends,
            eval_t=flat_eval_t,
            ray_indices=flat_ray_indices_i32,
            packed_info=packed_info,
            metadata={
                **packed_meta,
                "num_rays": float(n_rays),
                "_drjit_cache": {
                    "starts": flat_starts_dr,
                    "ends": flat_ends_dr,
                    "eval_t": flat_eval_t_dr,
                    "ray_indices": flat_ray_indices_dr,
                    "sample_counts": sample_cursor_dr,
                    "ray_offsets": ray_offsets_dr,
                    "deltas": flat_ends_dr - flat_starts_dr,
                },
            },
        )
        return packed
