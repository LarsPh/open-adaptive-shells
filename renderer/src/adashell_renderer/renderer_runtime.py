from __future__ import annotations

import time
from typing import Any, Callable

from adashell_renderer.config import RendererConfig
from adashell_renderer.drjit_backend.stage_adapter import build_fastpath_pipeline
from adashell_renderer.profiling import ProfileBook
from adashell_renderer.renderer_pipeline import RendererPipeline
from adashell_renderer.types import DrRenderBuffer, RayBatch


def build_runtime_pipeline(config: RendererConfig) -> RendererPipeline:
    """Build the only public RC runtime: OptiX metadata sampler + Dr field island."""

    return build_fastpath_pipeline(config)


def _merge_profiles(accum: dict[str, float], profile: dict[str, object]) -> None:
    timings = profile.get("stage_ms", profile.get("timings", {}))
    if isinstance(timings, dict):
        for key, value in timings.items():
            accum[key] = accum.get(key, 0.0) + float(value)
    counters = profile.get("counters", {})
    if isinstance(counters, dict):
        for key, value in counters.items():
            accum[key] = accum.get(key, 0.0) + float(value)


def _profile_total_ms(profile: dict[str, float]) -> float:
    # Nested field timings overlap; only the enclosing pipeline timer is a total.
    return float(profile.get("pipeline_wall_ms", 0.0))


def _merge_numeric(accum: dict[str, object], profile: dict[str, Any]) -> None:
    for key, value in profile.items():
        if isinstance(value, (int, float)):
            accum[key] = float(accum.get(key, 0.0)) + float(value)
        elif isinstance(value, str) and key not in accum:
            accum[key] = value


def _collect_render_metadata(
    detail_totals: dict[str, object],
    result_metadata: dict[str, Any],
) -> None:
    trace_profile = result_metadata.get("trace", {}).get("trace_profile", {})
    if isinstance(trace_profile, dict):
        _merge_numeric(detail_totals, trace_profile)

    top_profile = result_metadata.get("profile_stage_ms", {})
    if isinstance(top_profile, dict):
        _merge_numeric(detail_totals, top_profile)

    for section in ("geo", "colors"):
        profile = result_metadata.get(section, {}).get("profile_stage_ms", {})
        if isinstance(profile, dict):
            _merge_numeric(detail_totals, profile)

    kernel_history = result_metadata.get("drjit_kernel_history", {})
    if isinstance(kernel_history, dict):
        _merge_numeric(detail_totals, kernel_history)

    packed_meta = result_metadata.get("packed", {})
    if isinstance(packed_meta, dict):
        _merge_numeric(
            detail_totals,
            {
                key: packed_meta[key]
                for key in (
                    "sample_budget",
                    "budget_per_ray",
                    "valid_samples",
                    "metadata_sparse_pack",
                    "metadata_direct_cupy_pack",
                    "trace_metadata_flat_fill",
                    "trace_metadata_offsets_ms",
                    "trace_metadata_fill_ms",
                    "trace_metadata_total_samples_ms",
                    "trace_metadata_cuda_compile_ms",
                    "trace_metadata_cuda_launch_ms",
                    "trace_metadata_cuda_sync_ms",
                    "trace_metadata_build_dr_ms",
                    "trace_metadata_eval_ms",
                    "trace_metadata_sync_ms",
                    "kernel_history_count",
                    "kernel_history_execution_ms",
                    "kernel_history_codegen_ms",
                    "kernel_history_backend_ms",
                    "kernel_history_optix_count",
                    "kernel_history_jit_count",
                    "kernel_history_top_exec_ms",
                    "kernel_history_top_codegen_ms",
                )
                if key in packed_meta
            },
        )

def render_full_image_buffer(
    pipeline: RendererPipeline,
    rays: RayBatch,
    before_field: Callable[[], None] | None = None,
) -> tuple[DrRenderBuffer, dict[str, float], dict[str, object]]:
    full_start = time.perf_counter()
    profile = ProfileBook()
    render_buffer = pipeline.run_render_buffer(rays, profile=profile, before_field=before_field)
    profile_totals: dict[str, float] = {}
    detail_totals: dict[str, object] = {}
    _merge_profiles(profile_totals, render_buffer.metadata.get("profile", {}))
    _collect_render_metadata(detail_totals, render_buffer.metadata)
    full_wall_ms = (time.perf_counter() - full_start) * 1000.0
    detail_totals["render_full_image_wall_ms"] = full_wall_ms
    detail_totals["render_full_image_profile_total_ms"] = _profile_total_ms(profile_totals)
    detail_totals["render_full_image_profile_gap_ms"] = full_wall_ms - _profile_total_ms(profile_totals)
    return render_buffer, profile_totals, detail_totals
