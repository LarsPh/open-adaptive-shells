from __future__ import annotations

from typing import Any

from adashell_renderer.config import RendererConfig


def require_optix_trace_config(config: RendererConfig) -> None:
    if config.mesh_assets.dilation_mesh is None or config.mesh_assets.erosion_mesh is None:
        raise ValueError("OptiX tracing requires mesh_assets.dilation_mesh and mesh_assets.erosion_mesh.")


def build_project_ray_tracer(config: RendererConfig) -> Any:
    require_optix_trace_config(config)
    from adashell_renderer.optix_trace import ProjectOptixRayTracer

    return ProjectOptixRayTracer(
        mesh_assets=config.mesh_assets,
        trace_config=config.trace,
        cuda_src_filename=config.assets.cuda_src_filename,
        strict_timing=config.profiling.strict_timing,
        trace_blocking=config.profiling.trace_blocking,
    )
