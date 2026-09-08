from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from adashell_renderer.config import MeshAssetPaths, TraceConfig
from adashell_renderer.optix_utils import (
    create_acceleration_structure,
    create_context,
    create_module,
    create_pipeline,
    create_program_groups,
    create_sbt,
    launch_pipeline_ray_batch,
)
from adashell_renderer.types import RayBatch, TraceHitBuffers

log = logging.getLogger(__name__)


def _require_path(path: Optional[Path], name: str) -> Path:
    if path is None:
        raise ValueError(f"{name} is required.")
    return path


@dataclass(slots=True)
class ProjectOptixRayTracer:
    mesh_assets: MeshAssetPaths
    trace_config: TraceConfig
    cuda_src_filename: str = "narrow_band.cu"
    strict_timing: bool = False
    trace_blocking: bool = True
    pipeline: Any = field(init=False, repr=False)
    sbt: Any = field(init=False, repr=False)
    dilation_gas: Any = field(init=False, repr=False)
    erosion_gas: Any = field(init=False, repr=False)
    _last_trace_profile: Optional[dict] = field(init=False, default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            import optix as ox  # type: ignore[import-not-found]
            import trimesh  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Project-owned tracing requires `optix` and `trimesh` in the new renderer environment."
            ) from exc

        ctx = create_context(log=log)
        pipeline_options = ox.PipelineCompileOptions(
            traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_SINGLE_GAS,
            num_payload_values=2,
            num_attribute_values=2,
            exception_flags=ox.ExceptionFlags.NONE,
            pipeline_launch_params_variable_name="params",
        )
        cuda_src_path = Path(__file__).with_name(self.cuda_src_filename)
        module = create_module(ctx, pipeline_options, cuda_src_path)
        program_groups = create_program_groups(ctx, module)
        self.pipeline = create_pipeline(ctx, program_groups, pipeline_options)
        self.sbt = create_sbt(program_groups)

        dilation_path = _require_path(self.mesh_assets.dilation_mesh, "mesh_assets.dilation_mesh")
        erosion_path = _require_path(self.mesh_assets.erosion_mesh, "mesh_assets.erosion_mesh")
        dilation_mesh = trimesh.load(dilation_path, file_type="ply")
        erosion_mesh = trimesh.load(erosion_path, file_type="ply")
        self.dilation_gas = create_acceleration_structure(ctx, dilation_mesh)
        self.erosion_gas = create_acceleration_structure(ctx, erosion_mesh)
        self._last_trace_profile = None

    def _sync_cuda_for_profile(self, rays: RayBatch) -> float:
        if not self.strict_timing:
            return 0.0
        sync_start = time.perf_counter()
        rays.origins.device.synchronize()
        return (time.perf_counter() - sync_start) * 1000.0

    def trace(self, rays: RayBatch) -> TraceHitBuffers:
        trace_pre_sync_ms = self._sync_cuda_for_profile(rays)
        launch_start = time.perf_counter()
        _, hit_t, hit_count, erosion_first_t, dilation_hit_flag, pack_metadata, launch_profile = launch_pipeline_ray_batch(
            self.pipeline,
            self.sbt,
            self.dilation_gas,
            self.erosion_gas,
            rays.origins,
            rays.directions,
            rays.t_near,
            rays.t_far,
            n_depth_peeling_max=self.trace_config.n_depth_peeling_max,
            blocking_launch=bool(self.trace_blocking),
            n_sample_max=self.trace_config.n_sample_max,
            min_interval_size=self.trace_config.min_interval_size,
            single_sample_threshold=self.trace_config.single_sample_threshold,
            sample_spacing=self.trace_config.sample_spacing,
            compress_same_flag_runs=self.trace_config.compress_same_flag_runs,
        )
        trace_post_sync_ms = self._sync_cuda_for_profile(rays)
        trace_total_ms = (time.perf_counter() - launch_start) * 1000.0
        input_prep_ms = float(launch_profile.get("input_prep_ms", 0.0))
        optix_device_ms = float(launch_profile.get("optix_device_ms", 0.0))
        self._last_trace_profile = {
            "trace_call_ms": trace_total_ms,
            "trace_pre_sync_ms": trace_pre_sync_ms,
            "input_prep_ms": input_prep_ms,
            "optix_launch_ms": optix_device_ms,
            "trace_launch_block_ms": float(launch_profile.get("trace_launch_block_ms", optix_device_ms)),
            "trace_post_sync_ms": trace_post_sync_ms,
            "trace_h2d_ms": input_prep_ms,
            "trace_return_ms": 0.0,
            "trace_return_to_torch_ms": 0.0,
            "zero_copy_optix_interop": float(1.0 if launch_profile.get("interop_zero_copy", False) else 0.0),
            "trace_launch_blocking": float(1.0 if self.trace_blocking else 0.0),
        }
        interval_starts, interval_ends, interval_counts, sample_counts = pack_metadata
        metadata = {
            "ray_tmin": rays.t_near,
            "ray_tmax": rays.t_far,
            "launch_profile": launch_profile,
            "trace_profile": dict(self._last_trace_profile),
            "trace_pack_backend": "optix_metadata",
            "optix_interval_starts": interval_starts,
            "optix_interval_ends": interval_ends,
            "optix_interval_counts": interval_counts,
            "optix_sample_counts": sample_counts,
        }
        return TraceHitBuffers(
            hit_t=hit_t,
            hit_count=hit_count,
            inner_first_t=erosion_first_t,
            outer_hit_flag=dilation_hit_flag,
            metadata=metadata,
        )

    def get_last_trace_profile(self) -> Optional[dict]:
        return self._last_trace_profile
