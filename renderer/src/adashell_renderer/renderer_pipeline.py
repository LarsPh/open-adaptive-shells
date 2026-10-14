from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

from adashell_renderer.drjit_backend.optix_metadata_sampler import OptixMetadataSamplerCore
from adashell_renderer.profiling import ProfileBook
from adashell_renderer.optix_trace import ProjectOptixRayTracer
from adashell_renderer.types import DrRenderBuffer, RayBatch, TraceHitBuffers


@dataclass(slots=True)
class RendererPipeline:
    trace_stage: ProjectOptixRayTracer
    sampler: OptixMetadataSamplerCore
    field_island: Any
    fast_render_aux_outputs: bool = False
    fast_render_diagnostic_mode: str = "none"
    collect_stats: bool = False

    def _count_active_metadata_rays(self, trace_hits: TraceHitBuffers) -> int:
        import cupy as cp  # type: ignore[import-not-found]

        sample_counts = trace_hits.metadata["optix_sample_counts"]
        if not isinstance(sample_counts, cp.ndarray):
            raise TypeError("OptiX metadata sampling requires CuPy arrays.")
        return int(cp.asnumpy(cp.count_nonzero(sample_counts > 0)).item())

    def run_render_buffer(
        self,
        rays: RayBatch,
        profile: ProfileBook | None = None,
        before_field: Callable[[], None] | None = None,
        trace_hits: TraceHitBuffers | None = None,
        trace_ms: float = 0.0,
    ) -> DrRenderBuffer:
        """Render one frame.

        `trace_hits` lets a caller launch the trace earlier, from `trace()`;
        `trace_ms` is then its launch time for the profile.

        `before_field` runs after the sample count has been read back and before
        the field kernels are submitted. That readback already waited for all
        earlier GPU work, so the callback can consume the previous frame's
        buffers without another host wait, and its GPU work runs ahead of this
        frame's field network.
        """
        pipeline_wall_start = time.perf_counter()
        profile = profile or ProfileBook()
        profile.set_counter("num_rays", rays.num_rays)

        if trace_hits is None:
            with profile.time("trace_ms"):
                trace_hits = self.trace_stage.trace(rays)
        else:
            profile.add_time_ms("trace_ms", trace_ms)
        # Counting reads back to the host, so it waits for the trace; only stats mode pays for it.
        active_ray_count = self._count_active_metadata_rays(trace_hits) if self.collect_stats else None
        if active_ray_count is not None:
            profile.set_counter("active_rays", float(active_ray_count))

        sample_start = time.perf_counter()
        packed = self.sampler.sample(rays, trace_hits)
        sample_elapsed_ms = (time.perf_counter() - sample_start) * 1000.0
        profile.add_time_ms("sample_ms", sample_elapsed_ms)
        packed.metadata["source_num_rays"] = float(rays.num_rays)
        budget_per_ray = float(packed.metadata.get("budget_per_ray", 0.0))
        if active_ray_count is not None:
            packed.metadata["active_rays"] = float(active_ray_count)
            packed.metadata["active_sample_budget"] = float(active_ray_count) * budget_per_ray
        packed.metadata["sample_budget"] = float(rays.num_rays) * budget_per_ray
        valid_samples = float(packed.metadata.get("valid_samples", packed.valid_samples))
        profile.set_counter("valid_samples", valid_samples)

        if before_field is not None:
            before_field()

        field_start = time.perf_counter()
        render_buffer = self.field_island.render_rgb_buffer(
            rays,
            packed,
            materialize_aux_outputs=bool(self.fast_render_aux_outputs),
            diagnostic_mode=str(self.fast_render_diagnostic_mode),
        )
        field_elapsed_ms = (time.perf_counter() - field_start) * 1000.0

        profile.add_time_ms("field_island_ms", field_elapsed_ms)

        profile.add_time_ms("pipeline_wall_ms", (time.perf_counter() - pipeline_wall_start) * 1000.0)
        render_buffer.metadata["profile"] = profile.to_dict()
        render_buffer.metadata["trace"] = trace_hits.metadata
        render_buffer.metadata["packed"] = packed.metadata
        render_buffer.metadata.setdefault("diagnostics", render_buffer.diagnostics)
        return render_buffer
