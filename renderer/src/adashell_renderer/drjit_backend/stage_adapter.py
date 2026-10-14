from __future__ import annotations

import torch

from adashell_renderer.config import RendererConfig
from adashell_renderer.drjit_backend.field_island import DrJitFieldIsland
from adashell_renderer.drjit_backend.optix_metadata_sampler import OptixMetadataSamplerCore
from adashell_renderer.legacy_field import build_legacy_field_from_checkpoint
from adashell_renderer.renderer_pipeline import RendererPipeline
from adashell_renderer.trace_backend import build_project_ray_tracer


def apply_fastpath_defaults(config: RendererConfig) -> None:
    """Apply the release-candidate fastpath preset.

    This mutates user config by design. It keeps the public RC command on the
    known fast route, so callers should only invoke it when `use_fastpath` is
    enabled.
    """

    config.render.alpha_mode = "pred_gradient"
    config.field.color_normal_source = "pred"
    config.field.use_drjit_geo_mlp = True
    config.field.use_drjit_color_mlp = True
    config.field.use_drjit_hash_encoding = True
    config.profiling.strict_timing = False
    config.profiling.trace_blocking = False
    config.profiling.drjit_kernel_history = False


def _require_fastpath_config(config: RendererConfig) -> None:
    if not str(config.device).startswith("cuda"):
        raise ValueError("Fastpath renderer requires a CUDA device.")
    if config.assets.checkpoint is None:
        raise ValueError("Fastpath renderer requires config.assets.checkpoint.")
    if config.render.alpha_mode != "pred_gradient":
        raise ValueError("Fastpath renderer only supports render.alpha_mode='pred_gradient'.")
    if config.field.color_normal_source != "pred":
        raise ValueError("Fastpath renderer only supports predicted-normal color input.")
    if not bool(config.field.use_drjit_hash_encoding):
        raise ValueError("Fastpath renderer requires Dr.Jit hashgrid encoding.")
    if not (bool(config.field.use_drjit_geo_mlp) and bool(config.field.use_drjit_color_mlp)):
        raise ValueError("Fastpath renderer requires Dr.Jit geo/color MLPs.")


def build_fastpath_pipeline(config: RendererConfig) -> RendererPipeline:
    if config.use_fastpath:
        apply_fastpath_defaults(config)
    _require_fastpath_config(config)  # fail fast

    tracer = build_project_ray_tracer(config)
    field = build_legacy_field_from_checkpoint(
        path=config.assets.checkpoint,
        config=config.field,
        device=config.device,
    )
    sampler_core = OptixMetadataSamplerCore(
        config.trace,
        capture_kernel_history=bool(config.profiling.drjit_kernel_history),
        host_sync=bool(config.profiling.render_stats),
    )
    pipeline = RendererPipeline(
        trace_stage=tracer,
        sampler=sampler_core,
        fast_render_aux_outputs=False,
        collect_stats=bool(config.profiling.render_stats),
        field_island=DrJitFieldIsland(
            field_model=field,
            alpha_mode="pred_gradient",
            white_background=config.render.white_background,
            profile_geo=bool(config.profiling.geo_island_profile),
            prewarm=bool(config.profiling.geo_island_prewarm),
        ),
    )
    if torch.cuda.is_available() and str(config.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return pipeline
