from __future__ import annotations

# differentiate from field in radiance field sense...
from dataclasses import dataclass, field as data_field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class MeshAssetPaths:
    """Paths to shell meshes produced by the training/export pipeline."""

    base_mesh: Optional[Path] = None
    dilation_mesh: Optional[Path] = None
    erosion_mesh: Optional[Path] = None


@dataclass(slots=True)
class TraceConfig:
    """OptiX narrow band tracing configuration."""

    n_sample_max: int = 16
    n_depth_peeling_max: int = 20
    single_sample_threshold: float = 0.02
    sample_spacing: float = 0.01
    min_interval_size: float = 1e-4
    compress_same_flag_runs: bool = True


@dataclass(slots=True)
class RendererFieldConfig:
    """mlp/hashgrid configuration."""
    geo_feat_dim: int = 26
    appearance_embedding_dim: int = 32
    use_average_appearance_embedding: bool = False
    use_reflections: bool = False
    use_n_dot_v: bool = False
    rgb_padding: float = 0.001
    use_grid_feature: bool = True
    num_layers: int = 1
    hidden_dim: int = 64
    num_layers_color: int = 2
    hidden_dim_color: int = 64
    num_levels: int = 14
    max_res: int = 4096
    base_res: int = 16
    log2_hashmap_size: int = 22
    hash_features_per_level: int = 2
    hash_smoothstep: bool = True
    kernel_size_transform_mode: str = "exp"
    kernel_size_transform_exp_factor: float = 10.0
    pred_gradient_alpha_min_norm: float = 1e-3
    pred_gradient_alpha_max_norm: float = 10.0
    color_normal_source: str = "pred"
    drjit_direct_torch_bridge: bool = True
    use_drjit_geo_mlp: bool = True
    use_drjit_color_mlp: bool = True
    use_drjit_hash_encoding: bool = False
    drjit_hashgrid_torchngp_compat: bool = False
    drjit_mlp_mode: str = "fp16_fast"


@dataclass(slots=True)
class ProfilingConfig:
    strict_timing: bool = False
    drjit_kernel_history: bool = False
    trace_blocking: bool = False
    geo_island_profile: bool = False
    geo_island_prewarm: bool = True
    # Per-stage host waits and the active-ray count; off keeps the frame asynchronous.
    render_stats: bool = False


@dataclass(slots=True)
class RenderConfig:
    camera_index: int = 0
    ray_t_near: float = 0.05
    ray_t_far: float = 4.0
    pixel_center_offset: float = 0.5
    white_background: bool = True
    alpha_mode: str = "pred_gradient"


@dataclass(slots=True)
class RendererAssetConfig:
    config_root: Optional[Path] = None
    checkpoint: Optional[Path] = None
    camera_path: Optional[Path] = None
    provenance_config: Optional[Path] = None
    config_path: Optional[Path] = None
    cuda_src_filename: str = "narrow_band.cu"


@dataclass(slots=True)
class RendererConfig:
    mesh_assets: MeshAssetPaths = data_field(default_factory=MeshAssetPaths)
    trace: TraceConfig = data_field(default_factory=TraceConfig)
    field: RendererFieldConfig = data_field(default_factory=RendererFieldConfig)
    assets: RendererAssetConfig = data_field(default_factory=RendererAssetConfig)
    render: RenderConfig = data_field(default_factory=RenderConfig)
    profiling: ProfilingConfig = data_field(default_factory=ProfilingConfig)
    use_fastpath: bool = True
    device: str = "cuda"
