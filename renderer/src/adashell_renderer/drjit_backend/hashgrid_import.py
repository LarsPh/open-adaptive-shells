from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch

from adashell_renderer.config import RendererFieldConfig


@dataclass(frozen=True, slots=True)
class HashGridLevelLayout:
    level: int
    dr_vertex_offset: int
    dr_vertex_count: int
    dr_param_offset: int
    dr_param_count: int
    tcnn_vertex_offset: int
    tcnn_vertex_count: int
    tcnn_param_offset: int
    tcnn_param_count: int


def dr_smoothstep(x: Any) -> Any:
    # tiny-cuda-nn's Smoothstep interpolation applies the cubic ramp to the
    # local cell coordinate before multilinear corner accumulation:
    # s(x) = x^2 * (3 - 2x)
    return x * x * (3.0 - 2.0 * x)


def _build_hashgrid_cls(*, smoothstep: bool) -> Any:
    import drjit.hashgrid as hg  # type: ignore[import-not-found]
    import drjit as dr  # type: ignore[import-not-found]

    if not smoothstep:
        return hg.HashGridEncoding

    class SmoothstepHashGridEncoding(hg.HashGridEncoding):
        def __call__(self, p, active=True):
            _, PositionFloatXf = self._position_types(p)

            p = PositionFloatXf(p)
            assert len(p) == self.dimension, (
                f"This hash grid expected an input of feature dimension {self.dimension}"
                f" but got {len(p)}."
            )

            grid_offsets = [
                self.ArrayXu([(i >> j) & 1 for j in range(self.dimension)])
                for i in range(2**self.dimension)
            ]

            out_values = [self.StorageFloat(0.0)] * (
                self.n_features_per_level * self.n_levels
            )

            for level_i in range(self.n_levels):
                scale = self._level_scale(level_i)
                p_offset: float = (
                    0.5 if not self.align_corners or self.torchngp_compat else 0.0
                )
                pos = dr.fma(p, scale, p_offset)
                pos0 = self.ArrayXu(dr.floor(pos))

                w1 = dr_smoothstep(pos - pos0)
                w0 = 1.0 - w1

                for offset in grid_offsets:
                    pos_grid = pos0 + offset
                    weight = dr.select(offset == 0, w0, w1)
                    weight = dr.prod(weight, axis=0)

                    index = self.indexing_function(pos_grid, level_i)
                    self._acc_features(level_i, weight, index, out_values, active)

            return self.StorageFloatXf(*out_values) & active

    return SmoothstepHashGridEncoding


def hashgrid_per_level_scale(config: RendererFieldConfig) -> float:
    return math.exp((math.log(config.max_res) - math.log(config.base_res)) / (config.num_levels - 1))


def build_dr_hashgrid_encoding(
    config: RendererFieldConfig,
    *,
    dimension: int = 3,
    param_type: Any | None = None,
    torchngp_compat: bool = False,
    smoothstep: bool = False,
) -> Any:
    from drjit.cuda import Float16  # type: ignore[import-not-found]

    if param_type is None:
        param_type = Float16

    hashgrid_cls = _build_hashgrid_cls(smoothstep=smoothstep)

    return hashgrid_cls(
        param_type,
        dimension,
        n_levels=config.num_levels,
        n_features_per_level=config.hash_features_per_level,
        hashmap_size=2 ** config.log2_hashmap_size,
        base_resolution=config.base_res,
        per_level_scale=hashgrid_per_level_scale(config),
        torchngp_compat=torchngp_compat,
    )


def dr_hashgrid_level_layout(dr_encoding: Any) -> list[HashGridLevelLayout]:
    n_levels = int(dr_encoding.n_levels)
    n_features_per_level = int(dr_encoding.n_features_per_level)

    layouts: list[HashGridLevelLayout] = []
    tcnn_vertex_offset = 0
    tcnn_param_offset = 0

    for level in range(n_levels):
        dr_vertex_offset = int(dr_encoding.level_offset(level))
        dr_next_vertex_offset = int(dr_encoding.level_offset(level + 1))
        dr_vertex_count = dr_next_vertex_offset - dr_vertex_offset
        dr_param_offset = dr_vertex_offset * n_features_per_level
        dr_param_count = dr_vertex_count * n_features_per_level

        # tiny-cuda-nn pads each level's table to multiples of 8 vertices.
        tcnn_vertex_count = ((dr_vertex_count + 7) // 8) * 8
        tcnn_param_count = tcnn_vertex_count * n_features_per_level

        layouts.append(
            HashGridLevelLayout(
                level=level,
                dr_vertex_offset=dr_vertex_offset,
                dr_vertex_count=dr_vertex_count,
                dr_param_offset=dr_param_offset,
                dr_param_count=dr_param_count,
                tcnn_vertex_offset=tcnn_vertex_offset,
                tcnn_vertex_count=tcnn_vertex_count,
                tcnn_param_offset=tcnn_param_offset,
                tcnn_param_count=tcnn_param_count,
            )
        )

        tcnn_vertex_offset += tcnn_vertex_count
        tcnn_param_offset += tcnn_param_count

    return layouts


def tcnn_padded_params_to_dr_params(tcnn_params: torch.Tensor, dr_encoding: Any) -> torch.Tensor:
    flat = tcnn_params.detach().flatten()
    parts: list[torch.Tensor] = []
    for level in dr_hashgrid_level_layout(dr_encoding):
        src_start = level.tcnn_param_offset
        src_end = src_start + level.dr_param_count
        parts.append(flat[src_start:src_end])
    return torch.cat(parts, dim=0)


def load_tcnn_hashgrid_params_into_dr_encoding(tcnn_params: torch.Tensor, dr_encoding: Any) -> torch.Tensor:
    flat = tcnn_padded_params_to_dr_params(tcnn_params, dr_encoding)
    current = dr_encoding.params.torch()
    target = flat.to(device=current.device, dtype=current.dtype).contiguous()
    dr_encoding.set_params(type(dr_encoding.params)(target))
    return target


def summarize_hashgrid_layout(dr_encoding: Any) -> dict[str, Any]:
    levels = dr_hashgrid_level_layout(dr_encoding)
    return {
        "n_levels": len(levels),
        "dr_param_count": sum(level.dr_param_count for level in levels),
        "tcnn_param_count": sum(level.tcnn_param_count for level in levels),
        "padding_param_delta": sum(level.tcnn_param_count - level.dr_param_count for level in levels),
        "levels": levels,
    }


def smoothstep_semantics_note() -> str:
    return (
        "tiny-cuda-nn exposes interpolation='Smoothstep', while upstream Dr.Jit HashGridEncoding only exposes "
        "smooth_weight_gradients/smooth_weight_lambda, which changes gradients but not forward interpolation. "
        "This renderer therefore provides its own Smoothstep forward variant on top of Dr.Jit's HashGridEncoding "
        "while reusing Dr's indexing and parameter layout."
    )
