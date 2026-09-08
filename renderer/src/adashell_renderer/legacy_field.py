# Copyright 2022 The Nerfstudio Team. All rights reserved.
# Modifications Copyright (c) 2026 Zhaorong Wang.
# SPDX-License-Identifier: Apache-2.0
# Adapted for standalone checkpoint loading and Dr.Jit field evaluation.

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from adashell_renderer.config import RendererFieldConfig
from adashell_renderer.drjit_backend.hashgrid_import import (
    build_dr_hashgrid_encoding,
    load_tcnn_hashgrid_params_into_dr_encoding,
)
from adashell_renderer.drjit_backend.mlp import DrJitMLP
from adashell_renderer.types import IntervalPack, RayBatch


class LaplaceDensityModule(nn.Module):
    def __init__(self, init_val: float = 0.1, beta_min: float = 0.0001) -> None:
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))


def _require_tcnn() -> Any:
    try:
        import tinycudann as tcnn  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Loading the standalone legacy field requires `tinycudann` in the new renderer environment."
        ) from exc
    return tcnn


class AppearanceEmbedding(nn.Module):
    def __init__(self, num_images: int, embedding_dim: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_images, embedding_dim)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices)

    def mean(self, dim: int = 0) -> torch.Tensor:
        return self.embedding.weight.mean(dim=dim)

    def get_out_dim(self) -> int:
        return int(self.embedding.embedding_dim)


class ProjectSHEncoding(nn.Module):
    def __init__(self, levels: int = 4) -> None:
        super().__init__()
        self.levels = int(levels)

    def get_out_dim(self) -> int:
        return self.levels**2

class LegacyAdashellField(nn.Module):
    def __init__(self, config: RendererFieldConfig, num_images: int, aabb: torch.Tensor | None = None) -> None:
        super().__init__()
        self.config = config
        self.aabb = nn.Parameter(
            aabb if aabb is not None else torch.tensor([[-2.0, -2.0, -2.0], [2.0, 2.0, 2.0]], dtype=torch.float32),
            requires_grad=False,
        )
        self.embedding_appearance = AppearanceEmbedding(num_images, self.config.appearance_embedding_dim)
        self.use_average_appearance_embedding = self.config.use_average_appearance_embedding
        self.use_grid_feature = self.config.use_grid_feature

        tcnn = _require_tcnn()
        growth_factor = np.exp(
            (np.log(self.config.max_res) - np.log(self.config.base_res)) / (self.config.num_levels - 1)
        )
        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.config.num_levels,
                "n_features_per_level": self.config.hash_features_per_level,
                "log2_hashmap_size": self.config.log2_hashmap_size,
                "base_resolution": self.config.base_res,
                "per_level_scale": growth_factor,
                "interpolation": "Smoothstep" if self.config.hash_smoothstep else "Linear",
            },
        )
        self.register_buffer(
            "hash_encoding_mask",
            torch.ones(self.config.num_levels * self.config.hash_features_per_level, dtype=torch.float32),
            persistent=False,
        )
        self.direction_encoding = ProjectSHEncoding(levels=4)
        self.laplace_density = LaplaceDensityModule()

        geo_dims = [3 + self.encoding.n_output_dims]
        geo_dims.extend([self.config.hidden_dim] * self.config.num_layers)
        geo_dims.append(self.config.geo_feat_dim + 1 + 1 + 3)
        self.num_layers = len(geo_dims)
        self.skip_in = [4]
        for idx in range(self.num_layers - 1):
            out_dim = geo_dims[idx + 1]
            if idx + 1 in self.skip_in:
                out_dim -= geo_dims[0]
            layer = nn.Linear(geo_dims[idx], out_dim)
            layer = nn.utils.weight_norm(layer)
            setattr(self, f"glin{idx}", layer)

        color_in_dim = (
            3
            + self.direction_encoding.get_out_dim()
            + 3
            + self.config.geo_feat_dim
            + self.embedding_appearance.get_out_dim()
        )
        if self.config.use_n_dot_v:
            color_in_dim += 1
        color_dims = [color_in_dim]
        color_dims.extend([self.config.hidden_dim_color] * self.config.num_layers_color)
        color_dims.append(3)
        self.num_layers_color = len(color_dims)
        for idx in range(self.num_layers_color - 1):
            layer = nn.Linear(color_dims[idx], color_dims[idx + 1])
            layer = nn.utils.weight_norm(layer)
            setattr(self, f"clin{idx}", layer)

        self.softplus = nn.Softplus(beta=100)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self._refresh_drjit_mlps()
        self._refresh_drjit_hash_encoding()

    def _refresh_drjit_mlps(self) -> None:
        # Rebuild Dr.Jit-side MLP wrappers from the *current* torch layer
        # parameters. This is crucial after checkpoint loading: otherwise the
        # Dr.Jit path would keep using the random initialization captured during
        # `__init__`, while the torch path uses checkpoint weights.
        self.geo_mlp_dr = DrJitMLP.from_torch_linears(
            [getattr(self, f"glin{idx}") for idx in range(self.num_layers - 1)],
            hidden_activation="softplus",
            output_activation="none",
            unsafe_direct_torch_bridge=bool(self.config.drjit_direct_torch_bridge),
            mode=self.config.drjit_mlp_mode,
        )
        self.color_mlp_dr = DrJitMLP.from_torch_linears(
            [getattr(self, f"clin{idx}") for idx in range(self.num_layers_color - 1)],
            hidden_activation="relu",
            output_activation="sigmoid",
            unsafe_direct_torch_bridge=bool(self.config.drjit_direct_torch_bridge),
            mode=self.config.drjit_mlp_mode,
        )

    def _refresh_drjit_hash_encoding(self) -> None:
        self.hash_encoding_dr = None
        self._dr_hash_encoding_enabled = False

        if not self.use_grid_feature or not bool(self.config.use_drjit_hash_encoding):
            return

        self.hash_encoding_dr = build_dr_hashgrid_encoding(
            self.config,
            torchngp_compat=bool(self.config.drjit_hashgrid_torchngp_compat),
            smoothstep=bool(self.config.hash_smoothstep),
        )
        load_tcnn_hashgrid_params_into_dr_encoding(self.encoding.params, self.hash_encoding_dr)
        self._dr_hash_encoding_enabled = True

    def load_state_dict(self, state_dict, strict: bool = True):
        result = super().load_state_dict(state_dict, strict=strict)
        self._refresh_drjit_mlps()
        self._refresh_drjit_hash_encoding()
        return result

    def _transform_inv_s_dr(self, inv_s_raw: Any) -> Any:
        dr = self.geo_mlp_dr.interop.dr
        if self.config.kernel_size_transform_mode == "exp":
            return dr.minimum(
                dr.maximum(dr.exp(inv_s_raw * float(self.config.kernel_size_transform_exp_factor)), 1e-6),
                1e6,
            )
        if self.config.kernel_size_transform_mode == "clip":
            return dr.minimum(dr.maximum(inv_s_raw, 1e-6), 1e6)
        if self.config.kernel_size_transform_mode == "none":
            return inv_s_raw
        raise ValueError(f"Unsupported inv_s transform mode: {self.config.kernel_size_transform_mode}")

    def _inv_s_to_kernel_size_dr(self, inv_s: Any) -> Any:
        return 1.0 / (inv_s + 1e-6)

    def _dr_ray_columns(self, array: Any) -> tuple[Any, ...]:
        import cupy as cp  # type: ignore[import-not-found]

        if not isinstance(array, cp.ndarray) or array.dtype != cp.float32 or array.ndim != 2 or array.shape[1] != 3:
            raise TypeError("Field ray inputs must be CuPy float32 arrays with shape (R, 3).")
        dr = self.geo_mlp_dr.interop.dr
        return tuple(dr.cuda.Float(cp.ascontiguousarray(array[:, idx])) for idx in range(3))

    def _rays_dr_cache(self, rays: RayBatch) -> dict[str, Any]:
        cache = rays.metadata.setdefault("_drjit_cache", {})
        if "origins_xyz" not in cache:
            cache["origins_xyz"] = self._dr_ray_columns(rays.origins)
        if "directions_xyz" not in cache:
            cache["directions_xyz"] = self._dr_ray_columns(rays.directions)
        return cache

    def _packed_query_dr_cache(self, rays: RayBatch, packed: IntervalPack) -> tuple[tuple[Any, Any, Any], tuple[Any, Any, Any]] | None:
        packed_cache = packed.metadata.get("_drjit_cache", {})
        if "ray_indices" not in packed_cache or "eval_t" not in packed_cache:
            return None
        dr = self.geo_mlp_dr.interop.dr
        rays_cache = self._rays_dr_cache(rays)
        ray_indices_dr = packed_cache["ray_indices"]
        eval_t_dr = packed_cache["eval_t"]
        origin_cols = rays_cache["origins_xyz"]
        direction_cols = rays_cache["directions_xyz"]
        ray_dir_xyz = tuple(dr.gather(dr.cuda.Float, col, ray_indices_dr) for col in direction_cols)
        query_point_xyz = tuple(
            dr.gather(dr.cuda.Float, origin_cols[idx], ray_indices_dr) + ray_dir_xyz[idx] * eval_t_dr
            for idx in range(3)
        )
        return query_point_xyz, ray_dir_xyz

    def _normalize_dr3(self, cols: tuple[Any, Any, Any], eps: float = 1e-6) -> tuple[Any, Any, Any]:
        dr = self.geo_mlp_dr.interop.dr
        x, y, z = cols
        inv_norm = 1.0 / dr.sqrt(dr.maximum(x * x + y * y + z * z, eps))
        return x * inv_norm, y * inv_norm, z * inv_norm

    def _sh_encoding_dr(self, directions: tuple[Any, Any, Any], levels: int = 4) -> tuple[Any, ...]:
        if not 1 <= levels <= 4:
            raise ValueError(f"SH levels must be in [1, 4], got {levels}")
        x, y, z = directions
        xx = x * x
        yy = y * y
        zz = z * z
        dr = self.geo_mlp_dr.interop.dr
        comps: list[Any] = [x * 0 + 0.28209479177387814]
        if levels > 1:
            comps.extend(
                (
                    0.4886025119029199 * y,
                    0.4886025119029199 * z,
                    0.4886025119029199 * x,
                )
            )
        if levels > 2:
            comps.extend(
                (
                    1.0925484305920792 * x * y,
                    1.0925484305920792 * y * z,
                    0.9461746957575601 * zz - 0.31539156525252,
                    1.0925484305920792 * x * z,
                    0.5462742152960396 * (xx - yy),
                )
            )
        if levels > 3:
            comps.extend(
                (
                    0.5900435899266435 * y * (3 * xx - yy),
                    2.890611442640554 * x * y * z,
                    0.4570457994644658 * y * (5 * zz - 1),
                    0.3731763325901154 * z * (5 * zz - 3),
                    0.4570457994644658 * x * (5 * zz - 1),
                    1.445305721320277 * z * (xx - yy),
                    0.5900435899266435 * x * (xx - 3 * yy),
                )
            )
        return tuple(dr.cuda.Float(comp) for comp in comps)

@dataclass(slots=True)
class LegacyCheckpoint:
    field_state: dict[str, torch.Tensor]
    num_images: int
    aabb: torch.Tensor | None


def load_legacy_checkpoint(path: Path) -> LegacyCheckpoint:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("pipeline", checkpoint.get("state_dict", {}))
    prefix = "_model.field."
    field_state = {
        key[len(prefix) :]: value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }
    if not field_state:
        raise ValueError(f"No field weights found in checkpoint: {path}")
    embedding = field_state.get("embedding_appearance.embedding.weight")
    if embedding is None:
        raise ValueError("Checkpoint is missing `embedding_appearance.embedding.weight`.")
    aabb = field_state.get("aabb")
    return LegacyCheckpoint(field_state=field_state, num_images=int(embedding.shape[0]), aabb=aabb)


def build_legacy_field_from_checkpoint(path: Path, config: RendererFieldConfig, device: str) -> LegacyAdashellField:
    payload = load_legacy_checkpoint(path)
    field = LegacyAdashellField(config=config, num_images=payload.num_images, aabb=payload.aabb)
    missing, unexpected = field.load_state_dict(payload.field_state, strict=False)
    missing = [key for key in missing if not key.startswith("hash_encoding_mask")]
    if missing or unexpected:
        raise RuntimeError(
            f"Legacy field state mismatch. Missing={missing}, unexpected={unexpected}"
        )
    field.eval()
    return field.to(device)
