from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import torch

from adashell_renderer.drjit_backend.interop import DrJitInterop
from adashell_renderer.legacy_field import LegacyAdashellField
from adashell_renderer.types import DrRenderBuffer, IntervalPack, RayBatch, torch_device_for_array


@dataclass(slots=True)
class DrJitFieldIsland:
    """Dr-resident geo, predicted-gradient alpha, color, and RGB compositing."""

    field_model: LegacyAdashellField
    alpha_mode: str
    white_background: bool = True
    profile_geo: bool = False
    prewarm: bool = True
    _interop: DrJitInterop = field(init=False, repr=False)
    _prewarmed_devices: set[str] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._interop = DrJitInterop.create()
        self._prewarmed_devices = set()

    def _prewarm_device(self, device: torch.device, profile: dict[str, float]) -> None:
        if not self.prewarm:
            profile["field_island_prewarm_enabled"] = 0.0
            return
        profile["field_island_prewarm_enabled"] = 1.0
        key = str(device)
        if key in self._prewarmed_devices:
            profile["field_island_prewarm_cache_hit"] = 1.0
            return
        start = time.perf_counter()
        self.field_model.geo_mlp_dr._dr_layers(device)
        self.field_model.color_mlp_dr._dr_layers(device)
        profile["field_island_prewarm_ms"] = (time.perf_counter() - start) * 1000.0
        profile["field_island_prewarm_cache_hit"] = 0.0
        self._prewarmed_devices.add(key)

    def _require_supported(self, rays: RayBatch, packed: IntervalPack) -> None:
        if self.alpha_mode != "pred_gradient":
            raise ValueError(f"field island requires alpha_mode='pred_gradient', got {self.alpha_mode!r}.")
        if bool(rays.metadata.get("compute_gradients", False)):
            raise ValueError("field island does not support torch autograd gradients.")
        config = self.field_model.config
        if not (config.use_drjit_geo_mlp and config.use_drjit_color_mlp):
            raise ValueError("field island requires Dr.Jit geo and color MLPs.")
        if config.color_normal_source != "pred":
            raise ValueError("field island currently requires predicted-normal color input.")
        if self.field_model.use_grid_feature and not (
            self.field_model._dr_hash_encoding_enabled and self.field_model.hash_encoding_dr is not None
        ):
            raise ValueError("field island requires Dr hash encoding when grid features are enabled.")
        cache = packed.metadata.get("_drjit_cache", {})
        required = ("ray_indices", "eval_t", "deltas", "ray_offsets", "sample_counts")
        missing = [key for key in required if key not in cache]
        if missing:
            raise ValueError(f"field island missing packed Dr cache keys: {missing}.")

    def _exclusive_transmittance(self, alpha_dr: Any, packed_cache: dict[str, Any]) -> Any:
        dr = self._interop.dr
        survival_dr = 1.0 - alpha_dr + 1e-6
        log_survival_dr = dr.log(dr.maximum(survival_dr, 1e-6))
        sample_mask_dr = packed_cache.get("sample_mask")
        if sample_mask_dr is not None:
            log_survival_dr = dr.select(dr.cuda.Bool(sample_mask_dr), log_survival_dr, 0.0)
        # Subtracting long float32 prefixes loses the per-ray optical depth.
        global_log_prefix_dr = dr.prefix_reduce(dr.ReduceOp.Add, dr.cuda.Float64(log_survival_dr), exclusive=True)
        ray_indices_dr = packed_cache["ray_indices"]
        offsets_dr = packed_cache["ray_offsets"]
        ray_start_offsets_dr = dr.gather(dr.cuda.UInt, offsets_dr, ray_indices_dr)
        ray_start_log_prefix_dr = dr.gather(dr.cuda.Float64, global_log_prefix_dr, ray_start_offsets_dr)
        return dr.exp(dr.cuda.Float(global_log_prefix_dr - ray_start_log_prefix_dr))

    def _pred_gradient_alpha(
        self,
        *,
        sdf_dr: Any,
        inv_s_dr: Any,
        pred_gradient_cols: tuple[Any, Any, Any],
        ray_dir_cols: tuple[Any, Any, Any],
        deltas_dr: Any,
    ) -> Any:
        dr = self._interop.dr
        pred_x_dr, pred_y_dr, pred_z_dr = pred_gradient_cols
        norm_dr = dr.sqrt(dr.maximum(pred_x_dr * pred_x_dr + pred_y_dr * pred_y_dr + pred_z_dr * pred_z_dr, 1e-24))
        min_norm = max(float(self.field_model.config.pred_gradient_alpha_min_norm), 1e-12)
        max_norm = max(float(self.field_model.config.pred_gradient_alpha_max_norm), min_norm)
        clamped_norm_dr = dr.minimum(dr.maximum(norm_dr, min_norm), max_norm)
        scale_dr = clamped_norm_dr / norm_dr
        grad_x_dr = pred_x_dr * scale_dr
        grad_y_dr = pred_y_dr * scale_dr
        grad_z_dr = pred_z_dr * scale_dr
        ray_x_dr, ray_y_dr, ray_z_dr = ray_dir_cols
        directional_gradient_dr = ray_x_dr * grad_x_dr + ray_y_dr * grad_y_dr + ray_z_dr * grad_z_dr
        iter_cos_dr = -dr.maximum(-directional_gradient_dr, 0.0)
        estimated_next_sdf_dr = sdf_dr + iter_cos_dr * deltas_dr * 0.5
        estimated_prev_sdf_dr = sdf_dr - iter_cos_dr * deltas_dr * 0.5
        prev_cdf_dr = 1.0 / (1.0 + dr.exp(-(estimated_prev_sdf_dr * inv_s_dr)))
        next_cdf_dr = 1.0 / (1.0 + dr.exp(-(estimated_next_sdf_dr * inv_s_dr)))
        return dr.clip(((prev_cdf_dr - next_cdf_dr) + 1e-5) / (prev_cdf_dr + 1e-5), 0.0, 1.0)

    def _color_channels(
        self,
        *,
        query_point_cols: tuple[Any, Any, Any],
        ray_dir_cols: tuple[Any, Any, Any],
        feature_cols: tuple[Any, ...],
        pred_normal_cols: tuple[Any, Any, Any],
        width: int,
        device: torch.device,
    ) -> tuple[Any, Any, Any]:
        field_model = self.field_model
        dr = self._interop.dr
        shading_cols = field_model._normalize_dr3(pred_normal_cols)
        normals_cols = field_model._normalize_dr3(shading_cols)
        dir_cols = ray_dir_cols
        if field_model.config.use_reflections:
            dot = normals_cols[0] * (-dir_cols[0]) + normals_cols[1] * (-dir_cols[1]) + normals_cols[2] * (-dir_cols[2])
            dir_cols = (
                2.0 * dot * normals_cols[0] + dir_cols[0],
                2.0 * dot * normals_cols[1] + dir_cols[1],
                2.0 * dot * normals_cols[2] + dir_cols[2],
            )
        dir_features_cols = field_model._sh_encoding_dr(dir_cols, field_model.direction_encoding.levels)
        if field_model.use_average_appearance_embedding:
            mean_app = field_model.embedding_appearance.mean(dim=0).to(device=device, dtype=torch.float32)
            appearance_cols = tuple(dr.zeros(dr.cuda.Float, shape=width) + float(mean_app[idx].item()) for idx in range(mean_app.shape[0]))
        else:
            appearance_cols = tuple(dr.zeros(dr.cuda.Float, shape=width) for _ in range(field_model.config.appearance_embedding_dim))
        input_cols = tuple(query_point_cols) + tuple(dir_features_cols) + tuple(shading_cols) + tuple(feature_cols) + tuple(appearance_cols)
        if field_model.config.use_n_dot_v:
            n_dot_v = normals_cols[0] * dir_cols[0] + normals_cols[1] * dir_cols[1] + normals_cols[2] * dir_cols[2]
            input_cols = input_cols + (dr.cuda.Float(n_dot_v),)
        rgb_dr = field_model.color_mlp_dr.forward_dr(input_cols)
        if field_model.config.rgb_padding != 0.0:
            rgb_dr = field_model.color_mlp_dr.affine(
                rgb_dr,
                scale=(1.0 + 2.0 * float(field_model.config.rgb_padding)),
                bias=-float(field_model.config.rgb_padding),
            )
        return field_model.color_mlp_dr.tensor_columns(rgb_dr)

    def _new_render_profile(self) -> dict[str, float]:
        profile = {
            "field_island_geo_profile_enabled": float(1.0 if self.profile_geo else 0.0),
        }
        return profile

    def _render_rgb_core_dr(
        self,
        *,
        query_point_cols: tuple[Any, Any, Any] | None,
        ray_dir_cols: tuple[Any, Any, Any] | None,
        packed_cache: dict[str, Any],
        num_rays: int,
        valid_samples: int,
        image_width: int,
        image_height: int,
        device: torch.device,
        materialize_aux_outputs: bool,
        diagnostic_mode: str,
        profile: dict[str, float],
    ) -> DrRenderBuffer:
        """Evaluate packed samples and composite into per-ray Dr buffers."""

        dr = self._interop.dr
        sample_mask_dr = packed_cache.get("sample_mask")
        if sample_mask_dr is not None:
            sample_mask_dr = dr.cuda.Bool(sample_mask_dr)
        if valid_samples <= 0:
            bg = 1.0 if self.white_background else 0.0
            rgb_channels = tuple(dr.full(dr.cuda.Float, bg, shape=num_rays) for _ in range(3))
            zeros = dr.zeros(dr.cuda.Float, shape=num_rays)
            profile["field_island_empty"] = 1.0
            profile["rgb_materialize_ms"] = 0.0
            profile["aux_materialize_ms"] = 0.0
            profile["composite_to_torch_ms"] = 0.0
            return DrRenderBuffer(
                rgb_channels=rgb_channels,
                depth=zeros,
                accumulation=zeros,
                num_rays=num_rays,
                image_width=image_width,
                image_height=image_height,
                metadata={
                    "backend": "drjit_field_island",
                    "impl": "dr_render_buffer_empty",
                    "profile_stage_ms": profile,
                    "aux_outputs_materialized": float(1.0 if materialize_aux_outputs else 0.0),
                    "diagnostic_mode": diagnostic_mode,
                    "diagnostics": {},
                },
            )
        if query_point_cols is None or ray_dir_cols is None:
            raise ValueError("field island shared core requires query and ray-direction columns for non-empty samples.")

        geo_start = time.perf_counter()
        hash_ms = 0.0
        geo_mlp_ms = 0.0
        columns_ms = 0.0
        density_ms = 0.0
        if self.field_model.use_grid_feature:
            pos_start = time.perf_counter()
            pos_cols = tuple((col + 2.0) * 0.25 for col in query_point_cols)
            profile["field_island_pos_remap_ms"] = (time.perf_counter() - pos_start) * 1000.0
            hash_start = time.perf_counter()
            feature_cols = tuple(self.field_model.hash_encoding_dr(pos_cols))
            hash_ms = (time.perf_counter() - hash_start) * 1000.0
            inputs_full = tuple(query_point_cols) + feature_cols
        else:
            feature_cols = tuple()
            inputs_full = tuple(query_point_cols)
        mlp_profile = profile if self.profile_geo else None
        mlp_start = time.perf_counter()
        geo_out_dr = self.field_model.geo_mlp_dr.forward_dr(inputs_full, profile=mlp_profile, profile_prefix="geo_mlp")
        geo_mlp_ms = (time.perf_counter() - mlp_start) * 1000.0
        columns_start = time.perf_counter()
        raw_geo_cols = self.field_model.geo_mlp_dr.tensor_columns(geo_out_dr)
        columns_ms = (time.perf_counter() - columns_start) * 1000.0
        density_start = time.perf_counter()
        sdf_dr = raw_geo_cols[0]
        inv_s_dr = self.field_model._transform_inv_s_dr(raw_geo_cols[1])
        density_ms = (time.perf_counter() - density_start) * 1000.0
        geo_feature_cols = raw_geo_cols[2 : 2 + self.field_model.config.geo_feat_dim]
        pred_normal_cols = raw_geo_cols[2 + self.field_model.config.geo_feat_dim : 2 + self.field_model.config.geo_feat_dim + 3]
        geo_ms = (time.perf_counter() - geo_start) * 1000.0
        profile["field_island_geo_ms"] = geo_ms
        profile["field_island_hash_ms"] = hash_ms
        profile["field_island_geo_mlp_ms"] = geo_mlp_ms
        profile["field_island_geo_columns_ms"] = columns_ms
        profile["field_island_density_ms"] = density_ms
        profile["field_island_alpha_mode_pred_gradient"] = float(self.alpha_mode == "pred_gradient")
        profile["field_island_geo_other_ms"] = max(
            geo_ms
            - (
                float(profile.get("field_island_pos_remap_ms", 0.0))
                + hash_ms
                + geo_mlp_ms
                + columns_ms
                + density_ms
            ),
            0.0,
        )

        weight_start = time.perf_counter()
        pred_alpha_start = time.perf_counter()
        alpha_dr = self._pred_gradient_alpha(
            sdf_dr=sdf_dr,
            inv_s_dr=inv_s_dr,
            pred_gradient_cols=tuple(pred_normal_cols),
            ray_dir_cols=ray_dir_cols,
            deltas_dr=packed_cache["deltas"],
        )
        profile["field_island_pred_gradient_alpha_ms"] = (time.perf_counter() - pred_alpha_start) * 1000.0
        if sample_mask_dr is not None:
            alpha_dr = dr.select(sample_mask_dr, alpha_dr, 0.0)
            profile["field_island_sample_mask_enabled"] = 1.0
        else:
            profile["field_island_sample_mask_enabled"] = 0.0
        trans_dr = self._exclusive_transmittance(alpha_dr, packed_cache)
        weights_dr = alpha_dr * trans_dr
        profile["field_island_weight_ms"] = (time.perf_counter() - weight_start) * 1000.0

        color_start = time.perf_counter()
        rgb_sample_channels = self._color_channels(
            query_point_cols=query_point_cols,
            ray_dir_cols=ray_dir_cols,
            feature_cols=tuple(geo_feature_cols),
            pred_normal_cols=tuple(pred_normal_cols),
            width=dr.width(query_point_cols[0]),
            device=device,
        )
        profile["field_island_color_ms"] = (time.perf_counter() - color_start) * 1000.0

        composite_start = time.perf_counter()
        ray_indices_dr = packed_cache["ray_indices"]
        rgb_r_dr = dr.zeros(dr.cuda.Float, shape=num_rays)
        rgb_g_dr = dr.zeros(dr.cuda.Float, shape=num_rays)
        rgb_b_dr = dr.zeros(dr.cuda.Float, shape=num_rays)
        accum_dr = dr.zeros(dr.cuda.Float, shape=num_rays) if materialize_aux_outputs else None
        depth_dr = dr.zeros(dr.cuda.Float, shape=num_rays) if materialize_aux_outputs else None
        diagnostic_scalars_dr: dict[str, Any] = {}
        diagnostic_vectors_dr: dict[str, tuple[Any, Any, Any]] = {}
        if materialize_aux_outputs:
            dr.scatter_reduce(dr.ReduceOp.Add, accum_dr, weights_dr, ray_indices_dr)
            dr.scatter_reduce(dr.ReduceOp.Add, depth_dr, weights_dr * packed_cache["eval_t"], ray_indices_dr)
        if diagnostic_mode in {"weight_sum", "kernel_size", "pred_gradient_norm"}:
            diag_dr = dr.zeros(dr.cuda.Float, shape=num_rays)
            if diagnostic_mode == "weight_sum":
                dr.scatter_reduce(dr.ReduceOp.Add, diag_dr, weights_dr, ray_indices_dr)
            elif diagnostic_mode == "kernel_size":
                denom_dr = dr.zeros(dr.cuda.Float, shape=num_rays)
                kernel_size_dr = self.field_model._inv_s_to_kernel_size_dr(inv_s_dr)
                dr.scatter_reduce(dr.ReduceOp.Add, diag_dr, weights_dr * kernel_size_dr, ray_indices_dr)
                dr.scatter_reduce(dr.ReduceOp.Add, denom_dr, weights_dr, ray_indices_dr)
                diag_dr = diag_dr / dr.maximum(denom_dr, 1e-6)
            else:
                denom_dr = dr.zeros(dr.cuda.Float, shape=num_rays)
                pred_x_dr, pred_y_dr, pred_z_dr = pred_normal_cols
                norm_dr = dr.sqrt(dr.maximum(pred_x_dr * pred_x_dr + pred_y_dr * pred_y_dr + pred_z_dr * pred_z_dr, 1e-24))
                dr.scatter_reduce(dr.ReduceOp.Add, diag_dr, weights_dr * norm_dr, ray_indices_dr)
                dr.scatter_reduce(dr.ReduceOp.Add, denom_dr, weights_dr, ray_indices_dr)
                diag_dr = diag_dr / dr.maximum(denom_dr, 1e-6)
            diagnostic_scalars_dr[diagnostic_mode] = diag_dr
        elif diagnostic_mode in {"pred_normal", "pred_gradient"}:
            vec_x_dr = dr.zeros(dr.cuda.Float, shape=num_rays)
            vec_y_dr = dr.zeros(dr.cuda.Float, shape=num_rays)
            vec_z_dr = dr.zeros(dr.cuda.Float, shape=num_rays)
            if diagnostic_mode == "pred_normal":
                pred_x_dr, pred_y_dr, pred_z_dr = self.field_model._normalize_dr3(tuple(pred_normal_cols))
            else:
                pred_x_dr, pred_y_dr, pred_z_dr = pred_normal_cols
            dr.scatter_reduce(dr.ReduceOp.Add, vec_x_dr, weights_dr * pred_x_dr, ray_indices_dr)
            dr.scatter_reduce(dr.ReduceOp.Add, vec_y_dr, weights_dr * pred_y_dr, ray_indices_dr)
            dr.scatter_reduce(dr.ReduceOp.Add, vec_z_dr, weights_dr * pred_z_dr, ray_indices_dr)
            diagnostic_vectors_dr[diagnostic_mode] = (vec_x_dr, vec_y_dr, vec_z_dr)
        if self.white_background:
            dr.scatter_reduce(dr.ReduceOp.Add, rgb_r_dr, weights_dr * (rgb_sample_channels[0] - 1.0), ray_indices_dr)
            dr.scatter_reduce(dr.ReduceOp.Add, rgb_g_dr, weights_dr * (rgb_sample_channels[1] - 1.0), ray_indices_dr)
            dr.scatter_reduce(dr.ReduceOp.Add, rgb_b_dr, weights_dr * (rgb_sample_channels[2] - 1.0), ray_indices_dr)
            rgb_r_dr = rgb_r_dr + 1.0
            rgb_g_dr = rgb_g_dr + 1.0
            rgb_b_dr = rgb_b_dr + 1.0
        else:
            dr.scatter_reduce(dr.ReduceOp.Add, rgb_r_dr, weights_dr * rgb_sample_channels[0], ray_indices_dr)
            dr.scatter_reduce(dr.ReduceOp.Add, rgb_g_dr, weights_dr * rgb_sample_channels[1], ray_indices_dr)
            dr.scatter_reduce(dr.ReduceOp.Add, rgb_b_dr, weights_dr * rgb_sample_channels[2], ray_indices_dr)
        profile["field_island_rgb_scatter_count"] = 3.0
        profile["field_island_aux_scatter_count"] = 2.0 if materialize_aux_outputs else 0.0
        if diagnostic_mode in {"kernel_size", "pred_gradient_norm"}:
            diagnostic_scatter_count = 2.0
        elif diagnostic_mode == "weight_sum":
            diagnostic_scatter_count = 1.0
        elif diagnostic_mode in {"pred_normal", "pred_gradient"}:
            diagnostic_scatter_count = 3.0
        else:
            diagnostic_scatter_count = 0.0
        profile["field_island_diagnostic_scatter_count"] = diagnostic_scatter_count
        profile["field_island_diagnostic_mode_" + diagnostic_mode] = 1.0
        profile["field_island_accum_scatter_skipped"] = 0.0 if materialize_aux_outputs else 1.0
        profile["field_island_composite_dr_ms"] = (time.perf_counter() - composite_start) * 1000.0
        profile["rgb_materialize_ms"] = 0.0
        profile["aux_materialize_ms"] = 0.0
        profile["composite_to_torch_ms"] = 0.0

        diagnostics_dr: dict[str, Any] = {}
        diagnostics_dr.update(diagnostic_scalars_dr)
        diagnostics_dr.update(diagnostic_vectors_dr)
        return DrRenderBuffer(
            rgb_channels=(rgb_r_dr, rgb_g_dr, rgb_b_dr),
            depth=depth_dr,
            accumulation=accum_dr,
            diagnostics=diagnostics_dr,
            num_rays=num_rays,
            image_width=image_width,
            image_height=image_height,
            metadata={
                "backend": "drjit_field_island",
                "impl": "dr_render_buffer",
                "profile_stage_ms": profile,
                "aux_outputs_materialized": float(1.0 if materialize_aux_outputs else 0.0),
                "diagnostic_mode": diagnostic_mode,
                "diagnostics": diagnostics_dr,
            },
        )

    def render_rgb_buffer(
        self,
        rays: RayBatch,
        packed: IntervalPack,
        *,
        materialize_aux_outputs: bool = False,
        diagnostic_mode: str = "none",
    ) -> DrRenderBuffer:
        self._require_supported(rays, packed)
        diagnostic_mode = str(diagnostic_mode or "none")
        supported_diagnostics = {"none", "sample_count", "pred_normal", "pred_gradient", "pred_gradient_norm", "kernel_size", "weight_sum"}
        if diagnostic_mode not in supported_diagnostics:
            raise ValueError(f"Unsupported field island diagnostic_mode={diagnostic_mode!r}.")

        device = torch_device_for_array(packed.eval_t)
        profile = self._new_render_profile()
        packed_cache = packed.metadata["_drjit_cache"]
        self._prewarm_device(device, profile)

        query_point_cols: tuple[Any, Any, Any] | None = None
        ray_dir_cols: tuple[Any, Any, Any] | None = None
        if packed.valid_samples > 0:
            query_start = time.perf_counter()
            dr_query_cache = self.field_model._packed_query_dr_cache(rays, packed)
            if dr_query_cache is None:
                raise ValueError("field island could not build Dr query-point cache.")
            query_point_cols, ray_dir_cols = dr_query_cache
            query_ms = (time.perf_counter() - query_start) * 1000.0
            profile["field_island_query_ms"] = query_ms
            profile["field_island_query_gather_ms"] = query_ms

        render_buffer = self._render_rgb_core_dr(
            query_point_cols=query_point_cols,
            ray_dir_cols=ray_dir_cols,
            packed_cache=packed_cache,
            num_rays=packed.num_rays,
            valid_samples=packed.valid_samples,
            image_width=int(rays.metadata.get("image_width", 0)),
            image_height=int(rays.metadata.get("image_height", 0)),
            device=device,
            materialize_aux_outputs=materialize_aux_outputs,
            diagnostic_mode=diagnostic_mode,
            profile=profile,
        )
        render_buffer.diagnostics["sample_count"] = self._interop.dr.cuda.Float(packed_cache["sample_counts"])
        render_buffer.metadata["diagnostics"] = render_buffer.diagnostics
        return render_buffer
