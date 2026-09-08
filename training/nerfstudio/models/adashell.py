"""
Nerfstudio Template Config

Define your custom method here that registers with Nerfstudio CLI.
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Type, List, Dict, Tuple, Optional, cast, Literal
import math
import numpy as np
import time


import nerfacc
import torch
import trimesh
from PIL import Image

from nerfstudio.models.neus import NeuSModel, NeuSModelConfig

from nerfstudio.cameras.rays import RayBundle
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.models.neus import NeuSModel, NeuSModelConfig
from nerfstudio.model_components.losses import pred_normal_l2_loss, pred_normal_loss
from nerfstudio.fields.adashell_field import (
    AdashellField,
    AdashellFieldConfig,
    AdashellTCNNField,
    AdashellTorchCompatField,
)
from nerfstudio.model_components.ray_samplers import ShellSampler, ShellSamplerConfig
from nerfstudio.utils import writer
from nerfstudio.cameras.cameras import Cameras

@dataclass
class AdashellModelConfig(NeuSModelConfig):
    """Adashell Model Configuration. Based on NeuSFacto.
    """
    _target: Type = field(default_factory=lambda: AdashellModel)
    sdf_field: AdashellFieldConfig = AdashellFieldConfig()    
    field_backend: Literal["torch", "tcnn", "torch_tcnn_compat"] = "torch"
    """Field implementation backend for Adashell."""
    # num_proposal_samples_per_ray: Tuple[int, ...] = (256, 96)
    # """Number of samples per ray for the proposal network."""
    # num_neus_samples_per_ray: int = 48
    # """Number of samples per ray for the nerf network."""
    # proposal_update_every: int = 5
    # """Sample every n steps after the warmup"""
    # proposal_warmup: int = 5000
    # """Scales n from 1 to proposal_update_every over this many steps"""
    # num_proposal_iterations: int = 2
    # """Number of proposal network iterations."""
    # use_same_proposal_network: bool = False
    # """Use the same proposal network. Otherwise use different ones."""
    # proposal_net_args_list: List[Dict] = field(
    #     default_factory=lambda: [
    #         {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 64},
    #         {"hidden_dim": 16, "log2_hashmap_size": 17, "num_levels": 5, "max_res": 256},
    #     ]
    # )
    # """Arguments for the proposal density fields."""
    # interlevel_loss_mult: float = 1.0
    # """Proposal loss multiplier."""
    # use_proposal_weight_anneal: bool = True
    # """Whether to use proposal weight annealing."""
    # proposal_weights_anneal_slope: float = 10.0
    # """Slope of the annealing function for the proposal weights."""
    # proposal_weights_anneal_max_num_iters: int = 1000
    # """Max num iterations for the annealing function."""
    # use_single_jitter: bool = True
    # """Whether use single jitter or not for the proposal networks."""
    num_samples: int = 64
    """Number of uniform samples"""
    num_samples_importance: int = 64
    """Number of importance samples"""
    use_anneal_beta: bool = False
    """whether to anneal beta of neus or not similar to bakedsdf"""
    beta_anneal_max_num_iters: int = 1000_000
    """max num iterations for the annealing function of beta"""
    beta_anneal_init: float = 0.05
    """initial beta for annealing function"""
    beta_anneal_end: float = 0.0002
    """final beta for annealing function"""
    # TODO move to base model config since it can be used in all models
    enable_progressive_hash_encoding: bool = False
    """whether to use progressive hash encoding"""
    enable_numerical_gradients_schedule: bool = False
    """whether to use numerical gradients delta schedule"""
    enable_curvature_loss_schedule: bool = False
    """whether to use curvature loss weight schedule"""
    curvature_loss_multi: float = 0.0
    """curvature loss weight"""
    curvature_loss_warmup_steps: int = 20_000
    """curvature loss warmup steps"""
    level_init: int = 8
    """initial level of multi-resolution hash encoding"""
    steps_per_level: int = 5_000
    """steps per level of multi-resolution hash encoding"""
    numerical_gradients_schedule_mode: Literal["fixed", "neuralangelo"] = "neuralangelo"
    """Finite-difference delta mode: fixed fallback or Neuralangelo-style anchored schedule."""
    numerical_gradients_fixed_delta: float = 1e-4
    """Finite-difference delta used in fixed mode."""
    numerical_gradients_schedule_base_res: float = 64.0
    """Base resolution anchor used by the Neuralangelo-style epsilon schedule."""
    numerical_gradients_schedule_max_res: Optional[float] = None
    """Maximum resolution cap used by the Neuralangelo-style epsilon schedule; defaults to the field max_res."""
    numerical_gradients_schedule_start_res: float = 64.0
    """Deprecated legacy field kept for config compatibility; no longer used by AdaShell delta scheduling."""
    disable_fd_losses_before_schedule: bool = True
    """Deprecated legacy field kept for config compatibility; no longer used by AdaShell delta scheduling."""
    numerical_gradients_delta_init: float = 1e-4
    """Deprecated legacy field kept for config compatibility; no longer used by AdaShell delta scheduling."""
    numerical_gradients_delta_final: float = 2.5e-5
    """Deprecated legacy field kept for config compatibility; no longer used by AdaShell delta scheduling."""
    numerical_gradients_delta_scale: float = 4.0
    """Deprecated legacy field kept for config compatibility; no longer used by AdaShell delta scheduling."""
    numerical_gradients_delta_clamp_factor: float = 4.0
    """Deprecated legacy field kept for config compatibility; no longer used by AdaShell delta scheduling."""
    pred_normal_loss_mult: float = 0.1
    """pred normal loss multiplier"""
    pred_normal_loss_mode: Literal["l2_paper", "weighted_cosine_legacy"] = "l2_paper"
    """Pred-normal residual mode; supports legacy weighted cosine formulation."""
    kernel_size_smoothness_loss_mult: float = 0.01
    """kernel size smoothness loss multiplier"""
    kernel_size_smoothness_loss_mode: Literal["log_l1_current", "log_l2_legacy", "kernel_l2_legacy"] = "log_l1_current"
    """Kernel-size smoothness residual mode."""
    use_learned_inv_s_for_upsampling: bool = False
    """Use field-predicted inv_s (instead of fixed NeuS schedule) in stage1 upsampling."""
    shell_inference_mode: bool = False
    """whether to use shell inference mode"""
    shell_finetune_mode: bool = False
    """Enable shell narrow-band sampling during training/eval for stage2 fine-tuning."""
    stage2_color_only_loss: bool = False
    """Use only rgb reconstruction loss during training, matching paper stage2."""
    stage2_enable_pred_normal_loss: bool = False
    """Allow pred-normal supervision during stage2 even when color-only mode is enabled."""
    stage2_enable_kernel_smoothness_loss: bool = False
    """Allow kernel-size smoothness regularization during stage2 color-only training."""
    shell_use_packed_samples: bool = False
    """Use packed valid shell samples during inference to skip padded empty samples."""
    enable_cuda_pair_kernel: Optional[bool] = None
    """Enable the OptiX interval-pair CUDA kernel. None falls back to env var."""
    enable_cuda_pack_kernel: Optional[bool] = None
    """Enable the OptiX interval-pack CUDA kernel. None falls back to env var."""
    # shell inference mode config
    dilation_mesh_path: str = "meshes/shell_debug-dilation.ply"
    """dilation mesh path"""
    erosion_mesh_path: str = "meshes/shell_debug-erosion.ply"
    """erosion mesh path"""
    n_sample_max: int = 16
    """max number of samples"""
    n_depth_peeling_max: int = 20
    """maximum ordered hit budget for shell tracing; effective interval budget is floor(max_hits / 2)"""
    shell_ray_tracing_backend: str = "cpu"
    """Tracing backend used by `ShellSampler`: 'cpu' or 'optix'."""
    shell_use_inner_mesh: bool = True
    """Whether shell tracing clips intervals by the inner shell. Set False for outer-only diagnostics."""
    single_sample_threshold: float = 0.02
    """single sample threshold"""
    sample_spacing: float = 0.01
    """sample spacing"""
    optix_debug_render: bool = True
    """Whether to render a full-image OptiX depth map during eval-image debug."""
    optix_debug_validate_camera_conversion: bool = False
    """Validate ns->optix camera conversion once per camera when rendering debug depth."""
    eval_visualize_kernel_size: bool = True
    """Whether to publish kernel-size renderings during eval image logging."""
    eval_visualize_kernel_smoothness: bool = True
    """Whether to publish image-space kernel smoothness proxy during eval image logging."""
    eval_visualize_surface_proxies: bool = True
    """Whether to publish surface readiness proxy maps (min|sdf| and surface kernel size)."""
    eval_visualize_shell_maps: bool = True
    """Whether to publish shell hit/count maps during eval image logging."""
    eval_export_shell_maps_to_disk: bool = False
    """Whether to save shell hit/count debug maps as npy/png files on disk during eval."""

class AdashellModel(NeuSModel):
    """Adashell Model."""

    config: AdashellModelConfig

    def populate_modules(self):
        """Set the fields and modules."""
        if self.config.field_backend == "tcnn":
            if self.config.sdf_field.encoding_type != "hash":
                raise ValueError(
                    "field_backend='tcnn' only supports sdf_field.encoding_type='hash' in v1."
                )
            self.config.sdf_field._target = AdashellTCNNField
        elif self.config.field_backend == "torch_tcnn_compat":
            if self.config.sdf_field.encoding_type != "hash":
                raise ValueError(
                    "field_backend='torch_tcnn_compat' only supports sdf_field.encoding_type='hash'."
                )
            self.config.sdf_field._target = AdashellTorchCompatField
        else:
            self.config.sdf_field._target = AdashellField
        super().populate_modules()
        self._shell_profile_options = {"strict_timing": False}
        self.reset_shell_profile()
        self._active_hash_levels = int(getattr(self.field, "num_levels", 1))
        self._active_hash_max_res = float(self._active_hash_max_resolution(self._active_hash_levels))
        self._fd_schedule_active = False
        if hasattr(self.sampler, "use_learned_inv_s_for_upsampling"):
            setattr(self.sampler, "use_learned_inv_s_for_upsampling", self.config.use_learned_inv_s_for_upsampling)
        self._optix_eval_cameras: Optional[Cameras] = None
        self._optix_tracer = None
        use_shell_sampler = self.config.shell_inference_mode or self.config.shell_finetune_mode
        if use_shell_sampler:
            dilation_mesh, erosion_mesh = self.load_shell_mesh(self.config.dilation_mesh_path, self.config.erosion_mesh_path)
            if dilation_mesh is None or erosion_mesh is None:
                raise ValueError("Shell sampling requires valid dilation and erosion meshes.")
            tracing_backend = self.config.shell_ray_tracing_backend.lower()
            shared_optix_tracer = None
            need_optix_tracer = tracing_backend == "optix" or (
                self.config.optix_debug_render and self.config.shell_inference_mode
            )
            if need_optix_tracer:
                from nerfstudio.optix.raytracer import ShellRayTracer

                shared_optix_tracer = ShellRayTracer(
                    dilation_mesh=cast(trimesh.Trimesh, dilation_mesh),
                    erosion_mesh=cast(trimesh.Trimesh, erosion_mesh),
                    n_sample_max=self.config.n_sample_max,
                    n_depth_peeling_max=self.config.n_depth_peeling_max,
                    use_inner_mesh=self.config.shell_use_inner_mesh,
                    single_sample_threshold=self.config.single_sample_threshold,
                    sample_spacing=self.config.sample_spacing,
                    enable_cuda_pair_kernel=self.config.enable_cuda_pair_kernel,
                    enable_cuda_pack_kernel=self.config.enable_cuda_pack_kernel,
                )
            self.sampler = ShellSampler(
                dilation_mesh=cast(trimesh.Trimesh, dilation_mesh),
                erosion_mesh=cast(trimesh.Trimesh, erosion_mesh),
                config=ShellSamplerConfig(
                    n_sample_max=self.config.n_sample_max,
                    n_depth_peeling_max=self.config.n_depth_peeling_max,
                    tracing_backend=tracing_backend,
                    use_inner_mesh=self.config.shell_use_inner_mesh,
                    single_sample_threshold=self.config.single_sample_threshold,
                    sample_spacing=self.config.sample_spacing,
                    enable_cuda_pair_kernel=self.config.enable_cuda_pair_kernel,
                    enable_cuda_pack_kernel=self.config.enable_cuda_pack_kernel,
                    debug_print_shell_mask=not self.config.shell_finetune_mode,
                    debug_export_points_ply=not self.config.shell_finetune_mode,
                    debug_export_meshes=not self.config.shell_finetune_mode,
                ),
                tracer_backend=shared_optix_tracer if tracing_backend == "optix" else None,
            )
            if self.config.optix_debug_render:
                self._optix_tracer = shared_optix_tracer
        self.set_shell_profile_options(strict_timing=False)

    def _resolve_active_hash_level(self, step: int) -> int:
        num_levels = int(getattr(self.field, "num_levels", 1))
        if self.config.steps_per_level <= 0:
            return max(1, min(num_levels, self.config.level_init))
        progressed_levels = int(step // self.config.steps_per_level)
        return max(1, min(num_levels, self.config.level_init + progressed_levels))

    def _active_hash_max_resolution(self, active_level: int) -> float:
        base_res = float(getattr(self.field, "base_res", 1.0))
        growth_factor = float(getattr(self.field, "growth_factor", 1.0))
        return float(base_res * (growth_factor ** max(active_level - 1, 0)))

    def _update_progressive_training_state(self, step: int) -> None:
        num_levels = int(getattr(self.field, "num_levels", 1))
        active_level = num_levels
        if self.config.enable_progressive_hash_encoding:
            active_level = self._resolve_active_hash_level(step)
            self.field.update_mask(active_level)

        self._active_hash_levels = active_level
        self._active_hash_max_res = self._active_hash_max_resolution(active_level)

        if not getattr(self.config.sdf_field, "use_numerical_gradients", False):
            self._fd_schedule_active = False
            return

        fixed_delta = max(float(self.config.numerical_gradients_fixed_delta), 1e-8)
        delta_value = fixed_delta
        schedule_active = False

        if self.config.enable_numerical_gradients_schedule:
            mode = self.config.numerical_gradients_schedule_mode
            if mode == "fixed":
                delta_value = fixed_delta
            elif mode == "neuralangelo":
                growth_factor = float(getattr(self.field, "growth_factor", 1.0))
                schedule_base_res = max(float(self.config.numerical_gradients_schedule_base_res), 1e-8)
                schedule_max_res_cfg = self.config.numerical_gradients_schedule_max_res
                schedule_max_res = (
                    float(schedule_max_res_cfg)
                    if schedule_max_res_cfg is not None
                    else float(getattr(self.field, "max_res", 1.0))
                )
                schedule_max_res = max(schedule_max_res, schedule_base_res)
                schedule_steps_per_level = max(int(self.config.steps_per_level), 1)
                delta_raw = 1.0 / (schedule_base_res * (growth_factor ** (step / schedule_steps_per_level)))
                delta_raw = max(1.0 / schedule_max_res, delta_raw)
                delta_value = delta_raw * 2.0
                schedule_active = True
            else:
                raise ValueError(f"Unknown numerical_gradients_schedule_mode: {mode}")

        self.field.set_numerical_gradients_delta(float(delta_value))
        self._fd_schedule_active = schedule_active

    def _finite_difference_losses_enabled(self) -> bool:
        if not self.training:
            return True
        if not self.config.enable_numerical_gradients_schedule:
            return True
        return True

    def _pred_normal_residual(
        self,
        normals: torch.Tensor,
        pred_normals: torch.Tensor,
        sample_weights: Optional[torch.Tensor],
    ) -> torch.Tensor:
        mode = self.config.pred_normal_loss_mode
        if mode == "weighted_cosine_legacy":
            if sample_weights is None:
                raise ValueError("pred_normal_loss_mode='weighted_cosine_legacy' requires sample weights.")
            return pred_normal_loss(sample_weights.detach(), normals.detach(), pred_normals)
        if mode == "l2_paper":
            return pred_normal_l2_loss(normals.detach(), pred_normals)
        raise ValueError(f"Unknown pred_normal_loss_mode: {mode}")

    def _kernel_size_smoothness_residual(
        self,
        inv_s_deviated: torch.Tensor,
        inv_s: torch.Tensor,
    ) -> torch.Tensor:
        mode = self.config.kernel_size_smoothness_loss_mode
        if mode == "log_l1_current":
            return torch.abs(torch.log(1.0 / inv_s_deviated) - torch.log(1.0 / inv_s))
        if mode == "log_l2_legacy":
            return torch.square(torch.log(1.0 / inv_s_deviated) - torch.log(1.0 / inv_s))
        if mode == "kernel_l2_legacy":
            return torch.square(1.0 / inv_s_deviated - 1.0 / inv_s)
        raise ValueError(f"Unknown kernel_size_smoothness_loss_mode: {mode}")

    @staticmethod
    def _add_numeric_tensor_metrics(metrics_dict: Dict, prefix: str, tensor: Optional[torch.Tensor]) -> None:
        if tensor is None or not torch.is_tensor(tensor) or tensor.numel() == 0:
            return
        with torch.no_grad():
            flat = tensor.detach().reshape(-1).to(torch.float32)
            finite = torch.isfinite(flat)
            metrics_dict[f"{prefix}/nonfinite_frac"] = float((~finite).to(torch.float32).mean().item())
            if not bool(torch.any(finite).item()):
                metrics_dict[f"{prefix}/finite_frac"] = 0.0
                return
            finite_values = flat[finite]
            abs_values = torch.abs(finite_values)
            metrics_dict[f"{prefix}/finite_frac"] = float(finite.to(torch.float32).mean().item())
            metrics_dict[f"{prefix}/min"] = float(finite_values.min().item())
            metrics_dict[f"{prefix}/max"] = float(finite_values.max().item())
            metrics_dict[f"{prefix}/mean"] = float(finite_values.mean().item())
            metrics_dict[f"{prefix}/abs_max"] = float(abs_values.max().item())
            metrics_dict[f"{prefix}/rms"] = float(torch.sqrt(torch.mean(finite_values.square())).item())

    def _add_stage1_numeric_metrics(self, metrics_dict: Dict, outputs: Dict) -> None:
        field_outputs = outputs.get("field_outputs")
        if not isinstance(field_outputs, dict):
            return

        raw_inv_s = field_outputs.get("raw_inv_s")
        inv_s = field_outputs.get("inv_s")
        density = field_outputs.get(FieldHeadNames.DENSITY)
        alpha = field_outputs.get(FieldHeadNames.ALPHA)
        sdf = field_outputs.get(FieldHeadNames.SDF)
        gradient = field_outputs.get(FieldHeadNames.GRADIENT)
        pred_gradient = field_outputs.get("pred_gradient")

        self._add_numeric_tensor_metrics(metrics_dict, "numeric/sdf", sdf)
        self._add_numeric_tensor_metrics(metrics_dict, "numeric/raw_inv_s", raw_inv_s)
        self._add_numeric_tensor_metrics(metrics_dict, "numeric/inv_s", inv_s)
        self._add_numeric_tensor_metrics(metrics_dict, "numeric/density", density)
        self._add_numeric_tensor_metrics(metrics_dict, "numeric/alpha", alpha)
        self._add_numeric_tensor_metrics(metrics_dict, "numeric/gradient_norm", gradient.norm(dim=-1) if torch.is_tensor(gradient) else None)
        self._add_numeric_tensor_metrics(
            metrics_dict,
            "numeric/pred_gradient_norm",
            pred_gradient.norm(dim=-1) if torch.is_tensor(pred_gradient) else None,
        )
        if torch.is_tensor(inv_s):
            kernel_size = 1.0 / (inv_s.detach().to(torch.float32) + 1e-6)
            self._add_numeric_tensor_metrics(metrics_dict, "numeric/kernel_size", kernel_size)
        if torch.is_tensor(raw_inv_s):
            exp_factor = float(getattr(self.field.config, "kernel_size_transform_exp_factor", 10.0))
            exp_arg = raw_inv_s.detach().to(torch.float32) * exp_factor
            self._add_numeric_tensor_metrics(metrics_dict, "numeric/kernel_exp_arg", exp_arg)
            finite = torch.isfinite(exp_arg.reshape(-1))
            if bool(torch.any(finite).item()):
                finite_arg = exp_arg.reshape(-1)[finite]
                upper_clip = math.log(1e6)
                lower_clip = math.log(1e-6)
                overflow = math.log(torch.finfo(torch.float32).max)
                metrics_dict["numeric/kernel_exp_arg/upper_clip_frac"] = float(
                    (finite_arg > upper_clip).to(torch.float32).mean().item()
                )
                metrics_dict["numeric/kernel_exp_arg/lower_clip_frac"] = float(
                    (finite_arg < lower_clip).to(torch.float32).mean().item()
                )
                metrics_dict["numeric/kernel_exp_arg/fp32_overflow_risk_frac"] = float(
                    (finite_arg > overflow).to(torch.float32).mean().item()
                )

    def set_shell_profile_options(self, strict_timing: bool = False) -> None:
        self._shell_profile_options["strict_timing"] = bool(strict_timing)
        if hasattr(self, "field") and hasattr(self.field, "set_profile_options"):
            self.field.set_profile_options(strict_timing=bool(strict_timing))
        if hasattr(self, "sampler") and hasattr(self.sampler, "set_profile_options"):
            self.sampler.set_profile_options(strict_timing=bool(strict_timing))
        optix_tracer = getattr(self, "_optix_tracer", None)
        if optix_tracer is not None and hasattr(optix_tracer, "set_profile_options"):
            optix_tracer.set_profile_options(strict_timing=bool(strict_timing))

    def _is_strict_shell_profile_enabled(self) -> bool:
        return bool(getattr(self, "_shell_profile_options", {}).get("strict_timing", False))

    def _sync_cuda_for_profile(self, device: Optional[torch.device] = None) -> None:
        if not self._is_strict_shell_profile_enabled() or not torch.cuda.is_available():
            return
        sync_device = device if device is not None else getattr(self, "device", None)
        if sync_device is None:
            return
        if not isinstance(sync_device, torch.device):
            sync_device = torch.device(sync_device)
        if sync_device.type == "cuda":
            torch.cuda.synchronize(device=sync_device)

    def reset_shell_profile(self) -> None:
        self._shell_profile = {
            "chunks": 0,
            "chunk_rays": 0,
            "trace_and_pack_ms": 0.0,
            "sampler_total_ms": 0.0,
            "torch_to_numpy_ms": 0.0,
            "numpy_nonzero_ms": 0.0,
            "numpy_to_torch_ms": 0.0,
            "ray_samples_build_ms": 0.0,
            "optix_launch_ms": 0.0,
            "interval_pair_ms": 0.0,
            "sample_pack_ms": 0.0,
            "far_fill_ms": 0.0,
            "trace_h2d_ms": 0.0,
            "trace_other_ms": 0.0,
            "input_prep_ms": 0.0,
            "valid_count_reduce_ms": 0.0,
            "valid_count_d2h_ms": 0.0,
            "valid_count_hist_ms": 0.0,
            "compact_indices_ms": 0.0,
            "shell_count_meta_ms": 0.0,
            "sample_gather_ms": 0.0,
            "sampler_other_ms": 0.0,
            "field_eval_ms": 0.0,
            "field_geonet_ms": 0.0,
            "field_gradient_ms": 0.0,
            "field_color_ms": 0.0,
            "field_alpha_ms": 0.0,
            "field_other_ms": 0.0,
            "alpha_weights_ms": 0.0,
            "sampler_profile_merge_ms": 0.0,
            "sample_return_build_ms": 0.0,
            "sample_forward_total_ms": 0.0,
            "sample_forward_other_ms": 0.0,
            "outputs_compose_ms": 0.0,
            "outputs_post_ms": 0.0,
            "get_outputs_total_ms": 0.0,
            "get_outputs_other_ms": 0.0,
            "num_rays": 0,
            "hit_rays": 0,
            "valid_samples": 0,
            "sample_budget": 0,
            "valid_count_hist": None,
            "single_sample_grad_count": 0,
            "single_sample_grad_norm_sum": 0.0,
            "single_sample_grad_norm_sq_sum": 0.0,
            "single_sample_grad_norm_min": float("inf"),
            "single_sample_grad_norm_max": 0.0,
            "single_sample_abs_true_cos_sum": 0.0,
        }
        if hasattr(self, "sampler") and hasattr(self.sampler, "reset_profile"):
            self.sampler.reset_profile()

    def get_shell_profile_summary(self) -> dict:
        profile = dict(self._shell_profile)
        hist = profile.pop("valid_count_hist", None)
        total_rays = int(profile["num_rays"])
        if hist is not None and total_rays > 0:
            cdf = np.cumsum(hist)
            p50 = int(np.searchsorted(cdf, max(0, int(np.ceil(0.50 * total_rays)) - 1), side="left"))
            p95 = int(np.searchsorted(cdf, max(0, int(np.ceil(0.95 * total_rays)) - 1), side="left"))
        else:
            p50 = 0
            p95 = 0
        single_sample_rays = int(hist[1]) if hist is not None and len(hist) > 1 else 0
        hit_rays = int(profile["hit_rays"])
        single_grad_count = int(profile.pop("single_sample_grad_count", 0))
        single_grad_norm_sum = float(profile.pop("single_sample_grad_norm_sum", 0.0))
        single_grad_norm_sq_sum = float(profile.pop("single_sample_grad_norm_sq_sum", 0.0))
        single_grad_norm_min_raw = float(profile.pop("single_sample_grad_norm_min", 0.0))
        single_grad_norm_max = float(profile.pop("single_sample_grad_norm_max", 0.0))
        single_abs_true_cos_sum = float(profile.pop("single_sample_abs_true_cos_sum", 0.0))
        profile["mean_valid_per_ray"] = float(profile["valid_samples"]) / max(float(total_rays), 1.0)
        profile["hit_frac"] = float(hit_rays) / max(float(total_rays), 1.0)
        profile["utilization"] = float(profile["valid_samples"]) / max(float(profile["sample_budget"]), 1.0)
        profile["budget_per_ray"] = float(profile["sample_budget"]) / max(float(total_rays), 1.0)
        profile["mean_valid_per_hit_ray"] = float(profile["valid_samples"]) / max(float(hit_rays), 1.0)
        profile["single_sample_hit_frac"] = float(single_sample_rays) / max(float(hit_rays), 1.0)
        profile["single_sample_rays"] = single_sample_rays
        profile["single_sample_grad_count"] = single_grad_count
        if single_grad_count > 0:
            grad_norm_mean = single_grad_norm_sum / float(single_grad_count)
            grad_norm_var = max(single_grad_norm_sq_sum / float(single_grad_count) - grad_norm_mean * grad_norm_mean, 0.0)
            profile["single_sample_grad_norm_mean"] = grad_norm_mean
            profile["single_sample_grad_norm_std"] = float(np.sqrt(grad_norm_var))
            profile["single_sample_grad_norm_min"] = single_grad_norm_min_raw
            profile["single_sample_grad_norm_max"] = single_grad_norm_max
            profile["single_sample_abs_true_cos_mean"] = single_abs_true_cos_sum / float(single_grad_count)
        else:
            profile["single_sample_grad_norm_mean"] = 0.0
            profile["single_sample_grad_norm_std"] = 0.0
            profile["single_sample_grad_norm_min"] = 0.0
            profile["single_sample_grad_norm_max"] = 0.0
            profile["single_sample_abs_true_cos_mean"] = 0.0
        profile["sample_forward_other_ms"] = max(
            float(profile["sample_forward_total_ms"])
            - (
                float(profile["sampler_total_ms"])
                + float(profile["field_eval_ms"])
                + float(profile["alpha_weights_ms"])
                + float(profile["sampler_profile_merge_ms"])
                + float(profile["sample_return_build_ms"])
            ),
            0.0,
        )
        profile["trace_other_ms"] = max(
            float(profile["trace_and_pack_ms"])
            - (
                float(profile["trace_h2d_ms"])
                + float(profile["optix_launch_ms"])
                + float(profile["interval_pair_ms"])
                + float(profile["sample_pack_ms"])
                + float(profile["far_fill_ms"])
            ),
            0.0,
        )
        profile["sampler_other_ms"] = max(
            float(profile["sampler_total_ms"])
            - (
                float(profile["input_prep_ms"])
                + float(profile["trace_and_pack_ms"])
                + float(profile["valid_count_reduce_ms"])
                + float(profile["valid_count_d2h_ms"])
                + float(profile["compact_indices_ms"])
                + float(profile["shell_count_meta_ms"])
                + float(profile["sample_gather_ms"])
                + float(profile["ray_samples_build_ms"])
                + float(profile["valid_count_hist_ms"])
            ),
            0.0,
        )
        profile["field_other_ms"] = max(
            float(profile["field_eval_ms"])
            - (
                float(profile["field_geonet_ms"])
                + float(profile["field_gradient_ms"])
                + float(profile["field_color_ms"])
                + float(profile["field_alpha_ms"])
            ),
            0.0,
        )
        profile["get_outputs_other_ms"] = max(
            float(profile["get_outputs_total_ms"])
            - (
                float(profile["sample_forward_total_ms"])
                + float(profile["outputs_compose_ms"])
                + float(profile["outputs_post_ms"])
            ),
            0.0,
        )
        profile["p50_valid_per_ray"] = p50
        profile["p95_valid_per_ray"] = p95
        return profile

    def _merge_sampler_profile(self, sampler_profile: dict, chunk_rays: int) -> float:
        merge_start = time.perf_counter()
        profile = self._shell_profile
        hist = sampler_profile["valid_count_hist"]
        if profile["valid_count_hist"] is None:
            profile["valid_count_hist"] = hist.copy()
        else:
            profile["valid_count_hist"] += hist
        profile["chunks"] += 1
        profile["chunk_rays"] += int(chunk_rays)
        numeric_fields = (
            "trace_and_pack_ms",
            "sampler_total_ms",
            "torch_to_numpy_ms",
            "numpy_nonzero_ms",
            "numpy_to_torch_ms",
            "ray_samples_build_ms",
            "optix_launch_ms",
            "interval_pair_ms",
            "sample_pack_ms",
            "far_fill_ms",
            "trace_h2d_ms",
            "trace_other_ms",
            "input_prep_ms",
            "valid_count_reduce_ms",
            "valid_count_d2h_ms",
            "valid_count_hist_ms",
            "compact_indices_ms",
            "shell_count_meta_ms",
            "sample_gather_ms",
            "sampler_other_ms",
        )
        for key in numeric_fields:
            profile[key] += float(sampler_profile.get(key, 0.0))
        profile["num_rays"] += int(sampler_profile["num_rays"])
        profile["hit_rays"] += int(sampler_profile["hit_rays"])
        profile["valid_samples"] += int(sampler_profile["valid_samples"])
        profile["sample_budget"] += int(sampler_profile["sample_budget"])
        return (time.perf_counter() - merge_start) * 1000.0

        # self.density_fns = []
        # num_prop_nets = self.config.num_proposal_iterations
        # Build the proposal network(s)
        # self.proposal_networks = torch.nn.ModuleList()
        # if self.config.use_same_proposal_network:
        #     assert len(self.config.proposal_net_args_list) == 1, "Only one proposal network is allowed."
        #     prop_net_args = self.config.proposal_net_args_list[0]
        #     network = HashMLPDensityField(
        #         self.scene_box.aabb, spatial_distortion=self.scene_contraction, **prop_net_args
        #     )
        #     self.proposal_networks.append(network)
        #     self.density_fns.extend([network.density_fn for _ in range(num_prop_nets)])
        # else:
        #     for i in range(num_prop_nets):
        #         prop_net_args = self.config.proposal_net_args_list[min(i, len(self.config.proposal_net_args_list) - 1)]
        #         network = HashMLPDensityField(
        #             self.scene_box.aabb,
        #             spatial_distortion=self.scene_contraction,
        #             **prop_net_args,
        #         )
        #         self.proposal_networks.append(network)
        #     self.density_fns.extend([network.density_fn for network in self.proposal_networks])

        # update proposal network every iterations
        # update_schedule = lambda step: -1

        # self.proposal_sampler = ProposalNetworkSampler(
        #     num_nerf_samples_per_ray=self.config.num_neus_samples_per_ray,
        #     num_proposal_samples_per_ray=self.config.num_proposal_samples_per_ray,
        #     num_proposal_network_iterations=self.config.num_proposal_iterations,
        #     use_uniform_sampler=False,
        #     single_jitter=self.config.use_single_jitter,
        #     update_sched=update_schedule,
        # )
        # self.sampler = NeuSAccSampler(
        #     aabb=self.scene_box.aabb,
        #     neus_sampler=self.sampler,
        #     num_samples=self.config.num_samples,
        #     num_samples_importance=self.config.num_samples_importance,
        # )

    # def get_param_groups(self) -> Dict[str, List[Parameter]]:
    #     param_groups = super().get_param_groups()
    #     param_groups["proposal_networks"] = list(self.proposal_networks.parameters())
    #     return param_groups

    def load_shell_mesh(self, dilation_path: str, erosion_path: str):
        dilation_mesh = None
        erosion_mesh = None
        try:
            dilation_mesh = trimesh.load(dilation_path, file_type="ply")
        except Exception as e:
            print(f"Error loading dilation mesh: {e}")
        try:
            erosion_mesh = trimesh.load(erosion_path, file_type="ply")
        except Exception as e:
            print(f"Error loading erosion mesh: {e}")
        # TODO check watertight and maybe fill holes
        return dilation_mesh, erosion_mesh

    def get_training_callbacks(
        self, training_callback_attributes: TrainingCallbackAttributes
    ) -> List[TrainingCallback]:
        callbacks = super().get_training_callbacks(training_callback_attributes)

        # if self.config.use_proposal_weight_anneal:
        #     # anneal the weights of the proposal network before doing PDF sampling
        #     N = self.config.proposal_weights_anneal_max_num_iters

        #     def set_anneal(step):
        #         # https://arxiv.org/pdf/2111.12077.pdf eq. 18
        #         train_frac = np.clip(step / N, 0, 1)
        #         bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
        #         anneal = bias(train_frac, self.config.proposal_weights_anneal_slope)
        #         self.proposal_sampler.set_anneal(anneal)

        #     callbacks.append(
        #         TrainingCallback(
        #             where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
        #             update_every_num_iters=1,
        #             func=set_anneal,
        #         )
        #     )
        #     callbacks.append(
        #         TrainingCallback(
        #             where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
        #             update_every_num_iters=1,
        #             func=self.proposal_sampler.step_cb,
        #         )
        #     )
        # add sampler call backs

        # sdf_fn = lambda x: self.field.forward_geonetwork(x)[:, 0].contiguous()
        # # inv_s = self.field.deviation_network.get_variance
        # callbacks.append(
        #     TrainingCallback(
        #         where_to_run=[TrainingCallbackLocation.AFTER_TRAIN_ITERATION],
        #         update_every_num_iters=1,
        #         func=self.sampler.update_binary_grid,
        #         kwargs={"sdf_fn": sdf_fn, "inv_s": inv_s},
        #     )
        # )

        # callbacks.append(
        #     TrainingCallback(
        #         where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
        #         update_every_num_iters=1,
        #         func=self.sampler.update_step_size,
        #         kwargs={"inv_s": inv_s},
        #     )
        # )

        # if self.config.use_anneal_beta:
        #     # anneal the beta of volsdf before each training iterations
        #     M = self.config.beta_anneal_max_num_iters
        #     beta_init = self.config.beta_anneal_init
        #     beta_end = self.config.beta_anneal_end

            # def set_beta(step):
            #     # bakedsdf's beta schedule adapted to neus
            #     train_frac = np.clip(step / M, 0, 1)
            #     beta = beta_init / (1 + (beta_init - beta_end) / beta_end * (train_frac**0.8))
            #     beta = np.log(1.0 / beta) / 10.0
            #     self.field.deviation_network.variance.data[...] = beta

            # callbacks.append(
            #     TrainingCallback(
            #         where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
            #         update_every_num_iters=1,
            #         func=set_beta,
            #     )
            # )

        # read the hash encoding parameters from field
        max_res = self.field.max_res
        base_res = self.field.base_res
        growth_factor = self.field.growth_factor
        steps_per_level = self.config.steps_per_level
        init_delta = 1.0 / base_res

        if self.config.enable_progressive_hash_encoding or self.config.enable_numerical_gradients_schedule:
            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=self._update_progressive_training_state,
                )
            )
        # schedule the curvature loss weight
        # linear warmup for 5000 steps to 5e-4 and then decay as delta
        self.curvature_loss_multi_factor = 1.0
        if self.config.enable_curvature_loss_schedule:

            def set_curvature_loss_mult_factor(step):
                if step < self.config.curvature_loss_warmup_steps:
                    factor = step / self.config.curvature_loss_warmup_steps
                else:
                    delta = 1.0 / (
                        base_res * growth_factor ** ((step - self.config.curvature_loss_warmup_steps) / steps_per_level)
                    )
                    delta = max(1.0 / (max_res * 10.0), delta)
                    factor = delta / init_delta

                self.curvature_loss_multi_factor = factor

            callbacks.append(
                TrainingCallback(
                    where_to_run=[TrainingCallbackLocation.BEFORE_TRAIN_ITERATION],
                    update_every_num_iters=1,
                    func=set_curvature_loss_mult_factor,
                )
            )

        # TODO switch to analytic gradients after delta is small enough?

        return callbacks

    # def sample_and_forward_field(self, ray_bundle: RayBundle):
    #     ray_samples, weights_list, ray_samples_list = self.proposal_sampler(ray_bundle, density_fns=self.density_fns)

    #     field_outputs = self.field(ray_samples, return_alphas=True)

    #     if self.config.background_model != "none":
    #         field_outputs = self.forward_background_field_and_merge(ray_samples, field_outputs)

    #     weights = ray_samples.get_weights_from_alphas(field_outputs[FieldHeadNames.ALPHA])

    #     weights_list.append(weights)
    #     ray_samples_list.append(ray_samples)

    #     samples_and_field_outputs = {
    #         "ray_samples": ray_samples,
    #         "field_outputs": field_outputs,
    #         "weights": weights,
    #         "weights_list": weights_list,
    #         "ray_samples_list": ray_samples_list,
    #     }
    #     return samples_and_field_outputs

    def _use_packed_shell_path(self) -> bool:
        """Whether to run the packed shell contract for the current mode."""
        use_shell_sampler = self.config.shell_inference_mode or self.config.shell_finetune_mode
        return (
            use_shell_sampler
            and self.config.shell_use_packed_samples
            and hasattr(self.sampler, "generate_packed_ray_samples")
        )

    def sample_and_forward_field(self, ray_bundle: RayBundle) -> Dict:
        profile_device = ray_bundle.origins.device
        self._sync_cuda_for_profile(profile_device)
        sample_forward_start = time.perf_counter()
        use_shell_sampler = self.config.shell_inference_mode or self.config.shell_finetune_mode
        use_packed_shell = self._use_packed_shell_path()
        sampler_profile_merge_ms = 0.0
        sample_return_build_ms = 0.0
        if use_packed_shell:
            ray_samples, ray_indices, num_rays, shell_valid_count, shell_hit = self.sampler.generate_packed_ray_samples(ray_bundle)
            sampler_profile = self.sampler.get_last_profile() if hasattr(self.sampler, "get_last_profile") else None
            if ray_samples is None or ray_indices is None:
                if sampler_profile is not None:
                    sampler_profile_merge_ms = self._merge_sampler_profile(sampler_profile, chunk_rays=int(num_rays))
                    self._shell_profile["sampler_profile_merge_ms"] += sampler_profile_merge_ms
                self._sync_cuda_for_profile(profile_device)
                return_build_start = time.perf_counter()
                empty_result = {
                    "ray_samples": None,
                    "field_outputs": None,
                    "weights": None,
                    "bg_transmittance": torch.ones(
                        (num_rays, 1), device=ray_bundle.origins.device, dtype=ray_bundle.origins.dtype
                    ),
                    "ray_indices": None,
                    "num_rays": num_rays,
                    "shell_valid_count": shell_valid_count,
                    "shell_hit": shell_hit,
                }
                self._sync_cuda_for_profile(profile_device)
                sample_return_build_ms = (time.perf_counter() - return_build_start) * 1000.0
                self._shell_profile["sample_return_build_ms"] += sample_return_build_ms
                if sampler_profile is not None:
                    self._sync_cuda_for_profile(profile_device)
                    self._shell_profile["sample_forward_total_ms"] += (time.perf_counter() - sample_forward_start) * 1000.0
                return empty_result
        else:
            if use_shell_sampler:
                ray_samples = self.sampler(ray_bundle)
                sampler_profile = self.sampler.get_last_profile() if hasattr(self.sampler, "get_last_profile") else None
            else:
                ray_samples = self.sampler(
                    ray_bundle,
                    sdf_fn=self.field.get_sdf,
                    sdf_inv_s_fn=self.field.get_sdf_and_inv_s if self.config.use_learned_inv_s_for_upsampling else None,
                )
                sampler_profile = None
        # save_points("a.ply", ray_samples.frustums.get_start_positions().reshape(-1, 3).detach().cpu().numpy())
        self._sync_cuda_for_profile(profile_device)
        field_eval_start = time.perf_counter()
        debug_surface_maps = (not self.training) and (
            self.config.eval_visualize_surface_proxies or self.config.eval_visualize_kernel_size
        )
        render_minimal = use_packed_shell and (not self.training) and (not debug_surface_maps)
        field_outputs = self.field(ray_samples, return_alphas=True, render_minimal=render_minimal)
        self._sync_cuda_for_profile(profile_device)
        field_eval_ms = (time.perf_counter() - field_eval_start) * 1000.0
        field_profile = self.field.get_last_profile() if hasattr(self.field, "get_last_profile") else None
        self._sync_cuda_for_profile(profile_device)
        alpha_weights_start = time.perf_counter()
        if use_packed_shell:
            weights = nerfacc.render_weight_from_alpha(
                field_outputs[FieldHeadNames.ALPHA], ray_indices=ray_indices, n_rays=num_rays
            )
            bg_transmittance = 1.0 - nerfacc.accumulate_along_rays(weights, ray_indices, None, num_rays)
            self._sync_cuda_for_profile(profile_device)
            alpha_weights_ms = (time.perf_counter() - alpha_weights_start) * 1000.0
        else:
            weights, transmittance = ray_samples.get_weights_and_transmittance_from_alphas(
                field_outputs[FieldHeadNames.ALPHA]
            )
            self._sync_cuda_for_profile(profile_device)
            alpha_weights_ms = (time.perf_counter() - alpha_weights_start) * 1000.0
            bg_transmittance = transmittance[:, -1, :]

        if sampler_profile is not None:
            sampler_profile_merge_ms = self._merge_sampler_profile(
                sampler_profile, chunk_rays=int(num_rays if use_packed_shell else weights.shape[0])
            )
            self._shell_profile["field_eval_ms"] += field_eval_ms
            if field_profile is not None:
                self._shell_profile["field_geonet_ms"] += float(field_profile.get("geonet_ms", 0.0))
                self._shell_profile["field_gradient_ms"] += float(field_profile.get("gradient_ms", 0.0))
                self._shell_profile["field_color_ms"] += float(field_profile.get("color_ms", 0.0))
                self._shell_profile["field_alpha_ms"] += float(field_profile.get("alpha_in_field_ms", 0.0))
                single_grad_count = int(field_profile.get("single_sample_count", 0))
                self._shell_profile["single_sample_grad_count"] += single_grad_count
                self._shell_profile["single_sample_grad_norm_sum"] += float(
                    field_profile.get("single_sample_grad_norm_sum", 0.0)
                )
                self._shell_profile["single_sample_grad_norm_sq_sum"] += float(
                    field_profile.get("single_sample_grad_norm_sq_sum", 0.0)
                )
                self._shell_profile["single_sample_abs_true_cos_sum"] += float(
                    field_profile.get("single_sample_abs_true_cos_sum", 0.0)
                )
                if single_grad_count > 0:
                    self._shell_profile["single_sample_grad_norm_min"] = min(
                        float(self._shell_profile["single_sample_grad_norm_min"]),
                        float(field_profile.get("single_sample_grad_norm_min", 0.0)),
                    )
                    self._shell_profile["single_sample_grad_norm_max"] = max(
                        float(self._shell_profile["single_sample_grad_norm_max"]),
                        float(field_profile.get("single_sample_grad_norm_max", 0.0)),
                    )
            self._shell_profile["alpha_weights_ms"] += alpha_weights_ms
            self._shell_profile["sampler_profile_merge_ms"] += sampler_profile_merge_ms

        self._sync_cuda_for_profile(profile_device)
        return_build_start = time.perf_counter()
        samples_and_field_outputs = {
            "ray_samples": ray_samples,
            "field_outputs": field_outputs,
            "weights": weights,
            "bg_transmittance": bg_transmittance,
        }
        if use_packed_shell:
            samples_and_field_outputs.update(
                {
                    "ray_indices": ray_indices,
                    "num_rays": num_rays,
                    "shell_valid_count": shell_valid_count,
                    "shell_hit": shell_hit,
                }
            )
        self._sync_cuda_for_profile(profile_device)
        sample_return_build_ms = (time.perf_counter() - return_build_start) * 1000.0
        self._shell_profile["sample_return_build_ms"] += sample_return_build_ms
        if sampler_profile is not None:
            self._sync_cuda_for_profile(profile_device)
            self._shell_profile["sample_forward_total_ms"] += (time.perf_counter() - sample_forward_start) * 1000.0
        return samples_and_field_outputs
    
    def get_outputs(self, ray_bundle: RayBundle):
        profile_device = ray_bundle.origins.device
        self._sync_cuda_for_profile(profile_device)
        get_outputs_start = time.perf_counter()
        use_packed_shell = self._use_packed_shell_path()
        if use_packed_shell:
            samples_and_field_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle)
            ray_indices = samples_and_field_outputs.get("ray_indices")
            num_rays = samples_and_field_outputs.get("num_rays", len(ray_bundle))
            shell_valid_count = samples_and_field_outputs.get("shell_valid_count")
            shell_hit = samples_and_field_outputs.get("shell_hit")
            if ray_indices is None or samples_and_field_outputs["ray_samples"] is None:
                self._sync_cuda_for_profile(profile_device)
                outputs_compose_start = time.perf_counter()
                accumulation = torch.zeros((num_rays, 1), device=ray_bundle.origins.device, dtype=ray_bundle.origins.dtype)
                depth = ray_bundle.fars.clone() if ray_bundle.fars is not None else torch.zeros_like(accumulation)
                outputs = {
                    "rgb": self.renderer_rgb(
                        rgb=torch.zeros((0, 3), device=ray_bundle.origins.device, dtype=ray_bundle.origins.dtype),
                        weights=torch.zeros((0, 1), device=ray_bundle.origins.device, dtype=ray_bundle.origins.dtype),
                        ray_indices=torch.zeros((0,), device=ray_bundle.origins.device, dtype=torch.long),
                        num_rays=num_rays,
                    ),
                    "accumulation": accumulation,
                    "depth": depth / ray_bundle.directions_norm,
                    "normal": torch.zeros((num_rays, 3), device=ray_bundle.origins.device, dtype=ray_bundle.origins.dtype),
                    "weights": accumulation,
                    "directions_norm": ray_bundle.directions_norm,
                    "packed_shell_contract": True,
                }
                if not self.training:
                    outputs["pred_normals"] = torch.zeros(
                        (num_rays, 3), device=ray_bundle.origins.device, dtype=ray_bundle.origins.dtype
                    )
                    if self.config.eval_visualize_surface_proxies:
                        zero_debug = torch.zeros(
                            (num_rays, 1), device=ray_bundle.origins.device, dtype=ray_bundle.origins.dtype
                        )
                        outputs.update(
                            {
                                "min_abs_sdf": zero_debug,
                                "surface_kernel_size": zero_debug,
                                "surface_alpha": zero_debug,
                                "max_alpha": zero_debug,
                                "max_alpha_kernel_size": zero_debug,
                                "weighted_kernel_size": zero_debug,
                            }
                        )
                self._sync_cuda_for_profile(profile_device)
                outputs_compose_ms = (time.perf_counter() - outputs_compose_start) * 1000.0
                self._sync_cuda_for_profile(profile_device)
                outputs_post_start = time.perf_counter()
                if shell_valid_count is not None and shell_hit is not None:
                    budget = max(1, (self.config.n_depth_peeling_max // 2) * self.config.n_sample_max)
                    outputs["shell_valid_count"] = shell_valid_count
                    outputs["shell_hit"] = shell_hit
                    outputs["shell_valid_frac"] = shell_valid_count / float(budget)
                self._sync_cuda_for_profile(profile_device)
                outputs_post_ms = (time.perf_counter() - outputs_post_start) * 1000.0
                self._shell_profile["outputs_compose_ms"] += outputs_compose_ms
                self._shell_profile["outputs_post_ms"] += outputs_post_ms
                self._sync_cuda_for_profile(profile_device)
                self._shell_profile["get_outputs_total_ms"] += (time.perf_counter() - get_outputs_start) * 1000.0
                return outputs

            field_outputs = samples_and_field_outputs["field_outputs"]
            ray_samples = samples_and_field_outputs["ray_samples"]
            weights = samples_and_field_outputs["weights"]
            self._sync_cuda_for_profile(profile_device)
            outputs_compose_start = time.perf_counter()
            accumulation = self.renderer_accumulation(weights=weights, ray_indices=ray_indices, num_rays=num_rays)
            rgb = self.renderer_rgb(
                rgb=field_outputs[FieldHeadNames.RGB],
                weights=weights,
                ray_indices=ray_indices,
                num_rays=num_rays,
            )
            depth = self.renderer_depth(
                weights=weights,
                ray_samples=ray_samples,
                ray_indices=ray_indices,
                num_rays=num_rays,
            )
            if ray_bundle.fars is not None:
                depth = torch.where(accumulation > 1e-6, depth, ray_bundle.fars)
            depth = depth / ray_bundle.directions_norm
            normal = nerfacc.accumulate_along_rays(
                weights, ray_indices, field_outputs[FieldHeadNames.NORMAL], num_rays
            )
            pred_normals = None
            if FieldHeadNames.PRED_NORMALS in field_outputs:
                pred_normals = nerfacc.accumulate_along_rays(
                    weights, ray_indices, field_outputs[FieldHeadNames.PRED_NORMALS], num_rays
                )
            self._sync_cuda_for_profile(profile_device)
            outputs_compose_ms = (time.perf_counter() - outputs_compose_start) * 1000.0

            self._sync_cuda_for_profile(profile_device)
            outputs_post_start = time.perf_counter()
            outputs = {
                "rgb": rgb,
                "accumulation": accumulation,
                "depth": depth,
                "normal": normal,
                "weights": accumulation,
                "directions_norm": ray_bundle.directions_norm,
                "packed_shell_contract": True,
            }
            if pred_normals is not None:
                outputs["pred_normals"] = pred_normals
            if (
                (not self.training)
                and self.config.eval_visualize_surface_proxies
                and FieldHeadNames.SDF in field_outputs
                and "inv_s" in field_outputs
                and FieldHeadNames.ALPHA in field_outputs
            ):
                outputs.update(
                    self._compute_surface_debug_packed(
                        field_outputs=field_outputs,
                        weights=weights,
                        ray_indices=ray_indices,
                        num_rays=num_rays,
                    )
                )
            if shell_valid_count is not None and shell_hit is not None:
                budget = max(1, (self.config.n_depth_peeling_max // 2) * self.config.n_sample_max)
                outputs["shell_valid_count"] = shell_valid_count
                outputs["shell_hit"] = shell_hit
                outputs["shell_valid_frac"] = shell_valid_count / float(budget)
            if self.training:
                field_outputs = samples_and_field_outputs["field_outputs"]
                packed_weights = samples_and_field_outputs["weights"]
                if field_outputs is not None and ray_indices is not None:
                    if FieldHeadNames.GRADIENT in field_outputs:
                        outputs["eik_grad"] = field_outputs[FieldHeadNames.GRADIENT]
                    if "points_norm" in field_outputs:
                        outputs["points_norm"] = field_outputs["points_norm"]
                    outputs["field_outputs"] = field_outputs
                    outputs["ray_samples"] = samples_and_field_outputs["ray_samples"]
                    outputs["ray_indices"] = ray_indices
                    outputs["num_rays"] = num_rays
                    outputs["packed_sample_weights"] = packed_weights
                    if samples_and_field_outputs.get("bg_transmittance") is not None:
                        outputs["bg_transmittance"] = samples_and_field_outputs["bg_transmittance"]
                    fd_losses_enabled = self._finite_difference_losses_enabled()
                    color_only_stage2 = self.config.stage2_color_only_loss
                    enable_stage2_pred_normal = color_only_stage2 and self.config.stage2_enable_pred_normal_loss
                    if fd_losses_enabled and self.config.pred_normal_loss_mult > 0.0 and (
                        not color_only_stage2 or enable_stage2_pred_normal
                    ):
                        if FieldHeadNames.PRED_NORMALS in field_outputs and FieldHeadNames.NORMAL in field_outputs:
                            outputs["rendered_pred_normal_loss"] = self._pred_normal_residual(
                                field_outputs[FieldHeadNames.NORMAL],
                                field_outputs[FieldHeadNames.PRED_NORMALS],
                                packed_weights if torch.is_tensor(packed_weights) else None,
                            )
                    enable_stage2_kernel_smoothness = (
                        color_only_stage2 and self.config.stage2_enable_kernel_smoothness_loss
                    )
                    if self.config.kernel_size_smoothness_loss_mult > 0.0 and (
                        not color_only_stage2 or enable_stage2_kernel_smoothness
                    ):
                        inv_s_deviated = field_outputs.get("inv_s_deviated")
                        inv_s = field_outputs.get("inv_s")
                        if inv_s_deviated is not None and inv_s is not None:
                            outputs["kernel_size_smoothness_loss"] = self._kernel_size_smoothness_residual(
                                inv_s_deviated=inv_s_deviated,
                                inv_s=inv_s,
                            )
            self._sync_cuda_for_profile(profile_device)
            outputs_post_ms = (time.perf_counter() - outputs_post_start) * 1000.0
            self._shell_profile["outputs_compose_ms"] += outputs_compose_ms
            self._shell_profile["outputs_post_ms"] += outputs_post_ms
            self._sync_cuda_for_profile(profile_device)
            self._shell_profile["get_outputs_total_ms"] += (time.perf_counter() - get_outputs_start) * 1000.0
            return outputs

        outputs = super().get_outputs(ray_bundle)
        if (not self.training) and self.config.eval_visualize_surface_proxies:
            debug_outputs = self.sample_and_forward_field(ray_bundle=ray_bundle)
            dense_field_outputs = debug_outputs.get("field_outputs")
            dense_weights = debug_outputs.get("weights")
            if (
                dense_field_outputs is not None
                and dense_weights is not None
                and FieldHeadNames.SDF in dense_field_outputs
                and "inv_s" in dense_field_outputs
                and FieldHeadNames.ALPHA in dense_field_outputs
            ):
                outputs.update(
                    self._compute_surface_debug_dense(
                        field_outputs=dense_field_outputs,
                        weights=dense_weights,
                    )
                )
        self._sync_cuda_for_profile(profile_device)
        self._shell_profile["get_outputs_total_ms"] += (time.perf_counter() - get_outputs_start) * 1000.0
        color_only_stage2 = self.training and self.config.stage2_color_only_loss
        enable_stage2_pred_normal = color_only_stage2 and self.config.stage2_enable_pred_normal_loss
        if (
            self.training
            and self._finite_difference_losses_enabled()
            and self.config.pred_normal_loss_mult > 0.0
            and (not color_only_stage2 or enable_stage2_pred_normal)
        ):
            field_outputs = outputs["field_outputs"]        
            outputs["rendered_pred_normal_loss"] = self._pred_normal_residual(
                field_outputs[FieldHeadNames.NORMAL],
                field_outputs[FieldHeadNames.PRED_NORMALS],
                outputs.get("weights")
                if torch.is_tensor(outputs.get("weights"))
                else outputs.get("packed_sample_weights"),
            )

        # kernel size smoothness loss
        enable_stage2_kernel_smoothness = color_only_stage2 and self.config.stage2_enable_kernel_smoothness_loss
        if self.training and self.config.kernel_size_smoothness_loss_mult > 0.0 and (
            not color_only_stage2 or enable_stage2_kernel_smoothness
        ):
            field_outputs = outputs["field_outputs"]
            inv_s_deviated = field_outputs["inv_s_deviated"]
            inv_s = field_outputs["inv_s"]
            outputs["kernel_size_smoothness_loss"] = self._kernel_size_smoothness_residual(
                inv_s_deviated=inv_s_deviated,
                inv_s=inv_s,
            )

        return outputs

    @staticmethod
    def _resolve_shell_debug_dir() -> Path:
        """Resolve shell debug output directory under current run log dir when possible."""
        debug_root = Path.cwd()
        for event_writer in writer.EVENT_WRITERS:
            tb_writer = getattr(event_writer, "tb_writer", None)
            if tb_writer is None:
                continue
            log_dir = getattr(tb_writer, "log_dir", None)
            if log_dir:
                debug_root = Path(log_dir)
                break
        debug_dir = debug_root / "shell_debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        return debug_dir

    @staticmethod
    def _save_map_png(path: Path, image01: np.ndarray) -> None:
        image01 = np.clip(image01, 0.0, 1.0)
        image8 = (image01 * 255.0).astype(np.uint8)
        Image.fromarray(image8, mode="L").save(path)

    @staticmethod
    def _save_count_heatmap_png(path: Path, count_map: np.ndarray) -> None:
        """Save valid sample count map as a jet-like heatmap.

        Visualization rule:
        - count == 0: white outlier marker
        - count >= 1: visible base color, then increases along jet colormap
        """
        count = np.asarray(count_map, dtype=np.float32)
        valid = count > 0.0
        if not np.any(valid):
            outlier = np.full((*count.shape, 3), 255, dtype=np.uint8)
            Image.fromarray(outlier, mode="RGB").save(path)
            return

        max_count = float(np.max(count))
        if max_count <= 1.0:
            level = np.zeros_like(count, dtype=np.float32)
        else:
            level = np.clip((count - 1.0) / (max_count - 1.0), 0.0, 1.0)

        # Give 1-sample pixels a clear base color instead of near-black.
        base_level = 0.25
        level = np.where(valid, base_level + (1.0 - base_level) * level, 0.0).astype(np.float32)

        # Jet-like RGB mapping in [0,1].
        r = np.clip(1.5 - np.abs(4.0 * level - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * level - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * level - 1.0), 0.0, 1.0)
        rgb = np.stack([r, g, b], axis=-1)
        outlier = np.ones_like(rgb)
        rgb = np.where(valid[..., None], rgb, outlier)
        rgb8 = (rgb * 255.0).astype(np.uint8)
        Image.fromarray(rgb8, mode="RGB").save(path)

    @staticmethod
    def _normalize_map01_torch(value_map: torch.Tensor) -> torch.Tensor:
        value_min = torch.amin(value_map)
        value_max = torch.amax(value_map)
        denom = torch.clamp(value_max - value_min, min=1e-6)
        return (value_map - value_min) / denom

    @staticmethod
    def _sanitize_eval_image_tensor(
        value: torch.Tensor,
        *,
        clamp_min: Optional[float] = None,
        clamp_max: Optional[float] = None,
        nan: float = 0.0,
        posinf: Optional[float] = None,
        neginf: Optional[float] = None,
    ) -> torch.Tensor:
        """Keep eval-only visualization tensors finite so image logging cannot crash."""
        if posinf is None:
            posinf = nan
        if neginf is None:
            neginf = nan
        value = torch.nan_to_num(value, nan=nan, posinf=posinf, neginf=neginf)
        if clamp_min is not None or clamp_max is not None:
            min_value = clamp_min if clamp_min is not None else float("-inf")
            max_value = clamp_max if clamp_max is not None else float("inf")
            value = torch.clamp(value, min=min_value, max=max_value)
        return value

    @staticmethod
    def _jet_colormap_torch(level01: torch.Tensor) -> torch.Tensor:
        """Map [0,1] scalar map to a jet-like RGB image."""
        level01 = torch.clamp(level01, 0.0, 1.0)
        r = torch.clamp(1.5 - torch.abs(4.0 * level01 - 3.0), 0.0, 1.0)
        g = torch.clamp(1.5 - torch.abs(4.0 * level01 - 2.0), 0.0, 1.0)
        b = torch.clamp(1.5 - torch.abs(4.0 * level01 - 1.0), 0.0, 1.0)
        return torch.stack([r, g, b], dim=-1)

    @staticmethod
    def _quantile_float(values: torch.Tensor, q: float) -> float:
        flat = values.reshape(-1).to(torch.float32)
        return float(torch.quantile(flat, q).item())

    def _compute_surface_debug_dense(
        self,
        field_outputs: Dict[str, torch.Tensor],
        weights: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        sdf = field_outputs[FieldHeadNames.SDF][..., 0]
        inv_s = field_outputs["inv_s"][..., 0]
        alpha = field_outputs[FieldHeadNames.ALPHA][..., 0]
        kernel_size = 1.0 / (inv_s + 1e-6)

        abs_sdf = torch.abs(sdf)
        min_abs_sdf, min_abs_idx = torch.min(abs_sdf, dim=1, keepdim=True)
        surface_kernel_size = torch.gather(kernel_size, dim=1, index=min_abs_idx)
        surface_alpha = torch.gather(alpha, dim=1, index=min_abs_idx)

        max_alpha, max_alpha_idx = torch.max(alpha, dim=1, keepdim=True)
        max_alpha_kernel_size = torch.gather(kernel_size, dim=1, index=max_alpha_idx)

        sample_weights = weights[..., 0]
        accum = torch.clamp(sample_weights.sum(dim=1, keepdim=True), min=1e-6)
        weighted_kernel_size = (sample_weights * kernel_size).sum(dim=1, keepdim=True) / accum

        return {
            "min_abs_sdf": min_abs_sdf,
            "surface_kernel_size": surface_kernel_size,
            "surface_alpha": surface_alpha,
            "max_alpha": max_alpha,
            "max_alpha_kernel_size": max_alpha_kernel_size,
            "weighted_kernel_size": weighted_kernel_size,
        }

    def _compute_surface_debug_packed(
        self,
        field_outputs: Dict[str, torch.Tensor],
        weights: torch.Tensor,
        ray_indices: torch.Tensor,
        num_rays: int,
    ) -> Dict[str, torch.Tensor]:
        device = weights.device
        sdf = field_outputs[FieldHeadNames.SDF][..., 0]
        inv_s = field_outputs["inv_s"][..., 0]
        alpha = field_outputs[FieldHeadNames.ALPHA][..., 0]
        kernel_size = 1.0 / (inv_s + 1e-6)
        abs_sdf = torch.abs(sdf)

        sample_count = torch.zeros((num_rays,), dtype=torch.float32, device=device)
        sample_count.scatter_add_(0, ray_indices, torch.ones_like(abs_sdf, dtype=torch.float32))
        has_sample = sample_count > 0.0

        min_abs_sdf = torch.full((num_rays,), float("inf"), dtype=abs_sdf.dtype, device=device)
        min_abs_sdf.scatter_reduce_(0, ray_indices, abs_sdf, reduce="amin", include_self=True)
        min_abs_sdf = torch.where(has_sample, min_abs_sdf, torch.zeros_like(min_abs_sdf))

        max_alpha = torch.zeros((num_rays,), dtype=alpha.dtype, device=device)
        max_alpha.scatter_reduce_(0, ray_indices, alpha, reduce="amax", include_self=True)

        tol = 1e-6
        min_mask = (abs_sdf - min_abs_sdf[ray_indices]).abs() <= tol
        max_mask = (alpha - max_alpha[ray_indices]).abs() <= tol

        min_mask_f = min_mask.to(kernel_size.dtype)
        max_mask_f = max_mask.to(kernel_size.dtype)

        surface_kernel_sum = torch.zeros((num_rays,), dtype=kernel_size.dtype, device=device)
        surface_kernel_sum.scatter_add_(0, ray_indices, kernel_size * min_mask_f)
        surface_kernel_count = torch.zeros((num_rays,), dtype=kernel_size.dtype, device=device)
        surface_kernel_count.scatter_add_(0, ray_indices, min_mask_f)
        surface_kernel_size = surface_kernel_sum / torch.clamp(surface_kernel_count, min=1.0)

        surface_alpha_sum = torch.zeros((num_rays,), dtype=alpha.dtype, device=device)
        surface_alpha_sum.scatter_add_(0, ray_indices, alpha * min_mask_f)
        surface_alpha = surface_alpha_sum / torch.clamp(surface_kernel_count, min=1.0)

        max_alpha_kernel_sum = torch.zeros((num_rays,), dtype=kernel_size.dtype, device=device)
        max_alpha_kernel_sum.scatter_add_(0, ray_indices, kernel_size * max_mask_f)
        max_alpha_kernel_count = torch.zeros((num_rays,), dtype=kernel_size.dtype, device=device)
        max_alpha_kernel_count.scatter_add_(0, ray_indices, max_mask_f)
        max_alpha_kernel_size = max_alpha_kernel_sum / torch.clamp(max_alpha_kernel_count, min=1.0)

        sample_weights = weights[..., 0]
        accum = torch.zeros((num_rays,), dtype=sample_weights.dtype, device=device)
        accum.scatter_add_(0, ray_indices, sample_weights)
        weighted_kernel_sum = torch.zeros((num_rays,), dtype=kernel_size.dtype, device=device)
        weighted_kernel_sum.scatter_add_(0, ray_indices, sample_weights * kernel_size)
        weighted_kernel_size = weighted_kernel_sum / torch.clamp(accum, min=1e-6)

        return {
            "min_abs_sdf": min_abs_sdf.unsqueeze(-1),
            "surface_kernel_size": surface_kernel_size.unsqueeze(-1),
            "surface_alpha": surface_alpha.unsqueeze(-1),
            "max_alpha": max_alpha.unsqueeze(-1),
            "max_alpha_kernel_size": max_alpha_kernel_size.unsqueeze(-1),
            "weighted_kernel_size": weighted_kernel_size.unsqueeze(-1),
        }

    def get_image_metrics_and_images(self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]):
        outputs = dict(outputs)
        if "accumulation" in outputs:
            outputs["accumulation"] = self._sanitize_eval_image_tensor(
                outputs["accumulation"],
                clamp_min=0.0,
                clamp_max=1.0,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
        if "depth" in outputs:
            outputs["depth"] = self._sanitize_eval_image_tensor(
                outputs["depth"],
                clamp_min=0.0,
                nan=0.0,
            )
        if "rgb" in outputs:
            outputs["rgb"] = self._sanitize_eval_image_tensor(
                outputs["rgb"],
                clamp_min=0.0,
                clamp_max=1.0,
                nan=0.0,
                posinf=1.0,
                neginf=0.0,
            )
        if "normal" in outputs:
            outputs["normal"] = self._sanitize_eval_image_tensor(outputs["normal"], nan=0.0)
        if "pred_normals" in outputs:
            outputs["pred_normals"] = self._sanitize_eval_image_tensor(outputs["pred_normals"], nan=0.0)

        metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
        images_dict.pop("accumulation", None)

        normal = outputs.get("normal")
        if normal is not None:
            images_dict["normal"] = ((normal + 1.0) * 0.5).clamp(0.0, 1.0)

        pred_normals = outputs.get("pred_normals")
        if pred_normals is not None:
            images_dict["pred_normal"] = ((pred_normals + 1.0) * 0.5).clamp(0.0, 1.0)
            if normal is not None:
                normal_error = torch.linalg.norm(pred_normals - normal, dim=-1)
                normal_error_01 = self._normalize_map01_torch(normal_error)
                images_dict["normal_error"] = normal_error_01.unsqueeze(-1).repeat(1, 1, 3)
                metrics_dict["pred_normal_error_mean"] = float(normal_error.mean().item())

        if self.config.eval_visualize_kernel_size:
            weighted_kernel = outputs.get("weighted_kernel_size", outputs.get("kernel_size"))
            if weighted_kernel is not None:
                weighted_kernel_map = weighted_kernel[..., 0]
                weighted_kernel_01 = self._normalize_map01_torch(weighted_kernel_map)
                images_dict["weighted_kernel_size"] = self._jet_colormap_torch(weighted_kernel_01)
                metrics_dict["weighted_kernel_size_mean"] = float(weighted_kernel_map.mean().item())

            if self.config.eval_visualize_surface_proxies and "surface_kernel_size" in outputs:
                surface_kernel = outputs["surface_kernel_size"][..., 0]
                surface_kernel_01 = self._normalize_map01_torch(surface_kernel)
                images_dict["surface_kernel_size"] = self._jet_colormap_torch(surface_kernel_01)
                metrics_dict["surface_kernel_size_mean"] = float(surface_kernel.mean().item())
                metrics_dict["surface_kernel_size_p90"] = self._quantile_float(surface_kernel, 0.9)

                if self.config.eval_visualize_kernel_smoothness:
                    diff_x = torch.zeros_like(surface_kernel)
                    diff_y = torch.zeros_like(surface_kernel)
                    diff_x[:, 1:] = torch.abs(surface_kernel[:, 1:] - surface_kernel[:, :-1])
                    diff_y[1:, :] = torch.abs(surface_kernel[1:, :] - surface_kernel[:-1, :])
                    kernel_smoothness = diff_x + diff_y
                    kernel_smoothness_01 = self._normalize_map01_torch(kernel_smoothness)
                    images_dict["kernel_smoothness"] = self._jet_colormap_torch(kernel_smoothness_01)
                    metrics_dict["kernel_smoothness_mean"] = float(kernel_smoothness.mean().item())

        if self.config.eval_visualize_surface_proxies and "min_abs_sdf" in outputs:
            min_abs_sdf = outputs["min_abs_sdf"][..., 0]
            min_abs_sdf_01 = self._normalize_map01_torch(min_abs_sdf)
            images_dict["min_abs_sdf"] = min_abs_sdf_01.unsqueeze(-1).repeat(1, 1, 3)
            images_dict["inv_min_abs_sdf"] = self._jet_colormap_torch(1.0 - min_abs_sdf_01)
            metrics_dict["min_abs_sdf_mean"] = float(min_abs_sdf.mean().item())
            metrics_dict["min_abs_sdf_p50"] = self._quantile_float(min_abs_sdf, 0.5)
            metrics_dict["min_abs_sdf_p90"] = self._quantile_float(min_abs_sdf, 0.9)
            surface_alpha = outputs.get("surface_alpha")
            if surface_alpha is not None:
                metrics_dict["surface_alpha_mean"] = float(surface_alpha[..., 0].mean().item())
            max_alpha = outputs.get("max_alpha")
            if max_alpha is not None:
                metrics_dict["max_alpha_mean"] = float(max_alpha[..., 0].mean().item())

        shell_hit = outputs.get("shell_hit")
        shell_valid_count = outputs.get("shell_valid_count")
        if shell_hit is None or shell_valid_count is None:
            return metrics_dict, images_dict

        if self.config.eval_visualize_shell_maps:
            shell_count_map = shell_valid_count[..., 0].to(torch.float32)
            shell_count_01 = self._normalize_map01_torch(shell_count_map)
            shell_count_rgb = self._jet_colormap_torch(shell_count_01)
            zero_sample_mask = shell_count_map <= 0.0
            outlier_color = torch.ones((3,), device=shell_count_rgb.device, dtype=shell_count_rgb.dtype)
            shell_count_rgb = torch.where(zero_sample_mask[..., None], outlier_color, shell_count_rgb)
            images_dict["shell_valid_count"] = shell_count_rgb
            metrics_dict["shell_valid_count_mean"] = float(shell_count_map.mean().item())
            metrics_dict["shell_zero_sample_frac"] = float(zero_sample_mask.to(torch.float32).mean().item())

        if not self.config.eval_export_shell_maps_to_disk:
            return metrics_dict, images_dict

        debug_dir = self._resolve_shell_debug_dir()
        shell_valid_count_np = shell_valid_count.detach().cpu().numpy()[..., 0].astype(np.float32)

        np.save(debug_dir / "shell_valid_count.npy", shell_valid_count_np)
        self._save_count_heatmap_png(debug_dir / "shell_valid_count.png", shell_valid_count_np)
        if self.config.eval_visualize_surface_proxies and "min_abs_sdf" in outputs:
            min_abs_sdf = outputs["min_abs_sdf"][..., 0]
            min_abs_sdf_np = min_abs_sdf.detach().cpu().numpy().astype(np.float32)
            np.save(debug_dir / "min_abs_sdf.npy", min_abs_sdf_np)
            min_abs_sdf_01_np = self._normalize_map01_torch(min_abs_sdf).detach().cpu().numpy().astype(np.float32)
            self._save_map_png(debug_dir / "min_abs_sdf.png", min_abs_sdf_01_np)
        if self.config.eval_visualize_surface_proxies and "surface_kernel_size" in outputs:
            surface_kernel_np = outputs["surface_kernel_size"][..., 0].detach().cpu().numpy().astype(np.float32)
            np.save(debug_dir / "surface_kernel_size.npy", surface_kernel_np)

        if self._optix_tracer is not None and self._optix_eval_cameras is not None and "image_idx" in batch:
            image_idx_raw = batch["image_idx"]
            image_idx = int(image_idx_raw.item()) if hasattr(image_idx_raw, "item") else int(image_idx_raw)
            optix_path = debug_dir / f"optix_depth_{image_idx:04d}.png"
            self._optix_tracer.render_vis(
                cameras=self._optix_eval_cameras,
                camera_index=image_idx,
                debug_path=str(optix_path),
                validate_conversion=self.config.optix_debug_validate_camera_conversion,
            )

        return metrics_dict, images_dict

    def set_optix_eval_cameras(self, cameras: Cameras) -> None:
        """Bind eval cameras so OptiX debug rendering can reuse Nerfstudio camera definitions."""
        self._optix_eval_cameras = cameras

    def get_export_contract_metadata(self) -> Dict[str, object]:
        """Return field export metadata when backend supports it."""
        if hasattr(self.field, "get_tcnn_compat_export_metadata"):
            return cast(Dict[str, object], self.field.get_tcnn_compat_export_metadata())
        if hasattr(self.field, "get_tcnn_export_metadata"):
            return cast(Dict[str, object], self.field.get_tcnn_export_metadata())
        raise AttributeError("Active field backend does not expose export metadata.")


    def get_loss_dict(self, outputs, batch, metrics_dict=None):
        fd_losses_enabled = self._finite_difference_losses_enabled()
        if self.training and outputs.get("packed_shell_contract", False):
            image = batch["image"].to(self.device)
            loss_dict = {"rgb_loss": self.rgb_loss(image, outputs["rgb"])}

            if fd_losses_enabled and (not self.config.stage2_color_only_loss) and self.config.eikonal_loss_mult > 0.0:
                grad_theta = outputs.get("eik_grad")
                if grad_theta is not None and torch.is_tensor(grad_theta) and grad_theta.numel() > 0:
                    loss_dict["eikonal_loss"] = (
                        (grad_theta.norm(2, dim=-1) - 1) ** 2
                    ).mean() * self.config.eikonal_loss_mult

            if fd_losses_enabled and outputs.get("rendered_pred_normal_loss") is not None:
                loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
            if outputs.get("kernel_size_smoothness_loss") is not None:
                loss_dict["kernel_size_smoothness_loss"] = self.config.kernel_size_smoothness_loss_mult * torch.mean(
                    outputs["kernel_size_smoothness_loss"]
                )

            if self.config.stage2_color_only_loss:
                stage2_loss_dict = {"rgb_loss": loss_dict["rgb_loss"]}
                if self.config.stage2_enable_pred_normal_loss and loss_dict.get("pred_normal_loss") is not None:
                    stage2_loss_dict["pred_normal_loss"] = loss_dict["pred_normal_loss"]
                if self.config.stage2_enable_kernel_smoothness_loss and loss_dict.get("kernel_size_smoothness_loss") is not None:
                    stage2_loss_dict["kernel_size_smoothness_loss"] = loss_dict["kernel_size_smoothness_loss"]
                return stage2_loss_dict
            return loss_dict

        loss_dict = super().get_loss_dict(outputs, batch, metrics_dict)
        if self.training and not fd_losses_enabled:
            loss_dict.pop("eikonal_loss", None)
            loss_dict.pop("mono_normal_loss", None)
        if self.training and self.config.stage2_color_only_loss:
            stage2_loss_dict = {"rgb_loss": loss_dict["rgb_loss"]}
            if (
                fd_losses_enabled
                and self.config.stage2_enable_pred_normal_loss
                and outputs.get("rendered_pred_normal_loss") is not None
            ):
                stage2_loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                    outputs["rendered_pred_normal_loss"]
                )
            if self.config.stage2_enable_kernel_smoothness_loss and outputs.get("kernel_size_smoothness_loss") is not None:
                stage2_loss_dict["kernel_size_smoothness_loss"] = self.config.kernel_size_smoothness_loss_mult * torch.mean(
                    outputs["kernel_size_smoothness_loss"]
                )
            return stage2_loss_dict

        # if self.training:
        #     loss_dict["interlevel_loss"] = self.config.interlevel_loss_mult * interlevel_loss_zip(
        #         outputs["weights_list"], outputs["ray_samples_list"]
        #     )

        # normal consistency loss
        if self.training and fd_losses_enabled and outputs.get("rendered_pred_normal_loss") is not None:
            loss_dict["pred_normal_loss"] = self.config.pred_normal_loss_mult * torch.mean(
                outputs["rendered_pred_normal_loss"]
            )
        if self.training and outputs.get("kernel_size_smoothness_loss") is not None:
            loss_dict["kernel_size_smoothness_loss"] = self.config.kernel_size_smoothness_loss_mult * torch.mean(
                outputs["kernel_size_smoothness_loss"]
            )
        
        # curvature loss
        if self.training and fd_losses_enabled and self.config.curvature_loss_multi > 0.0:
            delta = self.field.numerical_gradients_delta
            centered_sdf = outputs["field_outputs"][FieldHeadNames.SDF]
            sourounding_sdf = outputs["field_outputs"]["sampled_sdf"]

            sourounding_sdf = sourounding_sdf.reshape(centered_sdf.shape[:2] + (3, 2))

            # (a - b)/d - (b -c)/d = (a + c - 2b)/d
            # ((a - b)/d - (b -c)/d)/d = (a + c - 2b)/(d*d)
            curvature = (sourounding_sdf.sum(dim=-1) - 2 * centered_sdf) / (delta * delta)
            loss_dict["curvature_loss"] = (
                torch.abs(curvature).mean() * self.config.curvature_loss_multi * self.curvature_loss_multi_factor
            )

        return loss_dict

    def get_metrics_dict(self, outputs, batch) -> Dict:
        # metrics_dict = super().get_metrics_dict(outputs, batch)
        # by pass single learnbale inv_s in NeusModel
        # BTW the s_val and inv_s in NeuSModel seems reversed
        # in our model, network always output inv_s
        metrics_dict = super(NeuSModel, self).get_metrics_dict(outputs, batch)

        if self.training:
            # training statics
            metrics_dict["activated_encoding"] = self.field.hash_encoding_mask.mean().item()
            metrics_dict["active_hash_levels"] = float(self._active_hash_levels)
            metrics_dict["active_hash_max_res"] = float(self._active_hash_max_res)
            metrics_dict["numerical_gradients_delta"] = self.field.numerical_gradients_delta
            metrics_dict["cos_anneal_ratio"] = float(getattr(self.field, "_cos_anneal_ratio", 1.0))
            alpha_mode = getattr(self.field.config, "alpha_mode", "neus_gradient")
            metrics_dict["alpha_mode_laplace_density"] = 1.0 if alpha_mode == "laplace_density" else 0.0
            metrics_dict["alpha_mode_pred_gradient"] = 1.0 if alpha_mode == "pred_gradient" else 0.0
            metrics_dict["fd_schedule_active"] = 1.0 if self._fd_schedule_active else 0.0
            metrics_dict["fd_losses_enabled"] = 1.0 if self._finite_difference_losses_enabled() else 0.0
            metrics_dict["curvature_loss_multi"] = self.curvature_loss_multi_factor * self.config.curvature_loss_multi
            self._add_stage1_numeric_metrics(metrics_dict, outputs)
            use_shell_sampler = self.config.shell_inference_mode or self.config.shell_finetune_mode
            if use_shell_sampler:
                shell_profile = self.get_shell_profile_summary()
                chunks = int(shell_profile.get("chunks", 0))
                if chunks > 0:
                    inv_chunks = 1.0 / float(chunks)
                    metrics_dict["shell_train_model_ms"] = float(shell_profile.get("get_outputs_total_ms", 0.0)) * inv_chunks
                    metrics_dict["shell_train_sample_fwd_ms"] = float(shell_profile.get("sample_forward_total_ms", 0.0)) * inv_chunks
                    metrics_dict["shell_train_sampler_ms"] = float(shell_profile.get("sampler_total_ms", 0.0)) * inv_chunks
                    metrics_dict["shell_train_trace_ms"] = float(shell_profile.get("trace_and_pack_ms", 0.0)) * inv_chunks
                    metrics_dict["shell_train_trace_build_ms"] = float(shell_profile.get("ray_samples_build_ms", 0.0)) * inv_chunks
                    metrics_dict["shell_train_field_ms"] = float(shell_profile.get("field_eval_ms", 0.0)) * inv_chunks
                    metrics_dict["shell_train_field_geonet_ms"] = float(shell_profile.get("field_geonet_ms", 0.0)) * inv_chunks
                    metrics_dict["shell_train_field_grad_ms"] = float(shell_profile.get("field_gradient_ms", 0.0)) * inv_chunks
                    metrics_dict["shell_train_field_color_ms"] = float(shell_profile.get("field_color_ms", 0.0)) * inv_chunks
                    metrics_dict["shell_train_alpha_weights_ms"] = float(shell_profile.get("alpha_weights_ms", 0.0)) * inv_chunks
                    metrics_dict["shell_train_utilization"] = float(shell_profile.get("utilization", 0.0))
                    metrics_dict["shell_train_budget_per_ray"] = float(shell_profile.get("budget_per_ray", 0.0))
                    metrics_dict["shell_train_fg_mean_samples"] = float(shell_profile.get("mean_valid_per_hit_ray", 0.0))
                    metrics_dict["shell_train_single_hit_frac"] = float(shell_profile.get("single_sample_hit_frac", 0.0))
                    metrics_dict["shell_train_single_grad_norm_mean"] = float(
                        shell_profile.get("single_sample_grad_norm_mean", 0.0)
                    )
                self.reset_shell_profile()

        return metrics_dict

    # def get_image_metrics_and_images(
    #     self, outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]
    # ) -> Tuple[Dict[str, float], Dict[str, torch.Tensor]]:
    #     metrics_dict, images_dict = super().get_image_metrics_and_images(outputs, batch)
    #     for i in range(self.config.num_proposal_iterations):
    #         key = f"prop_depth_{i}"
    #         prop_depth_i = colormaps.apply_depth_colormap(
    #             outputs[key],
    #             accumulation=outputs["accumulation"],
    #         )
    #         images_dict[key] = prop_depth_i

    #     return metrics_dict, images_dict
