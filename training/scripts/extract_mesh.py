#!/usr/bin/env python
# Modified for Adaptive Shells training and rendering; see the repository NOTICE.
"""
eval.py
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
import tyro
from rich.console import Console

from nerfstudio.model_components.ray_samplers import save_points
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.marching_cubes import (
    get_surface_occupancy,
    get_surface_sliding,
    get_surface_sliding_with_contraction,
    extract_mesh_shell,
    extract_mesh_simple,
)

CONSOLE = Console(width=120)

# speedup for when input size to model doesn't change (much)
torch.backends.cudnn.benchmark = True  # type: ignore


@dataclass
class ExtractMesh:
    """Load a checkpoint, run marching cubes, extract mesh, and save it to a ply file."""

    # Path to config YAML file.
    load_config: Path
    # Marching cube resolution.
    resolution: int = 1024
    # Name of the output file.
    output_path: Path = Path("output.ply")
    # Whether to simplify the mesh.
    simplify_mesh: bool = False
    # extract the mesh using occupancy field (unisurf) or SDF, default sdf
    is_occupancy: bool = False
    """Minimum of the bounding box. Defaults to the checkpoint/dataparser scene box."""
    bounding_box_min: Optional[Tuple[float, float, float]] = None
    """Maximum of the bounding box. Defaults to the checkpoint/dataparser scene box."""
    bounding_box_max: Optional[Tuple[float, float, float]] = None
    """marching cube threshold"""
    marching_cube_threshold: float = 0.0
    """create visibility mask"""
    create_visibility_mask: bool = False
    """save visibility grid"""
    save_visibility_grid: bool = False
    """visibility grid resolution"""
    visibility_grid_resolution: int = 512
    """threshold for considering a points is valid when splat to visibility grid"""
    valid_points_thres: float = 0.005
    """sub samples factor of images when creating visibility grid"""
    sub_sample_factor: int = 8
    """torch precision"""
    torch_precision: Literal["highest", "high"] = "high"
    """use shell extraction"""
    use_shell: bool = False
    """isosurface level used for extracting the base mesh"""
    shell_level: float = 0.0
    """print shell extraction statistics"""
    shell_debug_stats: bool = False
    """shell extraction iterations"""
    shell_n_iter: int = 50
    """shell extraction time step"""
    shell_time_step: float = 0.01
    """optional outer dilation time step; defaults to shell_time_step"""
    shell_time_step_d: Optional[float] = None
    """optional inner erosion time step; defaults to shell_time_step"""
    shell_time_step_e: Optional[float] = None
    """minimum alpha for dilation velocity"""
    shell_sigma_min: float = 0.01
    """outer dilation velocity coefficient"""
    shell_beta_d: float = 1.0
    """inner erosion velocity coefficient"""
    shell_beta_e: float = 0.01
    """maximum erosion velocity"""
    shell_v_max: float = 0.025
    """erosion velocity mode; excess_inverse_alpha protects alpha≈1 sharp surfaces"""
    shell_erosion_velocity_mode: Literal[
        "inverse_alpha", "excess_inverse_alpha", "baseline_excess_inverse_alpha"
    ] = "baseline_excess_inverse_alpha"
    """erosion velocity mode; baseline_excess_inverse_alpha gives sharp surfaces a small shrink floor"""
    shell_erosion_excess_floor: float = 0.2
    """dimensionless baseline term for baseline_excess_inverse_alpha erosion velocity"""
    shell_smooth_velocity_d_iters: int = 1
    """number of 3D smoothing passes applied to erosion velocity before evolution"""
    shell_smooth_velocity_e_iters: int = 0
    """clamp dilation curvature term to this ratio of the motion term; None disables"""
    shell_curvature_motion_ratio_d: Optional[float] = None
    """clamp erosion curvature term to this ratio of the motion term; None disables"""
    shell_curvature_motion_ratio_e: Optional[float] = None
    """outer curvature regularization weight"""
    shell_weight_curv_d: float = 0.0001
    """outer evolution window width"""
    shell_window_width_d: float = 0.1
    """inner curvature regularization weight"""
    shell_weight_curv_e: float = 0.0
    """inner evolution window width"""
    shell_window_width_e: float = 0.04
    """kernel size transform mode used for shell extraction"""
    shell_kernel_size_transform_mode: Optional[Literal["exp", "clip", "none"]] = None
    """exponential factor for shell kernel transform"""
    shell_kernel_size_transform_exp_factor: Optional[float] = None
    """exponential factor for shell kernel transform"""
    shell_recompute_grad_each_iter: bool = False
    """recompute level-set gradients every evolution step, matching the paper pseudocode more closely"""
    eval_batch_points: int = 200000
    """Maximum number of regular-grid points evaluated per field forward chunk."""
    eval_slab_depth: int = 8
    """Number of z slices evaluated together when rasterizing the extraction grid."""

    def _resolve_bounding_box(self, pipeline) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
        """Use explicit CLI bounds when supplied; otherwise follow the training scene box."""
        if (self.bounding_box_min is None) != (self.bounding_box_max is None):
            raise ValueError("`bounding_box_min` and `bounding_box_max` must be provided together.")

        if self.bounding_box_min is not None and self.bounding_box_max is not None:
            return self.bounding_box_min, self.bounding_box_max

        scene_box = getattr(getattr(pipeline, "model", None), "scene_box", None)
        aabb = getattr(scene_box, "aabb", None)
        if aabb is None:
            CONSOLE.print("[yellow]No scene_box.aabb found on pipeline.model; falling back to [-1, 1]^3.[/yellow]")
            return (-1.0, -1.0, -1.0), (1.0, 1.0, 1.0)

        aabb_tensor = torch.as_tensor(aabb, dtype=torch.float32).detach().cpu()
        if tuple(aabb_tensor.shape) != (2, 3):
            raise ValueError(f"Expected scene_box.aabb shape (2, 3), got {tuple(aabb_tensor.shape)}.")

        bbox_min = tuple(float(x) for x in aabb_tensor[0].tolist())
        bbox_max = tuple(float(x) for x in aabb_tensor[1].tolist())
        return bbox_min, bbox_max

    def main(self) -> None:
        """Main function."""
        torch.set_float32_matmul_precision(self.torch_precision)
        assert str(self.output_path)[-4:] == ".ply"
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        _, pipeline, _ = eval_setup(self.load_config)
        field = pipeline.model.field
        bounding_box_min, bounding_box_max = self._resolve_bounding_box(pipeline)

        CONSOLE.print("Extract mesh with marching cubes and may take a while")
        CONSOLE.print(f"Extraction bounding box: min={bounding_box_min}, max={bounding_box_max}")

        def forward_geonetwork_with_distortion(points: torch.Tensor) -> torch.Tensor:
            if getattr(field, "spatial_distortion", None) is not None:
                points = field.spatial_distortion(points)
            return field.forward_geonetwork(points)

        if self.create_visibility_mask:
            assert self.resolution % 512 == 0

            coarse_mask = pipeline.get_visibility_mask(
                self.visibility_grid_resolution, self.valid_points_thres, self.sub_sample_factor
            )

            def inv_contract(x):
                mag = torch.linalg.norm(x, ord=pipeline.model.scene_contraction.order, dim=-1)
                mask = mag >= 1
                x_new = x.clone()
                x_new[mask] = (1 / (2 - mag[mask][..., None])) * (x[mask] / mag[mask][..., None])
                return x_new

            if self.save_visibility_grid:
                offset = torch.linspace(-2.0, 2.0, 512)
                x, y, z = torch.meshgrid(offset, offset, offset, indexing="ij")
                offset_cube = torch.stack([x, y, z], dim=-1).reshape(-1, 3).to(coarse_mask.device)
                points = offset_cube[coarse_mask.reshape(-1) > 0]
                points = inv_contract(points)
                save_points("mask.ply", points.cpu().numpy())
                torch.save(coarse_mask, "coarse_mask.pt")

            get_surface_sliding_with_contraction(
                sdf=lambda x: (
                    forward_geonetwork_with_distortion(x)[:, 0] - self.marching_cube_threshold
                ).contiguous(),
                resolution=self.resolution,
                bounding_box_min=bounding_box_min,
                bounding_box_max=bounding_box_max,
                coarse_mask=coarse_mask,
                output_path=self.output_path,
                simplify_mesh=self.simplify_mesh,
                inv_contraction=inv_contract,
            )
            return

        shell_mode = self.shell_kernel_size_transform_mode
        if shell_mode is None:
            shell_mode = getattr(field.config, "kernel_size_transform_mode", "exp")
        shell_exp_factor = self.shell_kernel_size_transform_exp_factor
        if shell_exp_factor is None:
            shell_exp_factor = float(getattr(field.config, "kernel_size_transform_exp_factor", 10.0))

        if self.is_occupancy:
            # for unisurf
            get_surface_occupancy(
                occupancy_fn=lambda x: torch.sigmoid(
                    10 * forward_geonetwork_with_distortion(x)[:, 0].contiguous()
                ),
                resolution=self.resolution,
                bounding_box_min=bounding_box_min,
                bounding_box_max=bounding_box_max,
                level=0.5,
                device=pipeline.model.device,
                output_path=self.output_path,
            )
        elif self.use_shell:
            extract_mesh_shell(
                field_fn=lambda x: forward_geonetwork_with_distortion(x)[:, :2].contiguous(),  # sdf and raw inv_s head
                resolution=self.resolution,
                level=self.shell_level,
                bounding_box_min=bounding_box_min,
                bounding_box_max=bounding_box_max,
                output_path=self.output_path,
                n_iter=self.shell_n_iter,
                time_step=self.shell_time_step,
                time_step_d=self.shell_time_step_d,
                time_step_e=self.shell_time_step_e,
                sigma_min=self.shell_sigma_min,
                beta_d=self.shell_beta_d,
                beta_e=self.shell_beta_e,
                v_max=self.shell_v_max,
                erosion_velocity_mode=self.shell_erosion_velocity_mode,
                erosion_excess_floor=self.shell_erosion_excess_floor,
                smooth_velocity_d_iters=self.shell_smooth_velocity_d_iters,
                smooth_velocity_e_iters=self.shell_smooth_velocity_e_iters,
                curvature_motion_ratio_d=self.shell_curvature_motion_ratio_d,
                curvature_motion_ratio_e=self.shell_curvature_motion_ratio_e,
                weight_curv_d=self.shell_weight_curv_d,
                window_width_d=self.shell_window_width_d,
                weight_curv_e=self.shell_weight_curv_e,
                window_width_e=self.shell_window_width_e,
                kernel_size_transform_mode=shell_mode,
                kernel_size_transform_exp_factor=shell_exp_factor,
                recompute_grad_each_iter=self.shell_recompute_grad_each_iter,
                debug_stats=self.shell_debug_stats,
                device=pipeline.model.device,
                slab_depth=self.eval_slab_depth,
                batch_points=self.eval_batch_points,
            )
        else:
            assert self.resolution % 512 == 0
            # for sdf we can multi-scale extraction.
            get_surface_sliding(
                sdf=lambda x: forward_geonetwork_with_distortion(x)[:, 0].contiguous(),
                resolution=self.resolution,
                bounding_box_min=bounding_box_min,
                bounding_box_max=bounding_box_max,
                coarse_mask=pipeline.model.scene_box.coarse_binary_gird,
                output_path=self.output_path,
                simplify_mesh=self.simplify_mesh,
            )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(tyro.conf.FlagConversionOff[ExtractMesh]).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(ExtractMesh)  # noqa
