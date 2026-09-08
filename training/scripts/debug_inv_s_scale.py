from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import torch
import tyro

from nerfstudio.fields.adashell_field import transform_inv_s
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.marching_cubes import get_alpha_from_field


@dataclass
class DebugInvSScale:
    load_config: Path
    resolution: int = 128
    bbox_min: tuple[float, float, float] = (-1.0, -1.0, -1.0)
    bbox_max: tuple[float, float, float] = (1.0, 1.0, 1.0)
    transform_mode: Literal["exp", "clip", "none", "all"] = "all"
    exp_factor: float | None = None

    @staticmethod
    def summarize(name: str, x: torch.Tensor) -> None:
        x = x.detach().float().cpu().reshape(-1)
        q = torch.quantile(x, torch.tensor([0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0]))
        print(
            f"[{name}] "
            f"min={q[0]:.6g} q01={q[1]:.6g} q05={q[2]:.6g} median={q[3]:.6g} "
            f"q95={q[4]:.6g} q99={q[5]:.6g} max={q[6]:.6g} "
            f"mean={x.mean():.6g} std={x.std(unbiased=False):.6g}"
        )

    @staticmethod
    def summarize_masked(name: str, x: torch.Tensor, mask: torch.Tensor) -> None:
        if int(mask.sum().item()) == 0:
            print(f"[{name}] empty")
            return
        DebugInvSScale.summarize(name, x[mask])

    def main(self) -> None:
        _, pipeline, _ = eval_setup(self.load_config)
        field = pipeline.model.field.eval().cuda()
        field_mode = getattr(field.config, "kernel_size_transform_mode", "exp")
        field_exp_factor = float(getattr(field.config, "kernel_size_transform_exp_factor", 10.0))
        exp_factor = field_exp_factor if self.exp_factor is None else float(self.exp_factor)

        def forward_geonetwork_with_distortion(points: torch.Tensor) -> torch.Tensor:
            if getattr(field, "spatial_distortion", None) is not None:
                points = field.spatial_distortion(points)
            return field.forward_geonetwork(points)

        x = torch.linspace(self.bbox_min[0], self.bbox_max[0], self.resolution, device="cuda")
        y = torch.linspace(self.bbox_min[1], self.bbox_max[1], self.resolution, device="cuda")
        z = torch.linspace(self.bbox_min[2], self.bbox_max[2], self.resolution, device="cuda")
        xx, yy, zz = torch.meshgrid(x, y, z, indexing="ij")
        points = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

        raw_second_channel = []
        sdf_values = []
        for pts in torch.split(points, 100000, dim=0):
            h = forward_geonetwork_with_distortion(pts)
            sdf, raw_inv_s, _, _ = field.split_geonet_out(h)
            sdf_values.append(sdf[:, 0].detach())
            raw_second_channel.append(raw_inv_s[:, 0].detach())

        sdf = torch.cat(sdf_values, dim=0)
        raw_second_channel = torch.cat(raw_second_channel, dim=0)

        spacing = (self.bbox_max[0] - self.bbox_min[0]) / (self.resolution - 1)
        half_spacing = torch.tensor(spacing * 0.5, device="cuda")

        print(
            f"[meta] resolution={self.resolution} spacing={spacing:.6g} "
            f"half_spacing={spacing * 0.5:.6g} total_points={points.shape[0]}"
        )
        self.summarize("raw_second_channel", raw_second_channel)
        self.summarize("sdf", sdf)

        if self.transform_mode == "all":
            ordered_modes = [field_mode] + [m for m in ("exp", "clip", "none") if m != field_mode]
        else:
            ordered_modes = [self.transform_mode]

        for mode in ordered_modes:
            if mode == "exp":
                inv_s = transform_inv_s(raw_second_channel, mode=mode, exp_factor=exp_factor)
            else:
                inv_s = transform_inv_s(raw_second_channel, mode=mode)
            kernel_size = 1.0 / (inv_s + 1e-12)
            kernel_over_spacing = kernel_size / spacing
            alpha = get_alpha_from_field(sdf, inv_s, half_spacing)

            self.summarize(f"inv_s_{mode}", inv_s)
            self.summarize(f"kernel_size_{mode}", kernel_size)
            self.summarize(f"kernel_over_spacing_{mode}", kernel_over_spacing)
            self.summarize(f"alpha_{mode}", alpha)
            for k in (1, 2, 4):
                mask = sdf.abs() < (k * spacing)
                self.summarize_masked(f"near_surface_k{k}_inv_s_{mode}", inv_s, mask)
                self.summarize_masked(f"near_surface_k{k}_kernel_{mode}", kernel_size, mask)
                self.summarize_masked(f"near_surface_k{k}_alpha_{mode}", alpha, mask)
            for sigma_min in (0.01, 0.05, 0.1):
                frac = (alpha > sigma_min).float().mean().item()
                print(f"[alpha_{mode}] frac_gt_{sigma_min:.2f}={frac:.6f}")
            for target_res in (128, 256, 512, 1024):
                target_spacing = (self.bbox_max[0] - self.bbox_min[0]) / (target_res - 1)
                ratio = kernel_size / target_spacing
                self.summarize(f"kernel_over_spacing_{mode}_res{target_res}", ratio)
            if mode == "exp":
                safe_alpha = torch.clamp(alpha, min=1e-12)
                erosion_masks = {
                    "global": torch.ones_like(safe_alpha, dtype=torch.bool),
                    "sdf_abs_lt_0.05": sdf.abs() < 0.05,
                    "sdf_abs_lt_0.10": sdf.abs() < 0.10,
                    "near_surface_k2": sdf.abs() < (2 * spacing),
                }
                for beta_e in (5e-5, 1e-4, 2e-4, 5e-4, 1e-3):
                    for mask_name, mask in erosion_masks.items():
                        row = []
                        masked_alpha = safe_alpha[mask]
                        for v_max in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0):
                            speed = torch.minimum(torch.tensor(v_max, device=masked_alpha.device), beta_e / masked_alpha)
                            sat_frac = (speed >= (v_max - 1e-6)).float().mean().item()
                            row.append(f"vmax={v_max:g}:sat={sat_frac:.3f}:med={speed.median().item():.3g}")
                        print(f"[erosion_grid beta_e={beta_e:g} mask={mask_name}] " + " | ".join(row))


def entrypoint() -> None:
    tyro.cli(DebugInvSScale).main()


if __name__ == "__main__":
    entrypoint()
