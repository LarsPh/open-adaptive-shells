from __future__ import annotations

import torch

from adashell_renderer.camera_path import CameraPathFrame
from adashell_renderer.types import RayBatch


def generate_rays_reference(
    frame: CameraPathFrame,
    device: str = "cuda",
    camera_index: int | None = None,
    t_near: float = 0.05,
    t_far: float = 4.0,
    pixel_center_offset: float = 0.5,
) -> RayBatch:
    ys, xs = torch.meshgrid(
        torch.arange(frame.height, dtype=torch.float32, device=device),
        torch.arange(frame.width, dtype=torch.float32, device=device),
        indexing="ij",
    )
    dir_x = (xs - frame.cx + pixel_center_offset) / frame.fx
    dir_y = -(ys - frame.cy + pixel_center_offset) / frame.fy
    dir_z = -torch.ones_like(dir_x)
    dirs_cam = torch.stack((dir_x, dir_y, dir_z), dim=-1)
    dirs_cam = dirs_cam / torch.linalg.norm(dirs_cam, dim=-1, keepdim=True).clamp_min(1e-8)

    rotation = frame.camera_to_world[:, :3].to(device=device)
    translation = frame.camera_to_world[:, 3].to(device=device)
    dirs_world = torch.matmul(dirs_cam.reshape(-1, 3), rotation.T)
    dirs_world = dirs_world / torch.linalg.norm(dirs_world, dim=-1, keepdim=True).clamp_min(1e-8)
    origins = translation.view(1, 3).expand(frame.height * frame.width, 3).contiguous()

    metadata: dict[str, object] = {
        "image_height": frame.height,
        "image_width": frame.width,
        "raygen_backend": "torch_reference",
    }
    if camera_index is not None:
        metadata["camera_index"] = int(camera_index)

    return RayBatch(
        origins=origins,
        directions=dirs_world,
        t_near=torch.full((origins.shape[0],), float(t_near), device=device, dtype=torch.float32),
        t_far=torch.full((origins.shape[0],), float(t_far), device=device, dtype=torch.float32),
        metadata=metadata,
    )
