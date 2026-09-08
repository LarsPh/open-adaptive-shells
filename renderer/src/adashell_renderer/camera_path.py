from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(slots=True)
class CameraPathFrame:
    camera_to_world: torch.Tensor
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(slots=True)
class CameraPath:
    frames: list[CameraPathFrame]


def create_default_camera_path(
    bounds: torch.Tensor,
    *,
    width: int = 1280,
    height: int = 720,
    vertical_fov_degrees: float = 50.0,
    margin: float = 1.1,
) -> tuple[CameraPath, float]:
    bounds = torch.as_tensor(bounds, dtype=torch.float32).detach().cpu()
    if tuple(bounds.shape) != (2, 3) or not torch.isfinite(bounds).all():
        raise ValueError("Default viewer camera requires finite scene bounds with shape (2, 3).")
    if width <= 0 or height <= 0:
        raise ValueError("Default viewer camera requires positive image dimensions.")
    if not 0.0 < vertical_fov_degrees < 180.0:
        raise ValueError("Default viewer camera FOV must be between 0 and 180 degrees.")
    if margin < 1.0:
        raise ValueError("Default viewer camera margin must be at least 1.0.")
    if torch.any(bounds[1] < bounds[0]):
        raise ValueError("Default viewer camera requires ordered scene bounds.")

    center = (bounds[0] + bounds[1]) * 0.5
    radius = float(torch.linalg.norm(bounds[1] - bounds[0]).item()) * 0.5
    if radius <= 1e-6:
        raise ValueError("Default viewer camera cannot fit degenerate scene bounds.")

    half_fov_radians = math.radians(vertical_fov_degrees) * 0.5
    horizontal_half_fov = math.atan(math.tan(half_fov_radians) * width / height)
    distance = margin * radius / math.sin(min(half_fov_radians, horizontal_half_fov))
    focal = _focal_from_angle(math.radians(vertical_fov_degrees), height)
    c2w = torch.eye(4, dtype=torch.float32)[:3, :4].clone()
    c2w[:, 3] = center + torch.tensor([0.0, 0.0, distance], dtype=torch.float32)
    frame = CameraPathFrame(
        camera_to_world=c2w,
        fx=focal,
        fy=focal,
        cx=width * 0.5,
        cy=height * 0.5,
        width=width,
        height=height,
    )
    required_far = distance + margin * radius
    return CameraPath(frames=[frame]), required_far


def load_camera_path(path: Path) -> CameraPath:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "frames" in payload:
        return _load_blender_transforms_camera_path(payload)
    return _load_viewer_camera_path(payload)


def _load_viewer_camera_path(payload: dict) -> CameraPath:
    image_height = int(payload["render_height"])
    image_width = int(payload["render_width"])
    cx = image_width / 2.0
    cy = image_height / 2.0

    frames: list[CameraPathFrame] = []
    for camera in payload["camera_path"]:
        fov = float(camera["fov"])
        focal = _focal_from_angle(math.radians(fov), image_height)
        c2w = torch.tensor(camera["camera_to_world"], dtype=torch.float32).view(4, 4)[:3, :4]
        frames.append(
            CameraPathFrame(
                camera_to_world=c2w,
                fx=focal,
                fy=focal,
                cx=cx,
                cy=cy,
                width=image_width,
                height=image_height,
            )
        )
    return CameraPath(frames=frames)


def _focal_from_angle(angle: float, size: int) -> float:
    return float(0.5 * float(size) / math.tan(0.5 * float(angle)))


def _load_blender_transforms_camera_path(payload: dict) -> CameraPath:
    image_width = int(payload.get("w", payload.get("width", 0)))
    image_height = int(payload.get("h", payload.get("height", 0)))
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Blender transforms camera path requires positive `w` and `h` image dimensions.")

    fx = float(payload["fl_x"]) if "fl_x" in payload else _focal_from_angle(float(payload["camera_angle_x"]), image_width)
    fy = float(payload["fl_y"]) if "fl_y" in payload else _focal_from_angle(float(payload.get("camera_angle_y", payload["camera_angle_x"])), image_height)
    cx = float(payload.get("cx", image_width / 2.0))
    cy = float(payload.get("cy", image_height / 2.0))

    frames: list[CameraPathFrame] = []
    for frame in payload["frames"]:
        c2w = torch.tensor(frame["transform_matrix"], dtype=torch.float32).view(4, 4)[:3, :4]
        frames.append(
            CameraPathFrame(
                camera_to_world=c2w,
                fx=fx,
                fy=fy,
                cx=cx,
                cy=cy,
                width=image_width,
                height=image_height,
            )
        )
    return CameraPath(frames=frames)


