from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np
import torch

from adashell_renderer.camera_path import CameraPath, CameraPathFrame, create_default_camera_path
from adashell_renderer.config import RendererConfig
from adashell_renderer.cuda_raygen import generate_rays_cupy
from adashell_renderer.drjit_backend.stage_adapter import apply_fastpath_defaults
from adashell_renderer.renderer_runtime import _profile_total_ms, render_full_image_buffer
from adashell_renderer.renderer_pipeline import RendererPipeline
from adashell_renderer.types import DrRenderBuffer

DisplayMode = Literal[
    "RGB",
    "sample_count",
    "depth",
    "pred_normal",
    "pred_gradient",
    "pred_gradient_norm",
    "kernel_size",
    "weight_sum",
]
RotateMode = Literal["Orbit", "Look"]


@dataclass(slots=True)
class ViewerTimings:
    core_ms: float = 0.0
    texture_upload_ms: float = 0.0
    total_wall_ms: float = 0.0

    @property
    def displayed_total_ms(self) -> float:
        return self.total_wall_ms + self.texture_upload_ms

    @property
    def displayed_fps(self) -> float:
        total_ms = self.displayed_total_ms
        return 1000.0 / total_ms if total_ms > 1e-6 else 0.0


@dataclass(slots=True)
class ViewerRenderFrame:
    frame: CameraPathFrame
    profile: dict[str, float]
    details: dict[str, object]
    wall_ms: float
    display_buffer: DrRenderBuffer


@dataclass(slots=True)
class BoundsOverlayLine:
    start: tuple[float, float]
    end: tuple[float, float]
    color: tuple[int, int, int, int]


@dataclass(slots=True)
class ViewerCameraState:
    base_frame: CameraPathFrame
    target: torch.Tensor
    world_up: np.ndarray = field(default_factory=lambda: np.asarray([0.0, 1.0, 0.0]))
    _pose: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._pose = self.base_frame.camera_to_world.detach().cpu().numpy().astype(np.float64, copy=True)
        self.world_up = np.asarray(self.world_up, dtype=np.float64).copy()
        if self.world_up.shape != (3,) or not np.isfinite(self.world_up).all() or np.linalg.norm(self.world_up) < 1e-8:
            raise ValueError("Camera world-up must be a finite nonzero vector.")
        self.world_up /= np.linalg.norm(self.world_up)
        self.set_target(self.target)

    @classmethod
    def from_frame(
        cls, frame: CameraPathFrame, target: torch.Tensor | None = None, *, world_up: np.ndarray | None = None,
    ) -> "ViewerCameraState":
        c2w = frame.camera_to_world.detach().cpu()
        if target is None:
            target = c2w[:, 3] - c2w[:, 2]
        return cls(base_frame=frame, target=target, world_up=np.asarray([0.0, 1.0, 0.0]) if world_up is None else world_up)

    @property
    def radius(self) -> float:
        return float(np.linalg.norm(self._pose[:, 3] - self.target.numpy()))

    @property
    def pitch(self) -> float:
        offset = self._pose[:, 3] - self.target.numpy()
        height = float(np.dot(offset, self.world_up))
        return math.atan2(height, float(np.linalg.norm(offset - height * self.world_up)))

    def set_target(self, target: torch.Tensor) -> None:
        """Select an orbit pivot without changing the saved camera pose."""
        self.target = target.detach().cpu().to(dtype=self.base_frame.camera_to_world.dtype).clone()
        if self.radius < 1e-12:
            self.target = torch.from_numpy((self._pose[:, 3] - self._pose[:, 2]).copy()).to(dtype=self.target.dtype)

    def reset(self, frame: CameraPathFrame, target: torch.Tensor | None = None) -> None:
        replacement = ViewerCameraState.from_frame(frame, target=target, world_up=self.world_up)
        self.base_frame = replacement.base_frame
        self.target = replacement.target
        self._pose = replacement._pose

    def orbit(
        self, dx: float, dy: float, *, invert_y: bool = False,
        viewport_size: tuple[float, float] | None = None,
    ) -> None:
        dx, dy = self._drag_pixels(dx, dy, viewport_size)
        if dx == 0.0 and dy == 0.0:
            return
        pivot = self.target.numpy()
        offset = self._pose[:, 3] - pivot
        rotation = _turntable_rotation(offset, self._pose[:, 0], dx, dy, invert_y=invert_y, world_up=self.world_up)
        self._pose[:, 3] = pivot + rotation @ offset
        self._pose[:, :3] = rotation @ self._pose[:, :3]

    def look(
        self, dx: float, dy: float, *, invert_y: bool = False,
        viewport_size: tuple[float, float] | None = None,
    ) -> None:
        dx, dy = self._drag_pixels(dx, dy, viewport_size)
        if dx == 0.0 and dy == 0.0:
            return
        radius = self.radius
        rotation = _turntable_rotation(self._pose[:, 2], self._pose[:, 0], dx, dy, invert_y=invert_y, world_up=self.world_up)
        self._pose[:, :3] = rotation @ self._pose[:, :3]
        self.target = torch.from_numpy((self._pose[:, 3] - self._pose[:, 2] * radius).copy()).to(dtype=self.target.dtype)

    def zoom(self, delta: float) -> None:
        if delta == 0.0:
            return
        # Relative limits work for both tiny and large scenes, without crossing the pivot.
        factor = math.exp(float(np.clip(-delta * 0.08, -20.0, 20.0)))
        self._pose[:, 3] = self.target.numpy() + (self._pose[:, 3] - self.target.numpy()) * factor

    def pan(self, dx: float, dy: float, *, viewport_size: tuple[float, float] | None = None) -> None:
        """Drag in base-frame pixels; render resolution scaling does not affect sensitivity."""
        dx, dy = self._drag_pixels(dx, dy, viewport_size)
        depth = float(np.dot(self._pose[:, 3] - self.target.numpy(), self._pose[:, 2]))
        distance = depth if depth > self.radius * 1e-6 else self.radius
        self.move_local(
            right=-float(dx) * distance / self.base_frame.fx,
            up=float(dy) * distance / self.base_frame.fy,
        )

    def move_local(self, *, right: float = 0.0, up: float = 0.0, forward: float = 0.0) -> None:
        delta = self._pose[:, :3] @ np.asarray([right, up, -forward], dtype=np.float64)
        self._pose[:, 3] += delta
        self.target = self.target + torch.from_numpy(delta).to(dtype=self.target.dtype)

    def _drag_pixels(
        self, dx: float, dy: float, viewport_size: tuple[float, float] | None,
    ) -> tuple[float, float]:
        if viewport_size is None:
            return float(dx), float(dy)
        width, height = viewport_size
        if not math.isfinite(width) or not math.isfinite(height) or width <= 0.0 or height <= 0.0:
            raise ValueError("Camera viewport dimensions must be finite and positive.")
        return float(dx) * self.base_frame.width / width, float(dy) * self.base_frame.height / height

    def to_frame(self, resolution_scale: float) -> CameraPathFrame:
        c2w = torch.from_numpy(self._pose.copy()).to(dtype=self.base_frame.camera_to_world.dtype)
        return scale_camera_frame(
            CameraPathFrame(
                camera_to_world=c2w,
                fx=self.base_frame.fx,
                fy=self.base_frame.fy,
                cx=self.base_frame.cx,
                cy=self.base_frame.cy,
                width=self.base_frame.width,
                height=self.base_frame.height,
            ),
            resolution_scale,
        )


@dataclass(slots=True)
class ViewerState:
    config: RendererConfig
    camera_path: CameraPath
    pipeline: RendererPipeline
    camera_index: int
    resolution_scale: float
    display_mode: DisplayMode
    object_center: torch.Tensor | None = None
    field_bounds: torch.Tensor | None = None
    shell_bounds: torch.Tensor | None = None
    show_bounds_overlay: bool = False
    rotate_mode: RotateMode = "Orbit"
    invert_y: bool = False
    move_speed: float = 1.0
    viewport_size: tuple[float, float] | None = None
    continuous: bool = False
    dirty: bool = True
    last_mouse_pos: tuple[float, float] | None = None
    timings: ViewerTimings = field(default_factory=ViewerTimings)
    profile: dict[str, float] = field(default_factory=dict)
    details: dict[str, object] = field(default_factory=dict)
    last_render_frame: CameraPathFrame | None = None
    camera: ViewerCameraState = field(init=False)

    def __post_init__(self) -> None:
        self.camera = ViewerCameraState.from_frame(
            self.camera_path.frames[self.camera_index], target=self.object_center, world_up=_camera_path_up(self.camera_path),
        )

    def fit_camera(self) -> None:
        bounds = self.shell_bounds if self.shell_bounds is not None else self.field_bounds
        if bounds is None:
            raise ValueError("Fit requires scene bounds.")
        frame = self.camera.base_frame
        fov = math.degrees(2.0 * math.atan(frame.height / (2.0 * frame.fy)))
        fitted, far = create_default_camera_path(bounds, width=frame.width, height=frame.height, vertical_fov_degrees=fov)
        center = (bounds[0] + bounds[1]) * 0.5
        fitted_frame = fitted.frames[0]
        distance = fitted_frame.camera_to_world[2, 3] - center[2]
        rotation = self.camera.to_frame(1.0).camera_to_world[:, :3]
        fitted_frame.camera_to_world[:, :3] = rotation
        fitted_frame.camera_to_world[:, 3] = center + rotation[:, 2] * distance
        self.camera.reset(fitted_frame, target=center)
        self.config.render.ray_t_far = max(self.config.render.ray_t_far, far)
        self.dirty = True


def _camera_path_up(camera_path: CameraPath) -> np.ndarray:
    poses = np.stack([frame.camera_to_world.detach().cpu().numpy() for frame in camera_path.frames])
    up = poses[:, :, 1].mean(axis=0)
    if np.linalg.norm(up) < 1e-6:
        up = poses[0, :, 1]
    if len(poses) >= 3:
        # Upright cameras share an up direction perpendicular to their screen-right axes.
        _, singular_values, basis = np.linalg.svd(poses[:, :, 0], full_matrices=False)
        if singular_values[1] > 1e-6 and singular_values[2] < 0.1 * singular_values[1]:
            axis = basis[-1]
            up = axis if np.dot(axis, up) >= 0.0 else -axis
    return up


def scale_camera_frame(frame: CameraPathFrame, resolution_scale: float) -> CameraPathFrame:
    scale = float(np.clip(resolution_scale, 0.05, 4.0))
    width = max(1, int(round(frame.width * scale)))
    height = max(1, int(round(frame.height * scale)))
    sx = width / float(frame.width)
    sy = height / float(frame.height)
    return CameraPathFrame(
        camera_to_world=frame.camera_to_world,
        fx=frame.fx * sx,
        fy=frame.fy * sy,
        cx=frame.cx * sx,
        cy=frame.cy * sy,
        width=width,
        height=height,
    )


def _camera_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    unit_axis = axis / np.linalg.norm(axis)
    x, y, z = unit_axis
    cross = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    cos_a = math.cos(angle)
    return np.eye(3) * cos_a + cross * math.sin(angle) + np.outer(unit_axis, unit_axis) * (1.0 - cos_a)


def _turntable_rotation(
    offset: np.ndarray, right: np.ndarray, dx: float, dy: float, *, invert_y: bool, world_up: np.ndarray,
) -> np.ndarray:
    yaw_rotation = _camera_rotation(world_up, -float(dx) * 0.006)
    direction = yaw_rotation @ (offset / np.linalg.norm(offset))
    pitch_axis = np.cross(world_up, direction)
    if np.linalg.norm(pitch_axis) < 1e-8:
        # At an imported pole there is no azimuth; use the saved screen-right heading.
        pitch_axis = yaw_rotation @ right
        pitch_axis -= world_up * np.dot(pitch_axis, world_up)
        if np.linalg.norm(pitch_axis) < 1e-8:
            pitch_axis = np.cross(world_up, np.eye(3)[np.argmin(np.abs(world_up))])
    height = float(np.dot(direction, world_up))
    pitch = math.atan2(height, float(np.linalg.norm(direction - height * world_up)))
    limit = math.pi * 0.5 - 1e-4
    requested = float(dy) * 0.006 * (-1.0 if invert_y else 1.0)
    # Imported poses beyond the interactive limit stay intact until dragged away.
    next_pitch = float(np.clip(pitch + requested, min(-limit, pitch), max(limit, pitch)))
    return _camera_rotation(pitch_axis, pitch - next_pitch) @ yaw_rotation


def diagnostic_mode_for_display(display_mode: DisplayMode) -> str:
    if display_mode in {"pred_normal", "pred_gradient", "pred_gradient_norm", "kernel_size", "weight_sum"}:
        return display_mode
    return "none"


def _field_aabb_from_pipeline(pipeline: RendererPipeline) -> torch.Tensor | None:
    aabb = getattr(pipeline.field_island.field_model, "aabb", None)
    if aabb is None:
        return None
    aabb_value = aabb.detach().cpu() if isinstance(aabb, torch.Tensor) else aabb
    aabb_tensor = torch.as_tensor(aabb_value, dtype=torch.float32)
    if tuple(aabb_tensor.shape) == (2, 3) and torch.isfinite(aabb_tensor).all():
        return aabb_tensor
    return None


BOUNDS_EDGES: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)


def _bounds_corners_np(bounds: torch.Tensor) -> np.ndarray:
    bounds_np = bounds.detach().cpu().to(dtype=torch.float32).numpy()
    mins = bounds_np[0]
    maxs = bounds_np[1]
    return np.asarray(
        [
            [mins[0], mins[1], mins[2]],
            [maxs[0], mins[1], mins[2]],
            [maxs[0], maxs[1], mins[2]],
            [mins[0], maxs[1], mins[2]],
            [mins[0], mins[1], maxs[2]],
            [maxs[0], mins[1], maxs[2]],
            [maxs[0], maxs[1], maxs[2]],
            [mins[0], maxs[1], maxs[2]],
        ],
        dtype=np.float32,
    )


def project_bounds_overlay_lines(
    frame: CameraPathFrame,
    bounds: torch.Tensor | None,
    *,
    color: tuple[int, int, int, int],
) -> list[BoundsOverlayLine]:
    """Project one 3D bounds box to image-space line segments without touching render tensors."""
    if bounds is None:
        return []
    bounds_tensor = bounds.detach().cpu().to(dtype=torch.float32)
    if tuple(bounds_tensor.shape) != (2, 3) or not bool(torch.isfinite(bounds_tensor).all().item()):
        return []

    corners_world = _bounds_corners_np(bounds_tensor)
    c2w = frame.camera_to_world.detach().cpu().to(dtype=torch.float32).numpy()
    rotation = c2w[:, :3]
    translation = c2w[:, 3]
    corners_cam = (corners_world - translation[None, :]) @ rotation
    z = corners_cam[:, 2]
    visible = z < -1e-5
    denom = np.maximum(-z, 1e-5)
    pixels = np.empty((8, 2), dtype=np.float32)
    pixels[:, 0] = float(frame.fx) * (corners_cam[:, 0] / denom) + float(frame.cx)
    pixels[:, 1] = -float(frame.fy) * (corners_cam[:, 1] / denom) + float(frame.cy)

    lines: list[BoundsOverlayLine] = []
    for start, end in BOUNDS_EDGES:
        if bool(visible[start] and visible[end]):
            lines.append(
                BoundsOverlayLine(
                    start=(float(pixels[start, 0]), float(pixels[start, 1])),
                    end=(float(pixels[end, 0]), float(pixels[end, 1])),
                    color=color,
                )
            )
    return lines


def project_debug_bounds_overlay_lines(
    frame: CameraPathFrame,
    *,
    field_bounds: torch.Tensor | None,
    shell_bounds: torch.Tensor | None,
) -> list[BoundsOverlayLine]:
    lines = project_bounds_overlay_lines(frame, field_bounds, color=(255, 8, 8, 255))
    lines.extend(project_bounds_overlay_lines(frame, shell_bounds, color=(0, 242, 255, 255)))
    return lines


def load_shell_mesh_bounds(config: RendererConfig) -> torch.Tensor | None:
    paths = [
        config.mesh_assets.base_mesh,
        config.mesh_assets.dilation_mesh,
        config.mesh_assets.erosion_mesh,
    ]
    bounds: list[np.ndarray] = []
    for path in paths:
        if path is None or not path.exists():
            continue
        try:
            import trimesh

            mesh = trimesh.load(path, process=False)
        except Exception:
            continue
        mesh_bounds = np.asarray(getattr(mesh, "bounds", None), dtype=np.float64)
        if mesh_bounds.shape != (2, 3) or not np.all(np.isfinite(mesh_bounds)):
            continue
        bounds.append(mesh_bounds)
    if not bounds:
        return None
    mins = np.stack([item[0] for item in bounds], axis=0).min(axis=0)
    maxs = np.stack([item[1] for item in bounds], axis=0).max(axis=0)
    return torch.tensor(np.stack((mins, maxs), axis=0), dtype=torch.float32)


def apply_fastpath_viewer_defaults(
    config: RendererConfig,
) -> None:
    if config.use_fastpath:
        apply_fastpath_defaults(config)


def render_viewer_frame(state: ViewerState, before_field: Callable[[], None] | None = None) -> ViewerRenderFrame:
    state.pipeline.fast_render_aux_outputs = state.display_mode == "depth"
    state.pipeline.fast_render_diagnostic_mode = diagnostic_mode_for_display(state.display_mode)
    frame_start = time.perf_counter()
    frame = state.camera.to_frame(state.resolution_scale)
    state.last_render_frame = frame
    frame_ms = (time.perf_counter() - frame_start) * 1000.0
    rays_start = time.perf_counter()
    rays = generate_rays_cupy(
        frame,
        device=state.config.device,
        camera_index=state.camera_index,
        t_near=state.config.render.ray_t_near,
        t_far=state.config.render.ray_t_far,
        pixel_center_offset=state.config.render.pixel_center_offset,
    )
    rays_ms = (time.perf_counter() - rays_start) * 1000.0

    wall_start = time.perf_counter()
    display_buffer, profile, details = render_full_image_buffer(
        pipeline=state.pipeline,
        rays=rays,
        before_field=before_field,
    )
    render_done = time.perf_counter()
    render_full_image_ms = (render_done - wall_start) * 1000.0
    details["viewer_frame_ms"] = frame_ms
    details["viewer_generate_rays_ms"] = rays_ms
    for key, value in rays.metadata.items():
        if key.startswith("raygen_"):
            details[key] = value
    details["viewer_render_full_image_ms"] = render_full_image_ms
    details["viewer_render_wall_ms"] = frame_ms + rays_ms + render_full_image_ms
    details["viewer_render_profile_gap_ms"] = details["viewer_render_wall_ms"] - _profile_total_ms(profile)
    state.profile = profile
    state.details = details
    state.timings.core_ms = _profile_total_ms(profile)
    state.timings.total_wall_ms = float(details["viewer_render_wall_ms"])
    return ViewerRenderFrame(
        frame=frame,
        profile=profile,
        details=details,
        wall_ms=state.timings.total_wall_ms,
        display_buffer=display_buffer,
    )
