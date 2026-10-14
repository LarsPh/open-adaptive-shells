from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

import torch

from adashell_renderer.camera_path import CameraPath, create_default_camera_path, load_camera_path
from adashell_renderer.config import RendererConfig
from adashell_renderer.config_io import load_renderer_config
from adashell_renderer.cuda_gl_interop import load_cuda_gl_extension, print_build_environment
from adashell_renderer.renderer_runtime import build_runtime_pipeline
from adashell_renderer.types import DrRenderBuffer
from adashell_renderer.viewer_support import (
    DisplayMode,
    PreparedViewerFrame,
    ViewerRenderFrame,
    ViewerState,
    _field_aabb_from_pipeline,
    apply_fastpath_viewer_defaults,
    finish_viewer_frame,
    load_shell_mesh_bounds,
    prepare_viewer_frame,
    project_debug_bounds_overlay_lines,
)


@dataclass(slots=True)
class GlDisplayTimings:
    event_ms: float = 0.0
    ui_ms: float = 0.0
    render_wall_ms: float = 0.0
    presenter_ms: float = 0.0
    interop_map_ms: float = 0.0
    cuda_pack_ms: float = 0.0
    interop_unmap_ms: float = 0.0
    gl_tex_ms: float = 0.0
    imgui_draw_ms: float = 0.0
    swap_ms: float = 0.0
    loop_total_ms: float = 0.0
    sync_debug_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.loop_total_ms

    @property
    def fps(self) -> float:
        return 1000.0 / self.total_ms if self.total_ms > 1e-6 else 0.0

    @property
    def measured_parts_ms(self) -> float:
        return (
            self.event_ms
            + self.ui_ms
            + self.render_wall_ms
            + self.display_ms
            + self.sync_debug_ms
        )

    @property
    def unattributed_ms(self) -> float:
        return self.loop_total_ms - self.measured_parts_ms

    @property
    def display_ms(self) -> float:
        return self.presenter_ms + self.imgui_draw_ms + self.swap_ms


@dataclass(slots=True)
class GlViewerUiState:
    sample_min: int
    sample_max: int
    depth_min: float
    depth_max: float
    kernel_min: float = 0.0
    kernel_max: float = 0.2
    gradient_norm_min: float = 0.0
    gradient_norm_max: float = 2.0
    weight_sum_min: float = 0.0
    weight_sum_max: float = 1.0

    def mode_range(self, mode: DisplayMode) -> tuple[float, float]:
        if mode == "sample_count":
            return (float(self.sample_min), float(self.sample_max))
        if mode == "depth":
            return (self.depth_min, self.depth_max)
        if mode == "kernel_size":
            return (self.kernel_min, self.kernel_max)
        if mode == "pred_gradient_norm":
            return (self.gradient_norm_min, self.gradient_norm_max)
        if mode == "weight_sum":
            return (self.weight_sum_min, self.weight_sum_max)
        return (0.0, 1.0)


@dataclass(slots=True)
class CudaGlPresenter:
    gl: Any
    ext: Any
    texture_id: int = 0
    pbo_id: int = 0
    cuda_handle: int = 0
    width: int = 0
    height: int = 0
    timings: GlDisplayTimings = field(default_factory=GlDisplayTimings)

    def close(self) -> None:
        if self.cuda_handle:
            self.ext.unregister_pbo(int(self.cuda_handle))
            self.cuda_handle = 0
        if self.pbo_id:
            self.gl.glDeleteBuffers(1, [self.pbo_id])
            self.pbo_id = 0
        if self.texture_id:
            self.gl.glDeleteTextures([self.texture_id])
            self.texture_id = 0

    def ensure_size(self, width: int, height: int) -> None:
        if self.width == width and self.height == height and self.texture_id and self.pbo_id:
            return
        self.close()
        self.width = int(width)
        self.height = int(height)
        self.texture_id = int(self.gl.glGenTextures(1))
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.texture_id)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MIN_FILTER, self.gl.GL_LINEAR)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_MAG_FILTER, self.gl.GL_LINEAR)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_WRAP_S, self.gl.GL_CLAMP_TO_EDGE)
        self.gl.glTexParameteri(self.gl.GL_TEXTURE_2D, self.gl.GL_TEXTURE_WRAP_T, self.gl.GL_CLAMP_TO_EDGE)
        self.gl.glTexImage2D(
            self.gl.GL_TEXTURE_2D,
            0,
            self.gl.GL_RGBA8,
            self.width,
            self.height,
            0,
            self.gl.GL_RGBA,
            self.gl.GL_UNSIGNED_BYTE,
            None,
        )
        self.pbo_id = int(self.gl.glGenBuffers(1))
        self.gl.glBindBuffer(self.gl.GL_PIXEL_UNPACK_BUFFER, self.pbo_id)
        self.gl.glBufferData(
            self.gl.GL_PIXEL_UNPACK_BUFFER,
            self.width * self.height * 4,
            None,
            self.gl.GL_STREAM_DRAW,
        )
        self.gl.glBindBuffer(self.gl.GL_PIXEL_UNPACK_BUFFER, 0)
        self.cuda_handle = int(self.ext.register_pbo(int(self.pbo_id)))

    def _dr_arrays_to_cupy(self, *arrays: Any, wait: bool = True) -> tuple[Any, ...]:
        import cupy as cp  # type: ignore[import-not-found]
        import drjit as dr  # type: ignore[import-not-found]

        live_arrays = tuple(array for array in arrays if array is not None)
        if live_arrays:
            dr.eval(*live_arrays)
            if wait:
                dr.sync_thread()
        cupy_arrays = tuple(cp.from_dlpack(array) for array in arrays)
        for array in cupy_arrays:
            if array.dtype != cp.float32:
                raise TypeError(f"Direct PBO display expects float32 Dr buffers, got {array.dtype}.")
            if not array.flags.c_contiguous:
                raise ValueError("Direct PBO display expects contiguous Dr/CuPy buffers.")
            if array.ndim != 1 or array.size != self.width * self.height:
                raise ValueError("Direct PBO display expects one float32 value per pixel and channel.")
            if array.device.id != cp.cuda.runtime.getDevice():
                raise ValueError("Display buffers must use the current CUDA/OpenGL device.")
        return cupy_arrays

    def _zero_cupy_float_buffer(self, total: int) -> Any:
        import cupy as cp  # type: ignore[import-not-found]

        return cp.zeros((int(total),), dtype=cp.float32)

    def _upload_display_buffer(
        self,
        result: ViewerRenderFrame,
        mode: DisplayMode,
        value_range: tuple[float, float],
        wait: bool = True,
    ) -> None:
        buffer = result.display_buffer
        if buffer is None:
            raise ValueError("GL display requires a DrRenderBuffer.")
        width = int(result.frame.width)
        height = int(result.frame.height)
        if int(buffer.num_rays) != width * height:
            raise RuntimeError(
                "Direct GL PBO display requires a full-frame Dr buffer; "
                f"got {buffer.num_rays} rays for {width}x{height}."
            )
        self.ensure_size(width, height)
        result.details["display_direct_pbo"] = 1.0
        result.details["display_torch_materialize_ms"] = 0.0
        result.details["rgb_materialize_ms"] = 0.0

        scalar_modes = {"sample_count", "depth", "pred_gradient_norm", "kernel_size", "weight_sum"}
        vector_modes = {"pred_normal", "pred_gradient"}
        dlpack_start = time.perf_counter()
        keepalive: tuple[Any, ...]
        if mode in scalar_modes:
            if mode == "depth":
                scalar = buffer.depth
            else:
                scalar = buffer.diagnostics.get(mode)
            if scalar is None:
                keepalive = (self._zero_cupy_float_buffer(width * height),)
            else:
                keepalive = self._dr_arrays_to_cupy(scalar, wait=wait)
            result.details["display_dr_to_cupy_ms"] = (time.perf_counter() - dlpack_start) * 1000.0
            timings = self.ext.copy_scalar_ptr_to_pbo(
                int(self.cuda_handle),
                int(keepalive[0].data.ptr),
                width,
                height,
                float(value_range[0]),
                float(value_range[1]),
                0,
            )
        elif mode in vector_modes:
            vector = buffer.diagnostics.get(mode)
            if isinstance(vector, tuple) and len(vector) == 3:
                keepalive = self._dr_arrays_to_cupy(vector[0], vector[1], vector[2], wait=wait)
            else:
                zero = self._zero_cupy_float_buffer(width * height)
                keepalive = (zero, zero, zero)
            result.details["display_dr_to_cupy_ms"] = (time.perf_counter() - dlpack_start) * 1000.0
            timings = self.ext.copy_vector_ptrs_to_pbo(
                int(self.cuda_handle),
                int(keepalive[0].data.ptr),
                int(keepalive[1].data.ptr),
                int(keepalive[2].data.ptr),
                width,
                height,
                True,
            )
        else:
            r, g, b = self._dr_arrays_to_cupy(*buffer.rgb_channels, wait=wait)
            keepalive = (r, g, b)
            result.details["display_dr_to_cupy_ms"] = (time.perf_counter() - dlpack_start) * 1000.0
            timings = self.ext.copy_rgb_ptrs_to_pbo(
                int(self.cuda_handle),
                int(r.data.ptr),
                int(g.data.ptr),
                int(b.data.ptr),
                width,
                height,
            )
        result.details["display_pbo_pack_ms"] = float(timings[1])
        self.timings.interop_map_ms = float(timings[0])
        self.timings.cuda_pack_ms = float(timings[1])
        self.timings.interop_unmap_ms = float(timings[2])

    def upload(
        self,
        result: ViewerRenderFrame,
        mode: DisplayMode,
        value_range: tuple[float, float],
        wait: bool = True,
    ) -> None:
        """Pack a frame into the PBO and texture.

        `wait=False` skips the host wait for Dr.Jit; only pass it when the
        caller has already synchronized past the frame's kernels.
        """
        self._upload_display_buffer(result, mode, value_range, wait=wait)

        tex_start = time.perf_counter()
        self.gl.glBindTexture(self.gl.GL_TEXTURE_2D, self.texture_id)
        self.gl.glBindBuffer(self.gl.GL_PIXEL_UNPACK_BUFFER, self.pbo_id)
        self.gl.glTexSubImage2D(
            self.gl.GL_TEXTURE_2D,
            0,
            0,
            0,
            self.width,
            self.height,
            self.gl.GL_RGBA,
            self.gl.GL_UNSIGNED_BYTE,
            None,
        )
        self.gl.glBindBuffer(self.gl.GL_PIXEL_UNPACK_BUFFER, 0)
        self.timings.gl_tex_ms = (time.perf_counter() - tex_start) * 1000.0


def _require_gl_stack():
    try:
        import glfw
        import imgui
        from imgui.integrations.glfw import GlfwRenderer
        from OpenGL import GL
    except ModuleNotFoundError as exc:
        raise RuntimeError("GL viewer requires: uv sync --extra viewer-gl") from exc
    return glfw, imgui, GlfwRenderer, GL


def _status_line(state: ViewerState, timings: GlDisplayTimings) -> str:
    return _timing_summary(state, timings)


def _timing_tree(state: ViewerState, timings: GlDisplayTimings) -> str:
    profile = state.profile
    details = state.details
    render_ms = float(details.get("viewer_render_wall_ms", timings.render_wall_ms))
    frame_ms = timings.total_ms
    display_ms = timings.display_ms
    rays = float(details.get("viewer_generate_rays_ms", 0.0))
    frame = float(details.get("viewer_frame_ms", 0.0))
    trace = float(profile.get("trace_ms", 0.0))
    sample = float(profile.get("sample_ms", 0.0))
    field = float(profile.get("field_island_ms", 0.0))
    render_known = (
        frame
        + rays + trace + sample
        + field
    )
    render_residual = render_ms - render_known
    display_residual = display_ms - (
        timings.presenter_ms + timings.imgui_draw_ms + timings.swap_ms
    )
    frame_residual = frame_ms - (render_ms + display_ms + timings.event_ms + timings.ui_ms + timings.sync_debug_ms)
    return (
        f"frame {frame_ms:.2f} ms ({timings.fps:.1f} fps)\n"
        f"  render {render_ms:.2f}\n"
        f"    camera/frame {frame:.2f}\n"
        f"    rays {rays:.2f}\n"
        f"    trace {trace:.2f}\n"
        f"    sample {sample:.2f}\n"
        f"    field {field:.2f}\n"
        f"    residual {render_residual:.2f}\n"
        f"  display {display_ms:.2f}\n"
        f"    present {timings.presenter_ms:.2f}\n"
        f"      direct_pbo {float(details.get('display_direct_pbo', 0.0)):.0f}\n"
        f"      dr_to_cupy {float(details.get('display_dr_to_cupy_ms', 0.0)):.2f}\n"
        f"      pbo_pack {float(details.get('display_pbo_pack_ms', timings.cuda_pack_ms)):.2f}\n"
        f"      map/unmap {timings.interop_map_ms + timings.interop_unmap_ms:.2f}\n"
        f"      gltex {timings.gl_tex_ms:.2f}\n"
        f"    imgui_draw {timings.imgui_draw_ms:.2f}\n"
        f"    swap {timings.swap_ms:.2f}\n"
        f"    residual {display_residual:.2f}\n"
        f"  ui/event {timings.ui_ms + timings.event_ms:.2f}\n"
        f"    event {timings.event_ms:.2f}\n"
        f"    ui {timings.ui_ms:.2f}\n"
        f"  frame residual {frame_residual:.2f}\n"
        f"samples {state.details.get('valid_samples', 0.0):.0f}"
    )


def _timing_summary(state: ViewerState, timings: GlDisplayTimings) -> str:
    render_ms = float(state.details.get("viewer_render_wall_ms", timings.render_wall_ms))
    if timings.render_wall_ms == 0.0:
        return f"Idle | last render {render_ms:.2f} ms | samples {state.details.get('valid_samples', 0.0):.0f}"
    display_ms = timings.display_ms
    frame_residual = timings.total_ms - (render_ms + display_ms + timings.event_ms + timings.ui_ms + timings.sync_debug_ms)
    return (
        f"frame {timings.total_ms:.2f} ms ({timings.fps:.1f} fps) | "
        f"render {render_ms:.2f} | "
        f"display {display_ms:.2f} | "
        f"ui/event {timings.ui_ms + timings.event_ms:.2f} | "
        f"residual {frame_residual:.2f} | "
        f"samples {state.details.get('valid_samples', 0.0):.0f}"
    )


def _handle_mouse_camera(imgui: Any, state: ViewerState, hovered: bool) -> None:
    io = imgui.get_io()
    mouse_pos = (float(io.mouse_pos.x), float(io.mouse_pos.y))
    if hovered:
        if state.last_mouse_pos is not None:
            dx = mouse_pos[0] - state.last_mouse_pos[0]
            dy = mouse_pos[1] - state.last_mouse_pos[1]
            if io.mouse_down[0]:
                if state.rotate_mode == "Look":
                    state.camera.look(dx, dy, invert_y=state.invert_y, viewport_size=state.viewport_size)
                else:
                    state.camera.orbit(dx, dy, invert_y=state.invert_y, viewport_size=state.viewport_size)
                state.dirty = True
            elif io.mouse_down[1] or io.mouse_down[2]:
                state.camera.pan(dx, dy, viewport_size=state.viewport_size)
                state.dirty = True
        if abs(float(io.mouse_wheel)) > 0.0:
            state.camera.zoom(float(io.mouse_wheel))
            state.dirty = True
        state.last_mouse_pos = mouse_pos
    else:
        state.last_mouse_pos = None


def _handle_keyboard_camera(glfw: Any, window: Any, state: ViewerState, hovered: bool, dt_s: float) -> None:
    if not hovered or state.rotate_mode != "Look":
        return
    speed = max(0.01, float(state.move_speed))
    if glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS:
        speed *= 4.0
    step = speed * min(max(dt_s, 0.0), 0.1)
    right = 0.0
    up = 0.0
    forward = 0.0
    if glfw.get_key(window, glfw.KEY_W) == glfw.PRESS:
        forward += step
    if glfw.get_key(window, glfw.KEY_S) == glfw.PRESS:
        forward -= step
    if glfw.get_key(window, glfw.KEY_D) == glfw.PRESS:
        right += step
    if glfw.get_key(window, glfw.KEY_A) == glfw.PRESS:
        right -= step
    if glfw.get_key(window, glfw.KEY_E) == glfw.PRESS:
        up += step
    if glfw.get_key(window, glfw.KEY_Q) == glfw.PRESS:
        up -= step
    if right != 0.0 or up != 0.0 or forward != 0.0:
        state.camera.move_local(right=right, up=up, forward=forward)
        state.dirty = True


def _imgui_vec2_xy(value: Any) -> tuple[float, float]:
    if hasattr(value, "x") and hasattr(value, "y"):
        return float(value.x), float(value.y)
    return float(value[0]), float(value[1])


def _draw_imgui_bounds_overlay(imgui: Any, state: ViewerState, frame: Any, image_min: Any, image_size: tuple[float, float]) -> None:
    if not state.show_bounds_overlay or frame is None:
        return
    min_x, min_y = _imgui_vec2_xy(image_min)
    sx = float(image_size[0]) / max(float(frame.width), 1.0)
    sy = float(image_size[1]) / max(float(frame.height), 1.0)
    draw_list = imgui.get_window_draw_list()
    for line in project_debug_bounds_overlay_lines(
        frame,
        field_bounds=state.field_bounds,
        shell_bounds=state.shell_bounds,
    ):
        r, g, b, a = line.color
        color = imgui.get_color_u32_rgba(r / 255.0, g / 255.0, b / 255.0, a / 255.0)
        x0 = min_x + line.start[0] * sx
        y0 = min_y + line.start[1] * sy
        x1 = min_x + line.end[0] * sx
        y1 = min_y + line.end[1] * sy
        try:
            draw_list.add_line(x0, y0, x1, y1, color, 2.0)
        except TypeError:
            draw_list.add_line((x0, y0), (x1, y1), color, 2.0)


@dataclass(slots=True)
class GlViewerSession:
    state: ViewerState
    glfw: Any
    imgui: Any
    gl: Any
    window: Any
    impl: Any
    presenter: CudaGlPresenter
    timing_debug: bool = False
    ui_state: GlViewerUiState = field(init=False)
    sample_limit: int = field(init=False)
    camera_items: list[str] = field(init=False)
    last_result: ViewerRenderFrame | None = None
    # Rendered but not yet shown; it is packed while the next frame renders.
    pending: tuple[ViewerRenderFrame, DisplayMode] | None = None
    # Rays generated and trace launched; sampled and evaluated next loop.
    inflight: PreparedViewerFrame | None = None
    status: str = "warming up"
    timing_tree: str = "frame\n  render\n  display"
    last_loop_start: float = field(default_factory=time.perf_counter)

    DISPLAY_ITEMS: ClassVar[list[DisplayMode]] = [
        "RGB",
        "sample_count",
        "depth",
        "pred_normal",
        "pred_gradient",
        "pred_gradient_norm",
        "kernel_size",
        "weight_sum",
    ]
    ROTATE_ITEMS: ClassVar[list[str]] = ["Orbit", "Look"]

    def __post_init__(self) -> None:
        self.camera_items = [str(i) for i in range(len(self.state.camera_path.frames))]
        self.sample_limit = max(
            1,
            int(self.state.config.trace.n_sample_max * self.state.config.trace.n_depth_peeling_max),
        )
        self.ui_state = GlViewerUiState(
            sample_min=0,
            sample_max=self.sample_limit,
            depth_min=float(self.state.config.render.ray_t_near),
            depth_max=float(self.state.config.render.ray_t_far),
        )

    def run(self) -> None:
        while not self.glfw.window_should_close(self.window):
            self._run_frame()

    def _run_frame(self) -> None:
        frame_start = time.perf_counter()
        dt_s = frame_start - self.last_loop_start
        self.last_loop_start = frame_start
        event_start = time.perf_counter()
        self.glfw.poll_events()
        self.impl.process_inputs()
        self.presenter.timings.event_ms = (time.perf_counter() - event_start) * 1000.0

        ui_start = time.perf_counter()
        self.imgui.new_frame()
        range_dirty = self._draw_controls()
        self.presenter.timings.ui_ms = (time.perf_counter() - ui_start) * 1000.0

        self._render_if_needed(range_dirty)
        hovered = self._draw_viewport()
        _handle_mouse_camera(self.imgui, self.state, hovered)
        if not self.imgui.get_io().want_text_input:
            _handle_keyboard_camera(self.glfw, self.window, self.state, hovered, dt_s)
        self._submit_frame(frame_start)

    def _draw_controls(self) -> bool:
        state = self.state
        imgui = self.imgui
        ui_state = self.ui_state
        range_dirty = False

        _, window_h = self.glfw.get_window_size(self.window)
        imgui.set_next_window_position(10, 10, condition=imgui.ALWAYS)
        imgui.set_next_window_size(410, max(400, window_h - 20), condition=imgui.ALWAYS)
        imgui.begin("Controls", flags=imgui.WINDOW_NO_MOVE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE)
        changed, camera_index = imgui.combo("Camera", state.camera_index, self.camera_items)
        if changed:
            state.camera_index = int(camera_index)
            state.camera.reset(state.camera_path.frames[state.camera_index], target=state.object_center)
            state.dirty = True
        display_index = self.DISPLAY_ITEMS.index(state.display_mode)
        changed, display_index = imgui.combo("View", display_index, self.DISPLAY_ITEMS)
        if changed:
            state.display_mode = self.DISPLAY_ITEMS[display_index]
            state.dirty = True
        rotate_index = self.ROTATE_ITEMS.index(state.rotate_mode)
        changed, rotate_index = imgui.combo("Rotate", rotate_index, self.ROTATE_ITEMS)
        if changed:
            state.rotate_mode = self.ROTATE_ITEMS[rotate_index]
        _changed, state.continuous = imgui.checkbox("Continuous", state.continuous)
        _changed, state.show_bounds_overlay = imgui.checkbox("Bounds", state.show_bounds_overlay)
        _changed, state.invert_y = imgui.checkbox("Invert Y", state.invert_y)
        changed_speed, state.move_speed = imgui.slider_float("Move speed", state.move_speed, 0.05, 10.0)
        if changed_speed:
            state.move_speed = max(0.01, float(state.move_speed))
        if imgui.button("Reset"):
            state.camera.reset(state.camera_path.frames[state.camera_index], target=state.object_center)
            state.dirty = True
        imgui.same_line()
        if imgui.button("Fit"):
            state.fit_camera()
        changed, state.resolution_scale = imgui.slider_float("Resolution", state.resolution_scale, 0.25, 1.0)
        if changed:
            state.dirty = True
        if state.display_mode == "sample_count":
            changed, sample_min = imgui.slider_int("Sample min", int(ui_state.sample_min), 0, self.sample_limit)
            if changed:
                ui_state.sample_min = min(int(sample_min), int(ui_state.sample_max) - 1)
                range_dirty = True
            changed, sample_max = imgui.slider_int("Sample max", int(ui_state.sample_max), 1, self.sample_limit)
            if changed:
                ui_state.sample_max = max(int(sample_max), int(ui_state.sample_min) + 1)
                range_dirty = True
        elif state.display_mode == "depth":
            changed, depth_min = imgui.slider_float("Depth min", ui_state.depth_min, 0.0, max(ui_state.depth_max, 1.0))
            if changed:
                ui_state.depth_min = min(float(depth_min), ui_state.depth_max - 1e-3)
                range_dirty = True
            changed, depth_max = imgui.slider_float(
                "Depth max",
                ui_state.depth_max,
                max(ui_state.depth_min + 1e-3, 1e-3),
                max(ui_state.depth_max, 1.0) * 2.0,
            )
            if changed:
                ui_state.depth_max = max(float(depth_max), ui_state.depth_min + 1e-3)
                range_dirty = True
        elif state.display_mode == "kernel_size":
            changed, ui_state.kernel_min = imgui.slider_float(
                "Kernel min",
                ui_state.kernel_min,
                0.0,
                max(ui_state.kernel_max, 1e-3),
            )
            if changed:
                ui_state.kernel_min = min(float(ui_state.kernel_min), ui_state.kernel_max - 1e-6)
                range_dirty = True
            changed, ui_state.kernel_max = imgui.slider_float(
                "Kernel max",
                ui_state.kernel_max,
                max(ui_state.kernel_min + 1e-6, 1e-6),
                max(ui_state.kernel_max, 1e-3) * 2.0,
            )
            if changed:
                ui_state.kernel_max = max(float(ui_state.kernel_max), ui_state.kernel_min + 1e-6)
                range_dirty = True
        elif state.display_mode == "pred_gradient_norm":
            changed, ui_state.gradient_norm_min = imgui.slider_float(
                "Grad norm min",
                ui_state.gradient_norm_min,
                0.0,
                max(ui_state.gradient_norm_max, 1.0),
            )
            if changed:
                ui_state.gradient_norm_min = min(
                    float(ui_state.gradient_norm_min),
                    ui_state.gradient_norm_max - 1e-6,
                )
                range_dirty = True
            changed, ui_state.gradient_norm_max = imgui.slider_float(
                "Grad norm max",
                ui_state.gradient_norm_max,
                max(ui_state.gradient_norm_min + 1e-6, 1e-6),
                max(ui_state.gradient_norm_max, 1.0) * 2.0,
            )
            if changed:
                ui_state.gradient_norm_max = max(
                    float(ui_state.gradient_norm_max),
                    ui_state.gradient_norm_min + 1e-6,
                )
                range_dirty = True
        elif state.display_mode == "weight_sum":
            changed, ui_state.weight_sum_min = imgui.slider_float(
                "Weight min",
                ui_state.weight_sum_min,
                0.0,
                max(ui_state.weight_sum_max, 1.0),
            )
            if changed:
                ui_state.weight_sum_min = min(float(ui_state.weight_sum_min), ui_state.weight_sum_max - 1e-6)
                range_dirty = True
            changed, ui_state.weight_sum_max = imgui.slider_float(
                "Weight max",
                ui_state.weight_sum_max,
                max(ui_state.weight_sum_min + 1e-6, 1e-6),
                max(ui_state.weight_sum_max, 1.0) * 2.0,
            )
            if changed:
                ui_state.weight_sum_max = max(float(ui_state.weight_sum_max), ui_state.weight_sum_min + 1e-6)
                range_dirty = True
        imgui.separator()
        imgui.text_wrapped(self.status)
        imgui.separator()
        imgui.text(self.timing_tree)
        imgui.end()
        return range_dirty

    def _present(self, result: ViewerRenderFrame, mode: DisplayMode, *, wait: bool) -> None:
        present_start = time.perf_counter()
        self.presenter.upload(result, mode, self.ui_state.mode_range(mode), wait=wait)
        self.presenter.timings.presenter_ms = (time.perf_counter() - present_start) * 1000.0
        self.last_result = result

    def _present_pending(self, *, wait: bool) -> None:
        if self.pending is not None:
            result, mode = self.pending
            self.pending = None
            self._present(result, mode, wait=wait)

    def _render_if_needed(self, range_dirty: bool) -> None:
        """Keep the GPU queue full across loop iterations.

        Each iteration finishes the frame launched in the previous one: reading
        back its sample count waits for the earlier field kernels, so the frame
        before it is packed into the PBO without a host wait, ahead of this
        frame's field kernels. The next frame's rays and trace are then queued
        behind those kernels before the GL upload, ImGui and swap run on the
        CPU. A frame reaches the screen two iterations after its camera was read.
        """
        state = self.state
        timings = self.presenter.timings
        timings.presenter_ms = 0.0
        render_start = time.perf_counter()
        finished = False
        if self.inflight is not None:
            prepared, self.inflight = self.inflight, None
            result = finish_viewer_frame(state, prepared, before_field=lambda: self._present_pending(wait=False))
            _schedule_display_buffers(result.display_buffer)
            self.pending = (result, prepared.display_mode)
            finished = True
        wants_frame = state.dirty or state.continuous or (self.last_result is None and self.pending is None)
        if wants_frame:
            self.inflight = prepare_viewer_frame(state)
            state.dirty = False
        if finished or wants_frame:
            timings.render_wall_ms = (time.perf_counter() - render_start) * 1000.0 - timings.presenter_ms
        else:
            timings.render_wall_ms = 0.0
        if not wants_frame and self.inflight is None:
            # Nothing new is coming: show the last frame now.
            if self.pending is not None:
                self._present_pending(wait=True)
            elif range_dirty and self.last_result is not None:
                self._present(self.last_result, state.display_mode, wait=True)

    def _draw_viewport(self) -> bool:
        imgui = self.imgui
        presenter = self.presenter
        self.gl.glClearColor(0.02, 0.02, 0.02, 1.0)
        self.gl.glClear(self.gl.GL_COLOR_BUFFER_BIT)
        hovered = False
        window_w, window_h = self.glfw.get_window_size(self.window)
        imgui.set_next_window_position(430, 10, condition=imgui.ALWAYS)
        imgui.set_next_window_size(max(400, window_w - 440), max(400, window_h - 20), condition=imgui.ALWAYS)
        viewport_flags = (
            imgui.WINDOW_NO_MOVE
            | imgui.WINDOW_NO_RESIZE
            | imgui.WINDOW_NO_COLLAPSE
            | imgui.WINDOW_NO_BRING_TO_FRONT_ON_FOCUS
        )
        imgui.begin("Viewport", True, flags=viewport_flags)
        if presenter.texture_id and presenter.width and presenter.height:
            avail_w, avail_h = imgui.get_content_region_available()
            scale = min(
                float(avail_w) / max(float(presenter.width), 1.0),
                float(avail_h) / max(float(presenter.height), 1.0),
                1.0,
            )
            draw_w = max(1.0, float(presenter.width) * scale)
            draw_h = max(1.0, float(presenter.height) * scale)
            self.state.viewport_size = (draw_w, draw_h)
            # Raygen and the PBO both store image rows top-to-bottom.
            imgui.image(presenter.texture_id, draw_w, draw_h, uv0=(0.0, 0.0), uv1=(1.0, 1.0))
            image_min = imgui.get_item_rect_min()
            if self.last_result is not None:
                _draw_imgui_bounds_overlay(
                    imgui,
                    self.state,
                    self.last_result.frame,
                    image_min,
                    (draw_w, draw_h),
                )
            hovered = imgui.is_item_hovered()
        imgui.end()
        return hovered

    def _submit_frame(self, frame_start: float) -> None:
        draw_start = time.perf_counter()
        self.imgui.render()
        width, height = self.glfw.get_framebuffer_size(self.window)
        self.gl.glViewport(0, 0, width, height)
        self.impl.render(self.imgui.get_draw_data())
        self.presenter.timings.imgui_draw_ms = (time.perf_counter() - draw_start) * 1000.0
        swap_start = time.perf_counter()
        self.glfw.swap_buffers(self.window)
        self.presenter.timings.swap_ms = (time.perf_counter() - swap_start) * 1000.0
        if self.timing_debug:
            sync_start = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.gl.glFinish()
            self.presenter.timings.sync_debug_ms = (time.perf_counter() - sync_start) * 1000.0
        else:
            self.presenter.timings.sync_debug_ms = 0.0
        self.presenter.timings.loop_total_ms = (time.perf_counter() - frame_start) * 1000.0
        self.status = _status_line(self.state, self.presenter.timings)
        if self.presenter.timings.render_wall_ms > 0.0:
            self.timing_tree = _timing_tree(self.state, self.presenter.timings)


def _schedule_display_buffers(buffer: DrRenderBuffer | None) -> None:
    """Queue evaluation of every displayable output without waiting for it."""
    if buffer is None:
        return
    import drjit as dr  # type: ignore[import-not-found]

    arrays: list[Any] = [*buffer.rgb_channels, buffer.depth]
    for value in buffer.diagnostics.values():
        arrays.extend(value if isinstance(value, tuple) else (value,))
    live = [array for array in arrays if array is not None]
    if live:
        dr.schedule(*live)
        dr.eval()


def run_gl_viewer(state: ViewerState, *, timing_debug: bool = False) -> None:
    glfw, imgui, GlfwRenderer, GL = _require_gl_stack()
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW.")
    window = None
    presenter: CudaGlPresenter | None = None
    impl = None
    try:
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        window = glfw.create_window(1920, 1080, "Adashell Viewer GL", None, None)
        if window is None:
            raise RuntimeError("Failed to create GLFW window.")
        glfw.make_context_current(window)
        glfw.swap_interval(0)
        imgui.create_context()
        impl = GlfwRenderer(window)
        presenter = CudaGlPresenter(gl=GL, ext=load_cuda_gl_extension())

        GlViewerSession(
            state=state,
            glfw=glfw,
            imgui=imgui,
            gl=GL,
            window=window,
            impl=impl,
            presenter=presenter,
            timing_debug=timing_debug,
        ).run()

    finally:
        if presenter is not None:
            presenter.close()
        if impl is not None:
            impl.shutdown()
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()


def run_interop_smoke() -> None:
    import drjit as dr
    import numpy as np

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for --interop-smoke.")
    glfw, _imgui, _GlfwRenderer, GL = _require_gl_stack()
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW.")
    window = None
    presenter: CudaGlPresenter | None = None
    try:
        glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
        window = glfw.create_window(16, 16, "interop smoke", None, None)
        if window is None:
            raise RuntimeError("Failed to create hidden GLFW window.")
        glfw.make_context_current(window)
        presenter = CudaGlPresenter(gl=GL, ext=load_cuda_gl_extension())
        rgb = (dr.full(dr.cuda.Float, 1.0, 256), dr.zeros(dr.cuda.Float, 256), dr.zeros(dr.cuda.Float, 256))
        dummy = ViewerRenderFrame(
            frame=type("_Frame", (), {"width": 16, "height": 16})(),
            profile={},
            details={},
            wall_ms=0.0,
            display_buffer=DrRenderBuffer(rgb_channels=rgb, num_rays=256, image_width=16, image_height=16),
        )
        presenter.upload(dummy, "RGB", (0.0, 1.0))
        GL.glBindTexture(GL.GL_TEXTURE_2D, presenter.texture_id)
        GL.glFinish()
        pixels = np.empty((16, 16, 4), dtype=np.uint8)
        GL.glGetTexImage(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, pixels)
        np.testing.assert_array_equal(pixels, np.broadcast_to([255, 0, 0, 255], pixels.shape))
        print(
            "interop_smoke ok "
            f"map={presenter.timings.interop_map_ms:.3f} "
            f"pack={presenter.timings.cuda_pack_ms:.3f} "
            f"unmap={presenter.timings.interop_unmap_ms:.3f} "
            f"gltex={presenter.timings.gl_tex_ms:.3f}"
        )
    finally:
        if presenter is not None:
            presenter.close()
        if window is not None:
            glfw.destroy_window(window)
        glfw.terminate()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Low-latency CUDA/OpenGL AdaShell viewer.")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to renderer config JSON. When use_fastpath=true, the RC fastpath preset may override config fields.",
    )
    parser.add_argument("--camera-index", type=int, default=None)
    parser.add_argument("--resolution-scale", type=float, default=1.0)
    parser.add_argument(
        "--display-mode",
        choices=["RGB", "sample_count", "depth", "pred_normal", "pred_gradient", "pred_gradient_norm", "kernel_size", "weight_sum"],
        default="RGB",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--interop-smoke", action="store_true")
    parser.add_argument("--print-build-env", action="store_true")
    parser.add_argument("--timing-debug", action="store_true", help="Synchronize CUDA/GL each frame to expose hidden timing stalls.")
    return parser.parse_args()


def _resolve_viewer_camera(
    config: RendererConfig,
    requested_camera_index: int | None,
    scene_bounds: torch.Tensor | None,
) -> tuple[CameraPath, int]:
    if config.assets.camera_path is not None:
        camera_path = load_camera_path(config.assets.camera_path)
        camera_index = config.render.camera_index if requested_camera_index is None else int(requested_camera_index)
    else:
        if requested_camera_index not in (None, 0):
            raise ValueError("The generated default camera only supports --camera-index 0.")
        if scene_bounds is None:
            raise ValueError(
                "Viewer config has no camera_path and no valid shell or field bounds "
                "from which to generate a default camera."
            )
        camera_path, required_far = create_default_camera_path(scene_bounds)
        config.render.ray_t_far = max(float(config.render.ray_t_far), float(required_far))
        camera_index = 0

    if not camera_path.frames:
        raise ValueError("Viewer camera path must contain at least one frame.")
    if camera_index < 0 or camera_index >= len(camera_path.frames):
        raise ValueError(f"Camera index out of range: {camera_index}, available=[0, {len(camera_path.frames) - 1}]")
    return camera_path, camera_index


def main() -> None:
    args = _parse_args()
    try:
        if args.print_build_env:
            print_build_environment()
            return
        if args.interop_smoke:
            run_interop_smoke()
            return
        if args.config is None:
            raise ValueError("--config is required unless --interop-smoke is used.")
        config = load_renderer_config(args.config)
        apply_fastpath_viewer_defaults(config)
        if args.device is not None:
            config.device = str(args.device)
        shell_bounds = load_shell_mesh_bounds(config)
        camera_path: CameraPath | None = None
        camera_index: int | None = None
        if config.assets.camera_path is not None:
            camera_path, camera_index = _resolve_viewer_camera(config, args.camera_index, scene_bounds=None)

        pipeline = build_runtime_pipeline(config)
        field_bounds = _field_aabb_from_pipeline(pipeline)
        if camera_path is None:
            scene_bounds = shell_bounds if shell_bounds is not None else field_bounds
            camera_path, camera_index = _resolve_viewer_camera(config, args.camera_index, scene_bounds)
        assert camera_path is not None and camera_index is not None
        object_bounds = shell_bounds if shell_bounds is not None else field_bounds
        object_center = None if object_bounds is None else (object_bounds[0] + object_bounds[1]) * 0.5
        state = ViewerState(
            config=config,
            camera_path=camera_path,
            pipeline=pipeline,
            camera_index=camera_index,
            resolution_scale=float(args.resolution_scale),
            display_mode=args.display_mode,
            object_center=object_center,
            field_bounds=field_bounds,
            shell_bounds=shell_bounds,
        )
        run_gl_viewer(state, timing_debug=bool(args.timing_debug))
    except RuntimeError as exc:
        print(f"adashell-viewer-gl error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
