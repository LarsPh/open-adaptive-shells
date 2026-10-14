from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import adashell_renderer.viewer_gl as viewer_gl_module
from adashell_renderer.camera_path import CameraPathFrame, create_default_camera_path
from adashell_renderer.config import RendererConfig
from adashell_renderer.cuda_raygen import generate_rays_cupy
from adashell_renderer.viewer_support import (
    ViewerCameraState,
    ViewerState,
    ViewerTimings,
    diagnostic_mode_for_display,
    project_debug_bounds_overlay_lines,
    scale_camera_frame,
)
from adashell_renderer.viewer_gl import (
    GlDisplayTimings,
    GlViewerSession,
    GlViewerUiState,
    _resolve_viewer_camera,
)
from reference_raygen import generate_rays_reference


def test_scale_camera_frame_scales_intrinsics_and_size() -> None:
    frame = CameraPathFrame(
        camera_to_world=torch.eye(4, dtype=torch.float32)[:3, :4],
        fx=100.0,
        fy=120.0,
        cx=50.0,
        cy=60.0,
        width=200,
        height=100,
    )

    scaled = scale_camera_frame(frame, 0.5)

    assert scaled.width == 100
    assert scaled.height == 50
    assert scaled.fx == 50.0
    assert scaled.fy == 60.0
    assert scaled.cx == 25.0
    assert scaled.cy == 30.0
    assert scaled.camera_to_world is frame.camera_to_world


@pytest.mark.parametrize("width,height", [(1280, 720), (720, 1280), (800, 800)])
def test_default_camera_fits_scene_bounds(width: int, height: int) -> None:
    bounds = torch.tensor([[-2.0, -1.0, -3.0], [4.0, 5.0, 1.0]], dtype=torch.float32)

    camera_path, required_far = create_default_camera_path(bounds, width=width, height=height)
    frame = camera_path.frames[0]
    center = (bounds[0] + bounds[1]) * 0.5
    expected_focal = (frame.height * 0.5) / math.tan(math.radians(50.0) * 0.5)

    assert len(camera_path.frames) == 1
    assert (frame.width, frame.height) == (width, height)
    assert frame.fx == pytest.approx(expected_focal)
    assert frame.fy == pytest.approx(expected_focal)
    assert (frame.cx, frame.cy) == (width * 0.5, height * 0.5)
    assert torch.allclose(frame.camera_to_world[:, :3], torch.eye(3))
    assert torch.allclose(frame.camera_to_world[:2, 3], center[:2])
    assert frame.camera_to_world[2, 3] > center[2]
    assert required_far > float(frame.camera_to_world[2, 3] - center[2])

    rotation = frame.camera_to_world[:, :3]
    position = frame.camera_to_world[:, 3]
    for x in (bounds[0, 0], bounds[1, 0]):
        for y in (bounds[0, 1], bounds[1, 1]):
            for z in (bounds[0, 2], bounds[1, 2]):
                local = rotation.T @ (torch.tensor([x, y, z]) - position)
                assert local[2] < 0.0
                pixel_x = frame.fx * float(local[0] / -local[2]) + frame.cx
                pixel_y = frame.cy - frame.fy * float(local[1] / -local[2])
                assert 0.0 <= pixel_x <= frame.width
                assert 0.0 <= pixel_y <= frame.height


def test_default_camera_rejects_degenerate_bounds() -> None:
    bounds = torch.ones((2, 3), dtype=torch.float32)

    with pytest.raises(ValueError, match="degenerate"):
        create_default_camera_path(bounds)


def test_default_camera_rejects_inverted_bounds() -> None:
    with pytest.raises(ValueError, match="ordered"):
        create_default_camera_path(torch.tensor([[1.0, -1.0, -1.0], [-1.0, 1.0, 1.0]]))


@pytest.mark.parametrize("use_shell", [True, False])
def test_fit_camera_uses_scene_bounds_and_preserves_intrinsics(use_shell: bool) -> None:
    bounds = torch.tensor([[-2.0, -1.0, -3.0], [4.0, 5.0, 1.0]])
    camera_path, _ = create_default_camera_path(bounds, width=720, height=1280)
    config = RendererConfig()
    config.render.ray_t_far = 1000.0
    state = ViewerState(
        config=config, camera_path=camera_path, pipeline=None,
        camera_index=0, resolution_scale=0.5, display_mode="RGB",
        shell_bounds=bounds if use_shell else None, field_bounds=bounds * 0.5 if use_shell else bounds,
    )
    state.camera.move_local(right=10.0, up=5.0)
    state.dirty = False

    state.fit_camera()

    fitted = state.camera.to_frame(1.0)
    torch.testing.assert_close(fitted.camera_to_world, camera_path.frames[0].camera_to_world)
    torch.testing.assert_close(state.camera.target, bounds.mean(dim=0))
    assert (fitted.width, fitted.height) == (720, 1280)
    assert fitted.fy == pytest.approx(camera_path.frames[0].fy)
    assert config.render.ray_t_far == 1000.0
    assert state.dirty is True


def test_default_camera_expands_far_plane_without_shrinking_it() -> None:
    bounds = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32)
    config = RendererConfig()
    config.render.ray_t_far = 1.0

    camera_path, camera_index = _resolve_viewer_camera(config, None, bounds)
    fitted_far = config.render.ray_t_far

    assert len(camera_path.frames) == 1
    assert camera_index == 0
    assert fitted_far > 1.0

    config.render.ray_t_far = 100.0
    _resolve_viewer_camera(config, None, bounds)
    assert config.render.ray_t_far == 100.0


def test_explicit_camera_path_keeps_config_and_cli_index_precedence(tmp_path: Path) -> None:
    camera_path_file = tmp_path / "viewer_cameras.json"
    camera_path_file.write_text(
        json.dumps(
            {
                "render_width": 64,
                "render_height": 32,
                "camera_path": [
                    {"fov": 50.0, "camera_to_world": torch.eye(4).reshape(-1).tolist()},
                    {"fov": 50.0, "camera_to_world": torch.eye(4).reshape(-1).tolist()},
                ],
            }
        ),
        encoding="utf-8",
    )
    config = RendererConfig()
    config.assets.camera_path = camera_path_file
    config.render.camera_index = 1

    _, config_index = _resolve_viewer_camera(config, None, scene_bounds=None)
    _, cli_index = _resolve_viewer_camera(config, 0, scene_bounds=None)

    assert config_index == 1
    assert cli_index == 0


def test_generated_camera_rejects_nonzero_index_and_missing_bounds() -> None:
    config = RendererConfig()

    with pytest.raises(ValueError, match="only supports --camera-index 0"):
        _resolve_viewer_camera(config, 1, torch.tensor([[-1.0] * 3, [1.0] * 3]))
    with pytest.raises(ValueError, match="no valid shell or field bounds"):
        _resolve_viewer_camera(config, None, scene_bounds=None)


def test_cupy_raygen_rejects_cpu_device() -> None:
    c2w = torch.eye(4, dtype=torch.float32)[:3, :4]
    c2w[:, 3] = torch.tensor([0.2, -0.1, 3.0])
    frame = CameraPathFrame(c2w, fx=80.0, fy=90.0, cx=2.0, cy=1.5, width=4, height=3)

    with pytest.raises(RuntimeError, match="CuPy ray generation"):
        generate_rays_cupy(frame, device="cpu", camera_index=7, t_near=0.05, t_far=4.0, pixel_center_offset=0.5)


def test_cupy_raygen_matches_reference_generate_rays_when_cuda_is_available() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for CuPy raygen parity.")
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("CuPy is required for CuPy raygen parity.")

    c2w = torch.eye(4, dtype=torch.float32)[:3, :4]
    c2w[:, 3] = torch.tensor([0.2, -0.1, 3.0])
    frame = CameraPathFrame(c2w, fx=80.0, fy=90.0, cx=2.0, cy=1.5, width=4, height=3)

    generated = generate_rays_cupy(frame, device="cuda:0", camera_index=7, t_near=0.05, t_far=4.0, pixel_center_offset=0.5)
    reference = generate_rays_reference(frame, device="cuda:0", camera_index=7, t_near=0.05, t_far=4.0, pixel_center_offset=0.5)

    assert generated.metadata["raygen_backend"] == "cupy_rawkernel"
    assert torch.allclose(torch.utils.dlpack.from_dlpack(generated.origins), reference.origins)
    assert torch.allclose(torch.utils.dlpack.from_dlpack(generated.directions), reference.directions, atol=1e-6)
    assert torch.allclose(torch.utils.dlpack.from_dlpack(generated.t_near), reference.t_near)
    assert torch.allclose(torch.utils.dlpack.from_dlpack(generated.t_far), reference.t_far)
    cp.get_default_memory_pool().free_all_blocks()


def test_viewer_camera_uses_explicit_orbit_target() -> None:
    c2w = torch.eye(4, dtype=torch.float32)[:3, :4]
    c2w[:, 3] = torch.tensor([0.0, 0.0, 5.0])
    frame = CameraPathFrame(c2w, fx=1.0, fy=1.0, cx=0.5, cy=0.5, width=2, height=2)
    target = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)

    camera = ViewerCameraState.from_frame(frame, target=target)

    assert torch.allclose(camera.target, target)
    assert camera.radius > 0.0


def test_viewer_camera_look_rotates_in_place() -> None:
    c2w = torch.eye(4, dtype=torch.float32)[:3, :4]
    c2w[:, 3] = torch.tensor([0.0, 0.0, 5.0])
    frame = CameraPathFrame(c2w, fx=1.0, fy=1.0, cx=0.5, cy=0.5, width=2, height=2)
    camera = ViewerCameraState.from_frame(frame, target=torch.zeros((3,), dtype=torch.float32))
    before_position = camera.to_frame(1.0).camera_to_world[:, 3].clone()

    camera.look(dx=20.0, dy=10.0)
    after_position = camera.to_frame(1.0).camera_to_world[:, 3]

    assert torch.allclose(after_position, before_position, atol=1e-6)
    assert not torch.allclose(camera.target, torch.zeros((3,), dtype=torch.float32))


def test_viewer_camera_orbit_horizontal_drag_changes_position_without_roll() -> None:
    c2w = torch.eye(4, dtype=torch.float32)[:3, :4]
    c2w[:, 3] = torch.tensor([0.0, 0.0, 5.0])
    frame = CameraPathFrame(c2w, fx=1.0, fy=1.0, cx=0.5, cy=0.5, width=2, height=2)
    camera = ViewerCameraState.from_frame(frame, target=torch.zeros((3,), dtype=torch.float32))

    camera.orbit(dx=30.0, dy=0.0)
    after = camera.to_frame(1.0).camera_to_world

    assert after[0, 3] < 0.0
    assert torch.allclose(after[:, 1], torch.tensor([0.0, 1.0, 0.0]), atol=1e-5)


def test_viewer_camera_look_move_local_moves_camera_and_target() -> None:
    c2w = torch.eye(4, dtype=torch.float32)[:3, :4]
    c2w[:, 3] = torch.tensor([0.0, 0.0, 5.0])
    frame = CameraPathFrame(c2w, fx=1.0, fy=1.0, cx=0.5, cy=0.5, width=2, height=2)
    camera = ViewerCameraState.from_frame(frame, target=torch.zeros((3,), dtype=torch.float32))
    before_position = camera.to_frame(1.0).camera_to_world[:, 3].clone()

    camera.move_local(forward=1.0, right=0.5, up=-0.25)
    after_frame = camera.to_frame(1.0)

    assert not torch.allclose(after_frame.camera_to_world[:, 3], before_position)
    assert torch.allclose(after_frame.camera_to_world[:, 3] - before_position, camera.target - torch.zeros((3,), dtype=torch.float32), atol=1e-5)


def test_display_mode_maps_to_fast_diagnostic_request() -> None:
    assert diagnostic_mode_for_display("RGB") == "none"
    assert diagnostic_mode_for_display("depth") == "none"
    assert diagnostic_mode_for_display("sample_count") == "none"
    assert diagnostic_mode_for_display("pred_gradient_norm") == "pred_gradient_norm"


def test_bounds_overlay_projection_returns_finite_cpu_lines() -> None:
    c2w = torch.eye(4, dtype=torch.float32)[:3, :4]
    c2w[:, 3] = torch.tensor([0.0, 0.0, 5.0])
    frame = CameraPathFrame(c2w, fx=40.0, fy=40.0, cx=10.0, cy=10.0, width=20, height=20)
    bounds = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32)

    lines = project_debug_bounds_overlay_lines(frame, field_bounds=bounds, shell_bounds=None)

    assert len(lines) > 0
    for line in lines:
        assert all(torch.isfinite(torch.tensor([*line.start, *line.end])).tolist())
        assert line.color == (255, 8, 8, 255)


def test_gl_viewer_ui_state_ranges_are_persistent() -> None:
    ui = GlViewerUiState(sample_min=1, sample_max=9, depth_min=0.2, depth_max=3.0)

    assert ui.mode_range("sample_count") == (1.0, 9.0)
    assert ui.mode_range("depth") == (0.2, 3.0)
    assert ui.mode_range("kernel_size") == (0.0, 0.2)


class FakeSessionPresenter:
    def __init__(self) -> None:
        self.timings = GlDisplayTimings()
        self.uploads: list[tuple[object, str, tuple[float, float]]] = []
        self.waits: list[bool] = []
        self.texture_id = 0
        self.width = 0
        self.height = 0

    def upload(self, result, mode, value_range, wait=True):  # noqa: ANN001
        self.uploads.append((result, mode, value_range))
        self.waits.append(wait)


class FakeViewerCamera:
    def __init__(self) -> None:
        self.resets: list[tuple[object, object]] = []

    def reset(self, frame, target=None):  # noqa: ANN001
        self.resets.append((frame, target))


class FakeControlsImgui:
    FIRST_USE_EVER = 1
    ALWAYS = 2
    WINDOW_NO_MOVE = 4
    WINDOW_NO_RESIZE = 8
    WINDOW_NO_COLLAPSE = 16

    def set_next_window_position(self, *_args, **_kwargs):
        pass

    def set_next_window_size(self, *_args, **_kwargs):
        pass

    def begin(self, *_args, **_kwargs):
        pass

    def combo(self, label, current, _items):  # noqa: ANN001
        selections = {"Camera": 1, "View": 1, "Rotate": 1}
        return True, selections.get(label, current)

    def checkbox(self, label, value):  # noqa: ANN001
        values = {"Continuous": True, "Bounds": True, "Invert Y": True}
        return True, values.get(label, value)

    def slider_float(self, label, value, _minimum, _maximum):  # noqa: ANN001
        values = {"Move speed": 2.0, "Resolution": 0.5}
        return (label in values), values.get(label, value)

    def slider_int(self, label, value, _minimum, _maximum):  # noqa: ANN001
        values = {"Sample min": 2, "Sample max": 8}
        return True, values.get(label, value)

    def button(self, label):
        return label == "Reset"

    def same_line(self):
        pass

    def separator(self):
        pass

    def text(self, _value):
        pass

    def text_wrapped(self, _value):
        pass

    def end(self):
        pass


def _make_gl_viewer_session(*, dirty: bool = True, continuous: bool = False, imgui=None):
    camera = FakeViewerCamera()
    state = SimpleNamespace(
        config=SimpleNamespace(
            trace=SimpleNamespace(n_sample_max=4, n_depth_peeling_max=3),
            render=SimpleNamespace(ray_t_near=0.05, ray_t_far=4.0),
        ),
        camera_path=SimpleNamespace(frames=["frame-0", "frame-1"]),
        camera_index=0,
        display_mode="RGB",
        rotate_mode="Orbit",
        continuous=continuous,
        show_bounds_overlay=False,
        invert_y=False,
        move_speed=1.0,
        resolution_scale=1.0,
        object_center="center",
        camera=camera,
        dirty=dirty,
    )
    presenter = FakeSessionPresenter()
    session = GlViewerSession(
        state=state,
        glfw=SimpleNamespace(get_window_size=lambda _: (1920, 1080)),
        imgui=imgui or object(),
        gl=object(),
        window=object(),
        impl=object(),
        presenter=presenter,
    )
    return session, state, presenter, camera


def test_gl_viewer_session_shows_each_frame_while_the_next_one_renders(monkeypatch) -> None:
    rendered = [SimpleNamespace(display_buffer=None) for _ in range(2)]
    events: list[object] = []

    def fake_render(state, before_field=None):  # noqa: ANN001
        # The pipeline calls before_field after the sample readback, before the field kernels.
        before_field()
        events.append(("field", len([e for e in events if e[0] == "field"])))
        return rendered[len([e for e in events if e[0] == "field"]) - 1]

    monkeypatch.setattr(viewer_gl_module, "render_viewer_frame", fake_render)
    session, state, presenter, _camera = _make_gl_viewer_session()

    session._render_if_needed(range_dirty=False)
    assert presenter.uploads == []
    assert session.pending == (rendered[0], "RGB")
    assert state.dirty is False

    state.continuous = True
    session._render_if_needed(range_dirty=False)
    # Frame 0 is packed without a host wait, inside frame 1's render, before its field kernels.
    assert presenter.uploads == [(rendered[0], "RGB", (0.0, 1.0))]
    assert presenter.waits == [False]
    assert session.last_result is rendered[0]
    assert session.pending == (rendered[1], "RGB")

    state.continuous = False
    session._render_if_needed(range_dirty=False)
    # With nothing new to render, the pending frame is shown at once and waits for its kernels.
    assert presenter.uploads[-1][0] is rendered[1]
    assert presenter.waits[-1] is True
    assert session.pending is None
    assert len([e for e in events if e[0] == "field"]) == 2


def test_gl_viewer_session_reuses_last_result_for_range_only_update(monkeypatch) -> None:
    monkeypatch.setattr(
        viewer_gl_module,
        "render_viewer_frame",
        lambda _state, before_field=None: pytest.fail("range-only update must not render"),
    )
    session, _state, presenter, _camera = _make_gl_viewer_session(dirty=False)
    previous = object()
    session.last_result = previous

    session._render_if_needed(range_dirty=True)

    assert presenter.uploads == [(previous, "RGB", (0.0, 1.0))]
    assert session.presenter.timings.render_wall_ms == 0.0


def test_gl_viewer_session_idle_frame_clears_render_timings(monkeypatch) -> None:
    monkeypatch.setattr(
        viewer_gl_module,
        "render_viewer_frame",
        lambda _state, before_field=None: pytest.fail("idle update must not render"),
    )
    session, _state, presenter, _camera = _make_gl_viewer_session(dirty=False)
    session.last_result = object()
    presenter.timings.render_wall_ms = 3.0
    presenter.timings.presenter_ms = 2.0

    session._render_if_needed(range_dirty=False)

    assert presenter.uploads == []
    assert presenter.timings.render_wall_ms == 0.0
    assert presenter.timings.presenter_ms == 0.0


def test_gl_viewer_session_controls_write_back_viewer_state() -> None:
    imgui = FakeControlsImgui()
    session, state, _presenter, camera = _make_gl_viewer_session(imgui=imgui)

    range_dirty = session._draw_controls()

    assert range_dirty is True
    assert state.camera_index == 1
    assert state.display_mode == "sample_count"
    assert state.rotate_mode == "Look"
    assert state.continuous is True
    assert state.show_bounds_overlay is True
    assert state.invert_y is True
    assert state.move_speed == 2.0
    assert state.resolution_scale == 0.5
    assert state.dirty is True
    assert session.ui_state.sample_min == 2
    assert session.ui_state.sample_max == 8
    assert camera.resets == [("frame-1", "center"), ("frame-1", "center")]


def test_viewer_timings_reports_fps_from_displayed_total() -> None:
    timings = ViewerTimings(total_wall_ms=15.0, texture_upload_ms=5.0)

    assert timings.displayed_total_ms == 20.0
    assert timings.displayed_fps == 50.0
