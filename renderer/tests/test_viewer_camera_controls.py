from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
import torch

from adashell_renderer.camera_path import CameraPath, CameraPathFrame, load_camera_path
from adashell_renderer.config_io import load_renderer_config
from adashell_renderer.config import RendererConfig
from adashell_renderer.viewer_support import ViewerCameraState, ViewerState, _camera_path_up


def _frame(*, roll: float = 0.0, elevation: float = 0.0, radius: float = 5.0) -> CameraPathFrame:
    c, s = math.cos(elevation), math.sin(elevation)
    rotation = np.asarray([[1.0, 0.0, 0.0], [0.0, c, s], [0.0, -s, c]])
    c, s = math.cos(roll), math.sin(roll)
    rotation = rotation @ np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    pose = np.column_stack((rotation, [0.0, radius * math.sin(elevation), radius * math.cos(elevation)]))
    return CameraPathFrame(torch.tensor(pose, dtype=torch.float32), 720.0, 610.0, 401.0, 299.0, 803, 601)


def _pose(camera: ViewerCameraState) -> np.ndarray:
    return camera.to_frame(1.0).camera_to_world.numpy()


def _project(frame: CameraPathFrame, point: np.ndarray) -> np.ndarray:
    pose = frame.camera_to_world.numpy()
    x, y, z = pose[:, :3].T @ (point - pose[:, 3])
    assert z < 0.0
    return np.asarray([frame.cx + frame.fx * x / -z, frame.cy - frame.fy * y / -z])


@pytest.mark.parametrize("roll", [0.0, 0.63, math.pi])
@pytest.mark.parametrize("target", [None, torch.tensor([1.0, -2.0, 0.4])])
def test_import_noop_and_reset_preserve_entire_pose(roll: float, target: torch.Tensor | None) -> None:
    frame = _frame(roll=roll, elevation=0.7)
    original = frame.camera_to_world.clone()
    camera = ViewerCameraState.from_frame(frame, target)
    camera.orbit(0.0, 0.0)
    camera.look(0.0, 0.0)
    camera.pan(0.0, 0.0)
    camera.zoom(0.0)
    camera.move_local()
    assert torch.equal(camera.to_frame(1.0).camera_to_world, original)
    camera.orbit(17.0, -23.0)
    camera.pan(9.0, 4.0)
    camera.zoom(2.0)
    camera.reset(frame, target)
    assert torch.equal(camera.to_frame(1.0).camera_to_world, original)
    assert torch.equal(frame.camera_to_world, original)


def test_pivot_selection_and_returned_frames_do_not_modify_pose() -> None:
    camera = ViewerCameraState.from_frame(_frame(roll=0.9))
    before = _pose(camera).copy()
    target = torch.tensor([1.0, 2.0, -3.0])
    camera.set_target(target)
    target.zero_()
    np.testing.assert_array_equal(_pose(camera), before)
    np.testing.assert_array_equal(camera.target.numpy(), [1.0, 2.0, -3.0])
    camera.to_frame(1.0).camera_to_world.zero_()
    np.testing.assert_array_equal(_pose(camera), before)
    assert camera.radius == pytest.approx(np.linalg.norm(before[:, 3] - camera.target.numpy()))


@pytest.mark.parametrize("dx,dy", [(12.0, 0.0), (-12.0, 0.0), (0.0, 12.0), (0.0, -12.0), (12.0, 9.0)])
@pytest.mark.parametrize("elevation", [0.0, 0.8, -0.8])
def test_orbit_drags_visible_object_surface_in_pointer_direction(dx: float, dy: float, elevation: float) -> None:
    frame = _frame(elevation=elevation)
    camera = ViewerCameraState.from_frame(frame, torch.zeros(3))
    # A visible surface point in front of the pivot, not the stationary pivot itself.
    point = frame.camera_to_world[:, 2].numpy()
    before = _project(frame, point)
    camera.orbit(dx, dy)
    movement = _project(camera.to_frame(1.0), point) - before
    for component, drag in zip(movement, (dx, dy)):
        if drag:
            assert component * drag > 0.0


def test_horizontal_orbit_is_world_up_turntable_even_with_roll() -> None:
    camera = ViewerCameraState.from_frame(_frame(roll=0.6, elevation=0.7), torch.zeros(3))
    before = _pose(camera)
    angle = -0.18
    c, s = math.cos(angle), math.sin(angle)
    expected = np.asarray([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]) @ before
    camera.orbit(30.0, 0.0)
    np.testing.assert_allclose(_pose(camera), expected, atol=3e-7)


def test_z_up_scene_horizontal_drag_does_not_roll_and_fit_preserves_orientation() -> None:
    frame = _frame()
    z_up = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    frame.camera_to_world = z_up @ frame.camera_to_world
    bounds = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])
    state = ViewerState(
        config=RendererConfig(), camera_path=CameraPath([frame]), pipeline=None,
        camera_index=0, resolution_scale=1.0, display_mode="RGB",
        object_center=torch.zeros(3), shell_bounds=bounds,
    )
    np.testing.assert_array_equal(state.camera.world_up, [0.0, 0.0, 1.0])
    state.camera.orbit(30.0, 0.0)
    pose = _pose(state.camera)
    np.testing.assert_allclose(pose[:, 1], [0.0, 0.0, 1.0], atol=1e-7)
    assert pose[2, 3] == pytest.approx(0.0)
    state.fit_camera()
    np.testing.assert_allclose(_pose(state.camera)[:, :3], pose[:, :3], atol=1e-7)
    state.camera.reset(frame, torch.zeros(3))
    np.testing.assert_array_equal(state.camera.world_up, [0.0, 0.0, 1.0])


@pytest.mark.parametrize("dx,dy", [(10.0, 0.0), (0.0, 12.0), (10.0, 12.0)])
def test_camera_controls_are_equivariant_to_scene_up_axis(dx: float, dy: float) -> None:
    frame = _frame(elevation=0.4)
    rotation = np.asarray([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    rotated_frame = _frame(elevation=0.4)
    rotated_frame.camera_to_world = torch.tensor(rotation, dtype=torch.float32) @ frame.camera_to_world
    y_camera = ViewerCameraState.from_frame(frame, torch.zeros(3))
    z_camera = ViewerCameraState.from_frame(rotated_frame, torch.zeros(3), world_up=np.asarray([0.0, 0.0, 1.0]))
    y_camera.orbit(dx, dy)
    z_camera.orbit(dx, dy)
    np.testing.assert_allclose(_pose(z_camera), rotation @ _pose(y_camera), atol=1e-7)
    y_camera.look(dx, dy)
    z_camera.look(dx, dy)
    np.testing.assert_allclose(_pose(z_camera), rotation @ _pose(y_camera), atol=1e-7)


@pytest.mark.parametrize("scene", ["khady", "horse"])
def test_demo_presets_infer_z_up_despite_uneven_pitch_distribution(scene: str) -> None:
    config = load_renderer_config(Path(__file__).resolve().parents[1] / f"repro/khady_stage2_infer_prednorm/{scene}.json")
    up = _camera_path_up(load_camera_path(config.assets.camera_path))
    assert up[2] / np.linalg.norm(up) > 0.999


def test_orbit_keeps_off_axis_pivot_at_same_projection_and_radius() -> None:
    frame = _frame(roll=0.4)
    target = torch.tensor([1.0, 0.7, 0.0])
    camera = ViewerCameraState.from_frame(frame, target)
    radius = camera.radius
    pixel = _project(frame, target.numpy())
    camera.orbit(20.0, 18.0)
    np.testing.assert_allclose(_project(camera.to_frame(1.0), target.numpy()), pixel, atol=3e-5)
    np.testing.assert_array_equal(camera.target.numpy(), target.numpy())
    assert camera.radius == pytest.approx(radius, rel=1e-12)


@pytest.mark.parametrize("invert_y", [False, True])
def test_look_rotates_at_stationary_position_and_moves_scene_opposite_pointer(invert_y: bool) -> None:
    frame = _frame()
    camera = ViewerCameraState.from_frame(frame, torch.zeros(3))
    before = _pose(camera).copy()
    pixel = _project(frame, np.zeros(3))
    camera.look(10.0, 10.0, invert_y=invert_y)
    after = _pose(camera)
    np.testing.assert_array_equal(after[:, 3], before[:, 3])
    assert not np.allclose(after[:, :3], before[:, :3])
    movement = _project(camera.to_frame(1.0), np.zeros(3)) - pixel
    assert movement[0] < 0.0
    assert movement[1] * (-1.0 if invert_y else 1.0) < 0.0
    np.testing.assert_allclose(camera.target.numpy(), after[:, 3] - after[:, 2] * 5.0, atol=1e-6)


def test_invert_y_changes_only_vertical_orbit_direction() -> None:
    cameras = [ViewerCameraState.from_frame(_frame(), torch.zeros(3)) for _ in range(2)]
    cameras[0].orbit(10.0, 12.0)
    cameras[1].orbit(10.0, 12.0, invert_y=True)
    a, b = (_pose(camera)[:, 3] for camera in cameras)
    np.testing.assert_allclose(a, b * [1.0, -1.0, 1.0], atol=1e-6)


@pytest.mark.parametrize("roll", [0.0, 0.7])
def test_pan_is_pixel_scaled_and_translates_camera_and_pivot_together(roll: float) -> None:
    frame = _frame(roll=roll)
    camera = ViewerCameraState.from_frame(frame, torch.zeros(3))
    before = _pose(camera).copy()
    pixel = _project(frame, np.zeros(3))
    camera.pan(23.0, -17.0)
    after = _pose(camera)
    np.testing.assert_array_equal(after[:, :3], before[:, :3])
    np.testing.assert_allclose(after[:, 3] - before[:, 3], camera.target.numpy(), atol=1e-6)
    np.testing.assert_allclose(_project(camera.to_frame(1.0), np.zeros(3)) - pixel, [23.0, -17.0], atol=1e-5)


def test_move_local_uses_imported_rolled_basis() -> None:
    camera = ViewerCameraState.from_frame(_frame(roll=0.8, elevation=0.5), torch.zeros(3))
    before = _pose(camera).copy()
    delta = before[:, :3] @ [0.2, -0.3, -0.7]
    camera.move_local(right=0.2, up=-0.3, forward=0.7)
    np.testing.assert_allclose(_pose(camera)[:, 3], before[:, 3] + delta, atol=3e-7)
    np.testing.assert_allclose(camera.target.numpy(), delta, atol=1e-7)
    np.testing.assert_array_equal(_pose(camera)[:, :3], before[:, :3])


@pytest.mark.parametrize("radius", [1e-5, 5.0, 1e5])
def test_zoom_is_relative_reversible_and_preserves_orientation(radius: float) -> None:
    camera = ViewerCameraState.from_frame(_frame(roll=0.7, radius=radius), torch.zeros(3))
    before = _pose(camera).copy()
    initial_radius = camera.radius
    camera.zoom(2.0)
    assert camera.radius == pytest.approx(initial_radius * math.exp(-0.16))
    np.testing.assert_array_equal(_pose(camera)[:, :3], before[:, :3])
    camera.zoom(-2.0)
    np.testing.assert_allclose(_pose(camera), before, atol=1e-7)


@pytest.mark.parametrize("sign", [-1.0, 1.0])
@pytest.mark.parametrize("method", ["orbit", "look"])
def test_imported_poles_preserve_pose_and_can_drag_back_into_range(sign: float, method: str) -> None:
    frame = _frame(elevation=sign * math.pi / 2.0)
    camera = ViewerCameraState.from_frame(frame, torch.zeros(3))
    np.testing.assert_array_equal(_pose(camera), frame.camera_to_world.numpy())
    rotate = getattr(camera, method)
    rotate(0.0, sign * 100.0)
    np.testing.assert_allclose(_pose(camera), frame.camera_to_world.numpy(), atol=1e-7)
    rotate(0.0, -sign * 1.0)
    pose = _pose(camera)
    np.testing.assert_allclose(pose[:, :3].T @ pose[:, :3], np.eye(3), atol=1e-6)
    assert np.isfinite(pose).all()
    assert np.linalg.norm(pose[:, :3] - frame.camera_to_world[:, :3].numpy()) < 0.01
    assert abs(float(pose[1, 2])) < 1.0


def test_large_vertical_drags_stop_before_pole_and_reverse_without_flipping() -> None:
    camera = ViewerCameraState.from_frame(_frame(), torch.zeros(3))
    camera.orbit(0.0, 10000.0)
    assert 0.0 < camera.pitch < math.pi / 2.0
    before = _pose(camera).copy()
    camera.orbit(0.0, 10000.0)
    np.testing.assert_allclose(_pose(camera), before, atol=1e-6)
    camera.orbit(20.0, -1.0)
    assert camera.pitch < math.pi / 2.0 - 0.001
    assert camera.radius == pytest.approx(5.0)
    np.testing.assert_allclose(_pose(camera)[:, :3].T @ _pose(camera)[:, :3], np.eye(3), atol=1e-6)


@pytest.mark.parametrize("method", ["orbit", "look", "pan"])
def test_viewport_mapping_has_identical_sensitivity_for_same_fractional_drag(method: str) -> None:
    frame = _frame(roll=0.3)
    cameras = [ViewerCameraState.from_frame(frame, torch.zeros(3)) for _ in range(2)]
    getattr(cameras[0], method)(24.0, 18.0)
    getattr(cameras[1], method)(12.0, 9.0, viewport_size=(frame.width / 2.0, frame.height / 2.0))
    np.testing.assert_array_equal(_pose(cameras[0]), _pose(cameras[1]))


@pytest.mark.parametrize("scale", [0.05, 0.333, 0.5, 1.0, 4.0])
def test_resolution_scale_preserves_pose_and_normalized_projection(scale: float) -> None:
    frame = _frame(roll=0.6)
    camera = ViewerCameraState.from_frame(frame, torch.zeros(3))
    scaled = camera.to_frame(scale)
    assert torch.equal(scaled.camera_to_world, frame.camera_to_world)
    point = np.asarray([0.4, 0.3, 0.0])
    np.testing.assert_allclose(
        _project(scaled, point) / [scaled.width, scaled.height],
        _project(frame, point) / [frame.width, frame.height], atol=1e-12,
    )
    camera.pan(10.0, 12.0)
    reference = ViewerCameraState.from_frame(frame, torch.zeros(3))
    reference.pan(10.0, 12.0)
    np.testing.assert_array_equal(_pose(camera), _pose(reference))


def test_coincident_pivot_falls_back_without_changing_pose() -> None:
    frame = _frame(roll=0.8)
    camera = ViewerCameraState.from_frame(frame, frame.camera_to_world[:, 3])
    assert camera.radius == pytest.approx(1.0)
    assert torch.equal(camera.to_frame(1.0).camera_to_world, frame.camera_to_world)
