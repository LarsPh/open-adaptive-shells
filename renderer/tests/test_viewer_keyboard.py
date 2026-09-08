from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import adashell_renderer.viewer_gl as viewer_gl
from adashell_renderer.viewer_gl import GlViewerSession, _handle_keyboard_camera


def _keyboard(pressed: set[str]) -> SimpleNamespace:
    keys = ("W", "S", "A", "D", "Q", "E", "LEFT_SHIFT", "RIGHT_SHIFT")
    return SimpleNamespace(
        PRESS=1,
        **{f"KEY_{key}": key for key in keys},
        get_key=Mock(side_effect=lambda window, key: int(key in pressed)),
        poll_events=Mock(),
    )


def _state(*, mode: str = "Look", speed: float = 2.0) -> SimpleNamespace:
    return SimpleNamespace(
        rotate_mode=mode, move_speed=speed, dirty=False,
        camera=SimpleNamespace(move_local=Mock()),
    )


def test_held_w_moves_each_frame_and_release_stops() -> None:
    pressed = {"W"}
    glfw = _keyboard(pressed)
    state = _state()
    for _ in range(3):
        state.dirty = False
        _handle_keyboard_camera(glfw, None, state, True, 0.025)
        state.camera.move_local.assert_called_with(right=0.0, up=0.0, forward=0.05)
        assert state.dirty
    assert state.camera.move_local.call_count == 3
    pressed.clear()
    state.dirty = False
    _handle_keyboard_camera(glfw, None, state, True, 0.025)
    assert state.camera.move_local.call_count == 3
    assert not state.dirty


@pytest.mark.parametrize(
    "pressed,expected",
    [
        ({"W", "D", "E"}, (0.1, 0.1, 0.1)),
        ({"S", "A", "Q"}, (-0.1, -0.1, -0.1)),
        ({"W", "S", "D"}, (0.1, 0.0, 0.0)),
    ],
)
def test_combined_directions(pressed: set[str], expected: tuple[float, float, float]) -> None:
    state = _state()
    _handle_keyboard_camera(_keyboard(pressed), None, state, True, 0.05)
    right, up, forward = expected
    state.camera.move_local.assert_called_once_with(right=right, up=up, forward=forward)
    assert state.dirty


@pytest.mark.parametrize("shifts", [{"LEFT_SHIFT"}, {"RIGHT_SHIFT"}, {"LEFT_SHIFT", "RIGHT_SHIFT"}])
def test_shift_multiplies_speed_once(shifts: set[str]) -> None:
    state = _state()
    _handle_keyboard_camera(_keyboard({"W"} | shifts), None, state, True, 0.025)
    state.camera.move_local.assert_called_once_with(right=0.0, up=0.0, forward=0.2)


@pytest.mark.parametrize("dt_s", [0.0, -0.01])
def test_nonpositive_dt_does_not_move_or_dirty(dt_s: float) -> None:
    state = _state()
    _handle_keyboard_camera(_keyboard({"W"}), None, state, True, dt_s)
    state.camera.move_local.assert_not_called()
    assert not state.dirty


@pytest.mark.parametrize("dt_s", [0.1, 0.5, 10.0])
def test_dt_is_capped_at_one_tenth_second(dt_s: float) -> None:
    state = _state()
    _handle_keyboard_camera(_keyboard({"W"}), None, state, True, dt_s)
    state.camera.move_local.assert_called_once_with(right=0.0, up=0.0, forward=0.2)
    assert state.dirty


@pytest.mark.parametrize("hovered,mode", [(False, "Look"), (True, "Orbit"), (False, "Orbit")])
def test_inactive_camera_does_not_poll_or_move(hovered: bool, mode: str) -> None:
    glfw = _keyboard({"W"})
    state = _state(mode=mode)
    _handle_keyboard_camera(glfw, None, state, hovered, 0.05)
    glfw.get_key.assert_not_called()
    state.camera.move_local.assert_not_called()
    assert not state.dirty


@pytest.mark.parametrize("pressed", [set(), {"W", "S", "A", "D", "Q", "E"}])
def test_idle_or_opposing_keys_do_not_move_or_clear_existing_dirty(pressed: set[str]) -> None:
    state = _state()
    state.dirty = True
    _handle_keyboard_camera(_keyboard(pressed), None, state, True, 0.05)
    state.camera.move_local.assert_not_called()
    assert state.dirty


@pytest.mark.parametrize("want_text_input", [False, True])
def test_session_text_input_gates_real_keyboard_handler(
    monkeypatch: pytest.MonkeyPatch, want_text_input: bool,
) -> None:
    io = SimpleNamespace(want_text_input=want_text_input)
    state = _state()
    glfw = _keyboard({"W"})
    session = SimpleNamespace(
        last_loop_start=0.95,
        glfw=glfw, window=object(), state=state,
        imgui=SimpleNamespace(new_frame=Mock(), get_io=lambda: io),
        impl=SimpleNamespace(process_inputs=Mock()),
        presenter=SimpleNamespace(timings=SimpleNamespace()),
        _draw_controls=Mock(return_value=False),
        _render_if_needed=Mock(),
        _draw_viewport=Mock(return_value=True),
        _submit_frame=Mock(),
    )
    monkeypatch.setattr(viewer_gl.time, "perf_counter", lambda: 1.0)
    monkeypatch.setattr(viewer_gl, "_handle_mouse_camera", Mock())
    GlViewerSession._run_frame(session)
    if want_text_input:
        glfw.get_key.assert_not_called()
        state.camera.move_local.assert_not_called()
        assert not state.dirty
        # Releasing text ownership resumes held-key movement without extra state.
        io.want_text_input = False
        session.last_loop_start = 0.95
        GlViewerSession._run_frame(session)
    state.camera.move_local.assert_called_once()
    assert state.camera.move_local.call_args.kwargs == pytest.approx(
        {"right": 0.0, "up": 0.0, "forward": 0.1}
    )
    assert state.dirty
