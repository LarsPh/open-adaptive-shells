import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from adashell_renderer.camera_path import CameraPathFrame
from adashell_renderer.viewer_support import ViewerCameraState


spec = importlib.util.spec_from_file_location("record_viewer", Path(__file__).parents[1] / "scripts/record_viewer.py")
recording = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recording)


def test_tour_moves_continuously_and_keeps_look_position():
    pose = torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 5.]])
    frame = CameraPathFrame(pose, 1000., 1000., 960., 540., 1920, 1080)
    target = torch.zeros(3)
    state = SimpleNamespace(camera=ViewerCameraState.from_frame(frame, target))

    def sample(seconds):
        recording.apply_camera_tour(state, frame, target, seconds, 60.)
        return state.camera.to_frame(1.).camera_to_world.numpy().copy()

    for seconds in (1., 19., 25., 31., 37., 43., 49., 55.):
        a, b = sample(seconds), sample(seconds + 1 / 30)
        assert not np.array_equal(a, b)
        assert np.max(np.abs(a - b)) < 0.15
        assert state.dirty and state.continuous
    a, b = sample(31.), sample(33.)
    np.testing.assert_allclose(a[:, 3], b[:, 3])
    assert state.rotate_mode == "Look"
    for seconds, mode in ((37., "pred_normal"), (43., "sample_count"), (49., "pred_gradient_norm"), (55., "depth")):
        sample(seconds)
        assert state.display_mode == mode
    np.testing.assert_array_equal(sample(3.), sample(3.))
