from types import SimpleNamespace

import numpy as np
import pytest

from adashell_renderer.optix_utils import _to_cupy_array


class DeviceArray:
    def __init__(self, shape=(4, 3), dtype=np.float32, contiguous=True, device=0):
        self.shape = shape
        self.dtype = dtype
        self.flags = SimpleNamespace(c_contiguous=contiguous)
        self.device = SimpleNamespace(id=device)


CP = SimpleNamespace(
    ndarray=DeviceArray,
    float32=np.float32,
    cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: 0)),
)


def test_optix_ray_buffer_keeps_owner():
    array = DeviceArray()
    assert _to_cupy_array(CP, array, shape=(4, 3), name="origins") is array


@pytest.mark.parametrize("array, message", [
    (DeviceArray(shape=(4, 2)), "shape"),
    (DeviceArray(shape=(3, 3)), "shape"),
    (DeviceArray(dtype=np.float64), "float32"),
    (DeviceArray(contiguous=False), "contiguous"),
    (DeviceArray(device=1), "current CUDA device"),
])
def test_optix_rejects_invalid_ray_buffer(array, message):
    with pytest.raises(ValueError, match=message):
        _to_cupy_array(CP, array, shape=(4, 3), name="directions")


def test_optix_rejects_host_buffer():
    with pytest.raises(TypeError, match="CuPy"):
        _to_cupy_array(CP, np.zeros((4, 3)), shape=(4, 3), name="origins")


def test_optix_rejects_column_near_buffer():
    with pytest.raises(ValueError, match="t_near.*shape"):
        _to_cupy_array(CP, DeviceArray(shape=(4, 1)), shape=(4,), name="t_near")


def test_scratch_trace_buffers_reuse_until_shape_changes(monkeypatch):
    from adashell_renderer import optix_utils

    monkeypatch.setattr(optix_utils, "_SCRATCH_TRACE_BUFFERS", {})
    cp = SimpleNamespace(empty=np.empty, cuda=SimpleNamespace(runtime=SimpleNamespace(getDevice=lambda: 0)))
    first = optix_utils._scratch_trace_buffers(cp, 4, 20, 10)
    assert [a.shape for a in first] == [(4, 20), (4, 20), (4, 10), (4, 10)]
    assert [a.dtype for a in first] == [np.float32, np.int32, np.float32, np.float32]
    assert all(a is b for a, b in zip(first, optix_utils._scratch_trace_buffers(cp, 4, 20, 10)))
    resized = optix_utils._scratch_trace_buffers(cp, 8, 20, 10)
    assert resized[0].shape == (8, 20)
    assert len(optix_utils._SCRATCH_TRACE_BUFFERS) == 1
