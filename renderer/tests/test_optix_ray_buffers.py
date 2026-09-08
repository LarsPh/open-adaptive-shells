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
