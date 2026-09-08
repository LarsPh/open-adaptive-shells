from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adashell_renderer.drjit_backend.mlp import DrJitMLP, TorchLinearLayerSpec
from adashell_renderer.types import RayBatch


@pytest.fixture
def mlp():
    model = object.__new__(DrJitMLP)
    model.interop = SimpleNamespace(dr=SimpleNamespace(
        cuda=SimpleNamespace(Float=np.float32),
        width=len,
        zeros=lambda dtype, shape: np.zeros(shape, dtype=dtype),
        fma=lambda x, weight, acc: x * weight + acc,
    ))
    return model


@pytest.mark.parametrize("count", [0, 1, 3])
def test_fp32_linear_rejects_wrong_feature_count(mlp, count):
    spec = TorchLinearLayerSpec(torch.tensor([[2.0, 3.0]]), None)
    with pytest.raises(ValueError, match="Expected 2"):
        mlp._linear_columns_fp32(tuple(np.ones(4) for _ in range(count)), spec)


def test_fp32_linear_rejects_incompatible_widths(mlp):
    spec = TorchLinearLayerSpec(torch.tensor([[2.0, 3.0]]), None)
    with pytest.raises(ValueError, match="compatible sample widths"):
        mlp._linear_columns_fp32((np.ones(2), np.ones(3)), spec)


@pytest.mark.parametrize("with_bias", [False, True])
def test_fp32_linear_matches_affine_and_broadcasts_constants(mlp, with_bias):
    weight = torch.tensor([[2.0, 3.0], [-1.0, 4.0]])
    bias = torch.tensor([0.5, -0.5]) if with_bias else None
    spec = TorchLinearLayerSpec(weight, bias)
    cols = (np.array([5.0], dtype=np.float32), np.array([1.0, 2.0, 3.0], dtype=np.float32))
    result = np.stack(mlp._linear_columns_fp32(cols, spec), axis=-1)
    inputs = np.column_stack((np.full(3, 5.0), cols[1]))
    expected = inputs @ weight.numpy().T
    if bias is not None:
        expected += bias.numpy()
    np.testing.assert_allclose(result, expected)


def test_fp32_linear_accepts_empty_batch(mlp):
    spec = TorchLinearLayerSpec(torch.tensor([[2.0, 3.0]]), None)
    result = mlp._linear_columns_fp32((np.empty(0), np.empty(0)), spec)
    assert result[0].shape == (0,)


def test_ray_batch_has_no_cache_unsafe_slice_api():
    assert not hasattr(RayBatch, "slice")
    rays = RayBatch(torch.zeros(3, 3), torch.zeros(3, 3), torch.zeros(3), torch.ones(3))
    assert rays.num_rays == 3
