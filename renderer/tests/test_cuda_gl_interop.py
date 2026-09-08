from __future__ import annotations

import os

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for Dr/CuPy DLPack smoke.")
def test_dr_rgb_channels_can_be_viewed_by_cupy_dlpack() -> None:
    try:
        import cupy as cp  # type: ignore[import-not-found]
        import drjit as dr  # type: ignore[import-not-found]
    except ImportError:
        pytest.skip("CuPy and Dr.Jit are required for Dr/CuPy DLPack smoke.")

    r_dr = dr.cuda.Float([0.0, 0.5, 1.0, 1.5])
    g_dr = dr.cuda.Float([1.0, 0.5, 0.0, -1.0])
    b_dr = dr.cuda.Float([0.25, 0.5, 0.75, 1.0])
    dr.eval(r_dr, g_dr, b_dr)
    dr.sync_thread()

    r_cp = cp.from_dlpack(r_dr)
    g_cp = cp.from_dlpack(g_dr)
    b_cp = cp.from_dlpack(b_dr)

    assert r_cp.shape == (4,)
    assert r_cp.dtype == cp.float32
    assert int(r_cp.data.ptr) != 0
    assert int(g_cp.data.ptr) != 0
    assert int(b_cp.data.ptr) != 0
    assert cp.asnumpy(r_cp).tolist() == pytest.approx([0.0, 0.5, 1.0, 1.5])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for CUDA/GL pack smoke.")
def test_cuda_gl_pack_rgb_opt_in() -> None:
    if os.environ.get("ADASHELL_TEST_CUDA_GL_EXTENSION") != "1":
        pytest.skip("Set ADASHELL_TEST_CUDA_GL_EXTENSION=1 to JIT-compile the CUDA/GL extension.")
    from adashell_renderer.cuda_gl_interop import pack_rgb_ptrs_to_rgba8, pack_rgb_to_rgba8, pack_scalar_to_rgba8, pack_vector_to_rgba8

    rgb = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.5, 1.0],
            [2.0, -1.0, 0.25],
            [0.1, 0.2, 0.3],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    rgba = pack_rgb_to_rgba8(rgb, 2, 2)
    torch.cuda.synchronize()

    assert tuple(rgba.shape) == (2, 2, 4)
    assert rgba.dtype == torch.uint8
    assert rgba.reshape(-1, 4)[0].detach().cpu().tolist() == [255, 0, 0, 255]

    r = rgb[:, 0].contiguous()
    g = rgb[:, 1].contiguous()
    b = rgb[:, 2].contiguous()
    ptr_rgba = pack_rgb_ptrs_to_rgba8(int(r.data_ptr()), int(g.data_ptr()), int(b.data_ptr()), 2, 2)
    torch.cuda.synchronize()
    assert torch.equal(ptr_rgba, rgba)

    scalar = torch.tensor([0.0, 0.5, 1.0, 2.0], device="cuda", dtype=torch.float32)
    scalar_rgba = pack_scalar_to_rgba8(scalar, 2, 2, 0.0, 1.0)
    scalar_rgba_shifted = pack_scalar_to_rgba8(scalar, 2, 2, 0.0, 2.0)
    torch.cuda.synchronize()

    assert tuple(scalar_rgba.shape) == (2, 2, 4)
    assert scalar_rgba.dtype == torch.uint8
    assert not torch.equal(scalar_rgba, scalar_rgba_shifted)

    vectors = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [10.0, 0.0, 0.0],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    vector_rgba = pack_vector_to_rgba8(vectors, 2, 2, normalize=True)
    torch.cuda.synchronize()

    assert tuple(vector_rgba.shape) == (2, 2, 4)
    assert vector_rgba.dtype == torch.uint8
    assert vector_rgba.reshape(-1, 4)[0].detach().cpu().tolist() == [255, 128, 128, 255]
    assert torch.equal(vector_rgba.reshape(-1, 4)[0], vector_rgba.reshape(-1, 4)[3])
