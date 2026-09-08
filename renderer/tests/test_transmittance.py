from types import SimpleNamespace

import numpy as np
import pytest


def test_transmittance_does_not_accumulate_error_from_previous_rays() -> None:
    cp = pytest.importorskip("cupy")
    dr = pytest.importorskip("drjit")
    if not dr.has_backend(dr.JitBackend.CUDA):
        pytest.skip("CUDA backend required")
    from adashell_renderer.drjit_backend.field_island import DrJitFieldIsland

    rays, samples = 100_000, 8
    alpha = dr.full(dr.cuda.Float, 0.75, rays * samples)
    ray_indices = dr.arange(dr.cuda.UInt, rays * samples) // samples
    offsets = dr.arange(dr.cuda.UInt, rays) * samples
    field = SimpleNamespace(_interop=SimpleNamespace(dr=dr))
    result = DrJitFieldIsland._exclusive_transmittance(
        field, alpha, {"ray_indices": ray_indices, "ray_offsets": offsets},
    )
    dr.eval(result)
    dr.sync_thread()
    actual = cp.asnumpy(cp.from_dlpack(result)).reshape(rays, samples)
    # Each ray has the same eight samples; unrelated rays cannot change its weights.
    expected = np.float64(1.0 - 0.75 + 1e-6) ** np.arange(samples)
    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), rtol=2e-5, atol=1e-7)
