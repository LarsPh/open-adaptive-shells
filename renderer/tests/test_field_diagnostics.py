from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adashell_renderer.drjit_backend.field_island import DrJitFieldIsland
from adashell_renderer.types import DrRenderBuffer


@pytest.mark.parametrize("counts", ([2, 0, 1], [0, 0, 0]))
def test_render_buffer_exports_full_frame_sample_counts_without_extra_queries(counts):
    result = DrRenderBuffer(rgb_channels=(None, None, None), num_rays=len(counts), image_width=len(counts), image_height=1)
    queries = []

    def query(rays, packed):
        queries.append(packed.valid_samples)
        return (None, None, None), (None, None, None)

    field = SimpleNamespace(
        _require_supported=lambda *args: None,
        _new_render_profile=dict,
        _prewarm_device=lambda *args: None,
        _render_rgb_core_dr=lambda **kwargs: result,
        _interop=SimpleNamespace(dr=SimpleNamespace(cuda=SimpleNamespace(Float=lambda x: np.asarray(x, dtype=np.float32)))),
        field_model=SimpleNamespace(_packed_query_dr_cache=query),
    )
    packed = SimpleNamespace(
        num_rays=len(counts), valid_samples=sum(counts), eval_t=torch.empty(sum(counts)),
        metadata={"_drjit_cache": {"sample_counts": counts}},
    )
    actual = DrJitFieldIsland.render_rgb_buffer(field, SimpleNamespace(metadata={}), packed, diagnostic_mode="sample_count")

    np.testing.assert_array_equal(actual.diagnostics["sample_count"], counts)
    assert actual.diagnostics["sample_count"].dtype == np.float32
    assert actual.metadata["diagnostics"] is actual.diagnostics
    assert queries == ([sum(counts)] if sum(counts) else [])
