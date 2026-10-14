from types import SimpleNamespace

import numpy as np
import pytest
import torch

from adashell_renderer.profiling import ProfileBook
from adashell_renderer.renderer_pipeline import RendererPipeline


@pytest.fixture(autouse=True)
def fake_cupy(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "cupy", SimpleNamespace(
        ndarray=np.ndarray, asnumpy=np.asarray, count_nonzero=np.count_nonzero,
    ))


@pytest.mark.parametrize("counts", ([2, 0, 1], [0, 0, 0]))
@pytest.mark.parametrize("collect_stats", (False, True))
def test_direct_dispatch_preserves_buffers_and_profile(counts, collect_stats):
    events = []
    rays = SimpleNamespace(num_rays=3, metadata={"user": "keep"})
    original_metadata = dict(rays.metadata)
    hits = SimpleNamespace(metadata={"optix_sample_counts": np.asarray(counts)})
    packed = SimpleNamespace(valid_samples=sum(counts), metadata={"budget_per_ray": 8.0})
    result = SimpleNamespace(metadata={"profile_stage_ms": {"field_eval_ms": 2.0}}, diagnostics={})

    def trace(actual_rays):
        assert actual_rays is rays
        events.append("trace")
        return hits

    def sample(actual_rays, actual_hits):
        assert actual_rays is rays and actual_hits is hits
        events.append("sample")
        return packed

    def render(actual_rays, actual_pack, **kwargs):
        assert actual_rays is rays and actual_pack is packed
        assert rays.metadata == original_metadata
        assert kwargs == {"materialize_aux_outputs": True, "diagnostic_mode": "sample_count"}
        events.append("field")
        return result

    pipeline = RendererPipeline(
        SimpleNamespace(trace=trace),
        SimpleNamespace(sample=sample),
        SimpleNamespace(render_rgb_buffer=render),
        fast_render_aux_outputs=True,
        fast_render_diagnostic_mode="sample_count",
        collect_stats=collect_stats,
    )
    profile = ProfileBook()
    assert pipeline.run_render_buffer(rays, profile) is result
    assert events == ["trace", "sample", "field"]
    assert rays.metadata == original_metadata
    assert result.metadata["packed"] is packed.metadata
    assert result.metadata["trace"] is hits.metadata
    assert result.metadata["diagnostics"] is result.diagnostics
    active = float(sum(c > 0 for c in counts))
    expected = {"num_rays": 3.0, "valid_samples": float(sum(counts))}
    if collect_stats:
        expected["active_rays"] = active
    assert profile.counters == expected
    assert packed.metadata["sample_budget"] == 24.0
    assert packed.metadata.get("active_sample_budget") == (active * 8.0 if collect_stats else None)
    assert set(profile.stage_ms) == {"trace_ms", "sample_ms", "field_island_ms", "pipeline_wall_ms"}
    assert result.metadata["profile_stage_ms"] == {"field_eval_ms": 2.0}
    assert not hasattr(pipeline, "run")
    assert not hasattr(pipeline, "geo_stage")


def test_field_failure_propagates():
    rays = SimpleNamespace(num_rays=1, metadata={})
    hits = SimpleNamespace(metadata={"optix_sample_counts": np.asarray([1])})
    packed = SimpleNamespace(valid_samples=1, metadata={})
    error = RuntimeError("field failed")

    def fail(*args, **kwargs):
        raise error

    pipeline = RendererPipeline(
        SimpleNamespace(trace=lambda rays: hits),
        SimpleNamespace(sample=lambda rays, hits: packed),
        SimpleNamespace(render_rgb_buffer=fail),
    )
    with pytest.raises(RuntimeError, match="field failed") as raised:
        pipeline.run_render_buffer(rays)
    assert raised.value is error


def test_active_count_rejects_torch_metadata():
    pipeline = RendererPipeline(None, None, None)
    hits = SimpleNamespace(metadata={"optix_sample_counts": torch.tensor([1])})
    with pytest.raises(TypeError, match="CuPy arrays"):
        pipeline._count_active_metadata_rays(hits)


def test_builder_constructs_only_runtime_dependencies(monkeypatch):
    import importlib.util
    import sys
    from pathlib import Path
    from types import ModuleType

    from adashell_renderer.config import RendererConfig
    from adashell_renderer import renderer_pipeline

    events = []
    field = object()
    tracer = SimpleNamespace(trace=object())
    island = object()
    sampler = object()

    def make_tracer(config):
        events.append("tracer")
        return tracer

    def load_field(*, path, config, device):
        events.append("checkpoint")
        return field

    def make_sampler(config, *, capture_kernel_history, host_sync):
        assert host_sync is False
        events.append("sampler")
        return sampler

    def make_island(*, field_model, alpha_mode, white_background, profile_geo, prewarm):
        assert field_model is field
        assert alpha_mode == "pred_gradient"
        events.append("island")
        return island

    dependencies = {
        "drjit_backend.field_island": {"DrJitFieldIsland": make_island},
        "drjit_backend.optix_metadata_sampler": {"OptixMetadataSamplerCore": make_sampler},
        "legacy_field": {"build_legacy_field_from_checkpoint": load_field},
        "trace_backend": {"build_project_ray_tracer": make_tracer},
    }
    for name, exports in dependencies.items():
        module = ModuleType("adashell_renderer." + name)
        module.__dict__.update(exports)
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    path = Path(renderer_pipeline.__file__).parent / "drjit_backend" / "stage_adapter.py"
    spec = importlib.util.spec_from_file_location("_test_stage_adapter", path)
    adapter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(adapter)
    config = RendererConfig()
    config.assets.checkpoint = Path("fake-checkpoint.pt")
    config.use_fastpath = True
    pipeline = adapter.build_fastpath_pipeline(config)
    assert pipeline.trace_stage is tracer
    assert pipeline.sampler is sampler
    assert pipeline.field_island is island
    assert events == ["tracer", "checkpoint", "sampler", "island"]
    assert pipeline.collect_stats is False
