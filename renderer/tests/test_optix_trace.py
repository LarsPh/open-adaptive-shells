import sys
import types
from pathlib import Path

import torch

from adashell_renderer.config import MeshAssetPaths, TraceConfig
from adashell_renderer.optix_trace import ProjectOptixRayTracer
from adashell_renderer.types import RayBatch


def test_project_optix_raytracer_initializes_slot_backed_members(monkeypatch, tmp_path: Path) -> None:
    dilation = tmp_path / "dilation.ply"
    erosion = tmp_path / "erosion.ply"
    dilation.write_text("ply", encoding="utf-8")
    erosion.write_text("ply", encoding="utf-8")

    fake_optix = types.SimpleNamespace(
        TraversableGraphFlags=types.SimpleNamespace(ALLOW_SINGLE_GAS=1),
        ExceptionFlags=types.SimpleNamespace(NONE=0),
        PipelineCompileOptions=lambda **kwargs: kwargs,
    )
    fake_trimesh = types.SimpleNamespace(load=lambda path, file_type=None: {"path": str(path), "file_type": file_type})

    monkeypatch.setitem(sys.modules, "optix", fake_optix)
    monkeypatch.setitem(sys.modules, "trimesh", fake_trimesh)
    monkeypatch.setattr("adashell_renderer.optix_trace.create_context", lambda log: "ctx")
    monkeypatch.setattr("adashell_renderer.optix_trace.create_module", lambda ctx, pipeline_options, source_file: "module")
    monkeypatch.setattr(
        "adashell_renderer.optix_trace.create_program_groups",
        lambda ctx, module: ("raygen", "miss", "hit"),
    )
    monkeypatch.setattr(
        "adashell_renderer.optix_trace.create_pipeline",
        lambda ctx, program_groups, pipeline_options: {"pipeline": True},
    )
    monkeypatch.setattr("adashell_renderer.optix_trace.create_sbt", lambda program_groups: {"sbt": True})
    monkeypatch.setattr(
        "adashell_renderer.optix_trace.create_acceleration_structure",
        lambda ctx, mesh: {"gas": mesh},
    )

    tracer = ProjectOptixRayTracer(
        mesh_assets=MeshAssetPaths(dilation_mesh=dilation, erosion_mesh=erosion),
        trace_config=TraceConfig(),
    )

    assert tracer.pipeline == {"pipeline": True}
    assert tracer.sbt == {"sbt": True}
    assert tracer.dilation_gas["gas"]["file_type"] == "ply"
    assert tracer.erosion_gas["gas"]["file_type"] == "ply"


def test_project_optix_trace_always_requests_metadata(monkeypatch) -> None:
    tracer = object.__new__(ProjectOptixRayTracer)
    tracer.trace_config = TraceConfig()
    tracer.strict_timing = False
    tracer.trace_blocking = False
    tracer.pipeline = "pipeline"
    tracer.sbt = "sbt"
    tracer.dilation_gas = "dilation"
    tracer.erosion_gas = "erosion"
    tracer._last_trace_profile = None
    captured: dict[str, object] = {}

    def fake_launch(*args, **kwargs):  # noqa: ANN002, ANN003
        del args
        captured.update(kwargs)
        hit_t = torch.full((2, 4), -1.0)
        hit_count = torch.tensor([1, 0], dtype=torch.int32)
        erosion_first = torch.full((2,), -1.0)
        flags = torch.zeros((2, 4), dtype=torch.int32)
        metadata = (
            torch.zeros((2, 2), dtype=torch.float32),
            torch.zeros((2, 2), dtype=torch.float32),
            torch.zeros((2,), dtype=torch.int32),
            torch.zeros((2,), dtype=torch.int32),
        )
        return torch.zeros((2,)), hit_t, hit_count, erosion_first, flags, metadata, {"return_to_torch_ms": 0.0}

    monkeypatch.setattr("adashell_renderer.optix_trace.launch_pipeline_ray_batch", fake_launch)
    rays = RayBatch(
        origins=torch.zeros((2, 3)),
        directions=torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]),
        t_near=torch.full((2,), 0.1),
        t_far=torch.full((2,), 1.0),
    )

    result = tracer.trace(rays)

    assert "trace_pack_metadata" not in captured
    assert result.metadata["trace_pack_backend"] == "optix_metadata"
    assert "optix_interval_starts" in result.metadata
