from pathlib import Path


def test_project_source_has_no_legacy_runtime_imports() -> None:
    src_root = (Path(__file__).resolve().parents[1] / "src/adashell_renderer")
    offenders: list[str] = []
    for path in src_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "from nerfstudio" in text or "import nerfstudio" in text:
            offenders.append(str(path))
    assert offenders == []


def test_rc_source_has_no_mitsuba_or_dearpygui_runtime() -> None:
    src_root = (Path(__file__).resolve().parents[1] / "src/adashell_renderer")
    offenders: list[str] = []
    for path in src_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "mitsuba" in text.lower() or "dearpygui" in text.lower():
            offenders.append(str(path))
    assert offenders == []


def test_rc_fastpath_source_has_no_torch_trace_return_or_active_compaction() -> None:
    src_root = (Path(__file__).resolve().parents[1] / "src/adashell_renderer")
    forbidden = (
        "return_torch",
        "_cupy_to_torch",
        "metadata_only_torch_pack",
        "sample_counts_torch",
        "torch.nonzero(trace_hits",
        "index_select(0, active_ray_indices",
    )
    offenders: list[str] = []
    for path in src_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in forbidden):
            offenders.append(str(path))
    assert offenders == []


def test_gl_runtime_does_not_import_torch_camera_raygen() -> None:
    src_root = (Path(__file__).resolve().parents[1] / "src/adashell_renderer")
    forbidden = (
        "from adashell_renderer.camera_path import generate_rays",
        "torch.meshgrid",
    )
    offenders: list[str] = []
    for path in src_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in forbidden):
            offenders.append(str(path))
    assert offenders == []


def test_gl_fast_display_source_does_not_materialize_dr_rgb_to_torch() -> None:
    checked = [
        (Path(__file__).resolve().parents[1] / "src/adashell_renderer/viewer_gl.py"),
        (Path(__file__).resolve().parents[1] / "src/adashell_renderer/renderer_runtime.py"),
    ]
    offenders: list[str] = []
    for path in checked:
        text = path.read_text(encoding="utf-8")
        if ".torch()" in text or "torch.stack" in text:
            offenders.append(str(path))
    assert offenders == []


def test_gl_runtime_has_no_chunk_selector_or_public_tracer_bundle() -> None:
    checked = [
        (Path(__file__).resolve().parents[1] / "src/adashell_renderer/config.py"),
        (Path(__file__).resolve().parents[1] / "src/adashell_renderer/renderer_runtime.py"),
        (Path(__file__).resolve().parents[1] / "src/adashell_renderer/viewer_gl.py"),
        (Path(__file__).resolve().parents[1] / "src/adashell_renderer/viewer_support.py"),
        (Path(__file__).resolve().parents[1] / "src/adashell_renderer/drjit_backend/stage_adapter.py"),
    ]
    forbidden = (
        "eval_num_rays_per_chunk",
        "--chunk-size",
        "FastpathPipelineBundle",
        "def render_full_image(",
    )
    offenders: list[str] = []
    for path in checked:
        text = path.read_text(encoding="utf-8")
        if any(token in text for token in forbidden):
            offenders.append(str(path))
    assert offenders == []
