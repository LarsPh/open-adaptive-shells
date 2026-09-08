from __future__ import annotations

import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

from adashell_renderer.config import (
    RendererConfig,
    MeshAssetPaths,
    ProfilingConfig,
    RenderConfig,
    RendererFieldConfig,
    TraceConfig,
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _path_or_none(value: str | None, root: Path) -> Path | None:
    if value is None:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    return (root / candidate).resolve()


def _dataclass_payload(cls: type, payload: dict[str, Any]) -> dict[str, Any]:
    allowed = {item.name for item in fields(cls)}
    return {key: value for key, value in payload.items() if key in allowed}


def load_renderer_config(path: Path) -> RendererConfig:
    payload = _read_json(path)
    root = path.parent
    mesh_assets = MeshAssetPaths(
        base_mesh=_path_or_none(payload.get("mesh_assets", {}).get("base_mesh"), root),
        dilation_mesh=_path_or_none(payload.get("mesh_assets", {}).get("dilation_mesh"), root),
        erosion_mesh=_path_or_none(payload.get("mesh_assets", {}).get("erosion_mesh"), root),
    )
    trace = TraceConfig(**_dataclass_payload(TraceConfig, payload.get("trace", {})))
    field = RendererFieldConfig(**_dataclass_payload(RendererFieldConfig, payload.get("field", {})))
    render = RenderConfig(**_dataclass_payload(RenderConfig, payload.get("render", {})))

    config = RendererConfig(
        mesh_assets=mesh_assets,
        trace=trace,
        field=field,
        render=render,
        profiling=ProfilingConfig(**_dataclass_payload(ProfilingConfig, payload.get("profiling", {}))),
        use_fastpath=bool(payload.get("use_fastpath", True)),
        device=payload.get("device", "cuda"),
    )
    config.assets.config_root = root
    config.assets.checkpoint = _path_or_none(payload.get("checkpoint"), root)
    config.assets.camera_path = _path_or_none(payload.get("camera_path"), root)
    config.assets.provenance_config = _path_or_none(payload.get("provenance_config"), root)
    config.assets.config_path = path
    config.assets.cuda_src_filename = str(payload.get("cuda_src_filename", "narrow_band.cu"))
    return config


def dump_renderer_config(config: RendererConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "device": config.device,
        "use_fastpath": config.use_fastpath,
        "mesh_assets": {
            "base_mesh": None if config.mesh_assets.base_mesh is None else str(config.mesh_assets.base_mesh),
            "dilation_mesh": None if config.mesh_assets.dilation_mesh is None else str(config.mesh_assets.dilation_mesh),
            "erosion_mesh": None if config.mesh_assets.erosion_mesh is None else str(config.mesh_assets.erosion_mesh),
        },
        "trace": asdict(config.trace),
        "field": asdict(config.field),
        "render": asdict(config.render),
        "profiling": asdict(config.profiling),
        "checkpoint": None if config.assets.checkpoint is None else str(config.assets.checkpoint),
        "camera_path": None if config.assets.camera_path is None else str(config.assets.camera_path),
        "provenance_config": None if config.assets.provenance_config is None else str(config.assets.provenance_config),
        "cuda_src_filename": config.assets.cuda_src_filename,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
