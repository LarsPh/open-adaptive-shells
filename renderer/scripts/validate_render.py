"""Capture fixed-ray GPU render evidence without a viewer or training runtime."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--camera-index", type=int, default=10)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    if args.source_root:
        sys.path.insert(0, str(args.source_root.resolve() / "src"))
    import cupy as cp
    import drjit as dr
    import torch
    from PIL import Image

    from adashell_renderer.camera_path import load_camera_path
    from adashell_renderer.config_io import load_renderer_config
    from adashell_renderer.cuda_raygen import generate_rays_cupy
    from adashell_renderer.renderer_runtime import build_runtime_pipeline, render_full_image_buffer
    from adashell_renderer.viewer_support import scale_camera_frame

    if args.iterations < 1 or args.warmup < 0 or not 0 < args.scale <= 1:
        parser.error("Require iterations >= 1, warmup >= 0 and 0 < scale <= 1.")
    config = load_renderer_config(args.config.resolve())
    frame = scale_camera_frame(load_camera_path(config.assets.camera_path).frames[args.camera_index], args.scale)
    args.output.mkdir(parents=True, exist_ok=True)
    build_start = time.perf_counter()
    pipeline = build_runtime_pipeline(config)
    build_ms = (time.perf_counter() - build_start) * 1000
    rays = generate_rays_cupy(
        frame, device=config.device, camera_index=args.camera_index,
        t_near=config.render.ray_t_near, t_far=config.render.ray_t_far,
        pixel_center_offset=config.render.pixel_center_offset,
    )
    np.savez(args.output / "rays.npz", origins=cp.asnumpy(rays.origins), directions=cp.asnumpy(rays.directions))
    records = []
    for index in range(args.warmup + args.iterations):
        cp.cuda.Device().synchronize()
        start = time.perf_counter()
        buffer, profile, detail = render_full_image_buffer(pipeline, rays)
        dr.eval(*buffer.rgb_channels)
        dr.sync_thread()
        cp.cuda.Device().synchronize()
        wall_ms = (time.perf_counter() - start) * 1000
        records.append({"index": index, "wall_ms": wall_ms, "profile": profile, "detail": detail})
        if index == 0:
            print(f"First frame: {wall_ms:.3f} ms", flush=True)
    rgb = np.stack([cp.asnumpy(cp.from_dlpack(channel)) for channel in buffer.rgb_channels], axis=-1)
    rgb = rgb.reshape(frame.height, frame.width, 3)
    if not np.isfinite(rgb).all():
        raise RuntimeError("RGB contains non-finite values.")
    np.save(args.output / "rgb.npy", rgb)
    Image.fromarray(np.uint8(np.clip(rgb, 0, 1) * 255 + 0.5)).save(args.output / "rgb.png")
    diagnostics = {}
    for mode in ("pred_gradient_norm", "weight_sum"):
        pipeline.fast_render_diagnostic_mode = mode
        result, _, _ = render_full_image_buffer(pipeline, rays)
        for name, value in result.diagnostics.items():
            if isinstance(value, tuple):
                continue
            dr.eval(value)
            dr.sync_thread()
            diagnostics[name] = cp.asnumpy(cp.from_dlpack(value))
    pipeline.fast_render_diagnostic_mode = "none"
    counts = buffer.metadata.get("diagnostics", {}).get("sample_count")
    if counts is not None:
        dr.eval(counts)
        dr.sync_thread()
        diagnostics["sample_count"] = cp.asnumpy(cp.from_dlpack(counts))
    np.savez(args.output / "diagnostics.npz", **diagnostics)
    for name, values in diagnostics.items():
        if values.size == frame.width * frame.height:
            values = values.reshape(frame.height, frame.width)
            limit = max(float(np.percentile(values, 99)), 1e-6)
            Image.fromarray(np.uint8(np.clip(values / limit, 0, 1) * 255)).save(args.output / f"{name}.png")
    # A full miss must still return one background pixel per input ray.
    rays.origins += cp.float32(10000)
    empty, _, empty_detail = render_full_image_buffer(pipeline, rays)
    dr.eval(*empty.rgb_channels)
    dr.sync_thread()
    empty_rgb = np.stack([cp.asnumpy(cp.from_dlpack(channel)) for channel in empty.rgb_channels], axis=-1)
    expected_background = 1.0 if config.render.white_background else 0.0
    np.testing.assert_allclose(empty_rgb, expected_background, atol=1e-6)
    measured = [record["wall_ms"] for record in records[args.warmup:]]
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True,
    ).strip()
    report = {
        "source_root": str(args.source_root.resolve()) if args.source_root else None,
        "config": str(args.config.resolve()), "camera_index": args.camera_index,
        "width": frame.width, "height": frame.height, "scale": args.scale,
        "python": platform.python_version(), "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(), "driver": driver, "torch_cuda": torch.version.cuda,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "drjit", "cupy-cuda12x", "python-optix", "tinycudann")},
        "build_ms": build_ms, "warmup": args.warmup, "iterations": args.iterations,
        "median_ms": float(np.median(measured)), "p95_ms": float(np.percentile(measured, 95)),
        "min_ms": min(measured), "max_ms": max(measured), "records": records,
        "empty_frame_passed": True, "empty_frame_detail": empty_detail,
        "timing_scope": "fixed pre-generated rays; full render plus final GPU synchronization; excludes raygen, PBO and GUI",
    }
    (args.output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved {args.output}: median {report['median_ms']:.3f} ms; p95 {report['p95_ms']:.3f} ms", flush=True)


if __name__ == "__main__":
    main()
