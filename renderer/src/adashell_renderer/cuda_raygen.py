from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from adashell_renderer.camera_path import CameraPathFrame
from adashell_renderer.types import RayBatch


_RAYGEN_CUDA = r"""
extern "C" __global__ void adashell_generate_camera_rays(
    const float* __restrict__ camera_to_world,
    float* __restrict__ origins,
    float* __restrict__ directions,
    float* __restrict__ t_near,
    float* __restrict__ t_far,
    const int width,
    const int height,
    const float fx,
    const float fy,
    const float cx,
    const float cy,
    const float pixel_center_offset,
    const float near_value,
    const float far_value
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = width * height;
    if (idx >= total) {
        return;
    }

    const int x = idx % width;
    const int y = idx / width;
    float dx = ((float)x - cx + pixel_center_offset) / fx;
    float dy = -(((float)y - cy + pixel_center_offset) / fy);
    float dz = -1.0f;
    float inv_len = rsqrtf(fmaxf(dx * dx + dy * dy + dz * dz, 1e-20f));
    dx *= inv_len;
    dy *= inv_len;
    dz *= inv_len;

    const float wx =
        camera_to_world[0] * dx +
        camera_to_world[1] * dy +
        camera_to_world[2] * dz;
    const float wy =
        camera_to_world[4] * dx +
        camera_to_world[5] * dy +
        camera_to_world[6] * dz;
    const float wz =
        camera_to_world[8] * dx +
        camera_to_world[9] * dy +
        camera_to_world[10] * dz;
    const float world_inv_len = rsqrtf(fmaxf(wx * wx + wy * wy + wz * wz, 1e-20f));

    const int base = idx * 3;
    origins[base + 0] = camera_to_world[3];
    origins[base + 1] = camera_to_world[7];
    origins[base + 2] = camera_to_world[11];
    directions[base + 0] = wx * world_inv_len;
    directions[base + 1] = wy * world_inv_len;
    directions[base + 2] = wz * world_inv_len;
    t_near[idx] = near_value;
    t_far[idx] = far_value;
}
"""


_RAYGEN_KERNEL: Any | None = None


def _require_cupy() -> Any:
    try:
        import cupy as cp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("CuPy ray generation requires cupy.") from exc
    return cp


def _device_id_from_string(device: str) -> int:
    if ":" not in str(device):
        return 0
    return int(str(device).split(":", 1)[1])


def generate_rays_cupy(
    frame: CameraPathFrame,
    *,
    device: str,
    camera_index: int | None,
    t_near: float,
    t_far: float,
    pixel_center_offset: float,
) -> RayBatch:
    global _RAYGEN_KERNEL
    if device != "cuda" and not str(device).startswith("cuda:"):
        raise RuntimeError("GL viewer runtime requires CUDA CuPy ray generation.")
    cp = _require_cupy()
    device_id = _device_id_from_string(device)
    with cp.cuda.Device(device_id):
        compile_start = time.perf_counter()
        if _RAYGEN_KERNEL is None:
            _RAYGEN_KERNEL = cp.RawKernel(_RAYGEN_CUDA, "adashell_generate_camera_rays")
        compile_ms = (time.perf_counter() - compile_start) * 1000.0

        total_rays = int(frame.height * frame.width)
        camera_to_world = cp.asarray(
            frame.camera_to_world.detach().cpu().to(dtype=torch.float32).numpy().reshape(-1),
            dtype=cp.float32,
        )
        origins = cp.empty((total_rays, 3), dtype=cp.float32)
        directions = cp.empty((total_rays, 3), dtype=cp.float32)
        near = cp.empty((total_rays,), dtype=cp.float32)
        far = cp.empty((total_rays,), dtype=cp.float32)

        launch_start = time.perf_counter()
        block_size = 256
        grid_size = (total_rays + block_size - 1) // block_size
        _RAYGEN_KERNEL(
            (grid_size,),
            (block_size,),
            (
                camera_to_world,
                origins,
                directions,
                near,
                far,
                np.int32(frame.width),
                np.int32(frame.height),
                np.float32(frame.fx),
                np.float32(frame.fy),
                np.float32(frame.cx),
                np.float32(frame.cy),
                np.float32(pixel_center_offset),
                np.float32(t_near),
                np.float32(t_far),
            ),
        )
        cp.cuda.get_current_stream().synchronize()
        launch_ms = (time.perf_counter() - launch_start) * 1000.0

    metadata: dict[str, object] = {
        "image_height": frame.height,
        "image_width": frame.width,
        "raygen_backend": "cupy_rawkernel",
        "raygen_compile_ms": compile_ms,
        "raygen_launch_ms": launch_ms,
    }
    if camera_index is not None:
        metadata["camera_index"] = int(camera_index)
    return RayBatch(origins=origins, directions=directions, t_near=near, t_far=far, metadata=metadata)
