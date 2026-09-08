from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

Tensor = Any


def torch_device_for_array(value: Any) -> torch.device:
    if torch.is_tensor(value):
        return value.device
    device = getattr(value, "device", None)
    device_id = getattr(device, "id", None)
    if device_id is not None:
        return torch.device(f"cuda:{int(device_id)}")
    return torch.device("cpu")


@dataclass(slots=True)
class RayBatch:
    origins: Tensor
    directions: Tensor
    t_near: Tensor
    t_far: Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_rays(self) -> int:
        return int(self.origins.shape[0])

@dataclass(slots=True)
class TraceHitBuffers:
    hit_t: Tensor
    hit_count: Tensor
    inner_first_t: Tensor
    outer_hit_flag: Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class IntervalPack:
    """Packed sample buffer after shell interval sampling.

    Shape contract:
    - `starts`:      `(N,)`
    - `ends`:        `(N,)`
    - `eval_t`:      `(N,)`
    - `ray_indices`: `(N,)`
    - `packed_info`: `(R, 2)`

    Meaning:
    - `N = valid_samples`: total number of valid samples across all active rays
    - `R = num_rays`: number of full-frame rays, including zero-sample rays

    Field semantics:
    - `starts[i]`, `ends[i]`
      The interval bounds of packed sample `i` along its ray.
    - `eval_t[i]`
      The representative point used to query the field for sample `i`.
      In the current shell path this is the per-bin center-like sample position.
    - `ray_indices[i]`
      Which full-frame ray sample `i` belongs to. This is in `[0, R)`.
    - `packed_info[r, 0]`
      Offset of ray `r` inside the flat packed sample arrays.
    - `packed_info[r, 1]`
      Number of packed samples belonging to ray `r`.

    Layout invariant:
    - packed samples are ray-major and contiguous
    - i.e. samples for a given ray live in one consecutive slice
      `[offset : offset + count)`
    """
    starts: Tensor
    ends: Tensor
    eval_t: Tensor
    ray_indices: Tensor
    packed_info: Tensor
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def valid_samples(self) -> int:
        return int(self.starts.shape[0])

    @property
    def num_rays(self) -> int:
        return int(self.packed_info.shape[0])


@dataclass(slots=True)
class DrRenderBuffer:
    """GPU-resident render output before any Torch/CPU/export boundary.

    `rgb_channels` are Dr.Jit arrays, one lane per rendered ray. This is the
    target endpoint for a Dr-native field path: the renderer has already
    produced an image-sized RGB buffer on the GPU, but it has not converted it
    to Torch for legacy saving or presentation code.
    """

    rgb_channels: tuple[Any, Any, Any]
    num_rays: int
    image_width: int
    image_height: int
    depth: Optional[Any] = None
    accumulation: Optional[Any] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
