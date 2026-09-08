from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import trimesh


@dataclass(frozen=True)
class ShellRayTracingConfig:
    """CPU reference config for Adaptive Shell narrow-band sampling."""

    n_sample_max: int = 16
    n_intervals_max: int = 20
    use_inner_mesh: bool = True
    single_sample_threshold: float = 0.02
    sample_spacing: float = 0.01
    # Numerical robustness
    t_eps: float = 1e-6
    boundary_eps: float = 1e-4


def _make_intersector(mesh: trimesh.Trimesh):
    """Return a trimesh ray intersector, preferring accelerated backends when available."""
    # trimesh chooses an intersector automatically, but being explicit helps reliability.
    try:
        # pyembree (if installed) can be much faster.
        from trimesh.ray.ray_pyembree import RayMeshIntersector  # type: ignore

        return RayMeshIntersector(mesh)
    except Exception:
        from trimesh.ray.ray_triangle import RayMeshIntersector  # type: ignore

        return RayMeshIntersector(mesh)


def _safe_mesh_contains(mesh: trimesh.Trimesh, points: np.ndarray) -> np.ndarray:
    """Robust-ish wrapper for mesh.contains for potentially non-watertight meshes."""
    try:
        # trimesh.contains expects shape (N, 3)
        inside = mesh.contains(points)
        if inside is None:
            return np.zeros((points.shape[0],), dtype=bool)
        return np.asarray(inside, dtype=bool)
    except Exception:
        return np.zeros((points.shape[0],), dtype=bool)


def _intersect_ts(
    intersector,
    origins: np.ndarray,
    directions: np.ndarray,
    *,
    t_min: np.ndarray,
    t_max: np.ndarray,
    t_eps: float,
) -> List[np.ndarray]:
    """Compute per-ray sorted intersection distances (t) within [t_min, t_max]."""
    # trimesh returns (locations, index_ray, index_tri)
    locations, index_ray, _ = intersector.intersects_location(
        origins, directions, multiple_hits=True
    )
    if len(locations) == 0:
        return [np.empty((0,), dtype=np.float32) for _ in range(origins.shape[0])]

    locations = np.asarray(locations, dtype=np.float32)
    index_ray = np.asarray(index_ray, dtype=np.int64)

    o = origins[index_ray]
    d = directions[index_ray]
    v = locations - o
    # Allow for non-unit directions.
    denom = np.sum(d * d, axis=-1)
    denom = np.where(denom > 0.0, denom, 1.0)
    ts = np.sum(v * d, axis=-1) / denom

    # Filter to forward hits, and within [t_min, t_max] with epsilon.
    tmin = t_min[index_ray] + t_eps
    tmax = t_max[index_ray] - t_eps
    valid = (ts > tmin) & (ts < tmax)
    ts = ts[valid]
    index_ray = index_ray[valid]

    per_ray: List[List[float]] = [[] for _ in range(origins.shape[0])]
    for ridx, t in zip(index_ray.tolist(), ts.tolist()):
        per_ray[ridx].append(t)

    return [np.sort(np.asarray(v, dtype=np.float32)) for v in per_ray]


class ShellCpuRayTracer:
    """CPU reference Adaptive Shell narrow-band sampler based on triangle meshes.

    This follows the paper's interval logic:
    - Enumerate ordered ray intersections with outer (dilation) and inner (erosion) meshes.
    - Construct shell intervals where ray is inside outer but not inside inner.
    - Stop once the ray enters the inner mesh (solid interior).
    - Within each interval, place equidistant samples and cap counts.
    """

    def __init__(
        self,
        dilation_mesh: trimesh.Trimesh,
        erosion_mesh: trimesh.Trimesh,
        config: ShellRayTracingConfig,
    ) -> None:
        self.dilation_mesh = dilation_mesh
        self.erosion_mesh = erosion_mesh
        self.config = config

        # Intersectors are stateful and relatively expensive to create.
        self._outer = _make_intersector(dilation_mesh)
        self._inner = _make_intersector(erosion_mesh)

    def compute_shell_intervals(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        nears: np.ndarray,
        fars: np.ndarray,
    ) -> List[List[Tuple[float, float]]]:
        """Return per-ray list of [t_enter, t_exit] shell intervals."""
        cfg = self.config
        num_rays = origins.shape[0]

        # Initial inside/outside states at ray origin.
        inside_outer0 = _safe_mesh_contains(self.dilation_mesh, origins)
        if cfg.use_inner_mesh:
            inside_inner0 = _safe_mesh_contains(self.erosion_mesh, origins)
        else:
            inside_inner0 = np.zeros((num_rays,), dtype=bool)

        outer_hits = _intersect_ts(self._outer, origins, directions, t_min=nears, t_max=fars, t_eps=cfg.t_eps)
        if cfg.use_inner_mesh:
            inner_hits = _intersect_ts(self._inner, origins, directions, t_min=nears, t_max=fars, t_eps=cfg.t_eps)
        else:
            inner_hits = [np.empty((0,), dtype=np.float32) for _ in range(num_rays)]

        intervals_per_ray: List[List[Tuple[float, float]]] = []

        for i in range(num_rays):
            t_near = float(nears[i])
            t_far = float(fars[i])
            if not (t_far > t_near):
                intervals_per_ray.append([])
                continue

            inside_outer = bool(inside_outer0[i])
            inside_inner = bool(inside_inner0[i])

            # If we start inside inner mesh, we are already in solid interior: no contribution.
            if inside_inner:
                intervals_per_ray.append([])
                continue

            # Build sorted events (t, is_inner)
            ts_outer = outer_hits[i]
            ts_inner = inner_hits[i]

            if ts_outer.size == 0 and ts_inner.size == 0:
                intervals_per_ray.append([])
                continue

            # Merge while tracking type: 0=outer, 1=inner
            t_all = np.concatenate([ts_outer, ts_inner], axis=0) if ts_outer.size + ts_inner.size > 0 else np.empty((0,))
            type_all = (
                np.concatenate(
                    [np.zeros((ts_outer.size,), dtype=np.int8), np.ones((ts_inner.size,), dtype=np.int8)], axis=0
                )
                if ts_outer.size + ts_inner.size > 0
                else np.empty((0,), dtype=np.int8)
            )
            order = np.argsort(t_all, kind="mergesort")
            t_all = t_all[order]
            type_all = type_all[order]

            intervals: List[Tuple[float, float]] = []
            t_cursor = t_near
            j = 0

            while j < t_all.size:
                t_evt = float(t_all[j])
                if t_evt > t_cursor and inside_outer and not inside_inner:
                    intervals.append((t_cursor, t_evt))
                    if len(intervals) >= cfg.n_intervals_max:
                        break

                # Process all coincident events at the same t (within tolerance)
                t_ref = t_evt
                while j < t_all.size and abs(float(t_all[j]) - t_ref) <= cfg.t_eps:
                    if int(type_all[j]) == 0:
                        inside_outer = not inside_outer
                    else:
                        inside_inner = not inside_inner
                    j += 1

                t_cursor = t_ref

                # Stop once the ray enters inner (solid interior).
                if inside_inner:
                    break

            if len(intervals) < cfg.n_intervals_max and (not inside_inner) and inside_outer and t_cursor < t_far:
                intervals.append((t_cursor, t_far))

            # Nudge boundaries inward for numerical robustness.
            robust_intervals: List[Tuple[float, float]] = []
            for a, b in intervals:
                a2 = a + cfg.boundary_eps
                b2 = b - cfg.boundary_eps
                if b2 > a2:
                    robust_intervals.append((a2, b2))
            intervals_per_ray.append(robust_intervals[: cfg.n_intervals_max])

        return intervals_per_ray

    def intervals_to_packed_bins(
        self,
        intervals_per_ray: Sequence[Sequence[Tuple[float, float]]],
        nears: np.ndarray,
        fars: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convert intervals to fixed-size packed bins (starts, ends, mask).

        Output shapes:
        - starts: (num_rays, N_total)
        - ends:   (num_rays, N_total)
        - mask:   (num_rays, N_total) in {0,1}
        - eval_t: (num_rays, N_total), paper tau_s for actual field queries
        """
        cfg = self.config
        num_rays = len(intervals_per_ray)
        n_intervals_max = cfg.n_intervals_max
        n_sample_max = cfg.n_sample_max
        n_total = n_intervals_max * n_sample_max

        starts = np.zeros((num_rays, n_total), dtype=np.float32)
        ends = np.zeros((num_rays, n_total), dtype=np.float32)
        mask = np.zeros((num_rays, n_total), dtype=np.float32)
        eval_t = np.zeros((num_rays, n_total), dtype=np.float32)

        # Initialize padded samples to far with zero delta.
        far_fill = fars.astype(np.float32).reshape(-1, 1)
        starts[:] = far_fill
        ends[:] = far_fill
        eval_t[:] = far_fill

        for ridx in range(num_rays):
            t_near = float(nears[ridx])
            t_far = float(fars[ridx])
            write_base = 0
            for interval_idx, (a, b) in enumerate(intervals_per_ray[ridx][:n_intervals_max]):
                if write_base >= n_total:
                    break

                a = float(np.clip(a, t_near, t_far))
                b = float(np.clip(b, t_near, t_far))
                w = b - a
                if w <= 0.0:
                    write_base += n_sample_max
                    continue

                # Paper rule:
                #   N = min( ceil(max(w-w_s,0)/delta_s) + 1, N_max )
                #   tau_s = Linspace(a, b, N+2)[1:-1]
                n = int(np.ceil(max(w - cfg.single_sample_threshold, 0.0) / cfg.sample_spacing) + 1)
                n = max(1, min(n, n_sample_max))

                tau = np.linspace(a, b, n + 2, dtype=np.float32)[1:-1]
                if n == 1:
                    bin_starts = np.asarray([a], dtype=np.float32)
                    bin_ends = np.asarray([b], dtype=np.float32)
                else:
                    edges = np.empty((n + 1,), dtype=np.float32)
                    edges[0] = a
                    edges[-1] = b
                    edges[1:-1] = 0.5 * (tau[:-1] + tau[1:])
                    bin_starts = edges[:-1]
                    bin_ends = edges[1:]

                slot = slice(write_base, write_base + n)
                starts[ridx, slot] = np.clip(bin_starts, t_near, t_far).astype(np.float32)
                ends[ridx, slot] = np.clip(bin_ends, t_near, t_far).astype(np.float32)
                mask[ridx, slot] = 1.0
                eval_t[ridx, slot] = np.clip(tau, t_near, t_far).astype(np.float32)

                write_base += n_sample_max

        return starts, ends, mask, eval_t

    def trace_and_pack(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        nears: np.ndarray,
        fars: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Convenience: intervals then packed bins."""
        intervals = self.compute_shell_intervals(origins, directions, nears, fars)
        return self.intervals_to_packed_bins(intervals, nears, fars)

