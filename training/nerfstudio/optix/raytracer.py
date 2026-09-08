import logging
import os
from pathlib import Path
import time
from typing import Any, Optional, Tuple, Union, cast

import cv2
import numpy as np
import optix as ox
import cupy as cp
import torch
import torch.nn as nn
import trimesh

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.optix.pack_cuda import PackIntervalsCudaLauncher
from nerfstudio.optix.pair_cuda import PairIntervalsCudaLauncher
from nerfstudio.optix import utils as u_ox

log = logging.getLogger(__name__)

ArrayLike = Union[np.ndarray, torch.Tensor]

# hit flag
ENTERING = 1
EXITING = 2

class ShellRayTracer(nn.Module):

    def __init__(self,
        dilation_mesh: trimesh.Trimesh,
        erosion_mesh: trimesh.Trimesh,
        n_sample_max: int = 16,
        n_depth_peeling_max: int = 20,
        use_inner_mesh: bool = True,
        single_sample_threshold: float = 0.02,
        sample_spacing: float = 0.01,
        enable_cuda_pair_kernel: Optional[bool] = None,
        enable_cuda_pack_kernel: Optional[bool] = None,
        cuda_src_filename = "narrow_band.cu",
    ):
        super().__init__()
        self.n_sample_max = n_sample_max
        self.n_depth_peeling_max = n_depth_peeling_max
        self.use_inner_mesh = use_inner_mesh
        self.single_sample_threshold = single_sample_threshold
        self.sample_spacing = sample_spacing
        self._validated_camera_indices = set()
        self._last_trace_profile = None
        self._profile_strict_timing = False
        self._pair_cuda_launcher: Optional[PairIntervalsCudaLauncher] = None
        self._pack_cuda_launcher: Optional[PackIntervalsCudaLauncher] = None
        self._enable_cuda_pair_kernel = (
            bool(enable_cuda_pair_kernel)
            if enable_cuda_pair_kernel is not None
            else bool(int(os.environ.get("ADASHELL_ENABLE_CUDA_PAIR_KERNEL", "0")))
        )
        self._enable_cuda_pack_kernel = (
            bool(enable_cuda_pack_kernel)
            if enable_cuda_pack_kernel is not None
            else bool(int(os.environ.get("ADASHELL_ENABLE_CUDA_PACK_KERNEL", "0")))
        )

        ctx=u_ox.create_context(log=log)

        pipeline_options = ox.PipelineCompileOptions(traversable_graph_flags=ox.TraversableGraphFlags.ALLOW_SINGLE_GAS,
                                                num_payload_values=2,
                                                num_attribute_values=2,
                                                exception_flags=ox.ExceptionFlags.NONE,
                                                pipeline_launch_params_variable_name="params")        
        module = u_ox.create_module(ctx, pipeline_options, cuda_src_filename)
        program_grps = u_ox.create_program_groups(ctx, module)
        self.pipeline = u_ox.create_pipeline(ctx, program_grps, pipeline_options)

        self.sbt = u_ox.create_sbt(program_grps)

        self.dilation_gas = u_ox.create_acceleration_structure(ctx, dilation_mesh)
        self.erosion_gas = u_ox.create_acceleration_structure(ctx, erosion_mesh)

    def set_profile_options(self, strict_timing: bool = False) -> None:
        self._profile_strict_timing = bool(strict_timing)

    def _sync_cuda_for_profile(self, device: Optional[torch.device]) -> None:
        if not self._profile_strict_timing or not torch.cuda.is_available() or device is None:
            return
        if device.type == "cuda":
            torch.cuda.synchronize(device=device)

    def set_pair_cuda_kernel_enabled(self, enabled: bool = True) -> None:
        self._enable_cuda_pair_kernel = bool(enabled)

    def _get_pair_cuda_launcher(self) -> PairIntervalsCudaLauncher:
        if self._pair_cuda_launcher is None:
            self._pair_cuda_launcher = PairIntervalsCudaLauncher()
        return self._pair_cuda_launcher

    def set_pack_cuda_kernel_enabled(self, enabled: bool = True) -> None:
        self._enable_cuda_pack_kernel = bool(enabled)

    def _get_pack_cuda_launcher(self) -> PackIntervalsCudaLauncher:
        if self._pack_cuda_launcher is None:
            self._pack_cuda_launcher = PackIntervalsCudaLauncher()
        return self._pack_cuda_launcher

    @staticmethod
    def pair_hit_intervals(
        hit_t: np.ndarray, hit_count: np.ndarray, min_interval_size: float = 1e-4
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Pair hit intervals.
        hit_t: (height, width, max_hits)
        hit_count: (height, width)
        interval_starts: (height, width, max_hits // 2)
        interval_ends: (height, width, max_hits // 2)
        interval_mask: (height, width, max_hits // 2)
        interval_count: (height, width)
        """
        h, w, max_hits = hit_t.shape
        max_intervals = max_hits // 2

        interval_starts = hit_t[..., ::2].copy()
        interval_ends = hit_t[..., 1::2].copy()

        # 0. filter interval from dangling odd hit
        pair_idx = np.arange(max_intervals, dtype=np.int32)[None, None, :]
        valid_pair_by_count = pair_idx < (hit_count[..., None] // 2)

        # 1. filter ends - starts < min_interval_size (including minus interval size)
        interval_size = interval_ends - interval_starts
        valid_size = interval_size >= min_interval_size

        # 2. filter invalid hit_t 
        valid_values = (interval_starts > 0.0) & (interval_ends > 0.0)
        
        interval_mask = valid_values & valid_size & valid_pair_by_count
        interval_count = np.sum(interval_mask, axis=-1).astype(np.uint32)

        return interval_starts, interval_ends, interval_mask, interval_count
    
    def pair_shell_hit_intervals(
        self,
        outer_hit_t: ArrayLike, outer_hit_flag: ArrayLike, outer_hit_count: ArrayLike,
        inner_first_t: ArrayLike, ray_tmin: ArrayLike,
        min_interval_size: float = 1e-4, compress_same_flag_runs: bool = True,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Pair shell hit intervals.
        outer_hit_t: (R, max_hits)
        outer_hit_flag: (R, max_hits)
        outer_hit_count: (R)
        inner_first_t: (R)
        ray_tmin: (R)
        interval_starts: (R, ceil(max_hits / 2))
        interval_ends: (R, ceil(max_hits / 2))
        interval_mask: (R, ceil(max_hits / 2))
        interval_count: (R)
        """
        if torch.is_tensor(outer_hit_t):
            if self._enable_cuda_pair_kernel:
                return self._get_pair_cuda_launcher().launch(
                    outer_hit_t=outer_hit_t,
                    outer_hit_flag=outer_hit_flag,
                    outer_hit_count=outer_hit_count,
                    inner_first_t=inner_first_t,
                    ray_tmin=ray_tmin,
                    min_interval_size=min_interval_size,
                    compress_same_flag_runs=compress_same_flag_runs,
                )
            n_rays, max_hits = outer_hit_t.shape
            max_intervals = (max_hits + 1) // 2
            device = outer_hit_t.device
            interval_starts = torch.full((n_rays, max_intervals), -1.0, dtype=torch.float32, device=device)
            interval_ends = torch.full((n_rays, max_intervals), -1.0, dtype=torch.float32, device=device)
            interval_mask = torch.zeros((n_rays, max_intervals), dtype=torch.bool, device=device)
            interval_count = torch.zeros((n_rays,), dtype=torch.int32, device=device)

            # states
            has_open_interval = torch.zeros((n_rays,), dtype=torch.bool, device=device)
            current_enter_t = torch.zeros((n_rays,), dtype=torch.float32, device=device)
            finished = torch.zeros((n_rays,), dtype=torch.bool, device=device)
            # set missed inner hits to infinity
            inner_t = torch.where(
                inner_first_t > 0.0,
                inner_first_t.to(torch.float32),
                torch.full_like(inner_first_t, float("inf"), dtype=torch.float32),
            )

            # iterate over outer hits
            for hit_idx in range(max_hits):
                active_rays = (~finished) & (outer_hit_count > hit_idx)
                if not bool(active_rays.any().item()):
                    break

                t_i = outer_hit_t[:, hit_idx]
                flag_i = outer_hit_flag[:, hit_idx]
                if compress_same_flag_runs:
                    # first entering
                    if hit_idx == 0:
                        first_enter = active_rays & (flag_i == ENTERING)
                    else:
                        prev_flag = outer_hit_flag[:, hit_idx - 1]
                        # first entering if prev hit is invalid or prev hit is not entering
                        first_enter = active_rays & (flag_i == ENTERING) & (prev_flag != ENTERING)
                    # last exiting
                    if hit_idx == max_hits - 1:
                        last_exit = active_rays & (flag_i == EXITING)
                    else:
                        next_invalid = outer_hit_count <= hit_idx + 1
                        next_flag = outer_hit_flag[:, hit_idx + 1]
                        # last exiting if next hit is invalid or next hit is not exiting
                        last_exit = active_rays & (flag_i == EXITING) & (next_invalid | (next_flag != EXITING))
                else:
                    first_enter = active_rays & (flag_i == ENTERING)
                    last_exit = active_rays & (flag_i == EXITING)

                # standard case
                new_open = first_enter & (~has_open_interval)
                current_enter_t = torch.where(new_open, t_i, current_enter_t)
                has_open_interval = has_open_interval | new_open

                # when first hit is exiting, clip with ray_tmin
                clipped_open = last_exit & (~has_open_interval)
                current_enter_t = torch.where(clipped_open, ray_tmin.to(torch.float32), current_enter_t)
                has_open_interval = has_open_interval | clipped_open

                # commit interval
                commit = last_exit & has_open_interval
                if bool(commit.any().item()):
                    d_exit = torch.minimum(t_i, inner_t).to(torch.float32)  # clip with inner hit
                    valid_interval = commit & ((d_exit - current_enter_t) >= min_interval_size)

                    # calculate row and column indices and filter for outputs
                    ray_idcs = torch.nonzero(valid_interval, as_tuple=False).squeeze(-1)
                    interval_idcs = interval_count[ray_idcs]
                    keep = interval_idcs < max_intervals
                    ray_idcs = ray_idcs[keep]
                    interval_idcs = interval_idcs[keep].to(torch.long)

                    if ray_idcs.numel() > 0:
                        interval_starts[ray_idcs, interval_idcs] = current_enter_t[ray_idcs]
                        interval_ends[ray_idcs, interval_idcs] = d_exit[ray_idcs]
                        interval_mask[ray_idcs, interval_idcs] = True
                        interval_count[ray_idcs] += 1

                    # close open interval
                    has_open_interval = has_open_interval & (~commit)
                    current_enter_t = torch.where(commit, torch.zeros_like(current_enter_t), current_enter_t)

                    # mark finished rays when t_i goes beyond inner hit
                    finished = finished | (commit & torch.isfinite(inner_t) & (inner_t < t_i))

            return interval_starts, interval_ends, interval_mask, interval_count

        n_rays, max_hits = outer_hit_t.shape
        max_intervals = (max_hits + 1) // 2
        interval_starts =  -1 * np.ones((n_rays, max_intervals), dtype=np.float32)
        interval_ends =  -1 * np.ones((n_rays, max_intervals), dtype=np.float32)
        interval_mask = np.zeros((n_rays, max_intervals), dtype=bool)
        interval_count = np.zeros((n_rays), dtype=np.uint32)

        # states
        has_open_interval = np.zeros((n_rays), dtype=bool)
        current_enter_t = np.zeros((n_rays), dtype=np.float32)
        finished = np.zeros((n_rays), dtype=bool)
        # set missed inner hits to infinity
        inner_t = np.where(inner_first_t > 0.0, inner_first_t, np.inf).astype(np.float32)

        # iterate over outer hits
        for hit_idx in range(max_hits):
            active_rays = ~finished & (outer_hit_count > hit_idx)
            if not np.any(active_rays):
                break
            
            t_i = outer_hit_t[:, hit_idx]
            flag_i = outer_hit_flag[:, hit_idx]
            if compress_same_flag_runs:
                # first entering
                if hit_idx == 0:
                    first_enter = active_rays & (flag_i == ENTERING)
                else:
                    prev_flag = outer_hit_flag[:, hit_idx - 1]
                    # first entering if prev hit is invalid or prev hit is not entering
                    first_enter = active_rays & (flag_i == ENTERING) & (prev_flag != ENTERING)
                # last exiting
                if hit_idx == max_hits - 1:
                    last_exit = active_rays & (flag_i == EXITING)
                else:
                    next_invalid = outer_hit_count <= hit_idx + 1
                    next_flag = outer_hit_flag[:, hit_idx + 1]
                    # last exiting if next hit is invalid or next hit is not exiting
                    last_exit = active_rays & (flag_i == EXITING) & (next_invalid | (next_flag != EXITING))

            else:
                first_enter = active_rays & (flag_i == ENTERING)
                last_exit = active_rays & (flag_i == EXITING)

            # standard case
            new_open = first_enter & ~has_open_interval
            current_enter_t[new_open] = t_i[new_open]
            has_open_interval[new_open] = True

            # when first hit is exiting, clip with ray_tmin
            clipped_open = last_exit & (~has_open_interval)
            current_enter_t[clipped_open] = ray_tmin[clipped_open]
            has_open_interval[clipped_open] = True

            # commit interval
            commit = last_exit & has_open_interval
            if np.any(commit):
                d_exit = np.minimum(t_i, inner_t).astype(np.float32) # clip with inner hit
                valid_interval = commit & (d_exit - current_enter_t >= min_interval_size)

                # calculate row and column indices and filter for outputs
                ray_idcs = np.nonzero(valid_interval)[0]
                interval_idcs = interval_count[ray_idcs]
                keep = interval_idcs < max_intervals
                ray_idcs, interval_idcs = ray_idcs[keep], interval_idcs[keep]

                interval_starts[ray_idcs, interval_idcs] = current_enter_t[ray_idcs]
                interval_ends[ray_idcs, interval_idcs] = d_exit[ray_idcs]
                interval_mask[ray_idcs, interval_idcs] = True
                interval_count[ray_idcs] += 1

                # close open interval
                has_open_interval[commit] = False
                current_enter_t[commit] = 0.0

                # mark finished rays when t_i goes beyond inner hit
                finished |= commit & np.isfinite(inner_t) & (inner_t < t_i)

        return interval_starts, interval_ends, interval_mask, interval_count



    def pack_interval_to_sample_bins(
        self,
        interval_starts: ArrayLike,
        interval_ends: ArrayLike,
        interval_mask: ArrayLike,
        interval_count: ArrayLike,
        ray_fars: ArrayLike,
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Pack intervals to compressed fixed-size bins using the paper's interior tau samples.
        interval_starts: (height, width, max_intervals)
        interval_ends: (height, width, max_intervals)
        interval_mask: (height, width, max_intervals)
        interval_count: (height, width)
        ray_fars: (height, width)
        """
        if torch.is_tensor(interval_starts):
            if self._enable_cuda_pack_kernel:
                return self._get_pack_cuda_launcher().launch(
                    interval_starts=interval_starts,
                    interval_ends=interval_ends,
                    interval_mask=interval_mask,
                    interval_count=interval_count,
                    ray_fars=ray_fars,
                    n_sample_max=self.n_sample_max,
                    single_sample_threshold=self.single_sample_threshold,
                    sample_spacing=self.sample_spacing,
                )
            h, w, max_intervals = [int(v) for v in interval_starts.shape]
            n_total = self.n_sample_max * max_intervals
            device = interval_starts.device

            tau_local, tau_valid, samples_per_interval = self._compute_tau_local(
                interval_starts, interval_ends, interval_mask
            )

            ## 2. compressed offsets for each interval's bins
            offsets = torch.cumsum(samples_per_interval, dim=-1) - samples_per_interval

            ## 3. local sample slots
            local_idx = torch.arange(self.n_sample_max, device=device, dtype=torch.int32)[None, None, None, :]
            local_idx_f = local_idx.to(torch.float32)
            interval_len = interval_ends - interval_starts
            denom = (torch.clamp(samples_per_interval, min=1)[..., None] + 1).to(torch.float32)
            step = interval_len[..., None] / denom

            ## 4. convert tau samples to bins for Nerfstudio.
            ## Use Voronoi-style boundaries between neighboring tau's, with interval boundaries at the ends.
            bin_starts_local = torch.where(
                local_idx == 0,
                interval_starts[..., None],
                0.5 * (
                    tau_local
                    + (interval_starts[..., None] + local_idx_f * step)
                ),
            ).to(torch.float32)
            bin_ends_local = torch.where(
                local_idx == (samples_per_interval[..., None] - 1),
                interval_ends[..., None],
                0.5 * (
                    tau_local
                    + (interval_starts[..., None] + (local_idx_f + 2.0) * step)
                ),
            ).to(torch.float32)

            # 5. compressed packed indices
            packed_idx = offsets[..., None] + local_idx  # idx between [0, n_intervals * n_samples - 1]
            packed_bin_valid = tau_valid & (packed_idx < n_total)

            # 6. flatten and write with per-ray base offset
            ray_idx = torch.arange(h * w, device=device, dtype=torch.int64).reshape(h, w, 1, 1)
            flat_out_idx = ray_idx * n_total + packed_idx.to(torch.int64)
            packed_idx_flat_valid = flat_out_idx[packed_bin_valid]

            far_values = ray_fars.reshape(h * w, 1).to(device=device, dtype=torch.float32)
            packed_bins_starts = far_values.repeat(1, n_total).reshape(h * w * n_total)
            packed_bins_ends = far_values.repeat(1, n_total).reshape(h * w * n_total)
            packed_eval_t = far_values.repeat(1, n_total).reshape(h * w * n_total)
            packed_mask = torch.zeros((h * w * n_total,), dtype=torch.uint8, device=device)
            packed_bins_starts[packed_idx_flat_valid] = bin_starts_local[packed_bin_valid]
            packed_bins_ends[packed_idx_flat_valid] = bin_ends_local[packed_bin_valid]
            packed_eval_t[packed_idx_flat_valid] = tau_local[packed_bin_valid]
            packed_mask[packed_idx_flat_valid] = 1
            packed_bins_starts = packed_bins_starts.reshape(h, w, n_total)
            packed_bins_ends = packed_bins_ends.reshape(h, w, n_total)
            packed_eval_t = packed_eval_t.reshape(h, w, n_total)
            packed_mask = packed_mask.reshape(h, w, n_total)
            return packed_bins_starts, packed_bins_ends, packed_mask, packed_eval_t

        h, w, max_intervals = interval_starts.shape
        n_total = self.n_sample_max * max_intervals

        tau_local, tau_valid, samples_per_interval = self._compute_tau_local(
            interval_starts, interval_ends, interval_mask
        )

        ## 2. compressed offsets for each interval's bins
        offsets = np.cumsum(samples_per_interval, axis=-1) - samples_per_interval

        ## 3. local sample slots
        local_idx = np.arange(self.n_sample_max, dtype=np.int32)[None, None, None, :] # (1, 1, 1, n_sample_max)
        interval_len = interval_ends - interval_starts
        denom = (np.maximum(samples_per_interval, 1)[..., None] + 1).astype(np.float32)
        step = interval_len[..., None] / denom

        ## 4. convert tau samples to bins for Nerfstudio.
        ## Use Voronoi-style boundaries between neighboring tau's, with interval boundaries at the ends.
        bin_starts_local = np.where(
            local_idx == 0,
            interval_starts[..., None],
            0.5 * (
                tau_local
                + (interval_starts[..., None] + local_idx * step)
            ),
        ).astype(np.float32)
        bin_ends_local = np.where(
            local_idx == (samples_per_interval[..., None] - 1),
            interval_ends[..., None],
            0.5 * (
                tau_local
                + (interval_starts[..., None] + (local_idx + 2.0) * step)
            ),
        ).astype(np.float32)

        # 5. compressed packed indices
        packed_idx = offsets[..., None] + local_idx # idx between [0, n_intervals * n_samples - 1], (h, w, max_intervals, n_sample_max)
        packed_bin_valid = tau_valid & (packed_idx < n_total)

        # 6. flatten and write with per-ray base offset
        ray_idx = np.arange(h * w, dtype=np.int64).reshape(h, w, 1, 1)
        flat_out_idx = ray_idx * n_total + packed_idx.astype(np.int64)
        packed_idx_flat_valid = flat_out_idx[packed_bin_valid]
        far_values = np.asarray(ray_fars, dtype=np.float32).reshape(h * w, 1)
        packed_bins_starts = np.broadcast_to(far_values, (h * w, n_total)).copy().reshape(h * w * n_total)
        packed_bins_ends = np.broadcast_to(far_values, (h * w, n_total)).copy().reshape(h * w * n_total)
        packed_eval_t = np.broadcast_to(far_values, (h * w, n_total)).copy().reshape(h * w * n_total)
        packed_mask = np.zeros((h * w * n_total), dtype=np.uint8)
        packed_bins_starts[packed_idx_flat_valid] = bin_starts_local[packed_bin_valid]
        packed_bins_ends[packed_idx_flat_valid] = bin_ends_local[packed_bin_valid]
        packed_eval_t[packed_idx_flat_valid] = tau_local[packed_bin_valid]
        packed_mask[packed_idx_flat_valid] = 1
        packed_bins_starts = packed_bins_starts.reshape(h, w, n_total)
        packed_bins_ends = packed_bins_ends.reshape(h, w, n_total)
        packed_eval_t = packed_eval_t.reshape(h, w, n_total)
        packed_mask = packed_mask.reshape(h, w, n_total)
        
        return packed_bins_starts, packed_bins_ends, packed_mask, packed_eval_t

    def _compute_tau_local(
        self, interval_starts: ArrayLike, interval_ends: ArrayLike, interval_mask: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
        """Compute paper-style interior tau samples per interval.

        Returns:
            tau_local: (H, W, max_intervals, n_sample_max)
            tau_valid: same shape, valid paper samples
            samples_per_interval: (H, W, max_intervals)
        """
        if torch.is_tensor(interval_starts):
            device = interval_starts.device
            interval_len = interval_ends - interval_starts
            samples_per_interval = (
                torch.ceil(torch.clamp(interval_len - self.single_sample_threshold, min=0.0) / self.sample_spacing)
                .to(torch.int32)
                + 1
            )
            samples_per_interval = torch.clamp(samples_per_interval, 1, self.n_sample_max)
            samples_per_interval = torch.where(
                interval_mask,
                samples_per_interval,
                torch.zeros_like(samples_per_interval),
            ).to(torch.int32)

            local_idx = torch.arange(self.n_sample_max, device=device, dtype=torch.int32)[None, None, None, :]
            tau_valid = local_idx < samples_per_interval[..., None]
            denom = (torch.clamp(samples_per_interval, min=1)[..., None] + 1).to(torch.float32)
            step = interval_len[..., None] / denom
            tau_local = interval_starts[..., None] + (local_idx.to(torch.float32) + 1.0) * step
            return tau_local.to(torch.float32), tau_valid, samples_per_interval

        interval_len = interval_ends - interval_starts
        samples_per_interval = (
            np.ceil(np.maximum(interval_len - self.single_sample_threshold, 0.0) / self.sample_spacing).astype(np.int32)
            + 1
        )
        samples_per_interval = np.clip(samples_per_interval, 1, self.n_sample_max)
        samples_per_interval = np.where(interval_mask, samples_per_interval, 0).astype(np.int32)

        local_idx = np.arange(self.n_sample_max, dtype=np.int32)[None, None, None, :]
        tau_valid = local_idx < samples_per_interval[..., None]
        denom = (np.maximum(samples_per_interval, 1)[..., None] + 1).astype(np.float32)
        step = interval_len[..., None] / denom
        tau_local = interval_starts[..., None] + (local_idx + 1.0) * step
        return tau_local.astype(np.float32), tau_valid, samples_per_interval

    def render_vis(
        self,
        cameras: Cameras,
        camera_index: int,
        debug_path: Optional[str] = None,
        validate_conversion: bool = False,
    ) -> np.ndarray:
        """Render a single-camera OptiX depth map and optionally save debug images."""
        cam_params = self.ns_cam_to_optix_cam(cameras, camera_index)
        depth, hit_t, hit_count, erosion_first_t, dilation_hit_flag = u_ox.launch_pipeline(
            self.pipeline, self.sbt, self.dilation_gas, self.erosion_gas, 
            cam_params, 
            n_sample_max=self.n_sample_max, 
            n_depth_peeling_max=self.n_depth_peeling_max, 
            single_sample_threshold=self.single_sample_threshold, 
            sample_spacing=self.sample_spacing
        )
        height, width, max_hits = hit_t.shape
        num_rays = height * width
        inner_first_t = erosion_first_t.reshape(num_rays)
        if not self.use_inner_mesh:
            inner_first_t = -np.ones_like(inner_first_t, dtype=np.float32)
        interval_starts_flat, interval_ends_flat, interval_mask_flat, interval_count_flat = self.pair_shell_hit_intervals(
            outer_hit_t=hit_t.reshape(num_rays, max_hits),
            outer_hit_flag=dilation_hit_flag.reshape(num_rays, max_hits),
            outer_hit_count=hit_count.reshape(num_rays),
            inner_first_t=inner_first_t,
            ray_tmin=np.full((num_rays,), 1e-4, dtype=np.float32),
            min_interval_size=1e-4,
        )
        max_intervals = interval_starts_flat.shape[-1]
        interval_starts = interval_starts_flat.reshape(height, width, max_intervals)
        interval_ends = interval_ends_flat.reshape(height, width, max_intervals)
        interval_mask = interval_mask_flat.reshape(height, width, max_intervals)
        interval_count = interval_count_flat.reshape(height, width)
        # starts_packed, ends_packed, mask_packed = self.pack_interval_to_sample_bins(
        #     interval_starts, interval_ends, interval_mask, interval_count
        # )

        if validate_conversion and camera_index not in self._validated_camera_indices:
            max_err = self.validate_ns_cam_conversion(cameras, camera_index, cam_params)
            log.info("[OptiX] camera conversion check camera=%d max_dir_err=%.6e", camera_index, max_err)
            self._validated_camera_indices.add(camera_index)

        if debug_path is not None:
            self.save_depth_map(depth, debug_path)
            self.export_interval_geometry(
                cam_params=cam_params,
                interval_starts=interval_starts,
                interval_ends=interval_ends,
                interval_mask=interval_mask,
                interval_count=interval_count,
                path=debug_path,
            )
            self.export_sample_geometry(
                cam_params=cam_params,
                interval_starts=interval_starts,
                interval_ends=interval_ends,
                interval_mask=interval_mask,
                path=debug_path,
            )
        return depth

    # Backward-compatible alias used during teaching iterations.
    def render(
        self,
        cameras: Cameras,
        camera_index: int,
        debug_path: Optional[str] = None,
        validate_conversion: bool = False,
    ) -> np.ndarray:
        return self.render_vis(
            cameras=cameras,
            camera_index=camera_index,
            debug_path=debug_path,
            validate_conversion=validate_conversion,
        )
    
    @staticmethod
    def ns_cam_to_optix_cam(cameras: Cameras, camera_idx: int) -> dict:
        """Convert Nerfstudio pinhole camera fields into OptiX raygen U/V/W contract."""
        c2w = cast(torch.Tensor, cameras.camera_to_worlds[camera_idx]).detach().cpu().to(torch.float32)  # type: ignore[index]
        fx = float(cast(torch.Tensor, cameras.fx[camera_idx]).item())  # type: ignore[index]
        fy = float(cast(torch.Tensor, cameras.fy[camera_idx]).item())  # type: ignore[index]
        cx = float(cast(torch.Tensor, cameras.cx[camera_idx]).item())  # type: ignore[index]
        cy = float(cast(torch.Tensor, cameras.cy[camera_idx]).item())  # type: ignore[index]
        width = int(cast(torch.Tensor, cameras.width[camera_idx]).item())  # type: ignore[index]
        height = int(cast(torch.Tensor, cameras.height[camera_idx]).item())  # type: ignore[index]

        # Nerfstudio ray model:
        # dir_local = [ (x-cx)/fx, -(y-cy)/fy, -1 ]
        # OptiX raygen model:
        # dxy = [2*(x+0.5)/W-1, 2*(y+0.5)/H-1]
        # dir_local = cam_u_local*dxy.x + cam_v_local*dxy.y + cam_w_local
        half_w_minus = 0.5 * float(width) - 0.5
        half_h_minus = 0.5 * float(height) - 0.5
        cam_u_local = torch.tensor([float(width) / (2.0 * fx), 0.0, 0.0], dtype=torch.float32)
        cam_v_local = torch.tensor([0.0, -float(height) / (2.0 * fy), 0.0], dtype=torch.float32)
        cam_w_local = torch.tensor(
            [
                (half_w_minus - cx) / fx,
                (cy - half_h_minus) / fy,
                -1.0,
            ],
            dtype=torch.float32,
        )

        r_mat = c2w[:, :3]
        t_vec = c2w[:, 3]
        cam_eye_np = t_vec.numpy().astype(np.float32)
        cam_u_np = (r_mat @ cam_u_local).numpy().astype(np.float32)
        cam_v_np = (r_mat @ cam_v_local).numpy().astype(np.float32)
        cam_w_np = (r_mat @ cam_w_local).numpy().astype(np.float32)

        return {
            "image_width": width,
            "image_height": height,
            "cam_eye": cam_eye_np,
            "cam_u": cam_u_np,
            "cam_v": cam_v_np,
            "cam_w": cam_w_np,
        }

    @staticmethod
    def save_depth_map(depth_map: np.ndarray, path: str) -> None:
        """Save raw depth and two visualizations; misses (<0) are kept black."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(out_path.with_suffix(".npy")), depth_map.astype(np.float32))

        valid = depth_map > 0.0
        vis = np.zeros_like(depth_map, dtype=np.float32)
        if np.any(valid):
            d_valid = depth_map[valid]
            near, far = np.percentile(d_valid, [2, 98])
            if far <= near:
                far = near + 1e-6
            vis_valid = 1.0 - np.clip((d_valid - near) / (far - near), 0.0, 1.0)
            vis[valid] = vis_valid

        vis_u8 = (vis * 255.0).astype(np.uint8)
        vis_jet = cv2.applyColorMap(vis_u8, cv2.COLORMAP_JET)  # type: ignore[attr-defined]
        vis_jet[~valid] = 0
        cv2.imwrite(str(out_path), vis_u8)  # type: ignore[attr-defined]
        cv2.imwrite(str(out_path.with_name(out_path.stem + "_jet.png")), vis_jet)  # type: ignore[attr-defined]

    @staticmethod
    def _compute_primary_rays(cam_params: dict) -> np.ndarray:
        """Recreate all primary ray directions on host for geometry export."""
        width = int(cam_params["image_width"])
        height = int(cam_params["image_height"])
        xs = np.arange(width, dtype=np.float32) + 0.5
        ys = np.arange(height, dtype=np.float32) + 0.5
        px, py = np.meshgrid(xs, ys, indexing="xy")
        dxy_x = 2.0 * (px / float(width)) - 1.0
        dxy_y = 2.0 * (py / float(height)) - 1.0
        dirs = (
            cam_params["cam_u"][None, None, :] * dxy_x[..., None]
            + cam_params["cam_v"][None, None, :] * dxy_y[..., None]
            + cam_params["cam_w"][None, None, :]
        ).astype(np.float32)
        norms = np.linalg.norm(dirs, axis=-1, keepdims=True) + 1e-12
        return dirs / norms

    def export_interval_geometry(
        self,
        cam_params: dict,
        interval_starts: np.ndarray,
        interval_ends: np.ndarray,
        interval_mask: np.ndarray,
        interval_count: np.ndarray,
        path: str,
        ray_stride: int = 4,
        max_segments: int = 12000,
        line_radius: float = 0.0005,
    ) -> None:
        """Export valid intervals as colored endpoints and thin cylinder segments."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        valid_idx = self._select_interval_export_indices(
            interval_mask=interval_mask,
            ray_stride=ray_stride,
            max_intervals=max_segments,
        )
        if valid_idx.shape[0] == 0:
            return

        rays = self._compute_primary_rays(cam_params)
        cam_eye = np.asarray(cam_params["cam_eye"], dtype=np.float32)

        ys = valid_idx[:, 0]
        xs = valid_idx[:, 1]
        interval_ids = valid_idx[:, 2]
        ray_dirs = rays[ys, xs].astype(np.float32)

        starts = interval_starts[ys, xs, interval_ids].astype(np.float32)
        ends = interval_ends[ys, xs, interval_ids].astype(np.float32)
        start_pts = cam_eye[None, :] + ray_dirs * starts[:, None]
        end_pts = cam_eye[None, :] + ray_dirs * ends[:, None]

        if start_pts.shape[0] > max_segments:
            choose = np.linspace(0, start_pts.shape[0] - 1, max_segments, dtype=np.int64)
            start_pts = start_pts[choose]
            end_pts = end_pts[choose]
            starts = starts[choose]
            ends = ends[choose]
            ray_dirs = ray_dirs[choose]

        endpoint_points = np.concatenate([start_pts, end_pts], axis=0).astype(np.float32)
        start_colors = np.repeat(np.array([[0, 255, 0, 255]], dtype=np.uint8), start_pts.shape[0], axis=0)
        end_colors = np.repeat(np.array([[255, 0, 0, 255]], dtype=np.uint8), end_pts.shape[0], axis=0)
        endpoint_colors = np.concatenate([start_colors, end_colors], axis=0)
        endpoint_cloud = trimesh.points.PointCloud(vertices=endpoint_points, colors=endpoint_colors)
        endpoint_cloud.export(str(out_path.with_name(out_path.stem + "_interval_endpoints.ply")))

        try:
            scene = trimesh.Scene()
            scene.add_geometry(endpoint_cloud, node_name="interval_endpoints")

            # GLB viewers often ignore line primitives, so export thin cylinders instead.
            line_meshes = []
            for a, b in zip(start_pts, end_pts):
                seg_len = float(np.linalg.norm(b - a))
                if seg_len <= 1e-8:
                    continue
                line_meshes.append(
                    trimesh.creation.cylinder(
                        radius=line_radius,
                        segment=np.stack([a, b], axis=0),
                        sections=6,
                    )
                )
            if line_meshes:
                line_mesh = trimesh.util.concatenate(line_meshes)
                line_mesh.visual.vertex_colors = np.tile(
                    np.array([[220, 220, 220, 255]], dtype=np.uint8), (line_mesh.vertices.shape[0], 1)
                )
                scene.add_geometry(line_mesh, node_name="interval_lines")
            scene.export(str(out_path.with_name(out_path.stem + "_intervals.glb")))
        except Exception as err:
            log.warning("[OptiX] failed to export interval line scene: %s", err)

    def export_sample_geometry(
        self,
        cam_params: dict,
        interval_starts: np.ndarray,
        interval_ends: np.ndarray,
        interval_mask: np.ndarray,
        path: str,
        ray_stride: int = 4,
        max_intervals: int = 12000,
        max_points: int = 60000,
    ) -> None:
        """Export paper tau samples for the same selected intervals as the GLB."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        tau_local, tau_valid, _ = self._compute_tau_local(interval_starts, interval_ends, interval_mask)
        selected_interval_idx = self._select_interval_export_indices(
            interval_mask=interval_mask,
            ray_stride=ray_stride,
            max_intervals=max_intervals,
        )
        if selected_interval_idx.shape[0] == 0:
            return

        ys = selected_interval_idx[:, 0]
        xs = selected_interval_idx[:, 1]
        interval_ids = selected_interval_idx[:, 2]
        selected_tau_valid = tau_valid[ys, xs, interval_ids]
        sample_coords = np.argwhere(selected_tau_valid)
        if sample_coords.shape[0] == 0:
            return

        valid_idx = np.column_stack(
            [
                ys[sample_coords[:, 0]],
                xs[sample_coords[:, 0]],
                interval_ids[sample_coords[:, 0]],
                sample_coords[:, 1],
            ]
        )

        if valid_idx.shape[0] > max_points:
            choose = np.linspace(0, valid_idx.shape[0] - 1, max_points, dtype=np.int64)
            valid_idx = valid_idx[choose]

        rays = self._compute_primary_rays(cam_params)
        cam_eye = np.asarray(cam_params["cam_eye"], dtype=np.float32)

        ys = valid_idx[:, 0]
        xs = valid_idx[:, 1]
        interval_ids = valid_idx[:, 2]
        sample_ids = valid_idx[:, 3]

        ray_dirs = rays[ys, xs].astype(np.float32)
        tau = tau_local[ys, xs, interval_ids, sample_ids].astype(np.float32)
        sample_pts = cam_eye[None, :] + ray_dirs * tau[:, None]

        denom = float(max(self.n_sample_max - 1, 1))
        sample_norm = np.clip(sample_ids.astype(np.float32) / denom, 0.0, 1.0)
        colors = np.stack(
            [
                40.0 + 40.0 * sample_norm,
                180.0 + 60.0 * (1.0 - sample_norm),
                255.0 * np.ones_like(sample_norm),
                255.0 * np.ones_like(sample_norm),
            ],
            axis=-1,
        ).astype(np.uint8)

        sample_cloud = trimesh.points.PointCloud(vertices=sample_pts.astype(np.float32), colors=colors)
        sample_cloud.export(str(out_path.with_name(out_path.stem + "_sample_points.ply")))

    @staticmethod
    def _select_interval_export_indices(
        interval_mask: np.ndarray,
        ray_stride: int,
        max_intervals: int,
    ) -> np.ndarray:
        """Select a stable subset of valid intervals for debug exports."""
        valid_idx = np.argwhere(interval_mask)
        if valid_idx.shape[0] == 0:
            return valid_idx

        if ray_stride > 1:
            keep = (valid_idx[:, 0] % ray_stride == 0) & (valid_idx[:, 1] % ray_stride == 0)
            valid_idx = valid_idx[keep]
            if valid_idx.shape[0] == 0:
                return valid_idx

        if valid_idx.shape[0] > max_intervals:
            choose = np.linspace(0, valid_idx.shape[0] - 1, max_intervals, dtype=np.int64)
            valid_idx = valid_idx[choose]

        return valid_idx

    @staticmethod
    def _optix_dir_from_params(cam_params: dict, px: float, py: float) -> np.ndarray:
        """Recreate raygen direction on host for validation."""
        width = float(cam_params["image_width"])
        height = float(cam_params["image_height"])
        dxy_x = 2.0 * (px / width) - 1.0
        dxy_y = 2.0 * (py / height) - 1.0
        direction = (
            cam_params["cam_u"] * dxy_x
            + cam_params["cam_v"] * dxy_y
            + cam_params["cam_w"]
        )
        norm = np.linalg.norm(direction) + 1e-12
        return (direction / norm).astype(np.float32)

    @staticmethod
    def _ns_ray_dir(cameras: Cameras, camera_idx: int, px: float, py: float) -> np.ndarray:
        coords = torch.tensor([[[py, px]]], dtype=torch.float32, device=cameras.device)
        rb = cameras.generate_rays(camera_indices=camera_idx, coords=cast(Any, coords))
        d = cast(torch.Tensor, rb.directions).reshape(-1, 3)[0].detach().cpu().numpy().astype(np.float32)
        return d / (np.linalg.norm(d) + 1e-12)

    def validate_ns_cam_conversion(self, cameras: Cameras, camera_idx: int, cam_params: Optional[dict] = None) -> float:
        """Compare OptiX camera rays against Nerfstudio generate_rays at 3 probe pixels."""
        if cam_params is None:
            cam_params = self.ns_cam_to_optix_cam(cameras, camera_idx)
        width = int(cam_params["image_width"])
        height = int(cam_params["image_height"])

        probes: Tuple[Tuple[float, float], ...] = (
            (0.5, 0.5),
            (0.5 * width, 0.5 * height),
            (width - 0.5, height - 0.5),
        )
        max_err = 0.0
        for px, py in probes:
            d_optix = self._optix_dir_from_params(cam_params, px, py)
            d_ns = self._ns_ray_dir(cameras, camera_idx, px, py)
            err = float(np.linalg.norm(d_optix - d_ns))
            max_err = max(max_err, err)
        return max_err

    def trace_and_pack(
        self, origins: ArrayLike, directions: ArrayLike, nears: ArrayLike, fars: ArrayLike, return_torch: bool = False
    ) -> Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]:
        """Trace an arbitrary ray batch and return packed bins compatible with `ShellSampler`."""
        if origins.ndim != 2 or origins.shape[-1] != 3:
            raise ValueError(f"Expected origins with shape (num_rays, 3), got {origins.shape}.")
        if directions.ndim != 2 or directions.shape[-1] != 3:
            raise ValueError(f"Expected directions with shape (num_rays, 3), got {directions.shape}.")
        if nears.ndim != 1 or fars.ndim != 1:
            raise ValueError("Expected `nears` and `fars` with shape (num_rays,).")
        num_rays = int(origins.shape[0])
        if directions.shape[0] != num_rays or nears.shape[0] != num_rays or fars.shape[0] != num_rays:
            raise ValueError("Origins, directions, nears, and fars must have the same number of rays.")

        launch_start = time.perf_counter()
        _, hit_t, hit_count, erosion_first_t, dilation_hit_flag, launch_profile = u_ox.launch_pipeline_ray_batch(
            self.pipeline,
            self.sbt,
            self.dilation_gas,
            self.erosion_gas,
            origins,
            directions,
            nears,
            fars,
            n_depth_peeling_max=self.n_depth_peeling_max,
            return_torch=return_torch,
        )
        launch_ms = (time.perf_counter() - launch_start) * 1000.0
        if return_torch:
            hit_t_t = hit_t.to(dtype=torch.float32)
            hit_count_t = hit_count.to(dtype=torch.int32)
            erosion_first_t_t = erosion_first_t.to(dtype=torch.float32)
            dilation_hit_flag_t = dilation_hit_flag.to(dtype=torch.int32)
            nears_t = nears.to(dtype=torch.float32) if torch.is_tensor(nears) else torch.from_numpy(nears).cuda().to(torch.float32)
            fars_t = fars.to(dtype=torch.float32) if torch.is_tensor(fars) else torch.from_numpy(fars).cuda().to(torch.float32)
            upload_ms = 0.0
            profile_device = nears_t.device

            inner_first_t_t = erosion_first_t_t
            if not self.use_inner_mesh:
                inner_first_t_t = -torch.ones_like(inner_first_t_t, dtype=torch.float32)
            self._sync_cuda_for_profile(profile_device)
            pair_start = time.perf_counter()
            interval_starts_flat, interval_ends_flat, interval_mask_flat, interval_count_flat = self.pair_shell_hit_intervals(
                outer_hit_t=hit_t_t,
                outer_hit_flag=dilation_hit_flag_t,
                outer_hit_count=hit_count_t,
                inner_first_t=inner_first_t_t,
                ray_tmin=nears_t,
                min_interval_size=1e-4,
            )
            self._sync_cuda_for_profile(profile_device)
            pair_ms = (time.perf_counter() - pair_start) * 1000.0
            interval_starts = interval_starts_flat[:, None, :]
            interval_ends = interval_ends_flat[:, None, :]
            interval_mask = interval_mask_flat[:, None, :]
            interval_count = interval_count_flat[:, None]
            self._sync_cuda_for_profile(profile_device)
            pack_start = time.perf_counter()
            starts_packed, ends_packed, mask_packed, eval_t_packed = self.pack_interval_to_sample_bins(
                interval_starts, interval_ends, interval_mask, interval_count, fars_t[:, None]
            )
            self._sync_cuda_for_profile(profile_device)
            pack_ms = (time.perf_counter() - pack_start) * 1000.0

            starts = starts_packed[:, 0, :]
            ends = ends_packed[:, 0, :]
            mask = mask_packed[:, 0, :].to(torch.float32)
            eval_t = eval_t_packed[:, 0, :]

            self._last_trace_profile = {
                "optix_launch_ms": launch_ms,
                "interval_pair_ms": pair_ms,
                "sample_pack_ms": pack_ms,
                "far_fill_ms": 0.0,
                "trace_h2d_ms": upload_ms,
                "zero_copy_optix_interop": float(1.0 if launch_profile.get("interop_zero_copy", False) else 0.0),
            }
            return starts, ends, mask, eval_t

        inner_first_t = erosion_first_t
        if not self.use_inner_mesh:
            inner_first_t = -np.ones_like(inner_first_t, dtype=np.float32)
        pair_start = time.perf_counter()
        interval_starts_flat, interval_ends_flat, interval_mask_flat, interval_count_flat = self.pair_shell_hit_intervals(
            outer_hit_t=hit_t,
            outer_hit_flag=dilation_hit_flag,
            outer_hit_count=hit_count,
            inner_first_t=inner_first_t,
            ray_tmin=nears,
            min_interval_size=1e-4,
        )
        pair_ms = (time.perf_counter() - pair_start) * 1000.0
        interval_starts = interval_starts_flat[:, None, :]
        interval_ends = interval_ends_flat[:, None, :]
        interval_mask = interval_mask_flat[:, None, :]
        interval_count = interval_count_flat[:, None]
        pack_start = time.perf_counter()
        starts_packed, ends_packed, mask_packed, eval_t_packed = self.pack_interval_to_sample_bins(
            interval_starts, interval_ends, interval_mask, interval_count, fars[:, None]
        )
        pack_ms = (time.perf_counter() - pack_start) * 1000.0

        starts = starts_packed[:, 0, :].astype(np.float32)
        ends = ends_packed[:, 0, :].astype(np.float32)
        mask = mask_packed[:, 0, :].astype(np.float32)
        eval_t = eval_t_packed[:, 0, :].astype(np.float32)

        self._last_trace_profile = {
            "optix_launch_ms": launch_ms,
            "interval_pair_ms": pair_ms,
            "sample_pack_ms": pack_ms,
            "far_fill_ms": 0.0,
            "trace_h2d_ms": 0.0,
            "zero_copy_optix_interop": float(0.0),
        }
        return starts, ends, mask, eval_t

    def get_last_trace_profile(self) -> Optional[dict]:
        return self._last_trace_profile

        
if __name__ == "__main__":
    dilation_mesh = cast(trimesh.Trimesh, trimesh.load("meshes/shell_debug-dilation.ply", file_type="ply"))
    erosion_mesh = cast(trimesh.Trimesh, trimesh.load("meshes/shell_debug-erosion.ply", file_type="ply"))
    tracer = ShellRayTracer(dilation_mesh, erosion_mesh)
