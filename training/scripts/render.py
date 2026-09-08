#!/usr/bin/env python
# Modified for Adaptive Shells training and rendering; see the repository NOTICE.
"""
render.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import mediapy as media
import numpy as np
import torch
import tyro
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from typing_extensions import Literal, assert_never

from nerfstudio.cameras.camera_paths import get_path_from_json, get_spiral_path
from nerfstudio.cameras.cameras import Cameras
from nerfstudio.configs.base_config import Config  # pylint: disable=unused-import
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import install_checks
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.utils.rich_utils import ItersPerSecColumn

CONSOLE = Console(width=120)


def _fmt_terms(row: dict, items: List[tuple[str, str]], threshold_ms: float = 0.05) -> str:
    parts = []
    for label, key in items:
        value = float(row.get(key, 0.0))
        if abs(value) >= threshold_ms:
            parts.append(f"{label}={value:.1f}")
    return " + ".join(parts) if parts else "0.0"


def _render_trajectory_video(
    pipeline: Pipeline,
    cameras: Cameras,
    output_filename: Path,
    rendered_output_names: List[str],
    rendered_resolution_scaling_factor: float = 1.0,
    seconds: float = 5.0,
    output_format: Literal["images", "video"] = "video",
    strict_shell_profile: bool = False,
) -> None:
    """Helper function to create a video of the spiral trajectory.

    Args:
        pipeline: Pipeline to evaluate with.
        cameras: Cameras to render.
        output_filename: Name of the output file.
        rendered_output_names: List of outputs to visualise.
        rendered_resolution_scaling_factor: Scaling factor to apply to the camera image resolution.
        seconds: Length of output video.
        output_format: How to save output data.
        strict_shell_profile: If True, synchronize CUDA between profiled regions for additive wall-clock timings.
    """
    CONSOLE.print("[bold green]Creating trajectory video")
    if hasattr(pipeline.model, "set_shell_profile_options"):
        pipeline.model.set_shell_profile_options(strict_timing=strict_shell_profile)
    if strict_shell_profile:
        CONSOLE.print("[cyan]Shell profiling mode: strict wall-clock (CUDA synchronized)")
    images = []
    cameras.rescale_output_resolution(rendered_resolution_scaling_factor)
    cameras = cameras.to(pipeline.device)

    progress = Progress(
        TextColumn(":movie_camera: Rendering :movie_camera:"),
        BarColumn(),
        TaskProgressColumn(show_speed=True),
        ItersPerSecColumn(suffix="fps"),
        TimeRemainingColumn(elapsed_when_finished=True, compact=True),
    )
    output_image_dir = output_filename.parent / output_filename.stem
    profile_rows = []
    if output_format == "images":
        output_image_dir.mkdir(parents=True, exist_ok=True)
    with progress:
        for camera_idx in progress.track(range(cameras.size), description=""):
            camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx)
            if hasattr(pipeline.model, "reset_shell_profile"):
                pipeline.model.reset_shell_profile()
            if strict_shell_profile and pipeline.device.type == "cuda":
                torch.cuda.synchronize(device=pipeline.device)
            frame_start = time.perf_counter()
            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)
            if strict_shell_profile and pipeline.device.type == "cuda":
                torch.cuda.synchronize(device=pipeline.device)
            frame_total_ms = (time.perf_counter() - frame_start) * 1000.0
            if hasattr(pipeline.model, "get_shell_profile_summary"):
                shell_profile = pipeline.model.get_shell_profile_summary()
                if shell_profile.get("chunks", 0) > 0:
                    profile_row = {"frame_idx": int(camera_idx), "frame_total_ms": frame_total_ms, **shell_profile}
                    profile_rows.append(profile_row)
                    frame_driver_other_ms = max(
                        frame_total_ms - float(shell_profile.get("get_outputs_total_ms", 0.0)),
                        0.0,
                    )
                    trace_terms = _fmt_terms(
                        shell_profile,
                        [
                            ("input", "input_prep_ms"),
                            ("interop", "trace_h2d_ms"),
                            ("optix", "optix_launch_ms"),
                            ("pair", "interval_pair_ms"),
                            ("pack", "sample_pack_ms"),
                            ("fill", "far_fill_ms"),
                            ("other", "trace_other_ms"),
                        ],
                    )
                    sampler_terms = _fmt_terms(
                        shell_profile,
                        [
                            ("trace", "trace_and_pack_ms"),
                            ("count", "valid_count_reduce_ms"),
                            ("count_d2h", "valid_count_d2h_ms"),
                            ("compact", "compact_indices_ms"),
                            ("shell_meta", "shell_count_meta_ms"),
                            ("gather", "sample_gather_ms"),
                            ("build", "ray_samples_build_ms"),
                            ("hist", "valid_count_hist_ms"),
                            ("other", "sampler_other_ms"),
                        ],
                    )
                    sample_fwd_terms = _fmt_terms(
                        shell_profile,
                        [
                            ("sampler", "sampler_total_ms"),
                            ("field", "field_eval_ms"),
                            ("alpha", "alpha_weights_ms"),
                            ("merge", "sampler_profile_merge_ms"),
                            ("payload", "sample_return_build_ms"),
                            ("other", "sample_forward_other_ms"),
                        ],
                    )
                    field_terms = _fmt_terms(
                        shell_profile,
                        [
                            ("geonet", "field_geonet_ms"),
                            ("grad", "field_gradient_ms"),
                            ("color", "field_color_ms"),
                            ("alpha_in", "field_alpha_ms"),
                            ("other", "field_other_ms"),
                        ],
                    )
                    get_outputs_terms = _fmt_terms(
                        shell_profile,
                        [
                            ("sample_fwd", "sample_forward_total_ms"),
                            ("compose", "outputs_compose_ms"),
                            ("post", "outputs_post_ms"),
                            ("other", "get_outputs_other_ms"),
                        ],
                    )
                    CONSOLE.print(
                        "[cyan]"
                        f"[shell-profile frame={camera_idx}] "
                        f"frame_ms={frame_total_ms:.1f} = model_ms({shell_profile.get('get_outputs_total_ms', 0.0):.1f}) + "
                        f"driver_other({frame_driver_other_ms:.1f})"
                    )
                    CONSOLE.print(
                        "[cyan]"
                        f"  model_ms={shell_profile.get('get_outputs_total_ms', 0.0):.1f} = {get_outputs_terms}"
                    )
                    CONSOLE.print(
                        "[cyan]"
                        f"  sample_fwd={shell_profile.get('sample_forward_total_ms', 0.0):.1f} = {sample_fwd_terms}"
                    )
                    CONSOLE.print(
                        "[cyan]"
                        f"  field={shell_profile.get('field_eval_ms', 0.0):.1f} = {field_terms}"
                    )
                    CONSOLE.print(
                        "[cyan]"
                        f"  sampler={shell_profile['sampler_total_ms']:.1f} = {sampler_terms}"
                    )
                    CONSOLE.print(
                        "[cyan]"
                        f"  trace={shell_profile['trace_and_pack_ms']:.1f} = {trace_terms}; "
                        f"mean_valid={shell_profile['mean_valid_per_ray']:.2f} "
                        f"fg_mean={shell_profile.get('mean_valid_per_hit_ray', 0.0):.2f} "
                        f"p95={shell_profile['p95_valid_per_ray']} "
                        f"single_hit={100.0 * shell_profile.get('single_sample_hit_frac', 0.0):.1f}% "
                        f"single_grad={shell_profile.get('single_sample_grad_norm_mean', 0.0):.2f}"
                        f"+/-{shell_profile.get('single_sample_grad_norm_std', 0.0):.2f} "
                        f"|d.g|={shell_profile.get('single_sample_abs_true_cos_mean', 0.0):.2f} "
                        f"util={100.0 * shell_profile['utilization']:.2f}% "
                        f"budget/ray={shell_profile.get('budget_per_ray', 0.0):.1f}"
                    )
            render_image = []
            for rendered_output_name in rendered_output_names:
                if rendered_output_name not in outputs:
                    CONSOLE.rule("Error", style="red")
                    CONSOLE.print(f"Could not find {rendered_output_name} in the model outputs", justify="center")
                    CONSOLE.print(f"Please set --rendered_output_name to one of: {outputs.keys()}", justify="center")
                    sys.exit(1)
                output_image = outputs[rendered_output_name].cpu().numpy()
                render_image.append(output_image)
            render_image = np.concatenate(render_image, axis=1)
            if output_format == "images":
                media.write_image(output_image_dir / f"{camera_idx:05d}.png", render_image)
            else:
                images.append(render_image)

    if output_format == "video":
        fps = len(images) / seconds
        # make the folder if it doesn't exist
        output_filename.parent.mkdir(parents=True, exist_ok=True)
        with CONSOLE.status("[yellow]Saving video", spinner="bouncingBall"):
            media.write_video(output_filename, images, fps=fps)
    if profile_rows:
        frame_total_ms_mean = float(np.mean([row["frame_total_ms"] for row in profile_rows]))
        get_outputs_total_ms_mean = float(np.mean([row.get("get_outputs_total_ms", 0.0) for row in profile_rows]))
        sample_forward_total_ms_mean = float(np.mean([row.get("sample_forward_total_ms", 0.0) for row in profile_rows]))
        sample_forward_other_ms_mean = float(np.mean([row.get("sample_forward_other_ms", 0.0) for row in profile_rows]))
        trace_and_pack_ms_mean = float(np.mean([row["trace_and_pack_ms"] for row in profile_rows]))
        trace_other_ms_mean = float(np.mean([row.get("trace_other_ms", 0.0) for row in profile_rows]))
        sampler_total_ms_mean = float(np.mean([row["sampler_total_ms"] for row in profile_rows]))
        sampler_other_ms_mean = float(np.mean([row.get("sampler_other_ms", 0.0) for row in profile_rows]))
        torch_to_numpy_ms_mean = float(np.mean([row.get("torch_to_numpy_ms", 0.0) for row in profile_rows]))
        numpy_nonzero_ms_mean = float(np.mean([row.get("numpy_nonzero_ms", 0.0) for row in profile_rows]))
        numpy_to_torch_ms_mean = float(np.mean([row.get("numpy_to_torch_ms", 0.0) for row in profile_rows]))
        input_prep_ms_mean = float(np.mean([row.get("input_prep_ms", 0.0) for row in profile_rows]))
        valid_count_reduce_ms_mean = float(np.mean([row.get("valid_count_reduce_ms", 0.0) for row in profile_rows]))
        valid_count_d2h_ms_mean = float(np.mean([row.get("valid_count_d2h_ms", 0.0) for row in profile_rows]))
        valid_count_hist_ms_mean = float(np.mean([row.get("valid_count_hist_ms", 0.0) for row in profile_rows]))
        compact_indices_ms_mean = float(np.mean([row.get("compact_indices_ms", 0.0) for row in profile_rows]))
        shell_count_meta_ms_mean = float(np.mean([row.get("shell_count_meta_ms", 0.0) for row in profile_rows]))
        sample_gather_ms_mean = float(np.mean([row.get("sample_gather_ms", 0.0) for row in profile_rows]))
        ray_samples_build_ms_mean = float(np.mean([row.get("ray_samples_build_ms", 0.0) for row in profile_rows]))
        optix_launch_ms_mean = float(np.mean([row.get("optix_launch_ms", 0.0) for row in profile_rows]))
        interval_pair_ms_mean = float(np.mean([row.get("interval_pair_ms", 0.0) for row in profile_rows]))
        sample_pack_ms_mean = float(np.mean([row.get("sample_pack_ms", 0.0) for row in profile_rows]))
        far_fill_ms_mean = float(np.mean([row.get("far_fill_ms", 0.0) for row in profile_rows]))
        trace_h2d_ms_mean = float(np.mean([row.get("trace_h2d_ms", 0.0) for row in profile_rows]))
        outputs_compose_ms_mean = float(np.mean([row.get("outputs_compose_ms", 0.0) for row in profile_rows]))
        outputs_post_ms_mean = float(np.mean([row.get("outputs_post_ms", 0.0) for row in profile_rows]))
        get_outputs_other_ms_mean = float(np.mean([row.get("get_outputs_other_ms", 0.0) for row in profile_rows]))
        field_eval_ms_mean = float(np.mean([row["field_eval_ms"] for row in profile_rows]))
        field_geonet_ms_mean = float(np.mean([row.get("field_geonet_ms", 0.0) for row in profile_rows]))
        field_gradient_ms_mean = float(np.mean([row.get("field_gradient_ms", 0.0) for row in profile_rows]))
        field_color_ms_mean = float(np.mean([row.get("field_color_ms", 0.0) for row in profile_rows]))
        field_alpha_ms_mean = float(np.mean([row.get("field_alpha_ms", 0.0) for row in profile_rows]))
        field_other_ms_mean = float(np.mean([row.get("field_other_ms", 0.0) for row in profile_rows]))
        alpha_weights_ms_mean = float(np.mean([row["alpha_weights_ms"] for row in profile_rows]))
        sampler_profile_merge_ms_mean = float(np.mean([row.get("sampler_profile_merge_ms", 0.0) for row in profile_rows]))
        sample_return_build_ms_mean = float(np.mean([row.get("sample_return_build_ms", 0.0) for row in profile_rows]))
        profile_path = output_filename.parent / f"{output_filename.stem}_shell_profile.json"
        aggregate = {
            "frames": len(profile_rows),
            "strict_shell_profile": bool(strict_shell_profile),
            "frame_total_ms_mean": frame_total_ms_mean,
            "get_outputs_total_ms_mean": get_outputs_total_ms_mean,
            "sample_forward_total_ms_mean": sample_forward_total_ms_mean,
            "sample_forward_other_ms_mean": sample_forward_other_ms_mean,
            "trace_and_pack_ms_mean": trace_and_pack_ms_mean,
            "trace_other_ms_mean": trace_other_ms_mean,
            "sampler_total_ms_mean": sampler_total_ms_mean,
            "sampler_other_ms_mean": sampler_other_ms_mean,
            "torch_to_numpy_ms_mean": torch_to_numpy_ms_mean,
            "numpy_nonzero_ms_mean": numpy_nonzero_ms_mean,
            "numpy_to_torch_ms_mean": numpy_to_torch_ms_mean,
            "input_prep_ms_mean": input_prep_ms_mean,
            "valid_count_reduce_ms_mean": valid_count_reduce_ms_mean,
            "valid_count_d2h_ms_mean": valid_count_d2h_ms_mean,
            "valid_count_hist_ms_mean": valid_count_hist_ms_mean,
            "compact_indices_ms_mean": compact_indices_ms_mean,
            "shell_count_meta_ms_mean": shell_count_meta_ms_mean,
            "sample_gather_ms_mean": sample_gather_ms_mean,
            "ray_samples_build_ms_mean": ray_samples_build_ms_mean,
            "optix_launch_ms_mean": optix_launch_ms_mean,
            "interval_pair_ms_mean": interval_pair_ms_mean,
            "sample_pack_ms_mean": sample_pack_ms_mean,
            "far_fill_ms_mean": far_fill_ms_mean,
            "trace_h2d_ms_mean": trace_h2d_ms_mean,
            "outputs_compose_ms_mean": outputs_compose_ms_mean,
            "outputs_post_ms_mean": outputs_post_ms_mean,
            "get_outputs_other_ms_mean": get_outputs_other_ms_mean,
            "field_eval_ms_mean": field_eval_ms_mean,
            "field_geonet_ms_mean": field_geonet_ms_mean,
            "field_gradient_ms_mean": field_gradient_ms_mean,
            "field_color_ms_mean": field_color_ms_mean,
            "field_alpha_ms_mean": field_alpha_ms_mean,
            "field_other_ms_mean": field_other_ms_mean,
            "alpha_weights_ms_mean": alpha_weights_ms_mean,
            "sampler_profile_merge_ms_mean": sampler_profile_merge_ms_mean,
            "sample_return_build_ms_mean": sample_return_build_ms_mean,
            "mean_valid_per_ray_mean": float(np.mean([row["mean_valid_per_ray"] for row in profile_rows])),
            "mean_valid_per_hit_ray_mean": float(np.mean([row.get("mean_valid_per_hit_ray", 0.0) for row in profile_rows])),
            "p95_valid_per_ray_max": int(max(row["p95_valid_per_ray"] for row in profile_rows)),
            "budget_per_ray_mean": float(np.mean([row.get("budget_per_ray", 0.0) for row in profile_rows])),
            "utilization_mean": float(np.mean([row["utilization"] for row in profile_rows])),
            "single_sample_hit_frac_mean": float(np.mean([row.get("single_sample_hit_frac", 0.0) for row in profile_rows])),
            "single_sample_grad_norm_mean": float(
                np.mean([row.get("single_sample_grad_norm_mean", 0.0) for row in profile_rows])
            ),
            "single_sample_grad_norm_std_mean": float(
                np.mean([row.get("single_sample_grad_norm_std", 0.0) for row in profile_rows])
            ),
            "single_sample_abs_true_cos_mean": float(
                np.mean([row.get("single_sample_abs_true_cos_mean", 0.0) for row in profile_rows])
            ),
            "frame_driver_other_ms_mean": max(frame_total_ms_mean - get_outputs_total_ms_mean, 0.0),
            "trace_and_pack_ratio": trace_and_pack_ms_mean / max(frame_total_ms_mean, 1e-6),
            "trace_other_ratio": trace_other_ms_mean / max(frame_total_ms_mean, 1e-6),
            "sampler_total_ratio": sampler_total_ms_mean / max(frame_total_ms_mean, 1e-6),
            "sampler_other_ratio": sampler_other_ms_mean / max(frame_total_ms_mean, 1e-6),
            "sample_forward_total_ratio": sample_forward_total_ms_mean / max(frame_total_ms_mean, 1e-6),
            "sample_forward_other_ratio": sample_forward_other_ms_mean / max(frame_total_ms_mean, 1e-6),
            "get_outputs_total_ratio": get_outputs_total_ms_mean / max(frame_total_ms_mean, 1e-6),
            "get_outputs_other_ratio": get_outputs_other_ms_mean / max(frame_total_ms_mean, 1e-6),
            "torch_to_numpy_ratio": torch_to_numpy_ms_mean / max(frame_total_ms_mean, 1e-6),
            "numpy_nonzero_ratio": numpy_nonzero_ms_mean / max(frame_total_ms_mean, 1e-6),
            "numpy_to_torch_ratio": numpy_to_torch_ms_mean / max(frame_total_ms_mean, 1e-6),
            "input_prep_ratio": input_prep_ms_mean / max(frame_total_ms_mean, 1e-6),
            "valid_count_reduce_ratio": valid_count_reduce_ms_mean / max(frame_total_ms_mean, 1e-6),
            "valid_count_d2h_ratio": valid_count_d2h_ms_mean / max(frame_total_ms_mean, 1e-6),
            "valid_count_hist_ratio": valid_count_hist_ms_mean / max(frame_total_ms_mean, 1e-6),
            "compact_indices_ratio": compact_indices_ms_mean / max(frame_total_ms_mean, 1e-6),
            "shell_count_meta_ratio": shell_count_meta_ms_mean / max(frame_total_ms_mean, 1e-6),
            "sample_gather_ratio": sample_gather_ms_mean / max(frame_total_ms_mean, 1e-6),
            "ray_samples_build_ratio": ray_samples_build_ms_mean / max(frame_total_ms_mean, 1e-6),
            "optix_launch_ratio": optix_launch_ms_mean / max(frame_total_ms_mean, 1e-6),
            "interval_pair_ratio": interval_pair_ms_mean / max(frame_total_ms_mean, 1e-6),
            "sample_pack_ratio": sample_pack_ms_mean / max(frame_total_ms_mean, 1e-6),
            "far_fill_ratio": far_fill_ms_mean / max(frame_total_ms_mean, 1e-6),
            "trace_h2d_ratio": trace_h2d_ms_mean / max(frame_total_ms_mean, 1e-6),
            "outputs_compose_ratio": outputs_compose_ms_mean / max(frame_total_ms_mean, 1e-6),
            "outputs_post_ratio": outputs_post_ms_mean / max(frame_total_ms_mean, 1e-6),
            "field_eval_ratio": field_eval_ms_mean / max(frame_total_ms_mean, 1e-6),
            "field_geonet_ratio": field_geonet_ms_mean / max(frame_total_ms_mean, 1e-6),
            "field_gradient_ratio": field_gradient_ms_mean / max(frame_total_ms_mean, 1e-6),
            "field_color_ratio": field_color_ms_mean / max(frame_total_ms_mean, 1e-6),
            "field_alpha_ratio": field_alpha_ms_mean / max(frame_total_ms_mean, 1e-6),
            "field_other_ratio": field_other_ms_mean / max(frame_total_ms_mean, 1e-6),
            "alpha_weights_ratio": alpha_weights_ms_mean / max(frame_total_ms_mean, 1e-6),
            "sampler_profile_merge_ratio": sampler_profile_merge_ms_mean / max(frame_total_ms_mean, 1e-6),
            "sample_return_build_ratio": sample_return_build_ms_mean / max(frame_total_ms_mean, 1e-6),
            "frame_driver_other_ratio": max(frame_total_ms_mean - get_outputs_total_ms_mean, 0.0) / max(frame_total_ms_mean, 1e-6),
        }
        profile_path.write_text(json.dumps({"aggregate": aggregate, "frames": profile_rows}, indent=2), encoding="utf-8")
        CONSOLE.print(f"[green]Saved shell profile to {profile_path}")
    CONSOLE.rule("[green] :tada: :tada: :tada: Success :tada: :tada: :tada:")
    CONSOLE.print(f"[green]Saved video to {output_filename}", justify="center")


@dataclass
class RenderTrajectory:
    """Load a checkpoint, render a trajectory, and save to a video file."""

    # Path to config YAML file.
    load_config: Path
    # Name of the renderer outputs to use. rgb, depth, etc. concatenates them along y axis
    rendered_output_names: List[str] = field(default_factory=lambda: ["rgb"])
    #  Trajectory to render.
    traj: Literal["spiral", "filename"] = "spiral"
    # Scaling factor to apply to the camera image resolution.
    downscale_factor: int = 1
    # Filename of the camera path to render.
    camera_path_filename: Path = Path("camera_path.json")
    # Name of the output file.
    output_path: Path = Path("renders/output.mp4")
    # How long the video should be.
    seconds: float = 5.0
    # How to save output data.
    output_format: Literal["images", "video"] = "video"
    # Specifies number of rays per chunk during eval.
    eval_num_rays_per_chunk: Optional[int] = None
    # Start frame index for profiling or partial renders.
    frame_start: int = 0
    # Optional number of frames to render from the selected trajectory.
    num_frames: Optional[int] = None
    # If True, insert CUDA synchronizations around profiled shell stages for strict additive wall-clock timings.
    strict_shell_profile: bool = False

    def main(self) -> None:
        """Main function."""
        _, pipeline, _ = eval_setup(
            self.load_config,
            eval_num_rays_per_chunk=self.eval_num_rays_per_chunk,
            test_mode="test" if self.traj == "spiral" else "inference",
        )

        install_checks.check_ffmpeg_installed()

        seconds = self.seconds

        # TODO(ethan): use camera information from parsing args
        if self.traj == "spiral":
            camera_start = pipeline.datamanager.eval_dataloader.get_camera(image_idx=0).flatten()
            # TODO(ethan): pass in the up direction of the camera
            camera_path = get_spiral_path(camera_start, steps=30, radius=0.1)
        elif self.traj == "filename":
            with open(self.camera_path_filename, "r", encoding="utf-8") as f:
                camera_path = json.load(f)
            seconds = camera_path["seconds"]
            camera_path = get_path_from_json(camera_path)
        else:
            assert_never(self.traj)

        total_frames = int(camera_path.size)
        start_idx = max(0, int(self.frame_start))
        end_idx = total_frames if self.num_frames is None else min(total_frames, start_idx + int(self.num_frames))
        camera_path = camera_path[start_idx:end_idx]
        if total_frames > 0 and end_idx > start_idx:
            seconds = seconds * (float(end_idx - start_idx) / float(total_frames))

        _render_trajectory_video(
            pipeline,
            camera_path,
            output_filename=self.output_path,
            rendered_output_names=self.rendered_output_names,
            rendered_resolution_scaling_factor=1.0 / self.downscale_factor,
            seconds=seconds,
            output_format=self.output_format,
            strict_shell_profile=self.strict_shell_profile,
        )


def entrypoint():
    """Entrypoint for use with pyproject scripts."""
    tyro.extras.set_accent_color("bright_yellow")
    tyro.cli(RenderTrajectory).main()


if __name__ == "__main__":
    entrypoint()

# For sphinx docs
get_parser_fn = lambda: tyro.extras.get_parser(RenderTrajectory)  # noqa
