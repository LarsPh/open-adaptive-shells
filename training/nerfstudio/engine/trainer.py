# Copyright 2022 The Nerfstudio Team. All rights reserved.
# Modified for Adaptive Shells training and rendering; see the repository NOTICE.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Code to train model.
"""
from __future__ import annotations

import dataclasses
import functools
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from PIL import Image
from rich.console import Console
from torch.cuda.amp.grad_scaler import GradScaler
from typing_extensions import Literal

from nerfstudio.configs import base_config as cfg
from nerfstudio.engine.callbacks import (
    TrainingCallback,
    TrainingCallbackAttributes,
    TrainingCallbackLocation,
)
from nerfstudio.engine.optimizers import Optimizers, setup_optimizers
from nerfstudio.pipelines.base_pipeline import VanillaPipeline
from nerfstudio.utils import profiler, writer
from nerfstudio.utils.decorators import (
    check_eval_enabled,
    check_main_thread,
    check_viewer_enabled,
)
from nerfstudio.utils.misc import step_check
from nerfstudio.utils.marching_cubes import extract_mesh_shell
from nerfstudio.utils.writer import EventName, TimeWriter
from nerfstudio.viewer.server import viewer_utils

CONSOLE = Console(width=120)


def _load_checkpoint_file(load_path):
    """Load legacy nerfstudio checkpoints across PyTorch versions."""
    try:
        return torch.load(load_path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(load_path, map_location="cpu")


class Trainer:
    """Trainer class

    Args:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.

    Attributes:
        config: The configuration object.
        local_rank: Local rank of the process.
        world_size: World size of the process.
        device: The device to run the training on.
        pipeline: The pipeline object.
        optimizers: The optimizers object.
        callbacks: The callbacks object.
    """

    pipeline: VanillaPipeline
    optimizers: Optimizers
    callbacks: List[TrainingCallback]

    def __init__(self, config: cfg.Config, local_rank: int = 0, world_size: int = 1):
        self.config = config
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = "cpu" if world_size == 0 else f"cuda:{local_rank}"
        self.mixed_precision = self.config.trainer.mixed_precision
        if self.device == "cpu":
            self.mixed_precision = False
            CONSOLE.print("Mixed precision is disabled for CPU training.")
        self._start_step = 0
        # optimizers
        self.grad_scaler = GradScaler(enabled=self.mixed_precision)

        self.base_dir = config.get_base_dir()
        # directory to save checkpoints
        self.checkpoint_dir = config.get_checkpoint_dir()
        CONSOLE.log(f"Saving checkpoints to: {self.checkpoint_dir}")
        # set up viewer if enabled
        viewer_log_path = self.base_dir / config.viewer.relative_log_filename
        self.viewer_state, banner_messages = None, None
        if self.config.is_viewer_enabled() and local_rank == 0:
            self.viewer_state, banner_messages = viewer_utils.setup_viewer(config.viewer, log_filename=viewer_log_path)
        self._check_viewer_warnings()
        # set up writers/profilers if enabled
        writer_log_path = self.base_dir / config.logging.relative_log_dir
        writer.setup_event_writer(config, log_dir=writer_log_path)
        writer.setup_local_writer(
            config.logging, max_iter=config.trainer.max_num_iterations, banner_messages=banner_messages
        )
        writer.put_config(name="config", config_dict=dataclasses.asdict(config), step=0)
        profiler.setup_profiler(config.logging)
        self._eval_milestone_steps: Set[int] = set()
        self._shell_milestone_steps: Set[int] = set()

    def setup(self, test_mode: Literal["test", "val", "inference"] = "val"):
        """Setup the Trainer by calling other setup functions.

        Args:
            test_mode:
                'val': loads train/val datasets into memory
                'test': loads train/test datset into memory
                'inference': does not load any dataset into memory
        """
        self.pipeline = self.config.pipeline.setup(
            device=self.device, test_mode=test_mode, world_size=self.world_size, local_rank=self.local_rank
        )
        self.optimizers = setup_optimizers(self.config, self.pipeline.get_param_groups())

        self._load_checkpoint()

        self.callbacks = self.pipeline.get_training_callbacks(
            TrainingCallbackAttributes(
                optimizers=self.optimizers,  # type: ignore
                grad_scaler=self.grad_scaler,  # type: ignore
                pipeline=self.pipeline,  # type: ignore
                config=self.config.trainer,  # type: ignore
            )
        )
        self._init_milestone_schedule()

    def _resolve_milestone_steps(self, milestone_values: Tuple[float, ...]) -> Set[int]:
        """Convert percentages/fractions to concrete global step indices."""
        resolved_steps: Set[int] = set()
        num_iterations = max(int(self.config.trainer.max_num_iterations), 1)
        for raw_value in milestone_values:
            value = float(raw_value)
            if value <= 0.0:
                continue
            ratio = value / 100.0 if value > 1.0 else value
            ratio = min(max(ratio, 0.0), 1.0)
            if ratio <= 0.0:
                continue
            offset = max(1, int(round(num_iterations * ratio)))
            offset = min(offset, num_iterations)
            resolved_steps.add(self._start_step + offset - 1)
        return resolved_steps

    def _init_milestone_schedule(self) -> None:
        self._eval_milestone_steps = self._resolve_milestone_steps(self.config.trainer.eval_milestone_percentages)
        self._shell_milestone_steps = self._resolve_milestone_steps(
            self.config.trainer.shell_extraction_milestone_percentages
        )
        if self._eval_milestone_steps:
            preview = ", ".join(str(step) for step in sorted(self._eval_milestone_steps))
            CONSOLE.print(f"[cyan]Eval milestones (steps): {preview}")
        if self._shell_milestone_steps:
            preview = ", ".join(str(step) for step in sorted(self._shell_milestone_steps))
            CONSOLE.print(f"[cyan]Shell extraction milestones (steps): {preview}")

    @staticmethod
    def _save_image_tensor_png(path: Path, image: torch.Tensor) -> None:
        image_cpu = image.detach().float().cpu()
        if image_cpu.ndim != 3:
            return
        if image_cpu.shape[-1] == 1:
            image8 = (image_cpu[..., 0].clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()
            Image.fromarray(image8, mode="L").save(path)
            return
        if image_cpu.shape[-1] < 3:
            return
        image8 = (image_cpu[..., :3].clamp(0.0, 1.0) * 255.0).to(torch.uint8).numpy()
        Image.fromarray(image8, mode="RGB").save(path)

    def _save_eval_images_to_disk(self, step: int, image_idx: int, images_dict: Dict[str, torch.Tensor]) -> None:
        if not self.config.trainer.save_eval_images_to_disk:
            return
        output_dir = (
            self.base_dir
            / self.config.trainer.eval_images_output_subdir
            / f"step-{step:09d}"
            / f"image-{int(image_idx):04d}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        for image_name, image_tensor in images_dict.items():
            safe_name = image_name.replace("/", "_")
            self._save_image_tensor_png(output_dir / f"{safe_name}.png", image_tensor)

    def _resolve_eval_milestone_indices(self) -> Tuple[int, ...]:
        trainer_indices = self.config.trainer.eval_milestone_image_indices
        if trainer_indices is not None and len(trainer_indices) > 0:
            return tuple(int(idx) for idx in trainer_indices)
        datamanager_indices = getattr(self.config.pipeline.datamanager, "eval_image_indices", None)
        if datamanager_indices is not None and len(datamanager_indices) > 0:
            return tuple(int(idx) for idx in datamanager_indices)
        return (0,)

    def _run_milestone_eval(self, step: int) -> None:
        if self.config.trainer.run_eval_batch_on_milestones:
            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(step=step)
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
            writer.put_scalar(name="Milestone Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(name="Milestone Eval Loss Dict", scalar_dict=eval_loss_dict, step=step)
            writer.put_dict(name="Milestone Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step)

        if not self.config.trainer.run_eval_images_on_milestones:
            return
        if not hasattr(self.pipeline, "get_eval_image_metrics_and_images_for_indices"):
            return

        image_indices = self._resolve_eval_milestone_indices()
        render_fn = getattr(self.pipeline, "get_eval_image_metrics_and_images_for_indices")
        camera_results = render_fn(step=step, image_indices=image_indices)
        if len(camera_results) == 0:
            return

        aggregate_metrics: Dict[str, float] = {}
        for image_idx, metrics_dict, images_dict in camera_results:
            writer.put_dict(
                name=f"Milestone Eval Image Metrics/{int(image_idx):04d}",
                scalar_dict=metrics_dict,
                step=step,
            )
            for image_name, image in images_dict.items():
                writer.put_image(
                    name=f"Milestone Eval Images/cam_{int(image_idx):04d}/{image_name}",
                    image=image,
                    step=step,
                )
            self._save_eval_images_to_disk(step=step, image_idx=int(image_idx), images_dict=images_dict)

            for key, value in metrics_dict.items():
                if isinstance(value, (int, float)):
                    aggregate_metrics[key] = aggregate_metrics.get(key, 0.0) + float(value)

        inv_count = 1.0 / float(len(camera_results))
        for key in list(aggregate_metrics.keys()):
            aggregate_metrics[key] *= inv_count
        writer.put_dict(name="Milestone Eval Image Metrics/Average", scalar_dict=aggregate_metrics, step=step)

    def _run_shell_extraction(self, step: int) -> bool:
        if not self.config.trainer.run_shell_extraction_on_milestones:
            return False
        model = self.pipeline.model
        field = getattr(model, "field", None)
        if field is None or not hasattr(field, "forward_geonetwork"):
            return False

        output_dir = self.base_dir / self.config.trainer.shell_extraction_output_subdir / f"step-{step:09d}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "shell.ply"

        def forward_geonetwork_with_distortion(points: torch.Tensor) -> torch.Tensor:
            if getattr(field, "spatial_distortion", None) is not None:
                points = field.spatial_distortion(points)
            return field.forward_geonetwork(points)

        shell_mode = self.config.trainer.shell_extraction_kernel_size_transform_mode
        if shell_mode is None:
            shell_mode = getattr(field.config, "kernel_size_transform_mode", "exp")
        shell_exp_factor = self.config.trainer.shell_extraction_kernel_size_transform_exp_factor
        if shell_exp_factor is None:
            shell_exp_factor = float(getattr(field.config, "kernel_size_transform_exp_factor", 10.0))

        try:
            extract_mesh_shell(
                field_fn=lambda x: forward_geonetwork_with_distortion(x)[:, :2].contiguous(),
                resolution=self.config.trainer.shell_extraction_resolution,
                level=self.config.trainer.shell_extraction_level,
                bounding_box_min=self.config.trainer.shell_extraction_bounding_box_min,
                bounding_box_max=self.config.trainer.shell_extraction_bounding_box_max,
                output_path=output_path,
                simplify_mesh=self.config.trainer.shell_extraction_simplify_mesh,
                n_iter=self.config.trainer.shell_extraction_n_iter,
                time_step=self.config.trainer.shell_extraction_time_step,
                time_step_d=getattr(self.config.trainer, "shell_extraction_time_step_d", None),
                time_step_e=getattr(self.config.trainer, "shell_extraction_time_step_e", None),
                sigma_min=self.config.trainer.shell_extraction_sigma_min,
                beta_d=self.config.trainer.shell_extraction_beta_d,
                beta_e=self.config.trainer.shell_extraction_beta_e,
                v_max=self.config.trainer.shell_extraction_v_max,
                erosion_velocity_mode=getattr(
                    self.config.trainer, "shell_extraction_erosion_velocity_mode", "inverse_alpha"
                ),
                erosion_excess_floor=getattr(self.config.trainer, "shell_extraction_erosion_excess_floor", 0.0),
                smooth_velocity_d_iters=getattr(self.config.trainer, "shell_extraction_smooth_velocity_d_iters", 0),
                smooth_velocity_e_iters=getattr(self.config.trainer, "shell_extraction_smooth_velocity_e_iters", 0),
                curvature_motion_ratio_d=getattr(
                    self.config.trainer, "shell_extraction_curvature_motion_ratio_d", None
                ),
                curvature_motion_ratio_e=getattr(
                    self.config.trainer, "shell_extraction_curvature_motion_ratio_e", None
                ),
                weight_curv_d=self.config.trainer.shell_extraction_weight_curv_d,
                window_width_d=self.config.trainer.shell_extraction_window_width_d,
                weight_curv_e=self.config.trainer.shell_extraction_weight_curv_e,
                window_width_e=self.config.trainer.shell_extraction_window_width_e,
                kernel_size_transform_mode=shell_mode,
                kernel_size_transform_exp_factor=float(shell_exp_factor),
                recompute_grad_each_iter=getattr(
                    self.config.trainer, "shell_extraction_recompute_grad_each_iter", False
                ),
                debug_stats=self.config.trainer.shell_extraction_debug_stats,
                device=str(model.device),
                slab_depth=self.config.trainer.shell_extraction_eval_slab_depth,
                batch_points=self.config.trainer.shell_extraction_eval_batch_points,
            )
            writer.put_scalar(name="Shell Extraction/Triggered", scalar=1.0, step=step)
            return True
        except Exception as exc:  # pragma: no cover - defensive logging for long-running jobs
            CONSOLE.print(f"[bold red]Shell extraction failed at step {step}: {exc}")
            writer.put_scalar(name="Shell Extraction/Triggered", scalar=0.0, step=step)
            return False

    def _run_milestone_tasks(self, step: int) -> bool:
        eval_hit = step in self._eval_milestone_steps
        shell_hit = step in self._shell_milestone_steps
        if not eval_hit and not shell_hit:
            return False

        if eval_hit:
            self._run_milestone_eval(step)

        shell_extraction_ran = False
        if shell_hit:
            shell_extraction_ran = self._run_shell_extraction(step)

        should_save = False
        if eval_hit and self.config.trainer.save_checkpoint_on_eval_milestones:
            should_save = True
        if shell_hit and shell_extraction_ran and self.config.trainer.save_checkpoint_on_shell_milestones:
            should_save = True
        if should_save:
            self.save_checkpoint(step)
        return should_save

    def train(self) -> None:
        """Train the model."""
        assert self.pipeline.datamanager.train_dataset is not None, "Missing DatsetInputs"

        self._init_viewer_state()
        with TimeWriter(writer, EventName.TOTAL_TRAIN_TIME):
            num_iterations = self.config.trainer.max_num_iterations
            step = 0
            for step in range(self._start_step, self._start_step + num_iterations):
                with TimeWriter(writer, EventName.ITER_TRAIN_TIME, step=step) as train_t:

                    self.pipeline.train()

                    # training callbacks before the training iteration
                    for callback in self.callbacks:
                        callback.run_callback_at_location(
                            step, location=TrainingCallbackLocation.BEFORE_TRAIN_ITERATION
                        )

                    # time the forward pass
                    loss, loss_dict, metrics_dict = self.train_iteration(step)

                    # training callbacks after the training iteration
                    for callback in self.callbacks:
                        callback.run_callback_at_location(step, location=TrainingCallbackLocation.AFTER_TRAIN_ITERATION)

                # Skip the first two steps to avoid skewed timings that break the viewer rendering speed estimate.
                if step > 1:
                    writer.put_time(
                        name=EventName.TRAIN_RAYS_PER_SEC,
                        duration=self.config.pipeline.datamanager.train_num_rays_per_batch / train_t.duration,
                        step=step,
                        avg_over_steps=True,
                    )

                self._update_viewer_state(step)

                # a batch of train rays
                if step_check(step, self.config.logging.steps_per_log, run_at_zero=True):
                    writer.put_scalar(name="Train Loss", scalar=loss, step=step)
                    writer.put_dict(name="Train Loss Dict", scalar_dict=loss_dict, step=step)
                    writer.put_dict(name="Train Metrics Dict", scalar_dict=metrics_dict, step=step)

                self.eval_iteration(step)
                saved_in_milestone = self._run_milestone_tasks(step)

                if step_check(step, self.config.trainer.steps_per_save):
                    if not saved_in_milestone:
                        self.save_checkpoint(step)

                writer.write_out_storage()
            # save checkpoint at the end of training
            self.save_checkpoint(step)

            CONSOLE.rule()
            CONSOLE.print("[bold green]:tada: :tada: :tada: Training Finished :tada: :tada: :tada:", justify="center")
            if not self.config.viewer.quit_on_train_completion:
                CONSOLE.print("Use ctrl+c to quit", justify="center")
                self._always_render(step)

    @check_main_thread
    def _always_render(self, step):
        if self.config.is_viewer_enabled():
            while True:
                self.viewer_state.vis["renderingState/isTraining"].write(False)
                self._update_viewer_state(step)

    @check_main_thread
    def _check_viewer_warnings(self) -> None:
        """Helper to print out any warnings regarding the way the viewer/loggers are enabled"""
        if self.config.is_viewer_enabled():
            string = (
                "[NOTE] Not running eval iterations since only viewer is enabled."
                " Use [yellow]--vis wandb[/yellow] or [yellow]--vis tensorboard[/yellow] to run with eval instead."
            )
            CONSOLE.print(f"{string}")

    @check_viewer_enabled
    def _init_viewer_state(self) -> None:
        """Initializes viewer scene with given train dataset"""
        assert self.viewer_state and self.pipeline.datamanager.train_dataset
        self.viewer_state.init_scene(
            dataset=self.pipeline.datamanager.train_dataset,
            start_train=self.config.viewer.start_train,
        )
        if not self.config.viewer.start_train:
            self._always_render(self._start_step)

    @check_viewer_enabled
    def _update_viewer_state(self, step: int):
        """Updates the viewer state by rendering out scene with current pipeline
        Returns the time taken to render scene.

        Args:
            step: current train step
        """
        assert self.viewer_state is not None
        with TimeWriter(writer, EventName.ITER_VIS_TIME, step=step) as _:
            num_rays_per_batch = self.config.pipeline.datamanager.train_num_rays_per_batch
            try:
                self.viewer_state.update_scene(self, step, self.pipeline.model, num_rays_per_batch)
            except RuntimeError:
                time.sleep(0.03)  # sleep to allow buffer to reset
                assert self.viewer_state.vis is not None
                self.viewer_state.vis["renderingState/log_errors"].write(
                    "Error: GPU out of memory. Reduce resolution to prevent viewer from crashing."
                )

    @check_viewer_enabled
    def _update_viewer_rays_per_sec(self, train_t: TimeWriter, vis_t: TimeWriter, step: int):
        """Performs update on rays/sec calclation for training

        Args:
            train_t: timer object carrying time to execute total training iteration
            vis_t: timer object carrying time to execute visualization step
            step: current step
        """
        train_num_rays_per_batch = self.config.pipeline.datamanager.train_num_rays_per_batch
        writer.put_time(
            name=EventName.TRAIN_RAYS_PER_SEC,
            duration=train_num_rays_per_batch / (train_t.duration - vis_t.duration),
            step=step,
            avg_over_steps=True,
        )

    def _load_checkpoint(self) -> None:
        """Helper function to load pipeline and optimizer from prespecified checkpoint"""
        load_dir = self.config.trainer.load_dir
        if load_dir is not None:
            load_step = self.config.trainer.load_step
            if load_step is None:
                print("Loading latest checkpoint from load_dir")
                # NOTE: this is specific to the checkpoint name format
                load_step = sorted(int(x[x.find("-") + 1 : x.find(".")]) for x in os.listdir(load_dir))[-1]
            load_path = load_dir / f"step-{load_step:09d}.ckpt"
            assert load_path.exists(), f"Checkpoint {load_path} does not exist"
            loaded_state = _load_checkpoint_file(load_path)
            self._start_step = loaded_state["step"] + 1
            # load the checkpoints for pipeline, optimizers, and gradient scalar
            self.pipeline.load_pipeline(loaded_state["pipeline"])
            self.optimizers.load_optimizers(loaded_state["optimizers"])
            if "schedulers" in loaded_state and self.config.trainer.load_scheduler:
                self.optimizers.load_schedulers(loaded_state["schedulers"])
            self.grad_scaler.load_state_dict(loaded_state["scalers"])
            CONSOLE.print(f"done loading checkpoint from {load_path}")
        else:
            CONSOLE.print("No checkpoints to load, training from scratch")

    @check_main_thread
    def save_checkpoint(self, step: int) -> None:
        """Save the model and optimizers

        Args:
            step: number of steps in training for given checkpoint
        """
        # possibly make the checkpoint directory
        if not self.checkpoint_dir.exists():
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # save the checkpoint
        ckpt_path = self.checkpoint_dir / f"step-{step:09d}.ckpt"
        torch.save(
            {
                "step": step,
                "pipeline": self.pipeline.module.state_dict()  # type: ignore
                if hasattr(self.pipeline, "module")
                else self.pipeline.state_dict(),
                "optimizers": {k: v.state_dict() for (k, v) in self.optimizers.optimizers.items()},
                "schedulers": {k: v.state_dict() for (k, v) in self.optimizers.schedulers.items()},
                "scalers": self.grad_scaler.state_dict(),
            },
            ckpt_path,
        )
        # possibly delete old checkpoints
        if self.config.trainer.save_only_latest_checkpoint:
            # delete everything else in the checkpoint folder
            for f in self.checkpoint_dir.glob("*"):
                if f != ckpt_path:
                    f.unlink()

    @profiler.time_function
    def train_iteration(self, step: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """Run one iteration with a batch of inputs. Returns dictionary of model losses.

        Args:
            step: Current training step.
        """
        self.optimizers.zero_grad_all()
        cpu_or_cuda_str = self.device.split(":")[0]
        for _ in range(self.config.trainer.accumulate_grad_steps):
            with torch.autocast(device_type=cpu_or_cuda_str, enabled=self.mixed_precision):
                _, loss_dict, metrics_dict = self.pipeline.get_train_loss_dict(step=step)
                loss = functools.reduce(torch.add, loss_dict.values())
            self.grad_scaler.scale(loss).backward()  # type: ignore
        lr_before_step = {
            name: float(optimizer.param_groups[0].get("lr", 0.0))
            for name, optimizer in self.optimizers.optimizers.items()
            if optimizer.param_groups
        }
        self.optimizers.optimizer_scaler_step_all(self.grad_scaler)
        self.grad_scaler.update()
        self.optimizers.scheduler_step_all(step)
        optimizer_state_log_steps = int(getattr(self.config.trainer, "optimizer_state_log_steps", 100))
        if optimizer_state_log_steps > 0 and step_check(step, optimizer_state_log_steps, run_at_zero=True):
            metrics_dict.update(self.optimizers.get_state_metrics(lr_before_step=lr_before_step))

        # only return the last accumulate_grad_step's loss and metric for logging
        # Merging loss and metrics dict into a single output.
        return loss, loss_dict, metrics_dict

    @check_eval_enabled
    @profiler.time_function
    def eval_iteration(self, step):
        """Run one iteration with different batch/image/all image evaluations depending on step size.

        Args:
            step: Current training step.
        """
        # a batch of eval rays
        if step_check(step, self.config.trainer.steps_per_eval_batch, run_at_zero=self.config.trainer.sanity_check):
            _, eval_loss_dict, eval_metrics_dict = self.pipeline.get_eval_loss_dict(step=step)
            eval_loss = functools.reduce(torch.add, eval_loss_dict.values())
            writer.put_scalar(name="Eval Loss", scalar=eval_loss, step=step)
            writer.put_dict(name="Eval Loss Dict", scalar_dict=eval_loss_dict, step=step)
            writer.put_dict(name="Eval Metrics Dict", scalar_dict=eval_metrics_dict, step=step)

        # one eval image
        if step_check(step, self.config.trainer.steps_per_eval_image, run_at_zero=self.config.trainer.sanity_check):
            with TimeWriter(writer, EventName.TEST_RAYS_PER_SEC, write=False) as test_t:
                metrics_dict, images_dict = self.pipeline.get_eval_image_metrics_and_images(step=step)
            writer.put_time(
                name=EventName.TEST_RAYS_PER_SEC,
                duration=metrics_dict["num_rays"] / test_t.duration,
                step=step,
                avg_over_steps=True,
            )
            writer.put_dict(name="Eval Images Metrics", scalar_dict=metrics_dict, step=step)
            group = "Eval Images"
            for image_name, image in images_dict.items():
                writer.put_image(name=group + "/" + image_name, image=image, step=step)

        # all eval images
        if step_check(step, self.config.trainer.steps_per_eval_all_images):
            metrics_dict, _ = self.pipeline.get_average_eval_image_metrics(step=step)
            writer.put_dict(name="Eval Images Metrics Dict (all images)", scalar_dict=metrics_dict, step=step)
