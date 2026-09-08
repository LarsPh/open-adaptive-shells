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

"""Base Configs"""

# pylint: disable=wrong-import-position

from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import yaml
from rich.console import Console
from typing_extensions import Literal

from nerfstudio.configs.config_utils import to_immutable_dict

# model instances
from nerfstudio.utils import writer

warnings.filterwarnings("ignore", module="torchvision")

CONSOLE = Console(width=120)

# Pretty printing class
class PrintableConfig:  # pylint: disable=too-few-public-methods
    """Printable Config defining str function"""

    def __str__(self):
        lines = [self.__class__.__name__ + ":"]
        for key, val in vars(self).items():
            if isinstance(val, Tuple):
                flattened_val = "["
                for item in val:
                    flattened_val += str(item) + "\n"
                flattened_val = flattened_val.rstrip("\n")
                val = flattened_val + "]"
            lines += f"{key}: {str(val)}".split("\n")
        return "\n    ".join(lines)


# Base instantiate configs
@dataclass
class InstantiateConfig(PrintableConfig):  # pylint: disable=too-few-public-methods
    """Config class for instantiating an the class specified in the _target attribute."""

    _target: Type

    def setup(self, **kwargs) -> Any:
        """Returns the instantiated object using the config."""
        return self._target(self, **kwargs)


# Machine related configs
@dataclass
class MachineConfig(PrintableConfig):
    """Configuration of machine setup"""

    seed: int = 42
    """random seed initilization"""
    num_gpus: int = 1
    """total number of gpus available for train/eval"""
    num_machines: int = 1
    """total number of distributed machines available (for DDP)"""
    machine_rank: int = 0
    """current machine's rank (for DDP)"""
    dist_url: str = "auto"
    """distributed connection point (for DDP)"""


@dataclass
class LocalWriterConfig(InstantiateConfig):
    """Local Writer config"""

    _target: Type = writer.LocalWriter
    """target class to instantiate"""
    enable: bool = False
    """if True enables local logging, else disables"""
    stats_to_track: Tuple[writer.EventName, ...] = (
        writer.EventName.ITER_TRAIN_TIME,
        writer.EventName.TRAIN_RAYS_PER_SEC,
        writer.EventName.CURR_TEST_PSNR,
        writer.EventName.VIS_RAYS_PER_SEC,
        writer.EventName.TEST_RAYS_PER_SEC,
    )
    """specifies which stats will be logged/printed to terminal"""
    max_log_size: int = 10
    """maximum number of rows to print before wrapping. if 0, will print everything."""

    def setup(self, banner_messages: Optional[List[str]] = None, **kwargs) -> Any:
        """Instantiate local writer

        Args:
            banner_messages: List of strings that always print at the bottom of screen.
        """
        return self._target(self, banner_messages=banner_messages, **kwargs)


@dataclass
class LoggingConfig(PrintableConfig):
    """Configuration of loggers and profilers"""

    relative_log_dir: Path = Path("./")
    """relative path to save all logged events"""
    steps_per_log: int = 10
    """number of steps between logging stats"""
    max_buffer_size: int = 20
    """maximum history size to keep for computing running averages of stats.
     e.g. if 20, averages will be computed over past 20 occurances."""
    local_writer: LocalWriterConfig = LocalWriterConfig(enable=True)
    """if provided, will print stats locally. if None, will disable printing"""
    enable_profiler: bool = True
    """whether to enable profiling code; prints speed of functions at the end of a program.
    profiler logs run times of functions and prints at end of training"""


# Trainer related configs
@dataclass
class TrainerConfig(PrintableConfig):
    """Configuration for training regimen"""

    sanity_check: bool = False
    """Whether run sanity check before training start."""
    steps_per_save: int = 1000
    """Number of steps between saves."""
    steps_per_eval_batch: int = 500
    """Number of steps between randomly sampled batches of rays."""
    steps_per_eval_image: int = 500
    """Number of steps between single eval images."""
    steps_per_eval_all_images: int = 25000
    """Number of steps between eval all images."""
    max_num_iterations: int = 1000000
    """Maximum number of iterations to run."""
    mixed_precision: bool = False
    """Whether or not to use mixed precision for training."""
    relative_model_dir: Path = Path("sdfstudio_models/")
    """Relative path to save all checkpoints."""
    save_only_latest_checkpoint: bool = True
    """Whether to only save the latest checkpoint or all checkpoints."""
    # optional parameters if we want to resume training
    load_dir: Optional[Path] = None
    """Optionally specify a pre-trained model directory to load from."""
    load_step: Optional[int] = None
    """Optionally specify model step to load from; if none, will find most recent model in load_dir."""
    load_config: Optional[Path] = None
    """Optionally specify model config to load from; if none, will use the default config?"""
    load_scheduler: bool = True
    """Whether to load the lr scheduler state_dict if exists"""
    accumulate_grad_steps: int = 1
    """Number of gradient steps to accumulate before taking an optimizer step."""
    optimizer_state_log_steps: int = 100
    """Number of steps between aggregated optimizer-state diagnostics; set <=0 to disable."""
    eval_milestone_percentages: Tuple[float, ...] = ()
    """Optional milestone percentages/fractions for extra eval runs (e.g. 3,5,10,20 or 0.03,0.05...)."""
    shell_extraction_milestone_percentages: Tuple[float, ...] = ()
    """Optional milestone percentages/fractions for periodic shell extraction."""
    run_eval_batch_on_milestones: bool = True
    """Whether milestone eval also runs eval batch loss/metrics logging."""
    run_eval_images_on_milestones: bool = True
    """Whether milestone eval also renders configured fixed eval images."""
    eval_milestone_image_indices: Optional[Tuple[int, ...]] = None
    """Fixed eval image indices for milestone rendering; if None, falls back to datamanager eval_image_indices."""
    save_checkpoint_on_eval_milestones: bool = True
    """Whether to save a checkpoint whenever an eval milestone is hit."""
    save_checkpoint_on_shell_milestones: bool = True
    """Whether to save a checkpoint whenever a shell extraction milestone is hit."""
    save_eval_images_to_disk: bool = True
    """Whether to save milestone eval images as PNG files on disk in addition to event writers."""
    eval_images_output_subdir: str = "milestone_eval_images"
    """Sub-directory under the run directory for saved milestone eval images."""
    run_shell_extraction_on_milestones: bool = True
    """Whether to run shell extraction at shell extraction milestones."""
    shell_extraction_output_subdir: str = "shell_extraction"
    """Sub-directory under the run directory for periodic shell extraction outputs."""
    shell_extraction_resolution: int = 512
    """Marching-cubes grid resolution for shell extraction (current implementation expects 512)."""
    shell_extraction_level: float = 0.0
    """Isosurface level used for extracting the base mesh before shell evolution."""
    shell_extraction_simplify_mesh: bool = False
    """Whether to run post simplification on extracted shell meshes."""
    shell_extraction_debug_stats: bool = False
    """Whether to print detailed shell extraction statistics."""
    shell_extraction_bounding_box_min: Tuple[float, float, float] = (-1.0, -1.0, -1.0)
    """Minimum corner of the extraction bounding box."""
    shell_extraction_bounding_box_max: Tuple[float, float, float] = (1.0, 1.0, 1.0)
    """Maximum corner of the extraction bounding box."""
    shell_extraction_n_iter: int = 50
    """Number of shell evolution iterations during extraction."""
    shell_extraction_time_step: float = 0.01
    """Time step used by shell field evolution."""
    shell_extraction_time_step_d: Optional[float] = None
    """Optional outer dilation time step; if None, uses shell_extraction_time_step."""
    shell_extraction_time_step_e: Optional[float] = None
    """Optional inner erosion time step; if None, uses shell_extraction_time_step."""
    shell_extraction_sigma_min: float = 0.01
    """Minimum alpha threshold used by shell velocity computation."""
    shell_extraction_beta_d: float = 1.0
    """Outer dilation velocity coefficient."""
    shell_extraction_beta_e: float = 0.01
    """Inner erosion velocity coefficient."""
    shell_extraction_v_max: float = 0.025
    """Maximum erosion velocity clamp."""
    shell_extraction_erosion_velocity_mode: Literal[
        "inverse_alpha", "excess_inverse_alpha", "baseline_excess_inverse_alpha"
    ] = "baseline_excess_inverse_alpha"
    """Erosion velocity mode; excess_inverse_alpha protects alpha≈1 sharp surfaces."""
    shell_extraction_erosion_excess_floor: float = 0.2
    """Dimensionless baseline term for baseline_excess_inverse_alpha erosion velocity."""
    shell_extraction_smooth_velocity_d_iters: int = 1
    """Number of 3D smoothing passes applied to dilation velocity before evolution."""
    shell_extraction_smooth_velocity_e_iters: int = 0
    """Number of 3D smoothing passes applied to erosion velocity before evolution."""
    shell_extraction_curvature_motion_ratio_d: Optional[float] = None
    """Clamp dilation curvature term to this ratio of the motion term; None disables."""
    shell_extraction_curvature_motion_ratio_e: Optional[float] = None
    """Clamp erosion curvature term to this ratio of the motion term; None disables."""
    shell_extraction_weight_curv_d: float = 0.0001
    """Outer curvature regularization weight."""
    shell_extraction_window_width_d: float = 0.1
    """Outer evolution window width."""
    shell_extraction_weight_curv_e: float = 0.0
    """Inner curvature regularization weight."""
    shell_extraction_window_width_e: float = 0.04
    """Inner evolution window width."""
    shell_extraction_recompute_grad_each_iter: bool = False
    """Whether shell extraction recomputes level-set gradients at every evolution step."""
    shell_extraction_eval_batch_points: int = 200000
    """Maximum number of regular-grid points evaluated per field forward chunk during shell extraction."""
    shell_extraction_eval_slab_depth: int = 8
    """Number of z slices evaluated together during shell extraction grid evaluation."""
    shell_extraction_kernel_size_transform_mode: Optional[Literal["exp", "clip", "none"]] = None
    """Optional override for kernel-size transform mode used by shell extraction."""
    shell_extraction_kernel_size_transform_exp_factor: Optional[float] = None
    """Optional override for kernel-size exp factor used by shell extraction."""


# Viewer related configs
@dataclass
class ViewerConfig(PrintableConfig):
    """Configuration for viewer instantiation"""

    relative_log_filename: str = "viewer_log_filename.txt"
    """Filename to use for the log file."""
    start_train: bool = True
    """whether to immediately start training upon loading viewer
    if False, will just visualize dataset but you can toggle training in viewer"""
    zmq_port: Optional[int] = None
    """The zmq port to connect to for communication. If None, find an available port."""
    launch_bridge_server: bool = True
    """whether or not to launch the bridge server"""
    websocket_port: Optional[int] = 7007
    """the default websocket port to connect to"""
    ip_address: str = "127.0.0.1"
    """the ip address where the bridge server is running"""
    num_rays_per_chunk: int = 32768
    """number of rays per chunk to render with viewer"""
    max_num_display_images: int = 512
    """Maximum number of training images to display in the viewer, to avoid lag. This does not change which images are
    actually used in training/evaluation. If -1, display all."""
    quit_on_train_completion: bool = False
    """Whether to kill the training job when it has completed. Note this will stop rendering in the viewer."""


from nerfstudio.engine.optimizers import OptimizerConfig
from nerfstudio.engine.schedulers import SchedulerConfig
from nerfstudio.pipelines.base_pipeline import VanillaPipelineConfig


@dataclass
class Config(PrintableConfig):
    """Full config contents"""

    output_dir: Path = Path("outputs")
    """relative or absolute output directory to save all checkpoints and logging"""
    method_name: Optional[str] = None
    """Method name. Required to set in python or via cli"""
    experiment_name: Optional[str] = None
    """Experiment name. If None, will automatically be set to dataset name"""
    timestamp: str = "{timestamp}"
    """Experiment timestamp."""
    machine: MachineConfig = MachineConfig()
    """Machine configuration"""
    logging: LoggingConfig = LoggingConfig()
    """Logging configuration"""
    viewer: ViewerConfig = ViewerConfig()
    """Viewer configuration"""
    trainer: TrainerConfig = TrainerConfig()
    """Trainer configuration"""
    pipeline: VanillaPipelineConfig = VanillaPipelineConfig()
    """Pipeline configuration"""
    optimizers: Dict[str, Any] = to_immutable_dict(
        {
            "fields": {
                "optimizer": OptimizerConfig(),
                "scheduler": SchedulerConfig(),
            }
        }
    )
    """Dictionary of optimizer groups and their schedulers"""
    vis: Literal["viewer", "wandb", "tensorboard"] = "wandb"
    """Which visualizer to use."""
    data: Optional[Path] = None
    """Alias for --pipeline.datamanager.dataparser.data"""

    def is_viewer_enabled(self) -> bool:
        """Checks if a viewer is enabled."""
        return "viewer" == self.vis

    def is_wandb_enabled(self) -> bool:
        """Checks if wandb is enabled."""
        return "wandb" == self.vis

    def is_tensorboard_enabled(self) -> bool:
        """Checks if tensorboard is enabled."""
        return "tensorboard" == self.vis

    def set_timestamp(self) -> None:
        """Dynamically set the experiment timestamp"""
        if self.timestamp == "{timestamp}":
            self.timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")

    def set_experiment_name(self) -> None:
        """Dynamically set the experiment name"""
        if self.experiment_name is None:
            data_path = self.pipeline.datamanager.dataparser.data
            if data_path is None:
                self.experiment_name = "unnamed"
            else:
                dataset_name = Path(data_path).name
                self.experiment_name = dataset_name or str(data_path).replace("\\", "-").replace("/", "-")

    def get_base_dir(self) -> Path:
        """Retrieve the base directory to set relative paths"""
        # check the experiment and method names
        assert self.method_name is not None, "Please set method name in config or via the cli"
        self.set_experiment_name()
        return Path(f"{self.output_dir}/{self.experiment_name}/{self.method_name}/{self.timestamp}")

    def get_checkpoint_dir(self) -> Path:
        """Retrieve the checkpoint directory"""
        return Path(self.get_base_dir() / self.trainer.relative_model_dir)

    def print_to_terminal(self) -> None:
        """Helper to pretty print config to terminal"""
        CONSOLE.rule("Config")
        CONSOLE.print(self)
        CONSOLE.rule("")

    def save_config(self) -> None:
        """Save config to base directory"""
        base_dir = self.get_base_dir()
        assert base_dir is not None
        base_dir.mkdir(parents=True, exist_ok=True)
        config_yaml_path = base_dir / "config.yml"
        CONSOLE.log(f"Saving config to: {config_yaml_path}")
        config_yaml_path.write_text(yaml.dump(self), "utf8")
