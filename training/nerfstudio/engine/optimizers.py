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
Optimizers class.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import torch
from torch.cuda.amp.grad_scaler import GradScaler
from torch.nn.parameter import Parameter

from nerfstudio.configs import base_config
from nerfstudio.utils import writer


# Optimizer related configs
@dataclass
class OptimizerConfig(base_config.PrintableConfig):
    """Basic optimizer config with RAdam"""

    _target: Type = torch.optim.Adam
    lr: float = 0.0005
    eps: float = 1e-08

    # TODO: somehow make this more generic. i dont like the idea of overriding the setup function
    # but also not sure how to go about passing things into predefined torch objects.
    def setup(self, params) -> Any:
        """Returns the instantiated object using the config."""
        kwargs = vars(self).copy()
        kwargs.pop("_target")
        return self._target(params, **kwargs)


@dataclass
class AdamOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with Adam"""

    _target: Type = torch.optim.Adam
    weight_decay: float = 0

@dataclass
class AdamWOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with AdamW"""

    _target: Type = torch.optim.AdamW
    weight_decay: float = 0

@dataclass
class RAdamOptimizerConfig(OptimizerConfig):
    """Basic optimizer config with RAdam"""

    _target: Type = torch.optim.RAdam


def setup_optimizers(config: base_config.Config, param_groups: Dict[str, List[Parameter]]) -> "Optimizers":
    """Helper to set up the optimizers

    Args:
        config: The trainer configuration object.
        param_groups: A dictionary of parameter groups to optimize.

    Returns:
        The optimizers object.
    """
    optimizer_config = config.optimizers.copy()

    # Add the camera optimizer if enabled.
    camera_optimizer_config = config.pipeline.datamanager.camera_optimizer
    if camera_optimizer_config.mode != "off":
        assert camera_optimizer_config.param_group not in optimizer_config
        optimizer_config[camera_optimizer_config.param_group] = {
            "optimizer": config.pipeline.datamanager.camera_optimizer.optimizer,
            "scheduler": config.pipeline.datamanager.camera_optimizer.scheduler,
        }
    return Optimizers(optimizer_config, param_groups)


class Optimizers:
    """A set of optimizers.

    Args:
        config: The optimizer configuration object.
        param_groups: A dictionary of parameter groups to optimize.
    """

    def __init__(self, config: Dict[str, Any], param_groups: Dict[str, List[Parameter]]):
        self.config = config
        self.optimizers = {}
        self.schedulers = {}
        for param_group_name, params in param_groups.items():
            lr_init = config[param_group_name]["optimizer"].lr
            self.optimizers[param_group_name] = config[param_group_name]["optimizer"].setup(params=params)
            if config[param_group_name]["scheduler"]:
                self.schedulers[param_group_name] = config[param_group_name]["scheduler"].setup(
                    optimizer=self.optimizers[param_group_name], lr_init=lr_init
                )

    def optimizer_step(self, param_group_name: str) -> None:
        """Fetch and step corresponding optimizer.

        Args:
            param_group_name: name of optimizer to step forward
        """
        self.optimizers[param_group_name].step()

    def scheduler_step(self, param_group_name: str) -> None:
        """Fetch and step corresponding scheduler.

        Args:
            param_group_name: name of scheduler to step forward
        """
        if self.config.param_group_name.scheduler:  # type: ignore
            self.schedulers[param_group_name].step()

    def zero_grad_all(self) -> None:
        """Zero the gradients for all optimizer parameters."""
        for _, optimizer in self.optimizers.items():
            optimizer.zero_grad()

    def optimizer_scaler_step_all(self, grad_scaler: GradScaler) -> None:
        """Take an optimizer step using a grad scaler.

        Args:
            grad_scaler: GradScaler to use
        """
        for _, optimizer in self.optimizers.items():
            grad_scaler.step(optimizer)

    def optimizer_step_all(self):
        """Run step for all optimizers."""
        for _, optimizer in self.optimizers.items():
            # note that they key is the parameter name
            optimizer.step()

    def scheduler_step_all(self, step: int) -> None:
        """Run step for all schedulers.

        Args:
            step: the current step
        """
        for param_group_name, scheduler in self.schedulers.items():
            scheduler.step()
            # TODO(ethan): clean this up. why is there indexing into a list?
            lr = scheduler.get_last_lr()[0]
            writer.put_scalar(name=f"learning_rate/{param_group_name}", scalar=lr, step=step)

    @torch.no_grad()
    def get_state_metrics(self, lr_before_step: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Summarize optimizer state for diagnosing warmup/zero-LR phases.

        These metrics intentionally aggregate over each optimizer group so that long
        runs can log Adam burn-in behavior without writing per-parameter histograms.
        """
        metrics: Dict[str, float] = {}
        for param_group_name, optimizer in self.optimizers.items():
            prefix = f"optimizer/{param_group_name}"
            params = [param for group in optimizer.param_groups for param in group["params"] if param is not None]
            current_lr = float(optimizer.param_groups[0].get("lr", 0.0)) if optimizer.param_groups else 0.0
            metrics[f"{prefix}/lr_after_scheduler"] = current_lr
            if lr_before_step is not None and param_group_name in lr_before_step:
                metrics[f"{prefix}/lr_used_for_step"] = float(lr_before_step[param_group_name])

            param_sq_sum = 0.0
            grad_sq_sum = 0.0
            exp_avg_sq_sum = 0.0
            exp_avg_sq_sq_sum = 0.0
            adam_update_sq_sum = 0.0
            param_count = 0
            grad_count = 0
            state_param_count = 0
            step_sum = 0.0
            step_max = 0.0
            eps = float(optimizer.param_groups[0].get("eps", 1e-8)) if optimizer.param_groups else 1e-8
            beta1, beta2 = optimizer.param_groups[0].get("betas", (0.9, 0.999)) if optimizer.param_groups else (0.9, 0.999)

            for param in params:
                param_data = param.detach().float()
                param_sq_sum += float(torch.sum(param_data * param_data).item())
                param_count += int(param_data.numel())

                if param.grad is not None:
                    grad_data = param.grad.detach().float()
                    grad_sq_sum += float(torch.sum(grad_data * grad_data).item())
                    grad_count += int(grad_data.numel())

                state = optimizer.state.get(param, {})
                if "step" not in state or "exp_avg" not in state or "exp_avg_sq" not in state:
                    continue

                step_value_raw = state["step"]
                step_value = float(step_value_raw.item() if torch.is_tensor(step_value_raw) else step_value_raw)
                step_sum += step_value
                step_max = max(step_max, step_value)
                state_param_count += 1

                exp_avg = state["exp_avg"].detach().float()
                exp_avg_sq = state["exp_avg_sq"].detach().float()
                exp_avg_sq_sum += float(torch.sum(exp_avg * exp_avg).item())
                exp_avg_sq_sq_sum += float(torch.sum(exp_avg_sq).item())

                bias_correction1 = 1.0 - float(beta1) ** max(step_value, 1.0)
                bias_correction2 = 1.0 - float(beta2) ** max(step_value, 1.0)
                exp_avg_hat = exp_avg / max(bias_correction1, 1e-16)
                exp_avg_sq_hat = exp_avg_sq / max(bias_correction2, 1e-16)
                adam_update = exp_avg_hat / (torch.sqrt(exp_avg_sq_hat) + eps)
                adam_update_sq_sum += float(torch.sum(adam_update * adam_update).item())

            metrics[f"{prefix}/num_params"] = float(param_count)
            if param_count > 0:
                metrics[f"{prefix}/param_norm"] = float(param_sq_sum ** 0.5)
            if grad_count > 0:
                metrics[f"{prefix}/grad_norm"] = float(grad_sq_sum ** 0.5)
                metrics[f"{prefix}/grad_rms"] = float((grad_sq_sum / grad_count) ** 0.5)
            if state_param_count > 0:
                metrics[f"{prefix}/adam_step_mean"] = float(step_sum / state_param_count)
                metrics[f"{prefix}/adam_step_max"] = float(step_max)
            if param_count > 0 and exp_avg_sq_sum > 0.0:
                metrics[f"{prefix}/adam_exp_avg_norm"] = float(exp_avg_sq_sum ** 0.5)
                metrics[f"{prefix}/adam_exp_avg_rms"] = float((exp_avg_sq_sum / param_count) ** 0.5)
            if param_count > 0 and exp_avg_sq_sq_sum > 0.0:
                metrics[f"{prefix}/adam_exp_avg_sq_rms"] = float((exp_avg_sq_sq_sum / param_count) ** 0.5)
            if param_count > 0 and adam_update_sq_sum > 0.0:
                metrics[f"{prefix}/adam_unit_update_rms"] = float((adam_update_sq_sum / param_count) ** 0.5)

            scheduler = self.schedulers.get(param_group_name)
            if scheduler is not None:
                metrics[f"{prefix}/scheduler_last_epoch"] = float(getattr(scheduler, "last_epoch", -1))
                metrics[f"{prefix}/scheduler_last_lr"] = float(scheduler.get_last_lr()[0])
        return metrics

    def load_optimizers(self, loaded_state: Dict[str, Any]) -> None:
        """Helper to load the optimizer state from previous checkpoint

        Args:
            loaded_state: the state from the previous checkpoint
        """
        for k, v in loaded_state.items():
            self.optimizers[k].load_state_dict(v)

    def load_schedulers(self, loaded_state: Dict[str, Any]) -> None:
        """Helper to load the schedulers state from previous checkpoint

        Args:
            loaded_state: the state from the previous checkpoint
        """
        for k, v in loaded_state.items():
            self.schedulers[k].load_state_dict(v)
