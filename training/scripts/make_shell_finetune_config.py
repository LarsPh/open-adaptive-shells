from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional

import tyro


def _windows_path_yaml(path: Path, indent: str = "  ") -> str:
    parts = path.resolve().parts
    lines = [f"{indent}!!python/object/apply:pathlib.WindowsPath"]
    for part in parts:
        lines.append(f"{indent}- {part}")
    return "\n".join(lines)


@dataclass
class MakeShellFineTuneConfig:
    load_config: Path
    output_config: Path
    checkpoint_dir: Path
    dilation_mesh_path: str
    erosion_mesh_path: str
    shell_ray_tracing_backend: str = "optix"
    shell_use_inner_mesh: bool = True
    stage2_num_iterations: int = 100000
    stage2_enable_pred_normal_loss: bool = False
    stage2_enable_kernel_smoothness_loss: bool = False
    enable_cuda_pair_kernel: bool = True
    enable_cuda_pack_kernel: bool = True
    load_step: Optional[int] = None

    def main(self) -> None:
        text = self.load_config.read_text()
        needle = (
            "  model: !!python/object:nerfstudio.models.adashell.AdashellModelConfig\n"
            "    _target: !!python/name:nerfstudio.models.adashell.AdashellModel ''\n"
        )
        insert = (
            needle
            + "    shell_inference_mode: false\n"
            + "    shell_finetune_mode: true\n"
            + "    stage2_color_only_loss: true\n"
            + f"    stage2_enable_pred_normal_loss: {'true' if self.stage2_enable_pred_normal_loss else 'false'}\n"
            + "    stage2_enable_kernel_smoothness_loss: "
            + f"{'true' if self.stage2_enable_kernel_smoothness_loss else 'false'}\n"
            + f"    enable_cuda_pair_kernel: {'true' if self.enable_cuda_pair_kernel else 'false'}\n"
            + f"    enable_cuda_pack_kernel: {'true' if self.enable_cuda_pack_kernel else 'false'}\n"
            + f"    shell_ray_tracing_backend: {self.shell_ray_tracing_backend}\n"
            + f"    shell_use_inner_mesh: {'true' if self.shell_use_inner_mesh else 'false'}\n"
            + f"    dilation_mesh_path: '{self.dilation_mesh_path}'\n"
            + f"    erosion_mesh_path: '{self.erosion_mesh_path}'\n"
            + "    optix_debug_render: false\n"
        )
        if needle not in text:
            raise ValueError("Could not locate Adashell model block in config file.")
        text = text.replace(needle, insert, 1)

        ckpt_yaml = _windows_path_yaml(self.checkpoint_dir, indent="    ")
        text = re.sub(r"(?m)^  load_dir:.*$", f"  load_dir: {ckpt_yaml}", text, count=1)
        if self.load_step is not None:
            text = re.sub(r"(?m)^  load_step:.*$", f"  load_step: {int(self.load_step)}", text, count=1)

        text = re.sub(
            r"(?m)^  max_num_iterations:.*$",
            f"  max_num_iterations: {int(self.stage2_num_iterations)}",
            text,
            count=1,
        )
        text = re.sub(r'(?m)^timestamp:.*$', 'timestamp: "{timestamp}"', text, count=1)
        self.output_config.write_text(text)
        print(f"Wrote shell fine-tune config to {self.output_config}")


def entrypoint() -> None:
    tyro.cli(MakeShellFineTuneConfig).main()


if __name__ == "__main__":
    entrypoint()
