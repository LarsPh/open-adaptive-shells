from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro


@dataclass
class MakeShellRenderConfig:
    load_config: Path
    output_config: Path
    dilation_mesh_path: str
    erosion_mesh_path: str
    shell_ray_tracing_backend: str = "optix"
    shell_use_inner_mesh: bool = True
    optix_debug_render: bool = True
    optix_debug_validate_camera_conversion: bool = False

    def main(self) -> None:
        text = self.load_config.read_text()
        needle = (
            "  model: !!python/object:nerfstudio.models.adashell.AdashellModelConfig\n"
            "    _target: !!python/name:nerfstudio.models.adashell.AdashellModel ''\n"
        )
        insert = (
            needle
            + f"    shell_inference_mode: true\n"
            + f"    shell_ray_tracing_backend: {self.shell_ray_tracing_backend}\n"
            + f"    shell_use_inner_mesh: {'true' if self.shell_use_inner_mesh else 'false'}\n"
            + f"    dilation_mesh_path: '{self.dilation_mesh_path}'\n"
            + f"    erosion_mesh_path: '{self.erosion_mesh_path}'\n"
            + f"    optix_debug_render: {'true' if self.optix_debug_render else 'false'}\n"
            + f"    optix_debug_validate_camera_conversion: {'true' if self.optix_debug_validate_camera_conversion else 'false'}\n"
        )
        if needle not in text:
            raise ValueError("Could not locate Adashell model block in config file.")
        self.output_config.write_text(text.replace(needle, insert, 1))
        print(f"Wrote shell render config to {self.output_config}")


def entrypoint() -> None:
    tyro.cli(MakeShellRenderConfig).main()


if __name__ == "__main__":
    entrypoint()
