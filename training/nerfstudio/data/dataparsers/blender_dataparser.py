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

"""Data parser for blender dataset"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Type

import imageio
import numpy as np
import torch

from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.colors import get_color
from nerfstudio.utils.io import load_from_json


def _scene_box_from_blender_meta(meta: dict, fallback_bound: float) -> SceneBox:
    """Build scene bounds from Blender-style metadata when available."""
    if "aabb" in meta:
        aabb = np.asarray(meta["aabb"], dtype=np.float32)
        if aabb.shape != (2, 3):
            raise ValueError(f"Expected transforms JSON `aabb` to have shape (2, 3), got {aabb.shape}.")
        return SceneBox(aabb=torch.from_numpy(aabb))

    if "aabb_scale" in meta:
        bound = float(meta["aabb_scale"])
    else:
        bound = float(fallback_bound)
    return SceneBox(aabb=torch.tensor([[-bound, -bound, -bound], [bound, bound, bound]], dtype=torch.float32))


@dataclass
class BlenderDataParserConfig(DataParserConfig):
    """Blender dataset parser config"""

    _target: Type = field(default_factory=lambda: Blender)
    """target class to instantiate"""
    data: Path = Path("data/blender/lego")
    """Directory specifying location of data."""
    scale_factor: float = 1.0
    """How much to scale the camera origins by."""
    scene_box_bound: float = 1.5
    """Half side length of the axis-aligned scene box."""
    alpha_color: str = "white"
    """alpha color of background"""


@dataclass
class Blender(DataParser):
    """Blender Dataset
    Some of this code comes from https://github.com/yenchenlin/nerf-pytorch/blob/master/load_blender.py#L37.
    """

    config: BlenderDataParserConfig

    def __init__(self, config: BlenderDataParserConfig):
        super().__init__(config=config)
        self.data: Path = config.data
        self.scale_factor: float = config.scale_factor
        self.alpha_color = config.alpha_color

    def _generate_dataparser_outputs(self, split="train"):
        if self.alpha_color is not None:
            alpha_color_tensor = get_color(self.alpha_color)
        else:
            alpha_color_tensor = None

        transform_path = self.data / f"transforms_{split}.json"
        if split == "val" and not transform_path.exists():
            test_transform_path = self.data / "transforms_test.json"
            if test_transform_path.exists():
                transform_path = test_transform_path
        meta = load_from_json(transform_path)
        image_filenames = []
        poses = []
        for frame in meta["frames"]:
            image_path = Path(frame["file_path"].replace("./", ""))
            if image_path.suffix == "":
                image_path = image_path.with_suffix(".png")
            fname = self.data / image_path
            image_filenames.append(fname)
            poses.append(np.array(frame["transform_matrix"]))
        poses = np.array(poses).astype(np.float32)

        img_0 = imageio.imread(image_filenames[0])
        image_height, image_width = img_0.shape[:2]
        camera_angle_x = float(meta["camera_angle_x"])
        focal_length = 0.5 * image_width / np.tan(0.5 * camera_angle_x)

        cx = image_width / 2.0
        cy = image_height / 2.0
        camera_to_world = torch.from_numpy(poses[:, :3])  # camera to world transform

        # in x,y,z order
        camera_to_world[..., 3] *= self.scale_factor
        scene_box = _scene_box_from_blender_meta(meta, fallback_bound=self.config.scene_box_bound)

        cameras = Cameras(
            camera_to_worlds=camera_to_world,
            fx=focal_length,
            fy=focal_length,
            cx=cx,
            cy=cy,
            camera_type=CameraType.PERSPECTIVE,
        )

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            alpha_color=alpha_color_tensor,
            scene_box=scene_box,
        )

        return dataparser_outputs
