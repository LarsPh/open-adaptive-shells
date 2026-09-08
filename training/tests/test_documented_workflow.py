"""Check the documented commands and config handoff without starting training."""

import json
from pathlib import Path
import unittest
from unittest.mock import patch

import scripts.train as train
import tyro
import yaml

from scripts.extract_mesh import ExtractMesh


ROOT = Path(__file__).resolve().parents[1]


def load_example(scene, stage):
    path = ROOT / f"configs/examples/{scene}_stage{stage}.yml"
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.Loader)


class DocumentedWorkflowTests(unittest.TestCase):
    def test_stage_and_renderer_paths(self):
        for scene in ("khady", "horse"):
            with self.subTest(scene=scene):
                first = load_example(scene, 1)
                second = load_example(scene, 2)
                self.assertEqual(second.trainer.load_dir, first.get_checkpoint_dir())
                self.assertEqual(first.pipeline.model.field_backend, "torch")
                self.assertEqual(second.pipeline.model.field_backend, "torch")
                self.assertFalse(first.pipeline.model.shell_finetune_mode)
                self.assertTrue(second.pipeline.model.shell_finetune_mode)
                self.assertEqual(second.pipeline.model.sdf_field.alpha_mode, "pred_gradient")
                self.assertEqual(second.pipeline.model.sdf_field.color_normal_source, "pred")
                self.assertEqual(first.data, Path("data") / scene)
                self.assertEqual(second.data, first.data)

                path = ROOT.parent / f"renderer/configs/{scene}.json"
                renderer = json.loads(path.read_text())
                def resolve(value):
                    return (path.parent / value).resolve()
                final_step = first.trainer.max_num_iterations + second.trainer.max_num_iterations - 1
                checkpoint = second.get_checkpoint_dir() / f"step-{final_step:09d}.ckpt"
                self.assertEqual(resolve(renderer["checkpoint"]), (ROOT / checkpoint).resolve())
                for name in ("dilation", "erosion"):
                    mesh = getattr(second.pipeline.model, f"{name}_mesh_path")
                    self.assertFalse(Path(mesh).is_absolute())
                    self.assertEqual(
                        resolve(renderer["mesh_assets"][f"{name}_mesh"]),
                        (ROOT / mesh).resolve(),
                    )
                self.assertEqual(
                    resolve(renderer["camera_path"]),
                    (ROOT / second.data / "transforms_test.json").resolve(),
                )

    def test_checkpoint_topology_is_consistent(self):
        shape_fields = (
            "num_layers", "num_layers_color", "hidden_dim", "hidden_dim_color",
            "geo_feat_dim", "appearance_embedding_dim", "num_levels",
            "hash_features_per_level", "log2_hashmap_size", "base_res", "max_res",
        )
        for scene in ("khady", "horse"):
            first = load_example(scene, 1).pipeline.model.sdf_field
            second = load_example(scene, 2).pipeline.model.sdf_field
            renderer = json.loads((ROOT.parent / f"renderer/configs/{scene}.json").read_text())
            for name in shape_fields:
                with self.subTest(scene=scene, field=name):
                    self.assertEqual(getattr(first, name), getattr(second, name))
                    self.assertEqual(getattr(second, name), renderer["field"][name])
            self.assertEqual(first.weight_norm, second.weight_norm)

    def test_training_cli_loads_the_examples(self):
        for scene in ("khady", "horse"):
            for stage in (1, 2):
                with self.subTest(scene=scene, stage=stage):
                    config_path = ROOT / f"configs/examples/{scene}_stage{stage}.yml"
                    argv = ["ns-train", "adashell", "--trainer.load-config", str(config_path)]
                    with (
                        patch("sys.argv", argv),
                        patch.object(train, "launch") as launch,
                        patch.object(train.cfg.Config, "save_config"),
                        patch.object(train.cfg.Config, "print_to_terminal"),
                        patch.object(train.CONSOLE, "log"),
                    ):
                        train.entrypoint()
                    config = launch.call_args.kwargs["config"]
                    self.assertEqual(config.experiment_name, scene)
                    self.assertEqual(config.timestamp, f"stage{stage}")
                    self.assertEqual(config.pipeline.datamanager.dataparser.data, Path("data") / scene)

    def test_shell_cli_arguments(self):
        command = tyro.cli(
            tyro.conf.FlagConversionOff[ExtractMesh],
            args=[
                "--load-config", "outputs/khady/adashell/stage1/config.yml",
                "--output-path", "outputs/khady/adashell/stage1/shell.ply",
                "--use-shell", "True", "--resolution", "512",
            ],
        )
        self.assertTrue(command.use_shell)
        self.assertEqual(command.resolution, 512)
        second = load_example("khady", 2)
        self.assertEqual(
            command.output_path.with_name("shell-dilation.ply"),
            Path(second.pipeline.model.dilation_mesh_path),
        )


if __name__ == "__main__":
    unittest.main()
