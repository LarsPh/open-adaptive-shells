import json
from pathlib import Path

from adashell_renderer.camera_path import load_camera_path
from adashell_renderer.config_io import load_renderer_config
from adashell_renderer.drjit_backend.stage_adapter import apply_fastpath_defaults, build_fastpath_pipeline
from reference_raygen import generate_rays_reference


def test_pred_gradient_config_points_to_local_assets() -> None:
    config_path = (Path(__file__).resolve().parents[1] / "repro/khady_stage2_infer_prednorm/khady.json")
    config = load_renderer_config(config_path)

    assert config.assets.checkpoint is not None
    assert config.assets.checkpoint.is_relative_to(config_path.parent)
    assert config.mesh_assets.dilation_mesh is not None
    assert config.assets.camera_path is not None
    assert config.assets.config_root == config_path.parent
    assert config.assets.config_path == config_path
    assert config.use_fastpath is True
    assert config.profiling.strict_timing is False


def test_pred_gradient_config_selects_pred_gradient_alpha() -> None:
    config_path = (Path(__file__).resolve().parents[1] / "repro/khady_stage2_infer_prednorm/khady.json")
    config = load_renderer_config(config_path)

    assert config.render.alpha_mode == "pred_gradient"
    assert config.field.pred_gradient_alpha_min_norm == 1e-3
    assert config.field.pred_gradient_alpha_max_norm == 10.0


def test_horse_pred_gradient_config_is_kept_for_rc_timing() -> None:
    config_path = (Path(__file__).resolve().parents[1] / "repro/khady_stage2_infer_prednorm/horse.json")
    config = load_renderer_config(config_path)

    assert config.assets.checkpoint is not None
    assert config.assets.camera_path is not None
    assert config.render.alpha_mode == "pred_gradient"
    assert config.field.color_normal_source == "pred"
    assert config.field.use_drjit_hash_encoding is True


def test_rc_configs_do_not_use_legacy_switches() -> None:
    config_paths = [
        (Path(__file__).resolve().parents[1] / "repro/khady_stage2_infer_prednorm/khady.json"),
        (Path(__file__).resolve().parents[1] / "repro/khady_stage2_infer_prednorm/horse.json"),
    ]
    for config_path in config_paths:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
        assert payload["use_fastpath"] is True
        assert "switches" not in payload
        assert "fast_render_path" not in payload.get("render", {})
        assert "drjit_first_contract" not in payload.get("profiling", {})
        assert "drjit_field_island" not in payload.get("profiling", {})
        assert "pair_metadata_only_pack" not in payload.get("profiling", {})
        assert "eval_num_rays_per_chunk" not in payload.get("render", {})


def test_fastpath_preset_overrides_supported_route_fields() -> None:
    config_path = (Path(__file__).resolve().parents[1] / "repro/khady_stage2_infer_prednorm/khady.json")
    config = load_renderer_config(config_path)
    config.render.alpha_mode = "laplace"
    config.field.color_normal_source = "autograd"
    config.field.use_drjit_hash_encoding = False

    apply_fastpath_defaults(config)

    assert config.render.alpha_mode == "pred_gradient"
    assert config.field.color_normal_source == "pred"
    assert config.field.use_drjit_hash_encoding is True


def test_use_fastpath_false_does_not_enable_unsupported_route() -> None:
    config_path = (Path(__file__).resolve().parents[1] / "repro/khady_stage2_infer_prednorm/khady.json")
    config = load_renderer_config(config_path)
    config.use_fastpath = False
    config.render.alpha_mode = "laplace"

    try:
        build_fastpath_pipeline(config)
    except ValueError as exc:
        assert "only supports render.alpha_mode='pred_gradient'" in str(exc)
    else:
        raise AssertionError("use_fastpath=False must still fail fast for unsupported RC routes.")


def test_camera_path_generates_expected_ray_count() -> None:
    camera_path = load_camera_path(
        (Path(__file__).resolve().parents[1] / "repro/khady_stage2_infer_prednorm/assets/camera_paths/khady_path.json")
    )
    frame = camera_path.frames[0]
    rays = generate_rays_reference(frame, device="cpu", camera_index=0)

    assert rays.num_rays == frame.width * frame.height
    assert tuple(rays.origins.shape) == (frame.width * frame.height, 3)
    assert tuple(rays.directions.shape) == (frame.width * frame.height, 3)


def test_blender_transforms_camera_path_preserves_intrinsics(tmp_path: Path) -> None:
    path = tmp_path / "transforms_test.json"
    path.write_text(
        json.dumps(
            {
                "w": 1920,
                "h": 1080,
                "fl_x": 2666.0,
                "fl_y": 2555.0,
                "cx": 960.0,
                "cy": 540.0,
                "frames": [
                    {
                        "file_path": "test/0001.png",
                        "transform_matrix": [
                            [1.0, 0.0, 0.0, 3.0],
                            [0.0, 1.0, 0.0, -1.0],
                            [0.0, 0.0, 1.0, -7.0],
                            [0.0, 0.0, 0.0, 1.0],
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    camera_path = load_camera_path(path)
    frame = camera_path.frames[0]

    assert frame.width == 1920
    assert frame.height == 1080
    assert frame.fx == 2666.0
    assert frame.fy == 2555.0
    assert frame.cx == 960.0
    assert frame.cy == 540.0
    assert frame.camera_to_world[0, 3].item() == 3.0
