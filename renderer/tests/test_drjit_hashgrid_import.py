from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import textwrap

import pytest
import torch


@pytest.mark.parametrize(
    ("smoothstep", "label"),
    [
        (False, "Linear"),
        (True, "Smoothstep"),
    ],
)
def test_hashgrid_importer_subprocess_parity(smoothstep: bool, label: str) -> None:
    pytest.importorskip("drjit")
    pytest.importorskip("tinycudann")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Dr.Jit/TCNN hashgrid importer test.")

    script = textwrap.dedent(
        f"""
        import torch
        import tinycudann as tcnn
        import drjit as dr

        from adashell_renderer.config import RendererFieldConfig
        from adashell_renderer.drjit_backend.hashgrid_import import (
            build_dr_hashgrid_encoding,
            load_tcnn_hashgrid_params_into_dr_encoding,
            summarize_hashgrid_layout,
        )

        cfg = RendererFieldConfig(
            num_levels=14,
            hash_features_per_level=2,
            log2_hashmap_size=22,
            base_res=16,
            max_res=4096,
            hash_smoothstep={smoothstep},
        )

        tenc = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={{
                "otype": "HashGrid",
                "n_levels": cfg.num_levels,
                "n_features_per_level": cfg.hash_features_per_level,
                "log2_hashmap_size": cfg.log2_hashmap_size,
                "base_resolution": cfg.base_res,
                "per_level_scale": __import__("math").exp((__import__("math").log(cfg.max_res) - __import__("math").log(cfg.base_res)) / (cfg.num_levels - 1)),
                "interpolation": "{label}",
            }},
            dtype=torch.float32,
        ).cuda()

        denc = build_dr_hashgrid_encoding(cfg, smoothstep=cfg.hash_smoothstep)
        layout = summarize_hashgrid_layout(denc)
        if layout["padding_param_delta"] != 24:
            raise SystemExit(f"unexpected padding delta: {{layout['padding_param_delta']}}")

        loaded = load_tcnn_hashgrid_params_into_dr_encoding(tenc.params, denc)
        if loaded.numel() != int(denc.params.shape[0]):
            raise SystemExit("imported flat parameter count does not match Dr encoding")

        x = torch.rand((128, 3), device="cuda", dtype=torch.float32)
        y_tcnn = tenc(x)
        y_dr = denc(dr.cuda.TensorXf(x.t().contiguous())).torch().to(device="cuda", dtype=torch.float32).t().contiguous()
        diff = (y_tcnn - y_dr).abs()
        mean = diff.mean().item()
        maxv = diff.max().item()
        if mean > 1e-4 or maxv > 5e-4:
            raise SystemExit(f"hashgrid parity mismatch ({label}): mean={{mean}}, max={{maxv}}")
        print("ok")
        """
    )

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr


@pytest.mark.parametrize(
    ("smoothstep", "label"),
    [
        (False, "Linear"),
        (True, "Smoothstep"),
    ],
)
def test_legacy_field_dr_hashgrid_subprocess_parity(smoothstep: bool, label: str) -> None:
    pytest.importorskip("drjit")
    pytest.importorskip("tinycudann")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the legacy field Dr hashgrid parity test.")

    script = textwrap.dedent(
        f"""
        import torch

        from adashell_renderer.config import RendererFieldConfig
        from adashell_renderer.legacy_field import LegacyAdashellField

        cfg = RendererFieldConfig(
            hash_smoothstep={smoothstep},
            use_drjit_hash_encoding=True,
            use_drjit_geo_mlp=True,
        )
        field = LegacyAdashellField(cfg, num_images=1).cuda()
        x = torch.rand((128, 3), device="cuda", dtype=torch.float32) * 4.0 - 2.0

        # Test oracle composes checkpoint layers directly, outside the runtime.
        features = field.encoding((x + 2.0) * 0.25).float()
        y_torch = torch.cat((x, features), dim=-1)
        for index in range(field.num_layers - 1):
            y_torch = getattr(field, f"glin{{index}}")(y_torch)
            if index < field.num_layers - 2:
                y_torch = torch.nn.functional.softplus(y_torch, beta=100)
        dr = field.geo_mlp_dr.interop.dr
        point_cols = tuple(dr.cuda.Float(x[:, index].contiguous()) for index in range(3))
        feature_cols = tuple(field.hash_encoding_dr(tuple((col + 2.0) * 0.25 for col in point_cols)))
        y_dr = field.geo_mlp_dr.tensor_to_torch(field.geo_mlp_dr.forward_dr(point_cols + feature_cols), x.device)
        diff = (y_torch - y_dr).abs()
        mean = diff.mean().item()
        maxv = diff.max().item()
        if mean > 5e-3 or maxv > 2e-2:
            raise SystemExit(f"legacy field Dr hashgrid geonet mismatch ({label}): mean={{mean}}, max={{maxv}}")
        print("ok")
        """
    )

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr

