"""
Tests for torch_tcnn_compat Adashell helpers.
"""
import torch

from nerfstudio.fields.adashell_field import (
    AdashellFieldConfig,
    AdashellTorchCompatField,
    mlp_export_layout,
)


def test_mlp_export_layout_alignment():
    """Export layout should carry TCNN alignment policy."""
    fused = mlp_export_layout(
        n_input_dims=31,
        n_hidden_layers=1,
        n_neurons=64,
        n_output_dims=31,
        target_otype="FullyFusedMLP",
    )
    cutlass = mlp_export_layout(
        n_input_dims=31,
        n_hidden_layers=1,
        n_neurons=64,
        n_output_dims=31,
        target_otype="CutlassMLP",
    )
    assert fused[0]["alignment"] == 16
    assert cutlass[0]["alignment"] == 8
    assert fused[0]["padded_in_dim"] >= fused[0]["in_dim"]
    assert cutlass[0]["padded_out_dim"] >= cutlass[0]["out_dim"]


def test_adashell_torch_compat_export_metadata():
    """Torch compat field should expose deterministic export metadata."""
    # pylint: disable=import-outside-toplevel
    try:
        import tinycudann as _tcnn  # noqa: F401
    except ImportError as err:
        print(err)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = AdashellFieldConfig(
        use_grid_feature=True,
        encoding_type="hash",
        num_layers=1,
        num_layers_color=2,
        hidden_dim=64,
        hidden_dim_color=64,
        geo_feat_dim=26,
        torch_tcnn_compat_no_bias=True,
        torch_tcnn_compat_disable_skip=True,
        torch_tcnn_compat_disable_weight_norm=True,
        torch_tcnn_compat_disable_geometric_init=True,
        geometric_init=False,
        weight_norm=False,
        tcnn_geo_network_otype="FullyFusedMLP",
        tcnn_color_network_otype="FullyFusedMLP",
    )
    aabb = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], dtype=torch.float32, device=device)
    field = AdashellTorchCompatField(config=cfg, aabb=aabb, num_images=1).to(device)

    metadata = field.get_tcnn_compat_export_metadata()
    tensors = field.get_tcnn_compat_export_tensors()
    geo_layers = metadata["geo_config"]["layers"]  # type: ignore[index]
    color_layers = metadata["color_config"]["layers"]  # type: ignore[index]
    geo_numel = sum(layer["numel"] for layer in geo_layers)  # type: ignore[arg-type]
    color_numel = sum(layer["numel"] for layer in color_layers)  # type: ignore[arg-type]

    assert metadata["field_backend"] == "torch_tcnn_compat"
    assert metadata["requires_bias_free"] is True
    assert int(tensors["geo_params"].numel()) == int(geo_numel)
    assert int(tensors["color_params"].numel()) == int(color_numel)
