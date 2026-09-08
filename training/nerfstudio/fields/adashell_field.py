# Copyright 2022 The Nerfstudio Team. All rights reserved.
# Modifications Copyright (c) 2026 Zhaorong Wang.
# SPDX-License-Identifier: Apache-2.0
# Adapted from SDFStudio for adaptive-shell fields and predicted gradients.

"""
Adashell Nerfstudio Field
"""

from typing import Any, Literal, Optional, cast, List, Tuple, Dict, Union
import time

from torch import Tensor

from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field  # for custom Field
import math
from dataclasses import dataclass, field
from typing import Optional, Type, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.parameter import Parameter
from torchtyping import TensorType
from typing_extensions import Literal

from nerfstudio.cameras.rays import RaySamples
from nerfstudio.field_components.embedding import Embedding
from nerfstudio.field_components.encodings import (
    SHEncoding,
    PeriodicVolumeEncoding,
    TensorVMEncoding,
)
from nerfstudio.field_components.field_heads import FieldHeadNames
from nerfstudio.field_components.spatial_distortions import SpatialDistortion
from nerfstudio.fields.base_field import Field, FieldConfig

tcnn: Any = None
try:
    import tinycudann as tcnn
except ImportError:
    # tinycudann module doesn't exist
    tcnn = None


class LaplaceDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Laplace density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(
        self, sdf: TensorType["bs":...], beta: Union[TensorType["bs":...], None] = None
    ) -> TensorType["bs":...]:
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""
        # beta is predicted inv_s in adashell
        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


class SigmoidDensity(nn.Module):  # alpha * Laplace(loc=0, scale=beta).cdf(-sdf)
    """Sigmoid density from VolSDF"""

    def __init__(self, init_val, beta_min=0.0001):
        super().__init__()
        self.register_parameter("beta_min", nn.Parameter(beta_min * torch.ones(1), requires_grad=False))
        self.register_parameter("beta", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(
        self, sdf: TensorType["bs":...], beta: Union[TensorType["bs":...], None] = None
    ) -> TensorType["bs":...]:
        """convert sdf value to density value with beta, if beta is missing, then use learable beta"""

        if beta is None:
            beta = self.get_beta()

        alpha = 1.0 / beta

        # negtive sdf will have large density
        return alpha * torch.sigmoid(-sdf * alpha)

    def get_beta(self):
        """return current beta value"""
        beta = self.beta.abs() + self.beta_min
        return beta


class SingleVarianceNetwork(nn.Module):
    """Variance network in NeuS

    Args:
        nn (_type_): init value in NeuS variance network
    """

    def __init__(self, init_val):
        super(SingleVarianceNetwork, self).__init__()
        self.register_parameter("variance", nn.Parameter(init_val * torch.ones(1), requires_grad=True))

    def forward(self, x):
        """Returns current variance value"""
        return torch.ones([len(x), 1], device=x.device) * torch.exp(self.variance * 10.0)

    def get_variance(self):
        """return current variance value"""
        return torch.exp(self.variance * 10.0).clip(1e-6, 1e6)

def transform_inv_s(nn_inv_s, mode="exp", exp_factor=10.0):
    if mode == "exp":
        # Clamp before exp so the forward range matches the old exp(...).clip(...)
        # intent without creating inf values that can poison the backward pass.
        exp_arg = nn_inv_s * exp_factor
        exp_arg = exp_arg.clamp(min=math.log(1e-6), max=math.log(1e6))
        return torch.exp(exp_arg)
    elif mode == "clip":
        return nn_inv_s.clip(1e-6, 1e6)
    elif mode == "none":
        return nn_inv_s
    else:
        raise ValueError(f"Invalid mode: {mode}")


def inv_s_to_kernel_size(inv_s: Tensor, eps: float = 1e-6) -> Tensor:
    """Convert NeuS sharpness inv_s to adaptive kernel size K."""
    return 1.0 / (inv_s + eps)


def tcnn_otype_alignment(otype: str) -> int:
    """Return expected alignment used by TCNN MLP layouts."""
    if otype == "FullyFusedMLP":
        return 16
    if otype == "CutlassMLP":
        return 8
    raise ValueError(f"Unsupported TCNN otype: {otype}")


def pad_dim_for_tcnn(dim: int, alignment: int) -> int:
    """Pad dimension to TCNN alignment."""
    if alignment <= 0:
        raise ValueError(f"Invalid alignment: {alignment}")
    return int(((dim + alignment - 1) // alignment) * alignment)


def mlp_layer_shapes(n_input_dims: int, n_hidden_layers: int, n_neurons: int, n_output_dims: int) -> List[Tuple[int, int]]:
    """Return visible layer (in_dim, out_dim) pairs for an MLP."""
    dims = [int(n_input_dims)]
    dims.extend([int(n_neurons)] * int(n_hidden_layers))
    dims.append(int(n_output_dims))
    return [(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]


def mlp_export_layout(
    n_input_dims: int,
    n_hidden_layers: int,
    n_neurons: int,
    n_output_dims: int,
    target_otype: str,
) -> List[Dict[str, int]]:
    """Build layer metadata including TCNN padding policy."""
    alignment = tcnn_otype_alignment(target_otype)
    layers: List[Dict[str, int]] = []
    offset = 0
    for idx, (in_dim, out_dim) in enumerate(
        mlp_layer_shapes(n_input_dims, n_hidden_layers, n_neurons, n_output_dims)
    ):
        numel = int(in_dim * out_dim)
        padded_in = pad_dim_for_tcnn(in_dim, alignment)
        padded_out = pad_dim_for_tcnn(out_dim, alignment)
        padded_numel = int(padded_in * padded_out)
        layers.append(
            {
                "layer_index": idx,
                "in_dim": in_dim,
                "out_dim": out_dim,
                "padded_in_dim": padded_in,
                "padded_out_dim": padded_out,
                "alignment": alignment,
                "numel": numel,
                "padded_numel": padded_numel,
                "offset": offset,
            }
        )
        offset += numel
    return layers


class _ExpActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.exp(x)


class _SineActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(x)


class _SquareplusActivation(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return 0.5 * (x + torch.sqrt(x * x + 4.0))


def build_torch_activation(activation: str) -> nn.Module:
    """Map TCNN-style activation name to torch module."""
    if activation == "ReLU":
        return nn.ReLU()
    if activation == "Exponential":
        return _ExpActivation()
    if activation == "Sine":
        return _SineActivation()
    if activation == "Sigmoid":
        return nn.Sigmoid()
    if activation == "Squareplus":
        return _SquareplusActivation()
    if activation == "Softplus":
        return nn.Softplus()
    if activation == "Tanh":
        return nn.Tanh()
    raise ValueError(f"Unsupported torch activation for TCNN-compat path: {activation}")


class TorchCompatMLP(nn.Module):
    """Plain MLP that mirrors TCNN hidden-layer semantics."""

    def __init__(
        self,
        n_input_dims: int,
        n_output_dims: int,
        n_hidden_layers: int,
        n_neurons: int,
        activation: str,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if n_hidden_layers < 0:
            raise ValueError(f"n_hidden_layers must be >= 0, got {n_hidden_layers}")
        self.n_input_dims = int(n_input_dims)
        self.n_output_dims = int(n_output_dims)
        self.n_hidden_layers = int(n_hidden_layers)
        self.n_neurons = int(n_neurons)
        self.activation_name = activation
        self.use_bias = bool(bias)
        self.hidden_activation = build_torch_activation(activation)
        dims = [self.n_input_dims]
        dims.extend([self.n_neurons] * self.n_hidden_layers)
        dims.append(self.n_output_dims)
        self.layer_dims = dims
        self.layers = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1], bias=self.use_bias) for i in range(len(dims) - 1)]
        )
        for layer in self.layers:
            torch.nn.init.kaiming_uniform_(layer.weight.data)
            if layer.bias is not None:
                torch.nn.init.zeros_(layer.bias.data)

    def forward(self, x: Tensor) -> Tensor:
        for idx, layer in enumerate(self.layers):
            x = layer(x)
            if idx < len(self.layers) - 1:
                x = self.hidden_activation(x)
        return x

    def get_export_layout(self, target_otype: str) -> List[Dict[str, int]]:
        return mlp_export_layout(
            n_input_dims=self.n_input_dims,
            n_hidden_layers=self.n_hidden_layers,
            n_neurons=self.n_neurons,
            n_output_dims=self.n_output_dims,
            target_otype=target_otype,
        )

    def flatten_weights(self) -> Tensor:
        chunks = [layer.weight.detach().reshape(-1) for layer in self.layers]
        if not chunks:
            return torch.empty(0, dtype=torch.float32)
        return torch.cat(chunks, dim=0)


@dataclass
class AdashellFieldConfig(FieldConfig):
    """Adashell Field Config"""

    _target: Type = field(default_factory=lambda: AdashellField)
    """Adashell Field Config"""
    num_layers: int = 1
    """Number of layers for geometric network"""
    hidden_dim: int = 64
    """Number of hidden dimension of geometric network"""
    geo_feat_dim: int = 26
    """Dimension of geometric feature"""
    num_layers_color: int = 2
    """Number of layers for color network"""
    hidden_dim_color: int = 64
    """Number of hidden dimension of color network"""
    appearance_embedding_dim: int = 32
    """Dimension of appearance embedding"""
    use_appearance_embedding: bool = False
    """Dimension of appearance embedding"""
    bias: float = 0.8
    """sphere size of geometric initializaion"""
    geometric_init: bool = True
    """Whether to use geometric initialization"""
    inside_outside: bool = True
    """whether to revert signed distance value, set to True for indoor scene"""
    weight_norm: bool = True
    """Whether to use weight norm for linear laer"""
    use_grid_feature: bool = False
    """Whether to use multi-resolution feature grids"""
    divide_factor: float = 2.0
    """Normalization factor for multi-resolution grids"""
    beta_init: float = 0.1
    """Init learnable beta value for transformation of sdf to density"""
    encoding_type: Literal["hash", "periodic", "tensorf_vm"] = "hash"
    """feature grid encoding type"""
    #position_encoding_max_degree: int = 6
    #"""positional encoding max degree"""
    use_diffuse_color: bool = False
    """whether to use diffuse color as in ref-nerf"""
    use_specular_tint: bool = False
    """whether to use specular tint as in ref-nerf"""
    use_reflections: bool = False
    """whether to use reflections as in ref-nerf"""
    use_n_dot_v: bool = False
    """whether to use n dot v as in ref-nerf"""
    rgb_padding: float = 0.001
    """Padding added to the RGB outputs"""
    off_axis: bool = False
    """whether to use off axis encoding from mipnerf360"""
    use_numerical_gradients: bool = True
    """whether to use numercial gradients"""
    inference_use_autograd_gradients: bool = True
    """Use autograd on the current sdf during inference instead of 6-point finite differences."""
    color_normal_source: Literal["gradient", "pred"] = "gradient"
    """Normal source used by the color network shading path."""
    alpha_mode: Literal["neus_gradient", "laplace_density", "pred_gradient"] = "neus_gradient"
    """Alpha computation path. Default preserves the original NeuS-style geometry-gradient path."""
    pred_gradient_alpha_min_norm: float = 1e-3
    """Minimum norm used when the predicted vector is interpreted as an unnormalized gradient for alpha."""
    pred_gradient_alpha_max_norm: float = 10.0
    """Maximum norm used when the predicted vector is interpreted as an unnormalized gradient for alpha."""
    gradient_compute_mode: Literal["always", "auto"] = "always"
    """Whether to always compute true geometry gradients or skip them when alpha/color do not need them."""
    tcnn_geo_network_otype: Literal["FullyFusedMLP", "CutlassMLP"] = "FullyFusedMLP"
    """TCNN geometry network implementation type when using the TCNN field backend."""
    tcnn_color_network_otype: Literal["FullyFusedMLP", "CutlassMLP"] = "FullyFusedMLP"
    """TCNN color network implementation type when using the TCNN field backend."""
    tcnn_geo_activation: Literal["ReLU", "Exponential", "Sine", "Sigmoid", "Squareplus", "Softplus", "Tanh"] = "ReLU"
    """TCNN geometry hidden activation when using the TCNN field backend."""
    tcnn_color_activation: Literal["ReLU", "Exponential", "Sine", "Sigmoid", "Squareplus", "Softplus", "Tanh"] = "ReLU"
    """TCNN color hidden activation when using the TCNN field backend."""
    torch_tcnn_compat_no_bias: bool = True
    """Disable bias in torch_tcnn_compat backend to mirror TCNN weight layout."""
    torch_tcnn_compat_disable_skip: bool = True
    """Disable geometry skip connection in torch_tcnn_compat backend."""
    torch_tcnn_compat_disable_weight_norm: bool = True
    """Disable weight norm in torch_tcnn_compat backend."""
    torch_tcnn_compat_disable_geometric_init: bool = True
    """Disable geometric init in torch_tcnn_compat backend."""
    num_levels: int = 14
    """number of levels for multi-resolution hash grids"""
    max_res: int = 4096
    """max resolution for multi-resolution hash grids"""
    base_res: int = 16
    """base resolution for multi-resolution hash grids"""
    log2_hashmap_size: int = 22
    """log2 hash map size for multi-resolution hash grids"""
    hash_features_per_level: int = 2
    """number of features per level for multi-resolution hash grids"""
    hash_smoothstep: bool = True
    """whether to use smoothstep for multi-resolution hash grids"""
    #use_position_encoding: bool = False
    #"""whether to use positional encoding as input for geometric network"""
    # no position encoding in adashell
    kernel_size_smoothness_sigma: float = 0.0001
    """sigma of Gaussian distribution, which generates position deviation for kernel size smoothness loss"""
    kernel_size_smoothness_sigma_mode: Literal["direct", "squared_legacy"] = "direct"
    """Interpretation of smoothness sigma: direct std or legacy sigma**2."""
    kernel_size_transform_mode: Literal["exp", "clip", "none"] = "clip"
    """Mode for transforming raw network output into NeuS inv_s."""
    kernel_size_transform_exp_factor: float = 10.0
    """Exponent factor for the inv_s transform when mode='exp'."""
    density_beta_mode: Literal["kernel_size", "inv_s_legacy"] = "kernel_size"
    """How VolSDF-style Laplace density beta is derived from inv_s."""


class AdashellField(Field):
    """_summary_

    Args:
        Field (_type_): _description_
    """

    config: AdashellFieldConfig

    def __init__(
        self,
        config: AdashellFieldConfig,
        aabb,
        num_images: int,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        super().__init__()
        self.config = config

        # TODO do we need aabb here?
        self.aabb = Parameter(aabb, requires_grad=False)

        self.spatial_distortion = spatial_distortion
        self.num_images = num_images

        self.embedding_appearance = Embedding(self.num_images, self.config.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_grid_feature = self.config.use_grid_feature
        self.divide_factor = self.config.divide_factor

        self.num_levels = self.config.num_levels
        self.max_res = self.config.max_res 
        self.base_res = self.config.base_res 
        self.log2_hashmap_size = self.config.log2_hashmap_size 
        self.features_per_level = self.config.hash_features_per_level 
        use_hash = True
        smoothstep = self.config.hash_smoothstep
        self.growth_factor = np.exp((np.log(self.max_res) - np.log(self.base_res)) / (self.num_levels - 1))

        if self.config.encoding_type == "hash":
            # feature encoding
            self.encoding = tcnn.Encoding(
                n_input_dims=3,
                encoding_config={
                    "otype": "HashGrid" if use_hash else "DenseGrid",
                    "n_levels": self.num_levels,
                    "n_features_per_level": self.features_per_level,
                    "log2_hashmap_size": self.log2_hashmap_size,
                    "base_resolution": self.base_res,
                    "per_level_scale": self.growth_factor,
                    "interpolation": "Smoothstep" if smoothstep else "Linear",
                },
            )
            self.hash_encoding_mask = torch.ones(
                self.num_levels * self.features_per_level,
                dtype=torch.float32,
            )

        elif self.config.encoding_type == "periodic":
            print("using periodic encoding")
            self.encoding = PeriodicVolumeEncoding(
                num_levels=self.num_levels,
                min_res=self.base_res,
                max_res=self.max_res,
                log2_hashmap_size=18,  # 64 ** 3 = 2^18
                features_per_level=self.features_per_level,
                smoothstep=smoothstep,
            )
        elif self.config.encoding_type == "tensorf_vm":
            print("using tensor vm")
            self.encoding = TensorVMEncoding(128, 24, smoothstep=smoothstep)

        # we concat inputs position ourselves
        # self.position_encoding = NeRFEncoding(
        #     in_dim=3,
        #     num_frequencies=self.config.position_encoding_max_degree,
        #     min_freq_exp=0.0,
        #     max_freq_exp=self.config.position_encoding_max_degree - 1,
        #     include_input=False,
        #     off_axis=self.config.off_axis,
        # )

        # self.direction_encoding = NeRFEncoding(
        #     in_dim=3, num_frequencies=4, min_freq_exp=0.0, max_freq_exp=3.0, include_input=True
        # )

        # use spherical harmonic encoding
        self.direction_encoding = SHEncoding(levels=4) # 16 coefficients

        # TODO move it to field components
        # MLP with geometric initialization
        dims = [self.config.hidden_dim for _ in range(self.config.num_layers)]
        in_dim = 3 + self.encoding.n_output_dims
        dims = [in_dim] + dims + [self.config.geo_feat_dim + 1 + 1 + 3] # 1 (sdf) + 1 (s) + 3 (normal)
        self.num_layers = len(dims)
        # TODO check how to merge skip_in to config
        self.skip_in = [4]

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if self.config.geometric_init:
                if l == self.num_layers - 2:
                    if not self.config.inside_outside:
                        torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, -self.config.bias)
                    else:
                        torch.nn.init.normal_(lin.weight, mean=-np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                        torch.nn.init.constant_(lin.bias, self.config.bias)
                elif l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3) :], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if self.config.weight_norm:
                lin = nn.utils.weight_norm(lin)
                # print("=======", lin.weight.shape)
            setattr(self, "glin" + str(l), lin)

        # laplace function for transform sdf to density from VolSDF
        self.laplace_density = LaplaceDensity(init_val=self.config.beta_init)
        # self.laplace_density = SigmoidDensity(init_val=self.config.beta_init)

        # TODO use different name for beta_init for config
        # deviation_network to compute alpha from sdf from NeuS
        # self.deviation_network = SingleVarianceNetwork(init_val=self.config.beta_init)

        # diffuse and specular tint layer
        if self.config.use_diffuse_color:
            self.diffuse_color_pred = nn.Linear(self.config.geo_feat_dim, 3)
        if self.config.use_specular_tint:
            self.specular_tint_pred = nn.Linear(self.config.geo_feat_dim, 3)

        # view dependent color network
        dims = [self.config.hidden_dim_color for _ in range(self.config.num_layers_color)]
        if self.config.use_diffuse_color:
            in_dim = (
                self.direction_encoding.get_out_dim()
                + self.config.geo_feat_dim
                + self.embedding_appearance.get_out_dim()
            )
        else:
            # point, view_direction, normal, feature, embedding
            in_dim = (
                3
                + self.direction_encoding.get_out_dim() # should be 16
                + 3 # normal
                + self.config.geo_feat_dim
                + self.embedding_appearance.get_out_dim()
            )
        if self.config.use_n_dot_v:
            in_dim += 1

        dims = [in_dim] + dims + [3]
        self.num_layers_color = len(dims)

        for l in range(0, self.num_layers_color - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            torch.nn.init.kaiming_uniform_(lin.weight.data)
            torch.nn.init.zeros_(lin.bias.data)

            if self.config.weight_norm:
                lin = nn.utils.weight_norm(lin)
            # print("=======", lin.weight.shape)
            setattr(self, "clin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

        self._cos_anneal_ratio = 1.0
        self.numerical_gradients_delta = 0.0001
        self._profile_strict_timing = False
        self._last_profile = None

    def set_profile_options(self, strict_timing: bool = False) -> None:
        self._profile_strict_timing = bool(strict_timing)

    def _sync_cuda_for_profile(self, device: torch.device) -> None:
        if self._profile_strict_timing and device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device=device)

    def get_last_profile(self) -> Optional[dict]:
        return self._last_profile

    def split_geonet_out(self, geo_feat):
        return torch.split(geo_feat, [1, 1, self.config.geo_feat_dim, 3], dim=-1)

    def _use_predicted_normals_for_color(self) -> bool:
        """Resolve shading normal source."""
        return self.config.color_normal_source == "pred"

    def _summarize_single_sample_gradients(
        self,
        ray_samples: RaySamples,
        surface_gradients: torch.Tensor,
        directions_flat: torch.Tensor,
    ) -> dict:
        """Collect gradient statistics for rays that contribute exactly one valid shell sample."""
        metadata = ray_samples.metadata or {}
        single_sample_mask = None

        shell_valid_count_per_sample = cast(Optional[torch.Tensor], metadata.get("shell_valid_count_per_sample"))
        if shell_valid_count_per_sample is not None:
            single_sample_mask = shell_valid_count_per_sample.reshape(-1) == 1
        else:
            shell_mask = cast(Optional[torch.Tensor], metadata.get("shell_mask"))
            if shell_mask is not None:
                shell_mask_2d = shell_mask.reshape(*shell_mask.shape[:-1])
                shell_mask_flat = shell_mask_2d.reshape(-1) > 0.0
                valid_per_ray = shell_mask_2d.sum(dim=-1, keepdim=True)
                single_sample_per_ray = (valid_per_ray == 1).expand_as(shell_mask_2d)
                single_sample_mask = shell_mask_flat & single_sample_per_ray.reshape(-1)

        if single_sample_mask is None or not bool(torch.any(single_sample_mask).item()):
            return {
                "single_sample_count": 0,
                "single_sample_grad_norm_sum": 0.0,
                "single_sample_grad_norm_sq_sum": 0.0,
                "single_sample_grad_norm_min": 0.0,
                "single_sample_grad_norm_max": 0.0,
                "single_sample_abs_true_cos_sum": 0.0,
            }

        single_gradients = surface_gradients[single_sample_mask]
        single_dirs = directions_flat[single_sample_mask]
        grad_norms = single_gradients.norm(dim=-1)
        abs_true_cos = (single_dirs * single_gradients).sum(dim=-1).abs()

        return {
            "single_sample_count": int(grad_norms.shape[0]),
            "single_sample_grad_norm_sum": float(grad_norms.sum().item()),
            "single_sample_grad_norm_sq_sum": float((grad_norms.square().sum()).item()),
            "single_sample_grad_norm_min": float(grad_norms.min().item()),
            "single_sample_grad_norm_max": float(grad_norms.max().item()),
            "single_sample_abs_true_cos_sum": float(abs_true_cos.sum().item()),
        }

    def set_cos_anneal_ratio(self, anneal: float) -> None:
        """Set the anneal value for the proposal network."""
        self._cos_anneal_ratio = anneal

    def update_mask(self, level: int):
        self.hash_encoding_mask[:] = 1.0
        self.hash_encoding_mask[level * self.features_per_level:] = 0
        
    def forward_geonetwork(self, inputs):
        """forward the geonetwork"""
        if self.use_grid_feature:
            #TODO normalize inputs depending on the whether we model the background or not
            positions = (inputs + 2.0) / 4.0
            # positions = (inputs + 1.0) / 2.0
            feature = self.encoding(positions)
            # mask feature
            feature = feature * self.hash_encoding_mask.to(feature.device)
        else:
            feature = torch.zeros_like(inputs[:, :1].repeat(1, self.encoding.n_output_dims))

        # pe = self.position_encoding(inputs)
        # if not self.config.use_position_encoding:
        #     pe = torch.zeros_like(pe)

        # inputs = torch.cat((inputs, pe, feature), dim=-1)
        inputs = torch.cat((inputs, feature), dim=-1)

        x = inputs

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "glin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)
        return x

    def get_sdf(self, ray_samples: RaySamples):
        """predict the sdf value for ray samples"""
        positions = self._get_query_positions(ray_samples)
        positions_flat = positions.view(-1, 3)
        h = self.forward_geonetwork(positions_flat).view(*ray_samples.frustums.shape, -1)
        sdf, _, _, _ = self.split_geonet_out(h)
        return sdf

    def set_numerical_gradients_delta(self, delta: float) -> None:
        """Set the delta value for numerical gradient."""
        self.numerical_gradients_delta = delta

    def get_deviated_inv_s(self, x):
        """get the inv_s after deviating the position"""
        sigma = self.config.kernel_size_smoothness_sigma
        if self.config.kernel_size_smoothness_sigma_mode == "squared_legacy":
            sigma = sigma * sigma
        x_deviated = x + torch.randn_like(x) * sigma
        h_deviated = self.forward_geonetwork(x_deviated)
        _, inv_s_deviated, _, _ = self.split_geonet_out(h_deviated)
        inv_s_deviated = transform_inv_s(inv_s_deviated, mode=self.config.kernel_size_transform_mode, exp_factor=self.config.kernel_size_transform_exp_factor)
        return inv_s_deviated

    def _laplace_beta_from_inv_s(self, inv_s: Tensor) -> Tensor:
        if self.config.density_beta_mode == "kernel_size":
            return inv_s_to_kernel_size(inv_s)
        if self.config.density_beta_mode == "inv_s_legacy":
            return inv_s
        raise ValueError(f"Unknown density_beta_mode: {self.config.density_beta_mode}")

    @staticmethod
    def _apply_shell_mask(alpha: Tensor, ray_samples: RaySamples) -> Tensor:
        if ray_samples.metadata is not None:
            shell_mask = ray_samples.metadata.get("shell_mask")
            if shell_mask is not None:
                alpha = alpha * shell_mask
        return alpha

    def _clamp_predicted_gradient_for_alpha(self, pred_gradient: Tensor) -> Tensor:
        min_norm = max(float(self.config.pred_gradient_alpha_min_norm), 1e-12)
        max_norm = max(float(self.config.pred_gradient_alpha_max_norm), min_norm)
        norm = pred_gradient.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        clamped_norm = norm.clamp(min=min_norm, max=max_norm)
        return pred_gradient * (clamped_norm / norm)

    def _alpha_from_neus_directional_gradient(
        self,
        ray_samples: RaySamples,
        sdf: Tensor,
        inv_s: Tensor,
        directional_gradient: Tensor,
    ) -> Tensor:
        cos_anneal_ratio = self._cos_anneal_ratio
        iter_cos = -(
            F.relu(-directional_gradient * 0.5 + 0.5) * (1.0 - cos_anneal_ratio)
            + F.relu(-directional_gradient) * cos_anneal_ratio
        )

        estimated_next_sdf = sdf + iter_cos * ray_samples.deltas * 0.5
        estimated_prev_sdf = sdf - iter_cos * ray_samples.deltas * 0.5

        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
        alpha = ((prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)).clip(0.0, 1.0)
        return self._apply_shell_mask(alpha, ray_samples)

    def _alpha_from_laplace_density(self, ray_samples: RaySamples, sdf: Tensor, inv_s: Tensor) -> Tensor:
        density = self.laplace_density(sdf, beta=inv_s_to_kernel_size(inv_s))
        alpha = (1.0 - torch.exp(-density * ray_samples.deltas)).clip(0.0, 1.0)
        return self._apply_shell_mask(alpha, ray_samples)

    def _should_compute_geometry_gradients(self, render_minimal: bool, return_occupancy: bool) -> bool:
        if self.config.gradient_compute_mode == "always":
            return True
        if self.config.gradient_compute_mode != "auto":
            raise ValueError(f"Unknown gradient_compute_mode: {self.config.gradient_compute_mode}")
        if self.config.alpha_mode == "neus_gradient":
            return True
        if not self._use_predicted_normals_for_color():
            return True
        # Stage2 color-only training can skip true gradients when alpha and color both use predicted quantities.
        if self.training and (not render_minimal):
            return False
        return return_occupancy

    def gradient(self, x, skip_spatial_distortion=False, return_sdf=False):
        """compute the gradient of the ray"""
        if self.spatial_distortion is not None and not skip_spatial_distortion:
            x = self.spatial_distortion(x)

        # compute gradient in contracted space
        if self.config.use_numerical_gradients:
            # https://github.com/bennyguo/instant-nsr-pl/blob/main/models/geometry.py#L173
            delta = self.numerical_gradients_delta
            points = torch.stack(
                [
                    x + torch.as_tensor([delta, 0.0, 0.0]).to(x),
                    x + torch.as_tensor([-delta, 0.0, 0.0]).to(x),
                    x + torch.as_tensor([0.0, delta, 0.0]).to(x),
                    x + torch.as_tensor([0.0, -delta, 0.0]).to(x),
                    x + torch.as_tensor([0.0, 0.0, delta]).to(x),
                    x + torch.as_tensor([0.0, 0.0, -delta]).to(x),
                ],
                dim=0,
            )

            points_sdf = self.forward_geonetwork(points.view(-1, 3))[..., 0].view(6, *x.shape[:-1])
            gradients = torch.stack(
                [
                    0.5 * (points_sdf[0] - points_sdf[1]) / delta,
                    0.5 * (points_sdf[2] - points_sdf[3]) / delta,
                    0.5 * (points_sdf[4] - points_sdf[5]) / delta,
                ],
                dim=-1,
            )
        else:
            x.requires_grad_(True)

            y = self.forward_geonetwork(x)[:, :1]
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            gradients = torch.autograd.grad(
                outputs=y, inputs=x, grad_outputs=d_output, create_graph=True, retain_graph=True, only_inputs=True
            )[0]
        if not return_sdf:
            return gradients
        else:
            return gradients, points_sdf

    @staticmethod
    def _gradient_from_sdf(sdf: Tensor, inputs: Tensor, create_graph: bool, retain_graph: bool) -> Tensor:
        d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
        return torch.autograd.grad(
            outputs=sdf,
            inputs=inputs,
            grad_outputs=d_output,
            create_graph=create_graph,
            retain_graph=retain_graph,
            only_inputs=True,
        )[0]

    def get_density(self, ray_samples: RaySamples):
        """Computes and returns the densities."""
        positions = self._get_query_positions(ray_samples)
        positions_flat = positions.view(-1, 3)
        h = self.forward_geonetwork(positions_flat).view(*ray_samples.frustums.shape, -1)
        sdf, inv_s, geo_feature, _ = self.split_geonet_out(h)
        inv_s = transform_inv_s(inv_s, mode=self.config.kernel_size_transform_mode, exp_factor=self.config.kernel_size_transform_exp_factor)
        density = self.laplace_density(sdf, beta=self._laplace_beta_from_inv_s(inv_s))
        return density, geo_feature

    def get_alpha(self, ray_samples: RaySamples, sdf=None, inv_s=None, gradients=None, pred_gradient=None):
        """Compute alpha using the configured stage1/stage2 path."""
        alpha_mode = self.config.alpha_mode
        if alpha_mode not in {"neus_gradient", "laplace_density", "pred_gradient"}:
            raise ValueError(f"Unknown alpha_mode: {alpha_mode}")
        if (sdf is None or inv_s is None) or (
            alpha_mode == "neus_gradient" and gradients is None
        ) or (alpha_mode == "pred_gradient" and pred_gradient is None):
            inputs = self._get_query_positions(ray_samples)
            inputs.requires_grad_(True)
            with torch.enable_grad():
                h = self.forward_geonetwork(inputs)
                sdf, inv_s, _, pred_gradient = self.split_geonet_out(h)
                inv_s = transform_inv_s(inv_s, mode=self.config.kernel_size_transform_mode, exp_factor=self.config.kernel_size_transform_exp_factor)
            if alpha_mode == "neus_gradient":
                d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
                gradients = torch.autograd.grad(
                    outputs=sdf,
                    inputs=inputs,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                )[0]

        if alpha_mode == "laplace_density":
            return self._alpha_from_laplace_density(ray_samples, sdf, inv_s)

        if alpha_mode == "pred_gradient":
            pred_gradient = self._clamp_predicted_gradient_for_alpha(cast(Tensor, pred_gradient))
            directional_gradient = (ray_samples.frustums.directions * pred_gradient).sum(-1, keepdim=True)
            return self._alpha_from_neus_directional_gradient(ray_samples, sdf, inv_s, directional_gradient)

        directional_gradient = (ray_samples.frustums.directions * gradients).sum(-1, keepdim=True)
        alpha = self._alpha_from_neus_directional_gradient(ray_samples, sdf, inv_s, directional_gradient)

        # HF-NeuS
        # # sigma
        # cdf = torch.sigmoid(sdf * inv_s)
        # e = inv_s * (1 - cdf) * (-iter_cos) * ray_samples.deltas
        # alpha = (1 - torch.exp(-e)).clip(0.0, 1.0)

        return alpha

    def get_occupancy(self, sdf):
        """compute occupancy as in UniSurf"""
        occupancy = self.sigmoid(-10.0 * sdf)
        return occupancy

    def get_colors(self, points, directions, shading_vectors, geo_features, camera_indices):
        """compute colors"""
        raw_rgb_diffuse: Optional[Tensor] = None
        tint: Optional[Tensor] = None

        # diffuse color and specular tint
        if self.config.use_diffuse_color:
            raw_rgb_diffuse = self.diffuse_color_pred(geo_features.view(-1, self.config.geo_feat_dim))
        if self.config.use_specular_tint:
            tint = self.sigmoid(self.specular_tint_pred(geo_features.view(-1, self.config.geo_feat_dim)))

        normals = F.normalize(shading_vectors, p=2, dim=-1)

        if self.config.use_reflections:
            # https://github.com/google-research/multinerf/blob/5d4c82831a9b94a87efada2eee6a993d530c4226/internal/ref_utils.py#L22
            refdirs = 2.0 * torch.sum(normals * -directions, axis=-1, keepdims=True) * normals + directions
            d = self.direction_encoding(refdirs)
        else:
            d = self.direction_encoding(directions)

        # appearance
        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
            # set it to zero if don't use it
            if not self.config.use_appearance_embedding:
                embedded_appearance = torch.zeros_like(embedded_appearance)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.config.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.config.appearance_embedding_dim), device=directions.device
                )
        if self.config.use_diffuse_color:
            h = [
                d,
                geo_features.view(-1, self.config.geo_feat_dim),
                embedded_appearance.view(-1, self.config.appearance_embedding_dim),
            ]
        else:
            h = [
                points,
                d,
                shading_vectors,
                geo_features.view(-1, self.config.geo_feat_dim),
                embedded_appearance.view(-1, self.config.appearance_embedding_dim),
            ]

        if self.config.use_n_dot_v:
            n_dot_v = torch.sum(normals * directions, dim=-1, keepdims=True)
            h.append(n_dot_v)

        h = torch.cat(h, dim=-1)

        for l in range(0, self.num_layers_color - 1):
            lin = getattr(self, "clin" + str(l))

            h = lin(h)

            if l < self.num_layers_color - 2:
                h = self.relu(h)

        rgb = self.sigmoid(h)

        if self.config.use_diffuse_color:
            assert raw_rgb_diffuse is not None
            # Initialize linear diffuse color around 0.25, so that the combined
            # linear color is initialized around 0.5.
            diffuse_linear = self.sigmoid(raw_rgb_diffuse - math.log(3.0))
            if self.config.use_specular_tint:
                assert tint is not None
                specular_linear = tint * rgb
            else:
                specular_linear = 0.5 * rgb

            # TODO linear to srgb?
            # Combine specular and diffuse components and tone map to sRGB.
            rgb = torch.clamp(specular_linear + diffuse_linear, 0.0, 1.0)

        # Apply padding, mapping color to [-rgb_padding, 1+rgb_padding].
        rgb = rgb * (1 + 2 * self.config.rgb_padding) - self.config.rgb_padding

        return rgb

    def get_outputs(self, ray_samples: RaySamples, return_alphas=False, return_occupancy=False, render_minimal: bool = False):
        """compute output of ray samples"""
        if ray_samples.camera_indices is None:
            raise AttributeError("Camera indices are not provided.")

        profile_device = ray_samples.frustums.origins.device
        outputs = {}

        camera_indices = ray_samples.camera_indices.squeeze()

        inputs = self._get_query_positions(ray_samples)
        inputs = inputs.view(-1, 3)

        directions = ray_samples.frustums.directions
        directions_flat = directions.reshape(-1, 3)

        if self.spatial_distortion is not None:
            inputs = self.spatial_distortion(inputs)
        points_norm = inputs.norm(dim=-1)
        compute_geometry_gradients = self._should_compute_geometry_gradients(
            render_minimal=render_minimal,
            return_occupancy=return_occupancy,
        )
        if compute_geometry_gradients:
            inputs.requires_grad_(True)
        self._sync_cuda_for_profile(profile_device)
        geonet_start = time.perf_counter()
        grad_context = torch.enable_grad()
        with grad_context:
            h = self.forward_geonetwork(inputs)
            sdf, inv_s, geo_feature, pred_gradient = self.split_geonet_out(h)
            raw_inv_s = inv_s
            inv_s = transform_inv_s(inv_s, mode=self.config.kernel_size_transform_mode, exp_factor=self.config.kernel_size_transform_exp_factor)
        self._sync_cuda_for_profile(profile_device)
        geonet_ms = (time.perf_counter() - geonet_start) * 1000.0
        pred_normal_unit = F.normalize(pred_gradient, dim=-1)

        self._sync_cuda_for_profile(profile_device)
        gradient_start = time.perf_counter()
        use_inference_autograd = (
            compute_geometry_gradients
            and self.config.use_numerical_gradients
            and (not self.training)
            and self.config.inference_use_autograd_gradients
        )
        if not compute_geometry_gradients:
            surface_gradients = None
            sampled_sdf = None
        elif use_inference_autograd:
            surface_gradients = self._gradient_from_sdf(
                sdf=sdf,
                inputs=inputs,
                create_graph=False,
                retain_graph=False,
            )
            sampled_sdf = None
        elif self.config.use_numerical_gradients:
            if render_minimal:
                surface_gradients = self.gradient(inputs, skip_spatial_distortion=True, return_sdf=False)
                sampled_sdf = None
            else:
                surface_gradients, sampled_sdf = self.gradient(
                    inputs,
                    skip_spatial_distortion=True,
                    return_sdf=True,
                )
                if len(ray_samples.frustums.directions.shape[:-1]) >= 2:
                    sampled_sdf = (
                        sampled_sdf.view(-1, *ray_samples.frustums.directions.shape[:-1]).permute(1, 2, 0).contiguous()
                    )
                else:
                    sampled_sdf = sampled_sdf.transpose(0, 1).contiguous()
        else:
            surface_gradients = self._gradient_from_sdf(
                sdf=sdf,
                inputs=inputs,
                create_graph=self.training,
                retain_graph=self.training,
            )
            sampled_sdf = None
        self._sync_cuda_for_profile(profile_device)
        gradient_ms = (time.perf_counter() - gradient_start) * 1000.0
        if surface_gradients is not None:
            surface_gradients = cast(torch.Tensor, surface_gradients)
            single_sample_gradient_stats = self._summarize_single_sample_gradients(
                ray_samples=ray_samples,
                surface_gradients=surface_gradients,
                directions_flat=directions_flat,
            )
        else:
            single_sample_gradient_stats = {
                "single_sample_count": 0,
                "single_sample_grad_norm_sum": 0.0,
                "single_sample_grad_norm_sq_sum": 0.0,
                "single_sample_grad_norm_min": 0.0,
                "single_sample_grad_norm_max": 0.0,
                "single_sample_abs_true_cos_sum": 0.0,
            }

        color_shading_vectors = pred_normal_unit if self._use_predicted_normals_for_color() else cast(Tensor, surface_gradients)

        self._sync_cuda_for_profile(profile_device)
        color_start = time.perf_counter()
        rgb = self.get_colors(
            inputs,
            directions_flat,
            color_shading_vectors,
            geo_feature,
            camera_indices,
        )
        self._sync_cuda_for_profile(profile_device)
        color_ms = (time.perf_counter() - color_start) * 1000.0

        rgb = rgb.view(*ray_samples.frustums.directions.shape[:-1], -1)
        pred_normal = pred_normal_unit.view(*ray_samples.frustums.directions.shape[:-1], -1)
        pred_gradient_for_alpha = pred_gradient.view(*ray_samples.frustums.directions.shape[:-1], -1)
        sdf_for_alpha = sdf.view(*ray_samples.frustums.directions.shape[:-1], -1)
        inv_s_for_alpha = inv_s.view(*ray_samples.frustums.directions.shape[:-1], -1)
        surface_gradients_shaped = (
            surface_gradients.view(*ray_samples.frustums.directions.shape[:-1], -1)
            if surface_gradients is not None
            else None
        )
        alpha_gradients = (
            pred_gradient_for_alpha if self.config.alpha_mode == "pred_gradient" else surface_gradients_shaped
        )
        render_normals = (
            F.normalize(surface_gradients_shaped, p=2, dim=-1)
            if surface_gradients_shaped is not None
            else pred_normal
        )

        alpha_in_field_ms = 0.0
        if render_minimal:
            outputs.update(
                {
                    FieldHeadNames.RGB: rgb,
                    FieldHeadNames.NORMAL: render_normals,
                }
            )
        else:
            density = self.laplace_density(sdf, beta=self._laplace_beta_from_inv_s(inv_s))

            if self.training and compute_geometry_gradients and self.config.kernel_size_smoothness_sigma > 0.0:
                inv_s_deviated = self.get_deviated_inv_s(inputs)
                inv_s_deviated = inv_s_deviated.view(*ray_samples.frustums.directions.shape[:-1], -1)
            else:
                inv_s_deviated = None

            density = density.view(*ray_samples.frustums.directions.shape[:-1], -1)
            points_norm = points_norm.view(*ray_samples.frustums.directions.shape[:-1], -1)

            outputs.update(
                {
                    FieldHeadNames.RGB: rgb,
                    FieldHeadNames.DENSITY: density,
                    FieldHeadNames.SDF: sdf_for_alpha,
                    FieldHeadNames.NORMAL: render_normals,
                    FieldHeadNames.PRED_NORMALS: pred_normal,
                    "points_norm": points_norm,
                    "sampled_sdf": sampled_sdf,
                    "inv_s_deviated": inv_s_deviated,
                    "raw_inv_s": raw_inv_s.view(*ray_samples.frustums.directions.shape[:-1], -1),
                    "inv_s": inv_s_for_alpha,
                    "pred_gradient": pred_gradient_for_alpha,
                }
            )
            if surface_gradients_shaped is not None:
                outputs[FieldHeadNames.GRADIENT] = surface_gradients_shaped

        if return_alphas:
            # TODO use mid point sdf for NeuS
            self._sync_cuda_for_profile(profile_device)
            alpha_in_field_start = time.perf_counter()
            alphas = self.get_alpha(
                ray_samples,
                sdf_for_alpha,
                inv_s_for_alpha,
                alpha_gradients,
                pred_gradient=pred_gradient_for_alpha,
            )
            self._sync_cuda_for_profile(profile_device)
            alpha_in_field_ms = (time.perf_counter() - alpha_in_field_start) * 1000.0
            outputs.update({FieldHeadNames.ALPHA: alphas})

        if return_occupancy:
            occupancy = self.get_occupancy(sdf)
            outputs.update({FieldHeadNames.OCCUPANCY: occupancy})

        self._last_profile = {
            "geonet_ms": geonet_ms,
            "gradient_ms": gradient_ms,
            "color_ms": color_ms,
            "alpha_in_field_ms": alpha_in_field_ms,
            **single_sample_gradient_stats,
        }
        return outputs

    def get_sdf_and_inv_s(self, ray_samples: RaySamples):
        """Predict sdf and transformed inv_s for sampler-side upsampling."""
        positions = self._get_query_positions(ray_samples)
        positions_flat = positions.view(-1, 3)
        if self.spatial_distortion is not None:
            positions_flat = self.spatial_distortion(positions_flat)
        h = self.forward_geonetwork(positions_flat).view(*ray_samples.frustums.shape, -1)
        sdf, inv_s, _, _ = self.split_geonet_out(h)
        inv_s = transform_inv_s(
            inv_s,
            mode=self.config.kernel_size_transform_mode,
            exp_factor=self.config.kernel_size_transform_exp_factor,
        )
        return sdf, inv_s

    @staticmethod
    def _get_query_positions(ray_samples: RaySamples):
        """Use paper-aligned shell tau positions when provided; otherwise fall back to start positions."""
        if ray_samples.metadata is not None:
            shell_eval_positions = ray_samples.metadata.get("shell_eval_positions")
            if shell_eval_positions is not None:
                return shell_eval_positions
        return ray_samples.frustums.get_start_positions()

    def forward(self, ray_samples: RaySamples, return_alphas=False, return_occupancy=False, render_minimal: bool = False):
        """Evaluates the field at points along the ray.

        Args:
            ray_samples: Samples to evaluate field on.
        """
        field_outputs = self.get_outputs(
            ray_samples,
            return_alphas=return_alphas,
            return_occupancy=return_occupancy,
            render_minimal=render_minimal,
        )
        return field_outputs

class AdashellTorchCompatField(AdashellField):
    """Torch MLP backend with topology aligned to TCNN export constraints."""

    def __init__(
        self,
        config: AdashellFieldConfig,
        aabb,
        num_images: int,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        Field.__init__(self)
        self.config = config
        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.num_images = num_images

        if tcnn is None:
            raise ImportError("tinycudann is required for hash encoding in torch_tcnn_compat backend.")
        if self.config.encoding_type != "hash":
            raise ValueError(
                "AdashellTorchCompatField only supports encoding_type='hash' to align with TCNN export."
            )
        if not self.config.torch_tcnn_compat_disable_skip:
            raise ValueError("AdashellTorchCompatField requires torch_tcnn_compat_disable_skip=True.")
        if not self.config.torch_tcnn_compat_disable_weight_norm:
            raise ValueError("AdashellTorchCompatField requires torch_tcnn_compat_disable_weight_norm=True.")
        if not self.config.torch_tcnn_compat_disable_geometric_init:
            raise ValueError("AdashellTorchCompatField requires torch_tcnn_compat_disable_geometric_init=True.")

        self.embedding_appearance = Embedding(self.num_images, self.config.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_grid_feature = self.config.use_grid_feature
        self.divide_factor = self.config.divide_factor

        self.num_levels = self.config.num_levels
        self.max_res = self.config.max_res
        self.base_res = self.config.base_res
        self.log2_hashmap_size = self.config.log2_hashmap_size
        self.features_per_level = self.config.hash_features_per_level
        smoothstep = self.config.hash_smoothstep
        self.growth_factor = np.exp((np.log(self.max_res) - np.log(self.base_res)) / (self.num_levels - 1))
        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.num_levels,
                "n_features_per_level": self.features_per_level,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": self.base_res,
                "per_level_scale": self.growth_factor,
                "interpolation": "Smoothstep" if smoothstep else "Linear",
            },
        )
        self.hash_encoding_mask = torch.ones(
            self.num_levels * self.features_per_level,
            dtype=torch.float32,
        )
        self.direction_encoding = SHEncoding(levels=4)
        self.geo_output_dim = self.config.geo_feat_dim + 1 + 1 + 3
        geo_in_dim = 3 + self.encoding.n_output_dims
        self.geo_network = TorchCompatMLP(
            n_input_dims=geo_in_dim,
            n_output_dims=self.geo_output_dim,
            n_hidden_layers=self.config.num_layers,
            n_neurons=self.config.hidden_dim,
            activation=self.config.tcnn_geo_activation,
            bias=not self.config.torch_tcnn_compat_no_bias,
        )

        if self.config.use_diffuse_color:
            self.diffuse_color_pred = nn.Linear(self.config.geo_feat_dim, 3)
        if self.config.use_specular_tint:
            self.specular_tint_pred = nn.Linear(self.config.geo_feat_dim, 3)

        if self.config.use_diffuse_color:
            color_in_dim = (
                self.direction_encoding.get_out_dim()
                + self.config.geo_feat_dim
                + self.embedding_appearance.get_out_dim()
            )
        else:
            color_in_dim = (
                3
                + self.direction_encoding.get_out_dim()
                + 3
                + self.config.geo_feat_dim
                + self.embedding_appearance.get_out_dim()
            )
        if self.config.use_n_dot_v:
            color_in_dim += 1
        self.color_network = TorchCompatMLP(
            n_input_dims=color_in_dim,
            n_output_dims=3,
            n_hidden_layers=self.config.num_layers_color,
            n_neurons=self.config.hidden_dim_color,
            activation=self.config.tcnn_color_activation,
            bias=not self.config.torch_tcnn_compat_no_bias,
        )
        self.laplace_density = LaplaceDensity(init_val=self.config.beta_init)
        self.softplus = nn.Softplus(beta=100)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self._cos_anneal_ratio = 1.0
        self.numerical_gradients_delta = 0.0001
        self._profile_strict_timing = False
        self._last_profile = None

    def _geo_export_layout(self) -> List[Dict[str, int]]:
        return self.geo_network.get_export_layout(self.config.tcnn_geo_network_otype)

    def _color_export_layout(self) -> List[Dict[str, int]]:
        return self.color_network.get_export_layout(self.config.tcnn_color_network_otype)

    def get_tcnn_compat_export_metadata(self) -> Dict[str, object]:
        """Return config + layout metadata used by future export tooling."""
        return {
            "schema_version": "0.1",
            "producer": "sdfstudio-adashell",
            "field_backend": "torch_tcnn_compat",
            "requires_bias_free": bool(self.config.torch_tcnn_compat_no_bias),
            "geo_config": {
                "otype": self.config.tcnn_geo_network_otype,
                "activation": self.config.tcnn_geo_activation,
                "n_input_dims": self.geo_network.n_input_dims,
                "n_hidden_layers": self.geo_network.n_hidden_layers,
                "n_neurons": self.geo_network.n_neurons,
                "n_output_dims": self.geo_network.n_output_dims,
                "layers": self._geo_export_layout(),
            },
            "color_config": {
                "otype": self.config.tcnn_color_network_otype,
                "activation": self.config.tcnn_color_activation,
                "n_input_dims": self.color_network.n_input_dims,
                "n_hidden_layers": self.color_network.n_hidden_layers,
                "n_neurons": self.color_network.n_neurons,
                "n_output_dims": self.color_network.n_output_dims,
                "layers": self._color_export_layout(),
            },
        }

    def get_tcnn_compat_export_tensors(self) -> Dict[str, Tensor]:
        """Return flattened torch tensors in deterministic layer order."""
        return {
            "geo_params": self.geo_network.flatten_weights(),
            "color_params": self.color_network.flatten_weights(),
        }

    def forward_geonetwork(self, inputs):
        if self.use_grid_feature:
            positions = (inputs + 2.0) / 4.0
            feature = self.encoding(positions)
            feature = feature * self.hash_encoding_mask.to(feature.device)
        else:
            feature = torch.zeros_like(inputs[:, :1].repeat(1, self.encoding.n_output_dims))
        geo_inputs = torch.cat((inputs, feature), dim=-1).to(torch.float32).contiguous()
        return self.geo_network(geo_inputs).to(inputs)

    def get_colors(
        self,
        points,
        directions,
        shading_vectors,
        geo_features,
        camera_indices,
    ):
        raw_rgb_diffuse: Optional[Tensor] = None
        tint: Optional[Tensor] = None
        if self.config.use_diffuse_color:
            raw_rgb_diffuse = self.diffuse_color_pred(geo_features.view(-1, self.config.geo_feat_dim))
        if self.config.use_specular_tint:
            tint = self.sigmoid(self.specular_tint_pred(geo_features.view(-1, self.config.geo_feat_dim)))

        normals = F.normalize(shading_vectors, p=2, dim=-1)
        if self.config.use_reflections:
            refdirs = 2.0 * torch.sum(normals * -directions, axis=-1, keepdims=True) * normals + directions
            d = self.direction_encoding(refdirs)
        else:
            d = self.direction_encoding(directions)

        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
            if not self.config.use_appearance_embedding:
                embedded_appearance = torch.zeros_like(embedded_appearance)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.config.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.config.appearance_embedding_dim), device=directions.device
                )

        if self.config.use_diffuse_color:
            h = [
                d,
                geo_features.view(-1, self.config.geo_feat_dim),
                embedded_appearance.view(-1, self.config.appearance_embedding_dim),
            ]
        else:
            h = [
                points,
                d,
                shading_vectors,
                geo_features.view(-1, self.config.geo_feat_dim),
                embedded_appearance.view(-1, self.config.appearance_embedding_dim),
            ]
        if self.config.use_n_dot_v:
            n_dot_v = torch.sum(normals * directions, dim=-1, keepdims=True)
            h.append(n_dot_v)
        h = torch.cat(h, dim=-1).to(torch.float32).contiguous()

        rgb = self.sigmoid(self.color_network(h).to(directions))
        if self.config.use_diffuse_color:
            assert raw_rgb_diffuse is not None
            diffuse_linear = self.sigmoid(raw_rgb_diffuse - math.log(3.0))
            if self.config.use_specular_tint:
                assert tint is not None
                specular_linear = tint * rgb
            else:
                specular_linear = 0.5 * rgb
            rgb = torch.clamp(specular_linear + diffuse_linear, 0.0, 1.0)
        rgb = rgb * (1 + 2 * self.config.rgb_padding) - self.config.rgb_padding
        return rgb


class AdashellTCNNField(AdashellField):
    """TCNN-backed Adashell field for hash-encoding training from scratch."""

    def __init__(
        self,
        config: AdashellFieldConfig,
        aabb,
        num_images: int,
        use_average_appearance_embedding: bool = False,
        spatial_distortion: Optional[SpatialDistortion] = None,
    ) -> None:
        Field.__init__(self)
        self.config = config
        self.aabb = Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.num_images = num_images

        if tcnn is None:
            raise ImportError("tinycudann is required for field_backend='tcnn'.")
        if self.config.encoding_type != "hash":
            raise ValueError(
                f"AdashellTCNNField only supports encoding_type='hash' in v1, got '{self.config.encoding_type}'."
            )

        self.embedding_appearance = Embedding(self.num_images, self.config.appearance_embedding_dim)
        self.use_average_appearance_embedding = use_average_appearance_embedding
        self.use_grid_feature = self.config.use_grid_feature
        self.divide_factor = self.config.divide_factor

        self.num_levels = self.config.num_levels
        self.max_res = self.config.max_res
        self.base_res = self.config.base_res
        self.log2_hashmap_size = self.config.log2_hashmap_size
        self.features_per_level = self.config.hash_features_per_level
        smoothstep = self.config.hash_smoothstep
        self.growth_factor = np.exp((np.log(self.max_res) - np.log(self.base_res)) / (self.num_levels - 1))

        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": self.num_levels,
                "n_features_per_level": self.features_per_level,
                "log2_hashmap_size": self.log2_hashmap_size,
                "base_resolution": self.base_res,
                "per_level_scale": self.growth_factor,
                "interpolation": "Smoothstep" if smoothstep else "Linear",
            },
        )
        self.hash_encoding_mask = torch.ones(
            self.num_levels * self.features_per_level,
            dtype=torch.float32,
        )

        self.direction_encoding = SHEncoding(levels=4)
        self.geo_output_dim = self.config.geo_feat_dim + 1 + 1 + 3
        geo_in_dim = 3 + self.encoding.n_output_dims
        self.geo_network = self._build_tcnn_network(
            n_input_dims=geo_in_dim,
            n_output_dims=self.geo_output_dim,
            n_hidden_layers=self.config.num_layers,
            n_neurons=self.config.hidden_dim,
            preferred_otype=self.config.tcnn_geo_network_otype,
            activation=self.config.tcnn_geo_activation,
            output_activation="None",
        )

        if self.config.use_diffuse_color:
            self.diffuse_color_pred = nn.Linear(self.config.geo_feat_dim, 3)
        if self.config.use_specular_tint:
            self.specular_tint_pred = nn.Linear(self.config.geo_feat_dim, 3)

        if self.config.use_diffuse_color:
            color_in_dim = (
                self.direction_encoding.get_out_dim()
                + self.config.geo_feat_dim
                + self.embedding_appearance.get_out_dim()
            )
        else:
            color_in_dim = (
                3
                + self.direction_encoding.get_out_dim()
                + 3
                + self.config.geo_feat_dim
                + self.embedding_appearance.get_out_dim()
            )
        if self.config.use_n_dot_v:
            color_in_dim += 1

        self.color_network = self._build_tcnn_network(
            n_input_dims=color_in_dim,
            n_output_dims=3,
            n_hidden_layers=self.config.num_layers_color,
            n_neurons=self.config.hidden_dim_color,
            preferred_otype=self.config.tcnn_color_network_otype,
            activation=self.config.tcnn_color_activation,
            output_activation="None",
        )

        self.laplace_density = LaplaceDensity(init_val=self.config.beta_init)
        self.softplus = nn.Softplus(beta=100)
        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self._cos_anneal_ratio = 1.0
        self.numerical_gradients_delta = 0.0001
        self._profile_strict_timing = False
        self._last_profile = None

    def get_tcnn_export_metadata(self) -> Dict[str, object]:
        """Return TCNN network metadata with explicit padding policy."""
        geo_in_dim = 3 + self.encoding.n_output_dims
        if self.config.use_diffuse_color:
            color_in_dim = (
                self.direction_encoding.get_out_dim()
                + self.config.geo_feat_dim
                + self.embedding_appearance.get_out_dim()
            )
        else:
            color_in_dim = (
                3
                + self.direction_encoding.get_out_dim()
                + 3
                + self.config.geo_feat_dim
                + self.embedding_appearance.get_out_dim()
            )
        if self.config.use_n_dot_v:
            color_in_dim += 1
        return {
            "schema_version": "0.1",
            "producer": "sdfstudio-adashell",
            "field_backend": "tcnn",
            "geo_config": {
                "otype": self.config.tcnn_geo_network_otype,
                "activation": self.config.tcnn_geo_activation,
                "n_input_dims": int(geo_in_dim),
                "n_hidden_layers": int(self.config.num_layers),
                "n_neurons": int(self.config.hidden_dim),
                "n_output_dims": int(self.geo_output_dim),
                "layers": mlp_export_layout(
                    n_input_dims=int(geo_in_dim),
                    n_hidden_layers=int(self.config.num_layers),
                    n_neurons=int(self.config.hidden_dim),
                    n_output_dims=int(self.geo_output_dim),
                    target_otype=self.config.tcnn_geo_network_otype,
                ),
            },
            "color_config": {
                "otype": self.config.tcnn_color_network_otype,
                "activation": self.config.tcnn_color_activation,
                "n_input_dims": int(color_in_dim),
                "n_hidden_layers": int(self.config.num_layers_color),
                "n_neurons": int(self.config.hidden_dim_color),
                "n_output_dims": 3,
                "layers": mlp_export_layout(
                    n_input_dims=int(color_in_dim),
                    n_hidden_layers=int(self.config.num_layers_color),
                    n_neurons=int(self.config.hidden_dim_color),
                    n_output_dims=3,
                    target_otype=self.config.tcnn_color_network_otype,
                ),
            },
        }

    @staticmethod
    def _build_tcnn_network(
        n_input_dims: int,
        n_output_dims: int,
        n_hidden_layers: int,
        n_neurons: int,
        preferred_otype: str,
        activation: str,
        output_activation: str,
    ):
        config = {
            "otype": preferred_otype,
            "activation": activation,
            "output_activation": output_activation,
            "n_neurons": n_neurons,
            "n_hidden_layers": n_hidden_layers,
        }
        try:
            return tcnn.Network(
                n_input_dims=n_input_dims,
                n_output_dims=n_output_dims,
                network_config=config,
            )
        except RuntimeError:
            if preferred_otype == "CutlassMLP":
                raise
            fallback = dict(config)
            fallback["otype"] = "CutlassMLP"
            return tcnn.Network(
                n_input_dims=n_input_dims,
                n_output_dims=n_output_dims,
                network_config=fallback,
            )

    def forward_geonetwork(self, inputs):
        if self.use_grid_feature:
            positions = (inputs + 2.0) / 4.0
            feature = self.encoding(positions)
            feature = feature * self.hash_encoding_mask.to(feature.device)
        else:
            feature = torch.zeros_like(inputs[:, :1].repeat(1, self.encoding.n_output_dims))
        geo_inputs = torch.cat((inputs, feature), dim=-1).to(torch.float32).contiguous()
        return self.geo_network(geo_inputs).to(inputs)

    def get_colors(
        self,
        points,
        directions,
        shading_vectors,
        geo_features,
        camera_indices,
    ):
        raw_rgb_diffuse: Optional[Tensor] = None
        tint: Optional[Tensor] = None
        if self.config.use_diffuse_color:
            raw_rgb_diffuse = self.diffuse_color_pred(geo_features.view(-1, self.config.geo_feat_dim))
        if self.config.use_specular_tint:
            tint = self.sigmoid(self.specular_tint_pred(geo_features.view(-1, self.config.geo_feat_dim)))

        normals = F.normalize(shading_vectors, p=2, dim=-1)
        if self.config.use_reflections:
            refdirs = 2.0 * torch.sum(normals * -directions, axis=-1, keepdims=True) * normals + directions
            d = self.direction_encoding(refdirs)
        else:
            d = self.direction_encoding(directions)

        if self.training:
            embedded_appearance = self.embedding_appearance(camera_indices)
            if not self.config.use_appearance_embedding:
                embedded_appearance = torch.zeros_like(embedded_appearance)
        else:
            if self.use_average_appearance_embedding:
                embedded_appearance = torch.ones(
                    (*directions.shape[:-1], self.config.appearance_embedding_dim), device=directions.device
                ) * self.embedding_appearance.mean(dim=0)
            else:
                embedded_appearance = torch.zeros(
                    (*directions.shape[:-1], self.config.appearance_embedding_dim), device=directions.device
                )

        if self.config.use_diffuse_color:
            h = [
                d,
                geo_features.view(-1, self.config.geo_feat_dim),
                embedded_appearance.view(-1, self.config.appearance_embedding_dim),
            ]
        else:
            h = [
                points,
                d,
                shading_vectors,
                geo_features.view(-1, self.config.geo_feat_dim),
                embedded_appearance.view(-1, self.config.appearance_embedding_dim),
            ]
        if self.config.use_n_dot_v:
            n_dot_v = torch.sum(normals * directions, dim=-1, keepdims=True)
            h.append(n_dot_v)
        h = torch.cat(h, dim=-1).to(torch.float32).contiguous()

        rgb = self.sigmoid(self.color_network(h).to(directions))
        if self.config.use_diffuse_color:
            assert raw_rgb_diffuse is not None
            diffuse_linear = self.sigmoid(raw_rgb_diffuse - math.log(3.0))
            if self.config.use_specular_tint:
                assert tint is not None
                specular_linear = tint * rgb
            else:
                specular_linear = 0.5 * rgb
            rgb = torch.clamp(specular_linear + diffuse_linear, 0.0, 1.0)
        rgb = rgb * (1 + 2 * self.config.rgb_padding) - self.config.rgb_padding
        return rgb
