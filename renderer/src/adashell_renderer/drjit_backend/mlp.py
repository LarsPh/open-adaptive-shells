from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
import torch
from torch import nn

from adashell_renderer.drjit_backend.interop import DrJitInterop

ActivationName = Literal["none", "relu", "softplus", "sigmoid", "squareplus", "sine", "tanh", "exp"]


@dataclass(slots=True)
class TorchLinearLayerSpec:
    """Materialized affine layer parameters extracted from a torch module."""

    weight: torch.Tensor
    bias: torch.Tensor | None

    @classmethod
    def from_torch_linear(cls, layer: nn.Linear) -> "TorchLinearLayerSpec":
        # Important subtlety:
        # the legacy field uses the older `nn.utils.weight_norm()` wrapper.
        # After checkpoint loading, directly reading `layer.weight` here can
        # observe a stale tensor before the forward pre-hook has recomputed the
        # effective normalized weight. To avoid that ambiguity, explicitly
        # reconstruct the affine weight from `weight_v` / `weight_g` when those
        # parameters are present.
        weight_v = getattr(layer, "weight_v", None)
        weight_g = getattr(layer, "weight_g", None)
        if isinstance(weight_v, torch.Tensor) and isinstance(weight_g, torch.Tensor):
            dim = 0
            for hook in layer._forward_pre_hooks.values():
                if hook.__class__.__name__ == "WeightNorm" and getattr(hook, "name", None) == "weight":
                    dim = int(getattr(hook, "dim", 0))
                    break
            v = weight_v.detach().to(dtype=torch.float32)
            g = weight_g.detach().to(dtype=torch.float32)
            # Match PyTorch's internal weight-norm reparameterization exactly.
            # Using the builtin avoids subtle mistakes about what `dim` means
            # for higher-rank tensors and keeps parity with the wrapped layer.
            weight = torch._weight_norm(v, g, dim).contiguous()
        else:
            weight = layer.weight.detach().to(dtype=torch.float32).contiguous()
        bias = None if layer.bias is None else layer.bias.detach().to(dtype=torch.float32).contiguous()
        return cls(weight=weight, bias=bias)


@dataclass(slots=True)
class DrJitMLP:
    """Packed Dr.Jit MLP for batched inference with torch-authored weights.

    This class now uses Dr.Jit's neural-network fast path
    fast path (`nn.pack()` + `nn.matvec()` + `nn.CoopVec`) because the local
    Dr.Jit 1.3.x runtime does not yet expose the newer dynamic tensor GEMM path
    that would make plain `TensorXf @ TensorXf` a good fit here.

    Practical mapping rule:
    - torch linear layer `(out, in)` + optional bias
    - -> one-time Dr.Jit `nn.pack(weight, bias, layout='inference')`
    - -> repeated runtime `nn.matvec()` on a cooperative vector batch

    This already puts the *heavy* part of the MLP, the matrix-vector products,
    onto the intended fast path. Custom activations are then applied between
    layers. When an activation is unsupported on `CoopVec`, we unpack its lanes,
    apply regular Dr.Jit math, and repack. That's not as ideal as a fully native
    fused activation, but it is a good first step and keeps the implementation
    faithful to the torch feature set.
    """

    layers: tuple[TorchLinearLayerSpec, ...]
    hidden_activation: ActivationName
    output_activation: ActivationName = "none"
    unsafe_direct_torch_bridge: bool = False
    mode: str = "fp16_fast"
    interop: DrJitInterop = field(init=False, repr=False)
    _cached_dr_layers: dict[tuple[str, torch.device], tuple[tuple[object, object, object | None], ...]] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        self.interop = DrJitInterop.create()
        self._cached_dr_layers = {}

    @classmethod
    def from_torch_linears(
        cls,
        layers: Sequence[nn.Linear],
        hidden_activation: ActivationName,
        output_activation: ActivationName = "none",
        unsafe_direct_torch_bridge: bool = False,
        mode: str = "fp16_fast",
    ) -> "DrJitMLP":
        return cls(
            layers=tuple(TorchLinearLayerSpec.from_torch_linear(layer) for layer in layers),
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            unsafe_direct_torch_bridge=unsafe_direct_torch_bridge,
            mode=mode,
        )

    def _dr_layers(self, device: torch.device) -> tuple[tuple[object, object, object | None], ...]:
        dr = self.interop.dr
        key = (str(device), device)
        cached = self._cached_dr_layers.get(key)
        if cached is not None:
            return cached

        packed_layers: list[tuple[object, object, object | None]] = []
        for spec in self.layers:
            # Current Dr.Jit cooperative-vector packing on CUDA expects half
            # precision buffers. We therefore cast the torch-authored weights
            # into fp16 during the one-time pack step, not every forward pass.
            weight_t = spec.weight.to(device=device, dtype=torch.float16).contiguous()
            weight_dr = dr.cuda.TensorXf16(weight_t)
            if spec.bias is None:
                buffer_dr, weight_view = dr.nn.pack(weight_dr, layout="inference")
                packed_layers.append((buffer_dr, weight_view, None))
            else:
                bias_t = spec.bias.to(device=device, dtype=torch.float16).contiguous()
                bias_dr = dr.cuda.TensorXf16(bias_t)
                buffer_dr, weight_view, bias_view = dr.nn.pack(weight_dr, bias_dr, layout="inference")
                packed_layers.append((buffer_dr, weight_view, bias_view))
        result = tuple(packed_layers)
        self._cached_dr_layers[key] = result
        return result

    def _torch_to_coopvec(self, x: torch.Tensor):
        dr = self.interop.dr
        x_half = x.to(dtype=torch.float16).contiguous()
        # CoopVec stores one feature per component and one sample per lane.
        # So an `(N, D)` torch matrix becomes `D` Dr.Jit arrays of width `N`.
        cols = [dr.cuda.Float16(x_half[:, idx].contiguous()) for idx in range(x_half.shape[1])]
        return dr.nn.CoopVec(*cols)

    def _is_coopvec(self, value: object) -> bool:
        return value.__class__.__name__ == "CoopVec"

    def _coopvec_columns(self, vec) -> tuple[object, ...]:
        # `CoopVec` is iterable even though it is not subscriptable. Iteration
        # yields one lane-packed column per feature channel.
        return tuple(vec)

    def _is_column_bundle(self, value: object) -> bool:
        if not isinstance(value, tuple) or not value:
            return False
        return all(getattr(v.__class__, "__module__", "").startswith("drjit") for v in value)

    def tensor_columns(self, tensor):
        # Downstream stages mostly want "one Dr.Jit flat array per feature
        # channel". Support both:
        # - true Dr.Jit tensors
        # - the opaque CoopVec representation we now keep internally
        if self._is_coopvec(tensor):
            return tuple(self.interop.dr.cuda.Float(col) for col in self._coopvec_columns(tensor))
        if self._is_column_bundle(tensor):
            return tuple(self.interop.dr.cuda.Float(col) for col in tensor)
        tensor_fp32 = self.interop.dr.cuda.TensorXf(tensor)
        return tuple(self.interop.dr.cuda.Float(tensor_fp32[:, idx]) for idx in range(tensor_fp32.shape[1]))

    def _tensor_to_torch(self, tensor, device: torch.device) -> torch.Tensor:
        # This is a deliberately conservative bridge at true island exits.
        #
        # Fast-but-risky path:
        # - keep data on GPU with direct Dr.Jit -> torch bridging
        #
        # Safe path:
        # - materialize per-column NumPy arrays and rebuild the batch tensor
        #
        # We intentionally support CoopVec here because `nn.matvec()` returns
        # that opaque layout directly, and converting it through
        # `TensorXf(vec, flip_axes=True)` proved unstable on the local runtime.
        #
        # Once the MLP path is fully integrated, downstream stages should keep
        # consuming Dr.Jit columns directly so this bridge becomes rare.
        if self._is_coopvec(tensor):
            cols = self._coopvec_columns(tensor)
            if self.unsafe_direct_torch_bridge:
                return torch.stack([col.torch().to(dtype=torch.float32) for col in cols], dim=-1).to(device=device)
            array = np.stack([np.array(col.numpy(), copy=True) for col in cols], axis=-1)
            return torch.from_numpy(array).to(device=device, dtype=torch.float32)
        if self._is_column_bundle(tensor):
            if self.unsafe_direct_torch_bridge:
                return torch.stack([col.torch().to(dtype=torch.float32) for col in tensor], dim=-1).to(device=device)
            array = np.stack([np.array(col.numpy(), copy=True) for col in tensor], axis=-1)
            return torch.from_numpy(array).to(device=device, dtype=torch.float32)
        if self.unsafe_direct_torch_bridge:
            return tensor.torch().to(device=device, dtype=torch.float32)
        array = np.array(tensor.numpy(), copy=True)
        return torch.from_numpy(array).to(device=device, dtype=torch.float32)

    def tensor_to_torch(self, tensor, device: torch.device) -> torch.Tensor:
        return self._tensor_to_torch(tensor, device)

    def affine(self, value, scale: float = 1.0, bias: float = 0.0):
        """Apply a scalar affine transform to any supported Dr.Jit MLP output."""
        if self._is_coopvec(value):
            return value * scale + bias
        if self._is_column_bundle(value):
            return tuple(col * scale + bias for col in value)
        return value * scale + bias

    def _apply_activation_scalar(self, x, activation: ActivationName):
        dr = self.interop.dr
        if activation == "none":
            return x
        if activation == "relu":
            return dr.maximum(x, 0.0)
        if activation == "tanh":
            return dr.tanh(x)
        if activation == "sigmoid":
            return 1.0 / (1.0 + dr.exp(-x))
        if activation == "squareplus":
            return 0.5 * (x + dr.sqrt(x * x + 4.0))
        if activation == "softplus":
            beta = 100.0
            scaled = x * beta
            return (dr.maximum(scaled, 0.0) + dr.log(1.0 + dr.exp(-dr.abs(scaled)))) / beta
        if activation == "sine":
            return dr.sin(x)
        if activation == "exp":
            return dr.exp(x)
        raise ValueError(f"Unsupported Dr.Jit activation: {activation}")

    def _apply_activation(self, vec, activation: ActivationName):
        # Keep activations in the opaque CoopVec layout instead of detouring
        # through `TensorXf(vec, flip_axes=True)`. The latter path turned out
        # to be fragile for `nn.matvec()` outputs on the local Dr.Jit runtime.
        #
        # Each component of the CoopVec is one feature channel stored as a flat
        # Dr.Jit array over all samples. Elementwise activations therefore map
        # naturally to "apply the scalar activation to every column".
        #
        # We intentionally cast the activated values back to `Float16` because
        # CUDA cooperative-vector `nn.matvec()` currently requires half inputs.
        dr = self.interop.dr
        activated_cols = [
            dr.cuda.Float16(self._apply_activation_scalar(dr.cuda.Float(col), activation))
            for col in self._coopvec_columns(vec)
        ]
        return dr.nn.CoopVec(*activated_cols)

    def _torch_to_columns_fp32(self, x: torch.Tensor) -> tuple[object, ...]:
        dr = self.interop.dr
        x_fp32 = x.to(dtype=torch.float32).contiguous()
        return tuple(dr.cuda.Float(x_fp32[:, idx].contiguous()) for idx in range(x_fp32.shape[1]))

    def _column_bundle_to_coopvec_fp16(self, cols: tuple[object, ...]):
        dr = self.interop.dr
        return dr.nn.CoopVec(*(dr.cuda.Float16(col) for col in cols))

    def _apply_activation_columns_fp32(self, cols: tuple[object, ...], activation: ActivationName) -> tuple[object, ...]:
        return tuple(self._apply_activation_scalar(col, activation) for col in cols)

    def _linear_columns_fp32(self, cols: tuple[object, ...], spec: TorchLinearLayerSpec) -> tuple[object, ...]:
        dr = self.interop.dr
        weight = spec.weight
        bias = spec.bias
        if len(cols) != weight.shape[1] or not cols:
            raise ValueError(f"Expected {weight.shape[1]} non-empty input columns, got {len(cols)}.")
        widths = tuple(dr.width(col) for col in cols)
        width = max(widths)
        if any(size not in (1, width) for size in widths):
            raise ValueError(f"MLP input columns must have compatible sample widths, got {widths}.")
        out_cols: list[object] = []
        for out_idx in range(weight.shape[0]):
            acc = dr.zeros(dr.cuda.Float, shape=width)
            if bias is not None:
                acc = acc + float(bias[out_idx].item())
            for in_idx, col in enumerate(cols):
                acc = dr.fma(col, float(weight[out_idx, in_idx].item()), acc)
            out_cols.append(acc)
        return tuple(out_cols)

    def _forward_dr_fp32_debug(self, x: torch.Tensor):
        cols = self._torch_to_columns_fp32(x)
        for idx, spec in enumerate(self.layers):
            cols = self._linear_columns_fp32(cols, spec)
            is_last = idx == len(self.layers) - 1
            cols = self._apply_activation_columns_fp32(cols, self.output_activation if is_last else self.hidden_activation)
        return cols

    def forward_dr(self, x, profile: dict[str, float] | None = None, profile_prefix: str = "mlp"):
        if self.mode == "fp32_debug":
            if isinstance(x, torch.Tensor):
                return self._forward_dr_fp32_debug(x)
            if self._is_column_bundle(x):
                cols = x
                for idx, spec in enumerate(self.layers):
                    cols = self._linear_columns_fp32(cols, spec)
                    is_last = idx == len(self.layers) - 1
                    cols = self._apply_activation_columns_fp32(cols, self.output_activation if is_last else self.hidden_activation)
                return cols
            raise TypeError(f"Unsupported DrJitMLP fp32_debug input type: {type(x)!r}")
        if self.mode != "fp16_fast":
            raise ValueError(f"Unsupported DrJitMLP mode: {self.mode}")

        dr = self.interop.dr
        to_coopvec_start = time.perf_counter()
        if isinstance(x, torch.Tensor):
            device = x.device
            vec = self._torch_to_coopvec(x)
        elif self._is_column_bundle(x):
            device = torch.device("cuda")
            vec = self._column_bundle_to_coopvec_fp16(x)
        else:
            raise TypeError(f"Unsupported DrJitMLP fp16_fast input type: {type(x)!r}")
        if profile is not None:
            profile[f"{profile_prefix}_to_coopvec_ms"] = (time.perf_counter() - to_coopvec_start) * 1000.0

        cache_key = (str(device), device)
        cache_hit = cache_key in self._cached_dr_layers
        layers_start = time.perf_counter()
        activation_ms = 0.0
        for idx, (_buffer_dr, weight_view, bias_view) in enumerate(self._dr_layers(device)):
            vec = dr.nn.matvec(weight_view, vec, bias_view)
            is_last = idx == len(self.layers) - 1
            activation_start = time.perf_counter()
            vec = self._apply_activation(vec, self.output_activation if is_last else self.hidden_activation)
            activation_ms += (time.perf_counter() - activation_start) * 1000.0
        if profile is not None:
            profile[f"{profile_prefix}_pack_cache_hit"] = float(1.0 if cache_hit else 0.0)
            profile[f"{profile_prefix}_layers_ms"] = (time.perf_counter() - layers_start) * 1000.0
            profile[f"{profile_prefix}_activation_ms"] = activation_ms

        # Return the opaque Dr.Jit representation directly so downstream stages
        # can keep the island alive. Callers that truly need torch should use
        # `tensor_to_torch()` at the actual stage boundary.
        return vec

    def forward_torch(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        tensor = self.forward_dr(x)
        return self._tensor_to_torch(tensor, device)
