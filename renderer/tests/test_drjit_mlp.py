from __future__ import annotations

import subprocess
import sys
from pathlib import Path
import textwrap

import pytest
import torch


def test_drjit_mlp_subprocess_parity() -> None:
    pytest.importorskip("drjit")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the Dr.Jit CUDA MLP tests.")

    script = textwrap.dedent(
        r"""
        import torch
        from torch import nn

        from adashell_renderer.drjit_backend.mlp import DrJitMLP

        def torch_activation(name: str, x: torch.Tensor) -> torch.Tensor:
            if name == "none":
                return x
            if name == "relu":
                return torch.relu(x)
            if name == "sigmoid":
                return torch.sigmoid(x)
            if name == "squareplus":
                return 0.5 * (x + torch.sqrt(x * x + 4.0))
            if name == "sine":
                return torch.sin(x)
            if name == "tanh":
                return torch.tanh(x)
            if name == "exp":
                return torch.exp(x)
            if name == "softplus":
                return torch.nn.functional.softplus(x, beta=100.0)
            raise ValueError(name)

        def half_reference(x: torch.Tensor, layers: list[nn.Linear], hidden_activation: str, output_activation: str = "none") -> torch.Tensor:
            y = x.to(dtype=torch.float16)
            for idx, layer in enumerate(layers):
                weight = layer.weight.detach().to(device=x.device, dtype=torch.float16)
                bias = None if layer.bias is None else layer.bias.detach().to(device=x.device, dtype=torch.float16)
                y = y @ weight.t()
                if bias is not None:
                    y = y + bias
                is_last = idx == len(layers) - 1
                y = torch_activation(output_activation if is_last else hidden_activation, y)
            return y.to(dtype=torch.float32)

        device = torch.device("cuda")
        hidden_activations = ["relu", "softplus", "squareplus", "sine", "tanh"]
        output_activations = ["sigmoid", "tanh", "exp", "none"]

        for use_bias in [False, True]:
            for hidden_activation in hidden_activations:
                torch.manual_seed(7)
                layer0 = nn.utils.weight_norm(nn.Linear(5, 7, bias=use_bias)).to(device)
                layer1 = nn.utils.weight_norm(nn.Linear(7, 3, bias=use_bias)).to(device)
                x = torch.randn((11, 5), device=device, dtype=torch.float32)
                y_ref = half_reference(x, [layer0, layer1], hidden_activation=hidden_activation)
                mlp = DrJitMLP.from_torch_linears([layer0, layer1], hidden_activation=hidden_activation)
                y = mlp.forward_torch(x)
                if not torch.allclose(y, y_ref, atol=5e-3, rtol=5e-3):
                    raise SystemExit(f"hidden activation mismatch: use_bias={use_bias}, act={hidden_activation}")

        for output_activation in output_activations:
            torch.manual_seed(11)
            layer0 = nn.Linear(4, 6, bias=False).to(device)
            layer1 = nn.Linear(6, 3, bias=False).to(device)
            x = torch.randn((13, 4), device=device, dtype=torch.float32)
            y_ref = half_reference(x, [layer0, layer1], hidden_activation="relu", output_activation=output_activation)
            mlp = DrJitMLP.from_torch_linears([layer0, layer1], hidden_activation="relu", output_activation=output_activation)
            y = mlp.forward_torch(x)
            if not torch.allclose(y, y_ref, atol=5e-3, rtol=5e-3):
                raise SystemExit(f"output activation mismatch: act={output_activation}")
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
