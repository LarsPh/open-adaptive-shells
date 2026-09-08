from __future__ import annotations

import importlib
import json
import platform
from typing import Any, Dict


def collect_env_report() -> Dict[str, Any]:
    report: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }

    torch = importlib.import_module("torch")
    report["torch_version"] = getattr(torch, "__version__", "unknown")
    report["torch_cuda"] = getattr(torch.version, "cuda", None)
    report["cuda_available"] = bool(torch.cuda.is_available())

    try:
        drjit = importlib.import_module("drjit")
    except ModuleNotFoundError:
        report["drjit_installed"] = False
    else:
        report["drjit_installed"] = True
        report["drjit_version"] = getattr(drjit, "__version__", "unknown")

    try:
        tcnn = importlib.import_module("tinycudann")
    except ModuleNotFoundError:
        report["tinycudann_installed"] = False
        return report

    report["tinycudann_installed"] = True
    report["tinycudann_version"] = getattr(tcnn, "__version__", "unknown")
    report["tinycudann_supports_jit_fusion"] = bool(getattr(tcnn, "supports_jit_fusion")())
    return report


def main() -> None:
    print(json.dumps(collect_env_report(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
