"""Run a child command with MSVC/CUDA initialized, without changing a profile."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def compiler_environment(cuda: Path | None = None, vcvars: Path | None = None) -> dict[str, str]:
    if os.name != "nt":
        raise RuntimeError("This release profile requires Windows x64")
    if vcvars is None:
        vswhere = (
            Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)"))
            / "Microsoft Visual Studio/Installer/vswhere.exe"
        )
        installation = subprocess.check_output(
            [
                str(vswhere),
                "-latest",
                "-products",
                "*",
                "-requires",
                "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property",
                "installationPath",
            ],
            text=True,
        ).strip()
        vcvars = Path(installation) / "VC/Auxiliary/Build/vcvars64.bat"
    if not vcvars.is_file():
        raise FileNotFoundError(vcvars)
    # Capture the batch file's environment in a child cmd; never edit the shell profile.
    output = subprocess.check_output(
        f'cmd.exe /d /s /c "call "{vcvars}" >nul && set"',
        text=True,
    )
    env = {key.upper(): value for key, value in os.environ.items()}
    for line in output.splitlines():
        key, sep, value = line.partition("=")
        if sep and key:
            env[key.upper()] = value
    cuda = cuda or Path(env.get("CUDA_PATH", ""))
    if not (cuda / "bin/nvcc.exe").is_file():
        raise FileNotFoundError(f"CUDA toolkit not found: {cuda}")
    env.update(
        CUDA_PATH=str(cuda),
        CUDA_HOME=str(cuda),
        DISTUTILS_USE_SDK="1",
        TCNN_CUDA_ARCHITECTURES="120",
        TORCH_CUDA_ARCH_LIST="12.0",
    )
    env["PATH"] = str(cuda / "bin") + os.pathsep + env["PATH"]
    for tool in ("cl.exe", "nvcc.exe"):
        if not shutil.which(tool, path=env["PATH"]):
            raise RuntimeError(f"{tool} unavailable after vcvars64.bat")
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env.pop("VIRTUAL_ENV", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", type=Path)
    parser.add_argument("--vcvars", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    env = compiler_environment(args.cuda, args.vcvars)
    if args.cache_root:
        cache = args.cache_root.resolve()
        env["TORCH_EXTENSIONS_DIR"] = str(cache / "torch")
        env["CUPY_CACHE_DIR"] = str(cache / "cupy")
    if not command:
        for name in ("cl.exe", "nvcc.exe"):
            print(shutil.which(name, path=env["PATH"]))
        return
    executable = Path(command[0])
    if executable.is_file():
        env["PATH"] = str(executable.resolve().parent) + os.pathsep + env["PATH"]
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()
