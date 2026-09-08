# Windows Installation

This guide builds and installs AdaShell Renderer from wheels on Windows.
Run the commands from `renderer/` in Nushell or PowerShell.

## Environment

See the shared [installation guide](../../docs/installation.md) for versions
and native source setup. The current tinycudann wheel targets `sm_120`.

## Driver Compatibility

See the [driver note](../../docs/installation.md#windows-environment).

## Requirements

- uv and CPython 3.10.17; uv can provision Python.
- CUDA Toolkit 12.8 and a compatible NVIDIA driver.
- Visual Studio 2022 C++ tools and Windows SDK.
- OptiX 7.6.0 SDK when building python-optix.
- Dependency source snapshots or the wheel files listed below.

The viewer compiles its CUDA/OpenGL bridge on first use, so CUDA Toolkit, MSVC,
and Ninja are required even when the Python packages are installed from wheels.

## Compiler Environment

`scripts/windows_env.py` runs `vcvars64.bat` in a child process, checks `cl.exe`
and `nvcc.exe`, and passes the environment to a command. Environment changes
are limited to that process and its children.

~~~text
uv run --no-project --python 3.10.17 scripts/windows_env.py --vcvars 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat'
~~~

Adjust `--vcvars` for your Visual Studio installation. Without it, the wrapper
uses vswhere to locate the latest C++ installation. `--cuda` selects a toolkit;
otherwise the wrapper uses `CUDA_PATH`. It sets `TCNN_CUDA_ARCHITECTURES=120`,
`TORCH_CUDA_ARCH_LIST=12.0`, and `DISTUTILS_USE_SDK=1` for this build profile.

## Dependency Wheels

The project uses these relative wheel paths:

~~~text
dist/wheels/python_optix-1.0.0-cp310-cp310-win_amd64.whl
dist/wheels/tinycudann-2.0-cp310-cp310-win_amd64.whl
~~~

Place matching wheels there, or build them from the sources described in
[Dependency Provenance](dependencies.md). The following example assumes the
dependency checkouts are siblings of this repository; replace those two paths
with your checkout locations.

~~~text
uv run --no-project --python 3.10.17 scripts/prepare_local_release.py snapshot --python-optix ../python-optix --tiny-cuda-nn ../tiny-cuda-nn
uv run --no-project --python 3.10.17 scripts/prepare_local_release.py build --optix-sdk 'C:/Program Files/NVIDIA Corporation/OptiX 7.6.0' --vcvars 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat'
uv lock
~~~

Run the commands in order. Snapshotting copies submodule contents and build-file
patches into `dist/sources/` and produces ZIPs, patches, and SHA-256 manifests.
It requires new snapshot directories. Compilation runs on further copies under
`dist/build/`, with a separate environment and `UV_LINK_MODE=copy`.

To rebuild from existing source archives at another repository location, place
the ZIPs and their `.manifest.json` and `.patch` sidecars in `dist/sources/`.
Expand the trusted source archives, then run `build` and `uv lock` above:

~~~text
uv run --no-project --python 3.10.17 python -m zipfile -e dist/sources/python-optix.zip dist/sources/python-optix
uv run --no-project --python 3.10.17 python -m zipfile -e dist/sources/tiny-cuda-nn.zip dist/sources/tiny-cuda-nn
~~~

The build checks extracted source files against their manifests. Wheel hashes
are recorded in `uv.lock`, so rebuilding dependency wheels requires updating
the lockfile. When using the existing matching wheels and lockfile, proceed
directly to installation.

## Install the Renderer

~~~text
uv run --no-project --python 3.10.17 scripts/prepare_local_release.py renderer-wheel
uv run --no-project --python 3.10.17 scripts/prepare_local_release.py validate
~~~

`renderer-wheel` uses the build environment created by the `build` step and
packages a new source copy. If starting with prebuilt dependency wheels, also
provide the renderer wheel in `dist/wheels/` and run only `validate`.

`validate` creates `dist/validation/.venv`, installs locked dependencies and
the renderer wheel, then checks dependencies and isolated imports. It checks the
native tinycudann extension path, packaged `narrow_band.cu`, and embedded OptiX headers.

Validation requires a new environment. For another installation, use
`validate --destination dist/validation-new` and the same destination for later
checks. Ordinary `uv sync` targets the repository's `.venv` instead. Registry
packages require network access or a populated uv cache.

To relocate an installation, copy the code, lockfile, wheels, manifests, license
notices, and scene bundles, then create a new environment at the destination.
The original dependency checkouts and `dist/build/` are not runtime inputs.

## Scene Bundles

The Khady and Horse configs reference checkpoints and meshes that are separate
from the code repository. With those assets in place, create the bundles:

~~~text
uv run --no-project --python 3.10.17 scripts/prepare_local_release.py scenes repro/khady_stage2_infer_prednorm/khady.json repro/khady_stage2_infer_prednorm/horse.json
~~~

To use existing bundles, place each ZIP and its `.manifest.json` sidecar in
`dist/scenes/`, then extract them:

~~~text
uv run --no-project --python 3.10.17 scripts/prepare_local_release.py extract dist/scenes/khady.zip dist/validation/scenes/khady
uv run --no-project --python 3.10.17 scripts/prepare_local_release.py extract dist/scenes/horse.zip dist/validation/scenes/horse
~~~

Each bundle includes its config, selected checkpoint, camera path, shell meshes,
provenance YAML, and file manifest. Extraction requires a new directory, preserves
relative paths, and verifies archive and file SHA-256 hashes. It rejects path
traversal and duplicate or unexpected entries. The manifests provide integrity
checks, not signatures.

## Check and Run

Compile and load the CUDA/OpenGL bridge, then run scalar Torch/CuPy/Dr.Jit CUDA
checks and verify the scene paths without opening a window:

~~~text
uv run --no-project --python 3.10.17 scripts/prepare_local_release.py compile-interop --vcvars 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat'
uv run --no-project --python 3.10.17 scripts/prepare_local_release.py gpu-smoke
~~~

Reports are written to `dist/validation/imports.json`, `interop-build.json`, and
`gpu-smoke.json`; command logs are in `dist/logs/`. These checks cover installation
and CUDA execution, while `--interop-smoke` exercises the CUDA/OpenGL display path.

Launch the installed viewer using the same compiler environment and cache:

~~~text
uv run --no-project --python 3.10.17 scripts/windows_env.py --vcvars 'C:/Program Files (x86)/Microsoft Visual Studio/2022/BuildTools/VC/Auxiliary/Build/vcvars64.bat' --cache-root dist/validation/cache -- dist/validation/.venv/Scripts/python.exe -I -m adashell_renderer.viewer_gl --config dist/validation/scenes/khady/khady.json
~~~

For Horse, replace the config with `dist/validation/scenes/horse/horse.json`.
Without a camera preset, the default camera uses world +Y up and looks from +Z;
a Z-up scene such as Khady can therefore appear from above. The supplied presets
provide scene-specific initial views.
