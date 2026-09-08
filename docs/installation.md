# Installation

Training and rendering have separate Python environments. Run `uv` from the
project directory you want to use.

## Windows Environment

The local setup uses RTX 5090, CUDA Toolkit 12.8, OptiX SDK 7.6.0 and Visual Studio
2022 C++ tools. Install uv and the native toolkits before building dependencies.

| Package | Training | Renderer |
| --- | --- | --- |
| Python | 3.10.17 | 3.10.17 |
| Torch | 2.7.1+cu128 | 2.11.0+cu128 |
| NumPy | 1.26.4 | 2.2.6 |
| CuPy | 13.4.1 | 14.0.1 |
| Dr.Jit | - | 1.3.1 |

Use Studio driver **596.36**. Driver **610.47** caused
`Target architecture invalid` during Dr.Jit/OptiX PTX compilation on RTX 5090,
with CoopVec matrix-vector operations and `ScatterReduceLocal` scatter reduction.
Rolling back to 596.36 restored operation.

## Native Sources

From the repository root:

```sh
git clone https://github.com/mortacious/python-optix.git python-optix
git -C python-optix checkout 34cb83d9d69fc4edf73f06ea1bd9704f21bcaf38
git -C python-optix apply ../docs/patches/python-optix-build.patch
git clone https://github.com/NVlabs/tiny-cuda-nn.git tiny-cuda-nn
git -C tiny-cuda-nn checkout 2e757bbe781db59c4980d389d7dccbf5edc09669
git -C tiny-cuda-nn submodule update --init --recursive
git -C tiny-cuda-nn apply ../docs/patches/tiny-cuda-nn-build.patch
```

The build patch adapts the Torch binding to the local setuptools/CUDA setup.
These source directories are ignored by Git. See
[dependency details](../renderer/docs/dependencies.md) for provenance and licenses.

## Training

The following commands use Nushell. Set the SDK location to your installation:

```nu
cd training
$env.OPTIX_PATH = 'C:/Program Files/NVIDIA Corporation/OptiX 7.6.0'
$env.OPTIX_EMBED_HEADERS = '1'
```

The environment wrapper locates Visual Studio, loads MSVC and selects `sm_120`:

```nu
uv run --no-project --python 3.10.17 ../renderer/scripts/windows_env.py -- uv sync --frozen --no-install-package tinycudann
uv run --no-project --python 3.10.17 ../renderer/scripts/windows_env.py -- uv sync --frozen --no-build-isolation
uv run --no-sync ns-train --help
```

Installing Torch first makes the native extension build against the training
environment's Torch. Continue with the [training walkthrough](../training/README.md).

## Renderer

From the repository root, build the dependency wheels:

```nu
cd renderer
uv run --no-project --python 3.10.17 scripts/prepare_local_release.py snapshot --python-optix ../python-optix --tiny-cuda-nn ../tiny-cuda-nn
uv run --no-project --python 3.10.17 scripts/prepare_local_release.py build --optix-sdk 'C:/Program Files/NVIDIA Corporation/OptiX 7.6.0'
uv lock
uv sync --frozen --extra viewer-gl --extra dev
```

With matching wheels already in `renderer/dist/wheels/`, only the last command
is needed. Rebuilt wheels need a refreshed lockfile because their hashes change.

Check the display bridge, then open the viewer:

```nu
uv run --no-sync python scripts/windows_env.py -- .venv/Scripts/python.exe -m adashell_renderer.viewer_gl --interop-smoke
uv run --no-sync python scripts/windows_env.py -- .venv/Scripts/python.exe -m adashell_renderer.viewer_gl --config repro/khady_stage2_infer_prednorm/khady.json
```

The CUDA/OpenGL bridge compiles on first use. For wheel packaging and scene
archives, see [Windows packaging](../renderer/docs/windows-install.md).
