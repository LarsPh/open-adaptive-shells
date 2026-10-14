# Adaptive Shell Renderer

An unofficial implementation of [Adaptive Shells for Efficient Neural Radiance
Field Rendering](https://research.nvidia.com/labs/toronto-ai/adaptive-shells/)
(SIGGRAPH Asia 2023), exploring shell-guided neural rendering with OptiX,
CuPy, Dr.Jit and OpenGL.

![Khady rendered in the GL viewer](docs/media/khady-viewer.png)

[Video](docs/media/viewer-demo.mp4)

The renderer traces an inner and outer shell to restrict neural-field queries
to a narrow band. It combines sparse volumetric sampling with learned gradients
and displays the result directly from GPU memory. This repository contains the
inference runtime and interactive viewer for checkpoints trained with the
Adaptive Shell experiments in SDFStudio/Nerfstudio.

## Rendering Pipeline

```text
CuPy camera rays -> OptiX shell intervals -> compact samples
                -> Dr.Jit hash encoding + MLPs -> volume compositing
                -> CUDA PBO -> OpenGL texture
```

Only samples inside the shell band are evaluated by the neural field. Rendering
and display stay on the GPU. See [Renderer design](docs/technical-overview.md)
for the sampling, predicted-gradient field and display implementation.

## Quick Start

Tested with Windows, RTX 5090, CUDA Toolkit 12.8 and Python 3.10.
See [installation](../docs/installation.md) for native dependencies and environment setup.
Checkpoints and shell meshes are stored separately from the source code.

```nu
uv run adashell-viewer-gl --config repro/khady_stage2_infer_prednorm/khady.json
```

The CUDA/OpenGL extension compiles on first use. If MSVC is not available in the
current shell, use the build-environment wrapper:

```nu
uv run --no-sync python scripts/windows_env.py -- .venv/Scripts/adashell-viewer-gl.exe --config repro/khady_stage2_infer_prednorm/khady.json
```

A scene config specifies its checkpoint and shell meshes. Camera presets are
optional; without one, the viewer fits a camera to the scene bounds.

The viewer supports orbit and free-look navigation, with selectable diagnostic views.

## Performance

Khady averages **8.3 ms** per frame for core rendering at **1920 x 1080** on
an RTX 5090, excluding ray generation and display.
See [Performance](docs/performance.md) for measurements and the benchmark command.

## Development

```nu
uv run ruff check src/adashell_renderer tests scripts
uv run pytest
uv run adashell-viewer-gl --interop-smoke
```

Set `ADASHELL_TEST_METADATA_SAMPLER_GPU=1` and `ADASHELL_TEST_CUDA_GL_EXTENSION=1`
to include the optional GPU tests.

### Compatibility

- The current native wheels target Windows x64 and `sm_120`.
- Dr.Jit 1.5.0 includes the workaround for the OptiX compile error seen with
  Dr.Jit 1.3.1 on driver 610.47; see the [installation notes](../docs/installation.md).
- Checkpoints must match the supported network topology and paired shell meshes.
- Torch and tiny-cuda-nn are used for checkpoint parameter compatibility;
  per-frame field evaluation runs in Dr.Jit.

## References

Zian Wang et al. **Adaptive Shells for Efficient Neural Radiance Field Rendering**.
ACM Transactions on Graphics, 2023.
[Paper](https://arxiv.org/abs/2311.10091) /
[Project](https://research.nvidia.com/labs/toronto-ai/adaptive-shells/)

Built with OptiX, Dr.Jit, CuPy, tiny-cuda-nn, GLFW and Dear ImGui.
See [dependency details](docs/dependencies.md) for versions and licenses.
Original renderer code is licensed under [MIT](LICENSE); the SDFStudio-derived
field adapter retains [Apache-2.0](LICENSE.Apache-2.0). See [Licenses](../LICENSE.md).
