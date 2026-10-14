# Dependency Provenance

## Source Revisions

| Dependency | Repository / Commit | Build Changes |
| --- | --- | --- |
| python-optix 1.0.0 | `https://github.com/mortacious/python-optix`, `34cb83d9d69fc4edf73f06ea1bd9704f21bcaf38` | `docs/patches/python-optix-build.patch` selects `cupy-cuda12x>=13.4.1` |
| tinycudann 2.0 | `https://github.com/NVlabs/tiny-cuda-nn`, `2e757bbe781db59c4980d389d7dccbf5edc09669` | Additional patch to `bindings/torch/setup.py` and added `bindings/torch/pyproject.toml`, preserved in the source snapshot |

The tinycudann snapshot includes changes beyond the listed commit. Its setup
patch replaces `pkg_resources` with `packaging.version`, avoids Torch imports
and extension construction during metadata generation, discovers `CUDA_HOME`
from `CUDA_PATH` or nvcc, and creates
`torch/include/c10/cuda/impl/cuda_cmake_macros.h` if that header is missing.
The added TOML declares NumPy, packaging, setuptools >=61,<82, Torch, and wheel
as build dependencies.

The setup also recreates the binding's RTC directory during metadata handling.
The build script therefore uses copied sources and a separate environment,
keeping generated files and any Torch header changes in the build directory.

## Source Snapshots

python-optix requires its Python/Cython sources, `.pxd` declarations,
`optix_includes.h`, build metadata, README, and MIT license. Compiled `.pyd`
files are rebuilt. The setup accepts OptiX 7.6.x; this build selects 7.6.0 and
embeds headers from the SDK specified by `--optix-sdk`.

tinycudann requires its repository-root layout: `CMakeLists.txt`, `include`,
`src`, `dependencies`, `bindings/torch`, and license notices. The build reads
sources relative to the Torch binding, so that directory alone is insufficient.
The snapshot includes these submodule revisions:

| Submodule | Commit |
| --- | --- |
| `dependencies/cmrc` | `952ffddba731fc110bd50409e8d2b8a06abbd237` |
| `dependencies/cutlass` | `1eb6355182a5124639ce9d3ff165732a94ed9a70` |
| `dependencies/fmt` | `fa2eb2d2e3ec5c21629f8ccd88ae05ec40b963fa` |

`dist/sources/*.manifest.json` records the source revision, remotes, submodule
pins, patch hashes, extra files, file hashes, and license paths. Source ZIPs
contain the patched working-tree files; `.patch` files describe tracked changes
relative to HEAD. Keep the ZIPs, manifests, and patches together when rebuilding
at another location. A top-level `git archive HEAD` omits submodule contents and
uncommitted build-file changes.

## Runtime Versions

| Component | Version |
| --- | --- |
| Python | CPython 3.10.17 x64 |
| Torch | 2.11.0+cu128 |
| CUDA Toolkit | 12.8 |
| GPU architecture | `sm_120` / compute capability 12.0 |
| NumPy | 2.2.6 |
| CuPy | cupy-cuda12x 14.0.1 |
| Dr.Jit | 1.5.0 |
| OptiX SDK | 7.6.0 |

Other resolved versions are fixed in `uv.lock`. The native wheels have CPython
3.10 / Windows x64 tags, and tinycudann contains the `_120_C` extension. Changes
to the Python, Torch, CUDA, or GPU architecture profile require compatible
extension builds.

The renderer wheel includes `adashell_renderer/narrow_band.cu`. OptiX compiles
it through CuPy NVRTC at runtime. The CUDA/OpenGL bridge uses Torch `load_inline`,
so CUDA Toolkit, MSVC, and Ninja remain runtime requirements. Training source
is not imported; the scene provenance YAML records the training configuration.

## Licenses

| Component | License or Terms |
| --- | --- |
| AdaShell Renderer | MIT for original code; Apache-2.0 for the SDFStudio-derived field adapter. Both license texts are included in the wheel. |
| python-optix | MIT; see its `LICENSE` |
| tiny-cuda-nn | BSD-3-Clause; see its `LICENSE.txt` |
| Bundled source dependencies | Component-specific notices, including CUTLASS, fmt, cmrc, JSON, and PCG32; PCG32 includes an Apache-2.0 notice |
| OptiX SDK headers | NVIDIA OptiX SDK terms in `OptiX_EndUserLicense.pdf`, with `OptiX_ThirdParty_Licenses.txt` |
| CUDA/RTC headers | CUDA Toolkit terms in `EULA.txt` and `LICENSE` |
| Scene checkpoints and meshes | No separate asset license text is included in the scene bundles |

The python-optix wheel embeds OptiX headers, and tinycudann packages selected
CUDA/RTC headers. These components retain their respective license terms rather
than inheriting the renderer's MIT license. `dist/licenses/` collects the source
and SDK notices; `dist/dependency-wheels.manifest.json` records their hashes and
the SDK header hashes. Redistribution conditions are specified in those licenses.

See [Windows Installation](windows-install.md) for wheel builds, source snapshot
restoration, scene extraction, and runtime checks.
