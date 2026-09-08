# Repository Layout

`training/` contains the SDFStudio-based AdaShell training and shell-extraction
code. `renderer/` contains the standalone inference runtime and GL viewer.
Each directory has its own Python project, lockfile and virtual environment.
Run installation, tests and project commands from the corresponding directory.

The two projects use different Torch and NumPy versions. There is no root
Python workspace or shared virtual environment.

## Training

Run training commands from `training/`, where relative dataset and output paths
are resolved. The existing `nerfstudio` package name is preserved so saved
training configurations can resolve their Python classes.

The lockfile expects patched native dependency sources at
`python-optix/` and `tiny-cuda-nn/` in the repository root. These are local build
inputs and are ignored by Git. They must be built against the training
environment; the renderer's native wheels target a different Torch version.

See [installation](installation.md#training) for environment setup and
[training](../training/README.md) for the scene workflow.

The inherited SDFStudio documentation and workflow files remain under
`training/`. Workflows in `training/.github/` are not active repository Actions.

## Renderer

```sh
cd renderer
uv run adashell-viewer-gl --config repro/khady_stage2_infer_prednorm/khady.json
```

See [installation](../renderer/docs/windows-install.md) for dependencies.
Scene configurations resolve asset paths relative to the configuration file.
Checkpoints, shell meshes and native dependency wheels remain local assets;
moving the source code alone does not copy them.

## Training Outputs

The renderer consumes a checkpoint, paired shell meshes and a scene config,
with an optional camera preset. It does not import the training package.
Keep checkpoint topology, shell coordinates and camera conventions consistent
when exporting a trained scene.
