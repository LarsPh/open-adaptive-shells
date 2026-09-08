# Training

AdaShell training is built on SDFStudio/Nerfstudio. The workflow fits a neural
field, extracts its bounding shells, then fine-tunes the field inside the shells.

Follow [environment setup](../docs/installation.md#training) first.
All training commands below run from `training/`.

## Dataset

Download Khady from the official [Shelly dataset](https://drive.google.com/drive/folders/1VyNLOx5qAJQmIQ_Iw1JSoQ4rnkCWg5uk),
linked on the [Adaptive Shells project page](https://research.nvidia.com/labs/toronto-ai/adaptive-shells/).
Place the scene under `training/data/khady/`, preserving its image paths:

```text
data/khady/
  transforms_train.json
  transforms_test.json
  train/
  test/
```

The Blender dataparser reads the camera JSON directly. It uses the validation
split when present, or the test split otherwise.

## Initial Training

```nu
uv run --no-sync ns-train adashell --trainer.load-config configs/examples/khady_stage1.yml
```

This fits the SDF, spatially varying kernel and color field for 200,000 steps.
The saved config and checkpoints are written to `outputs/khady/adashell/stage1/`.
Settings are in the YAML; `--data` and `--experiment-name` can also be supplied
on the command line.

## Shell Extraction

```nu
uv run --no-sync ns-extract-mesh --load-config outputs/khady/adashell/stage1/config.yml --output-path outputs/khady/adashell/stage1/shell.ply --use-shell True --resolution 512
```

The extractor writes the base mesh, `shell-dilation.ply` and
`shell-erosion.ply`. The outer and inner meshes delimit the sampling band.

## Fine-Tuning

```nu
uv run --no-sync ns-train adashell --trainer.load-config configs/examples/khady_stage2.yml
```

Stage 2 loads the latest stage-1 checkpoint and samples within the extracted
shells. It uses predicted-gradient alpha and predicted normals for color,
with an RGB reconstruction loss, for another 100,000 steps.
Outputs go to `outputs/khady/adashell/stage2/`.

For separate runs, change `timestamp` and the corresponding stage-2 input paths
in the example configs.

## Render

Switch to the renderer environment:

```nu
cd ../renderer
uv run --no-sync python scripts/windows_env.py -- .venv/Scripts/python.exe -m adashell_renderer.viewer_gl --config configs/khady.json
```

The renderer config points to the stage-2 checkpoint, stage-1 shells and dataset
test cameras. This Torch-field checkpoint can be loaded directly.
Update `checkpoint` in the JSON to view an earlier saved step.

## Horse

The same sequence uses `configs/examples/horse_stage1.yml`,
`configs/examples/horse_stage2.yml` and `renderer/configs/horse.json`.
Use `data/horse/` and `outputs/horse/adashell/stage1/` in the data and extraction
steps.

## Predicted Gradients

Computing SDF gradients adds automatic-differentiation work or extra field
queries for finite differences. The renderer instead uses the geometry
network's predicted vector: its magnitude and direction enter the opacity
calculation, while its normalized direction supplies the color network's normal
input. Stage-2 fine-tuning adapts the field to this rendering formulation.

Implementation details are in [Renderer design](../renderer/docs/technical-overview.md).
The inherited [SDFStudio README](docs/sdfstudio-readme.md) and
[data documentation](docs/sdfstudio-data.md) describe the underlying framework.
