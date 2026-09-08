# Local Demo Scenes

`khady.json` and `horse.json` load the demo checkpoints and shell meshes.

```nu
uv run adashell-viewer-gl --config repro/khady_stage2_infer_prednorm/khady.json
uv run adashell-viewer-gl --config repro/khady_stage2_infer_prednorm/horse.json
```

Scene assets are stored separately from the code. See
[scene packaging](../../docs/windows-install.md#scene-bundles) for archive setup,
or the [training walkthrough](../../../training/README.md) to train a new scene.
