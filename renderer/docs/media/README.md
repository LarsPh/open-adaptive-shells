# Recording

`record_viewer.py` captures the OpenGL framebuffer using FFmpeg. The
`--camera-tour` option runs a repeatable camera sequence through the viewer's
camera controller.

```nu
uv run python scripts/windows_env.py -- .venv/Scripts/python.exe scripts/record_viewer.py --camera-tour --output outputs/khady-teaser.mp4 --seconds 60 -- --config repro/khady_stage2_infer_prednorm/khady.json
```
