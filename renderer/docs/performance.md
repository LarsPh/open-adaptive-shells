# Performance

Core rendering on an RTX 5090 at 1920 x 1080:

| Scene | Average Time | FPS |
| --- | --- | --- |
| Khady | 8.3 ms | 120 |
| Horse | 8.3 ms | 120 |

Measured with camera preset 10 on Windows, driver 596.36 Studio, CUDA 12.8
and Dr.Jit 1.3.1. Times cover shell tracing, sampling, field evaluation and
compositing, including GPU synchronization. Ray generation and display are
excluded; the viewer reports their timings separately.

## Benchmark

```nu
uv run python scripts/validate_render.py --config repro/khady_stage2_infer_prednorm/khady.json --output outputs/validation/khady
```

The script warms up the renderer, then measures repeated frames from fixed rays.
It saves per-frame timings and rendered outputs in the selected directory.
FPS above is the number of measured frames divided by their total render time.
