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

These RTX 5090 figures predate the changes below and will be remeasured.

Core rendering on an RTX 4060 Ti at 1920 x 1080, before and after the
optimizations that removed per-frame buffer fills, field recomputation and
re-tracing, and redundant shell traversals:

| Scene | Before | After | FPS after |
| --- | --- | --- | --- |
| Khady | 12.47 ms | 3.75 ms | 267 |
| Horse | 16.05 ms | 5.78 ms | 173 |

Medians of three interleaved runs of 300 frames each, camera preset 10,
Windows, driver 610.47, CUDA 12.8 and Dr.Jit 1.5.0; "Before" is the same
code base with only the Dr.Jit upgrade. The rendered images match to within
1.2e-7, the run-to-run variation of the atomic compositing.

## Benchmark

```nu
uv run python scripts/validate_render.py --config repro/khady_stage2_infer_prednorm/khady.json --output outputs/validation/khady
```

The script warms up the renderer, then measures repeated frames from fixed rays.
It saves per-frame timings and rendered outputs in the selected directory.
FPS above is the number of measured frames divided by their total render time.
