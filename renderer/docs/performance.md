# Performance

Core rendering at 1920 x 1080, before and after the optimizations that removed
per-frame buffer fills, field recomputation and re-tracing, and redundant shell
traversals:

| GPU | Scene | Before | After | FPS after |
| --- | --- | --- | --- | --- |
| RTX 5090 | Khady | 4.63 ms | 1.18 ms | 851 |
| RTX 5090 | Horse | 5.00 ms | 1.49 ms | 672 |
| RTX 4060 Ti | Khady | 12.47 ms | 3.75 ms | 267 |
| RTX 4060 Ti | Horse | 16.05 ms | 5.78 ms | 173 |

Medians of three interleaved runs of 300 frames each, camera preset 10,
Windows, driver 610.47, CUDA 12.8 and Dr.Jit 1.5.0; "Before" is the same
code base with only the Dr.Jit upgrade. Times cover shell tracing, sampling,
field evaluation and compositing, including GPU synchronization. Ray
generation and display are excluded; the viewer reports their timings
separately. The rendered images match to within 1.2e-7, the run-to-run
variation of the atomic compositing.

In the GL viewer, a full frame at 1920 x 1080 on the RTX 4060 Ti, including ray
generation, display, ImGui and swap, takes 5.1 ms for Khady and 7.1 ms for Horse
(6.3 ms and 8.3 ms before display overlapped the next frame's rendering).
Medians of 340 continuous frames after 60 warmup frames, camera preset 0,
driver 610.88.

## Benchmark

```nu
uv run python scripts/validate_render.py --config repro/khady_stage2_infer_prednorm/khady.json --output outputs/validation/khady
```

The script warms up the renderer, then measures repeated frames from fixed rays.
It saves per-frame timings and rendered outputs in the selected directory.
FPS above is 1000 divided by the median frame time in milliseconds.
