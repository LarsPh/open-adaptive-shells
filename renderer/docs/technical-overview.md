# Renderer Design

This repository is an inference and interactive-inspection runtime for trained
Adaptive Shell neural fields. It loads existing checkpoints and shell meshes;
it does not train a field or extract a shell. The implementation combines OptiX
ray traversal, compact CuPy sampling, Dr.Jit neural-field evaluation, and a
CUDA/OpenGL display path.

The renderer uses predicted-gradient alpha,
predicted-normal color input, Dr.Jit hash encoding and geometry/color MLPs,
and a CUDA GL viewer. The `use_fastpath` preset applies optimized defaults.
With the preset disabled, explicit settings are preserved and validated against
the supported field topology.

## One Frame

```text
Camera pose + intrinsics
  | CuPy ray-generation kernel
  v
Full-frame rays [R, 3]
  | OptiX traversal of dilation and erosion meshes
  v
Intervals + per-ray interval/sample counts
  | exclusive offsets + CuPy flat-fill kernel
  v
Packed samples [S], S = sum(sample_counts)
  | Dr.Jit query points, hash encoding, geometry MLP
  v
SDF + inverse scale + geometry features + predicted gradient
  | directional-gradient alpha + color MLP + compositing
  v
DrRenderBuffer: three float32 RGB channels [R]
  | DLPack views, CUDA pack into mapped PBO
  v
RGBA8 OpenGL texture -> ImGui viewport -> buffer swap
```

`R` is the full image ray count. `S` includes only actual samples. Rays with
zero samples retain their output pixel but contribute no neural-network
queries.

## Assembly and Ownership

[`build_runtime_pipeline`](../src/adashell_renderer/renderer_runtime.py) applies
the supported configuration and delegates construction to
[`stage_adapter.py`](../src/adashell_renderer/drjit_backend/stage_adapter.py).
The returned `RendererPipeline` owns the tracer, sampler, and field island.

[`RendererPipeline.run_render_buffer`](../src/adashell_renderer/renderer_pipeline.py)
has three execution steps: trace, sample, and field evaluation. The buffer retains
its Dr arrays and associated diagnostic/profile metadata until the consumer
is finished with them.

The field loader in [`legacy_field.py`](../src/adashell_renderer/legacy_field.py)
preserves the training checkpoint's parameter names and topology. Torch and
tiny-cuda-nn remain dependencies for loading and interpreting those parameters.
Field inference runs in Dr.Jit. CPU camera objects retain Torch tensors;
the per-pixel CUDA ray-generation kernel is CuPy-native.

## Traversal and Sampling

[`ProjectOptixRayTracer`](../src/adashell_renderer/optix_trace.py) builds an
acceleration structure for each shell mesh and launches
[`narrow_band.cu`](../src/adashell_renderer/narrow_band.cu). Traversal produces
CuPy-owned metadata buffers, including interval starts/ends, interval counts,
and sample counts. The sampler consumes these buffers directly.

[`OptixMetadataSamplerCore`](../src/adashell_renderer/drjit_backend/optix_metadata_sampler.py)
computes offsets from sample counts and allocates flat arrays for sample start,
end, evaluation distance, and full-frame ray index. Its CUDA fill kernel writes
the samples selected by the trace metadata. Dr views of the packed arrays,
offsets, and counts form the field's input contract.

Sample storage scales with `S` rather than `R * maximum_samples`.
Sample sizing and profiling introduce GPU synchronization and host-scalar reads.

## Field Evaluation

[`DrJitFieldIsland`](../src/adashell_renderer/drjit_backend/field_island.py)
gathers ray origins and directions using packed ray indices and forms query
points at the packed evaluation distances. Query positions and hash-grid
features feed the geometry MLP. Its output is interpreted as SDF, inverse
scale, geometry features, and a predicted gradient vector.

The alpha path clamps the predicted gradient norm to the
configured range, then projects the vector onto the ray direction. The
non-positive directional derivative estimates SDF at the two ends of each
sample interval. Sigmoid CDF differences produce alpha. The color path uses
normalized predicted normals, encoded view/reflection directions, geometry
features, and the checkpoint's appearance-input convention.

Transmittance is an exclusive product within each ray, implemented through
log-survival prefix sums and subtraction of each ray's starting prefix.
Only that long-prefix accumulation and subtraction use float64; the result
returns to float32 before exponentiation and weighting. A float32 global
prefix loses local optical-depth precision on large images. The regression
in [`test_transmittance.py`](../tests/test_transmittance.py) checks that earlier,
unrelated rays cannot change the weights of identical later rays.

Weighted sample colors scatter into full-frame output channels. Empty rays
receive the configured background without network evaluation. Optional
diagnostics include weighted depth, predicted normals/gradients, gradient
norm, kernel size, and weight sum. `sample_count` comes directly from the
sampler's counts. Scalar diagnostic ranges
change display packing, not the underlying field result.

## Display

[`CudaGlPresenter`](../src/adashell_renderer/viewer_gl.py) evaluates the Dr output,
synchronizes its producer, and obtains contiguous float32 CuPy views through
DLPack. It validates one value per pixel per channel and the current device.
The views remain alive while native code consumes their device pointers.

The presenter registers an OpenGL pixel-unpack buffer with CUDA. For each
upload the extension maps that PBO, packs the RGB/scalar/vector input into
`uchar4` pixels, and unmaps it before OpenGL reads it. `glTexSubImage2D` uses
the bound PBO as its source to update the texture. ImGui draws the texture;
GLFW swaps the window buffers.

These are three different resources: Dr/CuPy field storage, a CUDA-writable
OpenGL transfer buffer, and a texture sampled by OpenGL. DLPack avoids a copy
at the Dr/CuPy boundary; CUDA packing and the PBO-to-texture transfer remain
separate GPU operations. The image stays on the GPU throughout presentation.

The framebuffer is full-frame and single-chunk. Resizing unregisters CUDA's
PBO resource before deleting/replacing the GL buffers. Image rows follow the
ray generator's top-to-bottom ordering; the ImGui UVs preserve that ordering.

## Viewer State

`run_gl_viewer` owns resource creation and cleanup. `GlViewerSession` owns
the event/UI/render/display loop and its previous result. `ViewerState` owns
camera and renderer settings. Dirty or continuous frames render; a display
range change repacks the previous buffer; idle frames reuse the texture.

Camera presets preserve their imported pose. Orbit rotates around the selected
center; Look rotates in place. Drag sensitivity follows the displayed viewport
size, independently of rendering resolution. Without a preset, the viewer fits
an initial camera to the shell bounds, falling back to the field AABB.

See [Performance](performance.md) for benchmarks and
[Windows Installation](windows-install.md) for the supported environment.
