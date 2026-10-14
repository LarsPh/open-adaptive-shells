from contextlib import nullcontext
import os
from types import SimpleNamespace

import numpy as np
import pytest

from adashell_renderer.config import TraceConfig
from adashell_renderer.drjit_backend import optix_metadata_sampler as sampler_module


@pytest.fixture
def fake_backend(monkeypatch):
    events = []
    history = []
    dr = SimpleNamespace(
        cuda=SimpleNamespace(
            Float=lambda a: np.asarray(a, dtype=np.float32),
            Int=lambda a: np.asarray(a, dtype=np.int32),
            UInt=lambda a: np.asarray(a, dtype=np.uint32),
        ),
        JitFlag=SimpleNamespace(KernelHistory="history"),
        scoped_set_flag=lambda *args: nullcontext(),
        eval=lambda *args: events.append("eval"),
        sync_thread=lambda: events.append("sync"),
        kernel_history=lambda: list(history),
    )
    monkeypatch.setattr(sampler_module.DrJitInterop, "create", lambda: SimpleNamespace(dr=dr))
    monkeypatch.setattr(sampler_module, "_require_cupy", lambda: np)

    def fill(**kwargs):
        events.append("fill")
        counts = kwargs["sample_counts"]
        offsets = np.cumsum(counts, dtype=np.int32) - counts
        np.testing.assert_array_equal(kwargs["ray_offsets"], offsets)
        indices = np.repeat(np.arange(len(counts), dtype=np.int32), counts)
        starts = np.arange(len(indices), dtype=np.float32)
        history.append({"type": "JIT", "execution_time": 1.25})
        return starts, starts + 1, starts + 0.5, indices, {"trace_metadata_cuda_sync_ms": 0.25}

    monkeypatch.setattr(sampler_module, "_trace_metadata_flat_fill_cuda", fill)
    return events


def make_hits(counts):
    n_rays = len(counts)
    return SimpleNamespace(metadata={
        "optix_interval_starts": np.zeros((n_rays, 2), dtype=np.float32),
        "optix_interval_ends": np.ones((n_rays, 2), dtype=np.float32),
        "optix_interval_counts": np.asarray([int(c > 0) for c in counts], dtype=np.int32),
        "optix_sample_counts": np.asarray(counts, dtype=np.int32),
    })


@pytest.mark.parametrize("counts", ([2, 0, 1], [0, 0, 0], []))
@pytest.mark.parametrize("capture_history", (False, True))
@pytest.mark.parametrize("host_sync", (False, True))
def test_sample_returns_owned_valid_only_pack(fake_backend, counts, capture_history, host_sync):
    sampler = sampler_module.OptixMetadataSamplerCore(TraceConfig(), capture_history, host_sync)
    rays = SimpleNamespace(num_rays=len(counts))
    hits = make_hits(counts)
    packed = sampler.sample(rays, hits)
    assert packed.valid_samples == sum(counts)
    assert packed.num_rays == len(counts)
    np.testing.assert_array_equal(packed.packed_info[:, 1], counts)
    np.testing.assert_array_equal(packed.ray_indices, np.repeat(np.arange(len(counts)), counts))
    assert packed.starts.dtype == np.float32
    assert packed.ray_indices.dtype == np.int32
    assert packed.packed_info.dtype == np.int32
    cache = packed.metadata["_drjit_cache"]
    assert set(cache) == {"starts", "ends", "eval_t", "ray_indices", "sample_counts", "ray_offsets", "deltas"}
    assert cache["starts"] is packed.starts
    np.testing.assert_array_equal(cache["ray_offsets"], packed.packed_info[:, 0])
    np.testing.assert_array_equal(cache["sample_counts"], counts)
    np.testing.assert_array_equal(cache["deltas"], packed.ends - packed.starts)
    # Rendering stays asynchronous unless stats or kernel history ask for host waits.
    assert fake_backend == ["fill", "eval"] + (["sync"] if host_sync or capture_history else [])
    assert packed.metadata["trace_metadata_cuda_sync_ms"] == 0.25
    assert ("kernel_history_count" in packed.metadata) == capture_history
    assert not hasattr(sampler, "_cache")
    assert not hasattr(sampler, "take_cached_pack")
    first_starts = packed.starts.copy()
    second = sampler.sample(rays, hits)
    assert second is not packed
    assert second.starts is not packed.starts
    np.testing.assert_array_equal(packed.starts, first_starts)


@pytest.mark.parametrize("key", (
    "optix_interval_starts", "optix_interval_ends",
    "optix_interval_counts", "optix_sample_counts",
))
def test_missing_metadata_fails_before_fill(fake_backend, key):
    sampler = sampler_module.OptixMetadataSamplerCore(TraceConfig())
    hits = make_hits([1])
    del hits.metadata[key]
    with pytest.raises(ValueError, match=key):
        sampler.sample(SimpleNamespace(num_rays=1), hits)
    assert fake_backend == []


def test_mismatched_ray_batch_fails_before_fill(fake_backend):
    sampler = sampler_module.OptixMetadataSamplerCore(TraceConfig())
    with pytest.raises(ValueError, match="input ray batch"):
        sampler.sample(SimpleNamespace(num_rays=2), make_hits([1]))
    assert fake_backend == []


def test_flat_fill_empty_does_not_compile_or_launch(monkeypatch):
    cp = SimpleNamespace(
        ndarray=np.ndarray, float32=np.float32, int32=np.int32,
        ascontiguousarray=np.ascontiguousarray, asnumpy=np.asarray, empty=np.empty,
    )
    monkeypatch.setattr(sampler_module, "_require_cupy", lambda: cp)
    outputs = sampler_module._trace_metadata_flat_fill_cuda(
        interval_starts=np.zeros((0, 2), dtype=np.float32),
        interval_ends=np.zeros((0, 2), dtype=np.float32),
        interval_counts=np.zeros(0, dtype=np.int32),
        sample_counts=np.zeros(0, dtype=np.int32),
        ray_offsets=np.zeros(0, dtype=np.int32),
        n_sample_max=16, single_sample_threshold=0.02, sample_spacing=0.01,
    )
    assert all(array.shape == (0,) for array in outputs[:4])
    assert "trace_metadata_cuda_launch_ms" not in outputs[4]


def test_flat_fill_launch_synchronizes_and_reuses_kernel(monkeypatch):
    events = []
    launches = []

    class Stream:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def synchronize(self):
            events.append("sync")

    def kernel(grid, block, args):
        events.append("launch")
        launches.append((grid, block, args))

    def compile_kernel(source, name):
        assert source == sampler_module._TRACE_METADATA_FLAT_FILL_CUDA
        assert name == "adashell_trace_metadata_flat_fill"
        events.append("compile")
        return kernel

    cp = SimpleNamespace(
        ndarray=np.ndarray, float32=np.float32, int32=np.int32,
        ascontiguousarray=np.ascontiguousarray, asnumpy=np.asarray, empty=np.empty,
        RawKernel=compile_kernel,
        cuda=SimpleNamespace(get_current_stream=lambda: Stream()),
    )
    monkeypatch.setattr(sampler_module, "_require_cupy", lambda: cp)
    monkeypatch.setattr(sampler_module, "_TRACE_METADATA_FLAT_FILL_KERNEL", None)
    hits = make_hits([2, 0, 1]).metadata
    for _ in range(2):
        outputs = sampler_module._trace_metadata_flat_fill_cuda(
            interval_starts=hits["optix_interval_starts"],
            interval_ends=hits["optix_interval_ends"],
            interval_counts=hits["optix_interval_counts"],
            sample_counts=hits["optix_sample_counts"],
            ray_offsets=np.asarray([0, 2, 2], dtype=np.int32),
            n_sample_max=16, single_sample_threshold=0.02, sample_spacing=0.01,
        )
        assert all(array.shape == (3,) for array in outputs[:4])
        assert "trace_metadata_cuda_sync_ms" in outputs[4]
    assert events == ["compile", "launch", "sync", "launch", "sync"]
    grid, block, args = launches[0]
    assert grid == (1,) and block == (256,)
    assert args[9:12] == (3, 2, 16)
    assert args[12] == np.float32(0.02)
    assert args[13] == np.float32(0.01)


def test_metadata_conversion_rejects_non_cupy_inputs():
    cp = SimpleNamespace(ndarray=np.ndarray, float32=np.float32)
    with pytest.raises(TypeError, match="CuPy arrays"):
        sampler_module._to_cupy(cp, [1.0], cp.float32)


def _flat_fill_cpu_oracle(intervals, n_sample_max, threshold, spacing, limits=None):
    """Build complete interval bins, then apply each ray's metadata budget."""
    starts, ends, eval_t, ray_indices, counts = [], [], [], [], []
    for ray, ray_intervals in enumerate(intervals):
        bins = []
        for enter, exit in ray_intervals:
            enter, exit = np.float32(enter), np.float32(exit)
            extra = np.maximum(np.float32(exit - enter - threshold), np.float32(0))
            count = min(n_sample_max, max(1, int(np.ceil(extra / spacing)) + 1))
            step = np.float32((exit - enter) / np.float32(count + 1))
            points = enter + np.arange(1, count + 1, dtype=np.float32) * step
            boundaries = np.concatenate((
                [enter], (points[:-1] + points[1:]) * np.float32(0.5), [exit],
            ))
            bins.extend(zip(boundaries[:-1], boundaries[1:], points))
        if limits is not None:
            bins = bins[:limits[ray]]
        counts.append(len(bins))
        for start, end, point in bins:
            starts.append(start)
            ends.append(end)
            eval_t.append(point)
            ray_indices.append(ray)
    return (
        np.asarray(starts, dtype=np.float32),
        np.asarray(ends, dtype=np.float32),
        np.asarray(eval_t, dtype=np.float32),
        np.asarray(ray_indices, dtype=np.int32),
        np.asarray(counts, dtype=np.int32),
    )


def test_flat_fill_cpu_oracle_known_bins():
    starts, ends, points, indices, counts = _flat_fill_cpu_oracle(
        [[(0.0, 1.0)], [], [(2.0, 2.125)]],
        3, np.float32(0.25), np.float32(0.125),
    )
    np.testing.assert_array_equal(starts, [0.0, 0.375, 0.625, 2.0])
    np.testing.assert_array_equal(ends, [0.375, 0.625, 1.0, 2.125])
    np.testing.assert_array_equal(points, [0.25, 0.5, 0.75, 2.0625])
    np.testing.assert_array_equal(indices, [0, 0, 0, 2])
    np.testing.assert_array_equal(counts, [3, 0, 1])


@pytest.mark.parametrize("intervals,limits", (
    ([], None),
    ([[], [], []], None),
    ([[(0.0, 0.0), (1.0, 1.125)], [], [(2.0, 2.25), (3.0, 3.5)]], None),
    ([[(0.0, float(np.nextafter(np.float32(0.25), np.float32(0))))],
      [(0.0, 0.25)],
      [(0.0, float(np.nextafter(np.float32(0.25), np.float32(1))))]], None),
    ([[(-2.0, 3.0)], [(1.0, 1.5), (2.0, 4.0)], []], [2, 4, 0]),
    ([[(0.0, 2.0)] if ray % 3 else [] for ray in range(260)], None),
), ids=("zero-rays", "empty-rays", "multiple-intervals", "threshold", "budget-prefix", "block-boundary"))
def test_cupy_flat_fill_gpu_opt_in(intervals, limits):
    if os.environ.get("ADASHELL_TEST_METADATA_SAMPLER_GPU") != "1":
        pytest.skip("Set ADASHELL_TEST_METADATA_SAMPLER_GPU=1 for CUDA flat fill parity.")
    import cupy as cp

    n_sample_max = 4
    threshold, spacing = np.float32(0.25), np.float32(0.125)
    expected = _flat_fill_cpu_oracle(intervals, n_sample_max, threshold, spacing, limits)
    n_rays = len(intervals)
    max_intervals = max(1, max((len(row) for row in intervals), default=0))
    interval_starts = np.zeros((n_rays, max_intervals), dtype=np.float32)
    interval_ends = np.zeros_like(interval_starts)
    for ray, row in enumerate(intervals):
        for slot, (enter, exit) in enumerate(row):
            interval_starts[ray, slot] = enter
            interval_ends[ray, slot] = exit
    counts = expected[4]
    offsets = np.cumsum(counts, dtype=np.int32) - counts
    # Exercise the helper's current-stream ownership handoff, not just stream 0.
    with cp.cuda.Stream(non_blocking=True):
        actual = sampler_module._trace_metadata_flat_fill_cuda(
            interval_starts=cp.asarray(interval_starts),
            interval_ends=cp.asarray(interval_ends),
            interval_counts=cp.asarray([len(row) for row in intervals], dtype=cp.int32),
            sample_counts=cp.asarray(counts),
            ray_offsets=cp.asarray(offsets),
            n_sample_max=n_sample_max,
            single_sample_threshold=float(threshold),
            sample_spacing=float(spacing),
        )
        for output, reference in zip(actual[:3], expected[:3]):
            assert output.dtype == cp.float32
            np.testing.assert_allclose(cp.asnumpy(output), reference, rtol=2e-6, atol=2e-6)
        assert actual[3].dtype == cp.int32
        np.testing.assert_array_equal(cp.asnumpy(actual[3]), expected[3])
