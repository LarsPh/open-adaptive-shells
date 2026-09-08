from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import numpy as np


def _import_optix_modules() -> tuple[Any, Any]:
    try:
        import cupy as cp  # type: ignore[import-not-found]
        import optix as ox  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "Project-owned OptiX tracing requires the `cupy` and `optix` Python packages in the new environment."
        ) from exc
    return ox, cp


def _to_cupy_array(cp: Any, array: Any, *, shape: tuple[int, ...], name: str) -> Any:
    if not isinstance(array, cp.ndarray):
        raise TypeError("OptiX tracing requires CuPy ray buffers.")
    if array.dtype != cp.float32 or not array.flags.c_contiguous:
        raise ValueError("OptiX ray buffers must be contiguous float32 arrays.")
    if array.shape != shape:
        raise ValueError(f"OptiX {name} must have shape {shape}, got {array.shape}.")
    if array.device.id != cp.cuda.runtime.getDevice():
        raise ValueError(f"OptiX {name} must be on the current CUDA device.")
    return array


def create_context(log: logging.Logger):
    ox, _ = _import_optix_modules()
    logger = ox.Logger(log)
    ctx = ox.DeviceContext(validation_mode=False, log_callback_function=logger, log_callback_level=4)
    ctx.cache_enabled = False
    return ctx


def create_acceleration_structure(ctx: Any, mesh: Any):
    ox, _ = _import_optix_modules()
    build_input = ox.BuildInputTriangleArray(
        vertex_buffers=mesh.vertices.astype(np.float32),
        index_buffer=mesh.faces.astype(np.uint32),
        num_sbt_records=1,
        flags=[ox.GeometryFlags.NONE],
    )
    return ox.AccelerationStructure(ctx, build_input, compact=True, allow_update=False, random_vertex_access=True)


def create_module(ctx: Any, pipeline_opts: Any, source_file: Path):
    ox, _ = _import_optix_modules()
    compile_opts = ox.ModuleCompileOptions(
        debug_level=ox.CompileDebugLevel.DEFAULT,
        opt_level=ox.CompileOptimizationLevel.DEFAULT,
    )
    return ox.Module(ctx, str(source_file), compile_opts, pipeline_opts)


def create_program_groups(ctx: Any, module: Any):
    ox, _ = _import_optix_modules()
    raygen_grp = ox.ProgramGroup.create_raygen(ctx, module, "__raygen__rg")
    miss_grp = ox.ProgramGroup.create_miss(ctx, module, "__miss__ms")
    hit_grp = ox.ProgramGroup.create_hitgroup(ctx, module, entry_function_CH="__closesthit__ch")
    return raygen_grp, miss_grp, hit_grp


def create_pipeline(ctx: Any, program_grps: Any, pipeline_options: Any):
    ox, _ = _import_optix_modules()
    link_opts = ox.PipelineLinkOptions(max_trace_depth=1, debug_level=ox.CompileDebugLevel.DEFAULT)
    pipeline = ox.Pipeline(ctx, compile_options=pipeline_options, link_options=link_opts, program_groups=program_grps)
    pipeline.compute_stack_sizes(1, 0, 0)
    return pipeline


def create_sbt(program_grps: Any):
    ox, _ = _import_optix_modules()
    raygen_grp, miss_grp, hit_grp = program_grps
    raygen_sbt = ox.SbtRecord(raygen_grp)
    miss_sbt = ox.SbtRecord(miss_grp)
    hit_sbt = ox.SbtRecord(hit_grp)
    return ox.ShaderBindingTable(raygen_record=raygen_sbt, miss_records=miss_sbt, hitgroup_records=hit_sbt)


def _create_launch_params(
    dilation_gas: Any,
    erosion_gas: Any,
    depth_buffer: Any,
    hit_t_buffer: Any,
    hit_count_buffer: Any,
    image_width: int,
    image_height: int,
    max_hits: int,
    ray_origins: Any,
    ray_directions: Any,
    ray_tmins: Any,
    ray_tmaxs: Any,
    erosion_first_t_buffer: Any,
    dilation_hit_flag_buffer: Any,
    interval_start_buffer: Any | None = None,
    interval_end_buffer: Any | None = None,
    interval_count_buffer: Any | None = None,
    sample_count_buffer: Any | None = None,
    max_intervals: int = 0,
    n_sample_max: int = 0,
    min_interval_size: float = 1e-4,
    single_sample_threshold: float = 0.02,
    sample_spacing: float = 0.01,
    compress_same_flag_runs: bool = True,
):
    ox, _ = _import_optix_modules()
    params_template = [
        ("u8", "depth_buffer"),
        ("u8", "hit_t_buffer"),
        ("u8", "hit_count_buffer"),
        ("u4", "image_width"),
        ("u4", "image_height"),
        ("u4", "max_hits"),
        ("3f4", "cam_eye"),
        ("3f4", "cam_u"),
        ("3f4", "cam_v"),
        ("3f4", "cam_w"),
        ("u8", "ray_origins_buffer"),
        ("u8", "ray_directions_buffer"),
        ("u8", "ray_tmin_buffer"),
        ("u8", "ray_tmax_buffer"),
        ("u8", "dilation_trav_handle"),
        ("u8", "erosion_trav_handle"),
        ("u8", "erosion_first_t_buffer"),
        ("u8", "dilation_hit_flag_buffer"),
        ("u8", "interval_start_buffer"),
        ("u8", "interval_end_buffer"),
        ("u8", "interval_count_buffer"),
        ("u8", "sample_count_buffer"),
        ("u4", "max_intervals"),
        ("u4", "n_sample_max"),
        ("f4", "min_interval_size"),
        ("f4", "single_sample_threshold"),
        ("f4", "sample_spacing"),
        ("u4", "compress_same_flag_runs"),
    ]
    launch_params = ox.LaunchParamsRecord(
        names=[p[1] for p in params_template],
        formats=[p[0] for p in params_template],
    )
    launch_params["depth_buffer"] = depth_buffer.data.ptr
    launch_params["hit_t_buffer"] = hit_t_buffer.data.ptr
    launch_params["hit_count_buffer"] = hit_count_buffer.data.ptr
    launch_params["image_width"] = int(image_width)
    launch_params["image_height"] = int(image_height)
    launch_params["max_hits"] = int(max_hits)
    zero3 = np.zeros((3,), dtype=np.float32)
    launch_params["cam_eye"] = zero3
    launch_params["cam_u"] = zero3
    launch_params["cam_v"] = zero3
    launch_params["cam_w"] = zero3
    launch_params["ray_origins_buffer"] = ray_origins.data.ptr
    launch_params["ray_directions_buffer"] = ray_directions.data.ptr
    launch_params["ray_tmin_buffer"] = ray_tmins.data.ptr
    launch_params["ray_tmax_buffer"] = ray_tmaxs.data.ptr
    launch_params["dilation_trav_handle"] = dilation_gas.handle
    launch_params["erosion_trav_handle"] = erosion_gas.handle
    launch_params["erosion_first_t_buffer"] = erosion_first_t_buffer.data.ptr
    launch_params["dilation_hit_flag_buffer"] = dilation_hit_flag_buffer.data.ptr
    launch_params["interval_start_buffer"] = 0 if interval_start_buffer is None else interval_start_buffer.data.ptr
    launch_params["interval_end_buffer"] = 0 if interval_end_buffer is None else interval_end_buffer.data.ptr
    launch_params["interval_count_buffer"] = 0 if interval_count_buffer is None else interval_count_buffer.data.ptr
    launch_params["sample_count_buffer"] = 0 if sample_count_buffer is None else sample_count_buffer.data.ptr
    launch_params["max_intervals"] = int(max_intervals)
    launch_params["n_sample_max"] = int(n_sample_max)
    launch_params["min_interval_size"] = float(min_interval_size)
    launch_params["single_sample_threshold"] = float(single_sample_threshold)
    launch_params["sample_spacing"] = float(sample_spacing)
    launch_params["compress_same_flag_runs"] = 1 if compress_same_flag_runs else 0
    return launch_params


def launch_pipeline_ray_batch(
    pipeline: Any,
    sbt: Any,
    dilation_gas: Any,
    erosion_gas: Any,
    ray_origins: Any,
    ray_directions: Any,
    ray_tmins: Any,
    ray_tmaxs: Any,
    n_depth_peeling_max: int,
    blocking_launch: bool = True,
    n_sample_max: int = 0,
    min_interval_size: float = 1e-4,
    single_sample_threshold: float = 0.02,
    sample_spacing: float = 0.01,
    compress_same_flag_runs: bool = True,
):
    _, cp = _import_optix_modules()
    input_prep_t0 = time.perf_counter()
    if not isinstance(ray_origins, cp.ndarray) or ray_origins.ndim != 2:
        raise ValueError("OptiX ray origins must be a CuPy array with shape (R, 3).")
    num_rays = int(ray_origins.shape[0])
    ray_origins_gpu = _to_cupy_array(cp, ray_origins, shape=(num_rays, 3), name="origins")
    ray_directions_gpu = _to_cupy_array(cp, ray_directions, shape=(num_rays, 3), name="directions")
    ray_tmins_gpu = _to_cupy_array(cp, ray_tmins, shape=(num_rays,), name="t_near")
    ray_tmaxs_gpu = _to_cupy_array(cp, ray_tmaxs, shape=(num_rays,), name="t_far")
    input_prep_ms = (time.perf_counter() - input_prep_t0) * 1000.0

    depth_buffer = -1 * cp.ones((num_rays,), dtype=np.float32)
    hit_t_buffer = -1 * cp.ones((num_rays, n_depth_peeling_max), dtype=np.float32)
    hit_count_buffer = cp.zeros((num_rays,), dtype=np.int32)
    erosion_first_t_buffer = -1 * cp.ones((num_rays,), dtype=np.float32)
    dilation_hit_flag_buffer = cp.zeros((num_rays, n_depth_peeling_max), dtype=np.int32)
    max_intervals = (int(n_depth_peeling_max) + 1) // 2
    interval_start_buffer = -1 * cp.ones((num_rays, max_intervals), dtype=np.float32)
    interval_end_buffer = -1 * cp.ones((num_rays, max_intervals), dtype=np.float32)
    interval_count_buffer = cp.zeros((num_rays,), dtype=np.int32)
    sample_count_buffer = cp.zeros((num_rays,), dtype=np.int32)
    launch_params = _create_launch_params(
        dilation_gas=dilation_gas,
        erosion_gas=erosion_gas,
        depth_buffer=depth_buffer,
        hit_t_buffer=hit_t_buffer,
        hit_count_buffer=hit_count_buffer,
        image_width=num_rays,
        image_height=1,
        max_hits=n_depth_peeling_max,
        ray_origins=ray_origins_gpu,
        ray_directions=ray_directions_gpu,
        ray_tmins=ray_tmins_gpu,
        ray_tmaxs=ray_tmaxs_gpu,
        erosion_first_t_buffer=erosion_first_t_buffer,
        dilation_hit_flag_buffer=dilation_hit_flag_buffer,
        interval_start_buffer=interval_start_buffer,
        interval_end_buffer=interval_end_buffer,
        interval_count_buffer=interval_count_buffer,
        sample_count_buffer=sample_count_buffer,
        max_intervals=max_intervals,
        n_sample_max=int(n_sample_max),
        min_interval_size=float(min_interval_size),
        single_sample_threshold=float(single_sample_threshold),
        sample_spacing=float(sample_spacing),
        compress_same_flag_runs=bool(compress_same_flag_runs),
    )
    stream = cp.cuda.get_current_stream()
    optix_launch_t0 = time.perf_counter()
    pipeline.launch(sbt, dimensions=(num_rays, 1), params=launch_params, stream=stream)
    if blocking_launch:
        stream.synchronize()
    optix_launch_ms = (time.perf_counter() - optix_launch_t0) * 1000.0
    launch_profile = {
        "interop_zero_copy": True,
        "trace_pack_metadata": 1.0,
        "input_prep_ms": float(input_prep_ms),
        "optix_device_ms": float(optix_launch_ms),
        "trace_launch_block_ms": float(optix_launch_ms),
        "trace_launch_blocking": float(1.0 if blocking_launch else 0.0),
    }
    metadata_outputs = (interval_start_buffer, interval_end_buffer, interval_count_buffer, sample_count_buffer)
    return (
        depth_buffer,
        hit_t_buffer,
        hit_count_buffer,
        erosion_first_t_buffer,
        dilation_hit_flag_buffer,
        metadata_outputs,
        launch_profile,
    )
