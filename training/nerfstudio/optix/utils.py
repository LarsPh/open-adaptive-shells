import optix as ox
import logging
import trimesh
import numpy as np
import cupy as cp
import os
import torch
from torch.utils.dlpack import from_dlpack


def _torch_to_cupy(tensor: torch.Tensor, dtype: torch.dtype) -> cp.ndarray:
    tensor = tensor.detach()
    if not tensor.is_cuda:
        raise ValueError("Expected CUDA tensor for zero-copy OptiX interop.")
    tensor = tensor.to(dtype=dtype).contiguous()
    return cp.from_dlpack(tensor)


def _to_cupy_array(array, dtype, use_torch_interop: bool):
    if use_torch_interop:
        return _torch_to_cupy(array, dtype)
    return cp.asarray(np.ascontiguousarray(array.astype(cp.dtype(dtype).name)))


def _cupy_to_torch(array: cp.ndarray) -> torch.Tensor:
    return from_dlpack(array)


def _create_launch_params(
    dilation_gas,
    erosion_gas,
    depth_buffer,
    hit_t_buffer,
    hit_count_buffer,
    image_width,
    image_height,
    max_hits,
    cam_params=None,
    ray_origins=None,
    ray_directions=None,
    ray_tmins=None,
    ray_tmaxs=None,
    erosion_first_t_buffer=None,
    dilation_hit_flag_buffer=None,
):
    params_tmp = [
        ('u8', 'depth_buffer'),
        ('u8', 'hit_t_buffer'),
        ('u8', 'hit_count_buffer'),
        ('u4', 'image_width'),
        ('u4', 'image_height'),
        ('u4', 'max_hits'),
        ('3f4', 'cam_eye'),
        ('3f4', 'cam_u'),
        ('3f4', 'cam_v'),
        ('3f4', 'cam_w'),
        ('u8', 'ray_origins_buffer'),
        ('u8', 'ray_directions_buffer'),
        ('u8', 'ray_tmin_buffer'),
        ('u8', 'ray_tmax_buffer'),
        ('u8', 'dilation_trav_handle'),
        ('u8', 'erosion_trav_handle'),
        ('u8', 'erosion_first_t_buffer'),
        ('u8', 'dilation_hit_flag_buffer'),
    ]
    launch_params = ox.LaunchParamsRecord(
        names=[p[1] for p in params_tmp],
        formats=[p[0] for p in params_tmp]
    )
    launch_params['depth_buffer'] = depth_buffer.data.ptr
    launch_params['hit_t_buffer'] = hit_t_buffer.data.ptr
    launch_params['hit_count_buffer'] = hit_count_buffer.data.ptr
    launch_params['image_width'] = int(image_width)
    launch_params['image_height'] = int(image_height)
    launch_params['max_hits'] = int(max_hits)
    zero3 = np.zeros((3,), dtype=np.float32)
    launch_params['cam_eye'] = cam_params["cam_eye"] if cam_params is not None else zero3
    launch_params['cam_u'] = cam_params["cam_u"] if cam_params is not None else zero3
    launch_params['cam_v'] = cam_params["cam_v"] if cam_params is not None else zero3
    launch_params['cam_w'] = cam_params["cam_w"] if cam_params is not None else zero3
    launch_params['ray_origins_buffer'] = 0 if ray_origins is None else ray_origins.data.ptr
    launch_params['ray_directions_buffer'] = 0 if ray_directions is None else ray_directions.data.ptr
    launch_params['ray_tmin_buffer'] = 0 if ray_tmins is None else ray_tmins.data.ptr
    launch_params['ray_tmax_buffer'] = 0 if ray_tmaxs is None else ray_tmaxs.data.ptr
    launch_params['dilation_trav_handle'] = dilation_gas.handle
    launch_params['erosion_trav_handle'] = erosion_gas.handle
    launch_params['erosion_first_t_buffer'] = 0 if erosion_first_t_buffer is None else erosion_first_t_buffer.data.ptr
    launch_params['dilation_hit_flag_buffer'] = 0 if dilation_hit_flag_buffer is None else dilation_hit_flag_buffer.data.ptr
    return launch_params

def create_context(log: logging.Logger):
    logger = ox.Logger(log)
    ctx = ox.DeviceContext(validation_mode=False, log_callback_function=logger, log_callback_level=4)
    ctx.cache_enabled = False
    return ctx

def create_acceleration_structure(
        ctx: ox.DeviceContext,
        mesh: trimesh.Trimesh):
    build_input = ox.BuildInputTriangleArray(
        vertex_buffers=mesh.vertices.astype(np.float32),
        index_buffer=mesh.faces.astype(np.uint32), 
        num_sbt_records=1,
        flags=[ox.GeometryFlags.NONE]
    )
    gas = ox.AccelerationStructure(ctx, build_input, compact=True, allow_update=False, random_vertex_access=True)
    return gas

def create_module(ctx, pipeline_opts, source_file):
    script_dir = os.path.dirname(__file__)
    try:
        cuda_src = os.path.join(script_dir, source_file)
    except:
        raise ValueError(f"Source file not found: {source_file}")

    compile_opts = ox.ModuleCompileOptions(debug_level=ox.CompileDebugLevel.DEFAULT , opt_level=ox.CompileOptimizationLevel.DEFAULT )
    module = ox.Module(ctx, cuda_src, compile_opts, pipeline_opts)
    return module

def create_program_groups(ctx, module):
    raygen_grp = ox.ProgramGroup.create_raygen(ctx, module, "__raygen__rg")
    miss_grp = ox.ProgramGroup.create_miss(ctx, module, "__miss__ms")
    hit_grp = ox.ProgramGroup.create_hitgroup(ctx, module,
                                            entry_function_CH="__closesthit__ch",
    )
                                            # entry_function_AH="__anyhit__ah")
    return raygen_grp, miss_grp, hit_grp

def create_pipeline(ctx, program_grps, pipeline_options,debug_level=ox.CompileDebugLevel.DEFAULT):
    link_opts = ox.PipelineLinkOptions(max_trace_depth=1, debug_level=debug_level)

    pipeline = ox.Pipeline(ctx, compile_options=pipeline_options, link_options=link_opts, program_groups=program_grps)
    pipeline.compute_stack_sizes(1,  # max_trace_depth
                                 0,  # max_cc_depth
                                 0)  # max_dc_depth
    return pipeline

def create_sbt(program_grps):
    raygen_grp, miss_grp, hit_grp = program_grps

    raygen_sbt = ox.SbtRecord(raygen_grp)
    miss_sbt = ox.SbtRecord(miss_grp)
    hit_sbt = ox.SbtRecord(hit_grp)
    sbt = ox.ShaderBindingTable(raygen_record=raygen_sbt, miss_records=miss_sbt, hitgroup_records=hit_sbt)
    return sbt


def launch_pipeline(pipeline, sbt, dilation_gas, erosion_gas, cam_params, **kwargs):
    height, width = cam_params["image_height"], cam_params["image_width"]
    depth_buffer = -1 * cp.ones((height, width), dtype=np.float32)
    n_peeling_max = kwargs.get("n_depth_peeling_max", 20)
    hit_t_buffer = -1 * cp.ones((height, width, n_peeling_max), dtype=np.float32)
    hit_count_buffer = cp.zeros((height, width), dtype=np.uint32)
    erosion_first_t_buffer = -1 * cp.ones((height, width), dtype=np.float32)
    dilation_hit_flag_buffer = cp.zeros((height, width, n_peeling_max), dtype=np.uint32)
    launch_params = _create_launch_params(
        dilation_gas=dilation_gas,
        erosion_gas=erosion_gas,
        depth_buffer=depth_buffer,
        hit_t_buffer=hit_t_buffer,
        hit_count_buffer=hit_count_buffer,
        image_width=width,
        image_height=height,
        max_hits=n_peeling_max,
        cam_params=cam_params,
        erosion_first_t_buffer=erosion_first_t_buffer,
        dilation_hit_flag_buffer=dilation_hit_flag_buffer,
    )
    # cupy stream
    stream = cp.cuda.Stream()
    pipeline.launch(sbt, dimensions=(width, height), params=launch_params, stream=stream)
    stream.synchronize()
    return cp.asnumpy(depth_buffer), cp.asnumpy(hit_t_buffer), cp.asnumpy(hit_count_buffer), cp.asnumpy(erosion_first_t_buffer), cp.asnumpy(dilation_hit_flag_buffer)


def launch_pipeline_ray_batch(
    pipeline,
    sbt,
    dilation_gas,
    erosion_gas,
    ray_origins,
    ray_directions,
    ray_tmins,
    ray_tmaxs,
    **kwargs,
):
    num_rays = int(ray_origins.shape[0])
    n_peeling_max = kwargs.get("n_depth_peeling_max", 20)
    return_torch = kwargs.get("return_torch", False)
    use_torch_interop = any(torch.is_tensor(x) for x in (ray_origins, ray_directions, ray_tmins, ray_tmaxs))

    ray_origins_gpu = _to_cupy_array(ray_origins, torch.float32 if use_torch_interop else np.float32, use_torch_interop)
    ray_directions_gpu = _to_cupy_array(
        ray_directions, torch.float32 if use_torch_interop else np.float32, use_torch_interop
    )
    ray_tmins_gpu = _to_cupy_array(ray_tmins, torch.float32 if use_torch_interop else np.float32, use_torch_interop)
    ray_tmaxs_gpu = _to_cupy_array(ray_tmaxs, torch.float32 if use_torch_interop else np.float32, use_torch_interop)
    depth_buffer = -1 * cp.ones((num_rays,), dtype=np.float32)
    hit_t_buffer = -1 * cp.ones((num_rays, n_peeling_max), dtype=np.float32)
    hit_count_buffer = cp.zeros((num_rays,), dtype=np.int32)
    erosion_first_t_buffer = -1 * cp.ones((num_rays,), dtype=np.float32)
    dilation_hit_flag_buffer = cp.zeros((num_rays, n_peeling_max), dtype=np.int32)
    launch_params = _create_launch_params(
        dilation_gas=dilation_gas,
        erosion_gas=erosion_gas,
        depth_buffer=depth_buffer,
        hit_t_buffer=hit_t_buffer,
        hit_count_buffer=hit_count_buffer,
        image_width=num_rays,
        image_height=1,
        max_hits=n_peeling_max,
        ray_origins=ray_origins_gpu,
        ray_directions=ray_directions_gpu,
        ray_tmins=ray_tmins_gpu,
        ray_tmaxs=ray_tmaxs_gpu,
        erosion_first_t_buffer=erosion_first_t_buffer,
        dilation_hit_flag_buffer=dilation_hit_flag_buffer,
    )
    if use_torch_interop:
        torch_stream = torch.cuda.current_stream(ray_origins.device)
        stream = cp.cuda.ExternalStream(torch_stream.cuda_stream)
    else:
        stream = cp.cuda.Stream()
    pipeline.launch(sbt, dimensions=(num_rays, 1), params=launch_params, stream=stream)
    stream.synchronize()
    launch_profile = {
        "interop_zero_copy": bool(use_torch_interop and return_torch),
    }
    if return_torch:
        return (
            _cupy_to_torch(depth_buffer),
            _cupy_to_torch(hit_t_buffer),
            _cupy_to_torch(hit_count_buffer),
            _cupy_to_torch(erosion_first_t_buffer),
            _cupy_to_torch(dilation_hit_flag_buffer),
            launch_profile,
        )
    return (
        cp.asnumpy(depth_buffer),
        cp.asnumpy(hit_t_buffer),
        cp.asnumpy(hit_count_buffer),
        cp.asnumpy(erosion_first_t_buffer),
        cp.asnumpy(dilation_hit_flag_buffer),
        launch_profile,
    )