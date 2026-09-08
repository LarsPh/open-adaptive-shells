from __future__ import annotations

from functools import lru_cache
import os
from typing import Any

import torch


_CPP_SOURCE = r"""
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cstdint>
#include <tuple>

uint64_t register_pbo(int64_t pbo_id);
void unregister_pbo(uint64_t handle);
std::tuple<double, double, double> copy_rgb_to_pbo(uint64_t handle, at::Tensor rgb, int64_t width, int64_t height);
std::tuple<double, double, double> copy_rgb_ptrs_to_pbo(
    uint64_t handle,
    uint64_t r_ptr,
    uint64_t g_ptr,
    uint64_t b_ptr,
    int64_t width,
    int64_t height);
std::tuple<double, double, double> copy_scalar_to_pbo(
    uint64_t handle,
    at::Tensor values,
    int64_t width,
    int64_t height,
    double vmin,
    double vmax,
    int64_t colormap_id);
std::tuple<double, double, double> copy_scalar_ptr_to_pbo(
    uint64_t handle,
    uint64_t values_ptr,
    int64_t width,
    int64_t height,
    double vmin,
    double vmax,
    int64_t colormap_id);
std::tuple<double, double, double> copy_vector_to_pbo(
    uint64_t handle,
    at::Tensor vectors,
    int64_t width,
    int64_t height,
    bool normalize);
std::tuple<double, double, double> copy_vector_ptrs_to_pbo(
    uint64_t handle,
    uint64_t x_ptr,
    uint64_t y_ptr,
    uint64_t z_ptr,
    int64_t width,
    int64_t height,
    bool normalize);
at::Tensor pack_rgb_to_rgba8(at::Tensor rgb, int64_t width, int64_t height);
at::Tensor pack_rgb_ptrs_to_rgba8(uint64_t r_ptr, uint64_t g_ptr, uint64_t b_ptr, int64_t width, int64_t height);
at::Tensor pack_scalar_to_rgba8(at::Tensor values, int64_t width, int64_t height, double vmin, double vmax, int64_t colormap_id);
at::Tensor pack_vector_to_rgba8(at::Tensor vectors, int64_t width, int64_t height, bool normalize);
"""


_CUDA_SOURCE = r"""
#ifdef _WIN32
#include <windows.h>
#endif
#include <GL/gl.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <tuple>

namespace {

inline void check_cuda(cudaError_t err, const char* expr) {
    if (err != cudaSuccess) {
        std::ostringstream oss;
        oss << expr << " failed: " << cudaGetErrorString(err);
        throw std::runtime_error(oss.str());
    }
}

inline cudaGraphicsResource_t as_resource(uint64_t handle) {
    return reinterpret_cast<cudaGraphicsResource_t>(static_cast<uintptr_t>(handle));
}

// Own CUDA resources across all throwing launch and timing operations.
class CudaEvent {
public:
    CudaEvent() { check_cuda(cudaEventCreate(&event_), "cudaEventCreate"); }
    ~CudaEvent() noexcept {
        if (event_ != nullptr) {
            cudaEventDestroy(event_);
        }
    }
    CudaEvent(const CudaEvent&) = delete;
    CudaEvent& operator=(const CudaEvent&) = delete;
    operator cudaEvent_t() const noexcept { return event_; }
    void close() {
        check_cuda(cudaEventDestroy(event_), "cudaEventDestroy");
        event_ = nullptr;
    }
private:
    cudaEvent_t event_ = nullptr;
};

class MappedPbo {
public:
    MappedPbo(cudaGraphicsResource_t resource, cudaStream_t stream)
        : resource_(resource), stream_(stream) {
        check_cuda(cudaGraphicsMapResources(1, &resource_, stream_), "cudaGraphicsMapResources");
    }
    ~MappedPbo() noexcept {
        if (mapped_) {
            cudaGraphicsUnmapResources(1, &resource_, stream_);
        }
    }
    MappedPbo(const MappedPbo&) = delete;
    MappedPbo& operator=(const MappedPbo&) = delete;
    void close() {
        check_cuda(cudaGraphicsUnmapResources(1, &resource_, stream_), "cudaGraphicsUnmapResources");
        mapped_ = false;
    }
private:
    cudaGraphicsResource_t resource_;
    cudaStream_t stream_;
    bool mapped_ = true;
};

__device__ __forceinline__ unsigned char to_u8(float value) {
    value = fminf(fmaxf(value, 0.0f), 1.0f);
    return static_cast<unsigned char>(value * 255.0f + 0.5f);
}

__device__ __forceinline__ uchar4 jet(float x) {
    x = fminf(fmaxf(x, 0.0f), 1.0f);
    float r = fminf(fmaxf(1.5f - fabsf(4.0f * x - 3.0f), 0.0f), 1.0f);
    float g = fminf(fmaxf(1.5f - fabsf(4.0f * x - 2.0f), 0.0f), 1.0f);
    float b = fminf(fmaxf(1.5f - fabsf(4.0f * x - 1.0f), 0.0f), 1.0f);
    return make_uchar4(to_u8(r), to_u8(g), to_u8(b), 255);
}

__global__ void pack_rgb_kernel(const float* __restrict__ rgb, uchar4* __restrict__ out, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const float* src = rgb + idx * 3;
    out[idx] = make_uchar4(to_u8(src[0]), to_u8(src[1]), to_u8(src[2]), 255);
}

__global__ void pack_rgb_channels_kernel(
    const float* __restrict__ r,
    const float* __restrict__ g,
    const float* __restrict__ b,
    uchar4* __restrict__ out,
    int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    out[idx] = make_uchar4(to_u8(r[idx]), to_u8(g[idx]), to_u8(b[idx]), 255);
}

__global__ void pack_scalar_kernel(
    const float* __restrict__ values,
    uchar4* __restrict__ out,
    int total,
    float vmin,
    float vmax,
    int colormap_id) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    float denom = fmaxf(vmax - vmin, 1e-8f);
    float x = (values[idx] - vmin) / denom;
    if (colormap_id == 0) {
        out[idx] = jet(x);
    } else {
        unsigned char y = to_u8(x);
        out[idx] = make_uchar4(y, y, y, 255);
    }
}

__global__ void pack_vector_kernel(const float* __restrict__ vectors, uchar4* __restrict__ out, int total, bool normalize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    const float* src = vectors + idx * 3;
    float x = src[0];
    float y = src[1];
    float z = src[2];
    if (normalize) {
        float inv_norm = rsqrtf(fmaxf(x * x + y * y + z * z, 1e-20f));
        x *= inv_norm;
        y *= inv_norm;
        z *= inv_norm;
    }
    out[idx] = make_uchar4(to_u8(x * 0.5f + 0.5f), to_u8(y * 0.5f + 0.5f), to_u8(z * 0.5f + 0.5f), 255);
}

__global__ void pack_vector_channels_kernel(
    const float* __restrict__ x_ptr,
    const float* __restrict__ y_ptr,
    const float* __restrict__ z_ptr,
    uchar4* __restrict__ out,
    int total,
    bool normalize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) {
        return;
    }
    float x = x_ptr[idx];
    float y = y_ptr[idx];
    float z = z_ptr[idx];
    if (normalize) {
        float inv_norm = rsqrtf(fmaxf(x * x + y * y + z * z, 1e-20f));
        x *= inv_norm;
        y *= inv_norm;
        z *= inv_norm;
    }
    out[idx] = make_uchar4(to_u8(x * 0.5f + 0.5f), to_u8(y * 0.5f + 0.5f), to_u8(z * 0.5f + 0.5f), 255);
}

inline const float* as_float_ptr(uint64_t ptr) {
    return reinterpret_cast<const float*>(static_cast<uintptr_t>(ptr));
}

std::tuple<double, double, double> launch_rgb_channels_to_pointer(
    uchar4* out,
    uint64_t r_ptr,
    uint64_t g_ptr,
    uint64_t b_ptr,
    int64_t width,
    int64_t height,
    cudaStream_t stream) {
    int64_t total = width * height;
    CudaEvent start_event;
    CudaEvent stop_event;
    check_cuda(cudaEventRecord(start_event, stream), "cudaEventRecord(start)");
    int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    pack_rgb_channels_kernel<<<grid, block, 0, stream>>>(
        as_float_ptr(r_ptr),
        as_float_ptr(g_ptr),
        as_float_ptr(b_ptr),
        out,
        static_cast<int>(total));
    check_cuda(cudaGetLastError(), "pack_rgb_channels_kernel");
    check_cuda(cudaEventRecord(stop_event, stream), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)");
    float kernel_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&kernel_ms, start_event, stop_event), "cudaEventElapsedTime");
    start_event.close();
    stop_event.close();
    return std::make_tuple(0.0, static_cast<double>(kernel_ms), 0.0);
}

std::tuple<double, double, double> launch_scalar_ptr_to_pointer(
    uchar4* out,
    uint64_t values_ptr,
    int64_t width,
    int64_t height,
    double vmin,
    double vmax,
    int64_t colormap_id,
    cudaStream_t stream) {
    int64_t total = width * height;
    CudaEvent start_event;
    CudaEvent stop_event;
    check_cuda(cudaEventRecord(start_event, stream), "cudaEventRecord(start)");
    int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    pack_scalar_kernel<<<grid, block, 0, stream>>>(
        as_float_ptr(values_ptr),
        out,
        static_cast<int>(total),
        static_cast<float>(vmin),
        static_cast<float>(vmax),
        static_cast<int>(colormap_id));
    check_cuda(cudaGetLastError(), "pack_scalar_ptr_kernel");
    check_cuda(cudaEventRecord(stop_event, stream), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)");
    float kernel_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&kernel_ms, start_event, stop_event), "cudaEventElapsedTime");
    start_event.close();
    stop_event.close();
    return std::make_tuple(0.0, static_cast<double>(kernel_ms), 0.0);
}

std::tuple<double, double, double> launch_vector_channels_to_pointer(
    uchar4* out,
    uint64_t x_ptr,
    uint64_t y_ptr,
    uint64_t z_ptr,
    int64_t width,
    int64_t height,
    bool normalize,
    cudaStream_t stream) {
    int64_t total = width * height;
    CudaEvent start_event;
    CudaEvent stop_event;
    check_cuda(cudaEventRecord(start_event, stream), "cudaEventRecord(start)");
    int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    pack_vector_channels_kernel<<<grid, block, 0, stream>>>(
        as_float_ptr(x_ptr),
        as_float_ptr(y_ptr),
        as_float_ptr(z_ptr),
        out,
        static_cast<int>(total),
        normalize);
    check_cuda(cudaGetLastError(), "pack_vector_channels_kernel");
    check_cuda(cudaEventRecord(stop_event, stream), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)");
    float kernel_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&kernel_ms, start_event, stop_event), "cudaEventElapsedTime");
    start_event.close();
    stop_event.close();
    return std::make_tuple(0.0, static_cast<double>(kernel_ms), 0.0);
}

std::tuple<double, double, double> launch_rgb_to_pointer(
    uchar4* out,
    at::Tensor rgb,
    int64_t width,
    int64_t height,
    cudaStream_t stream) {
    auto contiguous = rgb.contiguous();
    int64_t total = width * height;
    if (!contiguous.is_cuda() || contiguous.scalar_type() != at::kFloat) {
        throw std::runtime_error("RGB tensor must be CUDA float32.");
    }
    if (contiguous.numel() < total * 3) {
        throw std::runtime_error("RGB tensor is smaller than width * height * 3.");
    }
    CudaEvent start_event;
    CudaEvent stop_event;
    check_cuda(cudaEventRecord(start_event, stream), "cudaEventRecord(start)");
    int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    pack_rgb_kernel<<<grid, block, 0, stream>>>(contiguous.data_ptr<float>(), out, static_cast<int>(total));
    check_cuda(cudaGetLastError(), "pack_rgb_kernel");
    check_cuda(cudaEventRecord(stop_event, stream), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)");
    float kernel_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&kernel_ms, start_event, stop_event), "cudaEventElapsedTime");
    start_event.close();
    stop_event.close();
    return std::make_tuple(0.0, static_cast<double>(kernel_ms), 0.0);
}

std::tuple<double, double, double> launch_scalar_to_pointer(
    uchar4* out,
    at::Tensor values,
    int64_t width,
    int64_t height,
    double vmin,
    double vmax,
    int64_t colormap_id,
    cudaStream_t stream) {
    auto contiguous = values.contiguous();
    int64_t total = width * height;
    if (!contiguous.is_cuda() || contiguous.scalar_type() != at::kFloat) {
        throw std::runtime_error("Scalar tensor must be CUDA float32.");
    }
    if (contiguous.numel() < total) {
        throw std::runtime_error("Scalar tensor is smaller than width * height.");
    }
    CudaEvent start_event;
    CudaEvent stop_event;
    check_cuda(cudaEventRecord(start_event, stream), "cudaEventRecord(start)");
    int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    pack_scalar_kernel<<<grid, block, 0, stream>>>(
        contiguous.data_ptr<float>(),
        out,
        static_cast<int>(total),
        static_cast<float>(vmin),
        static_cast<float>(vmax),
        static_cast<int>(colormap_id));
    check_cuda(cudaGetLastError(), "pack_scalar_kernel");
    check_cuda(cudaEventRecord(stop_event, stream), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)");
    float kernel_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&kernel_ms, start_event, stop_event), "cudaEventElapsedTime");
    start_event.close();
    stop_event.close();
    return std::make_tuple(0.0, static_cast<double>(kernel_ms), 0.0);
}

std::tuple<double, double, double> launch_vector_to_pointer(
    uchar4* out,
    at::Tensor vectors,
    int64_t width,
    int64_t height,
    bool normalize,
    cudaStream_t stream) {
    auto contiguous = vectors.contiguous();
    int64_t total = width * height;
    if (!contiguous.is_cuda() || contiguous.scalar_type() != at::kFloat) {
        throw std::runtime_error("Vector tensor must be CUDA float32.");
    }
    if (contiguous.numel() < total * 3) {
        throw std::runtime_error("Vector tensor is smaller than width * height * 3.");
    }
    CudaEvent start_event;
    CudaEvent stop_event;
    check_cuda(cudaEventRecord(start_event, stream), "cudaEventRecord(start)");
    int block = 256;
    int grid = static_cast<int>((total + block - 1) / block);
    pack_vector_kernel<<<grid, block, 0, stream>>>(contiguous.data_ptr<float>(), out, static_cast<int>(total), normalize);
    check_cuda(cudaGetLastError(), "pack_vector_kernel");
    check_cuda(cudaEventRecord(stop_event, stream), "cudaEventRecord(stop)");
    check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)");
    float kernel_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&kernel_ms, start_event, stop_event), "cudaEventElapsedTime");
    start_event.close();
    stop_event.close();
    return std::make_tuple(0.0, static_cast<double>(kernel_ms), 0.0);
}

} // namespace

uint64_t register_pbo(int64_t pbo_id) {
    cudaGraphicsResource_t resource = nullptr;
    check_cuda(
        cudaGraphicsGLRegisterBuffer(&resource, static_cast<GLuint>(pbo_id), cudaGraphicsRegisterFlagsWriteDiscard),
        "cudaGraphicsGLRegisterBuffer");
    return static_cast<uint64_t>(reinterpret_cast<uintptr_t>(resource));
}

void unregister_pbo(uint64_t handle) {
    if (handle == 0) {
        return;
    }
    auto resource = as_resource(handle);
    check_cuda(cudaGraphicsUnregisterResource(resource), "cudaGraphicsUnregisterResource");
}

std::tuple<double, double, double> copy_rgb_to_pbo(uint64_t handle, at::Tensor rgb, int64_t width, int64_t height) {
    auto resource = as_resource(handle);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto map_start = std::chrono::high_resolution_clock::now();
    MappedPbo mapping(resource, stream);
    uchar4* ptr = nullptr;
    size_t size_bytes = 0;
    check_cuda(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&ptr), &size_bytes, resource), "cudaGraphicsResourceGetMappedPointer");
    auto map_end = std::chrono::high_resolution_clock::now();
    if (size_bytes < static_cast<size_t>(width * height * 4)) {
        throw std::runtime_error("Mapped PBO is too small for requested frame.");
    }
    auto timings = launch_rgb_to_pointer(ptr, rgb, width, height, stream);
    auto unmap_start = std::chrono::high_resolution_clock::now();
    mapping.close();
    auto unmap_end = std::chrono::high_resolution_clock::now();
    double map_ms = std::chrono::duration<double, std::milli>(map_end - map_start).count();
    double unmap_ms = std::chrono::duration<double, std::milli>(unmap_end - unmap_start).count();
    return std::make_tuple(map_ms, std::get<1>(timings), unmap_ms);
}

std::tuple<double, double, double> copy_rgb_ptrs_to_pbo(
    uint64_t handle,
    uint64_t r_ptr,
    uint64_t g_ptr,
    uint64_t b_ptr,
    int64_t width,
    int64_t height) {
    auto resource = as_resource(handle);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto map_start = std::chrono::high_resolution_clock::now();
    MappedPbo mapping(resource, stream);
    uchar4* ptr = nullptr;
    size_t size_bytes = 0;
    check_cuda(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&ptr), &size_bytes, resource), "cudaGraphicsResourceGetMappedPointer");
    auto map_end = std::chrono::high_resolution_clock::now();
    if (size_bytes < static_cast<size_t>(width * height * 4)) {
        throw std::runtime_error("Mapped PBO is too small for requested frame.");
    }
    auto timings = launch_rgb_channels_to_pointer(ptr, r_ptr, g_ptr, b_ptr, width, height, stream);
    auto unmap_start = std::chrono::high_resolution_clock::now();
    mapping.close();
    auto unmap_end = std::chrono::high_resolution_clock::now();
    double map_ms = std::chrono::duration<double, std::milli>(map_end - map_start).count();
    double unmap_ms = std::chrono::duration<double, std::milli>(unmap_end - unmap_start).count();
    return std::make_tuple(map_ms, std::get<1>(timings), unmap_ms);
}

std::tuple<double, double, double> copy_scalar_to_pbo(
    uint64_t handle,
    at::Tensor values,
    int64_t width,
    int64_t height,
    double vmin,
    double vmax,
    int64_t colormap_id) {
    auto resource = as_resource(handle);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto map_start = std::chrono::high_resolution_clock::now();
    MappedPbo mapping(resource, stream);
    uchar4* ptr = nullptr;
    size_t size_bytes = 0;
    check_cuda(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&ptr), &size_bytes, resource), "cudaGraphicsResourceGetMappedPointer");
    auto map_end = std::chrono::high_resolution_clock::now();
    if (size_bytes < static_cast<size_t>(width * height * 4)) {
        throw std::runtime_error("Mapped PBO is too small for requested frame.");
    }
    auto timings = launch_scalar_to_pointer(ptr, values, width, height, vmin, vmax, colormap_id, stream);
    auto unmap_start = std::chrono::high_resolution_clock::now();
    mapping.close();
    auto unmap_end = std::chrono::high_resolution_clock::now();
    double map_ms = std::chrono::duration<double, std::milli>(map_end - map_start).count();
    double unmap_ms = std::chrono::duration<double, std::milli>(unmap_end - unmap_start).count();
    return std::make_tuple(map_ms, std::get<1>(timings), unmap_ms);
}

std::tuple<double, double, double> copy_scalar_ptr_to_pbo(
    uint64_t handle,
    uint64_t values_ptr,
    int64_t width,
    int64_t height,
    double vmin,
    double vmax,
    int64_t colormap_id) {
    auto resource = as_resource(handle);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto map_start = std::chrono::high_resolution_clock::now();
    MappedPbo mapping(resource, stream);
    uchar4* ptr = nullptr;
    size_t size_bytes = 0;
    check_cuda(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&ptr), &size_bytes, resource), "cudaGraphicsResourceGetMappedPointer");
    auto map_end = std::chrono::high_resolution_clock::now();
    if (size_bytes < static_cast<size_t>(width * height * 4)) {
        throw std::runtime_error("Mapped PBO is too small for requested frame.");
    }
    auto timings = launch_scalar_ptr_to_pointer(ptr, values_ptr, width, height, vmin, vmax, colormap_id, stream);
    auto unmap_start = std::chrono::high_resolution_clock::now();
    mapping.close();
    auto unmap_end = std::chrono::high_resolution_clock::now();
    double map_ms = std::chrono::duration<double, std::milli>(map_end - map_start).count();
    double unmap_ms = std::chrono::duration<double, std::milli>(unmap_end - unmap_start).count();
    return std::make_tuple(map_ms, std::get<1>(timings), unmap_ms);
}

std::tuple<double, double, double> copy_vector_to_pbo(
    uint64_t handle,
    at::Tensor vectors,
    int64_t width,
    int64_t height,
    bool normalize) {
    auto resource = as_resource(handle);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto map_start = std::chrono::high_resolution_clock::now();
    MappedPbo mapping(resource, stream);
    uchar4* ptr = nullptr;
    size_t size_bytes = 0;
    check_cuda(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&ptr), &size_bytes, resource), "cudaGraphicsResourceGetMappedPointer");
    auto map_end = std::chrono::high_resolution_clock::now();
    if (size_bytes < static_cast<size_t>(width * height * 4)) {
        throw std::runtime_error("Mapped PBO is too small for requested frame.");
    }
    auto timings = launch_vector_to_pointer(ptr, vectors, width, height, normalize, stream);
    auto unmap_start = std::chrono::high_resolution_clock::now();
    mapping.close();
    auto unmap_end = std::chrono::high_resolution_clock::now();
    double map_ms = std::chrono::duration<double, std::milli>(map_end - map_start).count();
    double unmap_ms = std::chrono::duration<double, std::milli>(unmap_end - unmap_start).count();
    return std::make_tuple(map_ms, std::get<1>(timings), unmap_ms);
}

std::tuple<double, double, double> copy_vector_ptrs_to_pbo(
    uint64_t handle,
    uint64_t x_ptr,
    uint64_t y_ptr,
    uint64_t z_ptr,
    int64_t width,
    int64_t height,
    bool normalize) {
    auto resource = as_resource(handle);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    auto map_start = std::chrono::high_resolution_clock::now();
    MappedPbo mapping(resource, stream);
    uchar4* ptr = nullptr;
    size_t size_bytes = 0;
    check_cuda(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&ptr), &size_bytes, resource), "cudaGraphicsResourceGetMappedPointer");
    auto map_end = std::chrono::high_resolution_clock::now();
    if (size_bytes < static_cast<size_t>(width * height * 4)) {
        throw std::runtime_error("Mapped PBO is too small for requested frame.");
    }
    auto timings = launch_vector_channels_to_pointer(ptr, x_ptr, y_ptr, z_ptr, width, height, normalize, stream);
    auto unmap_start = std::chrono::high_resolution_clock::now();
    mapping.close();
    auto unmap_end = std::chrono::high_resolution_clock::now();
    double map_ms = std::chrono::duration<double, std::milli>(map_end - map_start).count();
    double unmap_ms = std::chrono::duration<double, std::milli>(unmap_end - unmap_start).count();
    return std::make_tuple(map_ms, std::get<1>(timings), unmap_ms);
}

at::Tensor pack_rgb_to_rgba8(at::Tensor rgb, int64_t width, int64_t height) {
    auto out = at::empty({height, width, 4}, at::TensorOptions().device(rgb.device()).dtype(at::kByte));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_rgb_to_pointer(reinterpret_cast<uchar4*>(out.data_ptr<unsigned char>()), rgb, width, height, stream);
    return out;
}

at::Tensor pack_rgb_ptrs_to_rgba8(uint64_t r_ptr, uint64_t g_ptr, uint64_t b_ptr, int64_t width, int64_t height) {
    auto out = at::empty({height, width, 4}, at::TensorOptions().device(at::kCUDA).dtype(at::kByte));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_rgb_channels_to_pointer(reinterpret_cast<uchar4*>(out.data_ptr<unsigned char>()), r_ptr, g_ptr, b_ptr, width, height, stream);
    return out;
}

at::Tensor pack_scalar_to_rgba8(at::Tensor values, int64_t width, int64_t height, double vmin, double vmax, int64_t colormap_id) {
    auto out = at::empty({height, width, 4}, at::TensorOptions().device(values.device()).dtype(at::kByte));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_scalar_to_pointer(reinterpret_cast<uchar4*>(out.data_ptr<unsigned char>()), values, width, height, vmin, vmax, colormap_id, stream);
    return out;
}

at::Tensor pack_vector_to_rgba8(at::Tensor vectors, int64_t width, int64_t height, bool normalize) {
    auto out = at::empty({height, width, 4}, at::TensorOptions().device(vectors.device()).dtype(at::kByte));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    launch_vector_to_pointer(reinterpret_cast<uchar4*>(out.data_ptr<unsigned char>()), vectors, width, height, normalize, stream);
    return out;
}
"""


@lru_cache(maxsize=1)
def load_cuda_gl_extension() -> Any:
    from torch.utils.cpp_extension import load_inline

    verbose = bool(int(os.environ.get("ADASHELL_CUDA_GL_VERBOSE", "0")))
    print(
        "Loading CUDA/OpenGL interop extension "
        f"(verbose={int(verbose)}, cl={_which('cl')}, nvcc={_which('nvcc')})...",
        flush=True,
    )
    try:
        return load_inline(
            name="adashell_cuda_gl_interop_v3",
            cpp_sources=[_CPP_SOURCE],
            cuda_sources=[_CUDA_SOURCE],
            functions=[
                "register_pbo",
                "unregister_pbo",
                "copy_rgb_to_pbo",
                "copy_rgb_ptrs_to_pbo",
                "copy_scalar_to_pbo",
                "copy_scalar_ptr_to_pbo",
                "copy_vector_to_pbo",
                "copy_vector_ptrs_to_pbo",
                "pack_rgb_to_rgba8",
                "pack_rgb_ptrs_to_rgba8",
                "pack_scalar_to_rgba8",
                "pack_vector_to_rgba8",
            ],
            extra_cflags=["/std:c++17"] if _is_windows() else ["-std=c++17"],
            extra_cuda_cflags=["--std=c++17"],
            verbose=verbose,
            with_cuda=True,
        )
    except Exception as exc:
        if _is_windows():
            raise RuntimeError(
                "Failed to JIT-compile the CUDA/OpenGL interop extension. "
                "On Windows this requires running from a Visual Studio Developer Prompt, "
                "or otherwise having cl.exe and nvcc.exe visible on PATH. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc
        raise


def _is_windows() -> bool:
    import sys

    return sys.platform.startswith("win")


def _which(name: str) -> str | None:
    import shutil

    return shutil.which(name)


def print_build_environment() -> None:
    import os
    import shutil

    keys = [
        "PATH",
        "Path",
        "CUDA_PATH",
        "VCToolsInstallDir",
        "VSINSTALLDIR",
        "INCLUDE",
        "LIB",
        "DISTUTILS_USE_SDK",
        "CXX",
        "CC",
    ]
    print("adashell CUDA/GL build environment:")
    print(f"  cl: {shutil.which('cl')}")
    print(f"  nvcc: {shutil.which('nvcc')}")
    for key in keys:
        value = os.environ.get(key)
        if value is None:
            print(f"  {key}: <unset>")
        elif key.upper() in {"PATH", "INCLUDE", "LIB"}:
            parts = value.split(os.pathsep)
            print(f"  {key}: {len(parts)} entries")
            for item in parts[:8]:
                print(f"    - {item}")
            if len(parts) > 8:
                print("    ...")
        else:
            print(f"  {key}: {value}")


def pack_rgb_to_rgba8(rgb: torch.Tensor, width: int, height: int) -> torch.Tensor:
    return load_cuda_gl_extension().pack_rgb_to_rgba8(rgb.contiguous(), int(width), int(height))


def pack_rgb_ptrs_to_rgba8(r_ptr: int, g_ptr: int, b_ptr: int, width: int, height: int) -> torch.Tensor:
    return load_cuda_gl_extension().pack_rgb_ptrs_to_rgba8(
        int(r_ptr),
        int(g_ptr),
        int(b_ptr),
        int(width),
        int(height),
    )


def pack_scalar_to_rgba8(
    values: torch.Tensor,
    width: int,
    height: int,
    vmin: float,
    vmax: float,
    colormap_id: int = 0,
) -> torch.Tensor:
    return load_cuda_gl_extension().pack_scalar_to_rgba8(
        values.contiguous(),
        int(width),
        int(height),
        float(vmin),
        float(vmax),
        int(colormap_id),
    )


def pack_vector_to_rgba8(vectors: torch.Tensor, width: int, height: int, *, normalize: bool = True) -> torch.Tensor:
    return load_cuda_gl_extension().pack_vector_to_rgba8(
        vectors.contiguous(),
        int(width),
        int(height),
        bool(normalize),
    )
