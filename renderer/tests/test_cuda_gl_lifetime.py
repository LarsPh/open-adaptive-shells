"""Compile production resource guards with a CPU-only, fault-injecting CUDA API."""

import re
import shutil
import subprocess

import pytest

from adashell_renderer.cuda_gl_interop import _CUDA_SOURCE


def function_source(name):
    start = _CUDA_SOURCE.index("std::tuple<double, double, double> " + name + "(")
    end = _CUDA_SOURCE.index("\n}\n", start) + 3
    return _CUDA_SOURCE[start:end]


def test_every_launch_and_mapping_has_a_resource_guard():
    launches = re.findall(r"std::tuple<double, double, double> (launch_\w+)\(", _CUDA_SOURCE)
    copies = re.findall(r"std::tuple<double, double, double> (copy_\w+)\(", _CUDA_SOURCE)
    assert len(launches) == len(copies) == 6
    for name in launches:
        body = function_source(name)
        assert "CudaEvent start_event;" in body
        assert "CudaEvent stop_event;" in body
        assert "cudaEventCreate(" not in body
        assert "cudaEventDestroy(" not in body
    for name in copies:
        body = function_source(name)
        assert "MappedPbo mapping(resource, stream);" in body
        assert "mapping.close();" in body
        assert "cudaGraphicsMapResources(" not in body
        assert "cudaGraphicsUnmapResources(" not in body


_STUBS = r"""
#include <cassert>
#include <chrono>
#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <tuple>
using cudaError_t = int;
using cudaEvent_t = void*;
using cudaGraphicsResource_t = void*;
using cudaStream_t = void*;
constexpr int cudaSuccess = 0;
int calls = 0, fail_at = 0, events = 0, mappings = 0;
size_t mapped_size = 4096;
bool fails() { return ++calls == fail_at; }
const char* cudaGetErrorString(int) { return "injected failure"; }
int cudaEventCreate(cudaEvent_t* p) {
    if (fails()) return 1;
    *p = new int(0); ++events; return 0;
}
int cudaEventDestroy(cudaEvent_t p) {
    if (fails()) return 1;
    delete static_cast<int*>(p); --events; return 0;
}
int cudaEventRecord(cudaEvent_t, cudaStream_t) { return fails(); }
int cudaEventSynchronize(cudaEvent_t) { return fails(); }
int cudaEventElapsedTime(float* ms, cudaEvent_t, cudaEvent_t) {
    if (fails()) return 1;
    *ms = 1.0f; return 0;
}
int cudaGraphicsMapResources(int, cudaGraphicsResource_t*, cudaStream_t) {
    if (fails()) return 1;
    ++mappings; return 0;
}
int cudaGraphicsUnmapResources(int, cudaGraphicsResource_t*, cudaStream_t) {
    if (fails()) return 1;
    --mappings; return 0;
}
int cudaGraphicsResourceGetMappedPointer(void** p, size_t* size, cudaGraphicsResource_t) {
    if (fails()) return 1;
    *p = nullptr; *size = mapped_size; return 0;
}
int cudaGetLastError() { return fails(); }
namespace at {
constexpr int kFloat = 0;
struct Tensor {
    bool on_cuda = true;
    int dtype = kFloat;
    int count = 512;
    Tensor contiguous() const { return *this; }
    bool is_cuda() const { return on_cuda; }
    int scalar_type() const { return dtype; }
    int numel() const { return count; }
    template<class T> T* data_ptr() const { return nullptr; }
};
namespace cuda {
cudaStream_t getCurrentCUDAStream() { return nullptr; }
}}
struct uchar4 {};
const float* as_float_ptr(uint64_t p) { return reinterpret_cast<const float*>(p); }
template<class... T> void pack_rgb_channels_kernel(T...) {}
template<class... T> void pack_rgb_kernel(T...) {}
template<class... T> void pack_scalar_kernel(T...) {}
template<class... T> void pack_vector_channels_kernel(T...) {}
template<class... T> void pack_vector_kernel(T...) {}
"""

_MAIN = r"""
int main() {
    for (int mode = 0; mode < 6; ++mode) {
        auto run = [mode]() {
            if (mode == 0) copy_rgb_ptrs_to_pbo(1, 2, 3, 4, 8, 8);
            if (mode == 1) copy_scalar_ptr_to_pbo(1, 2, 8, 8, 0, 1, 0);
            if (mode == 2) copy_vector_ptrs_to_pbo(1, 2, 3, 4, 8, 8, true);
            if (mode == 3) copy_rgb_to_pbo(1, at::Tensor{}, 8, 8);
            if (mode == 4) copy_scalar_to_pbo(1, at::Tensor{}, 8, 8, 0, 1, 0);
            if (mode == 5) copy_vector_to_pbo(1, at::Tensor{}, 8, 8, true);
        };
        calls = 0; fail_at = 0;
        run();
        assert(events == 0 && mappings == 0);
        const int success_calls = calls;
        for (int failure = 1; failure <= success_calls; ++failure) {
            calls = 0; fail_at = failure;
            bool caught = false;
            try { run(); } catch (const std::runtime_error&) { caught = true; }
            assert(caught);
            assert(events == 0 && mappings == 0);
        }
        fail_at = 0; mapped_size = 0;
        bool caught = false;
        try { run(); } catch (const std::runtime_error&) { caught = true; }
        assert(caught && events == 0 && mappings == 0);
        mapped_size = 4096;
        run(); // The same resource remains reusable after a recoverable error.
        assert(events == 0 && mappings == 0);
    }
    for (int mode = 0; mode < 3; ++mode) {
        for (int invalid = 0; invalid < 3; ++invalid) {
            at::Tensor input;
            if (invalid == 0) input.on_cuda = false;
            if (invalid == 1) input.dtype = 1;
            if (invalid == 2) input.count = 0;
            bool caught = false;
            try {
                if (mode == 0) copy_rgb_to_pbo(1, input, 8, 8);
                if (mode == 1) copy_scalar_to_pbo(1, input, 8, 8, 0, 1, 0);
                if (mode == 2) copy_vector_to_pbo(1, input, 8, 8, true);
            } catch (const std::runtime_error&) { caught = true; }
            assert(caught && events == 0 && mappings == 0);
        }
    }
}
"""


def test_cpp_raii_releases_resources_on_each_failure(tmp_path):
    compiler = shutil.which("cl") or shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("A host C++ compiler is required; no CUDA compiler or GPU is used.")
    start = _CUDA_SOURCE.index("inline void check_cuda(")
    end = _CUDA_SOURCE.index("__device__ __forceinline__ unsigned char to_u8")
    guards = _CUDA_SOURCE[start:end]
    launch_names = re.findall(r"std::tuple<double, double, double> (launch_\w+)\(", _CUDA_SOURCE)
    copy_names = re.findall(r"std::tuple<double, double, double> (copy_\w+)\(", _CUDA_SOURCE)
    functions = "\n".join(function_source(name) for name in launch_names + copy_names)
    functions = re.sub(r"<<<[^>]+>>>", "", functions)
    source = tmp_path / "lifetime.cpp"
    source.write_text(_STUBS + guards + functions + _MAIN)
    binary = tmp_path / "lifetime.exe"
    if compiler.lower().endswith("cl.exe"):
        command = [compiler, "/nologo", "/EHsc", "/std:c++17", str(source), f"/Fe:{binary}"]
    else:
        command = [compiler, "-std=c++17", str(source), "-o", str(binary)]
    build = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert build.returncode == 0, build.stdout + build.stderr
    run = subprocess.run([str(binary)], cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert run.returncode == 0, run.stdout + run.stderr
