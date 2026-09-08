#include <optix.h>
#include <optix_device.h>

static __forceinline__ __device__ float3 add3(const float3 a, const float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

static __forceinline__ __device__ float3 mul3f(const float3 a, const float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}

static __forceinline__ __device__ float dot3(const float3 a, const float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __forceinline__ __device__ float3 normalize3(const float3 v) {
    const float inv_len = rsqrtf(dot3(v, v) + 1e-20f);
    return mul3f(v, inv_len);
}

struct Params {
    unsigned long long depth_buffer;
    unsigned long long hit_t_buffer;
    unsigned long long hit_count_buffer;
    unsigned int image_width;
    unsigned int image_height;
    unsigned int max_hits;
    float3 cam_eye;
    float3 cam_u;
    float3 cam_v;
    float3 cam_w;
    unsigned long long ray_origins_buffer;
    unsigned long long ray_directions_buffer;
    unsigned long long ray_tmin_buffer;
    unsigned long long ray_tmax_buffer;
    OptixTraversableHandle dilation_trav_handle;
    OptixTraversableHandle erosion_trav_handle;
    unsigned long long erosion_first_t_buffer;
    unsigned long long dilation_hit_flag_buffer;
    unsigned long long interval_start_buffer;
    unsigned long long interval_end_buffer;
    unsigned long long interval_count_buffer;
    unsigned long long sample_count_buffer;
    unsigned int max_intervals;
    unsigned int n_sample_max;
    float min_interval_size;
    float single_sample_threshold;
    float sample_spacing;
    unsigned int compress_same_flag_runs;
};

extern "C" __constant__ Params params;

extern "C" __global__ void __miss__ms() {
    optixSetPayload_0(__float_as_uint(-1.0f));
}

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    const unsigned int x = idx.x;
    const unsigned int y = idx.y;
    float* depth = (float*)params.depth_buffer;
    float* hit_t = (float*)params.hit_t_buffer;
    float* erosion_first_t = (float*)params.erosion_first_t_buffer;
    unsigned int* dilation_hit_flag = (unsigned int*)params.dilation_hit_flag_buffer;
    unsigned int* hit_count = (unsigned int*)params.hit_count_buffer;
    float* interval_start = (float*)params.interval_start_buffer;
    float* interval_end = (float*)params.interval_end_buffer;
    unsigned int* interval_count = (unsigned int*)params.interval_count_buffer;
    unsigned int* sample_count = (unsigned int*)params.sample_count_buffer;
    const bool write_interval_metadata = (
        interval_start != 0 &&
        interval_end != 0 &&
        interval_count != 0 &&
        sample_count != 0 &&
        params.max_intervals > 0 &&
        params.n_sample_max > 0
    );
    const unsigned int ray_idx = y * params.image_width + x;
    const unsigned int hit_buffer_base = ray_idx * params.max_hits;
    const unsigned int interval_buffer_base = ray_idx * params.max_intervals;

    float3 ray_o;
    float3 ray_d;
    float current_tmin;
    float ray_tmax;
    const float* ray_origins = (const float*)params.ray_origins_buffer;
    const float* ray_directions = (const float*)params.ray_directions_buffer;
    const float* ray_tmins = (const float*)params.ray_tmin_buffer;
    const float* ray_tmaxs = (const float*)params.ray_tmax_buffer;
    ray_o = make_float3(
        ray_origins[3 * ray_idx + 0],
        ray_origins[3 * ray_idx + 1],
        ray_origins[3 * ray_idx + 2]
    );
    ray_d = normalize3(make_float3(
        ray_directions[3 * ray_idx + 0],
        ray_directions[3 * ray_idx + 1],
        ray_directions[3 * ray_idx + 2]
    ));
    current_tmin = ray_tmins != 0 ? fmaxf(ray_tmins[ray_idx], 1e-4f) : 1e-4f;
    ray_tmax = ray_tmaxs != 0 ? ray_tmaxs[ray_idx] : 1e16f;

    if (!(ray_tmax > current_tmin)) {
        depth[ray_idx] = -1.0f;
        hit_count[ray_idx] = 0u;
        if (write_interval_metadata) {
            interval_count[ray_idx] = 0u;
            sample_count[ray_idx] = 0u;
        }
        return;
    }

    unsigned int p0 = __float_as_uint(-1.0f);
    unsigned int p1 = 0u;
    optixTrace(
        params.erosion_trav_handle,
        ray_o,
        ray_d,
        current_tmin,
        ray_tmax,
        0.0f,
        255,
        OPTIX_RAY_FLAG_DISABLE_ANYHIT,
        0,
        1,
        0,
        p0,
        p1
    );
    float erosion_t = __uint_as_float(p0);
    erosion_first_t[ray_idx] = erosion_t;

    unsigned int count = 0;
    for (unsigned int i = 0; i < params.max_hits; i++) {
        p0 = __float_as_uint(-1.0f);
        p1 = 0u;
        optixTrace(
            params.dilation_trav_handle,
            ray_o,
            ray_d,
            current_tmin,
            ray_tmax,
            0.0f,
            255,
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0,
            1,
            0,
            p0,
            p1
        );
        float t = __uint_as_float(p0);
        if (t < 0.0f) {
            break;
        }
        hit_t[hit_buffer_base + i] = t;
        if (i == 0) {
            depth[ray_idx] = t;
        }
        dilation_hit_flag[hit_buffer_base + i] = p1;
        ++count;
        current_tmin = t + 1e-4f;
        if (!(ray_tmax > current_tmin)) {
            break;
        }
    }
    hit_count[ray_idx] = count;
    if (write_interval_metadata) {
        unsigned int pair_count = 0u;
        unsigned int ray_sample_count = 0u;
        bool has_open = false;
        bool finished = false;
        float current_enter = 0.0f;
        const float inner_t = erosion_t > 0.0f ? erosion_t : 3.402823466e+38f;
        for (unsigned int i = 0; i < count; ++i) {
            if (finished) {
                break;
            }
            const float t = hit_t[hit_buffer_base + i];
            const unsigned int flag = dilation_hit_flag[hit_buffer_base + i];
            bool first_enter = false;
            bool last_exit = false;
            if (params.compress_same_flag_runs != 0u) {
                const unsigned int prev_flag = (i == 0u) ? 0u : dilation_hit_flag[hit_buffer_base + i - 1u];
                const unsigned int next_flag = (i + 1u >= count) ? 0u : dilation_hit_flag[hit_buffer_base + i + 1u];
                first_enter = (flag == 1u) && (i == 0u || prev_flag != 1u);
                last_exit = (flag == 2u) && (i + 1u >= count || next_flag != 2u);
            } else {
                first_enter = (flag == 1u);
                last_exit = (flag == 2u);
            }
            if (first_enter && !has_open) {
                current_enter = t;
                has_open = true;
            }
            if (last_exit && !has_open) {
                current_enter = ray_tmins != 0 ? ray_tmins[ray_idx] : 1e-4f;
                has_open = true;
            }
            const bool commit = last_exit && has_open;
            const float d_exit = fminf(t, inner_t);
            if (commit && ((d_exit - current_enter) >= params.min_interval_size) && pair_count < params.max_intervals) {
                interval_start[interval_buffer_base + pair_count] = current_enter;
                interval_end[interval_buffer_base + pair_count] = d_exit;
                const float interval_len = d_exit - current_enter;
                int n_samples = (int)ceilf(fmaxf(interval_len - params.single_sample_threshold, 0.0f) / params.sample_spacing) + 1;
                if (n_samples < 1) {
                    n_samples = 1;
                }
                if (n_samples > (int)params.n_sample_max) {
                    n_samples = (int)params.n_sample_max;
                }
                ray_sample_count += (unsigned int)n_samples;
                pair_count += 1u;
            }
            if (commit) {
                has_open = false;
                current_enter = 0.0f;
                finished = isfinite(inner_t) && (inner_t < t);
            }
        }
        interval_count[ray_idx] = pair_count;
        sample_count[ray_idx] = ray_sample_count;
    }
}

extern "C" __global__ void __closesthit__ch() {
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_1(optixIsFrontFaceHit() ? 1u : 2u);
}
