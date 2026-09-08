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

// launch parameters
struct Params
{
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
};

extern "C" __constant__ Params params;

// shaders

extern "C" __global__ void __miss__ms()
{
    optixSetPayload_0(__float_as_uint(-1.0f));
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 idx = optixGetLaunchIndex();
    const unsigned int x = idx.x;
    const unsigned int y = idx.y;
    float* depth = (float*)params.depth_buffer;
    float* hit_t = (float*)params.hit_t_buffer;
    float* erosion_first_t = (float*)params.erosion_first_t_buffer;
    unsigned int* dilation_hit_flag = (unsigned int*)params.dilation_hit_flag_buffer;
    unsigned int* hit_count = (unsigned int*)params.hit_count_buffer;
    const unsigned int ray_idx = y * params.image_width + x;
    const unsigned int hit_buffer_base = ray_idx * params.max_hits;

    float3 ray_o;
    float3 ray_d;
    float current_tmin;
    float ray_tmax;
    if (params.ray_origins_buffer != 0ull && params.ray_directions_buffer != 0ull) {
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
    } else {
        const uint3 dim = optixGetLaunchDimensions();
        const float2 dxy = make_float2(2.0f * (x + 0.5f) / dim.x - 1.0f, 2.0f * (y + 0.5f) / dim.y - 1.0f);
        ray_o = params.cam_eye;
        ray_d = normalize3(
            add3(
                add3(mul3f(params.cam_u, dxy.x), mul3f(params.cam_v, dxy.y)),
                params.cam_w
            )
        );
        current_tmin = 1e-4f;
        ray_tmax = 1e16f;
    }

    if (!(ray_tmax > current_tmin)) {
        depth[ray_idx] = -1.0f;
        hit_count[ray_idx] = 0u;
        return;
    }

    // trace erosion depth
    unsigned int p0 = __float_as_uint(-1.0f);
    unsigned int p1 = 0u;
    optixTrace(params.erosion_trav_handle, ray_o, ray_d, 
                current_tmin, ray_tmax, // tmin, tmax
                0.0f, 255, // time, visibilityMask
                OPTIX_RAY_FLAG_DISABLE_ANYHIT, // ray_flags
                0, 1, 0, // sbt_offset, sbt_stride, miss_sbt_idx
                p0, p1);
    float erosion_t = __uint_as_float(p0);
    erosion_first_t[ray_idx] = erosion_t;
    
    unsigned int count = 0;
    for (unsigned int i = 0; i < params.max_hits; i++) {
        // Initialize payload 
        p0 = __float_as_uint(-1.0f); // missed as default
        p1 = 0u; // flag
        // Call optixTrace with params.trav_handle
        // tmin, tmax, time, visibilityMask, ray_flags, sbt_record, sbt_offset, sbt_stride
        optixTrace(params.dilation_trav_handle, ray_o, ray_d, 
                current_tmin, ray_tmax, // tmin, tmax
                0.0f, 255, // time, visibilityMask
                OPTIX_RAY_FLAG_DISABLE_ANYHIT, // ray_flags
                0, 1, 0, // sbt_offset, sbt_stride, miss_sbt_idx
                p0, p1); 
        float t = __uint_as_float(p0);
        if (t < 0.0f) {
            // missed
            break;
        }
        // write buffers
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
}

extern "C" __global__ void __anyhit__ah() {

}

extern "C" __global__ void __closesthit__ch() {
    optixSetPayload_0(__float_as_uint(optixGetRayTmax()));
    optixSetPayload_1(optixIsFrontFaceHit() ? 1u : 2u);
}
