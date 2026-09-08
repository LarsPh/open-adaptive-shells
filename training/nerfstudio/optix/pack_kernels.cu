// CUDA kernel for packing shell intervals into fixed sample bins.
//
// This kernel mirrors the current Python/Torch contract:
// - input:
//     interval_starts: (R, max_intervals)
//     interval_ends:   (R, max_intervals)
//     interval_mask:   (R, max_intervals) as uint8
//     interval_count:  (R,)
// - output:
//     packed_starts: (R, n_total)
//     packed_ends:   (R, n_total)
//     packed_mask:   (R, n_total) as uint8
//     packed_eval_t: (R, n_total)
//

extern "C" __global__ void pack_interval_to_sample_bins_kernel(
    const float* interval_starts,
    const float* interval_ends,
    const unsigned char* interval_mask,
    const int* interval_count,
    const int n_rays,
    const int max_intervals,
    const int n_sample_max,
    const int n_total,
    const float single_sample_threshold,
    const float sample_spacing,
    float* packed_starts,
    float* packed_ends,
    unsigned char* packed_mask,
    float* packed_eval_t)
{
    const int ray_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (ray_idx >= n_rays) {
        return;
    }

    int valid_interval_count = interval_count[ray_idx];
    const int interval_base = ray_idx * max_intervals;
    const int packed_base = ray_idx * n_total;

    if (valid_interval_count <= 0) {
        return;
    }

    // ---------------------------------------------------------------------
    int write_offset = 0;
    //
    // Then for each interval k in [0, max_intervals):
    for (int k = 0; k < max_intervals; ++k) {
        //   1. Skip if k >= valid_interval_count
        if (k >= valid_interval_count) break;
        if (!interval_mask[interval_base + k]) continue;
    //   2. Load:
        float start = interval_starts[interval_base + k];
        float end = interval_ends[interval_base + k];
        float len = end - start;
    //   3. Compute samples_per_interval:
    //        ceil(max(len - single_sample_threshold, 0) / sample_spacing) + 1
        float excess = len - single_sample_threshold;
        if (excess < 0.0f) excess = 0.0f;
        int n = (int)ceilf(excess / sample_spacing) + 1;
    //        clamp to [1, n_sample_max]
        if (n < 1) n = 1;
        if (n > n_sample_max) n = n_sample_max;
    //   4. Compute:
        float step = len / (float)(n + 1);
        //   5. For each local sample j in [0, samples_per_interval):
        for (int j = 0; j < n; ++j) {
            float tau = start + (j + 1) * step; // evaluate at tau
            // midpoint between taus, for nerfstudio convention
            float bin_start = (j == 0) ? start : 0.5 * (tau + (start + j * step)); 
            float bin_end   = (j == n - 1)
                                ? end
                                : 0.5 * (tau + (start + (j + 2) * step));
            //   6. Write to packed slot:
            int local_idx = write_offset + j;
            if (local_idx >= n_total) break;
            int out_idx = packed_base + local_idx;
            packed_starts[out_idx] = bin_start;
            packed_ends[out_idx] = bin_end;
            packed_mask[out_idx] = 1;
            packed_eval_t[out_idx] = tau;
        }
    //   7. After the interval:
        write_offset += n;
    }
}
