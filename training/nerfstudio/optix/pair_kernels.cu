// CUDA kernel for pairing shell hit intervals.

#define ENTERING 1
#define EXITING 2
#define INFINITY 1e30f

extern "C" __global__ void pair_shell_hit_intervals_kernel(
    const float* outer_hit_t,
    const int* outer_hit_flag,
    const int* outer_hit_count,
    const float* inner_first_t,
    const float* ray_tmin,
    const int n_rays,
    const int max_hits,
    const int max_intervals,
    const float min_interval_size,
    const int compress_same_flag_runs,
    float* interval_starts,
    float* interval_ends,
    unsigned char* interval_mask,
    int* interval_count)
{
    const int ray_idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (ray_idx >= n_rays) {
        return;
    }

    const int out_base = ray_idx * max_intervals;
    const int hit_base = ray_idx * max_hits;

    for (int k = 0; k < max_intervals; ++k) {
        interval_starts[out_base + k] = -1.0f;
        interval_ends[out_base + k] = -1.0f;
        interval_mask[out_base + k] = 0;
    }
    interval_count[ray_idx] = 0;

    // ---------------------------------------------------------------------
    bool has_open_interval = false;
    bool finished = false;
    float current_enter_t = 0.0f;
    int out_count = 0;
    float inner_t = inner_first_t[ray_idx] > 0.0f ? inner_first_t[ray_idx] : INFINITY;
    int hit_count = outer_hit_count[ray_idx];
    //
    // Then loop:
    for (int hit_idx = 0; hit_idx < max_hits; ++hit_idx) { 
    // For each hit:
        // 1. Stop if finished or hit_idx >= hit_count.
        if (finished || hit_idx >= hit_count) {
            break;
        }
        // 2. Load:
        float t_i = outer_hit_t[hit_base + hit_idx];
        int flag_i = outer_hit_flag[hit_base + hit_idx];
        // 3. Compute `first_enter` and `last_exit`.
        bool first_enter = false;
        bool last_exit = false;
        if (compress_same_flag_runs) {
            if (flag_i == ENTERING) {
                if (hit_idx == 0) {
                    first_enter = true;
                }
                else {
                    int prev_flag = outer_hit_flag[hit_base + hit_idx - 1];
                    first_enter = prev_flag != ENTERING;
                }
            }
            if (flag_i == EXITING) {
                if (hit_idx == max_hits - 1 || hit_idx + 1 >= hit_count) {
                    last_exit = true;
                }
                else {
                    int next_flag = outer_hit_flag[hit_base + hit_idx + 1];
                    last_exit = next_flag != EXITING;
                }
            }

        }
        else {
            first_enter = flag_i == ENTERING;
            last_exit = flag_i == EXITING;
        }
        // 4. Open interval if entering.
        if (first_enter && !has_open_interval) {
            current_enter_t = t_i;
            has_open_interval = true;
        }
    
        // 5. If first valid event is EXITING, open from ray_tmin[ray_idx].
        if (last_exit && !has_open_interval) {
            current_enter_t = ray_tmin[ray_idx];
            has_open_interval = true;
        }
        // 6. If committing:
        bool commit = last_exit && has_open_interval;
        if (commit) {
            float d_exit = t_i < inner_t ? t_i : inner_t;
            bool valid_interval = (d_exit - current_enter_t) >= min_interval_size;

            if (valid_interval && out_count < max_intervals) {
                interval_starts[out_base + out_count] = current_enter_t;
                interval_ends[out_base + out_count] = d_exit;
                interval_mask[out_base + out_count] = 1;
                out_count++;
            }

            has_open_interval = false;
            current_enter_t = 0.0f;
            if (isfinite(inner_t) && inner_t < t_i) {
                finished = true;
            }
        }
    }
    // ---------------------------------------------------------------------
    interval_count[ray_idx] = out_count;
}
