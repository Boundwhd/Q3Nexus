#ifndef __ROPE_CUH__
#define __ROPE_CUH__

#include "common.cuh"

void launch_rope_cos_sin_bf16(
    const int32_t* positions,
    const __nv_bfloat16* inv_freq,
    __nv_bfloat16* cos,
    __nv_bfloat16* sin,
    const int seq_len,
    const int head_dim,
    cudaStream_t stream = 0
);

#endif