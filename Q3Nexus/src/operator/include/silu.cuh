#ifndef __SILU_CUH__
#define __SILU_CUH__

#include "common.cuh"

void launch_silu_bf16xbf16(
    const __nv_bfloat16* hidden_states,
    __nv_bfloat16* out,
    const int intermediate_size,
    const int batch_size,
    const int seq_len,
    cudaStream_t stream = 0
);

#endif