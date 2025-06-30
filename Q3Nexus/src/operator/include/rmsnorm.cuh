#ifndef __RMSNORM_CUH__
#define __RMSNORM_CUH__

#include "common.cuh"

void launch_rmsnorm_bf16xbf16(
    const __nv_bfloat16* hidden_state,  
    const __nv_bfloat16* weight,       
    __nv_bfloat16* output,              
    int batch_size,
    int seq_len,
    int hidden_size,
    float epsilon = 1e-6f,
    cudaStream_t stream = 0
);

#endif