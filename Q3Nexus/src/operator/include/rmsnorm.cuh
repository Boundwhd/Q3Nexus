#ifndef __RMSNORM_CUH__
#define __RMSNORM_CUH__

#include "common.cuh"

void launch_rmsnorm_bf16xbf16(
    const __nv_bfloat16* hidden_states,  
    const __nv_bfloat16* weight,       
    __nv_bfloat16* output,              
    int batch_size,
    int seq_len,
    int hidden_size,
    float epsilon = 1e-6f,
    cudaStream_t stream = 0
);

void launch_fused_add_rmsnorm_bf16xbf16(
    const __nv_bfloat16* hidden_states,
    const __nv_bfloat16* residual,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out_hidden_states,
    __nv_bfloat16* out_residual,
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const float epsilon = 1e-6,
    cudaStream_t stream = 0
);

#endif