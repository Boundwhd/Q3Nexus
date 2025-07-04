#include "../include/common.cuh"
#include "../include/linear.cuh"

// 设置128个线程
// 64 个线程搬运前面的 head_dim / 2
// 64 个线程搬运后面的 head_dim / 2
// head_dim 一般为 128

__global__ void rope_cos_sin_bf16(
    const uint32_t* positions,
    const __nv_bfloat16* inv_freq,
    __nv_bfloat16* cos,
    __nv_bfloat16* sin,
    const int seq_len,
    const int head_dim
) {
    const int tid = threadIdx.x;
    const int row = blockIdx.x;
    const int part = tid % (head_dim / 2);
    #pragma unroll
    for (int i = tid; i < head_dim; i += blockIdx.x) {
        float row_co = static_cast<float>(positions[row]);
        float inv = __bfloat162float(inv_freq[part]);
        float cos_value = cosf(row_co * inv);
        float sin_value = cosf(row_co * inv);
        cos[row * head_dim + tid] = __float2bfloat16(cos_value);
        sin[row * head_dim + tid] = __float2bfloat16(sin_value);
    }
}

void launch_rope_cos_sin_bf16(
    const uint32_t* positions,
    const __nv_bfloat16* inv_freq,
    __nv_bfloat16* cos,
    __nv_bfloat16* sin,
    const int seq_len,
    const int head_dim,
    cudaStream_t stream
) {
    constexpr int NUM_THREADS = 128;  
    dim3 grid(seq_len);
    dim3 block(NUM_THREADS);

    rope_cos_sin_bf16<<<grid, block, 0, stream>>>(
        positions,
        inv_freq,
        cos,
        sin,
        seq_len,
        head_dim
    );
}