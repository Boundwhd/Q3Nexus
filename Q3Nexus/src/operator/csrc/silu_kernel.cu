#include "../include/common.cuh"
#include "../include/silu.cuh"

/**
 * hidden_states: [batch, seq_len, 2 * intermediate_size]
 * out: [batch, seq_len, intermediate_size]
 * 
 * for each row :    x x x x | v v v v
 * x for gate output && v for up out put
 * 1. for x do silu
 * 2. for x do x * v
 * 3. out
 */

__device__ __forceinline__ void silu(float2& x) {
    x.x = x.x / (1.0f + expf(-x.x));
    x.y = x.y / (1.0f + expf(-x.y));
}

__global__ void silu_kernel_bf16xbf16_pack2(
    const __nv_bfloat16* hidden_states,  // [num_tokens, 2 * intermediate_size]
    __nv_bfloat16* out,                  // [num_tokens, intermediate_size]
    const int intermediate_size
) {
    const int token_idx = blockIdx.x;
    const int tid = threadIdx.x;

    #pragma unroll
    for (int i = (tid << 1); i < intermediate_size; i += (blockDim.x << 1)) {
        __nv_bfloat162 gate = reinterpret_cast<const __nv_bfloat162*>(&hidden_states[token_idx * 2 * intermediate_size + i])[0];
        __nv_bfloat162 up = reinterpret_cast<const __nv_bfloat162*>(&hidden_states[token_idx * 2 * intermediate_size + intermediate_size + i])[0];
        __nv_bfloat162 out_value;
        float2 gate_f;
        float2 up_f;
        float2 out_f;
        gate_f.x = __bfloat162float(gate.x); 
        gate_f.y = __bfloat162float(gate.y);
        up_f.x = __bfloat162float(up.x);
        up_f.y = __bfloat162float(up.y);
        silu(gate_f);
        out_f.x = gate_f.x * up_f.x;
        out_f.y = gate_f.y * up_f.y;
        out_value.x = __float2bfloat16(out_f.x);
        out_value.y = __float2bfloat16(out_f.y);
        reinterpret_cast<__nv_bfloat162*>(&out[token_idx * intermediate_size + i])[0] = out_value;
    }
}

void launch_silu_bf16xbf16(
    const __nv_bfloat16* hidden_states,
    __nv_bfloat16* out,
    const int intermediate_size,
    const int batch_size,
    const int seq_len,
    cudaStream_t stream

) {
    constexpr int NUM_THREADS = 512;  
    dim3 grid(batch_size * seq_len);
    dim3 block(NUM_THREADS);

    silu_kernel_bf16xbf16_pack2<<<grid, block, 0, stream>>>(
        hidden_states,
        out,
        intermediate_size
    );
}
