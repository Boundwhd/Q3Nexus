#include "../include/common.cuh"
#include "../include/rmsnorm.cuh"

// ------------ WarpReduceSum f32 -------------
// --------------------------------------------
template<const int kWarpSize = WARP_SIZE>
__device__ __forceinline__ float warp_reduce_sum_f32(float val) {
    #pragma unroll
    for (int mask = kWarpSize >> 1; mask >= 1; mask >>= 1) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

// ------------ BlockReduceSum f32 -------------
// ---------------------------------------------
template<const int NUM_THREADS>
__device__ __forceinline__ float block_reduce_sum_f32(float val) {
    constexpr int NUM_WARPS = (NUM_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    int warp = threadIdx.x / WARP_SIZE;
    int lane = threadIdx.x % WARP_SIZE;
    static __shared__ float shared[NUM_WARPS];
  
    val = warp_reduce_sum_f32(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();
    val = (lane < NUM_WARPS) ? shared[lane] : 0.0f;
    val = warp_reduce_sum_f32(val);
    return val;
}

// ------------ RMSNorm_bf16xbf16 ------------
// hidden_states [batch, seq_len, hidden_size]
// weight [hidden_size]
// out [batch, seq_len, hidden_size]
// -------------------------------------------
template<const int NUM_THREADS>
__global__ void rmsnorm_kernel_bf16xbf16(
    const __nv_bfloat16* hidden_states,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out,
    const int hidden_size,
    const float epsilon
) {
    __shared__ float s_variance;
    const int tid = threadIdx.x;
    const int token_idx = blockIdx.x;
    
    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float value = __bfloat162float(hidden_states[token_idx * hidden_size + i]);
        local_sum += value * value;
    }
    
    float total_sum = block_reduce_sum_f32<NUM_THREADS>(local_sum);
    if (tid == 0) {
        float mean = total_sum / static_cast<float>(hidden_size);
        s_variance = rsqrtf(mean + epsilon);
    }
    __syncthreads();

    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __bfloat162float(hidden_states[token_idx * hidden_size + i]);
        float w = __bfloat162float(weight[i]);
        out[token_idx * hidden_size + i] = __float2bfloat16(val * s_variance * w);
    }
}

// ------------ Fused_Add_RMSNorm_bf16xbf16 ------------
// hidden_states [batch, seq_len, hidden_size]
// residual [batch, seq_len, hidden_size]
// weight [hidden_size]
// out_hidden_states [batch, seq_len, hidden_size]
// out_residual [batch, seq_len, hidden_size]
// -----------------------------------------------------
template<const int NUM_THREADS>
__global__ void fused_add_rmsnorm_kernel_bf16xbf16(
    const __nv_bfloat16* hidden_states,
    const __nv_bfloat16* residual,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out_hidden_state,
    __nv_bfloat16* out_residual,
    const int hidden_size,
    const float epsilon
) {
    __shared__ float s_variance;
    const int tid = threadIdx.x;
    const int token_idx = blockIdx.x;

    float local_sum = 0.0f;
    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float input_val = __bfloat162float(hidden_states[token_idx * hidden_size + i]);
        float residual_val = __bfloat162float(residual[token_idx * hidden_size + i]);
        float hidden_states_new = input_val + residual_val;
        local_sum += hidden_states_new * hidden_states_new;
        out_residual[token_idx * hidden_size + i] = __float2bfloat16(hidden_states_new);
    }
    __syncthreads();

    float total_sum = block_reduce_sum_f32<NUM_THREADS>(local_sum);
    if (tid == 0) {
        float mean = total_sum / static_cast<float>(hidden_size);
        s_variance = rsqrtf(mean + epsilon);
    }
    __syncthreads();

    for (int i = tid; i < hidden_size; i += blockDim.x) {
        float val = __bfloat162float(out_residual[token_idx * hidden_size + i]);
        float w = __bfloat162float(weight[i]);
        out_hidden_state[token_idx * hidden_size + i] = __float2bfloat16(val * s_variance * w);
    }
}


void launch_rmsnorm_bf16xbf16(
    const __nv_bfloat16* hidden_states,  
    const __nv_bfloat16* weight,       
    __nv_bfloat16* output,              
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const float epsilon,
    cudaStream_t stream
) {
    constexpr int NUM_THREADS = 256;  
    dim3 grid(batch_size * seq_len);
    dim3 block(NUM_THREADS);

    rmsnorm_kernel_bf16xbf16<NUM_THREADS><<<grid, block, 0, stream>>>(
        hidden_states,
        weight,
        output,
        hidden_size,
        epsilon
    );
}

void launch_fused_add_rmsnorm_bf16xbf16(
    const __nv_bfloat16* hidden_states,
    const __nv_bfloat16* residual,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out_hidden_states,
    __nv_bfloat16* out_residual,
    const int batch_size,
    const int seq_len,
    const int hidden_size,
    const float epsilon,
    cudaStream_t stream
) {
    constexpr int NUM_THREADS = 256;  
    dim3 grid(batch_size * seq_len);
    dim3 block(NUM_THREADS);

    fused_add_rmsnorm_kernel_bf16xbf16<NUM_THREADS><<<grid, block, 0, stream>>>(
        hidden_states,
        residual,
        weight,
        out_hidden_states,
        out_residual,
        hidden_size,
        epsilon
    );
}