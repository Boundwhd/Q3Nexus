#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>

#include "../include/rmsnorm.cuh"

torch::Tensor rmsnorm_bf16xbf16_cuda(
    const torch::Tensor& hidden_states,  
    const torch::Tensor& weight,
    float epsilon
) {
    auto out = torch::empty_like(hidden_states);

    TORCH_CHECK(hidden_states.dim() == 3, "hidden_states must be 3D [batch, seq_len, hidden_size]");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D [hidden_size]");
    TORCH_CHECK(hidden_states.size(2) == weight.size(0), "Hidden size mismatch: hidden_states hidden_size=", hidden_states.size(2), " vs weight size=", weight.size(0));

    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be on CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be on CUDA");
    
    TORCH_CHECK(hidden_states.scalar_type() == torch::kBFloat16, "hidden_states must be BF16 tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight must be BF16 tensor");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be BF16 tensor");

    const int batch_size = hidden_states.size(0);
    const int seq_len = hidden_states.size(1);
    const int hidden_size = hidden_states.size(2);
    
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    launch_rmsnorm_bf16xbf16(
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<torch::BFloat16>()), 
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<torch::BFloat16>()),      
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<torch::BFloat16>()),             
        batch_size,
        seq_len,
        hidden_size,
        epsilon,
        stream
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, 
        "CUDA Error: ", cudaGetErrorString(err));
    
    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess,
        "CUDA Sync Error: ", cudaGetErrorString(err));
    
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_bf16xbf16", &rmsnorm_bf16xbf16_cuda, "RMSNorm for BF16 (CUDA)");
}