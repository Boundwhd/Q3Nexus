#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/extension.h>


#include "../include/rmsnorm.cuh"
#include "../include/linear.cuh"
#include "../include/silu.cuh"
#include "../include/rope.cuh"

torch::Tensor rmsnorm_bf16xbf16_cuda(
    const torch::Tensor& hidden_states,  
    const torch::Tensor& weight,
    float epsilon
) {
    auto out = torch::empty_like(hidden_states);

    const int batch_size = hidden_states.size(0);
    const int seq_len = hidden_states.size(1);
    const int hidden_size = hidden_states.size(2);

    TORCH_CHECK(hidden_states.dim() == 3, "hidden_states must be 3D [batch, seq_len, hidden_size]");
    TORCH_CHECK(weight.dim() == 1, "weight must be 1D [hidden_size]");
    TORCH_CHECK(hidden_states.size(2) == weight.size(0), "Hidden size mismatch: hidden_states hidden_size=", hidden_states.size(2), " vs weight size=", weight.size(0));

    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be on CUDA");
    TORCH_CHECK(out.is_cuda(), "out must be on CUDA");
    
    TORCH_CHECK(hidden_states.scalar_type() == torch::kBFloat16, "hidden_states must be BF16 tensor");
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight must be BF16 tensor");
    TORCH_CHECK(out.scalar_type() == torch::kBFloat16, "out must be BF16 tensor");

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

void fused_add_rmsnorm_bf16xbf16_cuda(
    const torch::Tensor& hidden_states, 
    const torch::Tensor& residual,
    const torch::Tensor& weight,
    torch::Tensor& out_hidden_states,
    torch::Tensor& out_residual,
    float epsilon
) {
    const auto batch_size = residual.size(0);
    const auto seq_len = residual.size(1);
    const auto hidden_size = residual.size(2);

    TORCH_CHECK(hidden_states.size(0) == batch_size && hidden_states.size(1) == seq_len && hidden_states.size(2) == hidden_size,"residual and hidden_states must have same dimensions");
    TORCH_CHECK(weight.size(0) == hidden_size,"weight size must match hidden_size");

    TORCH_CHECK(hidden_states.is_contiguous(), "hidden_states must be contiguous");
    TORCH_CHECK(residual.is_contiguous(), "residual must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");
    TORCH_CHECK(out_hidden_states.is_contiguous(), "out_hidden_states must be contiguous");
    TORCH_CHECK(out_residual.is_contiguous(), "out_residual must be contiguous");
    
    TORCH_CHECK(out_hidden_states.dim() == 3, "out_hidden_states must be 3D");
    TORCH_CHECK(out_hidden_states.size(0) == batch_size, "out_hidden_states batch size mismatch");
    TORCH_CHECK(out_hidden_states.size(1) == seq_len, "out_hidden_states seq_len mismatch");
    TORCH_CHECK(out_hidden_states.size(2) == hidden_size, "out_hidden_states hidden_size mismatch");
    
    TORCH_CHECK(out_residual.dim() == 3, "out_residual must be 3D");
    TORCH_CHECK(out_residual.size(0) == batch_size, "out_residual batch size mismatch");
    TORCH_CHECK(out_residual.size(1) == seq_len, "out_residual seq_len mismatch");
    TORCH_CHECK(out_residual.size(2) == hidden_size, "out_residual hidden_size mismatch");

    TORCH_CHECK(hidden_states.is_cuda(), "hidden_states must be on CUDA");
    TORCH_CHECK(residual.is_cuda(), "residual must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be on CUDA");
    TORCH_CHECK(out_hidden_states.is_cuda(), "out_hidden_states must be on CUDA");
    TORCH_CHECK(out_residual.is_cuda(), "out_residual must be on CUDA");

    TORCH_CHECK(residual.scalar_type() == torch::kBFloat16, "residual must be bfloat16");
    TORCH_CHECK(hidden_states.scalar_type() == torch::kBFloat16, "hidden_states must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == torch::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(out_hidden_states.scalar_type() == torch::kBFloat16, "out_hidden_states must be BF16 tensor");
    TORCH_CHECK(out_residual.scalar_type() == torch::kBFloat16,  "out_residual must be BF16 tensor");

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    launch_fused_add_rmsnorm_bf16xbf16(
        reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<torch::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(residual.data_ptr<torch::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(weight.data_ptr<torch::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out_hidden_states.data_ptr<torch::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(out_residual.data_ptr<torch::BFloat16>()),
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
}


torch::Tensor linear_bf16xbf16_cuda(
    const torch::Tensor& x,
    const torch::Tensor& A
) {
    TORCH_CHECK(x.device().is_cuda(), "x must be a CUDA tensor");
    TORCH_CHECK(A.device().is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "x must be BF16 tensor");
    TORCH_CHECK(A.scalar_type() == torch::kBFloat16, "A must be BF16 tensor");
    TORCH_CHECK(x.is_contiguous(), "x must be contiguous");
    TORCH_CHECK(A.is_contiguous(), "A must be contiguous");

    TORCH_CHECK(x.dim() >= 2 && x.dim() <= 4, "x must be 2D, 3D or 4D tensor");
    TORCH_CHECK(A.dim() == 2, "A must be 2D tensor [N, K]");

    int M = 1;
    std::vector<int64_t> original_sizes;
    for (int i = 0; i < x.dim() - 1; i++) {
        M *= x.size(i);
        original_sizes.push_back(x.size(i));
    }
    const int K = x.size(-1);
    const int N = A.size(0);

    TORCH_CHECK(K == A.size(1), 
        "x last dimension must match A's last dimension. Got ", K, " vs ", A.size(1));


    original_sizes.push_back(N);
    auto y = torch::empty(original_sizes, torch::TensorOptions().dtype(torch::kBFloat16).device(x.device()));
    launch_linear_bf16xbf16(
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<torch::BFloat16>()),
        reinterpret_cast<const __nv_bfloat16*>(A.data_ptr<torch::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(y.data_ptr<torch::BFloat16>()),
        M,
        N,
        K
    );

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, 
        "CUDA Error: ", cudaGetErrorString(err));
    
    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess,
        "CUDA Sync Error: ", cudaGetErrorString(err));
    
    return y;
}

torch::Tensor silu_bf16xbf16_cuda(const torch::Tensor& hidden_states) {
    
    TORCH_CHECK(hidden_states.dim() == 3, "hidden_states must have 3 dimensions [batch, seq_len, 2*intermediate_size]");

    int64_t batch_size = hidden_states.size(0);
    int64_t seq_len = hidden_states.size(1);
    int64_t two_intermediate = hidden_states.size(2);
    TORCH_CHECK(two_intermediate % 2 == 0, "The last dimension must be even (2*intermediate_size)");
    int64_t intermediate_size = two_intermediate / 2;

    auto out = torch::empty({batch_size, seq_len, intermediate_size}, hidden_states.options());

    const __nv_bfloat16* input_ptr = reinterpret_cast<const __nv_bfloat16*>(hidden_states.data_ptr<at::BFloat16>());
    __nv_bfloat16* output_ptr = reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());

    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
    launch_silu_bf16xbf16(
        input_ptr,
        output_ptr,
        static_cast<int>(intermediate_size),
        static_cast<int>(batch_size),
        static_cast<int>(seq_len),
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

void rope_cos_sin_bf16_cuda(
    torch::Tensor positions,
    torch::Tensor inv_freq,
    torch::Tensor cos,
    torch::Tensor sin
) {
    
    TORCH_CHECK(positions.is_cuda(), "positions must be a CUDA tensor");
    TORCH_CHECK(inv_freq.is_cuda(), "inv_freq must be a CUDA tensor");
    TORCH_CHECK(cos.is_cuda(), "cos must be a CUDA tensor");
    TORCH_CHECK(sin.is_cuda(), "sin must be a CUDA tensor");
    
    TORCH_CHECK(positions.scalar_type() == torch::kInt, "positions must be int32 tensor");
    TORCH_CHECK(inv_freq.scalar_type() == torch::kBFloat16, "inv_freq must be bfloat16 tensor");
    TORCH_CHECK(cos.scalar_type() == torch::kBFloat16, "cos must be bfloat16 tensor");
    TORCH_CHECK(sin.scalar_type() == torch::kBFloat16, "sin must be bfloat16 tensor");
    
    TORCH_CHECK(positions.dim() == 1, "positions must be 1D tensor");
    TORCH_CHECK(inv_freq.dim() == 1, "inv_freq must be 1D tensor");
    TORCH_CHECK(cos.dim() == 2, "cos must be 2D tensor");
    TORCH_CHECK(sin.dim() == 2, "sin must be 2D tensor");
    
    int seq_len = positions.size(0);
    int head_dim = cos.size(1);
    
    TORCH_CHECK(cos.size(0) == seq_len, "cos size(0) must match positions");
    TORCH_CHECK(sin.size(0) == seq_len, "sin size(0) must match positions");
    TORCH_CHECK(inv_freq.size(0) == head_dim / 2, 
                "inv_freq size(0) must be half of head_dim");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    launch_rope_cos_sin_bf16(
        reinterpret_cast<int32_t*>(positions.data_ptr<int32_t>()),
        reinterpret_cast<__nv_bfloat16*>(inv_freq.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(cos.data_ptr<at::BFloat16>()),
        reinterpret_cast<__nv_bfloat16*>(sin.data_ptr<at::BFloat16>()),
        seq_len,
        head_dim,
        stream
    );
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, 
        "CUDA Error: ", cudaGetErrorString(err));
    
    err = cudaDeviceSynchronize();
    TORCH_CHECK(err == cudaSuccess,
        "CUDA Sync Error: ", cudaGetErrorString(err));
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rmsnorm_bf16xbf16", &rmsnorm_bf16xbf16_cuda, "RMSNorm for BF16 (CUDA)");
    m.def("fused_add_rmsnorm_bf16xbf16", &fused_add_rmsnorm_bf16xbf16_cuda, "Fused add rmsorm for BF16 (CUDA)");
    m.def("linear_bf16xbf16", &linear_bf16xbf16_cuda, "Linear for BF16 (CUDA)");
    m.def("silu_bf16xbf16", &silu_bf16xbf16_cuda, "SiLU BF16 x BF16 (CUDA)");
    m.def("rope_cos_sin_bf16", &rope_cos_sin_bf16_cuda, "Rope BF16 (CUDA)");
}