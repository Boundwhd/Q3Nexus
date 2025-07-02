#include "../include/common.cuh"
#include "../include/linear.cuh"
#include <cublas_v2.h>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    }

#define CUBLAS_CHECK(err) \
    if (err != CUBLAS_STATUS_SUCCESS) { \
        fprintf(stderr, "cuBLAS Error %s:%d: %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

// y = xA^T + b
/**
 * x [M, K]
 * A [N, K]
 * y [M, N]
 * custom for Qwen3, no bias
 */
void launch_linear_bf16xbf16(
   const __nv_bfloat16* x,
   const __nv_bfloat16* A,
   __nv_bfloat16* y,
   const int M,
   const int N,
   const int K
) {
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    const float alpha = 1.0f, beta = 0.0f;
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_TENSOR_OP_MATH));
    CUBLAS_CHECK(cublasSgemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        N,
        M,
        K,
        &alpha,
        A,
        CUDA_R_16BF,
        K,
        x,
        CUDA_R_16BF,
        K,
        &beta,
        y,
        CUDA_R_16BF,
        N
    ));
    CUBLAS_CHECK(cublasDestroy(handle));
}