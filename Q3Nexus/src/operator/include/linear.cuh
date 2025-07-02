#ifndef __LINEAR_CUH__
#define __LINEAR_CUH__

#include "common.cuh"

void launch_linear_bf16xbf16(
   const __nv_bfloat16* x,
   const __nv_bfloat16* A,
   __nv_bfloat16* y,
   const int M,
   const int N,
   const int K
);

#endif