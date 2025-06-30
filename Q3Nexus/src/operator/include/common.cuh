#pragma once

#include <cstdio>
#include <cstdlib>     
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>

constexpr int WARP_SIZE = 32;