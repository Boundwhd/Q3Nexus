#pragma once

#include <cstdio>
#include <cstdlib>     
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>
#include <stdint.h>

constexpr int WARP_SIZE = 32;