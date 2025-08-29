#pragma once

#include <cuda_runtime.h>

// Welford's algorithm based LayerNorm kernel
template<int BLOCK_SIZE>
__global__ void layernorm_welford_kernel(const float* __restrict__ input,
                                        const float* __restrict__ gamma,
                                        const float* __restrict__ beta,
                                        float* __restrict__ output,
                                        int N, int D, float eps);

// Wrapper function
void layernorm_welford(const float* input, const float* gamma, const float* beta,
                      float* output, int N, int D, float eps,
                      cudaStream_t stream = 0);