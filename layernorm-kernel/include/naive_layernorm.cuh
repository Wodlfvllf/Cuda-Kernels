#pragma once

#include <cuda_runtime.h>

// Naive LayerNorm kernel declarations
__global__ void compute_mean_kernel(const float* __restrict__ input, 
                                   float* __restrict__ mean, 
                                   int N, int D);

__global__ void compute_variance_kernel(const float* __restrict__ input,
                                       const float* __restrict__ mean,
                                       float* __restrict__ variance,
                                       int N, int D);

__global__ void normalize_kernel(const float* __restrict__ input,
                                const float* __restrict__ mean,
                                const float* __restrict__ variance,
                                const float* __restrict__ gamma,
                                const float* __restrict__ beta,
                                float* __restrict__ output,
                                int N, int D, float eps);

// Wrapper function for naive implementation
void layernorm_naive(const float* input, const float* gamma, const float* beta,
                    float* output, float* mean, float* variance,
                    int N, int D, float eps, cudaStream_t stream = 0);