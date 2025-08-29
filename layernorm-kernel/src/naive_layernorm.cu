#include "naive_layernorm.cuh"
#include "common.cuh"
#include <cmath>

__global__ void compute_mean_kernel(const float* __restrict__ input, 
                                   float* __restrict__ mean, 
                                   int N, int D) {
    int tid = blockIdx.x;
    if (tid >= N) return;
    
    float sum = 0.0f;
    for (int i = 0; i < D; i++) {
        sum += input[tid * D + i];
    }
    mean[tid] = sum / D;
}

__global__ void compute_variance_kernel(const float* __restrict__ input,
                                       const float* __restrict__ mean,
                                       float* __restrict__ variance,
                                       int N, int D) {
    int tid = blockIdx.x;
    if (tid >= N) return;
    
    float m = mean[tid];
    float sum = 0.0f;
    for (int i = 0; i < D; i++) {
        float diff = input[tid * D + i] - m;
        sum += diff * diff;
    }
    variance[tid] = sum / D;
}

__global__ void normalize_kernel(const float* __restrict__ input,
                                const float* __restrict__ mean,
                                const float* __restrict__ variance,
                                const float* __restrict__ gamma,
                                const float* __restrict__ beta,
                                float* __restrict__ output,
                                int N, int D, float eps) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * D;
    
    if (idx >= total_elements) return;
    
    int row = idx / D;
    int col = idx % D;
    
    if (row >= N) return;
    
    float m = mean[row];
    float v = variance[row];
    float x = input[idx];
    
    output[idx] = gamma[col] * (x - m) / sqrtf(v + eps) + beta[col];
}

void layernorm_naive(const float* input, const float* gamma, const float* beta,
                    float* output, float* mean, float* variance,
                    int N, int D, float eps, cudaStream_t stream) {
    compute_mean_kernel<<<N, 1, 0, stream>>>(input, mean, N, D);
    compute_variance_kernel<<<N, 1, 0, stream>>>(input, mean, variance, N, D);
    
    dim3 block(256);
    dim3 grid((N * D + block.x - 1) / block.x);
    normalize_kernel<<<grid, block, 0, stream>>>(input, mean, variance, 
                                                gamma, beta, output, N, D, eps);
}