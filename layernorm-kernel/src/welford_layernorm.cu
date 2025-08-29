#include "welford_layernorm.cuh"
#include "common.cuh"
#include <cmath>

template<int BLOCK_SIZE>
__global__ void layernorm_welford_kernel(const float* __restrict__ input,
                                        const float* __restrict__ gamma,
                                        const float* __restrict__ beta,
                                        float* __restrict__ output,
                                        int N, int D, float eps) {
    extern __shared__ float shared[];
    float* shared_data = shared;
    float* shared_mean = &shared[D];
    float* shared_m2 = &shared_mean[32];
    int* shared_count = (int*)&shared_m2[32];
    
    int tid = threadIdx.x;
    int row = blockIdx.x;
    
    if (row >= N) return;
    
    const float* input_row = input + row * D;
    float* output_row = output + row * D;
    
    // Welford's algorithm for mean and variance in single pass
    float mean = 0.0f;
    float m2 = 0.0f;
    int count = 0;
    
    // Process elements with grid-stride loop
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float val = input_row[i];
        shared_data[i] = val;  // Cache for later use
        
        count++;
        float delta = val - mean;
        mean += delta / count;
        float delta2 = val - mean;
        m2 += delta * delta2;
    }
    
    // Reduce across threads
    int lane = tid % 32;
    int wid = tid / 32;
    
    // Warp-level reduction using Welford's method
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other_mean = __shfl_xor_sync(0xffffffff, mean, offset);
        float other_m2 = __shfl_xor_sync(0xffffffff, m2, offset);
        int other_count = __shfl_xor_sync(0xffffffff, count, offset);
        
        if (other_count > 0) {
            float delta = other_mean - mean;
            int total_count = count + other_count;
            float count_ratio = (float)count * other_count / total_count;
            mean = (mean * count + other_mean * other_count) / total_count;
            m2 = m2 + other_m2 + delta * delta * count_ratio;
            count = total_count;
        }
    }
    
    if (lane == 0) {
        shared_mean[wid] = mean;
        shared_m2[wid] = m2;
        shared_count[wid] = count;
    }
    __syncthreads();
    
    // Final reduction
    if (tid == 0) {
        mean = shared_mean[0];
        m2 = shared_m2[0];
        count = shared_count[0];
        
        for (int i = 1; i < (BLOCK_SIZE + 31) / 32; i++) {
            if (shared_count[i] > 0) {
                float delta = shared_mean[i] - mean;
                int total_count = count + shared_count[i];
                float count_ratio = (float)count * shared_count[i] / total_count;
                mean = (mean * count + shared_mean[i] * shared_count[i]) / total_count;
                m2 = m2 + shared_m2[i] + delta * delta * count_ratio;
                count = total_count;
            }
        }
        
        shared_mean[0] = mean;
        shared_m2[0] = rsqrtf(m2 / D + eps);
    }
    __syncthreads();
    
    mean = shared_mean[0];
    float inv_std = shared_m2[0];
    
    // Apply normalization
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float normalized = (shared_data[i] - mean) * inv_std;
        output_row[i] = gamma[i] * normalized + beta[i];
    }
}

// Explicit instantiation
template __global__ void layernorm_welford_kernel<256>(const float*, const float*, 
                                                       const float*, float*, int, int, float);

void layernorm_welford(const float* input, const float* gamma, const float* beta,
                      float* output, int N, int D, float eps, cudaStream_t stream) {
    const int block_size = 256;
    int shared_mem_size = D * sizeof(float) + 64 * sizeof(float) + 32 * sizeof(int);
    
    layernorm_welford_kernel<block_size><<<N, block_size, shared_mem_size, stream>>>(
        input, gamma, beta, output, N, D, eps);
}