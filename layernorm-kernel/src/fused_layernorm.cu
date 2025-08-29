#include "fused_layernorm.cuh"
#include "common.cuh"
#include <cmath>

template<int BLOCK_SIZE>
__global__ void layernorm_fused_kernel(const float* __restrict__ input,
                                      const float* __restrict__ gamma,
                                      const float* __restrict__ beta,
                                      float* __restrict__ output,
                                      int N, int D, float eps) {
    extern __shared__ float shared_data[];
    
    int tid = threadIdx.x;
    int row = blockIdx.x;
    
    if (row >= N) return;
    
    const float* input_row = input + row * D;
    float* output_row = output + row * D;
    
    // Phase 1: Compute mean using block reduction
    float sum = 0.0f;
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float val = input_row[i];
        // Fixed: proper indexing for shared memory
        if (i < D) {
            shared_data[i] = val;
        }
        sum += val;
    }
    
    sum = block_reduce_sum<float>(sum);
    __shared__ float mean;
    if (tid == 0) {
        mean = sum / D;
    }
    __syncthreads();
    
    // Phase 2: Compute variance using block reduction
    float var_sum = 0.0f;
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float diff = shared_data[i] - mean;
        var_sum += diff * diff;
    }
    
    var_sum = block_reduce_sum<float>(var_sum);
    __shared__ float variance;
    if (tid == 0) {
        variance = var_sum / D;
    }
    __syncthreads();
    
    // Phase 3: Normalize and apply affine transformation
    float inv_std = rsqrtf(variance + eps);
    for (int i = tid; i < D; i += BLOCK_SIZE) {
        float normalized = (shared_data[i] - mean) * inv_std;
        output_row[i] = gamma[i] * normalized + beta[i];
    }
}

// Explicit instantiation
template __global__ void layernorm_fused_kernel<128>(const float*, const float*, 
                                                     const float*, float*, int, int, float);
template __global__ void layernorm_fused_kernel<256>(const float*, const float*, 
                                                     const float*, float*, int, int, float);
template __global__ void layernorm_fused_kernel<512>(const float*, const float*, 
                                                     const float*, float*, int, int, float);

void layernorm_fused(const float* input, const float* gamma, const float* beta,
                    float* output, int N, int D, float eps, cudaStream_t stream) {
    const int block_size = 256;
    int shared_mem_size = D * sizeof(float);
    
    // Check shared memory limit
    int device;
    cudaGetDevice(&device);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    if (shared_mem_size > prop.sharedMemPerBlock) {
        // Fall back to smaller block size or alternative implementation
        shared_mem_size = prop.sharedMemPerBlock;
    }
    
    layernorm_fused_kernel<block_size><<<N, block_size, shared_mem_size, stream>>>(
        input, gamma, beta, output, N, D, eps);
}