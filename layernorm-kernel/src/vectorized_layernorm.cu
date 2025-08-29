#include "vectorized_layernorm.cuh"
#include "common.cuh"
#include <cmath>

template<int BLOCK_SIZE>
__global__ void layernorm_vectorized_kernel(const float* __restrict__ input,
                                           const float* __restrict__ gamma,
                                           const float* __restrict__ beta,
                                           float* __restrict__ output,
                                           int N, int D, float eps) {
    const int vec_size = 4;
    const int D_vec = D / vec_size;
    
    int tid = threadIdx.x;
    int row = blockIdx.x;
    
    if (row >= N) return;
    
    const float4* input_row = reinterpret_cast<const float4*>(input + row * D);
    float4* output_row = reinterpret_cast<float4*>(output + row * D);
    const float4* gamma_vec = reinterpret_cast<const float4*>(gamma);
    const float4* beta_vec = reinterpret_cast<const float4*>(beta);
    
    // Compute mean with vectorized loads
    float sum = 0.0f;
    for (int i = tid; i < D_vec; i += BLOCK_SIZE) {
        float4 val = input_row[i];
        sum += val.x + val.y + val.z + val.w;
    }
    
    sum = block_reduce_sum<float>(sum);
    __shared__ float mean;
    if (tid == 0) {
        mean = sum / D;
    }
    __syncthreads();
    
    // Compute variance with vectorized loads
    float var_sum = 0.0f;
    for (int i = tid; i < D_vec; i += BLOCK_SIZE) {
        float4 val = input_row[i];
        float4 diff;
        diff.x = val.x - mean;
        diff.y = val.y - mean;
        diff.z = val.z - mean;
        diff.w = val.w - mean;
        var_sum += diff.x * diff.x + diff.y * diff.y + 
                   diff.z * diff.z + diff.w * diff.w;
    }
    
    var_sum = block_reduce_sum<float>(var_sum);
    __shared__ float inv_std;
    if (tid == 0) {
        inv_std = rsqrtf(var_sum / D + eps);
    }
    __syncthreads();
    
    // Normalize with vectorized stores
    for (int i = tid; i < D_vec; i += BLOCK_SIZE) {
        float4 val = input_row[i];
        float4 g = gamma_vec[i];
        float4 b = beta_vec[i];
        float4 result;
        
        result.x = g.x * (val.x - mean) * inv_std + b.x;
        result.y = g.y * (val.y - mean) * inv_std + b.y;
        result.z = g.z * (val.z - mean) * inv_std + b.z;
        result.w = g.w * (val.w - mean) * inv_std + b.w;
        
        output_row[i] = result;
    }
    
    // Handle remaining elements if D is not divisible by 4
    if (tid == 0 && (D % vec_size != 0)) {
        for (int i = D_vec * vec_size; i < D; i++) {
            float val = input[row * D + i];
            float normalized = (val - mean) * inv_std;
            output[row * D + i] = gamma[i] * normalized + beta[i];
        }
    }
}

// Explicit instantiation
template __global__ void layernorm_vectorized_kernel<128>(const float*, const float*, 
                                                          const float*, float*, int, int, float);
template __global__ void layernorm_vectorized_kernel<256>(const float*, const float*, 
                                                          const float*, float*, int, int, float);

void layernorm_vectorized(const float* input, const float* gamma, const float* beta,
                         float* output, int N, int D, float eps, cudaStream_t stream) {
    const int block_size = 256;
    layernorm_vectorized_kernel<block_size><<<N, block_size, 0, stream>>>(
        input, gamma, beta, output, N, D, eps);
}