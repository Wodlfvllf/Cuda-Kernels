#include "groupnorm.cuh"
#include <hip/hip_runtime.h>

template<typename T>
__global__ void groupnorm_kernel(
    const T* input,
    const T* gamma,
    const T* beta,
    T* output,
    int N,
    int C,
    int H,
    int W,
    int G,
    T epsilon
) {
    int group_idx = blockIdx.x;
    int n_idx = group_idx / G;
    int g_idx = group_idx % G;

    int C_per_G = C / G;
    int group_size = C_per_G * H * W;

    const T* x = input + n_idx * C * H * W + g_idx * C_per_G * H * W;
    T* y = output + n_idx * C * H * W + g_idx * C_per_G * H * W;

    // Compute mean and variance
    T sum = 0;
    T sum_sq = 0;
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        T val = x[i];
        sum += val;
        sum_sq += val * val;
    }

    // Reduction in shared memory
    extern __shared__ T shared_data[];
    shared_data[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum = shared_data[0];

    shared_data[threadIdx.x] = sum_sq;
    __syncthreads();
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    sum_sq = shared_data[0];

    T mean = sum / group_size;
    T var = sum_sq / group_size - mean * mean;
    T std = sqrt(var + epsilon);
    T r_std = 1.0f / std;

    // Normalize
    for (int i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c_idx = (i / (H * W)) + g_idx * C_per_G;
        y[i] = (x[i] - mean) * r_std * gamma[c_idx] + beta[c_idx];
    }
}

void groupnorm_forward(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int N,
    int C,
    int H,
    int W,
    int G,
    float epsilon
) {
    int threads = 256;
    int blocks = N * G;
    int shared_mem_size = threads * sizeof(float);

    hipLaunchKernelGGL(groupnorm_kernel, dim3(blocks), dim3(threads), shared_mem_size, 0, input, gamma, beta, output, N, C, H, W, G, epsilon);
    hipDeviceSynchronize();
}
