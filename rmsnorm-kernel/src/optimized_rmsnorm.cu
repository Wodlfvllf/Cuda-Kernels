#include "optimized_rmsnorm.cuh"
#include <hip/hip_runtime.h>

template<typename T>
__global__ void optimized_rmsnorm_kernel(
    const T* input,
    const T* gamma,
    T* output,
    int rows,
    int cols,
    T epsilon
) {
    extern __shared__ T shared_data[];

    int row_idx = blockIdx.x;
    if (row_idx >= rows) return;

    const T* row_input = input + row_idx * cols;
    T* row_output = output + row_idx * cols;

    // Calculate sum of squares per thread
    T sum_sq = 0;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        T val = row_input[i];
        sum_sq += val * val;
    }

    // Reduce sum_sq in shared memory
    shared_data[threadIdx.x] = sum_sq;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    T total_sum_sq = shared_data[0];

    // Calculate rrms
    T mean_sq = total_sum_sq / cols;
    T rrms = rsqrt(mean_sq + epsilon);

    // Normalize and apply gamma
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_output[i] = row_input[i] * rrms * gamma[i];
    }
}

void optimized_rmsnorm_forward(
    const float* input,
    const float* gamma,
    float* output,
    int rows,
    int cols,
    float epsilon
) {
    int threads = 256;
    if (cols < 256) {
        threads = cols;
    }
    int blocks = rows;
    // Shared memory for reduction
    int shared_mem_size = threads * sizeof(float);

    hipLaunchKernelGGL(optimized_rmsnorm_kernel, dim3(blocks), dim3(threads), shared_mem_size, 0, input, gamma, output, rows, cols, epsilon);
    hipDeviceSynchronize();
}
