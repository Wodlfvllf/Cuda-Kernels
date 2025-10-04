#include "naive_rmsnorm.cuh"
#include <hip/hip_runtime.h>

template<typename T>
__global__ void naive_rmsnorm_kernel(
    const T* input,
    const T* gamma,
    T* output,
    int rows,
    int cols,
    T epsilon
) {
    int row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= rows) return;

    const T* row_input = input + row_idx * cols;
    T* row_output = output + row_idx * cols;

    // 1. Calculate sum of squares
    T sum_sq = 0;
    for (int i = 0; i < cols; ++i) {
        sum_sq += row_input[i] * row_input[i];
    }

    // 2. Calculate mean and rsqrt
    T mean_sq = sum_sq / cols;
    T rrms = rsqrt(mean_sq + epsilon);

    // 3. Normalize and apply gamma
    for (int i = 0; i < cols; ++i) {
        row_output[i] = row_input[i] * rrms * gamma[i];
    }
}

void naive_rmsnorm_forward(
    const float* input,
    const float* gamma,
    float* output,
    int rows,
    int cols,
    float epsilon
) {
    int threads = 256;
    int blocks = (rows + threads - 1) / threads;

    hipLaunchKernelGGL(naive_rmsnorm_kernel, dim3(blocks), dim3(threads), 0, 0, input, gamma, output, rows, cols, epsilon);
    hipDeviceSynchronize();
}
