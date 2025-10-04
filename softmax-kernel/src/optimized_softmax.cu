#include "optimized_softmax.cuh"
#include <hip/hip_runtime.h>

template<typename T>
__global__ void optimized_softmax_kernel(
    const T* input,
    T* output,
    int rows,
    int cols
) {
    extern __shared__ T shared_data[];

    int row_idx = blockIdx.x;
    if (row_idx >= rows) return;

    const T* row_input = input + row_idx * cols;
    T* row_output = output + row_idx * cols;

    // Load data to shared memory
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        shared_data[i] = row_input[i];
    }
    __syncthreads();

    // Find max value using reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] = max(shared_data[threadIdx.x], shared_data[threadIdx.x + s]);
        }
        __syncthreads();
    }
    T max_val = shared_data[0];
    __syncthreads();

    // Compute exp and sum
    T sum = 0;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        T exp_val = exp(row_input[i] - max_val);
        shared_data[i] = exp_val;
        sum += exp_val;
    }
    __syncthreads();

    // Reduce sum
    shared_data[threadIdx.x] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            shared_data[threadIdx.x] += shared_data[threadIdx.x + s];
        }
        __syncthreads();
    }
    T total_sum = shared_data[0];
    __syncthreads();

    // Re-calculate exp and normalize
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        row_output[i] = exp(row_input[i] - max_val) / total_sum;
    }
}

void optimized_softmax_forward(
    const float* input,
    float* output,
    int rows,
    int cols
) {
    int threads = 256;
    if (cols < 256) {
        threads = cols;
    }
    int blocks = rows;
    int shared_mem_size = cols * sizeof(float);

    hipLaunchKernelGGL(optimized_softmax_kernel, dim3(blocks), dim3(threads), shared_mem_size, 0, input, output, rows, cols);
    hipDeviceSynchronize();
}
