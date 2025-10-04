#include "grad_clip.cuh"
#include <hip/hip_runtime.h>

template<typename T>
__global__ void sum_squares_kernel(
    const T* input,
    T* output,
    int n
) {
    extern __shared__ T shared_data[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    T sum = 0;
    while (i < n) {
        T val = input[i];
        sum += val * val;
        i += gridDim.x * blockDim.x;
    }
    shared_data[tid] = sum;
    __syncthreads();

    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            shared_data[tid] += shared_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = shared_data[0];
    }
}

template<typename T>
__global__ void scale_kernel(
    T* input,
    T factor,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        input[idx] *= factor;
    }
}

void sum_squares_forward(
    const float* input,
    float* output,
    int n
) {
    int threads = 256;
    int blocks = 256; // This should be tuned
    int shared_mem_size = threads * sizeof(float);

    hipLaunchKernelGGL(sum_squares_kernel, dim3(blocks), dim3(threads), shared_mem_size, 0, input, output, n);
    hipDeviceSynchronize();
}

void scale_forward(
    float* input,
    float factor,
    int n
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    hipLaunchKernelGGL(scale_kernel, dim3(blocks), dim3(threads), 0, 0, input, factor, n);
    hipDeviceSynchronize();
}
