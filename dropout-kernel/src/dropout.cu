#include "dropout.cuh"
#include <hip/hip_runtime.h>

// Simple LCG random number generator
__device__ unsigned int lcg_rand(unsigned int &state) {
    const unsigned int a = 1103515245;
    const unsigned int c = 12345;
    state = a * state + c;
    return state >> 16; // Use the upper 16 bits
}

template<typename T>
__global__ void dropout_kernel(
    const T* input,
    T* output,
    float p,
    unsigned long long seed,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    // Initialize random state for each thread
    unsigned int rand_state = seed + idx;

    // Generate a random number between 0 and 1
    float rand_val = (float)lcg_rand(rand_state) / (float)0xFFFF;

    if (rand_val < p) {
        output[idx] = 0;
    } else {
        output[idx] = input[idx] / (1.0f - p);
    }
}

void dropout_forward(
    const float* input,
    float* output,
    float p,
    unsigned long long seed,
    int n
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    hipLaunchKernelGGL(dropout_kernel, dim3(blocks), dim3(threads), 0, 0, input, output, p, seed, n);
    hipDeviceSynchronize();
}
