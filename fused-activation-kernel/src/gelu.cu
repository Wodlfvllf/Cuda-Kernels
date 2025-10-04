#include "gelu.cuh"
#include <hip/hip_runtime.h>

template<typename T>
__global__ void gelu_kernel(
    const T* input,
    T* output,
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        T x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    }
}

void gelu_forward(
    const float* input,
    float* output,
    int n
) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    hipLaunchKernelGGL(gelu_kernel, dim3(blocks), dim3(threads), 0, 0, input, output, n);
    hipDeviceSynchronize();
}
