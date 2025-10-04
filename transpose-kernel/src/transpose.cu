#include "transpose.cuh"
#include <hip/hip_runtime.h>

template<typename T>
__global__ void naive_transpose_kernel(
    const T* input,
    T* output,
    int M,
    int N
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < N && y < M) {
        output[x * M + y] = input[y * N + x];
    }
}

void naive_transpose_forward(
    const float* input,
    float* output,
    int M,
    int N
) {
    dim3 threads(16, 16);
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    hipLaunchKernelGGL(naive_transpose_kernel, blocks, threads, 0, 0, input, output, M, N);
    hipDeviceSynchronize();
}
