#include "optimized_transpose.cuh"
#include <hip/hip_runtime.h>

#define TILE_DIM 16
#define TILE_PADDING 1

template<typename T>
__global__ void optimized_transpose_kernel(
    const T* input,
    T* output,
    int M,
    int N
) {
    __shared__ T tile[TILE_DIM][TILE_DIM + TILE_PADDING];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    if (y < M && x < N) {
        tile[threadIdx.y][threadIdx.x] = input[y * N + x];
    }

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    if (y < N && x < M) {
        output[y * M + x] = tile[threadIdx.x][threadIdx.y];
    }
}

void optimized_transpose_forward(
    const float* input,
    float* output,
    int M,
    int N
) {
    dim3 threads(TILE_DIM, TILE_DIM);
    dim3 blocks(
        (N + threads.x - 1) / threads.x,
        (M + threads.y - 1) / threads.y
    );

    hipLaunchKernelGGL(optimized_transpose_kernel, blocks, threads, 0, 0, input, output, M, N);
    hipDeviceSynchronize();
}
