#ifndef NAIVE_RMSNORM_CUH
#define NAIVE_RMSNORM_CUH

#include <hip/hip_runtime.h>

template<typename T>
__global__ void naive_rmsnorm_kernel(
    const T* input,
    const T* gamma,
    T* output,
    int rows,
    int cols,
    T epsilon
);

void naive_rmsnorm_forward(
    const float* input,
    const float* gamma,
    float* output,
    int rows,
    int cols,
    float epsilon
);

#endif
