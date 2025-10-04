#ifndef OPTIMIZED_RMSNORM_CUH
#define OPTIMIZED_RMSNORM_CUH

#include <hip/hip_runtime.h>

template<typename T>
__global__ void optimized_rmsnorm_kernel(
    const T* input,
    const T* gamma,
    T* output,
    int rows,
    int cols,
    T epsilon
);

void optimized_rmsnorm_forward(
    const float* input,
    const float* gamma,
    float* output,
    int rows,
    int cols,
    float epsilon
);

#endif
