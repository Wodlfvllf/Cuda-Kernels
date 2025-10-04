#ifndef OPTIMIZED_TRANSPOSE_CUH
#define OPTIMIZED_TRANSPOSE_CUH

#include <hip/hip_runtime.h>

template<typename T>
__global__ void optimized_transpose_kernel(
    const T* input,
    T* output,
    int M,
    int N
);

void optimized_transpose_forward(
    const float* input,
    float* output,
    int M,
    int N
);

#endif
