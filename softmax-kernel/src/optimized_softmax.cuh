#ifndef OPTIMIZED_SOFTMAX_CUH
#define OPTIMIZED_SOFTMAX_CUH

#include <hip/hip_runtime.h>

template<typename T>
__global__ void optimized_softmax_kernel(
    const T* input,
    T* output,
    int rows,
    int cols
);

void optimized_softmax_forward(
    const float* input,
    float* output,
    int rows,
    int cols
);

#endif
