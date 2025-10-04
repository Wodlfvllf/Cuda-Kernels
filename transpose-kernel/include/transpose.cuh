#ifndef TRANSPOSE_CUH
#define TRANSPOSE_CUH

#include <hip/hip_runtime.h> 

template<typename T>
__global__ void naive_transpose_kernel(
    const T* input,
    T* output,
    int M,
    int N
);

void naive_transpose_forward(
    const float* input,
    float* output,
    int M,
    int N
);

#endif
