#ifndef GELU_CUH
#define GELU_CUH

#include <hip/hip_runtime.h>

template<typename T>
__global__ void gelu_kernel(
    const T* input,
    T* output,
    int n
);

void gelu_forward(
    const float* input,
    float* output,
    int n
);

#endif
