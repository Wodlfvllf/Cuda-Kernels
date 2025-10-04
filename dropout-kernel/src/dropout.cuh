#ifndef DROPOUT_CUH
#define DROPOUT_CUH

#include <hip/hip_runtime.h>

template<typename T>
__global__ void dropout_kernel(
    const T* input,
    T* output,
    float p,
    unsigned long long seed,
    int n
);

void dropout_forward(
    const float* input,
    float* output,
    float p,
    unsigned long long seed,
    int n
);

#endif
