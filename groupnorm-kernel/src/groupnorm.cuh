#ifndef GROUPNORM_CUH
#define GROUPNORM_CUH

#include <hip/hip_runtime.h>

template<typename T>
__global__ void groupnorm_kernel(
    const T* input,
    const T* gamma,
    const T* beta,
    T* output,
    int N,
    int C,
    int H,
    int W,
    int G,
    T epsilon
);

void groupnorm_forward(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int N,
    int C,
    int H,
    int W,
    int G,
    float epsilon
);

#endif
