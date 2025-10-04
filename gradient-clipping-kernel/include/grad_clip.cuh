#ifndef GRAD_CLIP_CUH
#define GRAD_CLIP_CUH

#include <hip/hip_runtime.h>

template<typename T>
__global__ void sum_squares_kernel(
    const T* input,
    T* output,
    int n
);

template<typename T>
__global__ void scale_kernel(
    T* input,
    T factor,
    int n
);

void sum_squares_forward(
    const float* input,
    float* output,
    int n
);

void scale_forward(
    float* input,
    float factor,
    int n
);

#endif
