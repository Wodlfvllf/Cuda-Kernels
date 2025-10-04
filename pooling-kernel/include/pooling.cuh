#ifndef POOLING_CUH
#define POOLING_CUH

#include <hip/hip_runtime.h> 

template<typename T>
__global__ void max_pool2d_kernel(
    const T* input,
    T* output,
    int N,
    int C,
    int H,
    int W,
    int kH,
    int kW,
    int sH,
    int sW,
    int H_out,
    int W_out
);

void max_pool2d_forward(
    const float* input,
    float* output,
    int N,
    int C,
    int H,
    int W,
    int kH,
    int kW,
    int sH,
    int sW,
    int H_out,
    int W_out
);

#endif
