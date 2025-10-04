#ifndef ADAMW_CUH
#define ADAMW_CUH

#include <hip/hip_runtime.h>

template<typename T>
__global__ void adamw_kernel(
    T* params,
    const T* grads,
    T* exp_avg,
    T* exp_avg_sq,
    float beta1,
    float beta2,
    float lr,
    float weight_decay,
    float eps,
    int step,
    int num_elements
);

void adamw_forward(
    float* params,
    const float* grads,
    float* exp_avg,
    float* exp_avg_sq,
    float beta1,
    float beta2,
    float lr,
    float weight_decay,
    float eps,
    int step,
    int num_elements
);

#endif
