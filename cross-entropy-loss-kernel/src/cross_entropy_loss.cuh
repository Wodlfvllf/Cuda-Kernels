#ifndef CROSS_ENTROPY_LOSS_CUH
#define CROSS_ENTROPY_LOSS_CUH

#include <hip/hip_runtime.h>

template<typename T>
__global__ void cross_entropy_loss_kernel(
    const T* logits,
    const long* labels,
    T* loss,
    int batch_size,
    int num_classes
);

void cross_entropy_loss_forward(
    const float* logits,
    const long* labels,
    float* loss,
    int batch_size,
    int num_classes
);

#endif
