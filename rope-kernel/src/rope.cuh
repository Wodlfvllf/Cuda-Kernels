#ifndef ROPE_CUH
#define ROPE_CUH

#include <hip/hip_runtime.h>

template<typename T>
__global__ void rope_kernel(
    const T* input,
    const T* cos_sin_table,
    T* output,
    int batch_size,
    int seq_len,
    int hidden_dim
);

void rope_forward(
    const float* input,
    const float* cos_sin_table,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_dim
);

#endif
