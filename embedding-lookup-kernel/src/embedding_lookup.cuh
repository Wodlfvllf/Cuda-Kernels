#ifndef EMBEDDING_LOOKUP_CUH
#define EMBEDDING_LOOKUP_CUH

#include <hip/hip_runtime.h>

template<typename T>
__global__ void embedding_lookup_kernel(
    const long* indices,
    const T* embeddings,
    T* output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int vocab_size
);

void embedding_lookup_forward(
    const long* indices,
    const float* embeddings,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int vocab_size
);

#endif
