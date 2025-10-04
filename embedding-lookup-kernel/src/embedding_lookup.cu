#include "embedding_lookup.cuh"
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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = batch_size * seq_len * hidden_dim;

    if (idx >= total_elements) return;

    int h_dim_idx = idx % hidden_dim;
    int lookup_idx = idx / hidden_dim;
    
    long embedding_idx = indices[lookup_idx];

    if (embedding_idx < 0 || embedding_idx >= vocab_size) {
        output[idx] = 0;
    } else {
        output[idx] = embeddings[embedding_idx * hidden_dim + h_dim_idx];
    }
}

void embedding_lookup_forward(
    const long* indices,
    const float* embeddings,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_dim,
    int vocab_size
) {
    int total_elements = batch_size * seq_len * hidden_dim;
    dim3 threads(256);
    dim3 blocks((total_elements + threads.x - 1) / threads.x);

    hipLaunchKernelGGL(embedding_lookup_kernel, blocks, threads, 0, 0, indices, embeddings, output, batch_size, seq_len, hidden_dim, vocab_size);
    hipDeviceSynchronize();
}
