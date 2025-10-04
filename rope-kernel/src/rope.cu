#include "rope.cuh"
#include <hip/hip_runtime.h>

template<typename T>
__global__ void rope_kernel(
    const T* input,
    const T* cos_sin_table,
    T* output,
    int batch_size,
    int seq_len,
    int hidden_dim
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int half_hidden_dim = hidden_dim / 2;
    int total_pairs = batch_size * seq_len * half_hidden_dim;
    
    if (idx >= total_pairs) return;

    int pair_h_dim_idx = idx % half_hidden_dim;
    int seq_idx = (idx / half_hidden_dim) % seq_len;
    int batch_idx = idx / (seq_len * half_hidden_dim);

    int d_idx = pair_h_dim_idx * 2;

    int input_idx1 = batch_idx * seq_len * hidden_dim + seq_idx * hidden_dim + d_idx;
    int input_idx2 = input_idx1 + 1;

    T x1 = input[input_idx1];
    T x2 = input[input_idx2];

    const T* cos_sin_row = cos_sin_table + seq_idx * hidden_dim;
    T cos_val = cos_sin_row[d_idx];
    T sin_val = cos_sin_row[d_idx + 1];

    output[input_idx1] = x1 * cos_val - x2 * sin_val;
    output[input_idx2] = x1 * sin_val + x2 * cos_val;
}


void rope_forward(
    const float* input,
    const float* cos_sin_table,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_dim
) {
    int threads = 256;
    int total_pairs = batch_size * seq_len * hidden_dim / 2;
    int blocks = (total_pairs + threads - 1) / threads;

    hipLaunchKernelGGL(rope_kernel, dim3(blocks), dim3(threads), 0, 0, input, cos_sin_table, output, batch_size, seq_len, hidden_dim);
    hipDeviceSynchronize();
}
