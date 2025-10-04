#include "cross_entropy_loss.cuh"
#include <hip/hip_runtime.h>

template<typename T>
__global__ void cross_entropy_loss_kernel(
    const T* logits,
    const long* labels,
    T* loss,
    int batch_size,
    int num_classes
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_size) return;

    const T* row_logits = logits + idx * num_classes;
    long correct_class = labels[idx];

    // 1. Find max logit
    T max_logit = row_logits[0];
    for (int i = 1; i < num_classes; ++i) {
        max_logit = max(max_logit, row_logits[i]);
    }

    // 2. Sum of exps
    T sum_exp = 0;
    for (int i = 0; i < num_classes; ++i) {
        sum_exp += exp(row_logits[i] - max_logit);
    }

    // 3. Log of sum
    T log_sum_exp = log(sum_exp);

    // 4. Loss
    loss[idx] = -row_logits[correct_class] + max_logit + log_sum_exp;
}

void cross_entropy_loss_forward(
    const float* logits,
    const long* labels,
    float* loss,
    int batch_size,
    int num_classes
) {
    int threads = 256;
    int blocks = (batch_size + threads - 1) / threads;

    hipLaunchKernelGGL(cross_entropy_loss_kernel, dim3(blocks), dim3(threads), 0, 0, logits, labels, loss, batch_size, num_classes);
    hipDeviceSynchronize();
}
