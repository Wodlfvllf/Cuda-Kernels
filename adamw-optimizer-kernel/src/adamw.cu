#include "adamw.cuh"
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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    T grad = grads[idx];
    T param = params[idx];
    T m = exp_avg[idx];
    T v = exp_avg_sq[idx];

    // Weight decay
    param *= (1.0f - lr * weight_decay);

    // Update moments
    m = beta1 * m + (1.0f - beta1) * grad;
    v = beta2 * v + (1.0f - beta2) * grad * grad;

    // Bias correction
    float beta1_t = powf(beta1, step);
    float beta2_t = powf(beta2, step);
    float m_hat = m / (1.0f - beta1_t);
    float v_hat = v / (1.0f - beta2_t);

    // Update params
    param -= lr * m_hat / (sqrtf(v_hat) + eps);

    // Write back
    params[idx] = param;
    exp_avg[idx] = m;
    exp_avg_sq[idx] = v;
}

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
) {
    int threads = 256;
    int blocks = (num_elements + threads - 1) / threads;

    hipLaunchKernelGGL(adamw_kernel, dim3(blocks), dim3(threads), 0, 0, params, grads, exp_avg, exp_avg_sq, beta1, beta2, lr, weight_decay, eps, step, num_elements);
    hipDeviceSynchronize();
}
