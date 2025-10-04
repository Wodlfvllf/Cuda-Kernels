#include "pooling.cuh"
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
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = N * C * H_out * W_out;

    if (idx >= total_outputs) return;

    int w_out = idx % W_out;
    int h_out = (idx / W_out) % H_out;
    int c = (idx / (W_out * H_out)) % C;
    int n = idx / (C * W_out * H_out);

    int h_start = h_out * sH;
    int w_start = w_out * sW;

    T max_val = -1.0e+38; // A very small number

    for (int i = 0; i < kH; ++i) {
        for (int j = 0; j < kW; ++j) {
            int h = h_start + i;
            int w = w_start + j;
            if (h < H && w < W) {
                T val = input[n * C * H * W + c * H * W + h * W + w];
                max_val = max(max_val, val);
            }
        }
    }
    output[idx] = max_val;
}

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
) {
    int total_outputs = N * C * H_out * W_out;
    int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;

    hipLaunchKernelGGL(max_pool2d_kernel, dim3(blocks), dim3(threads), 0, 0, input, output, N, C, H, W, kH, kW, sH, sW, H_out, W_out);
    hipDeviceSynchronize();
}
