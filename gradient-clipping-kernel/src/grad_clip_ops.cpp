#include <torch/extension.h>
#include "grad_clip.cuh"

void sum_squares_forward_torch(
    torch::Tensor input,
    torch::Tensor output
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(output.is_cuda(), "Output must be a CUDA tensor");
    TORCH_CHECK(output.is_contiguous(), "Output must be contiguous");

    int n = input.numel();
    sum_squares_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
}

void scale_forward_torch(
    torch::Tensor input,
    float factor
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    int n = input.numel();
    scale_forward(
        input.data_ptr<float>(),
        factor,
        n
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("sum_squares_forward", &sum_squares_forward_torch, "Sum of squares forward pass (CUDA)");
    m.def("scale_forward", &scale_forward_torch, "Scale forward pass (CUDA)");
}
