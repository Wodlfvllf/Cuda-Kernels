#include <torch/extension.h>
#include "transpose.cuh"

torch::Tensor naive_transpose_forward_torch(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 2, "Input must be a 2D tensor");

    auto sizes = input.sizes();
    int M = sizes[0];
    int N = sizes[1];

    auto output = torch::empty({N, M}, input.options());

    naive_transpose_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        M,
        N
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_transpose_forward", &naive_transpose_forward_torch, "Naive transpose forward pass (CUDA)");
}
