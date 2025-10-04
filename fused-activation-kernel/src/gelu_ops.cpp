#include <torch/extension.h>
#include "gelu.cuh"

torch::Tensor gelu_forward_torch(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(input);
    int n = input.numel();

    gelu_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("gelu_forward", &gelu_forward_torch, "GELU forward pass (CUDA)");
}
