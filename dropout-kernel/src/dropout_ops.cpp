#include <torch/extension.h>
#include "dropout.cuh"

torch::Tensor dropout_forward_torch(
    torch::Tensor input,
    float p,
    unsigned long long seed
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");

    auto output = torch::empty_like(input);
    int n = input.numel();

    dropout_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        p,
        seed,
        n
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("dropout_forward", &dropout_forward_torch, "Dropout forward pass (CUDA)");
}
