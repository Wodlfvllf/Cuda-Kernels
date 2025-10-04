#include <torch/extension.h>
#include "naive_softmax.cuh"
#include "optimized_softmax.cuh"

torch::Tensor naive_softmax_forward_torch(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 3, "Input must be a 3D tensor");

    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int seq_len = sizes[1];
    int hidden_dim = sizes[2];

    auto output = torch::empty_like(input);

    int rows = batch_size * seq_len;
    int cols = hidden_dim;

    naive_softmax_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );

    return output;
}

torch::Tensor optimized_softmax_forward_torch(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(input.dim() == 3, "Input must be a 3D tensor");

    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int seq_len = sizes[1];
    int hidden_dim = sizes[2];

    auto output = torch::empty_like(input);

    int rows = batch_size * seq_len;
    int cols = hidden_dim;

    optimized_softmax_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_softmax_forward", &naive_softmax_forward_torch, "Naive softmax forward pass (CUDA)");
    m.def("optimized_softmax_forward", &optimized_softmax_forward_torch, "Optimized softmax forward pass (CUDA)");
}