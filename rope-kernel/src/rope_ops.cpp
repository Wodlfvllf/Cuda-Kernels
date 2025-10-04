#include <torch/extension.h>
#include "rope.cuh"

torch::Tensor rope_forward_torch(
    torch::Tensor input,
    torch::Tensor cos_sin_table
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(cos_sin_table.is_cuda(), "cos_sin_table must be a CUDA tensor");
    TORCH_CHECK(cos_sin_table.is_contiguous(), "cos_sin_table must be contiguous");

    auto input_sizes = input.sizes();
    auto table_sizes = cos_sin_table.sizes();

    TORCH_CHECK(input.dim() == 3, "Input must be a 3D tensor");
    int batch_size = input_sizes[0];
    int seq_len = input_sizes[1];
    int hidden_dim = input_sizes[2];

    TORCH_CHECK(table_sizes.size() == 2 && table_sizes[0] == seq_len && table_sizes[1] == hidden_dim, "cos_sin_table must be a 2D tensor of shape (seq_len, hidden_dim)");
    TORCH_CHECK(hidden_dim % 2 == 0, "hidden_dim must be even");

    auto output = torch::empty_like(input);

    rope_forward(
        input.data_ptr<float>(),
        cos_sin_table.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len,
        hidden_dim
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("rope_forward", &rope_forward_torch, "RoPE forward pass (CUDA)");
}
