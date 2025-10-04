#include <torch/extension.h>
#include "naive_rmsnorm.cuh"
#include "optimized_rmsnorm.cuh"

torch::Tensor naive_rmsnorm_forward_torch(
    torch::Tensor input,
    torch::Tensor gamma,
    float epsilon
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(gamma.is_cuda(), "Gamma must be a CUDA tensor");
    TORCH_CHECK(gamma.is_contiguous(), "Gamma must be contiguous");

    auto input_sizes = input.sizes();
    auto gamma_sizes = gamma.sizes();

    int rows;
    int cols;

    if (input.dim() == 2) {
        rows = input_sizes[0];
        cols = input_sizes[1];
    } else if (input.dim() == 3) {
        rows = input_sizes[0] * input_sizes[1];
        cols = input_sizes[2];
    } else {
        TORCH_CHECK(false, "Input must be a 2D or 3D tensor");
    }

    TORCH_CHECK(gamma_sizes.size() == 1 && gamma_sizes[0] == cols, "Gamma must be a 1D tensor of size cols");

    auto output = torch::empty_like(input);

    naive_rmsnorm_forward(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols,
        epsilon
    );

    return output;
}

torch::Tensor optimized_rmsnorm_forward_torch(
    torch::Tensor input,
    torch::Tensor gamma,
    float epsilon
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(gamma.is_cuda(), "Gamma must be a CUDA tensor");
    TORCH_CHECK(gamma.is_contiguous(), "Gamma must be contiguous");

    auto input_sizes = input.sizes();
    auto gamma_sizes = gamma.sizes();

    int rows;
    int cols;

    if (input.dim() == 2) {
        rows = input_sizes[0];
        cols = input_sizes[1];
    } else if (input.dim() == 3) {
        rows = input_sizes[0] * input_sizes[1];
        cols = input_sizes[2];
    } else {
        TORCH_CHECK(false, "Input must be a 2D or 3D tensor");
    }

    TORCH_CHECK(gamma_sizes.size() == 1 && gamma_sizes[0] == cols, "Gamma must be a 1D tensor of size cols");

    auto output = torch::empty_like(input);

    optimized_rmsnorm_forward(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        output.data_ptr<float>(),
        rows,
        cols,
        epsilon
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_rmsnorm_forward", &naive_rmsnorm_forward_torch, "Naive RMSNorm forward pass (CUDA)");
    m.def("optimized_rmsnorm_forward", &optimized_rmsnorm_forward_torch, "Optimized RMSNorm forward pass (CUDA)");
}