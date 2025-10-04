#include <torch/extension.h>
#include "groupnorm.cuh"

void groupnorm_forward_torch(
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    torch::Tensor output,
    int G,
    float epsilon
) {
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    TORCH_CHECK(gamma.is_cuda(), "Gamma must be a CUDA tensor");
    TORCH_CHECK(gamma.is_contiguous(), "Gamma must be contiguous");
    TORCH_CHECK(beta.is_cuda(), "Beta must be a CUDA tensor");
    TORCH_CHECK(beta.is_contiguous(), "Beta must be contiguous");
    TORCH_CHECK(output.is_cuda(), "Output must be a CUDA tensor");
    TORCH_CHECK(output.is_contiguous(), "Output must be contiguous");

    auto input_sizes = input.sizes();
    TORCH_CHECK(input.dim() == 4, "Input must be a 4D tensor");
    int N = input_sizes[0];
    int C = input_sizes[1];
    int H = input_sizes[2];
    int W = input_sizes[3];

    TORCH_CHECK(C % G == 0, "Number of channels must be divisible by number of groups");

    groupnorm_forward(
        input.data_ptr<float>(),
        gamma.data_ptr<float>(),
        beta.data_ptr<float>(),
        output.data_ptr<float>(),
        N, C, H, W, G, epsilon
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("groupnorm_forward", &groupnorm_forward_torch, "GroupNorm forward pass (CUDA)");
}
