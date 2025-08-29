#ifdef TORCH_EXTENSION

#include <torch/extension.h>
#include <cuda_runtime.h>
#include "fused_layernorm.cuh"
#include "welford_layernorm.cuh"

// Forward pass
torch::Tensor layernorm_forward(torch::Tensor input,
                               torch::Tensor gamma,
                               torch::Tensor beta,
                               float eps,
                               bool use_welford = false) {
    // Check inputs
    TORCH_CHECK(input.is_cuda(), "Input must be a CUDA tensor");
    TORCH_CHECK(gamma.is_cuda(), "Gamma must be a CUDA tensor");
    TORCH_CHECK(beta.is_cuda(), "Beta must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    const int N = input.size(0);
    const int D = input.size(1);
    
    TORCH_CHECK(gamma.size(0) == D, "Gamma size mismatch");
    TORCH_CHECK(beta.size(0) == D, "Beta size mismatch");
    
    auto output = torch::empty_like(input);
    
    if (use_welford) {
        layernorm_welford(
            input.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            output.data_ptr<float>(),
            N, D, eps,
            at::cuda::getCurrentCUDAStream()
        );
    } else {
        layernorm_fused(
            input.data_ptr<float>(),
            gamma.data_ptr<float>(),
            beta.data_ptr<float>(),
            output.data_ptr<float>(),
            N, D, eps,
            at::cuda::getCurrentCUDAStream()
        );
    }
    
    return output;
}

// Backward pass (simplified - you'd want a proper implementation)
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> layernorm_backward(
    torch::Tensor grad_output,
    torch::Tensor input,
    torch::Tensor gamma,
    torch::Tensor beta,
    float eps) {
    
    // This is a placeholder - implement proper backward pass
    auto grad_input = torch::empty_like(input);
    auto grad_gamma = torch::sum(grad_output * input, 0);
    auto grad_beta = torch::sum(grad_output, 0);
    
    return std::make_tuple(grad_input, grad_gamma, grad_beta);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &layernorm_forward, "LayerNorm forward (CUDA)",
          py::arg("input"), py::arg("gamma"), py::arg("beta"), 
          py::arg("eps"), py::arg("use_welford") = false);
    m.def("backward", &layernorm_backward, "LayerNorm backward (CUDA)");
}

#endif