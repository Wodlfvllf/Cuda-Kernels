#include <torch/extension.h>
#include "adamw.cuh"

void adamw_forward_torch(
    torch::Tensor params,
    torch::Tensor grads,
    torch::Tensor exp_avg,
    torch::Tensor exp_avg_sq,
    float beta1,
    float beta2,
    float lr,
    float weight_decay,
    float eps,
    int step
) {
    TORCH_CHECK(params.is_cuda(), "Params must be a CUDA tensor");
    TORCH_CHECK(params.is_contiguous(), "Params must be contiguous");
    TORCH_CHECK(grads.is_cuda(), "Grads must be a CUDA tensor");
    TORCH_CHECK(grads.is_contiguous(), "Grads must be contiguous");
    TORCH_CHECK(exp_avg.is_cuda(), "exp_avg must be a CUDA tensor");
    TORCH_CHECK(exp_avg.is_contiguous(), "exp_avg must be contiguous");
    TORCH_CHECK(exp_avg_sq.is_cuda(), "exp_avg_sq must be a CUDA tensor");
    TORCH_CHECK(exp_avg_sq.is_contiguous(), "exp_avg_sq must be contiguous");

    TORCH_CHECK(params.sizes() == grads.sizes(), "Params and Grads must have the same size");
    TORCH_CHECK(params.sizes() == exp_avg.sizes(), "Params and exp_avg must have the same size");
    TORCH_CHECK(params.sizes() == exp_avg_sq.sizes(), "Params and exp_avg_sq must have the same size");

    int num_elements = params.numel();

    adamw_forward(
        params.data_ptr<float>(),
        grads.data_ptr<float>(),
        exp_avg.data_ptr<float>(),
        exp_avg_sq.data_ptr<float>(),
        beta1,
        beta2,
        lr,
        weight_decay,
        eps,
        step,
        num_elements
    );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("adamw_forward", &adamw_forward_torch, "AdamW forward pass (CUDA)");
}
