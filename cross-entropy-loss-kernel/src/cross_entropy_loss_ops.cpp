#include <torch/extension.h>
#include "cross_entropy_loss.cuh"

torch::Tensor cross_entropy_loss_forward_torch(
    torch::Tensor logits,
    torch::Tensor labels
) {
    TORCH_CHECK(logits.is_cuda(), "Logits must be a CUDA tensor");
    TORCH_CHECK(logits.is_contiguous(), "Logits must be contiguous");
    TORCH_CHECK(labels.is_cuda(), "Labels must be a CUDA tensor");
    TORCH_CHECK(labels.is_contiguous(), "Labels must be contiguous");
    TORCH_CHECK(labels.scalar_type() == torch::kLong, "Labels must be of type long");

    auto logits_sizes = logits.sizes();
    auto labels_sizes = labels.sizes();

    TORCH_CHECK(logits.dim() == 2, "Logits must be a 2D tensor");
    int batch_size = logits_sizes[0];
    int num_classes = logits_sizes[1];

    TORCH_CHECK(labels.dim() == 1 && labels_sizes[0] == batch_size, "Labels must be a 1D tensor of size batch_size");

    auto loss = torch::empty({batch_size}, logits.options());

    cross_entropy_loss_forward(
        logits.data_ptr<float>(),
        labels.data_ptr<long>(),
        loss.data_ptr<float>(),
        batch_size,
        num_classes
    );

    return loss;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cross_entropy_loss_forward", &cross_entropy_loss_forward_torch, "Cross-Entropy Loss forward pass (CUDA)");
}
