#include <torch/extension.h>
#include "embedding_lookup.cuh"

torch::Tensor embedding_lookup_forward_torch(
    torch::Tensor indices,
    torch::Tensor embeddings
) {
    TORCH_CHECK(indices.is_cuda(), "Indices must be a CUDA tensor");
    TORCH_CHECK(indices.is_contiguous(), "Indices must be contiguous");
    TORCH_CHECK(indices.scalar_type() == torch::kLong, "Indices must be of type long");
    TORCH_CHECK(embeddings.is_cuda(), "Embeddings must be a CUDA tensor");
    TORCH_CHECK(embeddings.is_contiguous(), "Embeddings must be contiguous");

    auto indices_sizes = indices.sizes();
    auto embeddings_sizes = embeddings.sizes();

    TORCH_CHECK(indices.dim() == 2, "Indices must be a 2D tensor");
    int batch_size = indices_sizes[0];
    int seq_len = indices_sizes[1];

    TORCH_CHECK(embeddings.dim() == 2, "Embeddings must be a 2D tensor");
    int vocab_size = embeddings_sizes[0];
    int hidden_dim = embeddings_sizes[1];

    auto output = torch::empty({batch_size, seq_len, hidden_dim}, embeddings.options());

    embedding_lookup_forward(
        indices.data_ptr<long>(),
        embeddings.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size,
        seq_len,
        hidden_dim,
        vocab_size
    );

    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("embedding_lookup_forward", &embedding_lookup_forward_torch, "Embedding lookup forward pass (CUDA)");
}
