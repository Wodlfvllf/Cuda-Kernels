import torch
import embedding_lookup_cuda

def test_embedding_lookup_correctness():
    # Test dimensions
    batch_size = 4
    seq_len = 128
    hidden_dim = 768
    vocab_size = 10000
    
    # Create input
    indices = torch.randint(0, vocab_size, (batch_size, seq_len), device='cuda')
    embeddings = torch.randn(vocab_size, hidden_dim, device='cuda')
    
    # Compute with custom kernel
    custom_output = embedding_lookup_cuda.embedding_lookup_forward(indices, embeddings)
    
    # Compute with reference
    ref_output = torch.nn.functional.embedding(indices, embeddings)
    
    # Check correctness
    assert torch.allclose(custom_output, ref_output, rtol=1e-4, atol=1e-5)
    print("\u2713 Embedding Lookup correctness test passed")

if __name__ == "__main__":
    test_embedding_lookup_correctness()
