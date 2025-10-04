import torch
import gelu_cuda

def test_gelu_correctness():
    # Test dimensions
    batch_size = 4
    seq_len = 128
    hidden_dim = 768
    
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda')
    
    # Compute with custom kernel
    custom_output = gelu_cuda.gelu_forward(x)
    
    # Compute with reference
    ref_output = torch.nn.functional.gelu(x, approximate='tanh')
    
    # Check correctness
    assert torch.allclose(custom_output, ref_output, rtol=1e-4, atol=1e-5)
    print("\u2713 GELU correctness test passed")

if __name__ == "__main__":
    test_gelu_correctness()
