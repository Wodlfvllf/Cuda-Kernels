import torch
import rmsnorm_cuda

def rmsnorm_ref(input, gamma, epsilon):
    rrms = (input.pow(2).mean(dim=-1, keepdim=True) + epsilon).rsqrt()
    return input * rrms * gamma

def test_rmsnorm_correctness():
    # Test dimensions
    batch_size = 4
    seq_len = 128
    hidden_dim = 768
    epsilon = 1e-6
    
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda')
    gamma = torch.randn(hidden_dim, device='cuda')
    
    # Compute with custom kernel
    custom_output = rmsnorm_cuda.naive_rmsnorm_forward(x, gamma, epsilon)
    
    # Compute with reference
    ref_output = rmsnorm_ref(x, gamma, epsilon)
    
    # Check correctness
    assert torch.allclose(custom_output, ref_output, rtol=1e-4, atol=1e-5)
    print("\u2713 Naive RMSNorm correctness test passed")

def test_optimized_rmsnorm_correctness():
    # Test dimensions
    batch_size = 4
    seq_len = 128
    hidden_dim = 768
    epsilon = 1e-6
    
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda')
    gamma = torch.randn(hidden_dim, device='cuda')
    
    # Compute with custom kernel
    custom_output = rmsnorm_cuda.optimized_rmsnorm_forward(x, gamma, epsilon)
    
    # Compute with reference
    ref_output = rmsnorm_ref(x, gamma, epsilon)
    
    # Check correctness
    assert torch.allclose(custom_output, ref_output, rtol=1e-4, atol=1e-5)
    print("\u2713 Optimized RMSNorm correctness test passed")

if __name__ == "__main__":
    test_rmsnorm_correctness()
    test_optimized_rmsnorm_correctness()