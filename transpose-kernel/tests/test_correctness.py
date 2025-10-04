import torch
import transpose_cuda

def test_transpose_correctness():
    # Test dimensions
    M = 256
    N = 512
    
    # Create input
    x = torch.randn(M, N, device='cuda')
    
    # Compute with custom kernel
    custom_output = transpose_cuda.naive_transpose_forward(x)
    
    # Compute with reference
    ref_output = torch.transpose(x, 0, 1)
    
    # Check correctness
    assert torch.allclose(custom_output, ref_output, rtol=1e-4, atol=1e-5)
    print("\u2713 Naive transpose correctness test passed")

def test_optimized_transpose_correctness():
    # Test dimensions
    M = 256
    N = 512
    
    # Create input
    x = torch.randn(M, N, device='cuda')
    
    # Compute with custom kernel
    custom_output = transpose_cuda.optimized_transpose_forward(x)
    
    # Compute with reference
    ref_output = torch.transpose(x, 0, 1)
    
    # Check correctness
    assert torch.allclose(custom_output, ref_output, rtol=1e-4, atol=1e-5)
    print("\u2713 Optimized transpose correctness test passed")

if __name__ == "__main__":
    test_transpose_correctness()
    test_optimized_transpose_correctness()