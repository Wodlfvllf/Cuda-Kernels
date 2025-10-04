import torch
import groupnorm_cuda

def test_groupnorm_correctness():
    # Test dimensions
    N = 4
    C = 32
    H = 16
    W = 16
    G = 4
    epsilon = 1e-5
    
    # Create input
    x = torch.randn(N, C, H, W, device='cuda')
    gamma = torch.randn(C, device='cuda')
    beta = torch.randn(C, device='cuda')
    
    # Compute with custom kernel
    custom_output = torch.zeros_like(x)
    groupnorm_cuda.groupnorm_forward(x, gamma, beta, custom_output, G, epsilon)
    
    # Compute with reference
    ref_output = torch.nn.functional.group_norm(x, G, weight=gamma, bias=beta, eps=epsilon)
    
    # Check correctness
    assert torch.allclose(custom_output, ref_output, rtol=1e-4, atol=1e-5)
    print("\u2713 GroupNorm correctness test passed")

if __name__ == "__main__":
    test_groupnorm_correctness()
