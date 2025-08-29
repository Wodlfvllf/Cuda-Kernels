
import torch
import torch.nn as nn
import time
import numpy as np
import layernorm_cuda

def test_correctness():
    """Test correctness against PyTorch implementation"""
    print("Testing correctness...")
    
    batch_size = 32
    hidden_dim = 768
    eps = 1e-5
    
    # Create test tensors
    torch.manual_seed(42)
    input_tensor = torch.randn(batch_size, hidden_dim, device='cuda')
    gamma = torch.ones(hidden_dim, device='cuda')
    beta = torch.zeros(hidden_dim, device='cuda')
    
    # PyTorch reference
    layer_norm = nn.LayerNorm(hidden_dim, eps=eps).cuda()
    layer_norm.weight.data = gamma
    layer_norm.bias.data = beta
    
    with torch.no_grad():
        pytorch_output = layer_norm(input_tensor)
    
    # Custom CUDA implementation
    cuda_output = layernorm_cuda.forward(input_tensor, gamma, beta, eps, use_welford=False)
    cuda_output_welford = layernorm_cuda.forward(input_tensor, gamma, beta, eps, use_welford=True)
    
    # Check correctness
    diff = torch.abs(pytorch_output - cuda_output)
    diff_welford = torch.abs(pytorch_output - cuda_output_welford)
    
    print(f"Max error (fused): {diff.max().item():.2e}")
    print(f"Mean error (fused): {diff.mean().item():.2e}")
    print(f"Max error (Welford): {diff_welford.max().item():.2e}")
    print(f"Mean error (Welford): {diff_welford.mean().item():.2e}")
    
    assert diff.max().item() < 1e-5, "Fused implementation failed correctness test"
    assert diff_welford.max().item() < 1e-5, "Welford implementation failed correctness test"
    print("Correctness test PASSED!\n")

def benchmark_performance():
    """Benchmark performance against PyTorch"""
    print("Benchmarking performance...")
    
    configs = [
        (32, 768),   # BERT-base
        (64, 1024),  # BERT-large
        (128, 2048), # GPT-2 medium
        (256, 4096), # Large model
    ]
    
    for batch_size, hidden_dim in configs:
        print(f"\nBatch: {batch_size}, Hidden: {hidden_dim}")
        print("-" * 40)
        
        # Setup
        input_tensor = torch.randn(batch_size, hidden_dim, device='cuda')
        gamma = torch.ones(hidden_dim, device='cuda')
        beta = torch.zeros(hidden_dim, device='cuda')
        
        layer_norm = nn.LayerNorm(hidden_dim).cuda()
        
        # Warmup
        for _ in range(10):
            _ = layer_norm(input_tensor)
            _ = layernorm_cuda.forward(input_tensor, gamma, beta, 1e-5, False)
        
        torch.cuda.synchronize()
        
        # Benchmark PyTorch
        num_iters = 100
        start = time.time()
        for _ in range(num_iters):
            _ = layer_norm(input_tensor)
        torch.cuda.synchronize()
        pytorch_time = (time.time() - start) * 1000 / num_iters
        
        # Benchmark custom fused kernel
        start = time.time()
        for _ in range(num_iters):
            _ = layernorm_cuda.forward(input_tensor, gamma, beta, 1e-5, False)
        torch.cuda.synchronize()
        fused_time = (time.time() - start) * 1000 / num_iters
        
        # Benchmark Welford kernel
        start = time.time()
        for _ in range(num_iters):
            _ = layernorm_cuda.forward(input_tensor, gamma, beta, 1e-5, True)
        torch.cuda.synchronize()
        welford_time = (time.time() - start) * 1000 / num_iters
        
        print(f"PyTorch:  {pytorch_time:.3f} ms")
        print(f"Fused:    {fused_time:.3f} ms (Speedup: {pytorch_time/fused_time:.2f}x)")
        print(f"Welford:  {welford_time:.3f} ms (Speedup: {pytorch_time/welford_time:.2f}x)")

def test_gradient():
    """Test gradient computation"""
    print("\nTesting gradients...")
    
    batch_size = 8
    hidden_dim = 256
    
    input_tensor = torch.randn(batch_size, hidden_dim, device='cuda', requires_grad=True)
    gamma = torch.ones(hidden_dim, device='cuda', requires_grad=True)
    beta = torch.zeros(hidden_dim, device='cuda', requires_grad=True)
    
    # Forward pass
    output = layernorm_cuda.forward(input_tensor, gamma, beta, 1e-5, False)
    
    # Backward pass
    grad_output = torch.randn_like(output)
    output.backward(grad_output)
    
    print("Gradient test completed (basic check)")

if __name__ == "__main__":
    test_correctness()
    benchmark_performance()
    test_gradient()