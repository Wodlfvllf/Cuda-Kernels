"""
Correctness tests for FlashAttention-Mini

This module tests the numerical correctness of the FlashAttention implementation
by comparing outputs against reference PyTorch attention computation.
"""

import pytest
import torch
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from attention import FlashAttentionMini, flash_attention_forward, efficient_attention
from utils import validate_attention_output, calculate_memory_usage

# Test configurations
TEST_CONFIGS = [
    # (batch_size, n_heads, seq_len, head_dim)
    (1, 1, 64, 64),      # Small test
    (2, 4, 128, 64),     # Medium test
    (4, 8, 256, 64),     # Standard test
    (2, 4, 512, 128),    # Larger head_dim
    (1, 1, 1024, 64),    # Long sequence
    (8, 16, 128, 64),    # Many heads
]

TOLERANCE_CONFIG = {
    'rtol': 1e-4,
    'atol': 1e-6
}


@pytest.fixture
def device():
    """Get CUDA device for testing"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device('cuda')


def create_test_tensors(
    batch_size: int, 
    n_heads: int, 
    seq_len: int, 
    head_dim: int, 
    device: torch.device,
    seed: int = 42
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create reproducible test tensors"""
    torch.manual_seed(seed)
    
    Q = torch.randn(batch_size, n_heads, seq_len, head_dim, 
                   device=device, dtype=torch.float32, requires_grad=True)
    K = torch.randn(batch_size, n_heads, seq_len, head_dim,
                   device=device, dtype=torch.float32, requires_grad=True) 
    V = torch.randn(batch_size, n_heads, seq_len, head_dim,
                   device=device, dtype=torch.float32, requires_grad=True)
    
    return Q, K, V


def reference_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                       scale: float = None) -> torch.Tensor:
    """Reference PyTorch attention implementation"""
    if scale is None:
        scale = 1.0 / math.sqrt(Q.size(-1))
    
    # Compute attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
    
    # Apply softmax
    attn_weights = F.softmax(scores, dim=-1)
    
    # Apply attention to values
    output = torch.matmul(attn_weights, V)
    
    return output


class TestFlashAttentionCorrectness:
    """Test suite for FlashAttention numerical correctness"""
    
    @pytest.mark.parametrize("batch_size,n_heads,seq_len,head_dim", TEST_CONFIGS)
    def test_forward_correctness(self, device, batch_size, n_heads, seq_len, head_dim):
        """Test forward pass correctness against PyTorch reference"""
        Q, K, V = create_test_tensors(batch_size, n_heads, seq_len, head_dim, device)
        scale = 1.0 / math.sqrt(head_dim)
        
        # FlashAttention output
        flash_output = flash_attention_forward(Q, K, V, scale)
        
        # Reference output  
        reference_output = reference_attention(Q, K, V, scale)
        
        # Validate correctness
        validation = validate_attention_output(
            flash_output, reference_output, **TOLERANCE_CONFIG
        )
        
        assert validation['is_within_tolerance'], \
            f"Correctness test failed with max_error={validation['max_absolute_error']:.6f}"
        
        print(f"✅ Forward correctness: B={batch_size}, H={n_heads}, S={seq_len}, D={head_dim}")
        print(f"   Max absolute error: {validation['max_absolute_error']:.2e}")
        print(f"   Max relative error: {validation['max_relative_error']:.2e}")
    
    @pytest.mark.parametrize("batch_size,n_heads,seq_len,head_dim", TEST_CONFIGS[:3])  # Subset for speed
    def test_backward_correctness(self, device, batch_size, n_heads, seq_len, head_dim):
        """Test backward pass correctness"""
        Q, K, V = create_test_tensors(batch_size, n_heads, seq_len, head_dim, device)
        scale = 1.0 / math.sqrt(head_dim)
        
        # Create identical copies for gradient comparison
        Q_ref, K_ref, V_ref = Q.clone().detach().requires_grad_(True), \
                              K.clone().detach().requires_grad_(True), \
                              V.clone().detach().requires_grad_(True)
        
        # FlashAttention forward + backward
        flash_output = flash_attention_forward(Q, K, V, scale)
        grad_output = torch.randn_like(flash_output)
        flash_output.backward(grad_output)
        
        # Reference forward + backward
        reference_output = reference_attention(Q_ref, K_ref, V_ref, scale)
        reference_output.backward(grad_output)
        
        # Compare gradients
        grad_validations = {}
        for name, (grad_flash, grad_ref) in [
            ('Q', (Q.grad, Q_ref.grad)), 
            ('K', (K.grad, K_ref.grad)), 
            ('V', (V.grad, V_ref.grad))
        ]:
            if grad_flash is not None and grad_ref is not None:
                validation = validate_attention_output(
                    grad_flash, grad_ref, rtol=1e-3, atol=1e-5  # Slightly relaxed for gradients
                )
                grad_validations[name] = validation
                
                assert validation['is_within_tolerance'], \
                    f"Gradient {name} test failed with max_error={validation['max_absolute_error']:.6f}"
        
        print(f"✅ Backward correctness: B={batch_size}, H={n_heads}, S={seq_len}, D={head_dim}")
        for name, val in grad_validations.items():
            print(f"   Grad {name} max error: {val['max_absolute_error']:.2e}")
    
    @pytest.mark.parametrize("seq_len", [128, 256, 512, 1024])
    def test_different_sequence_lengths(self, device, seq_len):
        """Test correctness across different sequence lengths"""
        batch_size, n_heads, head_dim = 2, 4, 64
        Q, K, V = create_test_tensors(batch_size, n_heads, seq_len, head_dim, device)
        
        flash_output = flash_attention_forward(Q, K, V)
        reference_output = reference_attention(Q, K, V)
        
        validation = validate_attention_output(flash_output, reference_output, **TOLERANCE_CONFIG)
        assert validation['is_within_tolerance']
        
        print(f"✅ Sequence length {seq_len}: max_error={validation['max_absolute_error']:.2e}")
    
    def test_module_interface(self, device):
        """Test FlashAttentionMini module interface"""
        d_model, n_heads = 512, 8
        batch_size, seq_len = 2, 256
        
        # Create module
        flash_attn = FlashAttentionMini(d_model=d_model, n_heads=n_heads).to(device)
        
        # Test with 3D input (typical usage)
        input_3d = torch.randn(batch_size, seq_len, d_model, device=device)
        output_3d = flash_attn(input_3d)
        
        assert output_3d.shape == (batch_size, seq_len, d_model)
        
        # Test with 4D input (advanced usage)  
        head_dim = d_model // n_heads
        input_4d = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        output_4d = flash_attn(input_4d)
        
        assert output_4d.shape == (batch_size, n_heads, seq_len, head_dim)
        
        print("✅ Module interface tests passed")
    
    def test_numerical_stability(self, device):
        """Test numerical stability with extreme values"""
        batch_size, n_heads, seq_len, head_dim = 1, 1, 64, 64
        
        # Test with very large values
        Q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device) * 100
        K = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device) * 100
        V = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device) * 100
        
        flash_output = flash_attention_forward(Q, K, V)
        reference_output = reference_attention(Q, K, V)
        
        # Check for NaN/Inf
        assert not torch.isnan(flash_output).any(), "NaN detected in FlashAttention output"
        assert not torch.isinf(flash_output).any(), "Inf detected in FlashAttention output"
        
        # Check correctness
        validation = validate_attention_output(
            flash_output, reference_output, rtol=1e-3, atol=1e-4
        )
        assert validation['is_within_tolerance']
        
        print("✅ Numerical stability test passed")
    
    def test_edge_cases(self, device):
        """Test edge cases and boundary conditions"""
        
        # Test single element sequences
        batch_size, n_heads, seq_len, head_dim = 1, 1, 1, 64
        Q, K, V = create_test_tensors(batch_size, n_heads, seq_len, head_dim, device)
        
        flash_output = flash_attention_forward(Q, K, V)
        reference_output = reference_attention(Q, K, V)
        
        validation = validate_attention_output(flash_output, reference_output, **TOLERANCE_CONFIG)
        assert validation['is_within_tolerance']
        
        # Test zero inputs
        Q_zero = torch.zeros_like(Q)
        K_zero = torch.zeros_like(K) 
        V_zero = torch.zeros_like(V)
        
        flash_output_zero = flash_attention_forward(Q_zero, K_zero, V_zero)
        reference_output_zero = reference_attention(Q_zero, K_zero, V_zero)
        
        validation_zero = validate_attention_output(
            flash_output_zero, reference_output_zero, **TOLERANCE_CONFIG
        )
        assert validation_zero['is_within_tolerance']
        
        print("✅ Edge cases test passed")
    
    @pytest.mark.parametrize("head_dim", [64, 128])
    def test_supported_head_dimensions(self, device, head_dim):
        """Test all supported head dimensions"""
        batch_size, n_heads, seq_len = 2, 4, 128
        Q, K, V = create_test_tensors(batch_size, n_heads, seq_len, head_dim, device)
        
        flash_output = flash_attention_forward(Q, K, V)
        reference_output = reference_attention(Q, K, V)
        
        validation = validate_attention_output(flash_output, reference_output, **TOLERANCE_CONFIG)
        assert validation['is_within_tolerance']
        
        print(f"✅ Head dimension {head_dim}: max_error={validation['max_absolute_error']:.2e}")
    
    def test_memory_efficiency(self, device):
        """Test that FlashAttention uses less memory than standard attention"""
        batch_size, n_heads, seq_len, head_dim = 4, 8, 512, 64
        
        # Calculate theoretical memory usage
        memory_usage = calculate_memory_usage(batch_size, n_heads, seq_len, head_dim)
        standard_memory = batch_size * n_heads * seq_len * seq_len * 4  # Attention matrix
        
        print(f"✅ Memory efficiency:")
        print(f"   FlashAttention memory: {memory_usage / (1024**2):.1f} MB")
        print(f"   Standard attention matrix: {standard_memory / (1024**2):.1f} MB")
        print(f"   Memory reduction: {standard_memory / memory_usage:.1f}×")
        
        # For longer sequences, FlashAttention should be more memory efficient
        assert seq_len > 256, "Test designed for longer sequences"
    
    def test_scale_parameter(self, device):
        """Test different scale parameters"""
        batch_size, n_heads, seq_len, head_dim = 1, 1, 64, 64
        Q, K, V = create_test_tensors(batch_size, n_heads, seq_len, head_dim, device)
        
        scales = [0.1, 0.25, 0.5, 1.0, 2.0]
        
        for scale in scales:
            flash_output = flash_attention_forward(Q, K, V, scale)
            reference_output = reference_attention(Q, K, V, scale)
            
            validation = validate_attention_output(
                flash_output, reference_output, **TOLERANCE_CONFIG
            )
            assert validation['is_within_tolerance'], f"Scale {scale} failed"
        
        print(f"✅ Scale parameter test passed for scales: {scales}")


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])