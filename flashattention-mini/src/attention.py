"""
PyTorch wrapper for FlashAttention-Mini with autograd support

This module provides:
1. FlashAttentionMini: Main attention module with tiled computation  
2. FlashAttentionFunction: Autograd function for backward pass
3. Utility functions for configuration and optimization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
import math
import warnings
from typing import Optional, Tuple

try:
    import flashattention_mini_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    warnings.warn("FlashAttention-Mini CUDA extension not found. "
                  "Please build the extension with 'python setup.py develop'")


class FlashAttentionFunction(Function):
    """
    Autograd function for FlashAttention with efficient forward/backward passes
    
    This implements the autograd interface needed for training, including
    proper gradient computation that reuses intermediate values from forward pass.
    """
    
    @staticmethod
    def forward(ctx, Q, K, V, scale=None):
        """
        Forward pass using FlashAttention tiled computation
        
        Args:
            Q: Query tensor [batch, n_heads, seq_len, head_dim]
            K: Key tensor [batch, n_heads, seq_len, head_dim]  
            V: Value tensor [batch, n_heads, seq_len, head_dim]
            scale: Attention scale factor (default: 1/sqrt(head_dim))
            
        Returns:
            output: Attention output [batch, n_heads, seq_len, head_dim]
        """
        if not CUDA_AVAILABLE:
            raise RuntimeError("CUDA extension not available. Cannot run FlashAttention-Mini.")
            
        # Default scale
        if scale is None:
            scale = 1.0 / math.sqrt(Q.size(-1))
            
        # Ensure tensors are contiguous and on GPU
        Q = Q.contiguous()
        K = K.contiguous() 
        V = V.contiguous()
        
        # Call CUDA kernel
        output, l_buffer, m_buffer = flashattention_mini_cuda.flash_attention_forward(
            Q, K, V, scale
        )
        
        # Save tensors for backward pass
        ctx.save_for_backward(Q, K, V, output, l_buffer, m_buffer)
        ctx.scale = scale
        
        return output
    
    @staticmethod  
    def backward(ctx, grad_output):
        """
        Backward pass for FlashAttention
        
        Note: This is a simplified implementation. Full FlashAttention backward
        would recompute attention in blocks to maintain memory efficiency.
        """
        Q, K, V, output, l_buffer, m_buffer = ctx.saved_tensors
        scale = ctx.scale
        
        grad_output = grad_output.contiguous()
        
        # For this mini implementation, we'll use standard attention backward
        # A full implementation would use block-wise recomputation
        with torch.enable_grad():
            Q_temp = Q.detach().requires_grad_(True)
            K_temp = K.detach().requires_grad_(True) 
            V_temp = V.detach().requires_grad_(True)
            
            # Recompute forward for gradients (simplified)
            scores = torch.matmul(Q_temp, K_temp.transpose(-2, -1)) * scale
            attn_weights = F.softmax(scores, dim=-1)
            output_temp = torch.matmul(attn_weights, V_temp)
            
            # Compute gradients
            torch.autograd.backward(output_temp, grad_output)
            
        grad_Q = Q_temp.grad
        grad_K = K_temp.grad
        grad_V = V_temp.grad
        
        return grad_Q, grad_K, grad_V, None


# Convenient function interface
def flash_attention_forward(Q, K, V, scale=None):
    """
    Compute FlashAttention forward pass
    
    Args:
        Q: Query tensor [batch, n_heads, seq_len, head_dim]
        K: Key tensor [batch, n_heads, seq_len, head_dim]
        V: Value tensor [batch, n_heads, seq_len, head_dim]  
        scale: Attention scale factor (default: 1/sqrt(head_dim))
        
    Returns:
        output: Attention output [batch, n_heads, seq_len, head_dim]
    """
    return FlashAttentionFunction.apply(Q, K, V, scale)


class FlashAttentionMini(nn.Module):
    """
    FlashAttention-Mini: Memory-efficient tiled attention computation
    
    This implements a simplified version of FlashAttention that:
    1. Uses tiled computation to reduce memory usage from O(n²) to O(n)
    2. Computes attention in blocks using shared memory
    3. Supports efficient training with autograd integration
    
    Args:
        d_model: Model dimension (total)
        n_heads: Number of attention heads
        head_dim: Dimension per head (default: d_model // n_heads)
        block_size: Block size for tiled computation (auto-selected if None)
        dropout: Dropout probability (default: 0.0)
        bias: Whether to use bias in linear projections (default: True)
    """
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: Optional[int] = None,
        block_size: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = True
    ):
        super().__init__()
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim or d_model // n_heads
        self.dropout = dropout
        
        # Validate dimensions
        assert self.head_dim * n_heads == d_model, \
            f"d_model ({d_model}) must equal n_heads ({n_heads}) * head_dim ({self.head_dim})"
        assert self.head_dim in [64, 128], \
            f"Supported head_dim: 64, 128. Got: {self.head_dim}"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)  
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        
        # Dropout
        if dropout > 0.0:
            self.dropout_layer = nn.Dropout(dropout)
        else:
            self.dropout_layer = None
            
        # Attention scale
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Block size for tiling (will be auto-selected if None)
        self.block_size = block_size
        
        # Initialize weights
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters using Xavier uniform initialization"""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
    
    def forward(
        self, 
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        value: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of FlashAttention-Mini
        
        Args:
            query: Query tensor [batch, seq_len, d_model] or [batch, n_heads, seq_len, head_dim]
            key: Key tensor (defaults to query for self-attention)
            value: Value tensor (defaults to key)
            attn_mask: Attention mask (not implemented in mini version)
            key_padding_mask: Key padding mask (not implemented in mini version)
            
        Returns:
            output: Attention output with same shape as query
        """
        if key is None:
            key = query
        if value is None:
            value = key
            
        # Handle attention masks
        if attn_mask is not None or key_padding_mask is not None:
            warnings.warn("Attention masks not implemented in FlashAttention-Mini. Ignoring.")
            
        batch_size, seq_len = query.shape[:2]
        
        # Input format detection and projection
        if query.dim() == 3:  # [batch, seq_len, d_model]
            # Apply linear projections
            Q = self.q_proj(query)
            K = self.k_proj(key) 
            V = self.v_proj(value)
            
            # Reshape to [batch, n_heads, seq_len, head_dim]
            Q = Q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            K = K.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            V = V.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
            
            reshape_output = True
            
        elif query.dim() == 4:  # [batch, n_heads, seq_len, head_dim]
            # Already in correct format - use directly
            Q, K, V = query, key, value
            reshape_output = False
            
        else:
            raise ValueError(f"Expected query to be 3D or 4D, got {query.dim()}D")
        
        # Ensure tensors are on CUDA
        if not Q.is_cuda:
            raise ValueError("FlashAttention-Mini requires CUDA tensors")
            
        # Apply FlashAttention
        output = flash_attention_forward(Q, K, V, self.scale)
        
        # Apply dropout if specified
        if self.dropout_layer is not None and self.training:
            output = self.dropout_layer(output)
        
        # Reshape output if needed
        if reshape_output:
            # Convert back to [batch, seq_len, d_model]
            output = output.transpose(1, 2).contiguous().view(
                batch_size, seq_len, self.d_model
            )
            # Apply output projection
            output = self.out_proj(output)
            
        return output
    
    def extra_repr(self) -> str:
        """String representation for debugging"""
        return (f'd_model={self.d_model}, n_heads={self.n_heads}, '
                f'head_dim={self.head_dim}, dropout={self.dropout}')


class MultiHeadFlashAttention(FlashAttentionMini):
    """
    Alias for FlashAttentionMini with MultiHeadAttention-compatible interface
    
    This provides compatibility with torch.nn.MultiheadAttention usage patterns.
    """
    
    def __init__(self, embed_dim, num_heads, dropout=0.0, bias=True, **kwargs):
        super().__init__(
            d_model=embed_dim,
            n_heads=num_heads,
            dropout=dropout, 
            bias=bias,
            **kwargs
        )


# Utility function for easy attention computation
def efficient_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: Optional[float] = None,
    training: bool = False
) -> torch.Tensor:
    """
    Compute efficient attention using FlashAttention-Mini
    
    This is a functional interface similar to F.scaled_dot_product_attention
    
    Args:
        query: Query tensor [batch, n_heads, seq_len, head_dim] 
        key: Key tensor [batch, n_heads, seq_len, head_dim]
        value: Value tensor [batch, n_heads, seq_len, head_dim]
        scale: Attention scale factor (default: 1/sqrt(head_dim))
        training: Whether in training mode (affects dropout)
        
    Returns:
        output: Attention output [batch, n_heads, seq_len, head_dim]
    """
    return flash_attention_forward(query, key, value, scale)


# Configuration helpers
def get_recommended_config(seq_len: int, d_model: int, device: str = 'cuda') -> dict:
    """
    Get recommended FlashAttention-Mini configuration for given problem size
    
    Args:
        seq_len: Sequence length
        d_model: Model dimension
        device: Target device ('cuda' or 'cpu')
        
    Returns:
        Dictionary with recommended configuration parameters
    """
    if device == 'cpu':
        warnings.warn("FlashAttention-Mini is optimized for CUDA. CPU not supported.")
        return {}
        
    # Determine optimal head dimension
    if d_model % 128 == 0:
        head_dim = 128
        n_heads = d_model // 128
    elif d_model % 64 == 0:
        head_dim = 64
        n_heads = d_model // 64
    else:
        # Find largest supported head_dim that divides d_model
        head_dim = 64 if d_model % 64 == 0 else 128
        n_heads = d_model // head_dim
        
    # Get optimal block size
    if CUDA_AVAILABLE:
        block_size = flashattention_mini_cuda.get_optimal_block_size(seq_len, head_dim)
    else:
        block_size = 64  # Default fallback
        
    return {
        'n_heads': n_heads,
        'head_dim': head_dim, 
        'block_size': block_size,
        'dropout': 0.1,  # Default dropout
    }