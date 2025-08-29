"""
Utility functions for FlashAttention-Mini

This module provides:
1. Memory usage calculation and optimization utilities
2. Performance benchmarking helpers
3. Tensor validation and conversion functions
4. Configuration optimization helpers
"""

import torch
import time
import math
import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import flashattention_mini_cuda
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False


def calculate_memory_usage(
    batch_size: int,
    n_heads: int, 
    seq_len: int,
    head_dim: int,
    return_breakdown: bool = True
) -> Union[int, Dict[str, int]]:
    """
    Calculate memory usage for FlashAttention computation
    
    Args:
        batch_size: Batch size
        n_heads: Number of attention heads
        seq_len: Sequence length
        head_dim: Dimension per head
        return_breakdown: If True, return detailed breakdown
        
    Returns:
        Memory usage in bytes (int) or breakdown dict
    """
    if CUDA_AVAILABLE:
        memory_breakdown = flashattention_mini_cuda.estimate_memory_usage(
            batch_size, n_heads, seq_len, head_dim
        )
        if return_breakdown:
            return memory_breakdown
        else:
            return memory_breakdown['total']
    else:
        # Fallback calculation
        tensor_size = batch_size * n_heads * seq_len * head_dim * 4  # 4 bytes per float32
        input_memory = tensor_size * 3  # Q, K, V
        output_memory = tensor_size  # Output
        buffer_memory = batch_size * n_heads * seq_len * 4 * 2  # l_buffer, m_buffer
        
        total_memory = input_memory + output_memory + buffer_memory
        
        if return_breakdown:
            return {
                'input_tensors': input_memory,
                'output_tensor': output_memory,
                'buffers': buffer_memory,
                'total': total_memory
            }
        else:
            return total_memory


def get_optimal_block_size(
    seq_len: int,
    head_dim: int,
    device: Optional[torch.device] = None,
    max_shared_memory: int = 48 * 1024  # 48KB default
) -> int:
    """
    Determine optimal block size for tiled computation
    
    Args:
        seq_len: Sequence length
        head_dim: Head dimension
        device: CUDA device (auto-detect if None)
        max_shared_memory: Maximum shared memory per block in bytes
        
    Returns:
        Optimal block size for tiling
    """
    if CUDA_AVAILABLE and device is not None and device.type == 'cuda':
        return flashattention_mini_cuda.get_optimal_block_size(
            seq_len, head_dim, max_shared_memory
        )
    else:
        # Fallback heuristic
        candidate_sizes = [32, 64, 128, 256]
        
        for block_size in candidate_sizes:
            # Estimate memory usage: Q_block + K_block + S_block + buffer
            required_memory = (
                block_size * head_dim * 2 +  # Q and K blocks
                block_size * block_size +     # Attention scores
                1024                          # Reduction buffer
            ) * 4  # 4 bytes per float
            
            if required_memory <= max_shared_memory:
                return block_size
                
        return 32  # Minimum fallback


def validate_attention_output(
    output: torch.Tensor,
    reference: torch.Tensor,
    rtol: float = 1e-4,
    atol: float = 1e-6
) -> Dict[str, float]:
    """
    Validate FlashAttention output against reference implementation
    
    Args:
        output: FlashAttention output
        reference: Reference attention output (e.g., from torch.nn.functional)
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        Dictionary with validation metrics
    """
    # Convert to same device and dtype
    output = output.float()
    reference = reference.float()
    
    # Calculate metrics
    max_abs_error = torch.max(torch.abs(output - reference)).item()
    mean_abs_error = torch.mean(torch.abs(output - reference)).item()
    rel_error = torch.max(torch.abs((output - reference) / (reference + 1e-8))).item()
    
    # Check if within tolerance
    is_close = torch.allclose(output, reference, rtol=rtol, atol=atol)
    
    metrics = {
        'max_absolute_error': max_abs_error,
        'mean_absolute_error': mean_abs_error,
        'max_relative_error': rel_error,
        'is_within_tolerance': is_close,
        'rtol_used': rtol,
        'atol_used': atol
    }
    
    return metrics


def benchmark_attention(
    batch_size: int,
    n_heads: int,
    seq_len: int,
    head_dim: int,
    num_warmup: int = 10,
    num_trials: int = 100,
    device: str = 'cuda',
    compare_with_torch: bool = True
) -> Dict[str, float]:
    """
    Benchmark FlashAttention-Mini against standard PyTorch attention
    
    Args:
        batch_size, n_heads, seq_len, head_dim: Problem dimensions
        num_warmup: Number of warmup iterations
        num_trials: Number of timing trials  
        device: Device to run on ('cuda' or 'cpu')
        compare_with_torch: Whether to include PyTorch comparison
        
    Returns:
        Dictionary with timing results and speedup metrics
    """
    if device == 'cpu':
        warnings.warn("FlashAttention-Mini optimized for CUDA. CPU benchmarking not meaningful.")
        return {}
        
    device = torch.device(device)
    
    # Create test tensors
    Q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    K = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    V = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device, dtype=torch.float32)
    
    scale = 1.0 / math.sqrt(head_dim)
    results = {}
    
    if CUDA_AVAILABLE:
        # Benchmark FlashAttention-Mini
        flash_times = []
        
        # Warmup
        for _ in range(num_warmup):
            _ = flashattention_mini_cuda.flash_attention_forward_simple(Q, K, V, scale)
        torch.cuda.synchronize()
        
        # Timing
        for _ in range(num_trials):
            start_time = time.perf_counter()
            output_flash = flashattention_mini_cuda.flash_attention_forward_simple(Q, K, V, scale)
            torch.cuda.synchronize()
            flash_times.append(time.perf_counter() - start_time)
            
        results['flash_attention_mean_ms'] = np.mean(flash_times) * 1000
        results['flash_attention_std_ms'] = np.std(flash_times) * 1000
        
    if compare_with_torch:
        # Benchmark PyTorch attention
        torch_times = []
        
        # Warmup
        for _ in range(num_warmup):
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(scores, dim=-1)
            _ = torch.matmul(attn_weights, V)
        torch.cuda.synchronize()
        
        # Timing
        for _ in range(num_trials):
            start_time = time.perf_counter()
            scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
            attn_weights = torch.softmax(scores, dim=-1)
            output_torch = torch.matmul(attn_weights, V)
            torch.cuda.synchronize()
            torch_times.append(time.perf_counter() - start_time)
            
        results['pytorch_attention_mean_ms'] = np.mean(torch_times) * 1000
        results['pytorch_attention_std_ms'] = np.std(torch_times) * 1000
        
        # Calculate speedup
        if CUDA_AVAILABLE and 'flash_attention_mean_ms' in results:
            results['speedup'] = results['pytorch_attention_mean_ms'] / results['flash_attention_mean_ms']
            
            # Validate correctness
            validation = validate_attention_output(output_flash, output_torch)
            results.update({f'validation_{k}': v for k, v in validation.items()})
    
    # Memory usage
    memory_usage = calculate_memory_usage(batch_size, n_heads, seq_len, head_dim)
    results['memory_usage_mb'] = memory_usage / (1024 * 1024)
    
    # Compute throughput metrics
    num_elements = batch_size * n_heads * seq_len * head_dim
    if CUDA_AVAILABLE and 'flash_attention_mean_ms' in results:
        results['flash_throughput_gflops'] = (2 * seq_len * seq_len * head_dim * batch_size * n_heads) / \
                                           (results['flash_attention_mean_ms'] / 1000) / 1e9
    
    if 'pytorch_attention_mean_ms' in results:
        results['pytorch_throughput_gflops'] = (2 * seq_len * seq_len * head_dim * batch_size * n_heads) / \
                                             (results['pytorch_attention_mean_ms'] / 1000) / 1e9
    
    return results


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    Create causal (lower triangular) attention mask
    
    Args:
        seq_len: Sequence length
        device: Device to create mask on
        
    Returns:
        Causal mask tensor [seq_len, seq_len]
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
    return mask.bool()


def create_padding_mask(
    lengths: torch.Tensor, 
    max_len: Optional[int] = None
) -> torch.Tensor:
    """
    Create padding mask from sequence lengths
    
    Args:
        lengths: Tensor of sequence lengths [batch_size]
        max_len: Maximum sequence length (default: max of lengths)
        
    Returns:
        Padding mask [batch_size, max_len]
    """
    if max_len is None:
        max_len = lengths.max().item()
        
    batch_size = lengths.size(0)
    device = lengths.device
    
    # Create range tensor
    range_tensor = torch.arange(max_len, device=device).expand(batch_size, max_len)
    
    # Create mask
    mask = range_tensor >= lengths.unsqueeze(1)
    
    return mask


def profile_memory_usage(
    func: callable,
    *args,
    **kwargs
) -> Tuple[any, Dict[str, int]]:
    """
    Profile GPU memory usage of a function call
    
    Args:
        func: Function to profile
        *args: Function arguments
        **kwargs: Function keyword arguments
        
    Returns:
        Tuple of (function_result, memory_stats)
    """
    if not torch.cuda.is_available():
        warnings.warn("CUDA not available. Cannot profile GPU memory.")
        result = func(*args, **kwargs)
        return result, {}
    
    # Clear cache and reset stats
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Record initial memory
    initial_memory = torch.cuda.memory_allocated()
    
    # Run function
    result = func(*args, **kwargs)
    torch.cuda.synchronize()
    
    # Record final memory
    final_memory = torch.cuda.memory_allocated()
    peak_memory = torch.cuda.max_memory_allocated()
    
    memory_stats = {
        'initial_memory_mb': initial_memory / (1024 * 1024),
        'final_memory_mb': final_memory / (1024 * 1024),
        'peak_memory_mb': peak_memory / (1024 * 1024),
        'memory_increase_mb': (final_memory - initial_memory) / (1024 * 1024)
    }
    
    return result, memory_stats


def get_device_info() -> Dict[str, Union[str, int, float]]:
    """
    Get information about available CUDA devices
    
    Returns:
        Dictionary with device information
    """
    if not torch.cuda.is_available():
        return {'cuda_available': False}
    
    device_info = {
        'cuda_available': True,
        'device_count': torch.cuda.device_count(),
        'current_device': torch.cuda.current_device(),
    }
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        device_info[f'device_{i}'] = {
            'name': props.name,
            'total_memory_gb': props.total_memory / (1024**3),
            'major': props.major,
            'minor': props.minor,
            'multi_processor_count': props.multi_processor_count
        }
    
    return device_info


def optimize_for_inference(model: torch.nn.Module) -> torch.nn.Module:
    """
    Optimize FlashAttention model for inference
    
    Args:
        model: Model containing FlashAttention layers
        
    Returns:
        Optimized model
    """
    model.eval()
    
    # Disable dropout
    for module in model.modules():
        if hasattr(module, 'dropout_layer') and module.dropout_layer is not None:
            module.dropout_layer.p = 0.0
            
    # Enable inference optimizations
    with torch.no_grad():
        # Fuse operations where possible
        # This is a placeholder for potential future optimizations
        pass
        
    return model


class AttentionProfiler:
    """
    Profiler for FlashAttention operations with detailed metrics
    """
    
    def __init__(self):
        self.profiles = []
        self.current_profile = None
        
    def start_profile(self, name: str):
        """Start profiling an attention operation"""
        if not torch.cuda.is_available():
            return
            
        self.current_profile = {
            'name': name,
            'start_time': time.perf_counter(),
            'start_memory': torch.cuda.memory_allocated(),
        }
        torch.cuda.synchronize()
        
    def end_profile(self):
        """End current profile and record results"""
        if not torch.cuda.is_available() or self.current_profile is None:
            return
            
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        end_memory = torch.cuda.memory_allocated()
        
        profile = {
            **self.current_profile,
            'end_time': end_time,
            'end_memory': end_memory,
            'duration_ms': (end_time - self.current_profile['start_time']) * 1000,
            'memory_used_mb': (end_memory - self.current_profile['start_memory']) / (1024 * 1024)
        }
        
        self.profiles.append(profile)
        self.current_profile = None
        
    def get_summary(self) -> Dict[str, float]:
        """Get summary of all profiles"""
        if not self.profiles:
            return {}
            
        durations = [p['duration_ms'] for p in self.profiles]
        memory_usage = [p['memory_used_mb'] for p in self.profiles]
        
        return {
            'total_operations': len(self.profiles),
            'mean_duration_ms': np.mean(durations),
            'std_duration_ms': np.std(durations),
            'total_duration_ms': np.sum(durations),
            'mean_memory_mb': np.mean(memory_usage),
            'peak_memory_mb': np.max(memory_usage)
        }
    
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.current_profile is not None:
            self.end_profile()