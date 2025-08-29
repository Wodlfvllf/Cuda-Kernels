"""
FlashAttention-Mini: A simplified implementation of FlashAttention tiled attention mechanism.

This package provides:
- FlashAttentionMini: Main attention module with tiled computation
- Utility functions for memory management and benchmarking
- CUDA kernels for efficient attention computation
"""

from .attention import FlashAttentionMini, flash_attention_forward
from .utils import (
    calculate_memory_usage,
    benchmark_attention,
    get_optimal_block_size,
    validate_attention_output
)

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

__all__ = [
    "FlashAttentionMini",
    "flash_attention_forward", 
    "calculate_memory_usage",
    "benchmark_attention",
    "get_optimal_block_size",
    "validate_attention_output",
]

# Version check for dependencies
def _check_dependencies():
    """Check that required dependencies are available"""
    try:
        import torch
        if not torch.cuda.is_available():
            print("Warning: CUDA not available. FlashAttention-Mini requires CUDA.")
        
        # Check PyTorch version
        torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
        if torch_version < (1, 9):
            print(f"Warning: PyTorch {torch.__version__} detected. Recommended: >= 1.9.0")
            
    except ImportError:
        print("Error: PyTorch not found. Please install PyTorch with CUDA support.")
    
    try:
        import flashattention_mini_cuda
    except ImportError:
        print("Warning: CUDA extension not found. Please build the extension with 'python setup.py develop'")

# Run dependency check on import
_check_dependencies()