"""
Performance tests for FlashAttention-Mini

This module tests the performance characteristics of FlashAttention,
including throughput, memory usage, and scalability comparisons.
"""

import pytest
import torch
import time
import numpy as np
import gc
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from attention import FlashAttentionMini, flash_attention_forward
from utils import benchmark_attention, calculate_memory_usage, profile_memory_usage, get_device_info

# Performance test configurations
PERF_CONFIGS = [
    # (batch_size, n_heads, seq_len, head_dim)
    (1, 8, 512, 64),
    (2, 8, 512, 64),
    (4, 8, 512, 64),
    (8, 8, 512, 64),      # Batch scaling
    (2, 8, 256, 64),
    (2, 8, 512, 64),
    (2, 8, 1024, 64),
    (2, 8, 2048, 64),     # Sequence scaling  
    (2, 4, 512, 64),
    (2, 8, 512, 64),
    (2, 16, 512, 64),     # Head scaling
    (2, 8, 512, 64),
    (2, 8, 512, 128),     # Head dimension scaling
]

BENCHMARK_PARAMS = {
    'num_warmup': 10,
    'num_trials': 50
}


@pytest.fixture
def device():
    """Get CUDA device for testing"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return torch.device('cuda')


@pytest.fixture(scope="session")
def device_info():
    """Get device information for context"""
    return get_device_info()


class TestFlashAttentionPerformance:
    """Test suite for FlashAttention performance characteristics"""
    
    @pytest.mark.parametrize("batch_size,n_heads,seq_len,head_dim", PERF_CONFIGS)
    def test_throughput_benchmark(self, device, batch_size, n_heads, seq_len, head_dim):
        """Benchmark throughput against PyTorch attention"""
        results = benchmark_attention(
            batch_size, n_heads, seq_len, head_dim,
            device=str(device), **BENCHMARK_PARAMS
        )
        
        # Extract metrics
        flash_time = results.get('flash_attention_mean_ms', 0)
        torch_time = results.get('pytorch_attention_mean_ms', 0)
        speedup = results.get('speedup', 0)
        
        print(f"\n📊 Performance B={batch_size}, H={n_heads}, S={seq_len}, D={head_dim}:")
        print(f"   FlashAttention: {flash_time:.3f}ms ± {results.get('flash_attention_std_ms', 0):.3f}ms")
        print(f"   PyTorch:        {torch_time:.3f}ms ± {results.get('pytorch_attention_std_ms', 0):.3f}ms") 
        print(f"   Speedup:        {speedup:.2f}×")
        
        # Performance assertions
        assert flash_time > 0, "FlashAttention timing failed"
        if torch_time > 0:
            # For larger problems, FlashAttention should be competitive
            if seq_len >= 512:
                assert speedup >= 0.8, f"Performance regression: {speedup:.2f}× speedup"
    
    def test_memory_scaling(self, device):
        """Test memory scaling with sequence length"""
        head_dim = 64
        n_heads = 8
        batch_size = 2
        sequence_lengths = [128, 256, 512, 1024, 2048]
        
        memory_usage = []
        flash_times = []
        torch_times = []
        
        for seq_len in sequence_lengths:
            # Memory usage calculation
            mem_usage = calculate_memory_usage(batch_size, n_heads, seq_len, head_dim)
            memory_usage.append(mem_usage / (1024**2))  # Convert to MB
            
            # Performance benchmarking
            results = benchmark_attention(
                batch_size, n_heads, seq_len, head_dim,
                device=str(device), num_trials=20
            )
            flash_times.append(results.get('flash_attention_mean_ms', 0))
            torch_times.append(results.get('pytorch_attention_mean_ms', 0))
        
        print(f"\n📈 Memory and Performance Scaling:")
        print(f"{'Seq Len':<8} {'Memory (MB)':<12} {'Flash (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<8}")
        print("-" * 60)
        for i, seq_len in enumerate(sequence_lengths):
            speedup = torch_times[i] / flash_times[i] if flash_times[i] > 0 else 0
            print(f"{seq_len:<8} {memory_usage[i]:<12.1f} {flash_times[i]:<12.3f} {torch_times[i]:<14.3f} {speedup:<8.2f}")
        
        # Memory should scale linearly with sequence length
        memory_ratios = [memory_usage[i] / memory_usage[0] for i in range(len(memory_usage))]
        seq_ratios = [seq_len / sequence_lengths[0] for seq_len in sequence_lengths]
        
        # Check that memory scaling is roughly linear (within 20% tolerance)
        for i, (mem_ratio, seq_ratio) in enumerate(zip(memory_ratios, seq_ratios)):
            if i > 0:  # Skip first element
                assert abs(mem_ratio - seq_ratio) / seq_ratio < 0.2, \
                    f"Memory scaling not linear at seq_len={sequence_lengths[i]}"
    
    def test_batch_size_scaling(self, device):
        """Test performance scaling with batch size"""
        seq_len, head_dim, n_heads = 512, 64, 8
        batch_sizes = [1, 2, 4, 8, 16]
        
        results_data = []
        
        for batch_size in batch_sizes:
            results = benchmark_attention(
                batch_size, n_heads, seq_len, head_dim,
                device=str(device), num_trials=20
            )
            results_data.append(results)
        
        print(f"\n📊 Batch Size Scaling:")
        print(f"{'Batch':<6} {'Flash (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<8} {'Throughput (GFLOPS)':<18}")
        print("-" * 70)
        
        for i, batch_size in enumerate(batch_sizes):
            results = results_data[i]
            flash_time = results.get('flash_attention_mean_ms', 0)
            torch_time = results.get('pytorch_attention_mean_ms', 0)
            speedup = results.get('speedup', 0)
            throughput = results.get('flash_throughput_gflops', 0)
            
            print(f"{batch_size:<6} {flash_time:<12.3f} {torch_time:<14.3f} {speedup:<8.2f} {throughput:<18.1f}")
            
            # Larger batches should have better throughput
            if i > 0:
                prev_throughput = results_data[i-1].get('flash_throughput_gflops', 0)
                if prev_throughput > 0 and throughput > 0:
                    assert throughput >= prev_throughput * 0.8, \
                        "Throughput should not decrease significantly with batch size"
    
    def test_head_scaling(self, device):
        """Test performance scaling with number of heads"""
        seq_len, head_dim, batch_size = 512, 64, 2
        head_counts = [4, 8, 16, 32]
        
        print(f"\n🔄 Head Count Scaling:")
        print(f"{'Heads':<6} {'Flash (ms)':<12} {'PyTorch (ms)':<14} {'Speedup':<8}")
        print("-" * 50)
        
        for n_heads in head_counts:
            results = benchmark_attention(
                batch_size, n_heads, seq_len, head_dim,
                device=str(device), num_trials=20
            )
            
            flash_time = results.get('flash_attention_mean_ms', 0)
            torch_time = results.get('pytorch_attention_mean_ms', 0)  
            speedup = results.get('speedup', 0)
            
            print(f"{n_heads:<6} {flash_time:<12.3f} {torch_time:<14.3f} {speedup:<8.2f}")
    
    def test_memory_efficiency(self, device):
        """Test memory efficiency compared to standard attention"""
        batch_size, n_heads, head_dim = 2, 8, 64
        sequence_lengths = [256, 512, 1024, 2048]
        
        print(f"\n💾 Memory Efficiency Analysis:")
        print(f"{'Seq Len':<8} {'FlashAttn (MB)':<15} {'Standard (MB)':<15} {'Reduction':<10}")
        print("-" * 55)
        
        for seq_len in sequence_lengths:
            # FlashAttention memory usage
            flash_memory = calculate_memory_usage(batch_size, n_heads, seq_len, head_dim)
            flash_memory_mb = flash_memory / (1024**2)
            
            # Standard attention memory (includes attention matrix)
            attention_matrix_size = batch_size * n_heads * seq_len * seq_len * 4  # float32
            standard_memory_mb = (flash_memory + attention_matrix_size) / (1024**2)
            
            reduction_factor = standard_memory_mb / flash_memory_mb
            
            print(f"{seq_len:<8} {flash_memory_mb:<15.1f} {standard_memory_mb:<15.1f} {reduction_factor:<10.1f}×")
            
            # Memory reduction should increase with sequence length
            expected_reduction = seq_len / 64  # Rough estimate
            assert reduction_factor >= min(expected_reduction * 0.5, 2.0), \
                f"Memory reduction too low for seq_len={seq_len}"
    
    @pytest.mark.slow
    def test_large_scale_performance(self, device, device_info):
        """Test performance on large-scale problems"""
        if device_info.get('device_0', {}).get('total_memory_gb', 0) < 16:
            pytest.skip("Requires GPU with at least 16GB memory")
        
        large_configs = [
            (4, 16, 2048, 64),   # Large sequence
            (8, 8, 1024, 128),   # Large batch + head_dim
            (2, 32, 1024, 64),   # Many heads
        ]
        
        print(f"\n🚀 Large Scale Performance:")
        print(f"{'Config':<20} {'Flash (ms)':<12} {'Memory (MB)':<12} {'Throughput (GFLOPS)':<18}")
        print("-" * 70)
        
        for batch_size, n_heads, seq_len, head_dim in large_configs:
            config_str = f"B{batch_size}_H{n_heads}_S{seq_len}_D{head_dim}"
            
            try:
                # Test with reduced trials for speed
                results = benchmark_attention(
                    batch_size, n_heads, seq_len, head_dim,
                    device=str(device), num_trials=10, compare_with_torch=False
                )
                
                flash_time = results.get('flash_attention_mean_ms', 0)
                memory_mb = results.get('memory_usage_mb', 0)
                throughput = results.get('flash_throughput_gflops', 0)
                
                print(f"{config_str:<20} {flash_time:<12.3f} {memory_mb:<12.1f} {throughput:<18.1f}")
                
            except RuntimeError as e:
                print(f"{config_str:<20} {'FAILED':<12} {str(e)[:30]:<12}")
    
    def test_profiling_overhead(self, device):
        """Test that profiling doesn't significantly impact performance"""
        batch_size, n_heads, seq_len, head_dim = 2, 8, 512, 64
        
        # Create test tensors
        Q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        K = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)  
        V = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        
        # Benchmark without profiling
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        for _ in range(50):
            _ = flash_attention_forward(Q, K, V)
        torch.cuda.synchronize()
        time_without_profiling = time.perf_counter() - start_time
        
        # Benchmark with memory profiling
        def attention_call():
            return flash_attention_forward(Q, K, V)
        
        start_time = time.perf_counter()
        for _ in range(50):
            _, memory_stats = profile_memory_usage(attention_call)
        torch.cuda.synchronize()
        time_with_profiling = time.perf_counter() - start_time
        
        overhead_ratio = time_with_profiling / time_without_profiling
        
        print(f"\n⏱️  Profiling Overhead:")
        print(f"   Without profiling: {time_without_profiling*1000:.1f}ms")
        print(f"   With profiling:    {time_with_profiling*1000:.1f}ms")
        print(f"   Overhead ratio:    {overhead_ratio:.2f}×")
        
        # Profiling should not add significant overhead
        assert overhead_ratio < 1.5, f"Profiling overhead too high: {overhead_ratio:.2f}×"
    
    def test_warmup_effect(self, device):
        """Test the effect of warmup iterations on performance"""
        batch_size, n_heads, seq_len, head_dim = 2, 8, 512, 64
        
        Q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        K = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        V = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        
        # Measure first few iterations
        cold_times = []
        for _ in range(5):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = flash_attention_forward(Q, K, V)
            torch.cuda.synchronize()
            cold_times.append((time.perf_counter() - start_time) * 1000)
        
        # Warmup
        for _ in range(20):
            _ = flash_attention_forward(Q, K, V)
        torch.cuda.synchronize()
        
        # Measure warmed-up performance
        warm_times = []
        for _ in range(10):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            _ = flash_attention_forward(Q, K, V)
            torch.cuda.synchronize()
            warm_times.append((time.perf_counter() - start_time) * 1000)
        
        cold_mean = np.mean(cold_times)
        warm_mean = np.mean(warm_times)
        improvement_ratio = cold_mean / warm_mean
        
        print(f"\n🔥 Warmup Effect:")
        print(f"   Cold start:  {cold_mean:.3f}ms ± {np.std(cold_times):.3f}ms")
        print(f"   Warmed up:   {warm_mean:.3f}ms ± {np.std(warm_times):.3f}ms")
        print(f"   Improvement: {improvement_ratio:.2f}×")
        
        # Warmed up performance should be better
        assert improvement_ratio > 1.0, "Warmup should improve performance"


class TestMemoryBehavior:
    """Test memory allocation and deallocation behavior"""
    
    def test_memory_cleanup(self, device):
        """Test that memory is properly cleaned up after operations"""
        initial_memory = torch.cuda.memory_allocated(device)
        
        batch_size, n_heads, seq_len, head_dim = 4, 8, 512, 64
        
        # Allocate and compute
        Q = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        K = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        V = torch.randn(batch_size, n_heads, seq_len, head_dim, device=device)
        
        output = flash_attention_forward(Q, K, V)
        
        # Memory should increase
        peak_memory = torch.cuda.memory_allocated(device)
        assert peak_memory > initial_memory, "Memory should increase during computation"
        
        # Clean up
        del Q, K, V, output
        gc.collect()
        torch.cuda.empty_cache()
        
        # Memory should return close to initial
        final_memory = torch.cuda.memory_allocated(device)
        memory_leak = final_memory - initial_memory
        
        print(f"\n🧹 Memory Cleanup:")
        print(f"   Initial: {initial_memory / (1024**2):.1f}MB")
        print(f"   Peak:    {peak_memory / (1024**2):.1f}MB") 
        print(f"   Final:   {final_memory / (1024**2):.1f}MB")
        print(f"   Leak:    {memory_leak / (1024**2):.1f}MB")
        
        # Allow some tolerance for memory fragmentation
        assert abs(memory_leak) < 10 * 1024 * 1024, f"Memory leak detected: {memory_leak / (1024**2):.1f}MB"


if __name__ == "__main__":
    # Run performance tests
    pytest.main([__file__, "-v", "--tb=short", "-m", "not slow"])