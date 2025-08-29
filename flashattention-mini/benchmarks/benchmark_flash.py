"""
Comprehensive benchmarking suite for FlashAttention-Mini

This script runs extensive performance comparisons across different:
- Problem sizes (batch size, sequence length, heads, dimensions)
- Hardware configurations
- Memory usage patterns
- Throughput measurements

Results are saved and can be used for performance analysis and optimization.
"""

import argparse
import json
import time
import os
from datetime import datetime
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

import torch
import torch.nn.functional as F

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from attention import FlashAttentionMini, flash_attention_forward
from utils import (
    benchmark_attention, calculate_memory_usage, get_device_info,
    validate_attention_output, profile_memory_usage
)

# Comprehensive benchmark configurations
BENCHMARK_CONFIGS = {
    'quick': [
        # (batch_size, n_heads, seq_len, head_dim)
        (2, 8, 256, 64),
        (2, 8, 512, 64),
        (4, 8, 512, 64),
        (2, 8, 1024, 64),
    ],
    'standard': [
        # Batch size scaling
        (1, 8, 512, 64), (2, 8, 512, 64), (4, 8, 512, 64), (8, 8, 512, 64),
        # Sequence length scaling  
        (2, 8, 128, 64), (2, 8, 256, 64), (2, 8, 512, 64), (2, 8, 1024, 64), (2, 8, 2048, 64),
        # Head count scaling
        (2, 4, 512, 64), (2, 8, 512, 64), (2, 16, 512, 64), (2, 32, 512, 64),
        # Head dimension scaling
        (2, 8, 512, 64), (2, 4, 512, 128),
        # Mixed large configs
        (4, 16, 1024, 64), (8, 8, 1024, 64),
    ],
    'extensive': [
        # All combinations for thorough analysis
        (b, h, s, d) for b in [1, 2, 4, 8, 16]
                     for h in [4, 8, 16, 32]  
                     for s in [128, 256, 512, 1024, 2048]
                     for d in [64, 128]
                     if b * h * s * d <= 16 * 1024 * 1024  # Memory constraint
    ]
}


class FlashAttentionBenchmark:
    """Comprehensive benchmarking suite for FlashAttention-Mini"""
    
    def __init__(self, device='cuda', output_dir='benchmark_results'):
        self.device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        self.output_dir = output_dir
        self.device_info = get_device_info()
        self.results = []
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🚀 FlashAttention-Mini Benchmark Suite")
        print(f"   Device: {self.device}")
        if self.device.type == 'cuda':
            print(f"   GPU: {self.device_info.get('device_0', {}).get('name', 'Unknown')}")
            print(f"   Memory: {self.device_info.get('device_0', {}).get('total_memory_gb', 0):.1f}GB")
        print(f"   Output: {output_dir}")
        print("-" * 60)
        
    def reference_attention(self, Q, K, V, scale=None):
        """Reference PyTorch attention implementation"""
        if scale is None:
            scale = 1.0 / (Q.size(-1) ** 0.5)
            
        scores = torch.matmul(Q, K.transpose(-2, -1)) * scale
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output
        
    def run_single_benchmark(
        self, 
        batch_size: int, 
        n_heads: int, 
        seq_len: int, 
        head_dim: int,
        num_warmup: int = 10,
        num_trials: int = 50,
        validate_correctness: bool = True
    ) -> Dict:
        """Run benchmark for a single configuration"""
        
        # Create test tensors
        Q = torch.randn(batch_size, n_heads, seq_len, head_dim, 
                       device=self.device, dtype=torch.float32)
        K = torch.randn(batch_size, n_heads, seq_len, head_dim,
                       device=self.device, dtype=torch.float32)
        V = torch.randn(batch_size, n_heads, seq_len, head_dim,
                       device=self.device, dtype=torch.float32)
        
        scale = 1.0 / (head_dim ** 0.5)
        result = {
            'batch_size': batch_size,
            'n_heads': n_heads,
            'seq_len': seq_len, 
            'head_dim': head_dim,
            'total_elements': batch_size * n_heads * seq_len * head_dim,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Memory usage calculation
            memory_usage = calculate_memory_usage(batch_size, n_heads, seq_len, head_dim, True)
            result['memory_usage'] = memory_usage
            
            # Warmup
            for _ in range(num_warmup):
                try:
                    _ = flash_attention_forward(Q, K, V, scale)
                except Exception:
                    pass
                _ = self.reference_attention(Q, K, V, scale)
            torch.cuda.synchronize() if self.device.type == 'cuda' else None
            
            # Benchmark FlashAttention-Mini
            flash_times = []
            flash_memory_stats = []
            
            for _ in range(num_trials):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    start_memory = torch.cuda.memory_allocated()
                    
                start_time = time.perf_counter()
                try:
                    flash_output = flash_attention_forward(Q, K, V, scale)
                    success = True
                except Exception as e:
                    result['flash_error'] = str(e)
                    success = False
                    break
                    
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    end_memory = torch.cuda.memory_allocated()
                    flash_memory_stats.append(end_memory - start_memory)
                    
                flash_times.append((time.perf_counter() - start_time) * 1000)  # Convert to ms
            
            if success:
                result['flash_attention_mean_ms'] = np.mean(flash_times)
                result['flash_attention_std_ms'] = np.std(flash_times)
                result['flash_attention_median_ms'] = np.median(flash_times)
                
                if flash_memory_stats:
                    result['flash_memory_used_mb'] = np.mean(flash_memory_stats) / (1024**2)
            
            # Benchmark PyTorch reference
            torch_times = []
            torch_memory_stats = []
            
            for _ in range(num_trials):
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    start_memory = torch.cuda.memory_allocated()
                    
                start_time = time.perf_counter()
                torch_output = self.reference_attention(Q, K, V, scale)
                
                if self.device.type == 'cuda':
                    torch.cuda.synchronize()
                    end_memory = torch.cuda.memory_allocated()
                    torch_memory_stats.append(end_memory - start_memory)
                    
                torch_times.append((time.perf_counter() - start_time) * 1000)
            
            result['pytorch_attention_mean_ms'] = np.mean(torch_times)
            result['pytorch_attention_std_ms'] = np.std(torch_times)
            result['pytorch_attention_median_ms'] = np.median(torch_times)
            
            if torch_memory_stats:
                result['pytorch_memory_used_mb'] = np.mean(torch_memory_stats) / (1024**2)
            
            # Calculate metrics
            if success and result['flash_attention_mean_ms'] > 0:
                result['speedup'] = result['pytorch_attention_mean_ms'] / result['flash_attention_mean_ms']
                
                # Throughput calculation (approximate FLOPS)
                total_flops = 2 * batch_size * n_heads * seq_len * seq_len * head_dim  # QK^T + Attn*V
                result['flash_throughput_gflops'] = total_flops / (result['flash_attention_mean_ms'] / 1000) / 1e9
                result['pytorch_throughput_gflops'] = total_flops / (result['pytorch_attention_mean_ms'] / 1000) / 1e9
            
            # Correctness validation
            if success and validate_correctness:
                validation = validate_attention_output(flash_output, torch_output, rtol=1e-4, atol=1e-6)
                result['correctness'] = validation
                result['max_error'] = validation['max_absolute_error']
                result['is_correct'] = validation['is_within_tolerance']
            
        except Exception as e:
            result['error'] = str(e)
            result['success'] = False
            return result
        
        result['success'] = success
        return result
    
    def run_scaling_analysis(self):
        """Run scaling analysis across different dimensions"""
        scaling_results = {}
        
        print(f"\n📈 Running Scaling Analysis...")
        
        # Batch size scaling
        print("   Batch size scaling...")
        batch_configs = [(b, 8, 512, 64) for b in [1, 2, 4, 8, 16, 32]]
        scaling_results['batch_scaling'] = []
        for config in batch_configs:
            try:
                result = self.run_single_benchmark(*config, num_trials=20)
                scaling_results['batch_scaling'].append(result)
                if result.get('success', False):
                    print(f"     B={config[0]}: {result['flash_attention_mean_ms']:.2f}ms, "
                          f"Speedup: {result.get('speedup', 0):.2f}×")
            except Exception as e:
                print(f"     B={config[0]}: Failed - {e}")
        
        # Sequence length scaling
        print("   Sequence length scaling...")
        seq_configs = [(2, 8, s, 64) for s in [128, 256, 512, 1024, 2048, 4096]]
        scaling_results['sequence_scaling'] = []
        for config in seq_configs:
            try:
                result = self.run_single_benchmark(*config, num_trials=20)
                scaling_results['sequence_scaling'].append(result)
                if result.get('success', False):
                    print(f"     S={config[2]}: {result['flash_attention_mean_ms']:.2f}ms, "
                          f"Memory: {result['memory_usage']['total']/(1024**2):.1f}MB")
            except Exception as e:
                print(f"     S={config[2]}: Failed - {e}")
        
        # Head scaling
        print("   Head count scaling...")
        head_configs = [(2, h, 512, 64) for h in [4, 8, 16, 32, 64]]
        scaling_results['head_scaling'] = []
        for config in head_configs:
            try:
                result = self.run_single_benchmark(*config, num_trials=20)
                scaling_results['head_scaling'].append(result)
                if result.get('success', False):
                    print(f"     H={config[1]}: {result['flash_attention_mean_ms']:.2f}ms, "
                          f"Speedup: {result.get('speedup', 0):.2f}×")
            except Exception as e:
                print(f"     H={config[1]}: Failed - {e}")
        
        return scaling_results
    
    def run_benchmark_suite(self, config_name='standard', num_trials=50):
        """Run the full benchmark suite"""
        configs = BENCHMARK_CONFIGS.get(config_name, BENCHMARK_CONFIGS['standard'])
        
        print(f"🏃 Running {config_name} benchmark suite...")
        print(f"   Configurations: {len(configs)}")
        print(f"   Trials per config: {num_trials}")
        print()
        
        results = []
        total_configs = len(configs)
        
        for i, (batch_size, n_heads, seq_len, head_dim) in enumerate(configs):
            print(f"[{i+1:3d}/{total_configs}] B={batch_size}, H={n_heads}, S={seq_len}, D={head_dim}", end="")
            
            start_time = time.time()
            result = self.run_single_benchmark(
                batch_size, n_heads, seq_len, head_dim, num_trials=num_trials
            )
            elapsed = time.time() - start_time
            
            results.append(result)
            
            if result.get('success', False):
                flash_time = result.get('flash_attention_mean_ms', 0)
                speedup = result.get('speedup', 0)
                memory_mb = result.get('memory_usage', {}).get('total', 0) / (1024**2)
                print(f" → {flash_time:.2f}ms ({speedup:.2f}× speedup, {memory_mb:.1f}MB) [{elapsed:.1f}s]")
            else:
                error = result.get('error', result.get('flash_error', 'Unknown error'))
                print(f" → FAILED: {error}")
        
        self.results.extend(results)
        return results
    
    def save_results(self, filename=None):
        """Save benchmark results to files"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"flashattention_benchmark_{timestamp}"
        
        # Save as JSON
        json_path = os.path.join(self.output_dir, f"{filename}.json")
        with open(json_path, 'w') as f:
            json.dump({
                'device_info': self.device_info,
                'benchmark_info': {
                    'timestamp': datetime.now().isoformat(),
                    'total_configs': len(self.results),
                    'device': str(self.device)
                },
                'results': self.results
            }, f, indent=2)
        
        # Save as CSV for easy analysis
        csv_path = os.path.join(self.output_dir, f"{filename}.csv")
        df = pd.json_normalize(self.results)
        df.to_csv(csv_path, index=False)
        
        # Create summary report
        self.create_summary_report(filename)
        
        print(f"\n💾 Results saved:")
        print(f"   JSON: {json_path}")
        print(f"   CSV:  {csv_path}")
        print(f"   Summary: {os.path.join(self.output_dir, f'{filename}_summary.txt')}")
        
        return json_path, csv_path
    
    def create_summary_report(self, filename):
        """Create a human-readable summary report"""
        summary_path = os.path.join(self.output_dir, f"{filename}_summary.txt")
        
        successful_results = [r for r in self.results if r.get('success', False)]
        
        with open(summary_path, 'w') as f:
            f.write("FlashAttention-Mini Benchmark Summary\n")
            f.write("=" * 50 + "\n\n")
            
            # Device info
            f.write(f"Device: {self.device}\n")
            if self.device.type == 'cuda':
                device_0 = self.device_info.get('device_0', {})
                f.write(f"GPU: {device_0.get('name', 'Unknown')}\n")
                f.write(f"Memory: {device_0.get('total_memory_gb', 0):.1f}GB\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Overall statistics
            f.write("Overall Statistics:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Total configurations tested: {len(self.results)}\n")
            f.write(f"Successful runs: {len(successful_results)}\n")
            f.write(f"Failed runs: {len(self.results) - len(successful_results)}\n\n")
            
            if successful_results:
                # Performance statistics
                speedups = [r.get('speedup', 0) for r in successful_results if r.get('speedup', 0) > 0]
                flash_times = [r.get('flash_attention_mean_ms', 0) for r in successful_results]
                
                f.write("Performance Summary:\n")
                f.write("-" * 20 + "\n")
                if speedups:
                    f.write(f"Mean speedup: {np.mean(speedups):.2f}×\n")
                    f.write(f"Median speedup: {np.median(speedups):.2f}×\n")
                    f.write(f"Best speedup: {np.max(speedups):.2f}×\n")
                    f.write(f"Worst speedup: {np.min(speedups):.2f}×\n")
                f.write(f"Mean execution time: {np.mean(flash_times):.2f}ms\n")
                f.write(f"Median execution time: {np.median(flash_times):.2f}ms\n\n")
                
                # Top performers
                f.write("Top 5 Fastest Configurations:\n")
                f.write("-" * 30 + "\n")
                sorted_by_speed = sorted(successful_results, key=lambda x: x.get('flash_attention_mean_ms', float('inf')))
                for i, result in enumerate(sorted_by_speed[:5]):
                    config = f"B{result['batch_size']}_H{result['n_heads']}_S{result['seq_len']}_D{result['head_dim']}"
                    time_ms = result.get('flash_attention_mean_ms', 0)
                    speedup = result.get('speedup', 0)
                    f.write(f"{i+1}. {config}: {time_ms:.2f}ms ({speedup:.2f}× speedup)\n")
                f.write("\n")
                
                # Best speedups
                f.write("Top 5 Best Speedups:\n")
                f.write("-" * 20 + "\n")
                sorted_by_speedup = sorted([r for r in successful_results if r.get('speedup', 0) > 0], 
                                         key=lambda x: x.get('speedup', 0), reverse=True)
                for i, result in enumerate(sorted_by_speedup[:5]):
                    config = f"B{result['batch_size']}_H{result['n_heads']}_S{result['seq_len']}_D{result['head_dim']}"
                    speedup = result.get('speedup', 0)
                    time_ms = result.get('flash_attention_mean_ms', 0)
                    f.write(f"{i+1}. {config}: {speedup:.2f}× speedup ({time_ms:.2f}ms)\n")


def main():
    """Main benchmarking script"""
    parser = argparse.ArgumentParser(description='FlashAttention-Mini Comprehensive Benchmark')
    parser.add_argument('--config', choices=['quick', 'standard', 'extensive'], 
                       default='standard', help='Benchmark configuration')
    parser.add_argument('--trials', type=int, default=50, 
                       help='Number of trials per configuration')
    parser.add_argument('--output-dir', default='benchmark_results',
                       help='Output directory for results')
    parser.add_argument('--device', default='cuda',
                       help='Device to run benchmarks on')
    parser.add_argument('--scaling', action='store_true',
                       help='Run additional scaling analysis')
    
    args = parser.parse_args()
    
    # Create benchmark suite
    benchmark = FlashAttentionBenchmark(device=args.device, output_dir=args.output_dir)
    
    # Run main benchmark suite
    results = benchmark.run_benchmark_suite(config_name=args.config, num_trials=args.trials)
    
    # Run scaling analysis if requested
    if args.scaling:
        scaling_results = benchmark.run_scaling_analysis()
        # Add scaling results to main results
        for category, category_results in scaling_results.items():
            benchmark.results.extend(category_results)
    
    # Save results
    benchmark.save_results()
    
    print(f"\n✅ Benchmark completed!")
    print(f"   Total configurations: {len(benchmark.results)}")
    print(f"   Successful runs: {len([r for r in benchmark.results if r.get('success', False)])}")
    
    # Print quick summary
    successful_results = [r for r in benchmark.results if r.get('success', False)]
    if successful_results:
        speedups = [r.get('speedup', 0) for r in successful_results if r.get('speedup', 0) > 0]
        if speedups:
            print(f"   Mean speedup: {np.mean(speedups):.2f}×")
            print(f"   Best speedup: {np.max(speedups):.2f}×")


if __name__ == "__main__":
    main()