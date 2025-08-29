#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include "src/layernorm_benchmark.cu"

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Custom LayerNorm CUDA Kernel Benchmarks" << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "SM Count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Shared Memory per Block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Max Threads per Block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "========================================" << std::endl;
    
    // Test different configurations
    std::vector<std::pair<int, int>> configs = {
        {32, 768},    // BERT-base hidden dim
        {64, 1024},   // BERT-large hidden dim
        {128, 2048},  // GPT-2 medium
        {256, 4096},  // Large model
        {512, 8192},  // Very large model
    };
    
    for (auto& config : configs) {
        int batch_size = config.first;
        int hidden_dim = config.second;
        
        LayerNormBenchmark bench(batch_size, hidden_dim);
        bench.run_benchmarks();
    }
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Benchmark Complete!" << std::endl;
    std::cout << "========================================" << std::endl;
    
    return 0;
}
```

## File: python/setup.py

```python
from setuptools import setup, Extension
from torch.utils import cpp_extension
import os

# Get CUDA compute capability
import torch
cuda_version = torch.version.cuda
compute_capability = torch.cuda.get_device_capability()
arch_flag = f"sm_{compute_capability[0]}{compute_capability[1]}"

sources = [
    '../src/layernorm_ops.cpp',
    '../src/naive_layernorm.cu',
    '../src/fused_layernorm.cu',
    '../src/vectorized_layernorm.cu',
    '../src/welford_layernorm.cu',
]

include_dirs = [
    '../include',
]

setup(
    name='layernorm_cuda',
    ext_modules=[
        cpp_extension.CUDAExtension(
            name='layernorm_cuda',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    f'-arch={arch_flag}',
                    '-use_fast_math',
                    '--expt-relaxed-constexpr',
                    '-DTORCH_EXTENSION',
                ],
            },
        ),
    ],
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    },
    zip_safe=False,
)