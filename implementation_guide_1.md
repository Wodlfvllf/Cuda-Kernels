# CUDA Kernel Implementation Guide
## From Zero to Production-Ready Kernels

---

## Table of Contents
1. [Project Structure](#project-structure)
2. [Environment Setup](#environment-setup)
3. [Kernel Development Workflow](#kernel-development-workflow)
4. [Implementation Templates](#implementation-templates)
5. [Optimization Techniques](#optimization-techniques)
6. [Testing & Benchmarking](#testing-benchmarking)
7. [Common Pitfalls](#common-pitfalls)

---

## 1. Project Structure

### Standard Directory Layout
```
kernel-name/
├── README.md                    # Kernel description, usage, benchmarks
├── requirements.txt             # Python dependencies
├── setup.py                     # Build configuration
├── include/                     # Header files
│   ├── common.cuh              # Shared utilities
│   ├── naive_kernel.cuh        # Baseline implementation
│   ├── optimized_kernel.cuh    # Optimized versions
│   └── config.h                # Compile-time configuration
├── src/                        # Source files
│   ├── naive_kernel.cu         # Baseline CUDA implementation
│   ├── optimized_kernel.cu     # Optimized CUDA implementation
│   ├── kernel_ops.cpp          # PyTorch binding (C++)
│   └── kernel.py               # Python wrapper
├── python/                     # Additional Python code
│   ├── __init__.py
│   ├── setup.py
│   └── test_kernel.py
├── tests/                      # Test suite
│   ├── test_correctness.py    # Numerical correctness
│   ├── test_performance.py    # Speed benchmarks
│   └── test_edge_cases.py     # Edge case handling
├── benchmarks/                 # Performance analysis
│   ├── benchmark_kernel.py
│   ├── plots.ipynb
│   └── compare_backends.py
└── docs/                       # Documentation
    ├── design.md               # Design decisions
    ├── algorithm.md            # Algorithm explanation
    └── results.md              # Benchmark results
```

---

## 2. Environment Setup

### Step 1: Install CUDA Toolkit
```bash
# Check CUDA version
nvcc --version

# If not installed, download from NVIDIA
# https://developer.nvidia.com/cuda-downloads

# Verify installation
nvidia-smi
```

### Step 2: Setup Python Environment
```bash
# Create virtual environment
python -m venv cuda_env
source cuda_env/bin/activate  # Linux/Mac
# cuda_env\Scripts\activate   # Windows

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ninja pytest numpy matplotlib pandas
```

### Step 3: Verify PyTorch CUDA
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"Device: {torch.cuda.get_device_name(0)}")
```

---

## 3. Kernel Development Workflow

### Phase 1: Algorithm Design (Day 1)

#### A. Understand the Operation
```python
# Example: Softmax operation
# Input: tensor of shape (batch, seq_len, hidden_dim)
# Output: softmax along last dimension

# Mathematical formula:
# softmax(x_i) = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
```

#### B. Identify Parallelization Strategy
- **Element-wise**: Each thread handles one element
- **Row-wise**: Each block/warp handles one row
- **Block-wise**: Tile the computation
- **Reduction**: Use tree-based or warp-level reductions

#### C. Memory Access Pattern Analysis
```
1. Are memory accesses coalesced?
2. Is there data reuse (shared memory opportunity)?
3. What's the arithmetic intensity?
4. Are there reduction operations?
```

### Phase 2: Naive Implementation (Days 2-3)

#### Step 1: Create Header File
```cuda
// include/naive_softmax.cuh
#ifndef NAIVE_SOFTMAX_CUH
#define NAIVE_SOFTMAX_CUH

#include <cuda_runtime.h>

// Kernel declaration
template<typename T>
__global__ void naive_softmax_kernel(
    const T* input,
    T* output,
    int batch_size,
    int seq_len,
    int hidden_dim
);

// Host function
void naive_softmax_forward(
    const float* input,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_dim
);

#endif
```

#### Step 2: Implement CUDA Kernel
```cuda
// src/naive_softmax.cu
#include "naive_softmax.cuh"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

template<typename T>
__global__ void naive_softmax_kernel(
    const T* input,
    T* output,
    int batch_size,
    int seq_len,
    int hidden_dim
) {
    // Each thread handles one row
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < batch_size * seq_len) {
        const T* row_input = input + idx * hidden_dim;
        T* row_output = output + idx * hidden_dim;
        
        // Find max for numerical stability
        T max_val = row_input[0];
        for (int i = 1; i < hidden_dim; i++) {
            max_val = max(max_val, row_input[i]);
        }
        
        // Compute exp and sum
        T sum = 0;
        for (int i = 0; i < hidden_dim; i++) {
            T exp_val = exp(row_input[i] - max_val);
            row_output[i] = exp_val;
            sum += exp_val;
        }
        
        // Normalize
        for (int i = 0; i < hidden_dim; i++) {
            row_output[i] /= sum;
        }
    }
}

// Host function
void naive_softmax_forward(
    const float* input,
    float* output,
    int batch_size,
    int seq_len,
    int hidden_dim
) {
    int total_rows = batch_size * seq_len;
    int threads = 256;
    int blocks = (total_rows + threads - 1) / threads;
    
    naive_softmax_kernel<<<blocks, threads>>>(
        input, output, batch_size, seq_len, hidden_dim
    );
    
    cudaDeviceSynchronize();
}

// Explicit instantiation
template __global__ void naive_softmax_kernel<float>(
    const float*, float*, int, int, int
);
```

#### Step 3: Create PyTorch Binding
```cpp
// src/softmax_ops.cpp
#include <torch/extension.h>
#include "naive_softmax.cuh"

torch::Tensor naive_softmax_forward_torch(torch::Tensor input) {
    // Check input
    TORCH_CHECK(input.is_cuda(), "Input must be CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "Input must be contiguous");
    
    // Get dimensions
    auto sizes = input.sizes();
    int batch_size = sizes[0];
    int seq_len = sizes[1];
    int hidden_dim = sizes[2];
    
    // Allocate output
    auto output = torch::empty_like(input);
    
    // Launch kernel
    naive_softmax_forward(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        batch_size, seq_len, hidden_dim
    );
    
    return output;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("naive_softmax_forward", &naive_softmax_forward_torch, 
          "Naive softmax forward pass");
}
```

#### Step 4: Create Python Wrapper
```python
# src/softmax.py
import torch
from torch.utils.cpp_extension import load

# JIT compile
softmax_cuda = load(
    name="softmax_cuda",
    sources=["src/softmax_ops.cpp", "src/naive_softmax.cu"],
    extra_cuda_cflags=["-O3"],
    verbose=True
)

class NaiveSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = softmax_cuda.naive_softmax_forward(input)
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        # Softmax gradient: grad_input = output * (grad_output - sum(output * grad_output))
        grad_input = output * (grad_output - (output * grad_output).sum(dim=-1, keepdim=True))
        return grad_input

def naive_softmax(input):
    return NaiveSoftmax.apply(input)
```

#### Step 5: Write setup.py
```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='softmax_cuda',
    ext_modules=[
        CUDAExtension(
            name='softmax_cuda',
            sources=[
                'src/softmax_ops.cpp',
                'src/naive_softmax.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-use_fast_math',
                    '--expt-relaxed-constexpr',
                ]
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
```

### Phase 3: Testing (Day 4)

#### Create Correctness Test
```python
# tests/test_correctness.py
import torch
import pytest
from src.softmax import naive_softmax

def test_softmax_correctness():
    # Test dimensions
    batch_size = 4
    seq_len = 128
    hidden_dim = 768
    
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda')
    
    # Compute with custom kernel
    custom_output = naive_softmax(x)
    
    # Compute with PyTorch
    torch_output = torch.softmax(x, dim=-1)
    
    # Check correctness
    assert torch.allclose(custom_output, torch_output, rtol=1e-4, atol=1e-5)
    print("✓ Correctness test passed")

def test_softmax_gradient():
    x = torch.randn(2, 64, 256, device='cuda', requires_grad=True)
    
    # Forward
    y = naive_softmax(x)
    loss = y.sum()
    
    # Backward
    loss.backward()
    
    # Check gradient exists
    assert x.grad is not None
    print("✓ Gradient test passed")

if __name__ == "__main__":
    test_softmax_correctness()
    test_softmax_gradient()
```

### Phase 4: Optimization (Days 5-7)

#### Optimization 1: Shared Memory
```cuda
// Optimized with shared memory
template<typename T>
__global__ void optimized_softmax_kernel(
    const T* input,
    T* output,
    int batch_size,
    int seq_len,
    int hidden_dim
) {
    extern __shared__ char shared_mem[];
    T* shared_data = reinterpret_cast<T*>(shared_mem);
    
    int row_idx = blockIdx.x;
    if (row_idx >= batch_size * seq_len) return;
    
    const T* row_input = input + row_idx * hidden_dim;
    T* row_output = output + row_idx * hidden_dim;
    
    // Load data to shared memory
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        shared_data[i] = row_input[i];
    }
    __syncthreads();
    
    // Find max using reduction
    T thread_max = -INFINITY;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        thread_max = max(thread_max, shared_data[i]);
    }
    
    // Warp-level reduction for max
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_max = max(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    }
    
    // Broadcast max to shared memory
    __shared__ T max_val;
    if (threadIdx.x % 32 == 0) {
        atomicMax(&max_val, thread_max);
    }
    __syncthreads();
    
    // Compute exp and sum
    T thread_sum = 0;
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        T exp_val = exp(shared_data[i] - max_val);
        shared_data[i] = exp_val;
        thread_sum += exp_val;
    }
    
    // Warp-level reduction for sum
    for (int offset = 16; offset > 0; offset /= 2) {
        thread_sum += __shfl_down_sync(0xffffffff, thread_sum, offset);
    }
    
    // Broadcast sum
    __shared__ T sum_val;
    if (threadIdx.x % 32 == 0) {
        atomicAdd(&sum_val, thread_sum);
    }
    __syncthreads();
    
    // Normalize and write output
    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        row_output[i] = shared_data[i] / sum_val;
    }
}
```

#### Optimization 2: Vectorized Memory Access
```cuda
// Use float4 for coalesced memory access
template<typename T>
__global__ void vectorized_softmax_kernel(
    const T* input,
    T* output,
    int batch_size,
    int seq_len,
    int hidden_dim
) {
    // Assume hidden_dim is divisible by 4
    int row_idx = blockIdx.x;
    if (row_idx >= batch_size * seq_len) return;
    
    const float4* input4 = reinterpret_cast<const float4*>(input + row_idx * hidden_dim);
    float4* output4 = reinterpret_cast<float4*>(output + row_idx * hidden_dim);
    
    int vec_size = hidden_dim / 4;
    
    // Find max (vectorized)
    float thread_max = -INFINITY;
    for (int i = threadIdx.x; i < vec_size; i += blockDim.x) {
        float4 vals = input4[i];
        thread_max = fmaxf(thread_max, fmaxf(fmaxf(vals.x, vals.y), fmaxf(vals.z, vals.w)));
    }
    
    // Continue with reduction...
}
```

### Phase 5: Benchmarking (Day 8)

#### Create Benchmark Script
```python
# benchmarks/benchmark_softmax.py
import torch
import time
from src.softmax import naive_softmax

def benchmark_kernel(func, input, num_runs=100, warmup=10):
    # Warmup
    for _ in range(warmup):
        func(input)
    
    # Benchmark
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_runs):
        func(input)
    
    torch.cuda.synchronize()
    end = time.time()
    
    return (end - start) / num_runs * 1000  # ms

def run_benchmarks():
    configs = [
        (4, 128, 768),
        (8, 512, 1024),
        (16, 1024, 2048),
    ]
    
    for batch, seq_len, hidden in configs:
        x = torch.randn(batch, seq_len, hidden, device='cuda')
        
        # PyTorch
        torch_time = benchmark_kernel(lambda t: torch.softmax(t, dim=-1), x)
        
        # Custom
        custom_time = benchmark_kernel(naive_softmax, x)
        
        speedup = torch_time / custom_time
        print(f"Config: {batch}x{seq_len}x{hidden}")
        print(f"  PyTorch: {torch_time:.3f} ms")
        print(f"  Custom:  {custom_time:.3f} ms")
        print(f"  Speedup: {speedup:.2f}x\n")

if __name__ == "__main__":
    run_benchmarks()
```

---

## 4. Implementation Templates

### Template 1: Element-wise Operation
```cuda
// Fused GELU Activation
__global__ void gelu_kernel(const float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x = input[idx];
        output[idx] = 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x * x * x)));
    }
}
```

### Template 2: Reduction Operation
```cuda
// Sum reduction with warp primitives
__global__ void sum_reduce_kernel(const float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    // Load and reduce in shared memory
    sdata[tid] = (idx < n) ? input[idx] : 0.0f;
    __syncthreads();
    
    // Tree reduction
    for (int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Warp reduction
    if (tid < 32) {
        float val = sdata[tid];
        for (int offset = 16; offset > 0; offset /= 2) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (tid == 0) output[blockIdx.x] = val;
    }
}
```

### Template 3: Matrix Tiling
```cuda
// Tiled matrix multiplication
#define TILE_SIZE 16

__global__ void matmul_tiled_kernel(
    const float* A, const float* B, float* C,
    int M, int N, int K
) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    
    float sum = 0.0f;
    
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles
        if (row < M && t * TILE_SIZE + threadIdx.x < K)
            As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;
            
        if (col < N && t * TILE_SIZE + threadIdx.y < K)
            Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
            
        __syncthreads();
        
        // Compute
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

---

## 5. Optimization Techniques

### Memory Optimization

#### 1. Coalesced Memory Access
```cuda
// BAD: Strided access
for (int i = threadIdx.x; i < N; i += blockDim.x * stride) {
    output[i] = input[i];
}

// GOOD: Coalesced access
for (int i = threadIdx.x; i < N; i += blockDim.x) {
    output[i] = input[i];
}
```

#### 2. Shared Memory Bank Conflicts
```cuda
// BAD: Bank conflicts
__shared__ float data[32][32];
data[threadIdx.x][threadIdx.y] = value;  // Conflict!

// GOOD: Padded to avoid conflicts
__shared__ float data[32][33];  // +1 padding
data[threadIdx.x][threadIdx.y] = value;
```

#### 3. Vectorized Loads
```cuda
// Use float2, float4 for vectorized access
float4 val = reinterpret_cast<const float4*>(input)[idx];
// Process val.x, val.y, val.z, val.w
```

### Compute Optimization

#### 1. Warp-Level Primitives
```cuda
// Use shuffle instructions for warp reductions
float val = /* ... */;
for (int offset = 16; offset > 0; offset /= 2) {
    val += __shfl_down_sync(0xffffffff, val, offset);
}
```

#### 2. Fast Math
```cuda
// Use fast math intrinsics
__fdividef(a, b)  // Fast divide
__expf(x)          // Fast exp
__logf(x)          // Fast log
```

#### 3. Loop Unrolling
```cuda
#pragma unroll
for (int i = 0; i < 8; i++) {
    sum += data[i];
}
```

### Occupancy Optimization

```cuda
// Check occupancy
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, kernel, 0, 0);

// Profile register usage
nvcc --ptxas-options=-v kernel.cu
```

---

## 6. Testing & Benchmarking

### Profiling with Nsight Compute
```bash
# Profile kernel
ncu --set full -o profile python test.py

# View report
ncu-ui profile.ncu-rep
```

### Key Metrics to Track
1. **Memory Bandwidth Utilization**: Target >80%
2. **Compute Utilization**: Target >70% for compute-bound
3. **Occupancy**: Target >50%
4. **Register Usage**: Minimize to increase occupancy
5. **Shared Memory Bank Conflicts**: Should be near zero

---

## 7. Common Pitfalls

### Pitfall 1: Race Conditions
```cuda
// BAD: Race condition
__shared__ float sum;
sum += thread_value;  // Multiple threads write!

// GOOD: Use atomics
atomicAdd(&sum, thread_value);
```

### Pitfall 2: Insufficient Synchronization
```cuda
// BAD: Missing sync
__shared__ float data[256];
data[threadIdx.x] = input[idx];
// Missing __syncthreads()!
float val = data[threadIdx.x + 1];  // May read uninitialized!

// GOOD: Proper sync
data[threadIdx.x] = input[idx];
__syncthreads();
float val = data[threadIdx.x + 1];
```

### Pitfall 3: Numerical Instability
```cuda
// BAD: Numerically unstable softmax
exp(x[i]) / sum(exp(x))  // Can overflow!

// GOOD: Stable softmax
exp(x[i] - max(x)) / sum(exp(x - max(x)))
```

---

## Quick Reference Commands

```bash
# Compile and run
python setup.py install
python tests/test_correctness.py

# Profile
nsys profile python benchmarks/benchmark.py
ncu python benchmarks/benchmark.py

# Check CUDA errors
cuda-memcheck python test.py

# Build with debug info
CUDA_VISIBLE_DEVICES=0 python setup.py build_ext --inplace
```

---

## Next Steps

1. Start with the naive implementation
2. Write tests BEFORE optimizing
3. Profile to find bottlenecks
4. Optimize incrementally
5. Benchmark against PyTorch
6. Document your findings

Good luck with your CUDA kernel development! 🚀