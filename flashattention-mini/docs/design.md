# FlashAttention-Mini Design Document

## Overview

FlashAttention-Mini is a simplified, educational implementation of the FlashAttention algorithm that demonstrates the core tiling and memory optimization techniques. This document details the implementation design, algorithmic choices, and optimization strategies.

## Table of Contents

1. [Algorithm Foundation](#algorithm-foundation)
2. [Tiling Strategy](#tiling-strategy)  
3. [Online Softmax Implementation](#online-softmax-implementation)
4. [Memory Management](#memory-management)
5. [CUDA Kernel Design](#cuda-kernel-design)
6. [PyTorch Integration](#pytorch-integration)
7. [Performance Optimizations](#performance-optimizations)
8. [Limitations and Future Work](#limitations-and-future-work)

## Algorithm Foundation

### Mathematical Background

The attention mechanism computes:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- Q: Query matrix [seq_len, d_k]
- K: Key matrix [seq_len, d_k]  
- V: Value matrix [seq_len, d_v]
- d_k: Key dimension

### Problem with Standard Implementation

The naive implementation has quadratic memory complexity O(n²) due to materializing the attention matrix QK^T. For sequence length n=4096:
- Attention matrix: 4096² × 4 bytes = 64MB per head
- With 16 heads: 1GB just for attention matrices
- Memory becomes prohibitive for longer sequences

### FlashAttention Solution

FlashAttention solves this by:
1. **Tiling**: Divide computation into blocks that fit in fast memory (SRAM)
2. **Online Computation**: Compute softmax without materializing full matrix
3. **Recomputation**: Trade memory for computation in backward pass

## Tiling Strategy

### Block Decomposition

We divide the computation into blocks:
- Q blocks: [BLOCK_Q, head_dim]  
- K blocks: [BLOCK_K, head_dim]
- V blocks: [BLOCK_K, head_dim]
- Attention blocks: [BLOCK_Q, BLOCK_K]

```
For each Q block:
    For each K block:
        1. Load Q_block and K_block into shared memory
        2. Compute S_block = Q_block @ K_block^T  
        3. Update online softmax statistics
        4. Load V_block and accumulate output
```

### Block Size Selection

Block sizes are chosen based on:
- **Shared Memory Constraints**: Must fit Q_block + K_block + S_block in shared memory
- **Occupancy**: Optimize for GPU occupancy and memory bandwidth
- **Sequence Length**: Larger blocks for longer sequences

Heuristics:
```cpp
block_size = seq_len <= 512 ? 64 : 
             seq_len <= 1024 ? 128 : 256
```

### Memory Access Patterns

#### Coalesced Access
Threads access consecutive memory addresses for optimal bandwidth:
```cpp
// Coalesced loading
for (int i = tid; i < block_size * head_dim; i += blockDim.x) {
    int row = i / head_dim;
    int col = i % head_dim;
    shared_Q[row][col] = Q[base_offset + row * head_dim + col];
}
```

#### Shared Memory Layout
```
shared_memory:
├── Q_block     [BLOCK_Q × head_dim]
├── K_block     [BLOCK_K × head_dim]  
├── S_block     [BLOCK_Q × BLOCK_K]
└── reduction   [BLOCK_SIZE × 1]
```

## Online Softmax Implementation

### Problem with Naive Softmax

Standard softmax requires two passes:
1. Find maximum for numerical stability
2. Compute exponentials and normalize

This requires storing intermediate results or multiple passes over data.

### Online Softmax Algorithm

We use the online softmax algorithm that maintains running statistics:

```
For each new block of scores S_new:
    m_new = max(m_old, max(S_new))
    l_new = l_old * exp(m_old - m_new) + sum(exp(S_new - m_new))
    o_new = o_old * exp(m_old - m_new) + sum(exp(S_new - m_new) * V_new)
```

Where:
- m: Running maximum
- l: Running sum (denominator)  
- o: Running output

### Numerical Stability

The algorithm maintains numerical stability by:
1. **Scaling**: Always subtract maximum before exponential
2. **Incremental Updates**: Update statistics incrementally
3. **Rescaling**: Rescale previous outputs when maximum changes

### Implementation Details

```cpp
// Update maximum
float new_max = fmaxf(row_max, block_max);
float max_diff = row_max - new_max;

// Scale previous sum and add new contribution
row_sum = row_sum * expf(max_diff) + block_sum;

// Scale previous output  
for (int i = 0; i < head_dim; i++) {
    output_buffer[i] *= expf(max_diff);
}
```

## Memory Management

### Memory Hierarchy Utilization

```
GPU Memory Hierarchy:
├── Global Memory (HBM)     ~1TB/s,  high latency
├── L2 Cache               ~7TB/s,   medium latency  
├── L1 Cache/Texture       ~20TB/s,  low latency
└── Shared Memory (SRAM)   ~20TB/s,  lowest latency
```

### Data Movement Strategy

1. **Minimize HBM Access**: Keep intermediate results in shared memory
2. **Maximize Cache Hits**: Reuse data across thread blocks
3. **Coalesced Transfers**: Ensure aligned, consecutive memory access
4. **Prefetching**: Overlap computation with memory transfers

### Shared Memory Usage

Per thread block shared memory allocation:
```cpp
extern __shared__ float shared_mem[];
float* shared_Q = shared_mem;                           // Q block
float* shared_K = &shared_mem[BLOCK_Q * head_dim];      // K block
float* shared_S = &shared_K[BLOCK_K * head_dim];        // Scores  
float* reduction = &shared_S[BLOCK_Q * BLOCK_K];        // Temp buffer
```

Total: `(BLOCK_Q + BLOCK_K) * head_dim + BLOCK_Q * BLOCK_K + padding`

## CUDA Kernel Design

### Grid and Block Layout

```cpp
// Grid layout
dim3 grid(batch_size, n_heads, num_q_blocks);

// Block layout  
int threads_per_block = min(1024, max(256, BLOCK_Q * BLOCK_K / 4));
```

**Rationale**:
- Each thread block processes one Q block for one head in one batch
- Parallelization across batch items, heads, and Q blocks
- Thread count optimized for occupancy

### Thread Work Distribution

Each thread processes multiple elements to:
1. **Maximize Occupancy**: Use all available threads
2. **Minimize Divergence**: Uniform work distribution  
3. **Optimize Memory Access**: Coalesced patterns

```cpp
// Each thread loads multiple elements
for (int i = tid; i < q_size * head_dim; i += blockDim.x) {
    // Compute row and column indices
    int q_idx = i / head_dim;
    int dim_idx = i % head_dim;
    shared_Q[q_idx * head_dim + dim_idx] = Q[...];
}
```

### Synchronization Strategy

Minimal synchronization points:
1. After loading Q/K blocks: `__syncthreads()`
2. After computing attention scores: `__syncthreads()`  
3. After block reductions: `__syncthreads()`

### Warp-Level Optimizations

Use warp primitives for efficient reductions:
```cpp
__device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}
```

Benefits:
- No shared memory usage for reductions
- Faster than shared memory approach
- Hardware-optimized shuffle operations

## PyTorch Integration

### C++ Extension Architecture

```
Python Layer (attention.py)
    ↓ 
C++ Binding (attention_wrapper.cpp)
    ↓
CUDA Kernel (attention_kernel.cu)
```

### Tensor Validation

Comprehensive input validation:
```cpp
void validate_flash_attention_inputs(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
) {
    CHECK_CUDA(Q); CHECK_CUDA(K); CHECK_CUDA(V);
    CHECK_CONTIGUOUS(Q); CHECK_CONTIGUOUS(K); CHECK_CONTIGUOUS(V);
    CHECK_DTYPE(Q, torch::kFloat32);
    CHECK_4D(Q); CHECK_4D(K); CHECK_4D(V);
    CHECK_COMPATIBLE_SHAPES(Q, K, V);
}
```

### Memory Management

Automatic memory management using PyTorch's allocator:
```cpp
auto output = torch::zeros({batch, heads, seq, dim}, 
                          torch::TensorOptions()
                              .dtype(torch::kFloat32)
                              .device(Q.device())
                              .memory_format(torch::MemoryFormat::Contiguous));
```

### Autograd Integration

Custom autograd function for backward pass:
```python
class FlashAttentionFunction(Function):
    @staticmethod
    def forward(ctx, Q, K, V, scale):
        output, l_buffer, m_buffer = flashattention_mini_cuda.forward(Q, K, V, scale)
        ctx.save_for_backward(Q, K, V, l_buffer, m_buffer)
        return output
    
    @staticmethod  
    def backward(ctx, grad_output):
        # Simplified backward - would recompute attention in blocks
        return compute_gradients(ctx.saved_tensors, grad_output)
```

## Performance Optimizations

### Compiler Optimizations

NVCC flags for optimal performance:
```bash
-O3                          # Maximum optimization
--use_fast_math              # Fast math operations  
--restrict                   # Restrict keyword optimization
--extra-device-vectorization # Enable vectorization
-arch=sm_80                  # Target architecture
```

### Template Specialization

Compile-time optimization using templates:
```cpp
template<int BLOCK_Q, int BLOCK_K, int HEAD_DIM>
__global__ void flash_attention_kernel(...) {
    // Compile-time constants enable optimizations
    // Loop unrolling, constant propagation, etc.
}
```

### Memory Access Optimization

1. **Vectorized Loads**: Use float4 for 4x memory bandwidth
2. **Bank Conflict Avoidance**: Pad shared memory arrays  
3. **Cache Optimization**: Maximize L2 cache hit rate
4. **Prefetching**: Overlap memory and compute

### Occupancy Optimization

Balance between:
- **Thread Count**: More threads vs register pressure
- **Shared Memory**: Larger blocks vs occupancy  
- **Register Usage**: Complex computation vs spilling

Target: 75%+ theoretical occupancy for optimal performance.

## Limitations and Future Work

### Current Limitations

1. **Simplified Backward Pass**: Uses standard attention backward
2. **Limited Head Dimensions**: Only supports 64 and 128
3. **No Attention Masks**: Causal/padding masks not implemented
4. **Single Precision Only**: No FP16/BF16 support
5. **Memory Constraints**: Large sequences may exceed shared memory

### Potential Improvements

#### 1. True Recomputation Backward
Implement block-wise recomputation in backward pass:
```cpp
// Recompute attention scores in backward pass
// without storing attention matrix from forward
```

#### 2. Mixed Precision Support
Add FP16 support for 2x memory and compute efficiency:
```cpp
template<typename T>
__global__ void flash_attention_kernel_fp16(...)
```

#### 3. Advanced Masking
Support for causal and padding masks:
```cpp
// Apply mask during softmax computation
if (mask[i][j]) {
    scores[i][j] = -INFINITY;
}
```

#### 4. Multi-GPU Support
Scale across multiple GPUs for very long sequences:
```python
# Partition sequence across GPUs
# Communicate attention statistics between GPUs
```

#### 5. Adaptive Block Sizing
Dynamically choose block sizes based on:
- Available shared memory
- Sequence length  
- GPU architecture
- Register pressure

#### 6. Kernel Fusion
Fuse attention with other operations:
```cpp
// Fused attention + dropout + residual + layer norm
__global__ void fused_attention_dropout_layernorm(...)
```

### Research Directions

1. **Sparse Attention Patterns**: Block-sparse attention support
2. **Long Sequence Optimization**: Ring attention, sequence parallelism
3. **Hardware-Specific Tuning**: Tensor core utilization
4. **Memory Compression**: Attention score quantization
5. **Dynamic Attention**: Content-based block selection

## Conclusion

FlashAttention-Mini demonstrates the core principles of memory-efficient attention:
- **Tiling** reduces memory complexity from O(n²) to O(n)
- **Online softmax** enables single-pass computation
- **Memory hierarchy optimization** maximizes hardware utilization
- **Careful CUDA programming** achieves competitive performance

While simplified, this implementation provides a solid foundation for understanding and extending memory-efficient attention mechanisms. The modular design allows easy experimentation with different tiling strategies, block sizes, and optimization techniques.

The project serves as both an educational tool and a starting point for more advanced attention implementations, bridging the gap between theoretical understanding and practical GPU programming.