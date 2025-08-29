/*
FlashAttention-Mini CUDA Kernel Implementation

This implements a simplified version of FlashAttention using tiled computation:
1. Load Q/K blocks into shared memory
2. Compute QK^T for the block  
3. Apply online softmax with scaling
4. Multiply by V block and accumulate results

Key optimizations:
- Shared memory tiling for Q/K matrices
- Online softmax computation for numerical stability
- Coalesced memory access patterns
- Warp-level reductions for efficiency
*/

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
#include <math.h>

// Constants and macros
#define WARP_SIZE 32
#define MAX_BLOCK_SIZE 1024
#define FLOAT_MIN -1e30f

// Helper macros for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(error)); \
            exit(1); \
        } \
    } while(0)

// Warp-level reduction primitives
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// Block-level reduction using shared memory
template<typename T, int BLOCK_SIZE>
__device__ void block_reduce_sum(T* shared_mem, T val, int tid) {
    shared_mem[tid] = val;
    __syncthreads();
    
    // Reduce within block
    for (int stride = BLOCK_SIZE / 2; stride > WARP_SIZE; stride /= 2) {
        if (tid < stride) {
            shared_mem[tid] += shared_mem[tid + stride];
        }
        __syncthreads();
    }
    
    // Final warp reduction
    if (tid < WARP_SIZE) {
        T warp_val = (tid < BLOCK_SIZE / 2) ? shared_mem[tid] : T(0);
        warp_val = warp_reduce_sum(warp_val);
        if (tid == 0) {
            shared_mem[0] = warp_val;
        }
    }
    __syncthreads();
}

template<typename T, int BLOCK_SIZE>
__device__ void block_reduce_max(T* shared_mem, T val, int tid) {
    shared_mem[tid] = val;
    __syncthreads();
    
    for (int stride = BLOCK_SIZE / 2; stride > WARP_SIZE; stride /= 2) {
        if (tid < stride) {
            shared_mem[tid] = fmaxf(shared_mem[tid], shared_mem[tid + stride]);
        }
        __syncthreads();
    }
    
    if (tid < WARP_SIZE) {
        T warp_val = (tid < BLOCK_SIZE / 2) ? shared_mem[tid] : T(FLOAT_MIN);
        warp_val = warp_reduce_max(warp_val);
        if (tid == 0) {
            shared_mem[0] = warp_val;
        }
    }
    __syncthreads();
}

/*
Main FlashAttention kernel with tiled computation

Template parameters:
- BLOCK_Q: Block size for Q dimension (queries)  
- BLOCK_K: Block size for K dimension (keys)
- HEAD_DIM: Head dimension size

Grid/Block layout:
- blockIdx.x: Batch index
- blockIdx.y: Head index  
- blockIdx.z: Q block index
- Threads process elements within blocks

Memory layout:
- Q: [batch, n_heads, seq_len, head_dim]
- K: [batch, n_heads, seq_len, head_dim] 
- V: [batch, n_heads, seq_len, head_dim]
- Output: [batch, n_heads, seq_len, head_dim]
*/

template<int BLOCK_Q, int BLOCK_K, int HEAD_DIM>
__global__ void flash_attention_kernel(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ output,
    float* __restrict__ l_buffer,  // Softmax denominator buffer
    float* __restrict__ m_buffer,  // Row-wise max buffer
    const int batch_size,
    const int n_heads,
    const int seq_len,
    const int head_dim,
    const float scale
) {
    // Shared memory for Q and K blocks
    extern __shared__ float shared_mem[];
    float* shared_Q = shared_mem;                           // [BLOCK_Q, HEAD_DIM]
    float* shared_K = &shared_mem[BLOCK_Q * HEAD_DIM];      // [BLOCK_K, HEAD_DIM]  
    float* shared_S = &shared_mem[BLOCK_Q * HEAD_DIM + BLOCK_K * HEAD_DIM]; // [BLOCK_Q, BLOCK_K]
    float* reduction_buffer = &shared_S[BLOCK_Q * BLOCK_K]; // For reductions
    
    // Thread and block indices
    const int batch_idx = blockIdx.x;
    const int head_idx = blockIdx.y;
    const int q_block_idx = blockIdx.z;
    const int tid = threadIdx.x;
    
    // Calculate tensor strides
    const int batch_stride = n_heads * seq_len * head_dim;
    const int head_stride = seq_len * head_dim;
    const int seq_stride = head_dim;
    
    // Base pointers for current batch and head
    const float* Q_base = Q + batch_idx * batch_stride + head_idx * head_stride;
    const float* K_base = K + batch_idx * batch_stride + head_idx * head_stride;  
    const float* V_base = V + batch_idx * batch_stride + head_idx * head_stride;
    float* output_base = output + batch_idx * batch_stride + head_idx * head_stride;
    
    // Output buffer and online softmax states
    float output_buffer[HEAD_DIM] = {0.0f};
    float row_max = FLOAT_MIN;
    float row_sum = 0.0f;
    
    // Q block boundaries
    const int q_start = q_block_idx * BLOCK_Q;
    const int q_end = min(q_start + BLOCK_Q, seq_len);
    const int q_size = q_end - q_start;
    
    // Load Q block into shared memory (coalesced)
    for (int i = tid; i < q_size * HEAD_DIM; i += blockDim.x) {
        const int q_idx = i / HEAD_DIM;
        const int dim_idx = i % HEAD_DIM;
        if (q_start + q_idx < seq_len && dim_idx < head_dim) {
            shared_Q[q_idx * HEAD_DIM + dim_idx] = 
                Q_base[(q_start + q_idx) * seq_stride + dim_idx];
        } else {
            shared_Q[q_idx * HEAD_DIM + dim_idx] = 0.0f;
        }
    }
    __syncthreads();
    
    // Process K blocks sequentially
    for (int k_block_start = 0; k_block_start < seq_len; k_block_start += BLOCK_K) {
        const int k_end = min(k_block_start + BLOCK_K, seq_len);
        const int k_size = k_end - k_block_start;
        
        // Load K block into shared memory (coalesced)
        for (int i = tid; i < k_size * HEAD_DIM; i += blockDim.x) {
            const int k_idx = i / HEAD_DIM;
            const int dim_idx = i % HEAD_DIM;
            if (k_block_start + k_idx < seq_len && dim_idx < head_dim) {
                shared_K[k_idx * HEAD_DIM + dim_idx] = 
                    K_base[(k_block_start + k_idx) * seq_stride + dim_idx];
            } else {
                shared_K[k_idx * HEAD_DIM + dim_idx] = 0.0f;
            }
        }
        __syncthreads();
        
        // Compute QK^T for this block (each thread computes multiple elements)
        for (int qk_idx = tid; qk_idx < q_size * k_size; qk_idx += blockDim.x) {
            const int q_idx = qk_idx / k_size;
            const int k_idx = qk_idx % k_size;
            
            if (q_idx < q_size && k_idx < k_size) {
                float dot_product = 0.0f;
                
                // Compute dot product Q[q_idx] · K[k_idx]
                for (int d = 0; d < head_dim; d++) {
                    dot_product += shared_Q[q_idx * HEAD_DIM + d] * 
                                   shared_K[k_idx * HEAD_DIM + d];
                }
                
                shared_S[q_idx * BLOCK_K + k_idx] = dot_product * scale;
            }
        }
        __syncthreads();
        
        // Online softmax computation for each query
        for (int q_local = 0; q_local < q_size; q_local++) {
            // Find max in this row (block-wise reduction)
            float local_max = FLOAT_MIN;
            for (int k_local = tid; k_local < k_size; k_local += blockDim.x) {
                if (k_block_start + k_local < seq_len) {
                    local_max = fmaxf(local_max, shared_S[q_local * BLOCK_K + k_local]);
                }
            }
            
            block_reduce_max<float, MAX_BLOCK_SIZE>(reduction_buffer, local_max, tid);
            float block_max = reduction_buffer[0];
            
            // Update global max for this row
            float new_max = fmaxf(row_max, block_max);
            float max_diff = row_max - new_max;
            
            // Compute exp(S - new_max) and sum
            float local_sum = 0.0f;
            for (int k_local = tid; k_local < k_size; k_local += blockDim.x) {
                if (k_block_start + k_local < seq_len) {
                    float exp_val = expf(shared_S[q_local * BLOCK_K + k_local] - new_max);
                    shared_S[q_local * BLOCK_K + k_local] = exp_val;
                    local_sum += exp_val;
                }
            }
            
            block_reduce_sum<float, MAX_BLOCK_SIZE>(reduction_buffer, local_sum, tid);
            float block_sum = reduction_buffer[0];
            
            // Update running sum with scaling
            row_sum = row_sum * expf(max_diff) + block_sum;
            row_max = new_max;
            
            // Scale previous output buffer  
            if (k_block_start > 0 && tid < head_dim) {
                output_buffer[tid] *= expf(max_diff);
            }
        }
        __syncthreads();
        
        // Load V block and accumulate output
        for (int v_idx = tid; v_idx < k_size * HEAD_DIM; v_idx += blockDim.x) {
            const int k_idx = v_idx / HEAD_DIM;  
            const int dim_idx = v_idx % HEAD_DIM;
            if (k_block_start + k_idx < seq_len && dim_idx < head_dim) {
                float v_val = V_base[(k_block_start + k_idx) * seq_stride + dim_idx];
                
                // Accumulate weighted V values
                for (int q_local = 0; q_local < q_size; q_local++) {
                    if (tid == 0) {  // Single thread per query updates output
                        float attention_weight = shared_S[q_local * BLOCK_K + k_idx];
                        output_buffer[dim_idx] += attention_weight * v_val;
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // Final normalization and write output
    for (int q_local = 0; q_local < q_size; q_local++) {
        if (tid < head_dim) {
            float normalized_output = output_buffer[tid] / row_sum;
            output_base[(q_start + q_local) * seq_stride + tid] = normalized_output;
        }
    }
    
    // Store softmax normalization factors for backward pass
    if (tid == 0) {
        l_buffer[batch_idx * n_heads * seq_len + head_idx * seq_len + q_start] = row_sum;
        m_buffer[batch_idx * n_heads * seq_len + head_idx * seq_len + q_start] = row_max;
    }
}

/*
Launcher function for the FlashAttention kernel

Automatically selects optimal block sizes based on problem dimensions
and GPU capabilities.
*/
extern "C" {

void launch_flash_attention_kernel(
    const float* Q,
    const float* K, 
    const float* V,
    float* output,
    float* l_buffer,
    float* m_buffer,
    const int batch_size,
    const int n_heads,
    const int seq_len,
    const int head_dim,
    const float scale,
    cudaStream_t stream
) {
    // Determine optimal block sizes based on problem size and GPU capabilities
    const int BLOCK_Q = (seq_len <= 512) ? 64 : ((seq_len <= 1024) ? 128 : 256);
    const int BLOCK_K = BLOCK_Q;  // Square blocks for simplicity
    const int HEAD_DIM = head_dim;
    
    // Calculate grid dimensions
    const int num_q_blocks = (seq_len + BLOCK_Q - 1) / BLOCK_Q;
    dim3 grid(batch_size, n_heads, num_q_blocks);
    
    // Calculate block dimensions and shared memory
    const int threads_per_block = min(1024, max(256, BLOCK_Q * BLOCK_K / 4));
    const int shared_mem_size = (BLOCK_Q * HEAD_DIM + BLOCK_K * HEAD_DIM + 
                                BLOCK_Q * BLOCK_K + 1024) * sizeof(float);
    
    // Launch kernel with appropriate template instantiation
    if (head_dim == 64) {
        if (BLOCK_Q == 64) {
            flash_attention_kernel<64, 64, 64><<<grid, threads_per_block, shared_mem_size, stream>>>(
                Q, K, V, output, l_buffer, m_buffer,
                batch_size, n_heads, seq_len, head_dim, scale
            );
        } else if (BLOCK_Q == 128) {
            flash_attention_kernel<128, 128, 64><<<grid, threads_per_block, shared_mem_size, stream>>>(
                Q, K, V, output, l_buffer, m_buffer, 
                batch_size, n_heads, seq_len, head_dim, scale
            );
        } else {
            flash_attention_kernel<256, 256, 64><<<grid, threads_per_block, shared_mem_size, stream>>>(
                Q, K, V, output, l_buffer, m_buffer,
                batch_size, n_heads, seq_len, head_dim, scale
            );
        }
    } else if (head_dim == 128) {
        if (BLOCK_Q == 64) {
            flash_attention_kernel<64, 64, 128><<<grid, threads_per_block, shared_mem_size, stream>>>(
                Q, K, V, output, l_buffer, m_buffer,
                batch_size, n_heads, seq_len, head_dim, scale
            );
        } else {
            flash_attention_kernel<128, 128, 128><<<grid, threads_per_block, shared_mem_size, stream>>>(
                Q, K, V, output, l_buffer, m_buffer,
                batch_size, n_heads, seq_len, head_dim, scale
            );
        }
    } else {
        // Fallback for other head dimensions
        flash_attention_kernel<64, 64, 128><<<grid, threads_per_block, shared_mem_size, stream>>>(
            Q, K, V, output, l_buffer, m_buffer,
            batch_size, n_heads, seq_len, head_dim, scale
        );
    }
    
    CUDA_CHECK(cudaGetLastError());
}

} // extern "C"