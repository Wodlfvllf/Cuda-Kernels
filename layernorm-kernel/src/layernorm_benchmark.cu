#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include "common.cuh"
#include "naive_layernorm.cuh"
#include "fused_layernorm.cuh"
#include "vectorized_layernorm.cuh"
#include "welford_layernorm.cuh"

class LayerNormBenchmark {
private:
    int N, D;
    float eps;
    
    float *d_input, *d_output;
    float *d_gamma, *d_beta;
    float *d_mean, *d_variance;
    
    std::vector<float> h_input, h_output;
    std::vector<float> h_gamma, h_beta;
    
public:
    LayerNormBenchmark(int batch_size, int hidden_dim, float epsilon = 1e-5) 
        : N(batch_size), D(hidden_dim), eps(epsilon) {
        
        // Allocate host memory
        h_input.resize(N * D);
        h_output.resize(N * D);
        h_gamma.resize(D);
        h_beta.resize(D);
        
        // Initialize with random values
        srand(42);  // Fixed seed for reproducibility
        for (int i = 0; i < N * D; i++) {
            h_input[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
        }
        for (int i = 0; i < D; i++) {
            h_gamma[i] = 1.0f;
            h_beta[i] = 0.0f;
        }
        
        // Allocate device memory
        CUDA_CHECK(cudaMalloc(&d_input, N * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_output, N * D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_gamma, D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_beta, D * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_mean, N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_variance, N * sizeof(float)));
        
        // Copy to device
        CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), N * D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_gamma, h_gamma.data(), D * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_beta, h_beta.data(), D * sizeof(float), cudaMemcpyHostToDevice));
    }
    
    ~LayerNormBenchmark() {
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_gamma);
        cudaFree(d_beta);
        cudaFree(d_mean);
        cudaFree(d_variance);
    }
    
    float benchmark_naive() {
        const int num_runs = 100;
        
        // Warmup
        for (int i = 0; i < 10; i++) {
            layernorm_naive(d_input, d_gamma, d_beta, d_output, d_mean, d_variance, N, D, eps);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_runs; i++) {
            layernorm_naive(d_input, d_gamma, d_beta, d_output, d_mean, d_variance, N, D, eps);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        float ms = std::chrono::duration<float, std::milli>(end - start).count() / num_runs;
        return ms;
    }
    
    float benchmark_fused() {
        const int num_runs = 100;
        
        // Warmup
        for (int i = 0; i < 10; i++) {
            layernorm_fused(d_input, d_gamma, d_beta, d_output, N, D, eps);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_runs; i++) {
            layernorm_fused(d_input, d_gamma, d_beta, d_output, N, D, eps);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        float ms = std::chrono::duration<float, std::milli>(end - start).count() / num_runs;
        return ms;
    }
    
    float benchmark_vectorized() {
        const int num_runs = 100;
        
        // Warmup
        for (int i = 0; i < 10; i++) {
            layernorm_vectorized(d_input, d_gamma, d_beta, d_output, N, D, eps);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_runs; i++) {
            layernorm_vectorized(d_input, d_gamma, d_beta, d_output, N, D, eps);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        float ms = std::chrono::duration<float, std::milli>(end - start).count() / num_runs;
        return ms;
    }
    
    float benchmark_welford() {
        const int num_runs = 100;
        
        // Warmup
        for (int i = 0; i < 10; i++) {
            layernorm_welford(d_input, d_gamma, d_beta, d_output, N, D, eps);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_runs; i++) {
            layernorm_welford(d_input, d_gamma, d_beta, d_output, N, D, eps);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        auto end = std::chrono::high_resolution_clock::now();
        
        float ms = std::chrono::duration<float, std::milli>(end - start).count() / num_runs;
        return ms;
    }
    
    void verify_correctness() {
        // Run fused kernel
        layernorm_fused(d_input, d_gamma, d_beta, d_output, N, D, eps);
        CUDA_CHECK(cudaDeviceSynchronize());
        
        // Copy result to host
        std::vector<float> gpu_output(N * D);
        CUDA_CHECK(cudaMemcpy(gpu_output.data(), d_output, N * D * sizeof(float), cudaMemcpyDeviceToHost));
        
        // CPU reference implementation
        std::vector<float> cpu_output(N * D);
        for (int n = 0; n < N; n++) {
            // Compute mean
            float mean = 0.0f;
            for (int d = 0; d < D; d++) {
                mean += h_input[n * D + d];
            }
            mean /= D;
            
            // Compute variance
            float variance = 0.0f;
            for (int d = 0; d < D; d++) {
                float diff = h_input[n * D + d] - mean;
                variance += diff * diff;
            }
            variance /= D;
            
            // Normalize
            float inv_std = 1.0f / sqrtf(variance + eps);
            for (int d = 0; d < D; d++) {
                float normalized = (h_input[n * D + d] - mean) * inv_std;
                cpu_output[n * D + d] = h_gamma[d] * normalized + h_beta[d];
            }
        }
        
        // Compare results
        float max_error = 0.0f;
        float avg_error = 0.0f;
        for (int i = 0; i < N * D; i++) {
            float error = std::abs(gpu_output[i] - cpu_output[i]);
            max_error = std::max(max_error, error);
            avg_error += error;
        }
        avg_error /= (N * D);
        
        std::cout << "Correctness verification - Max error: " << std::scientific << max_error;
        std::cout << ", Avg error: " << avg_error;
        if (max_error < 1e-5) {
            std::cout << " [PASSED]" << std::endl;
        } else {
            std::cout << " [FAILED]" << std::endl;
        }
    }
    
    void run_benchmarks() {
        std::cout << "\nBatch Size: " << N << ", Hidden Dim: " << D << std::endl;
        std::cout << "----------------------------------------" << std::endl;
        
        // Verify correctness first
        verify_correctness();
        
        // Run benchmarks
        float naive_time = benchmark_naive();
        float fused_time = benchmark_fused();
        float vectorized_time = benchmark_vectorized();
        float welford_time = benchmark_welford();
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Naive (3 kernels):    " << naive_time << " ms" << std::endl;
        std::cout << "Fused:                " << fused_time << " ms";
        std::cout << " (Speedup: " << naive_time / fused_time << "x)" << std::endl;
        std::cout << "Vectorized (float4):  " << vectorized_time << " ms";
        std::cout << " (Speedup: " << naive_time / vectorized_time << "x)" << std::endl;
        std::cout << "Welford (stable):     " << welford_time << " ms";
        std::cout << " (Speedup: " << naive_time / welford_time << "x)" << std::endl;
        
        // Calculate memory bandwidth utilization
        float bytes_read = N * D * sizeof(float) + 2 * D * sizeof(float);  // input + gamma + beta
        float bytes_written = N * D * sizeof(float);
        float total_gb = (bytes_read + bytes_written) / 1e9;
        
        float bandwidth_fused = total_gb / (fused_time / 1000.0);
        
        // Get theoretical peak bandwidth
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        float theoretical_bw = prop.memoryBusWidth / 8.0 * prop.memoryClockRate * 2.0 / 1e6;
        
        std::cout << "\nFused kernel bandwidth: " << bandwidth_fused << " GB/s" << std::endl;
        std::cout << "Theoretical peak:       " << theoretical_bw << " GB/s" << std::endl;
        std::cout << "Efficiency:             " << (bandwidth_fused / theoretical_bw * 100) << "%" << std::endl;
    }
};