
```markdown
# Custom Fused LayerNorm CUDA Kernels

High-performance custom CUDA implementations of LayerNorm with multiple optimization strategies.

## Features

- **Multiple Implementations**:
  - Naive (3 separate kernels)
  - Fused single-pass kernel
  - Vectorized with float4 loads/stores
  - Welford's algorithm for numerical stability

- **Optimizations**:
  - Warp-level reductions
  - Shared memory caching
  - Coalesced memory access
  - Grid-stride loops for large dimensions

- **PyTorch Integration**:
  - Custom C++ extension
  - Python bindings
  - Gradient support (basic)

## Building

### Standalone CUDA Application
```bash
nvcc -O3 -arch=sm_80 -I include main.cu src/*.cu -o layernorm_bench
./layernorm_bench
```

### PyTorch Extension
```bash
cd python
pip install -e .
python test_layernorm.py
```

## Performance

On NVIDIA A100 (example results):
- Fused kernel: ~2-3x speedup over naive
- Vectorized: ~2.5-3.5x speedup over naive
- Welford: ~2-2.5x speedup with better numerical stability

## Architecture Support

- Requires CUDA compute capability 7.0+ (Volta or newer)
- Optimized for Ampere (sm_80) and Hopper (sm_90)

## Directory Structure

```
├── include/          # Header files
│   ├── common.cuh    # Common utilities
│   ├── naive_layernorm.cuh
│   ├── fused_layernorm.cuh
│   ├── vectorized_layernorm.cuh
│   └── welford_layernorm.cuh
├── src/             # Implementation files
│   ├── naive_layernorm.cu
│   ├── fused_layernorm.cu
│   ├── vectorized_layernorm.cu
│   ├── welford_layernorm.cu
│   ├── layernorm_benchmark.cu
│   └── layernorm_ops.cpp
├── python/          # Python bindings
│   ├── setup.py
│   └── test_layernorm.py
├── main.cu          # Main benchmark driver
└── README.md

```

## TODO

- [ ] Add proper backward pass implementation
- [ ] Support FP16/BF16 mixed precision
- [ ] Implement online softmax variant
- [ ] Add RMSNorm variant
- [ ] Support grouped/multi-head normalization
- [ ] Add CUTLASS integration example

## References

- [Layer Normalization Paper](https://arxiv.org/abs/1607.06450)
- [Welford's Online Algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm)
- [NVIDIA Apex LayerNorm](https://github.com/NVIDIA/apex)

## License

MIT