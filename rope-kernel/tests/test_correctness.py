import torch
import rope_cuda

def rope_ref(x, cos_sin_table):
    output = torch.zeros_like(x)
    for b in range(x.shape[0]):
        for s in range(x.shape[1]):
            for i in range(0, x.shape[2], 2):
                x1 = x[b, s, i]
                x2 = x[b, s, i+1]
                cos = cos_sin_table[s, i]
                sin = cos_sin_table[s, i+1]
                output[b, s, i] = x1 * cos - x2 * sin
                output[b, s, i+1] = x1 * sin + x2 * cos
    return output

def test_rope_correctness():
    # Test dimensions
    batch_size = 4
    seq_len = 128
    hidden_dim = 768
    
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda')
    cos_sin_table = torch.randn(seq_len, hidden_dim, device='cuda')
    
    # Compute with custom kernel
    custom_output = rope_cuda.rope_forward(x, cos_sin_table)
    
    # Compute with reference
    ref_output = rope_ref(x, cos_sin_table)
    
    # Check correctness
    assert torch.allclose(custom_output, ref_output, rtol=1e-4, atol=1e-5)
    print("\u2713 RoPE correctness test passed")

if __name__ == "__main__":
    test_rope_correctness()
