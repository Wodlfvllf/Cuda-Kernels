import torch
import dropout_cuda

def lcg_rand(state):
    a = 1103515245
    c = 12345
    state = (a * state + c) & 0xFFFFFFFF
    return state >> 16

def dropout_ref(x, p, seed):
    output = torch.zeros_like(x)
    scale = 1.0 / (1.0 - p)
    for i in range(x.numel()):
        rand_state = seed + i
        rand_val = float(lcg_rand(rand_state)) / float(0xFFFF)
        if rand_val < p:
            output.view(-1)[i] = 0
        else:
            output.view(-1)[i] = x.view(-1)[i] * scale
    return output

def test_dropout_correctness():
    # Test dimensions
    num_elements = 1024
    p = 0.5
    seed = 12345
    
    # Create input
    x = torch.randn(num_elements, device='cuda')
    
    # Compute with custom kernel
    custom_output = dropout_cuda.dropout_forward(x, p, seed)
    
    # Compute with reference
    ref_output = dropout_ref(x, p, seed)
    
    # Check correctness
    assert torch.allclose(custom_output, ref_output, rtol=1e-4, atol=1e-5)
    print("\u2713 Dropout correctness test passed")

if __name__ == "__main__":
    test_dropout_correctness()
