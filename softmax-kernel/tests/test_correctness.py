import torch
import softmax_cuda

class NaiveSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = softmax_cuda.naive_softmax_forward(input)
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_input = output * (grad_output - (output * grad_output).sum(dim=-1, keepdim=True))
        return grad_input

def naive_softmax(input):
    return NaiveSoftmax.apply(input)

class OptimizedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        output = softmax_cuda.optimized_softmax_forward(input)
        ctx.save_for_backward(output)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_input = output * (grad_output - (output * grad_output).sum(dim=-1, keepdim=True))
        return grad_input

def optimized_softmax(input):
    return OptimizedSoftmax.apply(input)

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
    print("\u2713 Naive softmax correctness test passed")

def test_softmax_gradient():
    x = torch.randn(2, 64, 256, device='cuda', requires_grad=True)
    
    # Forward
    y = naive_softmax(x)
    loss = y.sum()
    
    # Backward
    loss.backward()
    
    # Check gradient exists
    assert x.grad is not None
    print("\u2713 Naive softmax gradient test passed")

def test_optimized_softmax_correctness():
    # Test dimensions
    batch_size = 4
    seq_len = 128
    hidden_dim = 768
    
    # Create input
    x = torch.randn(batch_size, seq_len, hidden_dim, device='cuda')
    
    # Compute with custom kernel
    custom_output = optimized_softmax(x)
    
    # Compute with PyTorch
    torch_output = torch.softmax(x, dim=-1)
    
    # Check correctness
    assert torch.allclose(custom_output, torch_output, rtol=1e-4, atol=1e-5)
    print("\u2713 Optimized softmax correctness test passed")

def test_optimized_softmax_gradient():
    x = torch.randn(2, 64, 256, device='cuda', requires_grad=True)
    
    # Forward
    y = optimized_softmax(x)
    loss = y.sum()
    
    # Backward
    loss.backward()
    
    # Check gradient exists
    assert x.grad is not None
    print("\u2713 Optimized softmax gradient test passed")

if __name__ == "__main__":
    test_softmax_correctness()
    test_softmax_gradient()
    test_optimized_softmax_correctness()
    test_optimized_softmax_gradient()
