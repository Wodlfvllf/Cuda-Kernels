import torch
import grad_clip_cuda

def clip_grad_norm_ref(grad, max_norm):
    total_norm = torch.norm(grad)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        grad.mul_(clip_coef)
    return grad

def test_grad_clip_correctness():
    # Test dimensions
    num_elements = 1024
    max_norm = 1.0
    
    # Create input
    grad = torch.randn(num_elements, device='cuda') * 10
    
    # Compute with custom kernel
    grad_clone = grad.clone()
    
    # 1. Compute norm
    sum_sq_output = torch.zeros(256, device='cuda') # as defined in the kernel
    grad_clip_cuda.sum_squares_forward(grad_clone, sum_sq_output)
    total_norm = torch.sqrt(torch.sum(sum_sq_output))
    
    # 2. Scale if necessary
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + 1e-6)
        grad_clip_cuda.scale_forward(grad_clone, clip_coef)
    
    # Compute with reference
    ref_grad = grad.clone()
    ref_grad = clip_grad_norm_ref(ref_grad, max_norm)
    
    # Check correctness
    assert torch.allclose(grad_clone, ref_grad, rtol=1e-4, atol=1e-5)
    print("\u2713 Gradient Clipping correctness test passed")

if __name__ == "__main__":
    test_grad_clip_correctness()
