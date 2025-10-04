import torch
import adamw_cuda

def test_adamw_correctness():
    # Test dimensions
    num_elements = 1024
    lr = 0.001
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8
    weight_decay = 0.01
    
    # Create input
    params = torch.randn(num_elements, device='cuda')
    grads = torch.randn(num_elements, device='cuda')
    exp_avg = torch.zeros_like(params)
    exp_avg_sq = torch.zeros_like(params)
    
    # Copy for reference
    params_ref = params.clone()
    grads_ref = grads.clone()
    exp_avg_ref = exp_avg.clone()
    exp_avg_sq_ref = exp_avg_sq.clone()

    # Custom kernel
    for step in range(1, 10):
        adamw_cuda.adamw_forward(params, grads, exp_avg, exp_avg_sq, beta1, beta2, lr, weight_decay, eps, step)

    # Reference
    optimizer = torch.optim.AdamW([params_ref], lr=lr, betas=(beta1, beta2), eps=eps, weight_decay=weight_decay)
    for step in range(1, 10):
        params_ref.grad = grads_ref
        optimizer.step()

    # Check correctness
    assert torch.allclose(params, params_ref, rtol=1e-4, atol=1e-5)
    assert torch.allclose(exp_avg, optimizer.state[params_ref]['exp_avg'], rtol=1e-4, atol=1e-5)
    assert torch.allclose(exp_avg_sq, optimizer.state[params_ref]['exp_avg_sq'], rtol=1e-4, atol=1e-5)
    
    print("\u2713 AdamW correctness test passed")

if __name__ == "__main__":
    test_adamw_correctness()
