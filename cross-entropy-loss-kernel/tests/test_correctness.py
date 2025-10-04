import torch
import cross_entropy_loss_cuda

def test_cross_entropy_loss_correctness():
    # Test dimensions
    batch_size = 4
    num_classes = 1000
    
    # Create input
    logits = torch.randn(batch_size, num_classes, device='cuda')
    labels = torch.randint(0, num_classes, (batch_size,), device='cuda')
    
    # Compute with custom kernel
    custom_output = cross_entropy_loss_cuda.cross_entropy_loss_forward(logits, labels)
    
    # Compute with reference
    ref_output = torch.nn.functional.cross_entropy(logits, labels, reduction='none')
    
    # Check correctness
    assert torch.allclose(custom_output, ref_output, rtol=1e-4, atol=1e-5)
    print("\u2713 Cross-Entropy Loss correctness test passed")

if __name__ == "__main__":
    test_cross_entropy_loss_correctness()
