import torch
from torch.utils.cpp_extension import load
import os

# JIT compile the CUDA kernel
module_path = os.path.dirname(__file__)
softmax_cuda = load(
    name="softmax_cuda",
    sources=[
        os.path.join(module_path, "softmax_ops.cpp"),
        os.path.join(module_path, "naive_softmax.cu"),
    ],
    extra_include_paths=[os.path.join(module_path, "../include")],
    verbose=True
)

class NaiveSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Input validation is done in the C++ binding
        output = softmax_cuda.naive_softmax_forward(input)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        # Softmax gradient: grad_input = output * (grad_output - sum(output * grad_output))
        grad_input = output * (grad_output - (output * grad_output).sum(dim=-1, keepdim=True))
        return grad_input

def naive_softmax(input):
    return NaiveSoftmax.apply(input)
