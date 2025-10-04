from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='softmax_cuda',
    ext_modules=[
        CUDAExtension(
            name='softmax_cuda',
            sources=[
                'src/softmax_ops.cpp',
                'src/naive_softmax.cu',
                'src/optimized_softmax.cu',
            ],
            
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': []
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
