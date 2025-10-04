from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='gelu_cuda',
    ext_modules=[
        CUDAExtension(
            name='gelu_cuda',
            sources=[
                'src/gelu_ops.cpp',
                'src/gelu.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
