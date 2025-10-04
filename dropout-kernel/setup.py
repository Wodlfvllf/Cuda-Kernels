from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='dropout_cuda',
    ext_modules=[
        CUDAExtension(
            name='dropout_cuda',
            sources=[
                'src/dropout_ops.cpp',
                'src/dropout.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
