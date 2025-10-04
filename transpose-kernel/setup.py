from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='transpose_cuda',
    ext_modules=[
        CUDAExtension(
            name='transpose_cuda',
            sources=[
                'src/transpose_ops.cpp',
                'src/transpose.cu',
                'src/optimized_transpose.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
