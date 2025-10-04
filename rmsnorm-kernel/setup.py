from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rmsnorm_cuda',
    ext_modules=[
        CUDAExtension(
            name='rmsnorm_cuda',
            sources=[
                'src/rmsnorm_ops.cpp',
                'src/naive_rmsnorm.cu',
                'src/optimized_rmsnorm.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
