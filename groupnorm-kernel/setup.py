from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='groupnorm_cuda',
    ext_modules=[
        CUDAExtension(
            name='groupnorm_cuda',
            sources=[
                'src/groupnorm_ops.cpp',
                'src/groupnorm.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
