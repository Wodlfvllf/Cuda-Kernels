from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='adamw_cuda',
    ext_modules=[
        CUDAExtension(
            name='adamw_cuda',
            sources=[
                'src/adamw_ops.cpp',
                'src/adamw.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
