from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='rope_cuda',
    ext_modules=[
        CUDAExtension(
            name='rope_cuda',
            sources=[
                'src/rope_ops.cpp',
                'src/rope.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
