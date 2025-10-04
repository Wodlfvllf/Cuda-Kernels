from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='embedding_lookup_cuda',
    ext_modules=[
        CUDAExtension(
            name='embedding_lookup_cuda',
            sources=[
                'src/embedding_lookup_ops.cpp',
                'src/embedding_lookup.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
