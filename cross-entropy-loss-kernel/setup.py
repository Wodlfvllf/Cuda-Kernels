from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='cross_entropy_loss_cuda',
    ext_modules=[
        CUDAExtension(
            name='cross_entropy_loss_cuda',
            sources=[
                'src/cross_entropy_loss_ops.cpp',
                'src/cross_entropy_loss.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
