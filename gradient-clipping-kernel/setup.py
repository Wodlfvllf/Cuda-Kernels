from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='grad_clip_cuda',
    ext_modules=[
        CUDAExtension(
            name='grad_clip_cuda',
            sources=[
                'src/grad_clip_ops.cpp',
                'src/grad_clip.cu',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': ['-O3']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
