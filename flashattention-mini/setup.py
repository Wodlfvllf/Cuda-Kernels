
from setuptools import setup, find_packages
from torch.utils import cpp_extension
import torch
import os

# Check if CUDA is available
if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available. Please install CUDA toolkit and PyTorch with CUDA support.")

def get_cuda_version():
    """Get CUDA version for proper arch selection"""
    try:
        import subprocess
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            for line in result.stdout.split('\n'):
                if 'release' in line:
                    version = line.split('release ')[1].split(',')[0]
                    return version
    except:
        pass
    return "11.0"  # Default fallback

# Determine CUDA architectures based on GPU capability
def get_cuda_arch_flags():
    """Get appropriate CUDA architecture flags"""
    cuda_version = get_cuda_version()
    
    # Common architectures - adjust based on your target GPUs
    arch_flags = [
        '-arch=sm_70',  # V100
        '-arch=sm_75',  # T4, RTX 2080 
        '-arch=sm_80',  # A100
        '-arch=sm_86',  # RTX 30 series
    ]
    
    # Add newer architectures if CUDA version supports them
    if float(cuda_version) >= 11.1:
        arch_flags.append('-arch=sm_87')  # RTX 40 series
    if float(cuda_version) >= 11.8:
        arch_flags.append('-arch=sm_89')  # RTX 40 series high-end
        arch_flags.append('-arch=sm_90')  # H100
    
    return arch_flags

# CUDA extension
ext_modules = [
    cpp_extension.CUDAExtension(
        name='flashattention_mini_cuda',
        sources=[
            'src/attention_wrapper.cpp',
            'src/attention_kernel.cu',
        ],
        extra_compile_args={
            'cxx': [
                '-O3',
                '-std=c++14',
                '-fPIC',
            ],
            'nvcc': [
                '-O3',
                '--use_fast_math',
                '-lineinfo',
                '--extra-device-vectorization',
                '--restrict',
                '-std=c++14',
            ] + get_cuda_arch_flags()
        },
        include_dirs=[
            'src/',
        ],
        define_macros=[
            ('TORCH_API_INCLUDE_EXTENSION_H', None),
            ('TORCH_EXTENSION_NAME', 'flashattention_mini_cuda'),
        ],
    )
]

setup(
    name='flashattention-mini',
    version='0.1.0',
    description='A simplified FlashAttention implementation for educational purposes',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/flashattention-mini',
    
    packages=find_packages(),
    package_dir={'flashattention_mini': 'src'},
    
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.20.0',
    ],
    
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'matplotlib>=3.3.0',
            'seaborn>=0.11.0',
            'jupyter>=1.0.0',
            'tqdm>=4.60.0',
            'pandas>=1.3.0',
            'plotly>=5.0.0',
        ],
    },
    
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Researchers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: C++',
        'Programming Language :: CUDA',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    
    python_requires='>=3.8',
    
    # Include additional files
    include_package_data=True,
    package_data={
        'flashattention_mini': ['*.cu', '*.cuh', '*.cpp', '*.h'],
    },
    
    # Build options
    zip_safe=False,  # Required for C++ extensions
)

# Post-installation validation
def validate_installation():
    """Validate that CUDA extension built correctly"""
    try:
        import torch
        import flashattention_mini_cuda
        print("FlashAttention-Mini CUDA extension built successfully!")
        
        # Test basic functionality
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        if device.type == 'cuda':
            print(f"CUDA Device: {torch.cuda.get_device_name()}")
            print(f"CUDA Capability: {torch.cuda.get_device_capability()}")
            
    except ImportError as e:
        print(f"Failed to import CUDA extension: {e}")
        print("Please check your CUDA installation and rebuild.")
    except Exception as e:
        print(f"Validation error: {e}")

if __name__ == "__main__":
    # Run validation after setup
    import sys
    if "install" in sys.argv or "develop" in sys.argv:
        # Import here to avoid issues during build
        import atexit
        atexit.register(validate_installation)