"""Setup script for IRIS optimizer with CUDA extensions.

IRIS: InNOvation Variance Adaptive momentum estimation
Run from iris/ directory: cd iris && python setup.py build_ext --inplace
"""

from setuptools import setup, find_packages
import os

# Define extensions
ext_modules = []
cmdclass = {}

try:
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension
    import torch
    
    cuda_available = torch.cuda.is_available()
    
    if cuda_available:
        ext_modules = [
            CUDAExtension(
                # Module name for import: from iris.fused_iris import fused_iris_kernel
                name='iris.fused_iris.fused_iris_kernel',
                # Path relative to setup.py (package root)
                sources=['fused_iris/fused_iris_kernel.cu'],
                
                extra_compile_args={
                    'cxx': ['-O3', '-std=c++17'],
                    'nvcc': [
                        '-O3',
                        '--use_fast_math',
                        '-std=c++17',
                        '--expt-relaxed-constexpr',
                        '-gencode', 'arch=compute_80,code=sm_80',  # A100
                        '-gencode', 'arch=compute_86,code=sm_86',  # RTX 3090
                        '-gencode', 'arch=compute_89,code=sm_89',  # RTX 4090
                        '-gencode', 'arch=compute_90,code=sm_90',  # H100
                    ]
                }
            )
        ]
        cmdclass = {'build_ext': BuildExtension}
        print("✓ CUDA detected - building IRIS with fused kernels")
        print("  Innovation: I_t = g_t - m_{t-1}")
    else:
        print("! CUDA not available - building without fused kernels")

except ImportError:
    print("! PyTorch not found - skipping CUDA extensions")

setup(
    name='iris-optimizer',
    version='0.1.0',
    author='Udit Asthana',
    author_email='madhavasthana@gmail.com',
    description='IRIS: Innovation Residual Iterative Stabilization',
    # Package discovery from current directory
    packages=['iris', 'iris.fused_iris'],
    package_dir={'iris': '.'},  # Current dir (iris/) is the iris package
    
    package_data={
        'iris': ['*.so', '*.pyd'],
        'iris.fused_iris': ['*.cu', '*.so', '*.pyd']
    },
    
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    
    install_requires=['torch>=2.0.0'],
    python_requires='>=3.8',
    
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords='deep-learning optimizer pytorch adam kalman-filter innovation',
)