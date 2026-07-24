"""
IRIS Fused CUDA Kernel Module
================================
This module provides the compiled CUDA extension for fused IRIS operations.

The extension is built from fused_iris_kernel.cu and provides:
- iris_fused_cuda: Single-tensor fused IRIS update
- iris_multi_tensor_fused_cuda: Multi-tensor fused IRIS update
"""

# Try to import the compiled CUDA extension
try:
    from . import fused_iris_kernel
    __all__ = ['fused_iris_kernel']
except ImportError as e:
    # Extension not built or not available
    fused_iris_kernel = None
    import warnings
    warnings.warn(
        f"IRIS fused CUDA kernel not available: {e}\n"
        "The fused kernel will not be accessible. "
        "Please build the CUDA extension using: python setup.py build_ext --inplace",
        ImportWarning
    )
    __all__ = []

# Export availability check
def is_available():
    """Check if the fused CUDA kernel is available."""
    return fused_iris_kernel is not None

__all__.append('is_available')