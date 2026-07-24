"""
IRIS: Drift-Adaptive Optimizer for PyTorch
"""

__version__ = "0.1.0"
__author__ = "Udit Asthana"
__license__ = "MIT"

from .optimizer import IRIS
from .fused import is_fused_available, compile_cuda_extension

__all__ = [
    "IRIS",
    "is_fused_available",
    "compile_cuda_extension",
]

# Helpful info on import
def _check_fused():
    if is_fused_available():
        print("\u2714 IRIS initialized with fused CUDA kernels")
    else:
        print("\u2717 IRIS initialized without fused kernels (Python fallback)")
        print("  Install CUDA extension with: python setup.py install")

# Optionally auto-check on import
# _check_fused()