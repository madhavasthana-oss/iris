"""Fused CUDA kernel wrapper for IRIS optimizer."""

import torch
from typing import List, Optional, Tuple

_fused_kernel = None
_fused_available = False

try:
    from .fused_iris import fused_iris_kernel
    _fused_kernel = fused_iris_kernel
    _fused_available = True
except ImportError:
    try:
        from . import fused_iris_kernel
        _fused_kernel = fused_iris_kernel
        _fused_available = True
    except ImportError:
        _fused_available = False


def is_fused_available() -> bool:
    """Check if fused CUDA kernel is available."""
    return _fused_available


def _fused_iris(
    params: List[torch.Tensor],
    grads: List[torch.Tensor],
    grad_estimates: List[torch.Tensor],
    variance_estimates: List[torch.Tensor],
    max_variance_estimates: List[torch.Tensor],
    innovation_residuals: List[torch.Tensor],
    innovation_variances: List[torch.Tensor],
    state_steps: List[torch.Tensor],
    psi_1_prev: float,
    psi_2_prev: float,
    psi_3_prev: float,
    phi_prev: float,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    beta3: float,
    wd: float,
    eps: float,
    rho: Optional[float],
    amsgrad: bool,
    split_correction: Optional[float],
    has_complex: bool
) -> Tuple[float, float, float, float]:
    """
    Fused CUDA implementation of IRIS optimizer.

    Args:
        params: Parameters
        grads: Gradients g_t
        grad_estimates: Gradient estimates
        variance_estimates: Variance estimates
        max_variance_estimates: Max variance (AMSGrad)
        innovation_residuals: Innovation residuals
        innovation_variances: Innovation variances (heavy-ball only)
        state_steps: Step counters
        psi_1_prev: Previous bias-correction accumulator for g_est
        psi_2_prev: Previous bias-correction accumulator for innov_res
        psi_3_prev: Previous bias-correction accumulator for variance
        phi_prev: Previous bias-correction accumulator for innov_var
        lr: Learning rate
        beta1: EMA coefficient for gradient estimate
        beta2: EMA coefficient for innovation residual
        beta3: EMA coefficient for variance
        wd: Weight decay
        eps: Numerical stability constant
        rho: SNR threshold for clipping
        amsgrad: Use AMSGrad variant
        split_correction: Heavy-ball EMA coefficient (None = standard mode)
        has_complex: Complex parameters present

    Returns:
        Updated bias-correction accumulators (psi1, psi2, psi3, phi)
    """
    if not _fused_available:
        raise RuntimeError(
            "Fused IRIS kernel not available. Use foreach=True instead."
        )
    
    if has_complex:
        raise RuntimeError(
            "Fused IRIS kernel does not support complex parameters. Use foreach=True instead."
        )

    use_split_correction = split_correction is not None
    gamma = split_correction if use_split_correction else 0.0
    
    params_with_grad     = []
    grads_with_grad      = []
    grad_ests_with_grad  = []
    var_ests_with_grad   = []
    max_var_ests_with_grad = []
    innov_res_with_grad  = []
    innov_var_with_grad  = []
    steps_with_grad      = []
    
    for i, grad in enumerate(grads):
        if grad is not None:
            params_with_grad.append(params[i])
            grads_with_grad.append(grad)
            grad_ests_with_grad.append(grad_estimates[i])
            var_ests_with_grad.append(variance_estimates[i])
            if amsgrad:
                max_var_ests_with_grad.append(max_variance_estimates[i])
            innov_res_with_grad.append(innovation_residuals[i])
            if use_split_correction:
                innov_var_with_grad.append(innovation_variances[i])
            steps_with_grad.append(state_steps[i])
    
    if not params_with_grad:
        return psi_1_prev, psi_2_prev, psi_3_prev, phi_prev
    
    use_clipping = rho is not None
    rho_value = rho if rho is not None else 1.0
    
    psi_1_curr, psi_2_curr, psi_3_curr, phi_curr = _fused_kernel.iris_multi_tensor_fused_cuda(
        params_with_grad,
        grads_with_grad,
        grad_ests_with_grad,
        var_ests_with_grad,
        max_var_ests_with_grad if amsgrad else [],
        innov_res_with_grad,
        innov_var_with_grad if use_split_correction else [],
        steps_with_grad,
        psi_1_prev,
        psi_2_prev,
        psi_3_prev,
        phi_prev,
        lr,
        beta1,
        beta2,
        beta3,
        gamma,
        wd,
        eps,
        rho_value,
        use_clipping,
        amsgrad,
        use_split_correction,
    )
    
    return psi_1_curr, psi_2_curr, psi_3_curr, phi_curr


def compile_cuda_extension():
    """
    Compile the IRIS fused CUDA extension.
    
    Attempts to build the CUDA kernel from source.
    If successful, the fused kernel will be available.
    """
    try:
        from torch.utils.cpp_extension import load
        import os
        
        current_dir = os.path.dirname(os.path.abspath(__file__))
        kernel_path = os.path.join(current_dir, "fused_iris", "iris_fused_kernel.cu")
        
        if not os.path.exists(kernel_path):
            raise FileNotFoundError(
                f"CUDA kernel source not found at {kernel_path}"
            )
        
        print("Building IRIS fused CUDA kernel...")
        
        fused_iris = load(
            name="fused_iris_kernel",
            sources=[kernel_path],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            verbose=True
        )
        
        print("✓ IRIS fused kernel built successfully")
        return fused_iris
        
    except Exception as e:
        print(f"✗ Failed to build IRIS fused kernel: {e}")
        print("You can still use IRIS with foreach=True")
        raise


__all__ = ["is_fused_available", "compile_cuda_extension"]