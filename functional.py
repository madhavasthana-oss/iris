"""Functional API for IRIS optimizer - OPTIMIZED with lerp and innovation reuse"""
import warnings
from typing import List, Optional, Tuple
import torch
from torch import Tensor
from torch.optim.optimizer import _get_value

from .fused import _fused_iris, is_fused_available

def iris(
    params: List[Tensor],
    grads: List[Tensor],
    grad_estimates: List[Tensor],
    variance_estimates: List[Tensor],
    max_variance_estimates: List[Tensor],
    innovation_residuals: List[Tensor],
    state_steps: List[Tensor],
    psi_1_prev: float,
    psi_2_prev: float,
    psi_3_prev: float,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    beta3: float,
    wd: float,
    eps: float,
    rho: Optional[float],
    amsgrad: bool = False,
    maximize: bool = False,
    has_complex: bool = False,
    foreach: Optional[bool] = None,
    fused: bool = False,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
) -> Tuple[float, float, float]:
    """
    Functional IRIS optimizer step.

    Innovation-based error correction with variance matching.
    Reuses the innovation tensor to keep allocations low.

    Args:
        params: Parameters to optimize
        grads: Current gradients g_t
        grad_estimates: Bias-corrected gradient estimates
        variance_estimates: Bias-corrected variance estimates
        max_variance_estimates: Max variance for AMSGrad
        innovation_residuals: Bias-corrected innovation residual
        state_steps: Step counters
        psi_1_prev: Previous bias-correction accumulator for g_est
        psi_2_prev: Previous bias-correction accumulator for innov_res
        psi_3_prev: Previous bias-correction accumulator for variance
        lr: Learning rate
        beta1: EMA coefficient for gradient estimate
        beta2: EMA coefficient for innovation residual
        beta3: EMA coefficient for variance estimate
        wd: Weight decay
        eps: Numerical stability constant
        rho: SNR threshold for trust-region clipping
        amsgrad: Use AMSGrad variant
        maximize: Maximize objective instead of minimize
        has_complex: Parameters contain complex numbers
        foreach: Use multi-tensor operations
        fused: Use fused CUDA kernel
        grad_scale: Gradient scaler for mixed precision
        found_inf: Infinity flag for mixed precision

    Returns:
        Updated bias-correction accumulators (psi1, psi2, psi3)
    """
    if not torch.compiler.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if grad_scale is not None and found_inf is not None:
        if _get_value(found_inf):
            return psi_1_prev, psi_2_prev, psi_3_prev
        inv_scale = 1.0 / _get_value(grad_scale)
        grads = [g * inv_scale if g is not None else None for g in grads]
    
    if maximize:
        grads = [torch.neg(g) if g is not None else None for g in grads]

    if foreach is None:
        foreach = not torch.jit.is_scripting()
    
    if fused:
        if is_fused_available() and not has_complex:
            func = _fused_iris
        else:
            if has_complex:
                warnings.warn(
                    "IRIS: fused=True with complex parameters, falling back to foreach",
                    RuntimeWarning,
                )
            else:
                warnings.warn(
                    "IRIS: fused=True but CUDA extension unavailable, falling back to foreach",
                    RuntimeWarning,
                )
            func = _multi_tensor_iris
    elif foreach:
        func = _multi_tensor_iris
    else:
        func = _single_tensor_iris

    return func(
        params=params,
        grads=grads,
        grad_estimates=grad_estimates,
        variance_estimates=variance_estimates,
        max_variance_estimates=max_variance_estimates,
        innovation_residuals=innovation_residuals,
        state_steps=state_steps,
        psi_1_prev=psi_1_prev,
        psi_2_prev=psi_2_prev,
        psi_3_prev=psi_3_prev,
        lr=lr,
        beta1=beta1,
        beta2=beta2,
        beta3=beta3,
        wd=wd,
        eps=eps,
        rho=rho,
        amsgrad=amsgrad,
        has_complex=has_complex,
    )


def _single_tensor_iris(
    params: List[Tensor],
    grads: List[Tensor],
    grad_estimates: List[Tensor],
    variance_estimates: List[Tensor],
    max_variance_estimates: List[Tensor],
    innovation_residuals: List[Tensor],
    state_steps: List[Tensor],
    psi_1_prev: float,
    psi_2_prev: float,
    psi_3_prev: float,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    beta3: float,
    wd: float,
    eps: float,
    rho: Optional[float],
    amsgrad: bool,
    has_complex: bool,
) -> Tuple[float, float, float]:
    """Single-tensor IRIS implementation."""
    clip_updates = rho is not None

    psi_1_curr = beta1 * psi_1_prev + 1.0
    psi_2_curr = beta2 * psi_2_prev + 1.0
    psi_3_curr = beta3 * psi_3_prev + 1.0

    psi_inv_1 = 1.0 / psi_1_curr
    psi_inv_2 = 1.0 / psi_2_curr
    psi_inv_3 = 1.0 / psi_3_curr
        
    for i, param in enumerate(params):
        grad = grads[i]
        if grad is None:
            continue
        
        grad_est = grad_estimates[i]
        var_est = variance_estimates[i]
        innov_res = innovation_residuals[i]
        step_t = state_steps[i]
        
        if not torch.isfinite(grad).all():
            continue
        
        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            grad_est = torch.view_as_real(grad_est)
            var_est = torch.view_as_real(var_est)
            innov_res = torch.view_as_real(innov_res)
            param = torch.view_as_real(param)
            if amsgrad:
                max_variance_estimates[i] = torch.view_as_real(max_variance_estimates[i])
        
        step_t += 1

        innovation = grad.sub(grad_est)

        grad_est.add_(innovation, alpha=psi_inv_1)

        innov_res.lerp_(innovation, psi_inv_2)
        corrected_grad = grad.add(innovation, alpha=beta2)

        var_est.mul_(1.0 - psi_inv_3).addcmul_(corrected_grad, corrected_grad, value=psi_inv_3)

        if amsgrad:
            max_var_est = max_variance_estimates[i]
            torch.maximum(max_var_est, var_est, out=max_var_est)
            denom = max_var_est.sqrt()
        else:
            denom = var_est.sqrt()

        if clip_updates:
            denom.mul_(rho)
        denom.add_(eps)

        if wd != 0:
            param.mul_(1.0 - lr * wd)

        if rho is not None:
            update = grad_est.add(innov_res, alpha=beta2).div_(denom).clamp_(-1.0, 1.0)
            param.add_(update, alpha=-lr)
        else:
            numerator = grad_est.add(innov_res, alpha=beta2)
            param.addcdiv_(numerator, denom, value=-lr)
    
    return psi_1_curr, psi_2_curr, psi_3_curr


def _multi_tensor_iris(
    params: List[Tensor],
    grads: List[Tensor],
    grad_estimates: List[Tensor],
    variance_estimates: List[Tensor],
    max_variance_estimates: List[Tensor],
    innovation_residuals: List[Tensor],
    state_steps: List[Tensor],
    psi_1_prev: float,
    psi_2_prev: float,
    psi_3_prev: float,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    beta3: float,
    wd: float,
    eps: float,
    rho: Optional[float],
    amsgrad: bool,
    has_complex: bool,
) -> Tuple[float, float, float]:
    """Multi-tensor IRIS implementation."""
    clip_updates = rho is not None

    psi_1_curr = beta1 * psi_1_prev + 1.0
    psi_2_curr = beta2 * psi_2_prev + 1.0
    psi_3_curr = beta3 * psi_3_prev + 1.0

    psi_inv_1 = 1.0 / psi_1_curr
    psi_inv_2 = 1.0 / psi_2_curr
    psi_inv_3 = 1.0 / psi_3_curr

    # Filter valid gradients
    non_none_indices = []
    for i, g in enumerate(grads):
        if g is not None and torch.isfinite(g).all():
            non_none_indices.append(i)
    
    if not non_none_indices:
        return psi_1_curr, psi_2_curr, psi_3_curr
    
    # Group tensors by device, dtype, complex type
    grouped_tensors = {}
    for idx in non_none_indices:
        param = params[idx]
        group_key = (param.device, param.dtype, torch.is_complex(param))
        
        if group_key not in grouped_tensors:
            grouped_tensors[group_key] = {
                'params': [], 'grads': [], 'grad_estimates': [],
                'variance_estimates': [], 'max_variance_estimates': [], 
                'innovation_residuals': [], 'state_steps': []
            }
        
        grouped_tensors[group_key]['params'].append(param)
        grouped_tensors[group_key]['grads'].append(grads[idx])
        grouped_tensors[group_key]['grad_estimates'].append(grad_estimates[idx])
        grouped_tensors[group_key]['variance_estimates'].append(variance_estimates[idx])
        if amsgrad:
            grouped_tensors[group_key]['max_variance_estimates'].append(max_variance_estimates[idx])
        grouped_tensors[group_key]['innovation_residuals'].append(innovation_residuals[idx])
        grouped_tensors[group_key]['state_steps'].append(state_steps[idx])
    
    # Process each device group
    for (device, dtype, is_complex), tensors in grouped_tensors.items():
        device_params    = tensors['params']
        device_grads     = tensors['grads']
        device_grad_ests = tensors['grad_estimates']
        device_var_ests  = tensors['variance_estimates']
        device_max_var_ests = tensors['max_variance_estimates']
        device_innov_res = tensors['innovation_residuals']
        device_state_steps = tensors['state_steps']
        
        # Handle complex tensors
        if is_complex:
            device_params    = [torch.view_as_real(p) for p in device_params]
            device_grads     = [torch.view_as_real(g) for g in device_grads]
            device_grad_ests = [torch.view_as_real(m) for m in device_grad_ests]
            device_var_ests  = [torch.view_as_real(s) for s in device_var_ests]
            if amsgrad:
                device_max_var_ests = [torch.view_as_real(s) for s in device_max_var_ests]
            device_innov_res = [torch.view_as_real(s) for s in device_innov_res]
        
        # Increment step counter
        torch._foreach_add_(device_state_steps, 1)
        innovations = torch._foreach_sub(device_grads, device_grad_ests)
        torch._foreach_add_(device_grad_ests, innovations, alpha=psi_inv_1)
        torch._foreach_lerp_(device_innov_res, innovations, psi_inv_2)

        corrected_grads = torch._foreach_add(device_grads, innovations, alpha=beta2)

        torch._foreach_mul_(device_var_ests, 1.0 - psi_inv_3)
        torch._foreach_mul_(corrected_grads, corrected_grads)
        torch._foreach_add_(device_var_ests, corrected_grads, alpha=psi_inv_3)

        # Denominator
        if amsgrad:
            torch._foreach_maximum_(device_max_var_ests, device_var_ests)
            denoms = torch._foreach_sqrt(device_max_var_ests)
        else:
            denoms = torch._foreach_sqrt(device_var_ests)

        if clip_updates:
            torch._foreach_mul_(denoms, rho)
        torch._foreach_add_(denoms, eps)

        # Decoupled weight decay
        if wd != 0:
            torch._foreach_mul_(device_params, 1.0 - lr * wd)

        if rho is not None:
            updates = torch._foreach_mul(device_innov_res, beta2)
            torch._foreach_add_(updates, device_grad_ests)
            torch._foreach_div_(updates, denoms)
            torch._foreach_clamp_(updates, -1.0, 1.0)
            torch._foreach_add_(device_params, updates, alpha=-lr)
        else:
            torch._foreach_addcdiv_(device_params, device_grad_ests, denoms, value=-lr)
            torch._foreach_addcdiv_(device_params, device_innov_res, denoms, value=-lr * beta2)
    
    return psi_1_curr, psi_2_curr, psi_3_curr