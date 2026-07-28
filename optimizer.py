"""IRIS optimizer: Innovation Residual Iterative Stabilization"""
import torch
from torch import Tensor
from torch.optim.optimizer import (
    Optimizer,
    ParamsT,
    _get_scalar_dtype,
    _use_grad_for_differentiable,
)
from typing import List, Tuple, Optional

from .functional import iris


class IRIS(Optimizer):
    """
    IRIS: Innovation Residual Iterative Stabilization

    Error-correcting gradient estimation with innovation tracking.

    Treats the step as predictive error correction rather than pure momentum
    extrapolation. Tracks innovation I_t = g_t - g_est_{t-1} and folds
    accumulated innovation residuals back into the gradient estimate.

    Args:
        params: Parameters to optimize
        lr: Learning rate (default: 3e-3)
        betas: EMA coefficients (beta1, beta2, beta3) (default: (0.98, 0.92, 0.99))
            - beta1: gradient estimate (higher = smoother)
            - beta2: innovation residual (lower = faster correction)
            - beta3: variance (higher = more stable)
        snr_threshold: SNR threshold for trust-region clipping (default: None)
            None disables clipping. When set, clips updates to [-1, 1] after scaling.
        weight_decay: Decoupled weight decay (default: 0.01)
        eps: Numerical stability constant (default: 1e-8)
        amsgrad: Use AMSGrad variant (default: False)
        differentiable: Make optimizer differentiable (default: False)
        foreach: Use multi-tensor operations (default: None, auto-detect)
        fused: Use fused CUDA kernel (default: False)
    
    Example:
        >>> optimizer = IRIS(model.parameters(), lr=3e-3)
        >>> optimizer.zero_grad()
        >>> loss.backward()
        >>> optimizer.step()
    
    Reference:
        IRIS: Innovation Residual Iterative Stabilization
        https://arxiv.org/abs/XXXX.XXXXX
    """
    
    def __init__(
        self,
        params: ParamsT,
        lr: float = 3e-3,
        betas: Tuple[float, float, float] = (0.98, 0.92, 0.99),
        snr_threshold: Optional[float] = None,
        weight_decay: float = 0.01,
        eps: float = 1e-8,
        amsgrad: bool = False,
        differentiable: bool = False,
        foreach: Optional[bool] = None,
        fused: bool = False
    ):
        if not lr > 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        
        if snr_threshold is not None and not snr_threshold > 0.0:
            raise ValueError(f"Invalid snr_threshold: {snr_threshold}")
        
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta3: {betas[2]}")

        if not weight_decay >= 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        if not eps > 0.0:
            raise ValueError(f"Invalid eps: {eps}")

        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            beta3=betas[2],
            rho=snr_threshold,
            weight_decay=weight_decay,
            eps=eps,
            amsgrad=amsgrad,
            differentiable=differentiable,
            foreach=foreach,
            fused=fused,
            psi_1_curr=0.0,
            psi_2_curr=0.0,
            psi_3_curr=0.0,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("beta1", 0.98)
            group.setdefault("beta2", 0.92)
            group.setdefault("beta3", 0.99)
            group.setdefault("rho", None)
            group.setdefault("weight_decay", 0.01)
            group.setdefault("eps", 1e-8)
            group.setdefault("differentiable", False)
            group.setdefault("foreach", None)
            group.setdefault("fused", False)
            group.setdefault("psi_1_curr", 0.0)
            group.setdefault("psi_2_curr", 0.0)
            group.setdefault("psi_3_curr", 0.0)
            
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = torch.tensor(step_val, device=p.device)

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        amsgrad,
        grad_estimates,
        variance_estimates,
        max_variance_estimates,
        innovation_residuals,
        state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is None:
                continue

            has_complex |= torch.is_complex(p)
            params_with_grad.append(p)

            if p.grad.is_sparse:
                raise RuntimeError("IRIS does not support sparse gradients")

            grads.append(p.grad)
            state = self.state[p]

            if len(state) == 0:
                state["step"] = torch.zeros(
                    (), dtype=_get_scalar_dtype(is_fused=group.get("fused", False)), device=p.device
                )
                state["grad_estimate"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state["variance_estimate"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                if amsgrad:
                    state["max_variance_estimate"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                state["innovation_residual"] = torch.zeros_like(p, memory_format=torch.preserve_format)

            grad_estimates.append(state["grad_estimate"])
            variance_estimates.append(state["variance_estimate"])

            if amsgrad:
                max_variance_estimates.append(state["max_variance_estimate"])
            
            innovation_residuals.append(state["innovation_residual"])

            state_steps.append(state["step"])

        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.
        
        Args:
            closure: Closure that reevaluates the model and returns loss
            
        Returns:
            Loss if closure provided, otherwise None
        """
        # Present on PyTorch >= ~1.12; safe no-op on older installs.
        if hasattr(self, "_cuda_graph_capture_health_check"):
            self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad: List[Tensor] = []
            grads: List[Tensor] = []
            grad_estimates: List[Tensor] = []
            variance_estimates: List[Tensor] = []
            max_variance_estimates: List[Tensor] = []
            innovation_residuals: List[Tensor] = []
            state_steps: List[Tensor] = []

            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                group["amsgrad"],
                grad_estimates,
                variance_estimates,
                max_variance_estimates,
                innovation_residuals,
                state_steps,
            )

            if not params_with_grad:
                continue

            psi_1_curr, psi_2_curr, psi_3_curr = iris(
                params_with_grad,
                grads,
                grad_estimates,
                variance_estimates,
                max_variance_estimates,
                innovation_residuals,
                state_steps,
                psi_1_prev=group["psi_1_curr"],
                psi_2_prev=group["psi_2_curr"],
                psi_3_prev=group["psi_3_curr"],
                lr=group["lr"],
                beta1=group["beta1"],
                beta2=group["beta2"],
                beta3=group["beta3"],
                wd=group["weight_decay"],
                eps=group["eps"],
                rho=group["rho"],
                amsgrad=group["amsgrad"],
                has_complex=has_complex,
                foreach=group["foreach"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )
            
            group["psi_1_curr"] = psi_1_curr
            group["psi_2_curr"] = psi_2_curr
            group["psi_3_curr"] = psi_3_curr

        return loss