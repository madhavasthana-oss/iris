"""
Standalone script to compile CUDA extensions for IRIS optimizer.
Run this before using the optimizer if you haven't installed via setup.py.

IRIS: InNOvation Variance Adaptive momentum estimation
- Innovation-based variance (TRUE Kalman innovation)
- Trust-based clipping for robust updates
- No self-contamination (unlike AdaBelief)

Usage:
    python build_cuda.py
"""

import torch
import os
import sys

def build_cuda_extension():
    """Build the CUDA extension and place it in the IRIS package."""
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("✗ Error: CUDA is not available on this system")
        print("  Please ensure you have:")
        print("  1. NVIDIA GPU")
        print("  2. CUDA toolkit installed")
        print("  3. PyTorch with CUDA support")
        return False
    
    print(f"✓ CUDA {torch.version.cuda} detected")
    print(f"✓ PyTorch {torch.__version__} with CUDA support")
    
    # Find the CUDA source file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_source = os.path.join(script_dir, "fused_iris", "fused_iris_kernel.cu")
    
    if not os.path.exists(cuda_source):
        print(f"✗ Error: CUDA source file not found at {cuda_source}")
        print(f"  Current directory: {script_dir}")
        print(f"  Please ensure your directory structure is:")
        print(f"    iris/")
        print(f"    |-- fused_iris/")
        print(f"    |   |-- __init__.py")
        print(f"    |   `-- fused_iris_kernel.cu")
        print(f"    |-- __init__.py")
        print(f"    |-- build_cuda.py")
        print(f"    |-- diagnose_complete.py")
        print(f"    |-- functional.py")
        print(f"    |-- fused.py")
        print(f"    `-- optimizer.py")
        return False
    
    print(f"✓ Found CUDA source: {cuda_source}")
    
    try:
        from torch.utils.cpp_extension import load
        
        print("\n" + "="*60)
        print("Building IRIS CUDA extension...")
        print("This may take 2-5 minutes on first build...")
        print("="*60 + "\n")
        
        # Build the extension
        iris_cuda = load(
            name='fused_iris_kernel',
            sources=[cuda_source],
            extra_cflags=['-O3', '-std=c++14'],
            extra_cuda_cflags=[
                '-O3',
                '--use_fast_math',
                '-std=c++14',
                '--expt-relaxed-constexpr',
            ],
            verbose=True,
            with_cuda=True,
        )
        
        print("\n" + "="*60)
        print("✓ SUCCESS! CUDA extension compiled successfully!")
        print("="*60)
        print("\nYou can now use the fused optimizer:")
        print("  from iris import IRIS")
        print("  optimizer = IRIS(model.parameters(), fused=True)")
        print("\nIRIS innovation notes:")
        print("  - Innovation: I_t = g_t - m_{t-1} (Kalman-style)")
        print("  - Variance: v_t tracks E[I_t^2] (full signal strength)")
        print("  - Trust-based clipping for robust updates")
        print("\n")
        
        return True
        
    except Exception as e:
        print("\n" + "="*60)
        print("✗ COMPILATION FAILED")
        print("="*60)
        print(f"\nError: {e}")
        print("\nCommon issues:")
        print("1. CUDA toolkit version mismatch with PyTorch")
        print("   Check: nvcc --version")
        print("   Should match: python -c 'import torch; print(torch.version.cuda)'")
        print("\n2. Missing CUDA development tools")
        print("   Install: CUDA toolkit from nvidia.com/cuda-downloads")
        print("\n3. Insufficient GPU compute capability")
        print("   Minimum required: Compute Capability 3.5+")
        print("\nFallback: Use fused=False for pure PyTorch implementation")
        print("  optimizer = IRIS(model.parameters(), fused=False)")
        return False

def test_extension():
    """Test if the compiled extension works."""
    try:
        from fused_iris import fused_iris_kernel
        print("\n" + "="*60)
        print("Testing compiled extension...")
        print("="*60)
        
        # Create test tensors
        device = torch.device('cuda')
        param = torch.randn(100, device=device, requires_grad=True)
        grad = torch.randn(100, device=device)
        exp_avg = torch.zeros(100, device=device)
        exp_avg_sq = torch.zeros(100, device=device)
        param_uncertainty = torch.zeros(100, device=device)
        max_exp_avg = torch.zeros(100, device=device)
        step = torch.tensor(0, dtype=torch.int64, device=device)
        
        print("✓ Imported fused_iris_kernel successfully")
        print(f"  Available functions: {[x for x in dir(fused_iris_kernel) if not x.startswith('_')]}")
        
        # Call the function
        result = fused_iris_kernel.iris_fused_cuda(
            param, grad, exp_avg, exp_avg_sq, param_uncertainty,
            max_exp_avg, step,
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
            wd=0.01,
            eps=1e-8,
            trust=1.0,
            kalman_mode=False,
            process_noise=1e-8,
            amsgrad=False
        )
        
        print("✓ Extension test passed!")
        print(f"  Function executed successfully")
        print(f"  Parameters updated in-place")
        return True
        
    except Exception as e:
        print(f"✗ Extension test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("\n" + "="*60)
    print("IRIS Optimizer - CUDA Extension Builder")
    print("InNOvation Variance Adaptive")
    print("="*60 + "\n")
    
    success = build_cuda_extension()
    
    if success:
        print("\nTesting the compiled extension...\n")
        test_extension()
    else:
        sys.exit(1)