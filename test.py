#!/usr/bin/env python
"""Standalone test script for IRIS CUDA fused kernel.

Run this after building to verify the CUDA extension works correctly.
Place this file in the same directory as setup.py (inside iris/)
"""
import sys
import os

def main():
    print(f"Testing from Python {sys.version.split()[0]}")
    print()

    # Test 1: Direct import from fused_iris
    print("=== Test 1: Direct CUDA kernel import ===")
    try:
        # Try both import paths
        try:
            from iris.fused_iris import fused_iris_kernel
            print("✓ Import successful: from iris.fused_iris import fused_iris_kernel")
        except ImportError:
            from fused_iris import fused_iris_kernel
            print("✓ Import successful: from fused_iris import fused_iris_kernel")
        
        # Try to get file location
        try:
            file_path = fused_iris_kernel.__file__
            if file_path:
                print(f"  Loaded from: {file_path}")
            else:
                print("  Loaded as compiled extension (no __file__)")
        except AttributeError:
            print("  Loaded as compiled extension")
        
        funcs = [x for x in dir(fused_iris_kernel) if not x.startswith('_')]
        print(f"  Available functions: {funcs}")
        
        # Verify the critical functions exist
        required_funcs = ['iris_fused_cuda', 'iris_multi_tensor_fused_cuda']
        missing = [f for f in required_funcs if f not in funcs]
        if missing:
            print(f"✗ ERROR: Missing required functions: {missing}")
            return False
        
        print("  ✓ All required functions present")
        
    except ImportError as e:
        print(f"✗ Direct import failed: {e}")
        print()
        return False

    print()

    # Test 2: Package import
    print("=== Test 2: IRIS package import ===")
    try:
        from iris.fused import is_fused_available
        print("✓ Package import successful: from iris.fused import is_fused_available")
        
        available = is_fused_available()
        print(f"  CUDA available: {available}")
        
        if not available:
            print("✗ ERROR: CUDA should be available but isn't!")
            return False
            
        print("  ✓ Fused kernel detected correctly")
        
    except ImportError as e:
        print(f"✗ Package import failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()

    # Test 3: Create optimizer (minimal test)
    print("=== Test 3: IRIS optimizer with fused kernel ===")
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("✗ CUDA not available in PyTorch")
            return False
        
        from iris import IRIS
        
        # Create a tiny model
        model = torch.nn.Linear(2, 2).cuda()
        
        # Create optimizer with fused=True
        optimizer = IRIS(model.parameters(), fused=True)
        
        # Run one step
        output = model(torch.randn(1, 2, device='cuda'))
        loss = output.sum()
        loss.backward()
        optimizer.step()
        
        print("✓ Optimizer test successful")
        print("  Created IRIS optimizer with fused=True")
        print("  Performed one optimization step")
        
    except Exception as e:
        print(f"✗ Optimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
    print()
    print("IRIS is ready to use:")
    print()
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)