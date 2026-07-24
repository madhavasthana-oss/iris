"""
Complete diagnostic script for IRIS optimizer CUDA compilation debugging.
Run this to understand what's going wrong with IRIS and CUDA extensions.

Usage:
    python diagnose_complete.py
"""

import os
import sys
import subprocess
import importlib.util

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)

def check_cuda_setup():
    """Check basic CUDA setup."""
    print_section("1. CUDA Setup Check")
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✓ CUDA version: {torch.version.cuda}")
            print(f"✓ GPU count: {torch.cuda.device_count()}")
            print(f"✓ Current device: {torch.cuda.current_device()}")
            print(f"✓ Device name: {torch.cuda.get_device_name(0)}")
        else:
            print("✗ CUDA not available in PyTorch")
            return False
    except Exception as e:
        print(f"✗ Error checking CUDA: {e}")
        return False
    
    return True

def check_cuda_compiler():
    """Check if nvcc is available and working."""
    print_section("2. CUDA Compiler Check")
    
    # Check nvcc
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ nvcc found:")
            print(result.stdout)
        else:
            print("✗ nvcc command failed")
            print(result.stderr)
            return False
    except FileNotFoundError:
        print("✗ nvcc not found in PATH")
        print("\nPlease ensure CUDA is installed and nvcc is in your PATH:")
        print("  export PATH=/usr/local/cuda/bin:$PATH")
        return False
    except Exception as e:
        print(f"✗ Error running nvcc: {e}")
        return False
    
    # Check CUDA_HOME
    cuda_home = os.environ.get('CUDA_HOME')
    if cuda_home:
        print(f"✓ CUDA_HOME: {cuda_home}")
    else:
        print("! CUDA_HOME not set (may cause issues)")
        print("  Set with: export CUDA_HOME=/usr/local/cuda")
    
    return True

def check_directory_structure():
    """Check if files are in the right places."""
    print_section("3. Directory Structure Check")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Current directory: {script_dir}")
    
    # Check for CUDA source
    cuda_source = os.path.join(script_dir, "fused_iris_kernel.cu")
    if os.path.exists(cuda_source):
        print(f"✓ Found: {cuda_source}")
        size = os.path.getsize(cuda_source)
        print(f"  Size: {size} bytes")
        if size == 0:
            print("  ✗ WARNING: File is empty!")
            return False
    else:
        print(f"✗ Missing: {cuda_source}")
        print(f"  Expected location: {cuda_source}")
        return False
    
    # Check for Python files
    files_to_check = ["optimizer.py", "functional.py", "fused.py"]
    all_found = True
    for file in files_to_check:
        filepath = os.path.join(script_dir, file)
        if os.path.exists(filepath):
            print(f"✓ {file}")
        else:
            print(f"✗ {file} (missing)")
            all_found = False
    
    return all_found

def check_iris_import():
    """Check if IRIS package can be imported."""
    print_section("4. IRIS Package Import Check")
    
    try:
        import iris
        print(f"✓ IRIS package imported")
        print(f"  Location: {iris.__file__}")
        
        # Try to import IRIS optimizer
        from iris import IRIS
        print(f"✓ IRIS optimizer class imported")
        
        # Check for fused availability function
        from iris.fused import is_fused_available
        print(f"✓ is_fused_available function imported")
        
        return True
    except Exception as e:
        print(f"✗ Failed to import IRIS: {e}")
        import traceback
        traceback.print_exc()
        return False

def check_cuda_extension_import():
    """Try different ways to import the CUDA extension."""
    print_section("5. CUDA Extension Import Check")
    
    methods = [
        ("Direct import", "import fused_iris_kernel"),
        ("From iris.fused_iris", "from iris.fused_iris import fused_iris_kernel"),
    ]
    
    success = False
    
    for method_name, import_statement in methods:
        print(f"\nTrying: {import_statement}")
        try:
            exec(import_statement, globals())
            print(f"✓ Success!")
            if 'fused_iris_kernel' in globals():
                mod = globals()['fused_iris_kernel']
                print(f"  Location: {mod.__file__ if hasattr(mod, '__file__') else 'built-in'}")
                print(f"  Functions: {[x for x in dir(mod) if not x.startswith('_')]}")
            success = True
            break
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    # Check sys.modules
    print(f"\nChecking sys.modules:")
    if 'fused_iris_kernel' in sys.modules:
        print(f"✓ fused_iris_kernel found in sys.modules")
        print(f"  Location: {sys.modules['fused_iris_kernel']}")
        success = True
    else:
        print(f"✗ fused_iris_kernel not in sys.modules")
    
    return success

def check_torch_extensions_cache():
    """Check PyTorch extensions cache."""
    print_section("6. PyTorch Extensions Cache Check")
    
    try:
        import torch.utils.cpp_extension
        cache_dir = torch.utils.cpp_extension._get_build_directory('fused_iris_kernel', verbose=False)
        print(f"Cache directory: {cache_dir}")
        
        if os.path.exists(cache_dir):
            print(f"✓ Cache directory exists")
            files = os.listdir(cache_dir)
            print(f"  Contents ({len(files)} files):")
            for f in files[:10]:  # Show first 10 files
                print(f"    - {f}")
            if len(files) > 10:
                print(f"    ... and {len(files) - 10} more")
            
            # Look for compiled extension
            import glob
            so_files = glob.glob(os.path.join(cache_dir, "*.so")) + \
                      glob.glob(os.path.join(cache_dir, "*.pyd"))
            if so_files:
                print(f"\n✓ Found compiled extension(s):")
                for so in so_files:
                    print(f"    {os.path.basename(so)}")
                return True
            else:
                print(f"\n✗ No compiled extension (.so/.pyd) found")
                return False
        else:
            print(f"✗ Cache directory does not exist")
            return False
    except Exception as e:
        print(f"✗ Error checking cache: {e}")
        return False

def check_source_file():
    """Check the CUDA source file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cuda_source = os.path.join(script_dir, "fused_iris_kernel.cu")
    
    if not os.path.exists(cuda_source):
        return False, None
    
    try:
        with open(cuda_source, 'r') as f:
            lines = f.readlines()[:10]
            if len(lines) > 0:
                return True, cuda_source
    except:
        pass
    
    return False, cuda_source

def try_manual_compile(cuda_source):
    """Try to compile manually with nvcc to see errors."""
    print_section("7. Manual nvcc Compilation Test")
    
    if not cuda_source or not os.path.exists(cuda_source):
        print("✗ No valid source file provided")
        return False
    
    print("Attempting direct nvcc compilation...")
    
    try:
        import torch
        torch_include = os.path.join(os.path.dirname(torch.__file__), 'include')
        
        cmd = [
            'nvcc',
            '-c',  # Compile only, don't link
            cuda_source,
            f'-I{torch_include}',
            f'-I{torch_include}/torch/csrc/api/include',
            '-O3',
            '--use_fast_math',
            '-std=c++14',
            '-Xcompiler', '-fPIC',
            '--expt-relaxed-constexpr'
        ]
        
        print("Command:")
        print(' '.join(cmd))
        print("\n" + "-"*60)
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0:
            print("✓ Compilation succeeded!")
            return True
        else:
            print("✗ Compilation failed!")
            print("\nSTDOUT:")
            print(result.stdout)
            print("\nSTDERR:")
            print(result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("✗ Compilation timed out (>60s)")
        return False
    except Exception as e:
        print(f"✗ Error during compilation: {e}")
        import traceback
        traceback.print_exc()
        return False

def try_jit_compile():
    """Try JIT compilation with verbose output."""
    print_section("8. PyTorch JIT Compilation")
    
    try:
        from torch.utils.cpp_extension import load
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        cuda_source = os.path.join(script_dir, "fused_iris_kernel.cu")
        
        if not os.path.exists(cuda_source):
            print(f"✗ CUDA source not found: {cuda_source}")
            return False
        
        print(f"Compiling from: {cuda_source}")
        print(f"This may take 2-5 minutes...")
        print(f"(Verbose output enabled)\n")
        print("-"*60)
        
        fused_iris_kernel = load(
            name='fused_iris_kernel',
            sources=[cuda_source],
            extra_cuda_cflags=['-O3', '--use_fast_math', '-std=c++14', '--expt-relaxed-constexpr'],
            verbose=True
        )
        
        print("\n" + "-"*60)
        print(f"✓ JIT compilation succeeded!")
        print(f"  Module: {fused_iris_kernel}")
        print(f"  Functions: {[x for x in dir(fused_iris_kernel) if not x.startswith('_')]}")
        return True
        
    except Exception as e:
        print("\n" + "-"*60)
        print(f"✗ JIT compilation failed: {e}")
        print("\nFull traceback:")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("\n" + "="*60)
    print("  IRIS CUDA DIAGNOSTIC")
    print("  InNOvation Variance Adaptive Optimizer")
    print("="*60)
    
    results = {
        "CUDA Setup": check_cuda_setup(),
        "CUDA Compiler": check_cuda_compiler(),
        "Directory Structure": check_directory_structure(),
        "IRIS Import": check_iris_import(),
        "CUDA Extension": check_cuda_extension_import(),
        "Extensions Cache": check_torch_extensions_cache(),
    }
    
    # Summary
    print_section("SUMMARY")
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {check}")
    
    # Recommendations
    print_section("RECOMMENDATIONS")
    
    if not results["CUDA Setup"]:
        print("✗ CUDA is not properly set up")
        print("   -> Install CUDA toolkit and PyTorch with CUDA support")
        print("   -> Visit: https://pytorch.org/get-started/locally/")

    elif not results["CUDA Compiler"]:
        print("✗ CUDA compiler (nvcc) is not available")
        print("   -> Ensure CUDA toolkit is installed")
        print("   -> Add to PATH: export PATH=/usr/local/cuda/bin:$PATH")
        print("   -> Set CUDA_HOME: export CUDA_HOME=/usr/local/cuda")

    elif not results["Directory Structure"]:
        print("✗ Files are not in the right locations")
        print("   -> Ensure fused_iris_kernel.cu exists in the package root")
        print("   -> Ensure optimizer.py, functional.py, fused.py exist")

    elif not results["IRIS Import"]:
        print("✗ IRIS package cannot be imported")
        print("   -> Run: pip install -e . --no-cache-dir")
        print("   -> Or: pip install -e /path/to/iris")

    elif not results["CUDA Extension"]:
        print("✗ CUDA extension is not compiled or importable")
        print("\n   Detailed compilation diagnostics:")

        source_ok, cuda_source = check_source_file()
        if source_ok:
            print("\n   Choose diagnostic method:")
            print("   1. Manual nvcc compilation (raw compiler errors)")
            print("   2. PyTorch JIT compilation (full build process)")
            print("   3. Both")
            print("   4. Skip")

            choice = input("\n   Enter choice (1/2/3/4): ").strip()

            if choice == '1':
                try_manual_compile(cuda_source)
            elif choice == '2':
                try_jit_compile()
            elif choice == '3':
                try_manual_compile(cuda_source)
                try_jit_compile()

        print("\n   If compilation fails, you can still use IRIS without CUDA:")
        print("   optimizer = IRIS(model.parameters(), fused=False)")

    else:
        print("✓ Everything looks good!")
        print("   You should be able to use: IRIS(..., fused=True)")
        print("\n   IRIS uses innovation-based variance:")
        print("   - Innovation: I_t = g_t - m_{t-1} (Kalman-style)")
        print("   - Variance: v_t tracks E[I_t^2] (full signal strength)")
        print("   - Trust-based clipping for robust updates")

    print("\n" + "="*60)
    print("  DIAGNOSTIC COMPLETE")
    print("="*60)
    print("\nFor more info on IRIS theory vs AdaBelief:")
    print("  - AdaBelief: (g_t - m_t)^2 = dampened innovation")
    print("  - IRIS: (g_t - m_{t-1})^2 = pure innovation")
    print("  -> No self-contamination, full signal strength")

if __name__ == "__main__":
    main()