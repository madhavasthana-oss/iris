#!/bin/bash
# IRIS CUDA Extension Build Script
# InNOvation Variance Adaptive Optimizer
# Run from: iris/ directory (where setup.py is)

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}===============================================================${NC}"
echo -e "${BLUE}             IRIS CUDA Extension Build Script${NC}"
echo -e "${BLUE}       Innovation Residual Iterative Stabilization${NC}"
echo -e "${BLUE}===============================================================${NC}"
echo ""

echo "=== Step 1: Check PyTorch CUDA version ==="
PYTHON_VERSION=$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')")
PYTHON_SHORT=$(python -c "import sys; print(f'{sys.version_info.major}{sys.version_info.minor}')")
PYTORCH_VERSION=$(python -c "import torch; print(torch.__version__)")
PYTORCH_CUDA=$(python -c "import torch; print(torch.version.cuda)")

echo "Python version: $PYTHON_VERSION"
echo "PyTorch version: $PYTORCH_VERSION"
echo "PyTorch CUDA: $PYTORCH_CUDA"

echo ""
echo "=== Step 2: Install Ninja build system (recommended) ==="
if ! command -v ninja &> /dev/null; then
    echo "Ninja not found, installing..."
    if command -v apt-get &> /dev/null; then
        sudo apt-get update -qq
        sudo apt-get install -y ninja-build
    elif command -v conda &> /dev/null; then
        conda install -y ninja -c conda-forge
    else
        pip install ninja
    fi
    
    if command -v ninja &> /dev/null; then
        echo -e "${GREEN}✓ Ninja installed successfully${NC}"
    else
        echo -e "${YELLOW}! Ninja installation failed, will use slower distutils backend${NC}"
    fi
else
    echo -e "${GREEN}✓ Ninja already installed: $(which ninja)${NC}"
fi

echo ""
echo "=== Step 3: Verify/Install CUDA Toolkit ==="
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
    echo -e "${GREEN}✓ nvcc found: $(which nvcc)${NC}"
    echo "NVCC version: $NVCC_VERSION"
    
    CUDA_MAJOR=$(echo $NVCC_VERSION | cut -d'.' -f1)
    PYTORCH_CUDA_MAJOR=$(echo $PYTORCH_CUDA | cut -d'.' -f1)
    
    if [ "$CUDA_MAJOR" != "$PYTORCH_CUDA_MAJOR" ]; then
        echo -e "${YELLOW}! WARNING: CUDA toolkit ($NVCC_VERSION) differs from PyTorch CUDA ($PYTORCH_CUDA)${NC}"
        echo "This might cause issues, but continuing..."
    fi
else
    echo "nvcc not found, installing CUDA Toolkit..."
    
    PYTORCH_CUDA_MAJOR=$(echo $PYTORCH_CUDA | cut -d'.' -f1)
    PYTORCH_CUDA_MINOR=$(echo $PYTORCH_CUDA | cut -d'.' -f2)
    
    if [ "$PYTORCH_CUDA_MAJOR" == "12" ]; then
        [ "$PYTORCH_CUDA_MINOR" == "1" ] && CUDA_VERSION="12-1" || CUDA_VERSION="12-8"
    elif [ "$PYTORCH_CUDA_MAJOR" == "11" ]; then
        CUDA_VERSION="11-8"
    else
        echo -e "${RED}✗ Unsupported CUDA version: $PYTORCH_CUDA${NC}"
        exit 1
    fi
    
    echo "Installing CUDA Toolkit $CUDA_VERSION..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    sudo dpkg -i cuda-keyring_1.1-1_all.deb
    sudo apt-get update
    sudo apt-get -y install cuda-toolkit-$CUDA_VERSION
fi

echo ""
echo "=== Step 4: Set environment variables ==="
# Find CUDA installation
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
elif [ -d "/usr/local/cuda-12.8" ]; then
    export CUDA_HOME=/usr/local/cuda-12.8
elif [ -d "/usr/local/cuda-12.1" ]; then
    export CUDA_HOME=/usr/local/cuda-12.1
elif [ -d "/usr/local/cuda-11.8" ]; then
    export CUDA_HOME=/usr/local/cuda-11.8
else
    echo -e "${RED}✗ CUDA installation not found!${NC}"
    exit 1
fi

export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

echo "CUDA_HOME: $CUDA_HOME"
echo "nvcc location: $(which nvcc)"
nvcc --version | grep "release"

echo ""
echo "=== Step 5: Set PyTorch library path ==="
TORCH_LIB_PATH=$(python -c "import torch; import os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))")
export LD_LIBRARY_PATH=$TORCH_LIB_PATH:$LD_LIBRARY_PATH
echo "PyTorch libraries: $TORCH_LIB_PATH"

echo ""
echo "=== Step 6: Clean previous builds ==="
rm -rf build/ dist/ *.egg-info *.so fused_iris/*.so
echo -e "${GREEN}✓ Cleaned build artifacts${NC}"

echo ""
echo "=== Step 7: Verify project structure and source files ==="
echo "Current directory: $(pwd)"
echo ""
echo "Expected structure (setup.py INSIDE iris/):"
echo "  iris/"
echo "  |-- __init__.py"
echo "  |-- setup.py          <- HERE"
echo "  |-- functional.py"
echo "  |-- optimizer.py"
echo "  |-- fused.py"
echo "  \`-- fused_iris/"
echo "      |-- __init__.py"
echo "      \`-- fused_iris_kernel.cu"
echo ""

# Check for setup.py in current directory
if [ -f "setup.py" ]; then
    echo -e "${GREEN}✓ setup.py found in current directory${NC}"
else
    echo -e "${RED}✗ ERROR: setup.py not found!${NC}"
    echo "Are you running from the iris/ directory?"
    exit 1
fi

# Check for iris package files
if [ -f "__init__.py" ]; then
    echo -e "${GREEN}✓ __init__.py found${NC}"
else
    echo -e "${RED}✗ ERROR: __init__.py not found!${NC}"
    exit 1
fi

# Check for fused_iris subfolder
if [ -d "fused_iris" ]; then
    echo -e "${GREEN}✓ fused_iris/ directory found${NC}"
else
    echo -e "${RED}✗ ERROR: fused_iris/ directory not found!${NC}"
    exit 1
fi

# Check for CUDA kernel source
if [ -f "fused_iris/fused_iris_kernel.cu" ]; then
    echo -e "${GREEN}✓ CUDA kernel source found: fused_iris/fused_iris_kernel.cu${NC}"
    echo "  File size: $(wc -c < fused_iris/fused_iris_kernel.cu) bytes"
    echo "  Lines: $(wc -l < fused_iris/fused_iris_kernel.cu)"
else
    echo -e "${RED}✗ ERROR: fused_iris/fused_iris_kernel.cu not found!${NC}"
    echo "Looking for .cu files..."
    find . -name "*.cu" -type f
    exit 1
fi

# Check for __init__.py in fused_iris
if [ -f "fused_iris/__init__.py" ]; then
    echo -e "${GREEN}✓ fused_iris/__init__.py found${NC}"
else
    echo -e "${YELLOW}! fused_iris/__init__.py not found (creating)${NC}"
    touch fused_iris/__init__.py
fi

echo ""
echo "=== Step 8: Verify setup.py configuration ==="
if grep -q "fused_iris/fused_iris_kernel.cu" setup.py; then
    echo -e "${GREEN}✓ setup.py references correct path: fused_iris/fused_iris_kernel.cu${NC}"
else
    echo -e "${RED}✗ setup.py doesn't reference fused_iris/fused_iris_kernel.cu!${NC}"
    echo "Check your setup.py sources path"
    exit 1
fi

echo ""
echo "=== Step 9: Build CUDA extensions ==="
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export MAX_JOBS=4

echo "Building with:"
echo "  - Ninja: $(command -v ninja &> /dev/null && echo 'Yes' || echo 'No')"
echo "  - CUDA: $CUDA_HOME"
echo "  - Python: $PYTHON_VERSION"
echo "  - Architecture list: $TORCH_CUDA_ARCH_LIST"
echo ""

python setup.py build_ext --inplace --force 2>&1 | tee build.log

if grep -q "error:" build.log; then
    echo -e "${RED}✗ Build failed! Check build.log${NC}"
    tail -50 build.log
    exit 1
fi

echo ""
echo "=== Step 10: Verify .so file was created ==="
SO_FILE=""
if ls fused_iris/fused_iris_kernel*.so 1> /dev/null 2>&1; then
    SO_FILE=$(ls -1 fused_iris/fused_iris_kernel*.so 2>/dev/null | head -1)
    echo -e "${GREEN}✓ Shared library created: $SO_FILE${NC}"
else
    echo -e "${RED}✗ ERROR: No .so file created!${NC}"
    echo "Searching for .so files..."
    find . -name "*.so" -type f
    exit 1
fi

ls -lh "$SO_FILE"

if echo "$SO_FILE" | grep -q "cpython-$PYTHON_SHORT"; then
    echo -e "${GREEN}✓ Python version matches (cpython-$PYTHON_SHORT)${NC}"
else
    echo -e "${YELLOW}! Python version tag may differ${NC}"
fi

echo ""
echo "=== Step 11: Install package ==="
pip install -e . --no-build-isolation 2>&1 | tee install.log

if ! python -c "import iris" 2>/dev/null; then
    echo -e "${RED}✗ Installation failed! Package not importable${NC}"
    exit 1
fi

echo ""
echo "=== Step 12: Test imports and functionality ==="
python test.py

BUILD_STATUS=$?

if [ $BUILD_STATUS -eq 0 ]; then
    echo ""
    echo -e "${GREEN}===============================================================${NC}"
    echo -e "${GREEN}                 Build Complete and Verified!${NC}"
    echo -e "${GREEN}===============================================================${NC}"
    echo ""
    echo -e "${BLUE}Usage:${NC}"
    echo -e "${GREEN}  from iris import IRIS${NC}"
    echo -e "${GREEN}  optimizer = IRIS(model.parameters(), fused=True)${NC}"
    echo ""
    echo "IRIS: InNOvation Variance Adaptive"
else
    echo -e "${RED}===============================================================${NC}"
    echo -e "${RED}               Build completed but tests failed${NC}"
    echo -e "${RED}===============================================================${NC}"
    exit 1
fi