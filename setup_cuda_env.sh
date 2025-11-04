#!/bin/bash
# CUDA Environment Setup for Heimdall GPU Testing
# Run: source setup_cuda_env.sh

# Find CUDA installation
if [ -d "/usr/lib/cuda" ]; then
    CUDA_HOME="/usr/lib/cuda"
elif [ -d "/usr/local/cuda" ]; then
    CUDA_HOME="/usr/local/cuda"
elif [ -d "/usr/local/cuda-12.2" ]; then
    CUDA_HOME="/usr/local/cuda-12.2"
else
    echo "⚠️  CUDA installation not found. Install with:"
    echo "   sudo apt-get install nvidia-cuda-toolkit"
    return 1
fi

echo "Found CUDA at: $CUDA_HOME"

# Set CUDA environment variables
export CUDA_HOME="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# Activate the GPU test virtual environment
source "$(dirname "${BASH_SOURCE[0]}")/activate_gpu_test.sh"

echo ""
echo "✅ CUDA environment configured!"
echo "   CUDA_HOME: $CUDA_HOME"
echo ""
echo "Test GPU acceleration with:"
echo "   python scripts/test_gpu_acceleration.py"
echo "   python scripts/benchmark_gpu_synthetic.py"
