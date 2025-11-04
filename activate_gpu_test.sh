#!/bin/bash
# Helper script to activate the GPU test virtual environment
# Usage: source activate_gpu_test.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/venv_gpu_test"

if [ ! -d "$VENV_PATH" ]; then
    echo "Error: Virtual environment not found at $VENV_PATH"
    return 1
fi

# Activate the virtual environment
. "$VENV_PATH/bin/activate"

echo "GPU test environment activated!"
echo ""
echo "Installed packages:"
echo "  - numpy:     $(python -c 'import numpy; print(numpy.__version__)')"
echo "  - scipy:     $(python -c 'import scipy; print(scipy.__version__)')"
echo "  - structlog: $(python -c 'import structlog; print(structlog.__version__)')"
echo "  - cupy:      $(python -c 'try: import cupy; print(cupy.__version__)
except: print("not available - CPU only")')"
echo ""
echo "Ready to run test scripts:"
echo "  python scripts/test_gpu_acceleration.py"
echo "  python scripts/benchmark_gpu_synthetic.py"
echo ""
echo "To deactivate: deactivate"
