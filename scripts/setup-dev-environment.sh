#!/bin/bash
# Setup development environment for Heimdall on WSL/Linux
# This script installs all required dependencies for local development and testing

set -e

echo "========================================="
echo "Heimdall Development Environment Setup"
echo "========================================="
echo ""

# Check if running on WSL
if grep -qi microsoft /proc/version; then
    echo "✓ Detected WSL environment"
fi

# Update package lists
echo ""
echo "Step 1: Updating package lists..."
sudo apt-get update

# Install Python 3 and pip if not present
echo ""
echo "Step 2: Checking Python installation..."
if ! command -v python3 &> /dev/null; then
    echo "Installing Python 3..."
    sudo apt-get install -y python3 python3-pip python3-venv
else
    PYTHON_VERSION=$(python3 --version)
    echo "✓ $PYTHON_VERSION already installed"
fi

# Install pip if not present
if ! command -v pip3 &> /dev/null; then
    echo "Installing pip..."
    sudo apt-get install -y python3-pip
else
    echo "✓ pip3 already installed"
fi

# Install build dependencies
echo ""
echo "Step 3: Installing build dependencies..."
sudo apt-get install -y \
    build-essential \
    python3-dev \
    libpq-dev \
    curl \
    jq \
    git

# Create virtual environment
echo ""
echo "Step 4: Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✓ Virtual environment created in ./venv"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
echo ""
echo "Step 5: Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Step 6: Upgrading pip..."
pip install --upgrade pip setuptools wheel

# Install common dependencies
echo ""
echo "Step 7: Installing common Python dependencies..."
pip install \
    pytest \
    pytest-cov \
    pytest-asyncio \
    httpx \
    fastapi \
    uvicorn \
    sqlalchemy \
    psycopg2-binary \
    redis \
    numpy \
    pydantic \
    pydantic-settings \
    python-dotenv

# Install dependencies for each service
echo ""
echo "Step 8: Installing service-specific dependencies..."

for service in services/*/; do
    service_name=$(basename "$service")
    if [ -f "$service/requirements.txt" ]; then
        echo "Installing dependencies for $service_name..."
        pip install -r "$service/requirements.txt" || echo "⚠ Some dependencies failed for $service_name (non-critical)"
    fi
done

# Install Docker Compose if not present (for WSL)
echo ""
echo "Step 9: Checking Docker installation..."
if ! command -v docker &> /dev/null; then
    echo "⚠ WARNING: Docker not found. Install Docker Desktop for Windows with WSL integration."
    echo "   Visit: https://docs.docker.com/desktop/wsl/"
else
    echo "✓ Docker already installed"
fi

echo ""
echo "========================================="
echo "✓ Setup completed successfully!"
echo "========================================="
echo ""
echo "To activate the virtual environment in future sessions:"
echo "  source venv/bin/activate"
echo ""
echo "To run tests locally:"
echo "  make test-local"
echo ""
echo "To run tests in Docker containers:"
echo "  make test"
echo ""
