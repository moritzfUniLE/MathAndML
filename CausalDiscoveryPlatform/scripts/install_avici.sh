#!/bin/bash
"""
AVICI Installation Script for CausalDiscoveryPlatform
This script automatically sets up AVICI with proper conda environment
"""

set -e  # Exit on any error

echo "🚀 AVICI Installation Script"
echo "==============================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if script is run from correct directory
if [ ! -f "algorithms/avici_algorithm.py" ]; then
    print_error "Please run this script from the CausalDiscoveryPlatform root directory"
    exit 1
fi

# Detect platform
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    PLATFORM="Linux"
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    PLATFORM="macOS"
    MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh"
elif [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    PLATFORM="Windows"
    print_error "Please use WSL (Windows Subsystem for Linux) or install manually on Windows"
    print_error "See: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
else
    print_error "Unsupported platform: $OSTYPE"
    exit 1
fi

print_status "Detected platform: $PLATFORM"

# Check if conda is already available
if command -v conda &> /dev/null; then
    print_success "Conda is already installed"
    CONDA_PATH=$(which conda)
    CONDA_BASE=$(dirname $(dirname $CONDA_PATH))
else
    print_status "Installing Miniconda..."
    
    # Download Miniconda
    print_status "Downloading Miniconda for $PLATFORM..."
    wget -q --show-progress $MINICONDA_URL -O miniconda.sh
    
    # Install Miniconda
    print_status "Installing Miniconda to $HOME/miniconda..."
    bash miniconda.sh -b -p $HOME/miniconda
    
    # Set conda path
    CONDA_BASE="$HOME/miniconda"
    
    # Clean up installer
    rm miniconda.sh
    
    print_success "Miniconda installed successfully"
fi

# Initialize conda for bash
print_status "Initializing conda..."
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
else
    print_error "Could not find conda.sh at $CONDA_BASE/etc/profile.d/conda.sh"
    exit 1
fi

# Check if avici_env already exists
if conda info --envs | grep -q "avici_env"; then
    print_warning "Environment 'avici_env' already exists"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_status "Removing existing environment..."
        conda remove --name avici_env --all -y
    else
        print_status "Using existing environment..."
        conda activate avici_env
        
        # Check if AVICI is already installed
        if python -c "import avici" 2>/dev/null; then
            print_success "AVICI is already installed and working!"
            
            # Test AVICI
            print_status "Testing AVICI installation..."
            python -c "
import avici
import numpy as np
print('✅ AVICI version check passed')
# Quick test
X = np.random.randn(50, 3)
model = avici.load_pretrained(download='scm-v0')
W = model(x=X)
print(f'✅ AVICI inference test passed - output shape: {W.shape}')
"
            print_success "AVICI installation verified!"
            exit 0
        fi
    fi
fi

# Create conda environment with Python 3.10
print_status "Creating conda environment 'avici_env' with Python 3.10..."
conda create -n avici_env python=3.10 -y

# Activate environment
print_status "Activating conda environment..."
conda activate avici_env

# Verify Python version
PYTHON_VERSION=$(python --version)
print_status "Python version: $PYTHON_VERSION"

# Install AVICI
print_status "Installing AVICI and dependencies..."
print_status "This may take several minutes as it downloads TensorFlow, JAX, and other dependencies..."

pip install avici

# Test installation
print_status "Testing AVICI installation..."
python -c "
import avici
import numpy as np
print('✅ AVICI import successful')

# Test model loading
print('📥 Testing model download and loading...')
model = avici.load_pretrained(download='scm-v0')
print('✅ Model loaded successfully')

# Test inference
print('🧪 Testing inference...')
X = np.random.randn(50, 3)
W = model(x=X)
print(f'✅ Inference successful - output shape: {W.shape}')
print(f'✅ Non-zero elements: {int(np.sum(np.abs(W) > 1e-6))}')
"

if [ $? -eq 0 ]; then
    print_success "AVICI installation completed successfully!"
    echo ""
    print_status "Next steps:"
    echo "  1. The AVICI algorithm is now available in the web interface"
    echo "  2. Start the web app: python launch_gui.py"
    echo "  3. Navigate to http://localhost:5000"
    echo "  4. Select 'AVICI' from the algorithm dropdown"
    echo ""
    print_status "Manual activation (if needed):"
    echo "  source $CONDA_BASE/etc/profile.d/conda.sh"
    echo "  conda activate avici_env"
    echo ""
else
    print_error "AVICI installation test failed!"
    print_error "Please check the error messages above and try again"
    exit 1
fi