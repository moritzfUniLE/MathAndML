# AVICI Installation Guide

This guide helps you install the AVICI algorithm for the CausalDiscoveryPlatform.

## 🚨 **Important: Why Special Installation?**

AVICI requires specific versions of Python and PyArrow that conflict with the main application environment:

- **Main app**: Python 3.12 + PyArrow 19.x (compatible with bnlearn)
- **AVICI**: Python 3.10 + PyArrow 10.0.1 (has required `plasma` module)

The solution uses a **conda environment** to isolate AVICI dependencies while maintaining seamless integration.

## ⚡ **Quick Installation (Recommended)**

### Option 1: Automated Script

```bash
# From the CausalDiscoveryPlatform directory
./scripts/install_avici.sh
```

This script will:
- ✅ Install Miniconda (if not present)
- ✅ Create isolated Python 3.10 environment
- ✅ Install AVICI with correct dependencies
- ✅ Test the installation
- ✅ Verify integration works

### Option 2: Manual Installation

If you prefer manual control or the script fails:

```bash
# 1. Install Miniconda (if not already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda

# 2. Initialize conda
source $HOME/miniconda/etc/profile.d/conda.sh

# 3. Create environment
conda create -n avici_env python=3.10 -y
conda activate avici_env

# 4. Install AVICI
pip install avici

# 5. Test installation
python -c "import avici; print('AVICI installed successfully!')"
```

## 🖥️ **Platform-Specific Instructions**

### Linux / WSL (Recommended)
- Use the automated script above
- Tested on Ubuntu 20.04+ and WSL2

### macOS
- Replace the wget URL with: `Miniconda3-latest-MacOSX-x86_64.sh`
- Rest of the process is identical

### Windows (Native)
- **Recommended**: Use WSL2 with Linux instructions
- **Alternative**: Manual conda installation from https://docs.conda.io/en/latest/miniconda.html

## 🧪 **Testing the Installation**

After installation, test AVICI integration:

```bash
# From the main environment (not conda)
python test_avici_integration.py
```

Expected output:
```
✅ Algorithm loaded: AVICI
📋 Conda environment available: True
✅ AVICI completed successfully!
🎉 All tests passed!
```

## 🔧 **Troubleshooting**

### Issue: "AVICI (Environment Not Available)"

**Cause**: The conda environment is not properly set up or activated.

**Solution**:
```bash
# Check if environment exists
conda info --envs | grep avici_env

# If missing, reinstall
./scripts/install_avici.sh

# If present, test activation
source $HOME/miniconda/etc/profile.d/conda.sh
conda activate avici_env
python -c "import avici; print('OK')"
```

### Issue: Memory Errors During Execution

**Cause**: Dataset too large for available memory.

**Solutions**:
1. **Reduce max_samples**: Set to 1000-3000 for large datasets
2. **Increase memory_limit_gb**: If you have more RAM available
3. **Use smaller threshold**: Try 0.005 or 0.001 instead of 0.01

### Issue: "ModuleNotFoundError: No module named 'avici'"

**Cause**: Wrong Python environment or installation failed.

**Solution**:
```bash
# Reinstall in correct environment
conda activate avici_env
pip uninstall avici -y
pip install avici

# Test
python -c "import avici; print('Success')"
```

### Issue: JAX/CUDA Warnings

**Cause**: JAX detects GPU but falls back to CPU (normal behavior).

**Solution**: These warnings are harmless. AVICI runs on CPU for memory efficiency.

## 📊 **Usage Guidelines**

### Recommended Parameters by Dataset Size

| Dataset Size | max_samples | memory_limit_gb | threshold |
|--------------|-------------|-----------------|-----------|
| < 1,000 samples | 1000 | 4 | 0.01 |
| 1,000 - 5,000 | 3000 | 6 | 0.01 |
| 5,000 - 10,000 | 5000 | 8 | 0.005 |
| > 10,000 | 5000 | 8 | 0.005 |

### AVICI Model Versions

- **scm-v0**: General-purpose model (recommended)
- **neurips-linear**: Optimized for linear relationships
- **neurips-rff**: Random Fourier features model
- **neurips-grn**: Gene regulatory networks

## 🔄 **Updating AVICI**

To update AVICI to a newer version:

```bash
conda activate avici_env
pip install --upgrade avici
```

## 🗑️ **Uninstalling AVICI**

To completely remove AVICI:

```bash
# Remove conda environment
conda remove --name avici_env --all -y

# Remove any cached models (optional)
rm -rf ~/.cache/avici
rm -rf ./cache
```

## 💡 **Technical Details**

### Architecture Overview

```
Main App (Python 3.12)
├── Web Interface (Flask)
├── Algorithm Registry
└── AVICI Algorithm
    └── Subprocess Communication
        └── Conda Environment (Python 3.10)
            └── AVICI + Dependencies
```

### Subprocess Communication

- **Data Exchange**: NumPy arrays via temporary files
- **Progress Updates**: JSON files for real-time feedback
- **Error Handling**: Comprehensive error capture and forwarding
- **Timeout Protection**: Configurable execution limits

### Memory Management

- **Automatic Sampling**: Large datasets reduced to manageable size
- **JAX Configuration**: CPU-only execution with memory limits
- **Float32 Precision**: Reduces memory usage by 50%
- **Progressive Fallback**: Smaller subsets if memory issues persist

## 📚 **References**

- [AVICI Paper](https://arxiv.org/abs/2105.10438)
- [AVICI GitHub](https://github.com/larslorch/avici)
- [Conda Documentation](https://docs.conda.io/)
- [JAX Documentation](https://jax.readthedocs.io/)

## 🆘 **Getting Help**

If you encounter issues:

1. **Check logs**: Look for error messages in the console output
2. **Test environment**: Run `python test_avici_integration.py`
3. **Verify installation**: Check conda environment with `conda list -n avici_env`
4. **Memory issues**: Reduce `max_samples` parameter
5. **Report bugs**: Include full error messages and system details