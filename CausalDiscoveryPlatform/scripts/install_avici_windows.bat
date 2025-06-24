@echo off
REM AVICI Installation Script for Windows
REM This script helps Windows users install AVICI

echo 🚀 AVICI Installation Script for Windows
echo ==========================================

REM Check if we're in the right directory
if not exist "algorithms\avici_algorithm.py" (
    echo [ERROR] Please run this script from the CausalDiscoveryPlatform root directory
    pause
    exit /b 1
)

echo [INFO] Detected Windows environment

REM Check if conda is available
where conda >nul 2>&1
if %errorlevel% equ 0 (
    echo [SUCCESS] Conda is already installed
    goto :create_env
)

echo [INFO] Conda not found. Please install Miniconda first:
echo.
echo 1. Download Miniconda from: https://docs.conda.io/en/latest/miniconda.html
echo 2. Choose "Miniconda3 Windows 64-bit"
echo 3. Install with default settings
echo 4. Restart this script after installation
echo.
echo Alternative: Use WSL2 (Windows Subsystem for Linux) for easier installation
echo.
pause
exit /b 1

:create_env
echo [INFO] Creating conda environment 'avici_env' with Python 3.10...
conda create -n avici_env python=3.10 -y

if %errorlevel% neq 0 (
    echo [ERROR] Failed to create conda environment
    pause
    exit /b 1
)

echo [INFO] Activating conda environment...
call conda activate avici_env

echo [INFO] Installing AVICI and dependencies...
echo [INFO] This may take several minutes...
pip install avici

if %errorlevel% neq 0 (
    echo [ERROR] Failed to install AVICI
    pause
    exit /b 1
)

echo [INFO] Testing AVICI installation...
python -c "import avici; import numpy as np; print('✅ AVICI import successful'); X = np.random.randn(50, 3); model = avici.load_pretrained(download='scm-v0'); W = model(x=X); print(f'✅ Test complete - output shape: {W.shape}')"

if %errorlevel% equ 0 (
    echo.
    echo [SUCCESS] AVICI installation completed successfully!
    echo.
    echo Next steps:
    echo   1. The AVICI algorithm is now available in the web interface
    echo   2. Start the web app: python launch_gui.py
    echo   3. Navigate to http://localhost:5000
    echo   4. Select 'AVICI' from the algorithm dropdown
    echo.
) else (
    echo [ERROR] AVICI installation test failed!
    echo Please check the error messages above
)

pause