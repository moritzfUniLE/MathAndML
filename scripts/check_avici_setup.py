#!/usr/bin/env python3
"""
AVICI Setup Diagnostic Script
Checks if AVICI is properly installed and configured
"""
import sys
import subprocess
import os
from pathlib import Path

def print_header(title):
    print(f"\n{'='*50}")
    print(f"🔍 {title}")
    print(f"{'='*50}")

def print_status(status, message):
    symbols = {"✅": "✅", "❌": "❌", "⚠️": "⚠️", "ℹ️": "ℹ️"}
    print(f"{symbols.get(status, status)} {message}")

def check_file_exists(filepath, description):
    """Check if a file exists and report status."""
    if Path(filepath).exists():
        print_status("✅", f"{description}: Found")
        return True
    else:
        print_status("❌", f"{description}: Missing")
        return False

def run_command(cmd, description, shell=True):
    """Run a command and return success status."""
    try:
        result = subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print_status("✅", f"{description}: OK")
            return True, result.stdout.strip()
        else:
            print_status("❌", f"{description}: Failed")
            if result.stderr:
                print(f"   Error: {result.stderr.strip()}")
            return False, result.stderr.strip()
    except subprocess.TimeoutExpired:
        print_status("❌", f"{description}: Timeout")
        return False, "Command timed out"
    except Exception as e:
        print_status("❌", f"{description}: Error - {e}")
        return False, str(e)

def main():
    print_header("AVICI Setup Diagnostic")
    
    # Check basic files
    print_header("File Structure Check")
    files_ok = True
    files_ok &= check_file_exists("algorithms/avici_algorithm.py", "AVICI Algorithm")
    files_ok &= check_file_exists("algorithms/avici_runner.py", "AVICI Runner")
    files_ok &= check_file_exists("scripts/install_avici.sh", "Installation Script")
    files_ok &= check_file_exists("docs/AVICI_INSTALLATION.md", "Documentation")
    
    # Check main environment
    print_header("Main Environment Check")
    main_ok = True
    
    # Python version
    python_version = sys.version
    print_status("ℹ️", f"Python version: {python_version}")
    
    # Check if AVICI algorithm loads
    try:
        from algorithms import get_algorithm
        avici_alg = get_algorithm("avici")
        print_status("✅", f"AVICI algorithm loaded: {avici_alg.name}")
        conda_available = avici_alg.conda_env_available
        if conda_available:
            print_status("✅", "Conda environment detected: Available")
        else:
            print_status("❌", "Conda environment detected: Not available")
            main_ok = False
    except Exception as e:
        print_status("❌", f"AVICI algorithm loading failed: {e}")
        main_ok = False
        conda_available = False
    
    # Check conda installation
    print_header("Conda Environment Check")
    conda_ok = True
    
    # Check if conda command exists
    conda_success, conda_output = run_command("conda --version", "Conda installation")
    if conda_success:
        print_status("ℹ️", f"Conda version: {conda_output}")
    else:
        conda_ok = False
    
    # Check if avici_env exists
    if conda_success:
        env_success, env_output = run_command(
            "conda info --envs | grep avici_env", 
            "AVICI environment existence"
        )
        if not env_success:
            print_status("❌", "AVICI conda environment not found")
            conda_ok = False
    else:
        conda_ok = False
    
    # Test AVICI in conda environment
    if conda_ok and conda_available:
        print_header("AVICI Functionality Test")
        
        # Test basic import
        avici_import_cmd = (
            "source $HOME/miniconda/etc/profile.d/conda.sh && "
            "conda activate avici_env && "
            "python -c 'import avici; print(\"Import OK\")'"
        )
        import_success, import_output = run_command(
            avici_import_cmd, 
            "AVICI import in conda environment"
        )
        
        if import_success:
            # Test model loading
            model_cmd = (
                "source $HOME/miniconda/etc/profile.d/conda.sh && "
                "conda activate avici_env && "
                "python -c 'import avici; import numpy as np; "
                "X = np.random.randn(20, 3); "
                "model = avici.load_pretrained(download=\"scm-v0\"); "
                "W = model(x=X); "
                "print(f\"Test OK - Output shape: {W.shape}\")'"
            )
            model_success, model_output = run_command(
                model_cmd, 
                "AVICI inference test", 
            )
            if model_success:
                print_status("ℹ️", f"Test result: {model_output}")
    
    # Integration test
    if main_ok and conda_available:
        print_header("Integration Test")
        try:
            import numpy as np
            from algorithms import get_algorithm
            
            avici = get_algorithm("avici")
            params = avici.get_default_parameters()
            params['timeout'] = 60
            params['max_samples'] = 50
            
            print_status("ℹ️", "Running integration test (this may take a minute)...")
            
            X = np.random.randn(50, 3)
            W = avici.run(X, params, None)
            
            print_status("✅", f"Integration test passed - Output shape: {W.shape}")
            print_status("ℹ️", f"Edges found: {np.sum(np.abs(W) > 1e-6)}")
            
        except Exception as e:
            print_status("❌", f"Integration test failed: {e}")
    
    # Summary
    print_header("Summary")
    
    if files_ok and main_ok and conda_ok and conda_available:
        print_status("✅", "AVICI is properly installed and working!")
        print_status("ℹ️", "You can now use AVICI in the web interface")
    else:
        print_status("❌", "AVICI setup has issues")
        print()
        print("🔧 Recommended actions:")
        
        if not files_ok:
            print("   - Ensure you're running from the correct directory")
            print("   - Check if all AVICI files are present")
        
        if not conda_ok:
            print("   - Install Miniconda: https://docs.conda.io/en/latest/miniconda.html")
            print("   - Run: ./scripts/install_avici.sh")
        
        if not conda_available:
            print("   - Run: ./scripts/install_avici.sh")
            print("   - Check conda environment: conda info --envs")
        
        print("   - See docs/AVICI_INSTALLATION.md for detailed instructions")
    
    print()

if __name__ == "__main__":
    main()