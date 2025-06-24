"""
AVICI algorithm implementation using subprocess communication.
Integrates AVICI from conda environment with the main web application.
"""
import numpy as np
import subprocess
import tempfile
import json
import os
import time
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from .base_algorithm import BaseAlgorithm


class AVICI(BaseAlgorithm):
    """AVICI causal discovery algorithm using subprocess communication."""
    
    def __init__(self):
        # Check if conda environment is available
        self.conda_env_available = self._check_conda_environment()
        
        status_suffix = "" if self.conda_env_available else " (Environment Not Available)"
        super().__init__(
            name=f"AVICI{status_suffix}",
            description="AVICI (Amortized Variational Inference for Causal Inference) - A neural causal discovery method that uses amortized variational inference to learn causal graphs from observational data."
        )
    
    def _check_conda_environment(self) -> bool:
        """Check if conda environment with AVICI is available."""
        try:
            # Check if conda is available
            conda_cmd = [
                "bash", "-c", 
                "source $HOME/miniconda/etc/profile.d/conda.sh && conda activate avici_env && python -c 'import avici; print(\"AVICI available\")'"
            ]
            result = subprocess.run(conda_cmd, capture_output=True, text=True, timeout=30)
            return result.returncode == 0 and "AVICI available" in result.stdout
        except Exception as e:
            print(f"[AVICI] Conda environment check failed: {e}")
            return False
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for AVICI."""
        return {
            'download': "scm-v0",
            'timeout': 300,
            'max_samples': 5000,
            'memory_limit_gb': 8
        }
    
    def get_parameter_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter definitions for UI generation."""
        return {
            'download': {
                'type': 'select',
                'default': "scm-v0",
                'options': ["scm-v0", "neurips-linear", "neurips-rff", "neurips-grn"],
                'description': 'Pre-trained AVICI model version to use'
            },
            'timeout': {
                'type': 'int',
                'default': 300,
                'min': 60,
                'max': 1800,
                'description': 'Maximum execution time in seconds'
            },
            'max_samples': {
                'type': 'int',
                'default': 5000,
                'min': 100,
                'max': 20000,
                'description': 'Maximum number of samples to use (reduces memory usage for large datasets)'
            },
            'memory_limit_gb': {
                'type': 'int',
                'default': 8,
                'min': 2,
                'max': 32,
                'description': 'Memory limit in GB (helps with resource management)'
            }
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate AVICI parameters."""
        try:
            # Check required parameters exist
            required_params = ['download', 'timeout', 'max_samples', 'memory_limit_gb']
            for param in required_params:
                if param not in params:
                    return False
            
            # Validate parameter ranges
            if params['download'] not in ["scm-v0", "neurips-linear", "neurips-rff", "neurips-grn"]:
                return False
            
            # Validate timeout
            if not isinstance(params['timeout'], int) or params['timeout'] < 60 or params['timeout'] > 1800:
                return False
            
            # Validate max_samples
            if not isinstance(params['max_samples'], int) or params['max_samples'] < 100 or params['max_samples'] > 20000:
                return False
            
            # Validate memory_limit_gb
            if not isinstance(params['memory_limit_gb'], int) or params['memory_limit_gb'] < 2 or params['memory_limit_gb'] > 32:
                return False
            
            return True
        except (KeyError, TypeError, ValueError):
            return False
    
    def prepare_data(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Prepare data for AVICI algorithm."""
        # Default preprocessing - modify as needed:
        
        X_processed = X.astype(np.float64)
        
        # Example preprocessing steps (modify as needed):
        # 1. Center the data
        #X_processed = X_processed - np.mean(X_processed, axis=0, keepdims=True)
        
        # 2. Standardize if needed (uncomment if required)
        # X_processed = X_processed / np.std(X_processed, axis=0, keepdims=True)
        
        # 3. Handle any NaN values
        if np.any(np.isnan(X_processed)):
            print("[WARNING] NaN values detected in data, replacing with column means")
            col_means = np.nanmean(X_processed, axis=0)
            for j in range(X_processed.shape[1]):
                X_processed[np.isnan(X_processed[:, j]), j] = col_means[j]
        
        return X_processed
    
    def run(self, X: np.ndarray, params: Dict[str, Any], 
            progress_callback: Optional[Callable] = None) -> np.ndarray:
        """
        Run AVICI algorithm using subprocess communication.
        
        Args:
            X: Data matrix of shape (n_samples, n_variables)
            params: Algorithm parameters
            progress_callback: Optional callback for progress updates
                             Signature: callback(iteration, metric_value, additional_info)
            
        Returns:
            Adjacency matrix of shape (n_variables, n_variables)
        """
        if not self.validate_parameters(params):
            raise ValueError("Invalid parameters for AVICI algorithm")
        
        if not self.conda_env_available:
            raise RuntimeError("AVICI conda environment is not available. Please ensure conda environment 'avici_env' is properly set up with AVICI installed.")
        
        # Extract parameters
        avici_version = params['download']
        timeout = params.get('timeout', 300)
        max_samples = params.get('max_samples', 5000)
        memory_limit_gb = params.get('memory_limit_gb', 8)
        
        n_samples, n_variables = X.shape
        
        print(f"[INFO] Starting AVICI algorithm with {n_samples} samples, {n_variables} variables")
        print(f"[INFO] Parameters: version={avici_version}, timeout={timeout}s, max_samples={max_samples}, memory_limit={memory_limit_gb}GB")
        
        # Estimate memory requirements
        estimated_memory_gb = (n_samples * n_variables * n_variables * 8) / (1024**3)
        print(f"[INFO] Estimated memory requirement: {estimated_memory_gb:.2f} GB")
        
        if n_samples > max_samples:
            print(f"[INFO] Dataset will be reduced from {n_samples} to {max_samples} samples to manage memory usage")
        
        if estimated_memory_gb > memory_limit_gb:
            print(f"[WARNING] Estimated memory ({estimated_memory_gb:.2f} GB) exceeds limit ({memory_limit_gb} GB)")
            print(f"[WARNING] Consider reducing max_samples or increasing memory_limit_gb")
        
        # Create temporary files for communication
        with tempfile.TemporaryDirectory(prefix="avici_run_") as temp_dir:
            temp_path = Path(temp_dir)
            
            # File paths
            data_file = temp_path / "input_data.npy"
            params_file = temp_path / "params.json"
            output_file = temp_path / "output.npy"
            progress_file = temp_path / "progress.json"
            runner_script = Path(__file__).parent / "avici_runner.py"
            
            try:
                # Save input data and parameters
                np.save(data_file, X)
                with open(params_file, 'w') as f:
                    json.dump(params, f)
                
                # Initial progress callback
                if progress_callback:
                    progress_callback(0, 0.0, {"status": "initializing", "message": "Starting AVICI subprocess..."})
                
                # Prepare conda command
                conda_cmd = [
                    "bash", "-c", 
                    f"source $HOME/miniconda/etc/profile.d/conda.sh && "
                    f"conda activate avici_env && "
                    f"python {runner_script} {data_file} {params_file} {output_file} {progress_file}"
                ]
                
                # Start subprocess
                print(f"[INFO] Executing AVICI in conda environment...")
                process = subprocess.Popen(
                    conda_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # Monitor progress
                start_time = time.time()
                last_progress = 0.0
                
                while process.poll() is None:
                    # Check for timeout
                    if time.time() - start_time > timeout:
                        process.terminate()
                        try:
                            process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            process.kill()
                        raise TimeoutError(f"AVICI execution exceeded timeout of {timeout} seconds")
                    
                    # Read progress if available
                    if progress_file.exists() and progress_callback:
                        try:
                            with open(progress_file, 'r') as f:
                                progress_data = json.load(f)
                            
                            current_progress = progress_data.get('progress', last_progress)
                            if current_progress > last_progress:
                                progress_callback(
                                    int(current_progress * 100),
                                    current_progress,
                                    {
                                        "status": progress_data.get('status', 'running'),
                                        "message": progress_data.get('message', 'Processing...'),
                                        "result_info": progress_data.get('result_info', {})
                                    }
                                )
                                last_progress = current_progress
                        except (json.JSONDecodeError, FileNotFoundError):
                            pass
                    
                    time.sleep(0.5)  # Check every 500ms
                
                # Get final result
                stdout, stderr = process.communicate()
                
                if process.returncode != 0:
                    error_msg = f"AVICI subprocess failed with return code {process.returncode}"
                    if stderr:
                        error_msg += f"\nStderr: {stderr}"
                    if stdout:
                        error_msg += f"\nStdout: {stdout}"
                    raise RuntimeError(error_msg)
                
                # Load results
                if not output_file.exists():
                    raise RuntimeError("AVICI subprocess completed but no output file was generated")
                
                W = np.load(output_file)
                
                # Final progress update
                if progress_callback:
                    progress_callback(
                        100, 1.0,
                        {
                            "status": "completed",
                            "message": f"AVICI completed successfully. Matrix shape: {W.shape}",
                            "result_info": {
                                "shape": W.shape,
                                "non_zero_edges": int(np.sum(np.abs(W) > 1e-6)),
                                "execution_time": time.time() - start_time
                            }
                        }
                    )
                
                print(f"[INFO] AVICI completed successfully!")
                print(f"[INFO] Output matrix shape: {W.shape}")
                print(f"[INFO] Non-zero edges: {np.sum(np.abs(W) > 1e-6)}")
                print(f"[INFO] Execution time: {time.time() - start_time:.2f}s")
                
                return W
                
            except Exception as e:
                if progress_callback:
                    progress_callback(
                        0, 0.0,
                        {
                            "status": "error",
                            "message": f"AVICI failed: {str(e)}",
                            "error": str(e)
                        }
                    )
                raise
    
    def postprocess_result(self, W: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Post-process the algorithm result."""
        # Apply base class post-processing (thresholding)
        W_processed = super().postprocess_result(W, params)
        
        # TODO: Add any AVICI-specific post-processing here
        # Examples:
        # - Additional sparsity constraints
        # - Symmetry enforcement
        # - Custom thresholding rules
        
        return W_processed


# Auto-register the algorithm
try:
    from .algorithm_registry import algorithm_registry
    algorithm_registry.register_algorithm("avici", AVICI)
    print("[INFO] AVICI algorithm registered successfully")
except ImportError as e:
    print(f"[WARNING] Could not auto-register AVICI algorithm: {e}")
except Exception as e:
    print(f"[ERROR] Failed to register AVICI algorithm: {e}")