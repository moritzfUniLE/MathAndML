#!/usr/bin/env python3
"""
AVICI Runner Script - Executes AVICI in conda environment
This script runs in the conda environment with AVICI installed
"""
import sys
import json
import numpy as np
import traceback
from pathlib import Path

def run_avici(data_file, params_file, output_file, progress_file=None):
    """
    Run AVICI algorithm and save results
    
    Args:
        data_file: Path to .npy file containing input data
        params_file: Path to .json file containing parameters
        output_file: Path to save output adjacency matrix
        progress_file: Optional path to save progress updates
    """
    try:
        # Configure JAX BEFORE importing avici to ensure settings take effect
        import os
        os.environ['JAX_PLATFORM_NAME'] = 'cpu'
        os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '0.7'  # Use max 70% of available memory
        os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # Don't preallocate memory
        os.environ['XLA_PYTHON_CLIENT_ALLOCATOR'] = 'platform'
        
        # Import avici (should be available in conda environment)
        import avici
        import jax
        
        # Additional JAX configuration
        jax.config.update('jax_platform_name', 'cpu')
        
        print(f"[AVICI Runner] JAX configured for CPU with memory management")
        
        # Load input data and parameters
        X = np.load(data_file)
        with open(params_file, 'r') as f:
            params = json.load(f)
        
        print(f"[AVICI Runner] Loaded data shape: {X.shape}")
        print(f"[AVICI Runner] Parameters: {params}")
        
        # Check dataset size and apply memory-safe preprocessing
        n_samples, n_variables = X.shape
        memory_limit_gb = params.get('memory_limit_gb', 8)  # Default 8GB limit
        max_samples = params.get('max_samples', 5000)  # Default max samples
        
        # Estimate memory usage (rough heuristic)
        estimated_memory_gb = (n_samples * n_variables * n_variables * 8) / (1024**3)  # 8 bytes per float64
        
        print(f"[AVICI Runner] Estimated memory usage: {estimated_memory_gb:.2f} GB")
        
        # Apply data reduction if needed
        if n_samples > max_samples:
            print(f"[AVICI Runner] Dataset too large ({n_samples} samples). Sampling {max_samples} samples.")
            # Use random sampling to reduce dataset size
            np.random.seed(42)  # For reproducibility
            sample_indices = np.random.choice(n_samples, max_samples, replace=False)
            sample_indices = np.sort(sample_indices)  # Keep temporal order if relevant
            X = X[sample_indices]
            print(f"[AVICI Runner] Reduced data shape: {X.shape}")
        
        # Extract AVICI parameters
        download_version = params.get('download', 'scm-v0')
        
        # Write initial progress
        if progress_file:
            progress = {
                'status': 'loading_model',
                'message': f'Loading AVICI model: {download_version}',
                'progress': 0.1
            }
            with open(progress_file, 'w') as f:
                json.dump(progress, f)
        
        # Load pre-trained AVICI model
        print(f"[AVICI Runner] Loading model: {download_version}")
        model = avici.load_pretrained(download=download_version)
        
        # Update progress
        if progress_file:
            progress = {
                'status': 'running_inference',
                'message': 'Running AVICI inference...',
                'progress': 0.3
            }
            with open(progress_file, 'w') as f:
                json.dump(progress, f)
        
        # Run AVICI inference with memory management
        print(f"[AVICI Runner] Running inference...")
        
        # For very large datasets, we might need to use batch processing
        # But AVICI typically expects the full dataset, so we'll try direct inference first
        try:
            # Ensure data is float32 to save memory
            X_processed = X.astype(np.float32)
            print(f"[AVICI Runner] Data converted to float32, shape: {X_processed.shape}")
            
            # Run inference
            W = model(x=X_processed)
            
            # Convert result to numpy array with explicit memory management
            if hasattr(W, 'block_until_ready'):
                W.block_until_ready()  # Ensure computation is complete
            
            W = np.array(W, dtype=np.float32)  # Convert to numpy and reduce precision
            
        except Exception as inference_error:
            print(f"[AVICI Runner] Direct inference failed: {inference_error}")
            
            # If direct inference fails due to memory, try with even smaller sample
            if "out of memory" in str(inference_error).lower() or "buffer" in str(inference_error).lower():
                reduced_samples = min(1000, X.shape[0] // 2)
                print(f"[AVICI Runner] Trying with reduced sample size: {reduced_samples}")
                
                sample_indices = np.random.choice(X.shape[0], reduced_samples, replace=False)
                X_small = X[sample_indices].astype(np.float32)
                
                W = model(x=X_small)
                if hasattr(W, 'block_until_ready'):
                    W.block_until_ready()
                W = np.array(W, dtype=np.float32)
                
                print(f"[AVICI Runner] Inference completed with reduced dataset ({reduced_samples} samples)")
            else:
                raise inference_error
        
        # Update progress
        if progress_file:
            progress = {
                'status': 'saving_results',
                'message': 'Saving results...',
                'progress': 0.9
            }
            with open(progress_file, 'w') as f:
                json.dump(progress, f)
        
        # Save results
        np.save(output_file, W)
        
        # Final progress update
        if progress_file:
            progress = {
                'status': 'completed',
                'message': f'AVICI completed. Matrix shape: {W.shape}, Non-zero edges: {np.sum(np.abs(W) > 1e-6)}',
                'progress': 1.0,
                'result_info': {
                    'shape': W.shape,
                    'non_zero_edges': int(np.sum(np.abs(W) > 1e-6)),
                    'max_value': float(np.max(np.abs(W))),
                    'min_value': float(np.min(W))
                }
            }
            with open(progress_file, 'w') as f:
                json.dump(progress, f)
        
        print(f"[AVICI Runner] Success! Output shape: {W.shape}")
        print(f"[AVICI Runner] Non-zero edges: {np.sum(np.abs(W) > 1e-6)}")
        
        return True
        
    except Exception as e:
        error_msg = f"[AVICI Runner] Error: {str(e)}"
        print(error_msg)
        traceback.print_exc()
        
        # Write error to progress file
        if progress_file:
            progress = {
                'status': 'error',
                'message': error_msg,
                'progress': 0.0,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            with open(progress_file, 'w') as f:
                json.dump(progress, f)
        
        return False

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python avici_runner.py <data_file> <params_file> <output_file> [progress_file]")
        sys.exit(1)
    
    data_file = sys.argv[1]
    params_file = sys.argv[2]
    output_file = sys.argv[3]
    progress_file = sys.argv[4] if len(sys.argv) > 4 else None
    
    success = run_avici(data_file, params_file, output_file, progress_file)
    sys.exit(0 if success else 1)