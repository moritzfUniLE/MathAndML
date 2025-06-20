#!/usr/bin/env python3
"""
Script to integrate preprocessed SACHS datasets from NoTearsData directory.
"""
import os
import shutil
import json
import pandas as pd

def get_sachs_variable_names():
    """Get SACHS variable names from existing BIF file."""
    bif_path = "datasets/SACHSconglomeratedData/SACHSconglomeratedData.bif"
    if not os.path.exists(bif_path):
        # Fallback variable names based on SACHS network
        return ["Akt", "Erk", "Jnk", "Mek", "P38", "PIP2", "PIP3", "PKA", "PKC", "Plcg", "Raf"]
    
    variables = []
    with open(bif_path, 'r') as f:
        for line in f:
            if line.startswith('variable '):
                var_name = line.split()[1]
                variables.append(var_name)
    
    return variables

def integrate_sachs_preprocessed():
    """Integrate all preprocessed SACHS datasets."""
    
    # Define paths
    source_dir = "../NoTearsData/DataPreprocessed"
    datasets_dir = "datasets"
    
    # Get variable names
    variables = get_sachs_variable_names()
    print(f"Using variables: {variables}")
    
    # Define preprocessing variants
    variants = {
        "MainDataPreprocessedZscore.csv": {
            "name": "sachs_zscore",
            "description": "SACHS protein network data preprocessed with Z-score normalization"
        },
        "MainDataPreprocessedIQR.csv": {
            "name": "sachs_iqr",
            "description": "SACHS protein network data preprocessed with IQR outlier removal"
        },
        "MainDataPreprocessedIsolationForest.csv": {
            "name": "sachs_isolation_forest",
            "description": "SACHS protein network data preprocessed with Isolation Forest outlier detection"
        },
        "MainDataPreprocessedLogTransform.csv": {
            "name": "sachs_log_transform",
            "description": "SACHS protein network data preprocessed with log transformation"
        }
    }
    
    # Create datasets directory if it doesn't exist
    os.makedirs(datasets_dir, exist_ok=True)
    
    # Process each variant
    successful = []
    failed = []
    
    for source_file, info in variants.items():
        try:
            print(f"\nProcessing {info['name']}...")
            
            # Create dataset directory
            dataset_dir = os.path.join(datasets_dir, info['name'])
            os.makedirs(dataset_dir, exist_ok=True)
            
            # Copy and process data file
            source_path = os.path.join(source_dir, source_file)
            if not os.path.exists(source_path):
                print(f"ERROR: Source file not found: {source_path}")
                failed.append(info['name'])
                continue
            
            # Read data and add headers
            df = pd.read_csv(source_path, header=None)
            df.columns = variables[:len(df.columns)]  # Use only needed variable names
            
            # Save with headers
            data_dest = os.path.join(dataset_dir, f"{info['name']}_data.csv")
            df.to_csv(data_dest, index=False)
            print(f"✓ Saved data file with headers to {data_dest}")
            
            # Copy BIF file from existing SACHS dataset
            source_bif = "datasets/SACHSconglomeratedData/SACHSconglomeratedData.bif"
            if os.path.exists(source_bif):
                bif_dest = os.path.join(dataset_dir, f"{info['name']}.bif")
                shutil.copy2(source_bif, bif_dest)
                print(f"✓ Copied BIF file to {bif_dest}")
            
            # Analyze dataset
            n_samples, n_vars = df.shape
            
            # Count edges from BIF file (simplified)
            edge_count = 17  # Known SACHS network edge count
            
            # Create info.json
            dataset_info = {
                "name": info['name'],
                "description": info['description'],
                "type": "continuous",
                "nodes": n_vars,
                "variables": df.columns.tolist(),
                "edges": edge_count,
                "samples": n_samples,
                "bif_file": f"{info['name']}.bif",
                "data_file": f"{info['name']}_data.csv",
                "structure_parsed": True,
                "source": "SACHS protein signaling network (Sachs et al., 2005)",
                "domain": "Protein signaling pathways",
                "preprocessing": source_file.replace("MainDataPreprocessed", "").replace(".csv", "")
            }
            
            info_path = os.path.join(dataset_dir, "info.json")
            with open(info_path, 'w') as f:
                json.dump(dataset_info, f, indent=2)
            
            print(f"✓ Created dataset info file")
            print(f"  - Variables: {n_vars}")
            print(f"  - Samples: {n_samples}")
            print(f"  - Preprocessing: {dataset_info['preprocessing']}")
            
            successful.append(info['name'])
            
        except Exception as e:
            print(f"ERROR: Failed to process {info['name']}: {e}")
            failed.append(info['name'])
    
    return successful, failed

def main():
    """Main function."""
    print("SACHS Preprocessed Datasets Integration")
    print("=" * 50)
    
    successful, failed = integrate_sachs_preprocessed()
    
    print(f"\n\nIntegration Results:")
    print("=" * 30)
    
    if successful:
        print(f"✅ Successfully integrated: {len(successful)} datasets")
        for name in successful:
            print(f"  - {name}")
    
    if failed:
        print(f"❌ Failed: {len(failed)} datasets")
        for name in failed:
            print(f"  - {name}")
    
    if successful:
        print(f"\nYou can now:")
        print("1. Launch the web GUI: python notears_web_gui.py")
        print("2. Select any of the SACHS preprocessed datasets")
        print("3. Compare results across different preprocessing methods")
    
    return 0 if not failed else 1

if __name__ == "__main__":
    exit(main())