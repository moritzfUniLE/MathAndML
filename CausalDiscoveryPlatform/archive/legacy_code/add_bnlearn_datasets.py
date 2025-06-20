#!/usr/bin/env python3
"""
Add bnlearn datasets to the NOTEARS application.
This script downloads and integrates bnlearn datasets into the existing dataset structure.
"""
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path

try:
    import bnlearn as bn
    BNLEARN_AVAILABLE = True
except ImportError:
    BNLEARN_AVAILABLE = False
    print("bnlearn not available. Please install it: pip install bnlearn")
    exit(1)

def create_dataset_directory(dataset_name, base_dir="datasets"):
    """Create directory structure for a dataset."""
    dataset_dir = os.path.join(base_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    return dataset_dir

def save_dataset_info(dataset_dir, info):
    """Save dataset metadata to info.json."""
    info_path = os.path.join(dataset_dir, "info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)

def adjacency_matrix_to_bif(adj_matrix, node_names, dataset_name):
    """Create a simple BIF file from adjacency matrix."""
    bif_content = f"network {dataset_name} {{\n}}\n\n"
    
    # Variable declarations
    for node in node_names:
        bif_content += f"variable {node} {{\n"
        bif_content += f"  type discrete [ 2 ] {{ 0, 1 }};\n"
        bif_content += f"}}\n\n"
    
    # Probability tables
    for i, node in enumerate(node_names):
        parents = [node_names[j] for j in range(len(node_names)) if adj_matrix[j, i] != 0]
        
        if not parents:
            # No parents - simple probability
            bif_content += f"probability ( {node} ) {{\n"
            bif_content += f"  table 0.5, 0.5;\n"
            bif_content += f"}}\n\n"
        else:
            # Has parents - conditional probability
            bif_content += f"probability ( {node} | {', '.join(parents)} ) {{\n"
            # Simple conditional probabilities (this is a simplification)
            num_states = 2 ** len(parents)
            for state in range(num_states):
                bif_content += f"  ({', '.join(['0' if (state >> j) & 1 == 0 else '1' for j in range(len(parents))])}) 0.3, 0.7;\n"
            bif_content += f"}}\n\n"
    
    return bif_content

def add_bnlearn_dataset(dataset_name, n_samples=1000, verbose=0):
    """Add a bnlearn dataset to the NOTEARS application."""
    print(f"Processing bnlearn dataset: {dataset_name}")
    
    try:
        # Load dataset from bnlearn
        df = bn.import_example(data=dataset_name, n=n_samples, verbose=verbose)
        
        # Load DAG structure if available
        try:
            model = bn.import_DAG(dataset_name, verbose=verbose)
            has_structure = True
        except:
            model = None
            has_structure = False
        
        # Create dataset directory
        dataset_dir = create_dataset_directory(f"bnlearn_{dataset_name}")
        
        # Save data CSV
        data_file = os.path.join(dataset_dir, f"bnlearn_{dataset_name}_data.csv")
        df.to_csv(data_file, index=False)
        
        # Create adjacency matrix if we have structure
        adjacency_matrix = None
        if has_structure and 'adjmat' in model:
            adj_df = model['adjmat']
            adjacency_matrix = adj_df.values.tolist()
            node_names = list(adj_df.columns)
            
            # Create BIF file
            bif_content = adjacency_matrix_to_bif(adj_df.values, node_names, dataset_name)
            bif_file = os.path.join(dataset_dir, f"bnlearn_{dataset_name}.bif")
            with open(bif_file, 'w') as f:
                f.write(bif_content)
        else:
            node_names = list(df.columns)
            adjacency_matrix = [[0] * len(node_names) for _ in range(len(node_names))]
        
        # Create info.json
        info = {
            "name": f"bnlearn_{dataset_name}",
            "description": f"bnlearn {dataset_name} dataset - Bayesian network from bnlearn library",
            "type": "discrete",
            "nodes": len(df.columns),
            "variables": list(df.columns),
            "edges": sum(sum(row) for row in adjacency_matrix) if adjacency_matrix else 0,
            "samples": len(df),
            "bif_file": f"bnlearn_{dataset_name}.bif" if has_structure else None,
            "data_file": f"bnlearn_{dataset_name}_data.csv",
            "structure_parsed": has_structure,
            "adjacency_matrix": adjacency_matrix,
            "source": "bnlearn",
            "original_name": dataset_name
        }
        
        save_dataset_info(dataset_dir, info)
        
        print(f"✓ Added {dataset_name}: {len(df.columns)} variables, {len(df)} samples")
        return True
        
    except Exception as e:
        print(f"✗ Failed to add {dataset_name}: {e}")
        return False

def add_continuous_dataset(dataset_name, n_samples=1000):
    """Add a continuous bnlearn dataset."""
    print(f"Processing continuous bnlearn dataset: {dataset_name}")
    
    try:
        # Load dataset from bnlearn
        df = bn.import_example(data=dataset_name, verbose=0)
        
        # Create dataset directory
        dataset_dir = create_dataset_directory(f"bnlearn_{dataset_name}")
        
        # Save data CSV
        data_file = os.path.join(dataset_dir, f"bnlearn_{dataset_name}_data.csv")
        df.to_csv(data_file, index=False)
        
        # Create info.json
        info = {
            "name": f"bnlearn_{dataset_name}",
            "description": f"bnlearn {dataset_name} dataset - Continuous data from bnlearn library",
            "type": "continuous",
            "nodes": len(df.columns),
            "variables": list(df.columns),
            "edges": 0,  # No predefined structure for continuous data
            "samples": len(df),
            "bif_file": None,
            "data_file": f"bnlearn_{dataset_name}_data.csv",
            "structure_parsed": False,
            "adjacency_matrix": [[0] * len(df.columns) for _ in range(len(df.columns))],
            "source": "bnlearn",
            "original_name": dataset_name
        }
        
        save_dataset_info(dataset_dir, info)
        
        print(f"✓ Added {dataset_name}: {len(df.columns)} variables, {len(df)} samples")
        return True
        
    except Exception as e:
        print(f"✗ Failed to add {dataset_name}: {e}")
        return False

def main():
    """Main function to add all bnlearn datasets."""
    if not BNLEARN_AVAILABLE:
        return
    
    print("Adding bnlearn datasets to NOTEARS application...")
    print("=" * 50)
    
    # Discrete datasets with known structures
    discrete_datasets = [
        'sprinkler',
        'asia', 
        'alarm',
        'sachs',
        'water'
    ]
    
    # Continuous datasets
    continuous_datasets = [
        'auto_mpg'
    ]
    
    # Large datasets (smaller sample sizes)
    large_datasets = [
        'andes'  # 223 variables - use smaller sample
    ]
    
    success_count = 0
    total_count = 0
    
    # Add discrete datasets
    for dataset in discrete_datasets:
        total_count += 1
        if add_bnlearn_dataset(dataset, n_samples=1000):
            success_count += 1
    
    # Add continuous datasets
    for dataset in continuous_datasets:
        total_count += 1
        if add_continuous_dataset(dataset):
            success_count += 1
    
    # Add large datasets with smaller samples
    for dataset in large_datasets:
        total_count += 1
        if add_bnlearn_dataset(dataset, n_samples=200):  # Smaller sample for large networks
            success_count += 1
    
    print("=" * 50)
    print(f"Summary: {success_count}/{total_count} datasets added successfully")
    
    if success_count > 0:
        print("\nNew datasets are now available in the NOTEARS web interface!")
        print("Restart the web GUI to see the new datasets.")

if __name__ == "__main__":
    main()