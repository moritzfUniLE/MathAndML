#!/usr/bin/env python3
"""
Dataset Import Utility for NOTEARS
Provides functionality to import and validate datasets from various sources.
"""
import os
import json
import pandas as pd
import numpy as np
from typing import Dict, Optional
import argparse

class DatasetImporter:
    """Utility class for importing datasets into NOTEARS format."""
    
    def __init__(self, datasets_dir: str = "datasets"):
        self.datasets_dir = datasets_dir
        os.makedirs(datasets_dir, exist_ok=True)
    
    def validate_csv_data(self, data: pd.DataFrame) -> Dict:
        """Validate CSV data for NOTEARS compatibility."""
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': [],
            'info': {}
        }
        
        # Check basic requirements
        if data.empty:
            validation_result['valid'] = False
            validation_result['errors'].append('Dataset is empty')
            return validation_result
        
        if len(data.columns) < 2:
            validation_result['valid'] = False
            validation_result['errors'].append('Dataset must have at least 2 variables')
            return validation_result
        
        # Check for numeric data
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
        
        if len(non_numeric_cols) > 0:
            validation_result['errors'].append(f'Non-numeric columns found: {list(non_numeric_cols)}')
            validation_result['valid'] = False
        
        # Check for missing values
        missing_count = data.isnull().sum().sum()
        if missing_count > 0:
            validation_result['warnings'].append(f'Dataset contains {missing_count} missing values')
        
        # Check for sufficient samples
        n_samples, n_vars = data.shape
        if n_samples < n_vars * 10:
            validation_result['warnings'].append(
                f'Small sample size: {n_samples} samples for {n_vars} variables '
                f'(recommended: ≥{n_vars * 10})'
            )
        
        # Check for constant columns
        constant_cols = []
        for col in data.columns:
            if data[col].nunique() <= 1:
                constant_cols.append(col)
        
        if constant_cols:
            validation_result['warnings'].append(f'Constant columns detected: {constant_cols}')
        
        # Check for extreme values
        for col in numeric_cols:
            if data[col].dtype in [np.float64, np.float32]:
                if np.isinf(data[col]).any():
                    validation_result['errors'].append(f'Infinite values found in column: {col}')
                    validation_result['valid'] = False
                
                # Check for very large values that might cause overflow
                max_val = data[col].abs().max()
                if max_val > 1e6:
                    validation_result['warnings'].append(
                        f'Large values in column {col} (max: {max_val:.2e})'
                    )
        
        # Add dataset statistics
        validation_result['info'] = {
            'shape': data.shape,
            'numeric_columns': len(numeric_cols),
            'missing_values': int(missing_count),
            'memory_usage': f"{data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
        }
        
        return validation_result
    
    def list_available_datasets(self):
        """List all available datasets in the datasets directory."""
        if not os.path.exists(self.datasets_dir):
            print("No datasets directory found.")
            return
        
        datasets = [d for d in os.listdir(self.datasets_dir) 
                   if os.path.isdir(os.path.join(self.datasets_dir, d))]
        
        if not datasets:
            print("No datasets found.")
            return
        
        print("Available datasets:")
        print("-" * 70)
        print(f"{'Name':<20} {'Nodes':<6} {'Edges':<6} {'Samples':<8} {'Description'}")
        print("-" * 70)
        
        for dataset in sorted(datasets):
            info_path = os.path.join(self.datasets_dir, dataset, "info.json")
            if os.path.exists(info_path):
                try:
                    with open(info_path, 'r') as f:
                        info = json.load(f)
                    
                    name = info.get('name', dataset)[:20]
                    nodes = info.get('nodes', 0)
                    edges = info.get('edges', 0)
                    samples = info.get('samples', 0)
                    desc = info.get('description', '')[:40]
                    
                    print(f"{name:<20} {nodes:<6} {edges:<6} {samples:<8} {desc}")
                except:
                    print(f"{dataset:<20} {'?':<6} {'?':<6} {'?':<8} Error reading info")

def main():
    """Main CLI function."""
    parser = argparse.ArgumentParser(
        description="NOTEARS Dataset Import Utility",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available datasets
  python dataset_import_utility.py list
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    subparsers.add_parser('list', help='List available datasets')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    importer = DatasetImporter()
    
    if args.command == 'list':
        importer.list_available_datasets()
        return 0
    
    return 0

if __name__ == "__main__":
    exit(main())