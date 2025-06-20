"""
Automatic dataset downloader for bnlearn repository datasets.
Downloads BIF files and converts them to CSV format for NOTEARS testing.
"""
import os
import requests
import numpy as np
from urllib.parse import urljoin
import json
from typing import Dict, List, Tuple, Optional
import re

class BNLearnDatasetDownloader:
    """Downloads and processes bnlearn repository datasets for causal discovery testing."""
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = data_dir
        self.base_url = "https://www.bnlearn.com/bnrepository/"
        self.datasets = {
            # Small networks for initial testing
            "asia": {
                "nodes": 8, "arcs": 8,
                "description": "Small Asia network for lung cancer diagnosis",
                "type": "discrete"
            },
            "cancer": {
                "nodes": 5, "arcs": 4,
                "description": "Simple cancer diagnosis network",
                "type": "discrete"
            },
            "earthquake": {
                "nodes": 5, "arcs": 4,
                "description": "Simple earthquake alarm network",
                "type": "discrete"
            },
            "sachs": {
                "nodes": 11, "arcs": 17,
                "description": "Protein signaling network from Sachs et al.",
                "type": "discrete"
            },
            # Medium networks
            "child": {
                "nodes": 20, "arcs": 25,
                "description": "Child birth defects network",
                "type": "discrete"
            },
            "alarm": {
                "nodes": 37, "arcs": 46,
                "description": "Medical monitoring alarm network",
                "type": "discrete"
            },
            "barley": {
                "nodes": 48, "arcs": 84,
                "description": "Crop yield analysis network",
                "type": "discrete"
            },
            # Large networks
            "hailfinder": {
                "nodes": 56, "arcs": 66,
                "description": "Weather forecasting network",
                "type": "discrete"
            },
            "hepar2": {
                "nodes": 70, "arcs": 123,
                "description": "Hepatitis diagnosis network",
                "type": "discrete"
            },
            "win95pts": {
                "nodes": 76, "arcs": 112,
                "description": "Windows 95 troubleshooting network",
                "type": "discrete"
            },
            "andes": {
                "nodes": 223, "arcs": 338,
                "description": "Physics tutoring system network",
                "type": "discrete"
            }
        }
        
    def create_directories(self):
        """Create necessary directories for storing datasets."""
        os.makedirs(self.data_dir, exist_ok=True)
        for dataset_name in self.datasets:
            dataset_path = os.path.join(self.data_dir, dataset_name)
            os.makedirs(dataset_path, exist_ok=True)
    
    def download_bif_file(self, dataset_name: str) -> bool:
        """Download BIF file for a specific dataset."""
        if dataset_name not in self.datasets:
            print(f"[ERROR] Unknown dataset: {dataset_name}")
            return False
            
        bif_url = urljoin(self.base_url, f"discrete-small/{dataset_name}/{dataset_name}.bif")
        if self.datasets[dataset_name]["nodes"] > 50:
            bif_url = urljoin(self.base_url, f"discrete-large/{dataset_name}/{dataset_name}.bif")
        elif self.datasets[dataset_name]["nodes"] > 20:
            bif_url = urljoin(self.base_url, f"discrete-medium/{dataset_name}/{dataset_name}.bif")
        
        bif_path = os.path.join(self.data_dir, dataset_name, f"{dataset_name}.bif")
        
        try:
            print(f"[INFO] Downloading {dataset_name}.bif from {bif_url}")
            response = requests.get(bif_url, timeout=30)
            response.raise_for_status()
            
            with open(bif_path, 'w', encoding='utf-8') as f:
                f.write(response.text)
            
            print(f"[INFO] Successfully downloaded {bif_path}")
            return True
            
        except requests.RequestException as e:
            print(f"[ERROR] Failed to download {dataset_name}: {e}")
            return False
    
    def parse_bif_structure(self, bif_path: str) -> Tuple[Optional[np.ndarray], Optional[List[str]], bool]:
        """Parse BIF file to extract DAG structure."""
        try:
            with open(bif_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Extract variable names
            vars_pattern = r'variable\s+(\w+)\s*{'
            variables = re.findall(vars_pattern, text)
            
            if not variables:
                print(f"[ERROR] No variables found in {bif_path}")
                return None, None, False
            
            n = len(variables)
            var_idx = {v: i for i, v in enumerate(variables)}
            adj_matrix = np.zeros((n, n))
            
            # Extract parent-child relationships
            prob_pattern = r'probability\s*\(\s*(\w+)\s*(?:\|\s*([^)]+))?\s*\)'
            matches = re.findall(prob_pattern, text)
            
            for match in matches:
                child = match[0]
                parents_str = match[1]
                
                if parents_str:  # Has parents
                    parents = [p.strip() for p in parents_str.split(',')]
                    for parent in parents:
                        if parent in var_idx and child in var_idx:
                            adj_matrix[var_idx[parent], var_idx[child]] = 1
            
            return adj_matrix, variables, True
            
        except Exception as e:
            print(f"[ERROR] Failed to parse BIF file {bif_path}: {e}")
            return None, None, False
    
    def generate_synthetic_data(self, dataset_name: str, n_samples: int = 5000) -> bool:
        """Generate synthetic data from BIF structure using R bnlearn if available."""
        bif_path = os.path.join(self.data_dir, dataset_name, f"{dataset_name}.bif")
        csv_path = os.path.join(self.data_dir, dataset_name, f"{dataset_name}_data.csv")
        
        if not os.path.exists(bif_path):
            print(f"[ERROR] BIF file not found: {bif_path}")
            return False
        
        try:
            # Try to use R bnlearn to generate data
            r_script = f"""
library(bnlearn)
bn <- read.bif("{bif_path}")
data <- rbn(bn, {n_samples})
write.csv(data, "{csv_path}", row.names=FALSE)
"""
            
            r_script_path = os.path.join(self.data_dir, dataset_name, "generate_data.R")
            with open(r_script_path, 'w') as f:
                f.write(r_script)
            
            print(f"[INFO] R script created at {r_script_path}")
            print(f"[INFO] Run: Rscript {r_script_path} to generate data")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to create R script for {dataset_name}: {e}")
            return False
    
    def create_dataset_info(self, dataset_name: str):
        """Create information file for the dataset."""
        info_path = os.path.join(self.data_dir, dataset_name, "info.json")
        bif_path = os.path.join(self.data_dir, dataset_name, f"{dataset_name}.bif")
        
        # Parse BIF to get actual structure
        adj_matrix, variables, success = self.parse_bif_structure(bif_path)
        
        info = {
            "name": dataset_name,
            "description": self.datasets[dataset_name]["description"],
            "type": self.datasets[dataset_name]["type"],
            "nodes": len(variables) if variables else self.datasets[dataset_name]["nodes"],
            "variables": variables,
            "edges": int(np.sum(adj_matrix)) if adj_matrix is not None else self.datasets[dataset_name]["arcs"],
            "bif_file": f"{dataset_name}.bif",
            "data_file": f"{dataset_name}_data.csv",
            "structure_parsed": success
        }
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"[INFO] Dataset info saved to {info_path}")
    
    def download_dataset(self, dataset_name: str, generate_data: bool = True) -> bool:
        """Download and process a single dataset."""
        if self.download_bif_file(dataset_name):
            self.create_dataset_info(dataset_name)
            if generate_data:
                self.generate_synthetic_data(dataset_name)
            return True
        return False
    
    def download_all_datasets(self, sizes: List[str] = ["small", "medium"]) -> Dict[str, bool]:
        """Download multiple datasets based on size categories."""
        results = {}
        
        for dataset_name, info in self.datasets.items():
            nodes = info["nodes"]
            
            should_download = False
            if "small" in sizes and nodes <= 20:
                should_download = True
            elif "medium" in sizes and 20 < nodes <= 80:
                should_download = True
            elif "large" in sizes and nodes > 80:
                should_download = True
            
            if should_download:
                print(f"\n[INFO] Processing {dataset_name} ({nodes} nodes)...")
                results[dataset_name] = self.download_dataset(dataset_name)
        
        return results
    
    def list_available_datasets(self):
        """Print information about available datasets."""
        print("Available bnlearn datasets for causal discovery testing:")
        print("=" * 60)
        
        for name, info in self.datasets.items():
            print(f"{name:15} | {info['nodes']:3d} nodes | {info['arcs']:3d} arcs | {info['description']}")
        
        print("\nRecommendations:")
        print("- Start with small datasets (asia, cancer, earthquake) for initial testing")
        print("- Use medium datasets (child, alarm) for validation")
        print("- Test on large datasets (hepar2, andes) for robustness")


def main():
    """Main function to download and process datasets."""
    downloader = BNLearnDatasetDownloader()
    
    print("BnLearn Dataset Downloader for NOTEARS Testing")
    print("=" * 50)
    
    # Create directories
    downloader.create_directories()
    
    # List available datasets
    downloader.list_available_datasets()
    
    print("\nChoose an option:")
    print("1. Download small datasets (recommended for initial testing)")
    print("2. Download medium datasets")
    print("3. Download all datasets")
    print("4. Download specific dataset")
    print("5. Exit")
    
    try:
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == "1":
            results = downloader.download_all_datasets(["small"])
        elif choice == "2":
            results = downloader.download_all_datasets(["medium"])
        elif choice == "3":
            results = downloader.download_all_datasets(["small", "medium", "large"])
        elif choice == "4":
            dataset_name = input("Enter dataset name: ").strip().lower()
            results = {dataset_name: downloader.download_dataset(dataset_name)}
        elif choice == "5":
            print("Exiting...")
            return
        else:
            print("Invalid choice")
            return
        
        # Print results summary
        print("\nDownload Results:")
        print("-" * 30)
        successful = [name for name, success in results.items() if success]
        failed = [name for name, success in results.items() if not success]
        
        if successful:
            print(f"✓ Successfully downloaded: {', '.join(successful)}")
        if failed:
            print(f"✗ Failed to download: {', '.join(failed)}")
        
        print(f"\nDatasets saved in: {downloader.data_dir}/")
        print("\nNote: To generate CSV data files, you need R with bnlearn package installed.")
        print("Run the generated R scripts in each dataset directory.")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()