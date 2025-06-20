"""
Updated dataset downloader using Python bnlearn package.
Downloads BIF files and generates synthetic data for NOTEARS testing.
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import bnlearn as bn
    BNLEARN_AVAILABLE = True
except ImportError:
    BNLEARN_AVAILABLE = False
    print("[WARNING] bnlearn package not available. Please install with: pip install bnlearn")

class BNLearnDatasetDownloader:
    """Downloads and processes bnlearn datasets for causal discovery testing."""
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = data_dir
        self.datasets = {
            # Small networks for initial testing
            "asia": {
                "description": "Small Asia network for lung cancer diagnosis",
                "expected_nodes": 8,
                "expected_edges": 8,
                "type": "discrete"
            },
            "sprinkler": {
                "description": "Classic sprinkler example network",
                "expected_nodes": 4,
                "expected_edges": 4,
                "type": "discrete"
            },
            "alarm": {
                "description": "Medical monitoring alarm network",
                "expected_nodes": 37,
                "expected_edges": 46,
                "type": "discrete"
            },
            "andes": {
                "description": "Physics tutoring system network",
                "expected_nodes": 223,
                "expected_edges": 338,
                "type": "discrete"
            },
            "sachs": {
                "description": "Protein signaling network from Sachs et al.",
                "expected_nodes": 11,
                "expected_edges": 17,
                "type": "discrete"
            }
        }
    
    def create_directories(self):
        """Create necessary directories for storing datasets."""
        os.makedirs(self.data_dir, exist_ok=True)
        for dataset_name in self.datasets:
            dataset_path = os.path.join(self.data_dir, dataset_name)
            os.makedirs(dataset_path, exist_ok=True)
    
    def download_dataset_with_bnlearn(self, dataset_name: str) -> bool:
        """Download dataset using Python bnlearn package."""
        if not BNLEARN_AVAILABLE:
            print(f"[ERROR] bnlearn package not available")
            return False
            
        if dataset_name not in self.datasets:
            print(f"[ERROR] Unknown dataset: {dataset_name}")
            return False
        
        dataset_path = os.path.join(self.data_dir, dataset_name)
        
        try:
            print(f"[INFO] Loading {dataset_name} dataset...")
            
            # Load the DAG model
            model = bn.import_DAG(dataset_name)
            
            if model is None:
                print(f"[ERROR] Failed to load {dataset_name} model")
                return False
            
            # Extract adjacency matrix
            adj_matrix = model['adjmat'].values
            node_names = list(model['adjmat'].columns)
            
            print(f"[INFO] Loaded {dataset_name}: {len(node_names)} nodes, {np.sum(adj_matrix)} edges")
            
            # Save BIF file
            bif_path = os.path.join(dataset_path, f"{dataset_name}.bif")
            self.save_bif_file(bif_path, adj_matrix, node_names, dataset_name)
            
            # Generate synthetic data
            print(f"[INFO] Generating synthetic data for {dataset_name}...")
            data = bn.sampling(model, n=5000)
            
            # Convert to numeric data
            numeric_data = pd.DataFrame()
            for col in data.columns:
                if data[col].dtype == 'object' or data[col].dtype.name == 'category':
                    # Convert categorical to numeric
                    numeric_data[col] = pd.Categorical(data[col]).codes
                else:
                    numeric_data[col] = data[col]
            
            # Save CSV data
            csv_path = os.path.join(dataset_path, f"{dataset_name}_data.csv")
            numeric_data.to_csv(csv_path, index=False)
            print(f"[INFO] Saved data to {csv_path}")
            
            # Create dataset info
            self.create_dataset_info(dataset_name, adj_matrix, node_names)
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to process {dataset_name}: {e}")
            return False
    
    def save_bif_file(self, bif_path: str, adj_matrix: np.ndarray, node_names: List[str], dataset_name: str):
        """Create a simplified BIF file from adjacency matrix."""
        try:
            with open(bif_path, 'w') as f:
                f.write(f"network {dataset_name} {{\n}}\n\n")
                
                # Write variable declarations
                for node in node_names:
                    f.write(f"variable {node} {{\n")
                    f.write("  type discrete [ 2 ] { 0, 1 };\n")
                    f.write("}\n\n")
                
                # Write probability distributions
                for i, child in enumerate(node_names):
                    parents = [node_names[j] for j in range(len(node_names)) if adj_matrix[j, i] == 1]
                    
                    if parents:
                        f.write(f"probability ( {child} | {', '.join(parents)} ) {{\n")
                        # Simplified probability table
                        n_states = 2 ** len(parents)
                        for state in range(n_states):
                            f.write("  table 0.5, 0.5;\n")
                        f.write("}\n\n")
                    else:
                        f.write(f"probability ( {child} ) {{\n")
                        f.write("  table 0.5, 0.5;\n")
                        f.write("}\n\n")
            
            print(f"[INFO] Saved BIF file to {bif_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save BIF file: {e}")
    
    def create_dataset_info(self, dataset_name: str, adj_matrix: np.ndarray, node_names: List[str]):
        """Create information file for the dataset."""
        info_path = os.path.join(self.data_dir, dataset_name, "info.json")
        
        info = {
            "name": dataset_name,
            "description": self.datasets[dataset_name]["description"],
            "type": self.datasets[dataset_name]["type"],
            "nodes": len(node_names),
            "variables": node_names,
            "edges": int(np.sum(adj_matrix)),
            "bif_file": f"{dataset_name}.bif",
            "data_file": f"{dataset_name}_data.csv",
            "structure_parsed": True,
            "adjacency_matrix": adj_matrix.tolist()
        }
        
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"[INFO] Dataset info saved to {info_path}")
    
    def download_all_datasets(self, sizes: List[str] = ["small", "medium"]) -> Dict[str, bool]:
        """Download multiple datasets based on size categories."""
        results = {}
        
        for dataset_name, info in self.datasets.items():
            nodes = info["expected_nodes"]
            
            should_download = False
            if "small" in sizes and nodes <= 20:
                should_download = True
            elif "medium" in sizes and 20 < nodes <= 80:
                should_download = True
            elif "large" in sizes and nodes > 80:
                should_download = True
            
            if should_download:
                print(f"\n[INFO] Processing {dataset_name} ({nodes} expected nodes)...")
                results[dataset_name] = self.download_dataset_with_bnlearn(dataset_name)
        
        return results
    
    def list_available_datasets(self):
        """Print information about available datasets."""
        print("Available bnlearn datasets for causal discovery testing:")
        print("=" * 60)
        
        for name, info in self.datasets.items():
            print(f"{name:15} | {info['expected_nodes']:3d} nodes | {info['expected_edges']:3d} edges | {info['description']}")
        
        print("\nRecommendations:")
        print("- Start with small datasets (asia, sprinkler) for initial testing")
        print("- Use medium datasets (alarm) for validation")
        print("- Test on large datasets (andes) for robustness")


def main():
    """Main function to download and process datasets."""
    if not BNLEARN_AVAILABLE:
        print("Error: bnlearn package is required but not installed.")
        print("Please install it with: pip install bnlearn")
        return
    
    downloader = BNLearnDatasetDownloader()
    
    print("BnLearn Dataset Downloader for NOTEARS Testing (v2)")
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
            results = {dataset_name: downloader.download_dataset_with_bnlearn(dataset_name)}
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
        print("Data files are ready for NOTEARS testing!")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()