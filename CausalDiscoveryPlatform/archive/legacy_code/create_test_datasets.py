"""
Create simple test datasets for NOTEARS testing.
Generates synthetic data with known ground truth structures.
"""
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

class SimpleDatasetGenerator:
    """Generate simple synthetic datasets with known DAG structures."""
    
    def __init__(self, data_dir: str = "datasets"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
    
    def create_asia_network(self) -> Tuple[np.ndarray, List[str]]:
        """Create the classic Asia Bayesian network structure."""
        nodes = ["asia", "tub", "smoke", "lung", "bronc", "either", "xray", "dysp"]
        n = len(nodes)
        adj_matrix = np.zeros((n, n))
        
        # Define the Asia network structure
        # asia -> tub, smoke -> lung, smoke -> bronc, tub -> either, lung -> either
        # either -> xray, either -> dysp, bronc -> dysp
        node_idx = {node: i for i, node in enumerate(nodes)}
        
        edges = [
            ("asia", "tub"),
            ("smoke", "lung"), 
            ("smoke", "bronc"),
            ("tub", "either"),
            ("lung", "either"),
            ("either", "xray"),
            ("either", "dysp"),
            ("bronc", "dysp")
        ]
        
        for parent, child in edges:
            adj_matrix[node_idx[parent], node_idx[child]] = 1
        
        return adj_matrix, nodes
    
    def create_simple_chain(self) -> Tuple[np.ndarray, List[str]]:
        """Create a simple chain network: A -> B -> C -> D."""
        nodes = ["A", "B", "C", "D"]
        n = len(nodes)
        adj_matrix = np.zeros((n, n))
        
        # A -> B -> C -> D
        for i in range(n-1):
            adj_matrix[i, i+1] = 1
        
        return adj_matrix, nodes
    
    def create_fork_network(self) -> Tuple[np.ndarray, List[str]]:
        """Create a fork network: A -> B, A -> C, A -> D."""
        nodes = ["A", "B", "C", "D"]
        n = len(nodes)
        adj_matrix = np.zeros((n, n))
        
        # A -> B, A -> C, A -> D
        adj_matrix[0, 1] = 1
        adj_matrix[0, 2] = 1
        adj_matrix[0, 3] = 1
        
        return adj_matrix, nodes
    
    def create_collider_network(self) -> Tuple[np.ndarray, List[str]]:
        """Create a collider network: A -> D, B -> D, C -> D."""
        nodes = ["A", "B", "C", "D"]
        n = len(nodes)
        adj_matrix = np.zeros((n, n))
        
        # A -> D, B -> D, C -> D
        adj_matrix[0, 3] = 1
        adj_matrix[1, 3] = 1
        adj_matrix[2, 3] = 1
        
        return adj_matrix, nodes
    
    def create_complex_network(self) -> Tuple[np.ndarray, List[str]]:
        """Create a more complex network with multiple paths."""
        nodes = ["X1", "X2", "X3", "X4", "X5", "X6"]
        n = len(nodes)
        adj_matrix = np.zeros((n, n))
        
        # X1 -> X2, X1 -> X3, X2 -> X4, X3 -> X4, X3 -> X5, X4 -> X6, X5 -> X6
        edges = [
            (0, 1),  # X1 -> X2
            (0, 2),  # X1 -> X3
            (1, 3),  # X2 -> X4
            (2, 3),  # X3 -> X4
            (2, 4),  # X3 -> X5
            (3, 5),  # X4 -> X6
            (4, 5)   # X5 -> X6
        ]
        
        for parent, child in edges:
            adj_matrix[parent, child] = 1
        
        return adj_matrix, nodes
    
    def generate_linear_data(self, adj_matrix: np.ndarray, n_samples: int = 5000, 
                           noise_scale: float = 1.0) -> np.ndarray:
        """Generate synthetic data using linear structural equations."""
        n_vars = adj_matrix.shape[0]
        
        # Topological ordering
        topo_order = self.topological_sort(adj_matrix)
        
        # Generate data
        X = np.zeros((n_samples, n_vars))
        
        for node in topo_order:
            # Find parents
            parents = np.where(adj_matrix[:, node] == 1)[0]
            
            if len(parents) == 0:
                # Root node - sample from standard normal
                X[:, node] = np.random.normal(0, 1, n_samples)
            else:
                # Generate random weights
                weights = np.random.uniform(0.5, 2.0, len(parents))
                weights *= np.random.choice([-1, 1], len(parents))  # Random signs
                
                # Linear combination of parents + noise
                X[:, node] = np.dot(X[:, parents], weights) + np.random.normal(0, noise_scale, n_samples)
        
        return X
    
    def generate_nonlinear_data(self, adj_matrix: np.ndarray, n_samples: int = 5000,
                              noise_scale: float = 1.0) -> np.ndarray:
        """Generate synthetic data using nonlinear structural equations."""
        n_vars = adj_matrix.shape[0]
        
        # Topological ordering
        topo_order = self.topological_sort(adj_matrix)
        
        # Generate data
        X = np.zeros((n_samples, n_vars))
        
        for node in topo_order:
            # Find parents  
            parents = np.where(adj_matrix[:, node] == 1)[0]
            
            if len(parents) == 0:
                # Root node
                X[:, node] = np.random.normal(0, 1, n_samples)
            else:
                # Nonlinear combination of parents
                parent_values = X[:, parents]
                
                if len(parents) == 1:
                    # Single parent - use various nonlinear functions
                    func_choice = np.random.choice(['tanh', 'sin', 'quadratic'])
                    if func_choice == 'tanh':
                        X[:, node] = np.tanh(parent_values[:, 0]) + np.random.normal(0, noise_scale, n_samples)
                    elif func_choice == 'sin':
                        X[:, node] = np.sin(parent_values[:, 0]) + np.random.normal(0, noise_scale, n_samples)
                    else:
                        X[:, node] = parent_values[:, 0]**2 + np.random.normal(0, noise_scale, n_samples)
                else:
                    # Multiple parents - use combinations
                    weights = np.random.uniform(0.5, 1.5, len(parents))
                    linear_combo = np.dot(parent_values, weights)
                    X[:, node] = np.tanh(linear_combo) + np.random.normal(0, noise_scale, n_samples)
        
        return X
    
    def topological_sort(self, adj_matrix: np.ndarray) -> List[int]:
        """Perform topological sort on the adjacency matrix."""
        n = adj_matrix.shape[0]
        in_degree = np.sum(adj_matrix, axis=0)
        queue = [i for i in range(n) if in_degree[i] == 0]
        topo_order = []
        
        while queue:
            node = queue.pop(0)
            topo_order.append(node)
            
            # Update in-degrees
            for neighbor in range(n):
                if adj_matrix[node, neighbor] == 1:
                    in_degree[neighbor] -= 1
                    if in_degree[neighbor] == 0:
                        queue.append(neighbor)
        
        return topo_order
    
    def save_bif_file(self, bif_path: str, adj_matrix: np.ndarray, node_names: List[str], dataset_name: str):
        """Create a BIF file from adjacency matrix."""
        try:
            with open(bif_path, 'w') as f:
                f.write(f"network {dataset_name} {{\n}}\n\n")
                
                # Write variable declarations
                for node in node_names:
                    f.write(f"variable {node} {{\n")
                    f.write("  type discrete [ 2 ] { LOW, HIGH };\n")
                    f.write("}\n\n")
                
                # Write probability distributions
                for i, child in enumerate(node_names):
                    parents = [node_names[j] for j in range(len(node_names)) if adj_matrix[j, i] == 1]
                    
                    if parents:
                        f.write(f"probability ( {child} | {', '.join(parents)} ) {{\n")
                        # Simple probability table
                        n_parent_states = 2 ** len(parents)
                        for _ in range(n_parent_states):
                            f.write("  table 0.3, 0.7;\n")
                        f.write("}\n\n")
                    else:
                        f.write(f"probability ( {child} ) {{\n")
                        f.write("  table 0.5, 0.5;\n")
                        f.write("}\n\n")
            
            print(f"[INFO] Saved BIF file to {bif_path}")
            
        except Exception as e:
            print(f"[ERROR] Failed to save BIF file: {e}")
    
    def create_dataset(self, dataset_name: str, adj_matrix: np.ndarray, node_names: List[str], 
                      data_type: str = "nonlinear", n_samples: int = 5000):
        """Create a complete dataset with BIF file, data, and info."""
        dataset_path = os.path.join(self.data_dir, dataset_name)
        os.makedirs(dataset_path, exist_ok=True)
        
        # Generate data
        if data_type == "linear":
            X = self.generate_linear_data(adj_matrix, n_samples)
        else:
            X = self.generate_nonlinear_data(adj_matrix, n_samples)
        
        # Save data as CSV
        csv_path = os.path.join(dataset_path, f"{dataset_name}_data.csv")
        df = pd.DataFrame(X, columns=node_names)
        df.to_csv(csv_path, index=False)
        print(f"[INFO] Saved data to {csv_path}")
        
        # Save BIF file
        bif_path = os.path.join(dataset_path, f"{dataset_name}.bif")
        self.save_bif_file(bif_path, adj_matrix, node_names, dataset_name)
        
        # Save dataset info
        info = {
            "name": dataset_name,
            "description": f"Synthetic {data_type} dataset",
            "type": data_type,
            "nodes": len(node_names),
            "variables": node_names,
            "edges": int(np.sum(adj_matrix)),
            "samples": n_samples,
            "bif_file": f"{dataset_name}.bif",
            "data_file": f"{dataset_name}_data.csv",
            "structure_parsed": True,
            "adjacency_matrix": adj_matrix.tolist()
        }
        
        info_path = os.path.join(dataset_path, "info.json")
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"[INFO] Dataset {dataset_name} created successfully")
        
        return True
    
    def create_all_datasets(self):
        """Create all test datasets."""
        datasets = [
            ("asia", self.create_asia_network),
            ("chain", self.create_simple_chain),
            ("fork", self.create_fork_network),
            ("collider", self.create_collider_network),
            ("complex", self.create_complex_network)
        ]
        
        results = {}
        
        for name, structure_func in datasets:
            try:
                print(f"\n[INFO] Creating {name} dataset...")
                adj_matrix, node_names = structure_func()
                success = self.create_dataset(name, adj_matrix, node_names, "nonlinear")
                results[name] = success
            except Exception as e:
                print(f"[ERROR] Failed to create {name}: {e}")
                results[name] = False
        
        return results
    
    def list_created_datasets(self):
        """List all created datasets."""
        if not os.path.exists(self.data_dir):
            print("No datasets directory found.")
            return
        
        datasets = [d for d in os.listdir(self.data_dir) 
                   if os.path.isdir(os.path.join(self.data_dir, d))]
        
        if not datasets:
            print("No datasets found.")
            return
        
        print("Created datasets:")
        print("-" * 40)
        
        for dataset in datasets:
            info_path = os.path.join(self.data_dir, dataset, "info.json")
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    info = json.load(f)
                print(f"{dataset:10} | {info['nodes']:2d} nodes | {info['edges']:2d} edges | {info['description']}")


def main():
    """Main function to create test datasets."""
    generator = SimpleDatasetGenerator()
    
    print("Simple Dataset Generator for NOTEARS Testing")
    print("=" * 50)
    
    print("This will create synthetic datasets with known causal structures:")
    print("1. Asia network (8 nodes) - Classic Bayesian network")
    print("2. Chain network (4 nodes) - Simple linear chain")
    print("3. Fork network (4 nodes) - One parent, multiple children")
    print("4. Collider network (4 nodes) - Multiple parents, one child")
    print("5. Complex network (6 nodes) - Multiple paths and colliders")
    
    choice = input("\nCreate all datasets? (y/n): ").strip().lower()
    
    if choice == 'y':
        results = generator.create_all_datasets()
        
        print("\nCreation Results:")
        print("-" * 30)
        successful = [name for name, success in results.items() if success]
        failed = [name for name, success in results.items() if not success]
        
        if successful:
            print(f"✓ Successfully created: {', '.join(successful)}")
        if failed:
            print(f"✗ Failed to create: {', '.join(failed)}")
        
        print(f"\nDatasets saved in: {generator.data_dir}/")
        print("\nYou can now run: python test_datasets.py")
        
        # List created datasets
        print()
        generator.list_created_datasets()
        
    else:
        print("Operation cancelled.")


if __name__ == "__main__":
    main()