"""
Test NOTEARS nonlinear algorithm on multiple bnlearn datasets.
Compares results against ground truth and provides comprehensive evaluation.
"""
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from typing import Dict, List, Tuple, Optional
import time
import warnings
warnings.filterwarnings('ignore')

from notears_core import (
    NotearsMLP, 
    notears_nonlinear, 
    load_ground_truth_from_bif, 
    compute_metrics,
    apply_threshold
)

class NOTEARSDatasetTester:
    """Test NOTEARS algorithm on multiple bnlearn datasets."""
    
    def __init__(self, datasets_dir: str = "datasets", results_dir: str = "test_results"):
        self.datasets_dir = datasets_dir
        self.results_dir = results_dir
        self.results = {}
        os.makedirs(results_dir, exist_ok=True)
    
    def load_dataset_info(self, dataset_name: str) -> Optional[Dict]:
        """Load dataset information from info.json."""
        info_path = os.path.join(self.datasets_dir, dataset_name, "info.json")
        if not os.path.exists(info_path):
            return None
        
        with open(info_path, 'r') as f:
            return json.load(f)
    
    def load_data_and_ground_truth(self, dataset_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[List[str]]]:
        """Load CSV data and ground truth BIF for a dataset."""
        dataset_path = os.path.join(self.datasets_dir, dataset_name)
        
        # Load data
        csv_path = os.path.join(dataset_path, f"{dataset_name}_data.csv")
        if not os.path.exists(csv_path):
            print(f"[WARNING] Data file not found: {csv_path}")
            return None, None, None
        
        try:
            data = pd.read_csv(csv_path)
            # Convert categorical to numeric if needed
            for col in data.columns:
                if data[col].dtype == 'object':
                    data[col] = pd.Categorical(data[col]).codes
            X = data.values.astype(np.float32)
            print(f"[INFO] Loaded data: {X.shape}")
        except Exception as e:
            print(f"[ERROR] Failed to load data from {csv_path}: {e}")
            return None, None, None
        
        # Load ground truth
        bif_path = os.path.join(dataset_path, f"{dataset_name}.bif")
        W_true, node_names, success = load_ground_truth_from_bif(bif_path)
        
        if not success:
            print(f"[ERROR] Failed to load ground truth from {bif_path}")
            return X, None, None
        
        return X, W_true, node_names
    
    def test_dataset(self, dataset_name: str, 
                    lambda1: float = 0.01, 
                    lambda2: float = 0.01,
                    max_iter: int = 100,
                    threshold: float = 0.1) -> Dict:
        """Test NOTEARS on a single dataset."""
        print(f"\n{'='*60}")
        print(f"Testing NOTEARS on {dataset_name.upper()}")
        print(f"{'='*60}")
        
        # Load dataset info
        info = self.load_dataset_info(dataset_name)
        if info is None:
            return {"error": "Failed to load dataset info"}
        
        print(f"Dataset: {info['description']}")
        print(f"Nodes: {info['nodes']}, Expected edges: {info['edges']}")
        
        # Load data and ground truth
        X, W_true, node_names = self.load_data_and_ground_truth(dataset_name)
        if X is None:
            return {"error": "Failed to load data"}
        
        if W_true is None:
            return {"error": "Failed to load ground truth"}
        
        print(f"Data shape: {X.shape}")
        print(f"Ground truth shape: {W_true.shape}")
        
        # Initialize model
        d = X.shape[1]
        model = NotearsMLP(d, m_hidden=min(10, d*2))  # Adaptive hidden size
        
        # Run NOTEARS
        print(f"[INFO] Running NOTEARS with λ1={lambda1}, λ2={lambda2}, max_iter={max_iter}")
        start_time = time.time()
        
        try:
            W_learned = notears_nonlinear(
                model, X, 
                lambda1=lambda1, 
                lambda2=lambda2, 
                max_iter=max_iter
            )
            runtime = time.time() - start_time
            print(f"[INFO] NOTEARS completed in {runtime:.2f} seconds")
            
        except Exception as e:
            return {"error": f"NOTEARS failed: {str(e)}"}
        
        # Compute metrics
        metrics = compute_metrics(W_learned, W_true, thresh=threshold)
        if metrics['error']:
            return {"error": f"Metrics computation failed: {metrics['error']}"}
        
        # Additional analysis
        n_learned_edges = np.sum(apply_threshold(W_learned, threshold) != 0)
        n_true_edges = np.sum(W_true != 0)
        
        result = {
            "dataset": dataset_name,
            "dataset_info": info,
            "runtime_seconds": runtime,
            "learned_edges": int(n_learned_edges),
            "true_edges": int(n_true_edges),
            "parameters": {
                "lambda1": lambda1,
                "lambda2": lambda2, 
                "max_iter": max_iter,
                "threshold": threshold
            },
            "metrics": metrics,
            "adjacency_matrices": {
                "learned": W_learned.tolist(),
                "ground_truth": W_true.tolist()
            },
            "node_names": node_names
        }
        
        # Print results summary
        print(f"\nResults Summary for {dataset_name}:")
        print(f"  Runtime: {runtime:.2f}s")
        print(f"  Edges: {n_learned_edges} learned vs {n_true_edges} true")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
        print(f"  Hamming Distance: {metrics['hamming_distance']}")
        
        return result
    
    def save_adjacency_csv(self, dataset_name: str, W_learned: np.ndarray, W_true: np.ndarray, 
                          node_names: List[str], threshold: float):
        """Save adjacency matrices as CSV files."""
        dataset_result_dir = os.path.join(self.results_dir, dataset_name)
        os.makedirs(dataset_result_dir, exist_ok=True)
        
        # Apply threshold to learned matrix
        W_learned_thresh = apply_threshold(W_learned, threshold)
        
        # Create DataFrames with proper node names
        df_learned = pd.DataFrame(W_learned, index=node_names, columns=node_names)
        df_learned_thresh = pd.DataFrame(W_learned_thresh, index=node_names, columns=node_names)
        df_true = pd.DataFrame(W_true, index=node_names, columns=node_names)
        
        # Save CSV files
        df_learned.to_csv(os.path.join(dataset_result_dir, "adjacency_learned_raw.csv"))
        df_learned_thresh.to_csv(os.path.join(dataset_result_dir, "adjacency_learned_thresholded.csv"))
        df_true.to_csv(os.path.join(dataset_result_dir, "adjacency_ground_truth.csv"))
        
        print(f"[INFO] Adjacency matrices saved as CSV in {dataset_result_dir}/")
    
    def create_graph_from_adjacency(self, adj_matrix: np.ndarray, node_names: List[str], 
                                  pos: Dict = None) -> Tuple[nx.DiGraph, Dict]:
        """Create NetworkX graph from adjacency matrix with consistent layout."""
        G = nx.DiGraph()
        
        # Add nodes with labels
        for i, name in enumerate(node_names):
            G.add_node(i, label=name)
        
        # Add edges
        for i in range(len(node_names)):
            for j in range(len(node_names)):
                if adj_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=adj_matrix[i, j])
        
        # Create consistent layout if not provided
        if pos is None:
            # Use spring layout with fixed seed for consistency
            pos = nx.spring_layout(G, seed=42, k=3, iterations=50)
        
        return G, pos
    
    def visualize_side_by_side_graphs(self, dataset_name: str, W_learned: np.ndarray, 
                                    W_true: np.ndarray, node_names: List[str], threshold: float):
        """Create side-by-side graph visualization comparing learned vs ground truth."""
        W_learned_thresh = apply_threshold(W_learned, threshold)
        
        # Create graphs
        G_true, pos = self.create_graph_from_adjacency(W_true, node_names)
        G_learned, _ = self.create_graph_from_adjacency(W_learned_thresh, node_names, pos)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Ground Truth Graph
        ax1.set_title(f'Ground Truth DAG\n({dataset_name})', fontsize=14, fontweight='bold')
        
        # Draw nodes
        nx.draw_networkx_nodes(G_true, pos, ax=ax1, node_color='lightblue', 
                              node_size=1000, alpha=0.9)
        
        # Draw edges
        nx.draw_networkx_edges(G_true, pos, ax=ax1, edge_color='black', 
                              arrows=True, arrowsize=20, arrowstyle='->', width=2)
        
        # Draw labels
        labels_true = {i: name for i, name in enumerate(node_names)}
        nx.draw_networkx_labels(G_true, pos, labels_true, ax=ax1, font_size=10, font_weight='bold')
        
        ax1.set_aspect('equal')
        ax1.axis('off')
        
        # Learned Graph
        ax2.set_title(f'Learned DAG (threshold={threshold})\n({dataset_name})', 
                     fontsize=14, fontweight='bold')
        
        # Draw nodes
        nx.draw_networkx_nodes(G_learned, pos, ax=ax2, node_color='lightcoral', 
                              node_size=1000, alpha=0.9)
        
        # Categorize edges for different colors
        true_edges = set(G_true.edges())
        learned_edges = set(G_learned.edges())
        
        correct_edges = true_edges.intersection(learned_edges)  # True Positives
        missing_edges = true_edges.difference(learned_edges)    # False Negatives
        false_edges = learned_edges.difference(true_edges)      # False Positives
        
        # Draw correct edges (green)
        if correct_edges:
            nx.draw_networkx_edges(G_learned, pos, edgelist=list(correct_edges), 
                                  ax=ax2, edge_color='green', arrows=True, 
                                  arrowsize=20, arrowstyle='->', width=3, alpha=0.8)
        
        # Draw false positive edges (red)
        if false_edges:
            nx.draw_networkx_edges(G_learned, pos, edgelist=list(false_edges), 
                                  ax=ax2, edge_color='red', arrows=True, 
                                  arrowsize=20, arrowstyle='->', width=2, alpha=0.8)
        
        # Draw missing edges as dashed lines (orange)
        if missing_edges:
            nx.draw_networkx_edges(G_true, pos, edgelist=list(missing_edges), 
                                  ax=ax2, edge_color='orange', arrows=True, 
                                  arrowsize=15, arrowstyle='->', width=1, 
                                  alpha=0.6, style='dashed')
        
        # Draw labels
        labels_learned = {i: name for i, name in enumerate(node_names)}
        nx.draw_networkx_labels(G_learned, pos, labels_learned, ax=ax2, font_size=10, font_weight='bold')
        
        ax2.set_aspect('equal')
        ax2.axis('off')
        
        # Add legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', lw=3, label='Correct edges (TP)'),
            Line2D([0], [0], color='red', lw=2, label='False positive edges (FP)'),
            Line2D([0], [0], color='orange', lw=1, linestyle='--', label='Missing edges (FN)')
        ]
        ax2.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
        plt.tight_layout()
        
        # Save graph comparison
        graph_path = os.path.join(self.results_dir, dataset_name, f"{dataset_name}_graph_comparison.png")
        os.makedirs(os.path.dirname(graph_path), exist_ok=True)
        plt.savefig(graph_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Graph comparison saved to {graph_path}")
        
        # Print edge analysis
        print(f"[INFO] Edge Analysis:")
        print(f"  ✓ Correct edges (TP): {len(correct_edges)}")
        print(f"  ✗ False positive edges (FP): {len(false_edges)}")
        print(f"  ✗ Missing edges (FN): {len(missing_edges)}")
        if correct_edges:
            print(f"  Correct: {list(correct_edges)}")
        if false_edges:
            print(f"  False positives: {list(false_edges)}")
        if missing_edges:
            print(f"  Missing: {list(missing_edges)}")
    
    def visualize_results(self, dataset_name: str, result: Dict):
        """Create visualizations for the test results."""
        if "error" in result:
            return
        
        W_learned = np.array(result["adjacency_matrices"]["learned"])
        W_true = np.array(result["adjacency_matrices"]["ground_truth"])
        node_names = result["node_names"]
        threshold = result["parameters"]["threshold"]
        
        # Save adjacency matrices as CSV
        self.save_adjacency_csv(dataset_name, W_learned, W_true, node_names, threshold)
        
        # Create side-by-side graph comparison
        self.visualize_side_by_side_graphs(dataset_name, W_learned, W_true, node_names, threshold)
        
        # Create heatmap visualization (original style)
        self.create_heatmap_visualization(dataset_name, W_learned, W_true, node_names, threshold)
    
    def create_heatmap_visualization(self, dataset_name: str, W_learned: np.ndarray, W_true: np.ndarray,
                                   node_names: List[str], threshold: float):
        """Create heatmap visualization of adjacency matrices."""
        # Apply threshold to learned matrix
        W_learned_thresh = apply_threshold(W_learned, threshold)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'NOTEARS Results: {dataset_name.upper()}', fontsize=16)
        
        # Ground truth
        sns.heatmap(W_true, ax=axes[0,0], cmap='Blues', cbar=True, 
                   xticklabels=node_names, yticklabels=node_names)
        axes[0,0].set_title('Ground Truth Adjacency Matrix')
        axes[0,0].set_xlabel('To Node')
        axes[0,0].set_ylabel('From Node')
        
        # Learned (raw)
        sns.heatmap(W_learned, ax=axes[0,1], cmap='Reds', cbar=True,
                   xticklabels=node_names, yticklabels=node_names)
        axes[0,1].set_title('Learned Adjacency Matrix (Raw)')
        axes[0,1].set_xlabel('To Node')
        axes[0,1].set_ylabel('From Node')
        
        # Learned (thresholded)
        sns.heatmap(W_learned_thresh, ax=axes[1,0], cmap='Greens', cbar=True,
                   xticklabels=node_names, yticklabels=node_names)
        axes[1,0].set_title(f'Learned Adjacency Matrix (Threshold={threshold})')
        axes[1,0].set_xlabel('To Node')
        axes[1,0].set_ylabel('From Node')
        
        # Comparison
        comparison = np.zeros_like(W_true)
        comparison[(W_learned_thresh != 0) & (W_true != 0)] = 1  # True Positive
        comparison[(W_learned_thresh != 0) & (W_true == 0)] = 2  # False Positive
        comparison[(W_learned_thresh == 0) & (W_true != 0)] = 3  # False Negative
        
        colors = ['white', 'green', 'red', 'orange']
        sns.heatmap(comparison, ax=axes[1,1], cmap=colors, cbar=True,
                   xticklabels=node_names, yticklabels=node_names)
        axes[1,1].set_title('Edge Comparison')
        axes[1,1].set_xlabel('To Node')
        axes[1,1].set_ylabel('From Node')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.results_dir, dataset_name, f"{dataset_name}_heatmap_results.png")
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Heatmap visualization saved to {plot_path}")
    
    def save_results(self, dataset_name: str, result: Dict):
        """Save test results to JSON file."""
        dataset_result_dir = os.path.join(self.results_dir, dataset_name)
        os.makedirs(dataset_result_dir, exist_ok=True)
        result_path = os.path.join(dataset_result_dir, f"{dataset_name}_results.json")
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"[INFO] Results saved to {result_path}")
    
    def run_tests(self, dataset_names: List[str], 
                  lambda1: float = 0.01, 
                  lambda2: float = 0.01,
                  max_iter: int = 50) -> Dict[str, Dict]:
        """Run tests on multiple datasets."""
        results = {}
        
        for dataset_name in dataset_names:
            print(f"\n[INFO] Starting test for {dataset_name}")
            
            result = self.test_dataset(
                dataset_name, 
                lambda1=lambda1, 
                lambda2=lambda2, 
                max_iter=max_iter
            )
            
            results[dataset_name] = result
            
            if "error" not in result:
                self.visualize_results(dataset_name, result)
                self.save_results(dataset_name, result)
            else:
                print(f"[ERROR] Test failed for {dataset_name}: {result['error']}")
        
        # Save summary
        self.save_test_summary(results)
        return results
    
    def save_test_summary(self, results: Dict[str, Dict]):
        """Save a summary of all test results."""
        summary = {
            "total_datasets": len(results),
            "successful_tests": sum(1 for r in results.values() if "error" not in r),
            "failed_tests": sum(1 for r in results.values() if "error" in r),
            "results_summary": []
        }
        
        for dataset_name, result in results.items():
            if "error" not in result:
                summary["results_summary"].append({
                    "dataset": dataset_name,
                    "nodes": result["dataset_info"]["nodes"],
                    "true_edges": result["true_edges"],
                    "learned_edges": result["learned_edges"],
                    "precision": result["metrics"]["precision"],
                    "recall": result["metrics"]["recall"],
                    "f1_score": result["metrics"]["f1_score"],
                    "runtime": result["runtime_seconds"]
                })
            else:
                summary["results_summary"].append({
                    "dataset": dataset_name,
                    "error": result["error"]
                })
        
        summary_path = os.path.join(self.results_dir, "test_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n[INFO] Test summary saved to {summary_path}")
        self.print_summary(summary)
    
    def print_summary(self, summary: Dict):
        """Print a formatted summary of test results."""
        print(f"\n{'='*80}")
        print("NOTEARS TESTING SUMMARY")
        print(f"{'='*80}")
        print(f"Total datasets tested: {summary['total_datasets']}")
        print(f"Successful tests: {summary['successful_tests']}")
        print(f"Failed tests: {summary['failed_tests']}")
        
        if summary['results_summary']:
            print(f"\n{'Dataset':<15} {'Nodes':<6} {'Edges':<12} {'Precision':<10} {'Recall':<8} {'F1':<8} {'Time(s)':<8}")
            print("-" * 80)
            
            successful_results = [r for r in summary['results_summary'] if 'error' not in r]
            for result in successful_results:
                print(f"{result['dataset']:<15} {result['nodes']:<6} "
                      f"{result['learned_edges']:>3}/{result['true_edges']:<3} "
                      f"{result['precision']:<10.3f} {result['recall']:<8.3f} "
                      f"{result['f1_score']:<8.3f} {result['runtime']:<8.1f}")
            
            # Print failed tests
            failed_results = [r for r in summary['results_summary'] if 'error' in r]
            if failed_results:
                print(f"\nFailed tests:")
                for result in failed_results:
                    print(f"  {result['dataset']}: {result['error']}")


def main():
    """Main function to run NOTEARS tests on bnlearn datasets."""
    tester = NOTEARSDatasetTester()
    
    print("NOTEARS Algorithm Testing on BnLearn Datasets")
    print("=" * 50)
    
    # Check available datasets
    datasets_dir = tester.datasets_dir
    if not os.path.exists(datasets_dir):
        print(f"[ERROR] Datasets directory not found: {datasets_dir}")
        print("Please run dataset_downloader.py first to download datasets.")
        return
    
    available_datasets = [d for d in os.listdir(datasets_dir) 
                         if os.path.isdir(os.path.join(datasets_dir, d)) and 
                         os.path.exists(os.path.join(datasets_dir, d, "info.json"))]
    
    if not available_datasets:
        print("[ERROR] No datasets found. Please run dataset_downloader.py first.")
        return
    
    print(f"Available datasets: {', '.join(available_datasets)}")
    
    # Get user choice
    print("\nChoose testing option:")
    print("1. Test small datasets (recommended)")
    print("2. Test all available datasets")
    print("3. Test specific datasets")
    print("4. Test with custom parameters")
    
    try:
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            # Test small datasets
            small_datasets = []
            for dataset in available_datasets:
                info = tester.load_dataset_info(dataset)
                if info and info.get('nodes', 0) <= 20:
                    small_datasets.append(dataset)
            
            if not small_datasets:
                print("No small datasets available.")
                return
            
            print(f"Testing small datasets: {', '.join(small_datasets)}")
            results = tester.run_tests(small_datasets)
            
        elif choice == "2":
            print(f"Testing all datasets: {', '.join(available_datasets)}")
            results = tester.run_tests(available_datasets)
            
        elif choice == "3":
            selected = input("Enter dataset names (comma-separated): ").strip().split(',')
            selected = [s.strip() for s in selected if s.strip() in available_datasets]
            
            if not selected:
                print("No valid datasets selected.")
                return
            
            results = tester.run_tests(selected)
            
        elif choice == "4":
            # Custom parameters
            lambda1 = float(input("Enter lambda1 (default 0.01): ") or "0.01")
            lambda2 = float(input("Enter lambda2 (default 0.01): ") or "0.01")
            max_iter = int(input("Enter max_iter (default 50): ") or "50")
            
            datasets = input("Enter datasets (comma-separated, or 'all'): ").strip()
            if datasets.lower() == 'all':
                selected = available_datasets
            else:
                selected = [s.strip() for s in datasets.split(',') if s.strip() in available_datasets]
            
            results = tester.run_tests(selected, lambda1=lambda1, lambda2=lambda2, max_iter=max_iter)
            
        else:
            print("Invalid choice.")
            return
        
        print(f"\nTesting completed. Results saved in: {tester.results_dir}/")
        
    except KeyboardInterrupt:
        print("\nTesting interrupted by user.")
    except Exception as e:
        print(f"Error during testing: {e}")


if __name__ == "__main__":
    main()