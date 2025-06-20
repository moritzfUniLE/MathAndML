"""
Quick test of NOTEARS implementation on small synthetic data.
"""
import numpy as np
import pandas as pd
from notears_core import NotearsMLP, notears_nonlinear, compute_metrics

# Create very small test data (3 nodes, chain: A -> B -> C)
np.random.seed(42)
n_samples = 100  # Small sample size
n_vars = 3

# Ground truth: A -> B -> C
W_true = np.array([
    [0, 1, 0],  # A -> B
    [0, 0, 1],  # B -> C
    [0, 0, 0]   # C has no children
])

# Generate synthetic data
X = np.zeros((n_samples, n_vars))
X[:, 0] = np.random.normal(0, 1, n_samples)  # A
X[:, 1] = np.tanh(X[:, 0]) + np.random.normal(0, 0.5, n_samples)  # B = f(A) + noise
X[:, 2] = np.tanh(X[:, 1]) + np.random.normal(0, 0.5, n_samples)  # C = f(B) + noise

print("Quick NOTEARS Test")
print("=" * 30)
print(f"Data shape: {X.shape}")
print(f"Ground truth edges: {np.sum(W_true)}")
print(f"Ground truth structure:\n{W_true}")

# Run NOTEARS with minimal iterations
print("\nRunning NOTEARS...")
model = NotearsMLP(n_vars, m_hidden=5)

try:
    W_learned = notears_nonlinear(
        model, X, 
        lambda1=0.01, 
        lambda2=0.01, 
        max_iter=10  # Very few iterations for quick test
    )
    
    print(f"\nLearned adjacency matrix:\n{W_learned}")
    
    # Compute metrics
    metrics = compute_metrics(W_learned, W_true, thresh=0.1)
    print(f"\nMetrics (threshold=0.1):")
    print(f"  Precision: {metrics['precision']:.3f}")
    print(f"  Recall: {metrics['recall']:.3f}")
    print(f"  F1-Score: {metrics['f1_score']:.3f}")
    print(f"  Hamming Distance: {metrics['hamming_distance']}")
    
    print("\nTest completed successfully!")
    
except Exception as e:
    print(f"Error during NOTEARS execution: {e}")
    import traceback
    traceback.print_exc()