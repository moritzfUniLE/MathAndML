"""
Utility functions for NOTEARS causal discovery.
Common functions used across different algorithms and components.
"""
import numpy as np


def apply_threshold(W: np.ndarray, thresh: float | None) -> np.ndarray:
    """
    Apply threshold to adjacency matrix, zeroing small entries and diagonal.
    
    Args:
        W: Adjacency matrix
        thresh: Threshold value (None to skip thresholding)
        
    Returns:
        Thresholded adjacency matrix
    """
    W_thr = W.copy()
    if thresh is not None: 
        W_thr[np.abs(W_thr) < thresh] = 0.0
    np.fill_diagonal(W_thr, 0.0)
    return W_thr


def load_ground_truth_from_bif(bif_path: str) -> tuple[np.ndarray|None, list[str]|None, bool]:
    """
    Parse BIF file to extract ground-truth DAG and node names.
    
    Args:
        bif_path: Path to BIF file
        
    Returns:
        Tuple of (adjacency_matrix, node_names, success_flag)
    """
    import re
    try:
        with open(bif_path, 'r') as f:
            text = f.read()
        
        vars = re.findall(r'variable\s+(\w+)', text)
        if not vars: 
            raise ValueError("No variables found in BIF")
        
        n = len(vars)
        idx = {v: i for i, v in enumerate(vars)}
        W = np.zeros((n, n))
        
        # Extract parent-child relations
        for m in re.finditer(r'probability\s*\(\s*(\w+)\s*\|\s*([^\)]+)\)', text):
            child = m.group(1)
            parents = [p.strip() for p in m.group(2).split(',')]
            for p in parents:
                if p in idx:  # Safety check
                    W[idx[p], idx[child]] = 1
        
        print(f"[INFO] Ground Truth loaded: {n} nodes")
        return W, vars, True
        
    except Exception as e:
        print(f"[ERROR] BIF load failed: {e}")
        return None, None, False


def compute_metrics(W_learned: np.ndarray, W_true: np.ndarray, thresh: float | None = None) -> dict:
    """
    Compute performance metrics comparing learned and true adjacency matrices.
    
    Args:
        W_learned: Learned adjacency matrix
        W_true: Ground truth adjacency matrix  
        thresh: Threshold for learned matrix
        
    Returns:
        Dictionary with metrics (precision, recall, F1, etc.) and any error message
    """
    try:
        if W_learned.shape != W_true.shape:
            raise ValueError(f"Shape mismatch: learned {W_learned.shape} vs true {W_true.shape}")
        
        # Apply threshold and convert to binary
        Wl = (apply_threshold(W_learned, thresh) != 0).astype(int)
        Wt = (W_true != 0).astype(int)
        
        # Compute confusion matrix elements
        TP = int(np.sum((Wl == 1) & (Wt == 1)))
        FP = int(np.sum((Wl == 1) & (Wt == 0)))
        FN = int(np.sum((Wl == 0) & (Wt == 1)))
        
        # Compute metrics
        precision = TP / (TP + FP) if TP + FP > 0 else 0
        recall = TP / (TP + FN) if TP + FN > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        
        return {
            'hamming_distance': int(np.sum(Wl != Wt)),
            'true_positives': TP,
            'false_positives': FP,
            'false_negatives': FN,
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'error': None
        }
        
    except Exception as e:
        return {
            'hamming_distance': None,
            'true_positives': None,
            'false_positives': None,
            'false_negatives': None,
            'precision': None,
            'recall': None,
            'f1_score': None,
            'error': str(e)
        }