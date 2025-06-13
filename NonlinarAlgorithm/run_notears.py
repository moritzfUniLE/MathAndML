#!/usr/bin/env python3
"""
Entry point for running NOTEARS nonlinear causal discovery.
Loads CSV data, runs the Notears MLP optimizer, and saves or plots results.
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import networkx as nx

from notears_core import (
    NotearsMLP,
    notears_nonlinear,
    load_ground_truth_from_bif,
    compute_metrics,
)

# Use double precision for all Torch tensors by default
torch.set_default_dtype(torch.double)


def load_data(csv_path: str, impute_nan: bool = True) -> np.ndarray:
    """
    Load numerical data from a CSV file into a numpy array.

    Args:
        csv_path: Path to the CSV file containing the dataset.
        impute_nan: Whether to replace NaNs by column mean.

    Returns:
        A 2D numpy array of shape (n_samples, n_features).
    """
    df = pd.read_csv(csv_path)
    X = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    # Impute missing values by column mean if enabled
    if impute_nan and np.isnan(X).any():
        col_mean = np.nanmean(X, axis=0)
        nan_idx = np.where(np.isnan(X))
        X[nan_idx] = np.take(col_mean, nan_idx[1])
    return X


def standardise(X: np.ndarray) -> np.ndarray:
    """
    Standardise each feature: zero mean, unit variance.
    Small std-dev are clamped to prevent division by zero.

    Args:
        X: Input data array.

    Returns:
        Standardised data array.
    """
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd < 1e-10] = 1.0
    return (X - mu) / sd


def apply_threshold(W: np.ndarray, thresh: float | None) -> np.ndarray:
    """
    Zero out entries below a threshold to sparsify adjacency.
    Ensures diagonal entries remain zero.

    Args:
        W: Weight matrix (d x d).
        thresh: Threshold value; if None, no threshold.

    Returns:
        Thresholded adjacency matrix.
    """
    W_thr = W.copy()
    if thresh is not None:
        W_thr[np.abs(W_thr) < thresh] = 0.0
    np.fill_diagonal(W_thr, 0.0)
    return W_thr


def save_outputs(W: np.ndarray, outdir: Path, thresh: float | None = None) -> None:
    """
    Save adjacency matrix CSV and edge list JSON to disk.

    Args:
        W: Raw learned weight matrix.
        outdir: Output directory path.
        thresh: Threshold for sparsification.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    W_thr = apply_threshold(W, thresh)
    # CSV of adjacency
    np.savetxt(outdir / "adjacency.csv", W_thr, delimiter=",")
    # JSON edge list
    edges = [
        {"src": int(i), "dst": int(j), "weight": float(W_thr[i, j])}
        for i, j in zip(*np.nonzero(W_thr))
    ]
    (outdir / "edges.json").write_text(json.dumps(edges, indent=2))


def plot_matrix(
        W: np.ndarray,
        outdir: Path,
        node_names: list[str] | None = None,
        thresh: float | None = None,
) -> None:
    """
    Plot heatmap of the learned adjacency matrix.

    Args:
        W: Raw weight matrix.
        outdir: Directory for saving plot.
        node_names: Optional labels for rows/cols.
        thresh: Threshold before plotting.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W_thr = apply_threshold(W, thresh)
    vmax = max(abs(W_thr.min()), abs(W_thr.max()))

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(W_thr, cmap='bwr', vmin=-vmax, vmax=vmax)
    plt.colorbar(im)

    if node_names:
        ax.set_xticks(range(len(node_names)))
        ax.set_yticks(range(len(node_names)))
        ax.set_xticklabels(node_names, rotation=45, ha='right')
        ax.set_yticklabels(node_names)

    ax.set_title("Adjazenzmatrix")
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    plt.tight_layout()
    plt.savefig(outdir / "adjacency.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_ground_truth_matrix(
        W_true: np.ndarray,
        outdir: Path,
        node_names: list[str] | None = None,
) -> None:
    """
    Plot heatmap of the ground truth adjacency matrix.

    Args:
        W_true: Ground truth matrix.
        outdir: Directory for saving plot.
        node_names: Optional labels.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 8))
    vmax = max(abs(W_true.min()), abs(W_true.max()))
    im = ax.imshow(W_true, cmap='bwr', vmin=-vmax, vmax=vmax)
    plt.colorbar(im)

    if node_names:
        ax.set_xticks(range(len(node_names)))
        ax.set_yticks(range(len(node_names)))
        ax.set_xticklabels(node_names, rotation=45, ha='right')
        ax.set_yticklabels(node_names)

    ax.set_title("Ground Truth Adjazenzmatrix")
    ax.set_xlabel("To")
    ax.set_ylabel("From")
    plt.tight_layout()
    plt.savefig(outdir / "ground_truth_matrix.png", dpi=300, bbox_inches='tight')
    plt.close()


def get_fixed_layout(n_nodes: int) -> dict:
    """
    Create circular layout for graph nodes.

    Args:
        n_nodes: Number of nodes.

    Returns:
        Dict node index -> (x,y) coords.
    """
    import numpy as np
    radius = 1.0
    angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
    return {i: (radius * np.cos(a), radius * np.sin(a)) for i, a in enumerate(angles)}


def plot_graphs_side_by_side(
        W: np.ndarray,
        W_true: np.ndarray,
        outdir: Path,
        node_names: list[str] | None = None,
        thresh: float | None = None,
) -> None:
    """
    Compare learned DAG vs ground truth side by side.

    Args:
        W: Learned matrix.
        W_true: Ground truth matrix.
        outdir: Save directory.
        node_names: Optional labels.
        thresh: Threshold for W.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W_thr = apply_threshold(W, thresh)
    n = W_thr.shape[0]
    if not node_names or len(node_names) != n:
        print("[WARN] Invalid node_names, using indices")
        node_names = [str(i) for i in range(n)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    pos = get_fixed_layout(n)

    # Learned DAG
    G1 = nx.DiGraph()
    G1.add_nodes_from(range(n))
    G1.add_edges_from(zip(*np.nonzero(W_thr)))
    nx.draw_networkx(G1, pos, ax=ax1, node_color='lightblue', edge_color='gray', with_labels=True,
                     labels={i: node_names[i] for i in range(n)})
    ax1.set_title("Learned DAG")

    # Ground Truth DAG
    if W_true.shape != W.shape:
        print("[WARN] Shape mismatch, skipping ground truth plot")
        plt.close()
        return
    G2 = nx.DiGraph()
    G2.add_nodes_from(range(n))
    G2.add_edges_from(zip(*np.nonzero(W_true)))
    nx.draw_networkx(G2, pos, ax=ax2, node_color='lightgreen', edge_color='darkgreen', with_labels=True,
                     labels={i: node_names[i] for i in range(n)})
    ax2.set_title("Ground Truth DAG")

    plt.tight_layout()
    plt.savefig(outdir / "comparison.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_graph(
        W: np.ndarray,
        outdir: Path,
        node_names: list[str] | None = None,
        thresh: float | None = None,
) -> None:
    """
    Plot the learned DAG structure using spring layout.

    Args:
        W: Learned weight matrix.
        outdir: Save directory.
        node_names: Optional labels.
        thresh: Threshold for sparsity.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W_thr = apply_threshold(W, thresh)
    n = W_thr.shape[0]
    if not node_names or len(node_names) != n:
        print(f"[WARN] Invalid node_names, using indices ({len(node_names) if node_names else 0} vs {n})")
        node_names = [str(i) for i in range(n)]

    G = nx.DiGraph()
    for i in range(n): G.add_node(i, name=node_names[i])
    edges = [(i, j, {'weight': W_thr[i, j]}) for i, j in zip(*np.nonzero(W_thr))]
    G.add_edges_from(edges)
    if not G.edges(): print("[WARN] No edges above threshold"); return

    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, k=1 / np.sqrt(n), iterations=50)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500, alpha=0.8)
    weights = [abs(G[u][v]['weight']) * 2 for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=weights, arrowsize=20)
    nx.draw_networkx_labels(G, pos, labels={i: G.nodes[i]['name'] for i in G.nodes()})
    plt.title("Learned DAG Structure")
    plt.axis('off')
    plt.tight_layout()
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / "graph.png", dpi=300, bbox_inches='tight')
    plt.close()


def main() -> None:
    """
    Parse arguments, load data, run NOTEARS algorithm, and handle outputs.
    """
    parser = argparse.ArgumentParser(description="Run NOTEARS nonlinear causal discovery.")
    parser.add_argument("--csv", required=True, help="Input CSV file path.")
    parser.add_argument("--hidden", type=int, default=20, help="Hidden layer size.")
    parser.add_argument("--lambda1", type=float, default=0.01, help="L1 regularization weight.")
    parser.add_argument("--lambda2", type=float, default=0.01, help="L2 regularization weight.")
    parser.add_argument("--max-iter", type=int, default=100, help="Max optimization iterations.")
    parser.add_argument("--thresh", type=float, default=0.3, help="Threshold for edge pruning.")
    parser.add_argument("--out", default="results", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--no-impute", action="store_true", help="Disable NaN imputation.")
    parser.add_argument("--plot", action="store_true", help="Enable heatmap plotting.")
    parser.add_argument("--graph", action="store_true", help="Enable graph plotting.")
    parser.add_argument("--evaluate", action="store_true", help="Compute metrics with ground truth.")
    parser.add_argument("--ground-truth", help="BIF file path for ground truth.")
    parser.add_argument("--node-names", type=str, help="Comma-separated node names.")

    args = parser.parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load and preprocess data
    X = load_data(args.csv, impute_nan=not args.no_impute)
    if np.isnan(X).any():
        raise ValueError("Dataset contains NaN values after imputation")
    X = standardise(X)

    d = X.shape[1]
    model = NotearsMLP(d, m_hidden=args.hidden)

    print(f"[INFO] Dataset: {X.shape}, Hidden: {args.hidden}")
    print(f"[INFO] Lambda1: {args.lambda1}, Lambda2: {args.lambda2}")

    # Run optimization
    t0 = time.time()
    W = notears_nonlinear(
        model, X,
        lambda1=args.lambda1,
        lambda2=args.lambda2,
        max_iter=args.max_iter,
    )
    print(f"[INFO] Training finished in {time.time() - t0:.1f}s")

    # Parse node names if provided
    node_names = [n.strip() for n in args.node_names.split(',')] if args.node_names else None

    # Load ground truth if requested
    W_true = None
    if args.ground_truth:
        print(f"[INFO] Loading ground truth from {args.ground_truth}")
        W_true, node_names_bif, success = load_ground_truth_from_bif(args.ground_truth)
        if success and node_names is None:
            node_names = node_names_bif

    # Evaluate metrics
    if args.evaluate and W_true is not None:
        metrics = compute_metrics(W, W_true, thresh=args.thresh)
        if metrics['error']:
            print(f"[ERROR] Evaluation error: {metrics['error']}")
        else:
            print("\n[INFO] Evaluation metrics:")
            for key in ['hamming_distance', 'true_positives', 'false_positives', 'false_negatives']:
                if metrics[key] is not None:
                    print(f"{key.replace('_', ' ').title()}: {metrics[key]}")
            print(f"Precision: {metrics['precision']:.3f}")
            print(f"Recall: {metrics['recall']:.3f}")
            print(f"F1-Score: {metrics['f1_score']:.3f}")

    outdir = Path(args.out)
    # Save and plot outputs
    save_outputs(W, outdir, thresh=args.thresh)
    if W_true is not None:
        plot_ground_truth_matrix(W_true, outdir, node_names)
        if args.graph:
            plot_graphs_side_by_side(W, W_true, outdir, node_names, args.thresh)
    if args.plot:
        plot_matrix(W, outdir, node_names, args.thresh)
    if args.graph:
        plot_graph(W, outdir, node_names, args.thresh)

    print(f"[INFO] Results saved to {outdir.resolve()}")


if __name__ == "__main__":
    main()