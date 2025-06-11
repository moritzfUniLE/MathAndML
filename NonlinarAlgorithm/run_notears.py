#!/usr/bin/env python3
"""
run_notears.py — nonlinear NOTEARS + visualisation helper
=======================================================

• Lädt ein numerisches CSV, bereitet die Daten vor (optional: NaN‑Imputation).
• Lernt eine gerichtete azyklische Struktur mit der MLP‑Variante von NOTEARS.
• Speichert
    ‑ adjacency.csv  – d×d Kantengewichte‑Matrix
    ‑ edges.json     – Edge‑Liste (sparse) für interaktive Tools
    ‑ adjacency.png  – Heat‑Map (optional mit --plot)
    ‑ graph.png      – gerichteter Graph (optional mit --graph)

Beispiel
--------
    python run_notears.py --csv data.csv --hidden 20 \
                          --lambda1 1e-3 --lambda2 0 \
                          --max-iter 50 --thresh 0.02 \
                          --out results --plot --graph
"""
from __future__ import annotations

import argparse, json, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import networkx as nx

from notears_core import NotearsMLP, notears_nonlinear

torch.set_default_dtype(torch.double)  # globale Float‑Präzision = float64

# ---------------------------------------------------------------------------
# Hilfsfunktionen
# ---------------------------------------------------------------------------

def load_data(csv_path: str, impute_nan: bool = True) -> np.ndarray:
    """Liest *numerische* Spalten aus *csv_path* und imputiert optional NaNs."""
    df = pd.read_csv(csv_path)
    X = df.select_dtypes(include=[np.number]).to_numpy(dtype=np.float64)
    if impute_nan and np.isnan(X).any():
        col_mean = np.nanmean(X, axis=0)
        nan_idx = np.where(np.isnan(X))
        X[nan_idx] = np.take(col_mean, nan_idx[1])
    return X


def standardise(X: np.ndarray) -> np.ndarray:
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True)
    sd[sd == 0.0] = 1.0  # schützt gegen konstante Spalten
    return (X - mu) / sd


def apply_threshold(W: np.ndarray, thresh: float | None) -> np.ndarray:
    """Setzt Einträge mit |W| < thresh auf 0 und gibt *neue* Matrix zurück."""
    if thresh is None:
        return W
    W_thr = W.copy()
    W_thr[np.abs(W_thr) < thresh] = 0.0
    return W_thr


def save_outputs(W: np.ndarray, outdir: Path, thresh: float | None = None) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    W_thr = apply_threshold(W, thresh)
    np.savetxt(outdir / "adjacency.csv", W_thr, delimiter=",")
    edges = [
        {"src": int(i), "dst": int(j), "weight": float(W_thr[i, j])}
        for i, j in zip(*np.nonzero(W_thr))
    ]
    (outdir / "edges.json").write_text(json.dumps(edges, indent=2))


def plot_matrix(W: np.ndarray, outdir: Path, thresh: float | None = None, cmap: str = "bwr") -> None:
    """Speichert eine PNG‑Heat‑Map von *W* unter *adjacency.png*."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W_thr = apply_threshold(W, thresh)
    vmax = np.abs(W_thr).max() or 1.0
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(W_thr, cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xlabel("destination node j")
    ax.set_ylabel("source node i")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="edge weight")
    fig.tight_layout()
    plt.savefig(outdir / "adjacency.png", dpi=150)
    plt.close(fig)


def plot_graph(W: np.ndarray, outdir: Path, thresh: float | None = None, labels: list[str] | None = None) -> None:
    """Erzeugt eine gerichtete Graph‑PNG *graph.png* mittels NetworkX."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    W_thr = apply_threshold(W, thresh)
    d = W_thr.shape[0]
    node_labels = labels or [str(i) for i in range(d)]

    G = nx.DiGraph()
    for i in range(d):
        G.add_node(node_labels[i])
    for i in range(d):
        for j in range(d):
            w = W_thr[i, j]
            if abs(w) > 0.0:
                G.add_edge(node_labels[i], node_labels[j], weight=w)

    if not G.edges:
        print("[WARN] Keine Kanten über dem Threshold – graph.png wird nicht erzeugt")
        return

    pos = nx.spring_layout(G, seed=42)
    weights = [abs(G[u][v]['weight']) for u, v in G.edges]
    max_w = max(weights)
    lw = [1.0 + 4.0*(w / max_w) for w in weights]  # dynamische Linienbreite

    plt.figure(figsize=(max(4, d/2), max(4, d/2)))
    nx.draw_networkx_nodes(G, pos, node_size=800, node_color="#ffffff", edgecolors="#000000")
    nx.draw_networkx_edges(G, pos, width=lw, arrowstyle="-|>", arrowsize=12)
    nx.draw_networkx_labels(G, pos, font_size=9)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(outdir / "graph.png", dpi=150)
    plt.close()

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run nonlinear NOTEARS on a CSV file and save adjacency/graph visualisations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--csv", required=True, help="Pfad zur Eingabe‑CSV (nur numerische Spalten werden verwendet).")
    ap.add_argument("--hidden", type=int, default=10, help="Breite der versteckten Schicht (m) der MLP‑Variante.")
    ap.add_argument("--lambda1", type=float, default=0.1,
                    help="L1‑Regularisierung (fördert Sparsity auf der ersten Schicht).")
    ap.add_argument("--lambda2", type=float, default=0.1,
                    help="L2‑Regularisierung (glättet alle Gewichte, stärkt Numerik‑Stabilität).")
    ap.add_argument("--max-iter", type=int, default=20, help="Äußere Dual‑Ascent‑Iterationen.")
    ap.add_argument("--thresh", type=float, default=None,
                    help="Setzt |W|<thresh auf 0 *nach* der Optimierung und *vor* dem Speichern/Plotten.")
    ap.add_argument("--out", default="results", help="Ausgabeverzeichnis für alle Dateien.")
    ap.add_argument("--seed", type=int, default=42, help="Zufallssaat für NumPy + PyTorch.")
    ap.add_argument("--no-impute", action="store_true", help="Deaktiviert NaN‑Imputation (bricht bei NaNs ab).")
    ap.add_argument("--plot", action="store_true", help="Speichert eine Heat‑Map *adjacency.png*.")
    ap.add_argument("--graph", action="store_true", help="Speichert eine NetworkX‑Visualisierung *graph.png*.")

    args = ap.parse_args()

    # reproducible randomness
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # 1) Daten laden & skalieren
    X = load_data(args.csv, impute_nan=not args.no_impute)
    if np.isnan(X).any():
        raise ValueError("Datensatz enthält noch NaN – CSV säubern oder --no-impute weglassen")
    X = standardise(X)

    # 2) Modell & Optimierung
    d = X.shape[1]
    h = args.hidden or max(10, int(0.1 * d))
    print(f"[INFO] dataset shape = {X.shape}; hidden = {h}")

    model = NotearsMLP(d, m_hidden=h)
    with torch.no_grad():
        model.fc1_pos.weight.uniform_(1e-3, 1e-2)
        model.fc1_neg.weight.uniform_(1e-3, 1e-2)

    t0 = time.time()
    W = notears_nonlinear(model, X,
                          lambda1=args.lambda1,
                          lambda2=args.lambda2,
                          max_iter=args.max_iter)
    print(f"[INFO] finished in {time.time()-t0:.1f}s, |E| = {np.count_nonzero(apply_threshold(W, args.thresh))}")

    # 3) Outputs
    outdir = Path(args.out)
    save_outputs(W, outdir, thresh=args.thresh)
    if args.plot:
        plot_matrix(W, outdir, thresh=args.thresh)
    if args.graph:
        plot_graph(W, outdir, thresh=args.thresh)
    print(f"[INFO] results written to {outdir.resolve()}")


if __name__ == "__main__":
    main()

