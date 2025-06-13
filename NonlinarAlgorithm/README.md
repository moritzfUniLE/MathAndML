# Nonlinear NOTEARS DAG Learner

This repository contains a small, self‑contained wrapper around the **non‑linear NOTEARS** algorithm (`run_notears.py`, `notears_core.py`). It trains a directed acyclic graph (DAG) on a numerical data set and can visualise the learned structure as a **heat‑map** _and_ as a _directed graph_.

----------

## 1 Installation

```bash
# 1. Make sure the files `run_notears.py`, `notears_core.py`, `requirements.txt` and this `README.md` are located in **one directory** (e.g. `notears-mlp`).
cd <your-working-directory>

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate  # on Windows: venv\Scripts\activate

# 3. Install the required packages
pip install -r requirements.txt

```

All packages are available from PyPI; CUDA support for PyTorch is **optional**.

----------

## 2 Quick start

```bash
python run_notears.py 
    --csv <path-to-input.csv> 
    --hidden 20 
    --lambda1 0.1 
    --lambda2 0 
    --max-iter 50 
    --thresh 0.3 
    --out <path-to-output-dir> 
    --plot 
    --graph
    --evaluate 
    --ground-truth <path-to-ground-truth.bif>

```

After the run you will find in the chosen  `--out` directory:

| file | purpose |
|--|--|
| `adjacency.csv` |  learned **d × d** weight matrix (after optional threshold)
|`edges.json`| sparse edge list; convenient for D3, Gephi, etc.
|`adjacency.png`| colour heat‑map (only if `--plot`)
|`ground_truth_matrix.png`| colour heat‑map of the ground truth DAG (if --ground-truth)
|`comparison.png` | side‑by‑side comparison of learned vs. ground truth DAG (if --graph and --ground-truth)
|`graph.png`|tidy NetworkX drawing (only if `--graph`)

----------

## 3 CLI reference

Every flag has an in‑script help text (`python run_notears.py --help`). The table below gives a concise overview.

| flag             | type / default     | description |
|------------------|--------------------|--|
| `--csv`          | **required**       | Path to the input CSV; _only numeric columns_ are used.
| `--hidden`       | `int`, _10_        | Width _m_ of the first hidden layer of the MLP.
| `--lambda1`      | `float`, _0.1_     | L1 (lasso) penalty – enforces sparsity and DAG constraint.
| `--lambda2`      | `float`, _0.1_     | L2 (ridge) penalty – stabilises optimisation.
| `--max-iter`     | `int`, _20_        | Outer dual‑ascent iterations.
| `--thresh`       | `float`, _None_    | Zero‑out weights with `W < thresh` **after** training. None keeps raw values.
| `--out`          | `str`, _"results"_ | Output directory (created if absent).
| `--seed`         | `int`, _42_        | Global random seed for NumPy & PyTorch.
| `--no-impute`    | _(flag)_           | **Deactivate** NaN imputation (run will abort if NaN present).
| `--plot`         | _(flag)_           | Save a heat‑map of the adjacency matrix as `adjacency.png`.
| `--graph`        | _(flag)_           | Render a directed graph layout as `graph.png`.
| `--graph`        | (flag)             | Render a directed graph layout of the learned DAG as graph.png. 
| `--evaluate`     | (flag)             | Compute evaluation metrics (Hamming, precision, recall, F1) against ground truth.
| `--ground-truth` | str                | Path to a BIF-format file containing the ground truth DAG.
| `--node-names`   | str                | Comma-separated list of node names; overrides names from BIF if provided.

----------

## 4 Program functionality

In addition to learning and visualizing a DAG, the tool now offers:

- **Edge list export** (`edges.json`): JSON representation of directed edges with weights.

- **Ground truth support** (`--ground-truth`): Load a Bayesian Interchange Format (BIF) file to extract true parent-child relations.

- **Evaluation metrics** (`--evaluate`): Compute Hamming distance, true positives, false positives, false negatives, precision, recall, and F1 score comparing learned vs. ground truth.

- **Named nodes** (`--node-names`): Supply human-readable labels for nodes in all plots and outputs.

- **Comparison plot** (`comparison.png`): Side‑by‑side visualization of the learned DAG and the ground truth DAG when both --graph and --ground-truth are used.

----------

## 5 Troubleshooting and tips

-   **All‑zero matrix?**
    
    -   Lower or disable `--thresh`.
        
    -   Reduce `--lambda1` (e.g. `1e-3` → `1e-4`).
        
-   **NaN explosion during optimisation?**
    
    -   Keep the built‑in weight bounds _(0 … 1.5)_ or add a small `--lambda2`.
        
-   **Very small weights (< 0.05):**
    
    -   That is normal after z‑score scaling; compare _relative_ magnitudes or refit on the fixed structure without L1.
        

----------

## 6 Uninstall

```bash
# Inside the project root
pip uninstall -r requirements.txt -y
rm -rf venv  # if you created one

```

----------
