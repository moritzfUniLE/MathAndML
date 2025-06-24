# CLAUDE.md - CausalDiscoveryPlatform Analysis

## Project Overview
The **CausalDiscoveryPlatform** is a comprehensive web-based platform for causal discovery that learns directed acyclic graphs (DAGs) from observational data. It implements the NOTEARS algorithm family alongside classical Bayesian network methods.

## Key Components

### Core Algorithms
- **NOTEARS Linear**: Continuous optimization for linear structural equation models
- **NOTEARS Nonlinear**: Neural network-based approach for nonlinear relationships
- **bnlearn Integration**: Classical methods (Hill Climbing, PC Algorithm, Grow-Shrink)

### Architecture
- **Modular Design**: `BaseAlgorithm` abstract class with consistent interface
- **Algorithm Registry**: Dynamic loading and management system
- **Web Interface**: Flask + Socket.IO for real-time progress monitoring
- **Dataset Management**: 19 built-in datasets + custom upload support

### Key Files Structure
```
CausalDiscoveryPlatform/
├── algorithms/              # Algorithm implementations
│   ├── base_algorithm.py   # Abstract interface
│   ├── algorithm_registry.py # Factory pattern
│   ├── notears_linear.py   # Linear NOTEARS
│   ├── notears_nonlinear.py # Nonlinear NOTEARS
│   └── bnlearn_algorithms.py # Classical methods
├── datasets/               # 19 built-in datasets
├── notears_web_gui.py     # Main Flask application
├── notears_utils.py       # Utility functions
└── launch_gui.py          # Application launcher
```

### Datasets Available
- **Synthetic**: chain, fork, collider, complex structures
- **Classical**: Asia (8 vars), ALARM (37 vars), SACHS (11 vars)
- **Preprocessed**: Multiple SACHS variants with different preprocessing
- **bnlearn**: Sprinkler, Auto MPG, Water treatment, Andes (223 vars)

## Key Concepts

### NOTEARS Innovation
- "NO TEARS" = No acyclicity constraints using combinatorial search
- Reformulates acyclic constraint as continuous optimization problem
- Uses augmented Lagrangian method with penalty terms

### DAG Learning Problem
- Discovers causal relationships as directed acyclic graphs
- Handles both linear and nonlinear relationships
- Provides ground truth evaluation when available

### Web Interface Features
- Real-time algorithm progress monitoring
- Interactive Plotly visualizations
- Session isolation for concurrent users
- Results persistence and comparison tools
- Parameter tuning with validation

## Usage Patterns
1. **Web Interface**: Launch via `python launch_gui.py` → http://localhost:5000
2. **API Usage**: Import algorithms directly for programmatic use
3. **Dataset Management**: Built-in datasets + CSV upload support
4. **Results Analysis**: Performance metrics, graph visualization, export options

## Technical Notes
- Python 3.8+ required
- Key dependencies: torch, bnlearn, flask, plotly, networkx
- Docker support available
- Windows compatibility via conda (recommended)
- Comprehensive error handling and logging

## Synthetic Dataset Generation

### New Feature: Built-in Synthetic Data Generation
- **Purpose**: Generate datasets with known ground truth for algorithm testing
- **Access**: "Create New Dataset" → "Generate Synthetic Dataset" mode
- **Parameters**:
  - Number of samples (n): 10-10,000 data points
  - Number of nodes (d): 2-50 variables 
  - Expected edges (s0): Target number of causal connections
  - Graph types: Erdős–Rényi (ER), Scale-Free (SF), Bipartite (BP)
  - Noise distributions: Gaussian, Exponential, Gumbel, Uniform, Logistic, Poisson

### Implementation Details
- **Backend**: `create_artifical_dataset()` in notears_utils.py
- **BIF Generation**: `adjacency_matrix_to_bif()` converts ground truth to BIF format
- **API Endpoint**: `/api/generate_synthetic_dataset` for dataset creation
- **Frontend**: Dual-mode modal (Upload vs Generate) with parameter validation
- **Integration**: Generated datasets work seamlessly with existing visualization/evaluation

### Benefits
- Known ground truth enables performance evaluation
- Controlled complexity for algorithm benchmarking  
- Parameter sensitivity analysis capabilities
- Educational tool for understanding causal structures

## Performance Considerations
- Memory usage scales with dataset size and hidden units
- Iteration monitoring via h-value convergence
- Parameter sensitivity requires tuning
- Large datasets (>50 variables) may need optimization
- Synthetic generation: O(n*d) time complexity, recommended limits enforced