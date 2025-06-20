# NOTEARS Causal Discovery Platform

A comprehensive web-based platform for causal discovery using NOTEARS algorithms and classical Bayesian network methods. Features an intuitive web interface, multiple algorithms, extensive dataset collection, and advanced visualization capabilities.

## Overview

This platform provides a complete implementation of causal discovery algorithms with a focus on ease of use and comprehensive functionality:

- **Multiple Algorithms**: NOTEARS (Linear & Nonlinear) + bnlearn integration (Hill Climbing, PC, Grow-Shrink)
- **Web-based Interface**: Interactive GUI with real-time progress and visualizations
- **Extensive Datasets**: 19+ built-in datasets from synthetic and real-world sources
- **Session Management**: Concurrent multi-user support with isolated execution
- **Advanced Visualization**: Interactive causal graphs and adjacency matrix heatmaps
- **Results Management**: Persistent storage, export, and comparison capabilities

## Features

### 🧠 **Multiple Causal Discovery Algorithms**
- **NOTEARS Linear**: Fast linear structural equation models
- **NOTEARS Nonlinear**: Neural network-based nonlinear relationships  
- **Hill Climbing (bnlearn)**: Score-based structure learning
- **PC Algorithm (bnlearn)**: Constraint-based causal discovery
- **Grow-Shrink (bnlearn)**: Efficient constraint-based learning

### 📊 **Comprehensive Dataset Collection**
- **Synthetic Networks**: Chain, Fork, Collider, Complex structures
- **Classical Benchmarks**: Asia, ALARM, SACHS protein signaling
- **bnlearn Datasets**: Sprinkler, Auto MPG, Water treatment, Andes (223 variables)
- **Preprocessed Variants**: Multiple SACHS preprocessing approaches
- **Custom Upload**: Support for user datasets via CSV

### 🌐 **Advanced Web Interface**
- **Real-time Progress**: Live algorithm execution monitoring
- **Interactive Visualizations**: Plotly-based graph exploration
- **Session Isolation**: Multiple users can run algorithms simultaneously
- **Responsive Design**: Works on desktop, tablet, and mobile
- **Export Capabilities**: CSV, JSON, PNG format support

### 📈 **Performance & Analysis**
- **Comprehensive Metrics**: Precision, Recall, F1-score, Hamming distance
- **Ground Truth Comparison**: Automatic evaluation with known structures
- **Result Management**: Save, organize, and compare algorithm runs
- **Parameter Optimization**: Interactive parameter tuning with validation

## Installation & Setup

### Prerequisites
- Python 3.8+ 
- Operating System: Linux (tested on Fedora/WSL), macOS, or Windows

### 1. Clone Repository
```bash
git clone <repository-url>
cd CausalDiscoveryPlatform
```

### 2. Install System Dependencies

**Fedora/RHEL:**
```bash
sudo dnf install python3-devel gcc gcc-c++ make
```

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3-dev python3-pip build-essential
```

**macOS:**
```bash
xcode-select --install  # Install command line tools
```

### 3. Install Python Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `torch>=2.0.0` - Neural network computations
- `bnlearn>=0.7.0` - Classical Bayesian network algorithms
- `flask>=3.0.0` - Web framework
- `flask-socketio>=5.3.0` - Real-time communication
- `plotly>=5.17.0` - Interactive visualizations
- `networkx>=3.0.0` - Graph manipulation
- `pandas>=2.0.0` - Data processing

**Note on bnlearn**: The bnlearn library provides classical Bayesian network algorithms (Hill Climbing, PC, Grow-Shrink). If installation fails, you can also install via conda:
```bash
conda install -c conda-forge bnlearn
```

### 4. Verify Installation
```bash
python launch_gui.py
```
This will check all dependencies and launch the web interface. All 19 datasets (including bnlearn datasets) are pre-loaded and ready to use.

## Quick Start Guide

### Option 1: Web Interface (Recommended)

1. **Launch the web application:**
```bash
python launch_gui.py
```
*Alternative launch method:*
```bash
python notears_web_gui.py
```

2. **Access the interface:**
   - Opens automatically at `http://localhost:5000`
   - Or manually navigate to the URL in your browser

3. **Basic workflow:**
   - Select a dataset from the dropdown (start with `fork` or `chain`)
   - Choose an algorithm (`NOTEARS Linear` recommended for beginners)
   - Adjust parameters if needed (defaults work well)
   - Click "Run Algorithm" and monitor progress
   - View results in the Results tab

### Option 2: Command Line Interface

```python
import numpy as np
from algorithms import get_algorithm

# Load data (your n_samples x n_variables matrix)
X = np.random.randn(1000, 5)

# Get algorithm instance
algorithm = get_algorithm('notears_nonlinear')

# Set parameters
params = {
    'lambda1': 0.01,      # L1 regularization
    'lambda2': 0.01,      # L2 regularization
    'max_iter': 100,      # Maximum iterations
    'm_hidden': 10,       # Hidden units
    'threshold': 0.3      # Edge threshold
}

# Run algorithm
W_learned = algorithm.run(X, params)
print("Learned adjacency matrix:")
print(W_learned)
```

## Available Algorithms

### NOTEARS Family

#### **NOTEARS Linear**
- **Best for**: Linear relationships, fast execution
- **Parameters**:
  - `lambda1` (0.001-1.0): L1 sparsity regularization
  - `lambda2` (0.0-1.0): L2 stability regularization  
  - `max_iter` (10-500): Maximum optimization iterations
  - `threshold` (0.0-1.0): Edge detection cutoff

#### **NOTEARS Nonlinear** 
- **Best for**: Complex nonlinear relationships
- **Parameters**:
  - `lambda1`, `lambda2`: Regularization (same as linear)
  - `max_iter`: Iterations (typically needs more than linear)
  - `m_hidden` (1-100): Neural network hidden units
  - `threshold`: Edge detection threshold

### bnlearn Integration

#### **Hill Climbing**
- **Best for**: Score-based discrete data
- **Parameters**:
  - `scoring_method`: BIC, K2, or BDEU scoring
  - `max_iter`: Search iterations
  - `discretize`: Auto-discretize continuous data
  - `discretize_nbins`: Number of discretization bins

#### **PC Algorithm**
- **Best for**: Constraint-based learning with independence tests
- **Parameters**:
  - `alpha` (0.001-0.2): Significance level for independence tests
  - `ci_test`: Conditional independence test (chi_square, pearsonr, etc.)
  - `discretize`: Auto-discretization option

#### **Grow-Shrink**
- **Best for**: Efficient constraint-based learning
- **Parameters**: Same as PC algorithm

## Dataset Collection

### Built-in Datasets (19 total)

#### **Synthetic Structures**
- `chain` - Linear chain: A → B → C → D (4 variables)
- `fork` - Common cause: A → {B, C, D} (4 variables)
- `collider` - Common effect: {A, B, C} → D (4 variables)
- `complex` - Multi-path network (6 variables, 7 edges)

#### **Classical Benchmarks**
- `asia` - Medical diagnosis network (8 variables)
- `alarm` - Medical monitoring (37 variables)
- `SACHSconglomeratedData` - Protein signaling (11 variables)

#### **SACHS Preprocessing Variants**
- `sachs_zscore` - Z-score normalized
- `sachs_iqr` - IQR outlier removal
- `sachs_isolation_forest` - Isolation Forest outlier detection
- `sachs_log_transform` - Log transformation

#### **bnlearn Datasets**
- `bnlearn_sprinkler` - Classic sprinkler network (4 variables)
- `bnlearn_asia` - Asia network from bnlearn (8 variables)
- `bnlearn_alarm` - ALARM from bnlearn (37 variables)
- `bnlearn_sachs` - SACHS from bnlearn (11 variables)
- `bnlearn_water` - Water treatment network (32 variables)
- `bnlearn_auto_mpg` - Automotive continuous data (8 variables)
- `bnlearn_andes` - Large physics network (223 variables)

### Custom Dataset Upload

**Requirements:**
- CSV format with numerical data
- Column headers as variable names
- No missing values (clean beforehand)
- Recommended: ≤50 variables for performance

**Upload Process:**
1. Click "Upload Custom Dataset" in the web interface
2. Select CSV file and provide dataset name/description
3. Optional: Upload BIF file for ground truth structure
4. Dataset becomes available in dropdown

## Web Interface Guide

### Main Components

#### **1. Dataset Panel (Left Sidebar)**
- **Dataset Selection**: Dropdown with all available datasets
- **Dataset Info**: Displays variables, samples, edges, type
- **Algorithm Selection**: Choose from 5 available algorithms
- **Parameter Configuration**: Interactive parameter controls with validation
- **Upload Interface**: Custom dataset upload with BIF support

#### **2. Main Tabs**

**Data Preview Tab:**
- Tabular view of dataset with pagination
- Basic statistics (mean, std, min, max)
- Data type detection and validation warnings

**Results Tab:**
- Performance metrics cards (when ground truth available)
- Interactive causal graph visualization (Plotly)
- Adjacency matrix heatmap with node labels
- Execution time and convergence information

**Saved Results Tab:**
- Browse all saved algorithm runs
- Search and filter by dataset, algorithm, date
- View, compare, and delete saved results
- Bulk operations and result management

**Log Tab:**
- Real-time algorithm progress
- Parameter logging and validation messages
- Error reporting and debugging information
- Session-specific output (no mixed logs)

### Advanced Features

#### **Real-time Progress Monitoring**
- Live iteration updates during algorithm execution
- Convergence metrics (h-value, ρ penalty) for NOTEARS
- Progress bars for bnlearn algorithms
- Session isolation prevents mixed output

#### **Interactive Visualization**
- **Causal Graphs**: Node positioning, zoom, pan, hover
- **Edge Inspection**: Click edges to see weights/strengths
- **Layout Options**: Force-directed, circular, hierarchical
- **Export**: PNG, SVG, HTML formats

#### **Results Management System**
- **Persistent Storage**: Results saved across sessions
- **Metadata Tracking**: Algorithm, parameters, performance, timing
- **Export Options**: CSV (adjacency), JSON (complete), PNG (visualization)
- **Comparison Tools**: Side-by-side result comparison

### Performance Tips

#### **Parameter Tuning**
- **Start Small**: Use simple datasets (chain, fork) for parameter exploration
- **Threshold Sensitivity**: Lower values (0.05-0.15) for sparse networks
- **Regularization Balance**: Higher lambda1 for sparsity, lambda2 for stability
- **Iteration Monitoring**: Watch h-value convergence for NOTEARS

#### **Dataset Recommendations**
- **Learning**: Begin with `chain` or `fork` (4 variables)
- **Validation**: Use `asia` or `alarm` for realistic complexity
- **Benchmarking**: Try `bnlearn_sprinkler` for algorithm comparison
- **Challenge**: Attempt `bnlearn_andes` for large-scale testing

## Architecture & Components

### Modular Algorithm System
```
algorithms/
├── base_algorithm.py       # Abstract interface
├── notears_linear.py       # Linear implementation
├── notears_nonlinear.py    # Nonlinear implementation  
├── bnlearn_algorithms.py   # bnlearn integration
├── algorithm_registry.py   # Algorithm factory
└── __init__.py            # Module exports
```

### Web Application Stack
- **Backend**: Flask with Socket.IO for real-time communication
- **Frontend**: Bootstrap + JavaScript + Plotly
- **Session Management**: Per-session algorithm tracking
- **Data Storage**: JSON + CSV persistent storage
- **File Management**: Organized dataset and results structure

### Utilities & Tools
- **notears_utils.py**: BIF parsing, metrics, thresholding
- **launch_gui.py**: Application launcher with dependency checking

## Troubleshooting

### Common Issues

#### **Installation Problems**
```bash
# Missing Python headers
sudo dnf install python3-devel  # Fedora
sudo apt install python3-dev    # Ubuntu

# Compilation issues
pip install --upgrade pip setuptools wheel
pip install --no-cache-dir -r requirements.txt
```

#### **Runtime Issues**
- **Memory Errors**: Reduce dataset size or `m_hidden` parameter
- **Slow Convergence**: Increase `max_iter` or adjust regularization
- **Poor Results**: Try different algorithms or parameter combinations
- **Web Interface Lag**: Check browser console for JavaScript errors

#### **Algorithm-Specific**
- **NOTEARS Numerical Warnings**: Normal for complex optimization
- **bnlearn Import Errors**: If you see "bnlearn package is not available":
  - Ensure bnlearn is installed: `pip install bnlearn` or `conda install -c conda-forge bnlearn`
  - Check same Python environment is being used
  - Restart the web application after installation
- **bnlearn Performance Issues**: Try different discretization bin counts (2-20) or significance levels
- **Empty Results**: Lower threshold parameter or check data quality

### Performance Optimization

#### **System Level**
- Use SSD storage for faster dataset loading
- Ensure adequate RAM (8GB+ recommended)
- Close other applications during large algorithm runs

#### **Algorithm Level**
- Start with linear NOTEARS for baseline
- Use appropriate dataset size for algorithm choice
- Monitor convergence behavior in Log tab
- Save intermediate results for long runs

## Project Structure

```
CausalDiscoveryPlatform/
├── algorithms/                    # Algorithm implementations
│   ├── __init__.py               # Module exports
│   ├── base_algorithm.py         # Abstract base class
│   ├── notears_linear.py         # Linear NOTEARS
│   ├── notears_nonlinear.py      # Nonlinear NOTEARS
│   ├── bnlearn_algorithms.py     # bnlearn integration
│   └── algorithm_registry.py     # Algorithm factory
├── datasets/                      # Dataset collection (19 datasets)
│   ├── [dataset_name]/           # Individual dataset folders
│   │   ├── [name]_data.csv       # Data file
│   │   ├── [name].bif            # Ground truth structure
│   │   └── info.json             # Dataset metadata
│   └── ...
├── data/                         # Active data storage
│   ├── saved_results/            # Algorithm results
│   └── trashcan/                 # Deleted items (recoverable)
├── static/                       # Web interface assets
│   ├── css/style.css             # Styling
│   └── js/app.js                 # JavaScript application
├── templates/                    # HTML templates
│   └── index.html                # Main web interface
├── archive/                      # Legacy files
├── notears_web_gui.py            # Main web application
├── notears_utils.py              # Utility functions
├── launch_gui.py                 # Application launcher
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## API Reference

### Algorithm Interface
```python
from algorithms import get_algorithm, get_available_algorithms

# List available algorithms
algorithms = get_available_algorithms()

# Get specific algorithm
algorithm = get_algorithm('notears_nonlinear')

# Get parameter definitions
params = algorithm.get_parameter_definitions()

# Run algorithm
W_learned = algorithm.run(X, parameters, progress_callback)
```

### Web API Endpoints
- `GET /api/algorithms` - List available algorithms
- `GET /api/datasets` - List available datasets  
- `GET /api/load_dataset/<name>` - Load specific dataset
- `POST /api/upload_dataset` - Upload custom dataset
- `GET /api/saved_results` - List saved results
- Socket.IO events: `run_algorithm`, `algorithm_progress`, `algorithm_completed`

## Contributing

### Development Setup
1. Fork repository and create feature branch
2. Install development dependencies: `pip install -r requirements-dev.txt`
3. Follow PEP 8 style guidelines
4. Add tests for new functionality
5. Update documentation as needed

### Adding New Algorithms
1. Inherit from `BaseAlgorithm` class
2. Implement required methods: `run()`, `get_default_parameters()`, etc.
3. Register in `algorithm_registry.py`
4. Add parameter definitions for web interface
5. Test with various datasets

## References & Citations

- **NOTEARS**: Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018). DAGs with NO TEARS: Continuous optimization for structure learning. *Advances in Neural Information Processing Systems*, 31.
- **Nonlinear NOTEARS**: Zheng, X., Dan, C., Aragam, B., Ravikumar, P., & Xing, E. (2020). Learning sparse nonparametric DAGs. *International Conference on Artificial Intelligence and Statistics*.
- **bnlearn**: Scutari, M. (2010). Learning Bayesian Networks with the bnlearn R Package. *Journal of Statistical Software*, 35(3), 1-22.

## License

This implementation is provided for research and educational purposes. See individual algorithm implementations for specific licensing terms.

---

**Version**: 2.0.0  
**Last Updated**: June 2025  
**Platform**: Python 3.8+ / Web-based Interface