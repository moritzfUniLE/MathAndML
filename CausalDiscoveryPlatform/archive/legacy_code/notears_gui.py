"""
NOTEARS Graphical User Interface
A comprehensive GUI for running NOTEARS causal discovery algorithm with interactive controls.
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import json
import threading
import time
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import networkx as nx
from PIL import Image, ImageTk

# Import NOTEARS components
from notears_core import NotearsMLP, notears_nonlinear, load_ground_truth_from_bif, compute_metrics, apply_threshold
from test_datasets import NOTEARSDatasetTester


class NOTEARSGuiApp:
    """Main GUI application for NOTEARS causal discovery."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("NOTEARS Causal Discovery Tool")
        self.root.geometry("1200x800")
        self.root.configure(bg='#f0f0f0')
        
        # Application state
        self.current_dataset = None
        self.current_data = None
        self.current_ground_truth = None
        self.current_node_names = None
        self.last_result = None
        self.is_running = False
        
        # Algorithm parameters
        self.params = {
            'lambda1': tk.DoubleVar(value=0.01),
            'lambda2': tk.DoubleVar(value=0.01),
            'max_iter': tk.IntVar(value=50),
            'hidden_units': tk.IntVar(value=10),
            'threshold': tk.DoubleVar(value=0.1)
        }
        
        self.setup_gui()
        self.load_available_datasets()
    
    def setup_gui(self):
        """Initialize the GUI layout."""
        # Create main menu
        self.create_menu()
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create tabs
        self.create_dataset_tab()
        self.create_algorithm_tab()
        self.create_results_tab()
        
        # Status bar
        self.create_status_bar()
    
    def create_menu(self):
        """Create the application menu bar."""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Import Dataset...", command=self.import_dataset)
        file_menu.add_command(label="Export Results...", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        # Tools menu
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        tools_menu.add_command(label="Generate Synthetic Data...", command=self.generate_synthetic_data)
        tools_menu.add_command(label="Validate Dataset", command=self.validate_dataset)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)
        help_menu.add_command(label="User Guide", command=self.show_help)
    
    def create_dataset_tab(self):
        """Create the dataset selection and import tab."""
        dataset_frame = ttk.Frame(self.notebook)
        self.notebook.add(dataset_frame, text="Dataset")
        
        # Dataset selection section
        selection_frame = ttk.LabelFrame(dataset_frame, text="Dataset Selection", padding=10)
        selection_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Built-in datasets
        ttk.Label(selection_frame, text="Built-in Datasets:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.dataset_var = tk.StringVar()
        self.dataset_combo = ttk.Combobox(selection_frame, textvariable=self.dataset_var, 
                                         width=30, state="readonly")
        self.dataset_combo.grid(row=0, column=1, padx=10, pady=5)
        self.dataset_combo.bind('<<ComboboxSelected>>', self.on_dataset_selected)
        
        ttk.Button(selection_frame, text="Load Dataset", 
                  command=self.load_selected_dataset).grid(row=0, column=2, padx=5, pady=5)
        
        # Custom dataset import
        ttk.Label(selection_frame, text="Custom Dataset:").grid(row=1, column=0, sticky=tk.W, pady=5)
        ttk.Button(selection_frame, text="Import CSV File...", 
                  command=self.import_dataset).grid(row=1, column=1, sticky=tk.W, padx=10, pady=5)
        ttk.Button(selection_frame, text="Import Ground Truth (BIF)...", 
                  command=self.import_ground_truth).grid(row=1, column=2, padx=5, pady=5)
        
        # Dataset information
        info_frame = ttk.LabelFrame(dataset_frame, text="Dataset Information", padding=10)
        info_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.dataset_info = scrolledtext.ScrolledText(info_frame, height=10, wrap=tk.WORD)
        self.dataset_info.pack(fill=tk.BOTH, expand=True)
        
        # Data preview
        preview_frame = ttk.LabelFrame(dataset_frame, text="Data Preview", padding=10)
        preview_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Create treeview for data preview
        self.data_tree = ttk.Treeview(preview_frame, height=8)
        self.data_tree.pack(fill=tk.BOTH, expand=True)
        
        # Scrollbar for treeview
        tree_scroll = ttk.Scrollbar(preview_frame, orient=tk.VERTICAL, command=self.data_tree.yview)
        tree_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.data_tree.configure(yscrollcommand=tree_scroll.set)
    
    def create_algorithm_tab(self):
        """Create the algorithm configuration and execution tab."""
        algorithm_frame = ttk.Frame(self.notebook)
        self.notebook.add(algorithm_frame, text="Algorithm")
        
        # Algorithm selection
        algo_frame = ttk.LabelFrame(algorithm_frame, text="Algorithm Selection", padding=10)
        algo_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.algorithm_var = tk.StringVar(value="NOTEARS Nonlinear")
        ttk.Label(algo_frame, text="Algorithm:").grid(row=0, column=0, sticky=tk.W, pady=5)
        algo_combo = ttk.Combobox(algo_frame, textvariable=self.algorithm_var, 
                                 values=["NOTEARS Nonlinear"], state="readonly", width=30)
        algo_combo.grid(row=0, column=1, padx=10, pady=5)
        
        # Parameter configuration
        param_frame = ttk.LabelFrame(algorithm_frame, text="Parameters", padding=10)
        param_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Lambda1 (L1 regularization)
        ttk.Label(param_frame, text="Lambda1 (L1 reg):").grid(row=0, column=0, sticky=tk.W, pady=5)
        lambda1_spin = ttk.Spinbox(param_frame, from_=0.001, to=1.0, increment=0.001, 
                                  textvariable=self.params['lambda1'], width=15, format="%.3f")
        lambda1_spin.grid(row=0, column=1, padx=10, pady=5)
        ttk.Label(param_frame, text="Sparsity penalty").grid(row=0, column=2, sticky=tk.W, padx=5)
        
        # Lambda2 (L2 regularization)
        ttk.Label(param_frame, text="Lambda2 (L2 reg):").grid(row=1, column=0, sticky=tk.W, pady=5)
        lambda2_spin = ttk.Spinbox(param_frame, from_=0.0, to=1.0, increment=0.001, 
                                  textvariable=self.params['lambda2'], width=15, format="%.3f")
        lambda2_spin.grid(row=1, column=1, padx=10, pady=5)
        ttk.Label(param_frame, text="Weight decay").grid(row=1, column=2, sticky=tk.W, padx=5)
        
        # Max iterations
        ttk.Label(param_frame, text="Max Iterations:").grid(row=2, column=0, sticky=tk.W, pady=5)
        iter_spin = ttk.Spinbox(param_frame, from_=1, to=500, increment=1, 
                               textvariable=self.params['max_iter'], width=15)
        iter_spin.grid(row=2, column=1, padx=10, pady=5)
        ttk.Label(param_frame, text="Optimization steps").grid(row=2, column=2, sticky=tk.W, padx=5)
        
        # Hidden units
        ttk.Label(param_frame, text="Hidden Units:").grid(row=3, column=0, sticky=tk.W, pady=5)
        hidden_spin = ttk.Spinbox(param_frame, from_=1, to=100, increment=1, 
                                 textvariable=self.params['hidden_units'], width=15)
        hidden_spin.grid(row=3, column=1, padx=10, pady=5)
        ttk.Label(param_frame, text="Neural network width").grid(row=3, column=2, sticky=tk.W, padx=5)
        
        # Threshold
        ttk.Label(param_frame, text="Edge Threshold:").grid(row=4, column=0, sticky=tk.W, pady=5)
        thresh_spin = ttk.Spinbox(param_frame, from_=0.0, to=1.0, increment=0.01, 
                                 textvariable=self.params['threshold'], width=15, format="%.2f")
        thresh_spin.grid(row=4, column=1, padx=10, pady=5)
        ttk.Label(param_frame, text="Edge detection cutoff").grid(row=4, column=2, sticky=tk.W, padx=5)
        
        # Execution controls
        exec_frame = ttk.LabelFrame(algorithm_frame, text="Execution", padding=10)
        exec_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.run_button = ttk.Button(exec_frame, text="Run NOTEARS", command=self.run_algorithm)
        self.run_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(exec_frame, text="Stop", command=self.stop_algorithm, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(exec_frame, mode='indeterminate')
        self.progress.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        
        # Output log
        log_frame = ttk.LabelFrame(algorithm_frame, text="Execution Log", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)
    
    def create_results_tab(self):
        """Create the results visualization and analysis tab."""
        results_frame = ttk.Frame(self.notebook)
        self.notebook.add(results_frame, text="Results")
        
        # Results summary
        summary_frame = ttk.LabelFrame(results_frame, text="Results Summary", padding=10)
        summary_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.results_text = tk.Text(summary_frame, height=6, wrap=tk.WORD)
        self.results_text.pack(fill=tk.X)
        
        # Visualization controls
        viz_frame = ttk.LabelFrame(results_frame, text="Visualizations", padding=10)
        viz_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Button(viz_frame, text="Show Graph Comparison", 
                  command=self.show_graph_comparison).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_frame, text="Show Heatmaps", 
                  command=self.show_heatmaps).pack(side=tk.LEFT, padx=5)
        ttk.Button(viz_frame, text="Export Results", 
                  command=self.export_results).pack(side=tk.LEFT, padx=5)
        
        # Matplotlib canvas for inline plots
        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.figure, results_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    def create_status_bar(self):
        """Create the status bar at the bottom."""
        self.status_frame = ttk.Frame(self.root)
        self.status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_var = tk.StringVar(value="Ready")
        self.status_label = ttk.Label(self.status_frame, textvariable=self.status_var)
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Dataset status
        self.dataset_status_var = tk.StringVar(value="No dataset loaded")
        self.dataset_status_label = ttk.Label(self.status_frame, textvariable=self.dataset_status_var)
        self.dataset_status_label.pack(side=tk.RIGHT, padx=10, pady=5)
    
    def load_available_datasets(self):
        """Load list of available built-in datasets."""
        datasets_dir = "datasets"
        available_datasets = []
        
        if os.path.exists(datasets_dir):
            for item in os.listdir(datasets_dir):
                dataset_path = os.path.join(datasets_dir, item)
                if os.path.isdir(dataset_path):
                    info_file = os.path.join(dataset_path, "info.json")
                    if os.path.exists(info_file):
                        available_datasets.append(item)
        
        self.dataset_combo['values'] = available_datasets
        if available_datasets:
            self.dataset_combo.set(available_datasets[0])
    
    def on_dataset_selected(self, event=None):
        """Handle dataset selection from dropdown."""
        selected = self.dataset_var.get()
        if selected:
            self.load_dataset_info(selected)
    
    def load_dataset_info(self, dataset_name: str):
        """Load and display information about a dataset."""
        info_file = os.path.join("datasets", dataset_name, "info.json")
        if os.path.exists(info_file):
            try:
                with open(info_file, 'r') as f:
                    info = json.load(f)
                
                info_text = f"Dataset: {info['name']}\n"
                info_text += f"Description: {info['description']}\n"
                info_text += f"Type: {info['type']}\n"
                info_text += f"Nodes: {info['nodes']}\n"
                info_text += f"Edges: {info['edges']}\n"
                info_text += f"Samples: {info.get('samples', 'Unknown')}\n"
                info_text += f"Variables: {', '.join(info['variables'])}\n"
                
                self.dataset_info.delete(1.0, tk.END)
                self.dataset_info.insert(1.0, info_text)
                
            except Exception as e:
                self.log(f"Error loading dataset info: {e}")
    
    def load_selected_dataset(self):
        """Load the currently selected dataset."""
        dataset_name = self.dataset_var.get()
        if not dataset_name:
            messagebox.showwarning("No Selection", "Please select a dataset first.")
            return
        
        try:
            # Load data
            data_file = os.path.join("datasets", dataset_name, f"{dataset_name}_data.csv")
            if os.path.exists(data_file):
                self.current_data = pd.read_csv(data_file)
                self.current_dataset = dataset_name
                
                # Load ground truth if available
                bif_file = os.path.join("datasets", dataset_name, f"{dataset_name}.bif")
                if os.path.exists(bif_file):
                    W_true, node_names, success = load_ground_truth_from_bif(bif_file)
                    if success:
                        self.current_ground_truth = W_true
                        self.current_node_names = node_names
                
                self.update_data_preview()
                self.dataset_status_var.set(f"Loaded: {dataset_name} ({self.current_data.shape[0]} samples, {self.current_data.shape[1]} variables)")
                self.log(f"Loaded dataset: {dataset_name}")
                
            else:
                messagebox.showerror("Error", f"Data file not found: {data_file}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load dataset: {e}")
            self.log(f"Error loading dataset: {e}")
    
    def update_data_preview(self):
        """Update the data preview table."""
        if self.current_data is None:
            return
        
        # Clear existing data
        for item in self.data_tree.get_children():
            self.data_tree.delete(item)
        
        # Set columns
        columns = list(self.current_data.columns)
        self.data_tree['columns'] = columns
        self.data_tree['show'] = 'headings'
        
        # Configure column headings
        for col in columns:
            self.data_tree.heading(col, text=col)
            self.data_tree.column(col, width=80)
        
        # Insert data (first 100 rows)
        for idx, row in self.current_data.head(100).iterrows():
            values = [f"{val:.3f}" if isinstance(val, (int, float)) else str(val) for val in row]
            self.data_tree.insert('', 'end', values=values)
    
    def import_dataset(self):
        """Import a custom CSV dataset."""
        file_path = filedialog.askopenfilename(
            title="Select CSV Data File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                self.current_data = pd.read_csv(file_path)
                self.current_dataset = os.path.basename(file_path)
                self.current_ground_truth = None
                self.current_node_names = list(self.current_data.columns)
                
                self.update_data_preview()
                self.dataset_status_var.set(f"Imported: {self.current_dataset} ({self.current_data.shape[0]} samples, {self.current_data.shape[1]} variables)")
                self.log(f"Imported dataset from: {file_path}")
                
                # Update dataset info
                info_text = f"Custom Dataset: {self.current_dataset}\n"
                info_text += f"Samples: {self.current_data.shape[0]}\n"
                info_text += f"Variables: {self.current_data.shape[1]}\n"
                info_text += f"Columns: {', '.join(self.current_data.columns)}\n"
                info_text += f"Data types: {dict(self.current_data.dtypes)}\n"
                
                self.dataset_info.delete(1.0, tk.END)
                self.dataset_info.insert(1.0, info_text)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import dataset: {e}")
                self.log(f"Error importing dataset: {e}")
    
    def import_ground_truth(self):
        """Import ground truth BIF file."""
        file_path = filedialog.askopenfilename(
            title="Select BIF Ground Truth File",
            filetypes=[("BIF files", "*.bif"), ("All files", "*.*")]
        )
        
        if file_path:
            try:
                W_true, node_names, success = load_ground_truth_from_bif(file_path)
                if success:
                    self.current_ground_truth = W_true
                    if node_names:
                        self.current_node_names = node_names
                    self.log(f"Imported ground truth from: {file_path}")
                    messagebox.showinfo("Success", "Ground truth imported successfully!")
                else:
                    messagebox.showerror("Error", "Failed to parse BIF file")
                    
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import ground truth: {e}")
                self.log(f"Error importing ground truth: {e}")
    
    def run_algorithm(self):
        """Run the NOTEARS algorithm in a separate thread."""
        if self.current_data is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        if self.is_running:
            messagebox.showwarning("Already Running", "Algorithm is already running.")
            return
        
        # Start algorithm in separate thread
        self.is_running = True
        self.run_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress.start()
        self.status_var.set("Running NOTEARS algorithm...")
        
        # Clear previous results
        self.last_result = None
        self.results_text.delete(1.0, tk.END)
        self.figure.clear()
        self.canvas.draw()
        
        # Start algorithm thread
        self.algorithm_thread = threading.Thread(target=self._run_algorithm_thread)
        self.algorithm_thread.daemon = True
        self.algorithm_thread.start()
    
    def _run_algorithm_thread(self):
        """Execute the algorithm in a separate thread."""
        try:
            self.log("Starting NOTEARS algorithm...")
            
            # Get parameters
            lambda1 = self.params['lambda1'].get()
            lambda2 = self.params['lambda2'].get()
            max_iter = self.params['max_iter'].get()
            hidden_units = self.params['hidden_units'].get()
            threshold = self.params['threshold'].get()
            
            self.log(f"Parameters: λ1={lambda1}, λ2={lambda2}, max_iter={max_iter}, hidden={hidden_units}, threshold={threshold}")
            
            # Prepare data
            X = self.current_data.values.astype(np.float32)
            d = X.shape[1]
            
            # Initialize model
            model = NotearsMLP(d, m_hidden=hidden_units)
            
            # Run NOTEARS
            start_time = time.time()
            W_learned = notears_nonlinear(
                model, X,
                lambda1=lambda1,
                lambda2=lambda2,
                max_iter=max_iter
            )
            runtime = time.time() - start_time
            
            # Compute metrics if ground truth available
            metrics = None
            if self.current_ground_truth is not None:
                metrics = compute_metrics(W_learned, self.current_ground_truth, thresh=threshold)
            
            # Store results
            self.last_result = {
                'W_learned': W_learned,
                'W_true': self.current_ground_truth,
                'node_names': self.current_node_names or [f"X{i}" for i in range(d)],
                'runtime': runtime,
                'metrics': metrics,
                'parameters': {
                    'lambda1': lambda1,
                    'lambda2': lambda2,
                    'max_iter': max_iter,
                    'hidden_units': hidden_units,
                    'threshold': threshold
                }
            }
            
            # Update UI in main thread
            self.root.after(0, self._algorithm_completed)
            
        except Exception as e:
            self.log(f"Algorithm failed: {e}")
            self.root.after(0, lambda: self._algorithm_failed(str(e)))
    
    def _algorithm_completed(self):
        """Handle algorithm completion."""
        self.is_running = False
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress.stop()
        self.status_var.set("Algorithm completed successfully")
        
        # Display results
        self.display_results()
        self.log(f"Algorithm completed in {self.last_result['runtime']:.2f} seconds")
        
        # Switch to results tab
        self.notebook.select(2)
    
    def _algorithm_failed(self, error_msg: str):
        """Handle algorithm failure."""
        self.is_running = False
        self.run_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.progress.stop()
        self.status_var.set("Algorithm failed")
        
        messagebox.showerror("Algorithm Error", f"Algorithm failed: {error_msg}")
    
    def stop_algorithm(self):
        """Stop the running algorithm."""
        if self.is_running:
            self.is_running = False
            self.run_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.progress.stop()
            self.status_var.set("Algorithm stopped by user")
            self.log("Algorithm stopped by user")
    
    def display_results(self):
        """Display algorithm results."""
        if self.last_result is None:
            return
        
        # Create results summary
        summary = f"NOTEARS Results Summary\n"
        summary += f"{'='*40}\n"
        summary += f"Runtime: {self.last_result['runtime']:.2f} seconds\n"
        summary += f"Dataset: {self.current_dataset or 'Custom'}\n"
        summary += f"Variables: {len(self.last_result['node_names'])}\n"
        
        W_learned = self.last_result['W_learned']
        threshold = self.last_result['parameters']['threshold']
        W_learned_thresh = apply_threshold(W_learned, threshold)
        n_learned_edges = np.sum(W_learned_thresh != 0)
        summary += f"Learned edges: {n_learned_edges}\n"
        
        if self.last_result['metrics']:
            metrics = self.last_result['metrics']
            summary += f"\nEvaluation Metrics:\n"
            summary += f"Precision: {metrics['precision']:.3f}\n"
            summary += f"Recall: {metrics['recall']:.3f}\n"
            summary += f"F1-Score: {metrics['f1_score']:.3f}\n"
            summary += f"Hamming Distance: {metrics['hamming_distance']}\n"
            
            if self.last_result['W_true'] is not None:
                n_true_edges = np.sum(self.last_result['W_true'] != 0)
                summary += f"True edges: {n_true_edges}\n"
        
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, summary)
        
        # Show initial visualization
        self.show_graph_comparison()
    
    def show_graph_comparison(self):
        """Show side-by-side graph comparison."""
        if self.last_result is None:
            messagebox.showwarning("No Results", "No results to display. Run the algorithm first.")
            return
        
        self.figure.clear()
        
        W_learned = self.last_result['W_learned']
        W_true = self.last_result['W_true']
        node_names = self.last_result['node_names']
        threshold = self.last_result['parameters']['threshold']
        
        W_learned_thresh = apply_threshold(W_learned, threshold)
        
        if W_true is not None:
            # Create side-by-side comparison
            ax1 = self.figure.add_subplot(121)
            ax2 = self.figure.add_subplot(122)
            
            # Create graphs
            G_true = self._create_networkx_graph(W_true, node_names)
            G_learned = self._create_networkx_graph(W_learned_thresh, node_names)
            
            # Use consistent layout
            pos = nx.spring_layout(G_true, seed=42, k=2, iterations=50)
            
            # Ground truth
            ax1.set_title("Ground Truth DAG")
            nx.draw(G_true, pos, ax=ax1, with_labels=True, node_color='lightblue',
                   node_size=800, font_size=8, font_weight='bold', arrows=True)
            
            # Learned graph with colored edges
            ax2.set_title(f"Learned DAG (threshold={threshold})")
            
            # Categorize edges
            true_edges = set(G_true.edges())
            learned_edges = set(G_learned.edges())
            correct_edges = true_edges.intersection(learned_edges)
            false_edges = learned_edges.difference(true_edges)
            missing_edges = true_edges.difference(learned_edges)
            
            # Draw nodes
            nx.draw_networkx_nodes(G_learned, pos, ax=ax2, node_color='lightcoral', node_size=800)
            
            # Draw edges with colors
            if correct_edges:
                nx.draw_networkx_edges(G_learned, pos, edgelist=list(correct_edges), 
                                     ax=ax2, edge_color='green', width=2, arrows=True)
            if false_edges:
                nx.draw_networkx_edges(G_learned, pos, edgelist=list(false_edges), 
                                     ax=ax2, edge_color='red', width=2, arrows=True)
            if missing_edges:
                nx.draw_networkx_edges(G_true, pos, edgelist=list(missing_edges), 
                                     ax=ax2, edge_color='orange', width=1, style='dashed', arrows=True)
            
            # Draw labels
            nx.draw_networkx_labels(G_learned, pos, ax=ax2, font_size=8, font_weight='bold')
            
        else:
            # Only learned graph
            ax = self.figure.add_subplot(111)
            G_learned = self._create_networkx_graph(W_learned_thresh, node_names)
            pos = nx.spring_layout(G_learned, seed=42, k=2, iterations=50)
            
            ax.set_title(f"Learned DAG (threshold={threshold})")
            nx.draw(G_learned, pos, ax=ax, with_labels=True, node_color='lightcoral',
                   node_size=800, font_size=8, font_weight='bold', arrows=True)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def show_heatmaps(self):
        """Show adjacency matrix heatmaps."""
        if self.last_result is None:
            messagebox.showwarning("No Results", "No results to display. Run the algorithm first.")
            return
        
        self.figure.clear()
        
        W_learned = self.last_result['W_learned']
        W_true = self.last_result['W_true']
        node_names = self.last_result['node_names']
        threshold = self.last_result['parameters']['threshold']
        
        W_learned_thresh = apply_threshold(W_learned, threshold)
        
        if W_true is not None:
            # Show ground truth and learned matrices
            ax1 = self.figure.add_subplot(121)
            ax2 = self.figure.add_subplot(122)
            
            im1 = ax1.imshow(W_true, cmap='Blues', aspect='auto')
            ax1.set_title("Ground Truth")
            ax1.set_xticks(range(len(node_names)))
            ax1.set_yticks(range(len(node_names)))
            ax1.set_xticklabels(node_names, rotation=45)
            ax1.set_yticklabels(node_names)
            self.figure.colorbar(im1, ax=ax1)
            
            im2 = ax2.imshow(W_learned_thresh, cmap='Reds', aspect='auto')
            ax2.set_title(f"Learned (threshold={threshold})")
            ax2.set_xticks(range(len(node_names)))
            ax2.set_yticks(range(len(node_names)))
            ax2.set_xticklabels(node_names, rotation=45)
            ax2.set_yticklabels(node_names)
            self.figure.colorbar(im2, ax=ax2)
            
        else:
            # Only learned matrix
            ax = self.figure.add_subplot(111)
            im = ax.imshow(W_learned_thresh, cmap='Reds', aspect='auto')
            ax.set_title(f"Learned Adjacency Matrix (threshold={threshold})")
            ax.set_xticks(range(len(node_names)))
            ax.set_yticks(range(len(node_names)))
            ax.set_xticklabels(node_names, rotation=45)
            ax.set_yticklabels(node_names)
            self.figure.colorbar(im, ax=ax)
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def _create_networkx_graph(self, adj_matrix: np.ndarray, node_names: List[str]) -> nx.DiGraph:
        """Create NetworkX graph from adjacency matrix."""
        G = nx.DiGraph()
        
        # Add nodes
        for i, name in enumerate(node_names):
            G.add_node(i, label=name)
        
        # Add edges
        for i in range(len(node_names)):
            for j in range(len(node_names)):
                if adj_matrix[i, j] > 0:
                    G.add_edge(i, j, weight=adj_matrix[i, j])
        
        return G
    
    def export_results(self):
        """Export results to files."""
        if self.last_result is None:
            messagebox.showwarning("No Results", "No results to export. Run the algorithm first.")
            return
        
        # Select directory
        export_dir = filedialog.askdirectory(title="Select Export Directory")
        if not export_dir:
            return
        
        try:
            # Export adjacency matrices as CSV
            W_learned = self.last_result['W_learned']
            W_true = self.last_result['W_true']
            node_names = self.last_result['node_names']
            threshold = self.last_result['parameters']['threshold']
            
            W_learned_thresh = apply_threshold(W_learned, threshold)
            
            # Save learned matrix
            df_learned = pd.DataFrame(W_learned_thresh, index=node_names, columns=node_names)
            df_learned.to_csv(os.path.join(export_dir, "adjacency_learned.csv"))
            
            # Save ground truth if available
            if W_true is not None:
                df_true = pd.DataFrame(W_true, index=node_names, columns=node_names)
                df_true.to_csv(os.path.join(export_dir, "adjacency_ground_truth.csv"))
            
            # Save results summary
            with open(os.path.join(export_dir, "results_summary.json"), 'w') as f:
                export_data = {
                    'runtime': self.last_result['runtime'],
                    'parameters': self.last_result['parameters'],
                    'metrics': self.last_result['metrics'],
                    'node_names': node_names,
                    'dataset': self.current_dataset
                }
                json.dump(export_data, f, indent=2)
            
            # Save current visualization
            self.figure.savefig(os.path.join(export_dir, "visualization.png"), dpi=150, bbox_inches='tight')
            
            messagebox.showinfo("Export Complete", f"Results exported to: {export_dir}")
            self.log(f"Results exported to: {export_dir}")
            
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export results: {e}")
            self.log(f"Error exporting results: {e}")
    
    def generate_synthetic_data(self):
        """Open dialog to generate synthetic data."""
        # This could open the create_test_datasets.py functionality
        messagebox.showinfo("Feature", "Synthetic data generation will be implemented in a future version.\n\nFor now, please use: python create_test_datasets.py")
    
    def validate_dataset(self):
        """Validate the current dataset."""
        if self.current_data is None:
            messagebox.showwarning("No Data", "Please load a dataset first.")
            return
        
        try:
            # Basic validation
            issues = []
            
            # Check for missing values
            if self.current_data.isnull().any().any():
                issues.append("Dataset contains missing values")
            
            # Check data types
            non_numeric = []
            for col in self.current_data.columns:
                if not np.issubdtype(self.current_data[col].dtype, np.number):
                    non_numeric.append(col)
            
            if non_numeric:
                issues.append(f"Non-numeric columns found: {non_numeric}")
            
            # Check for constant columns
            constant_cols = []
            for col in self.current_data.columns:
                if self.current_data[col].nunique() == 1:
                    constant_cols.append(col)
            
            if constant_cols:
                issues.append(f"Constant columns found: {constant_cols}")
            
            # Display results
            if issues:
                issue_text = "Dataset validation issues found:\n\n" + "\n".join(f"• {issue}" for issue in issues)
                messagebox.showwarning("Validation Issues", issue_text)
            else:
                messagebox.showinfo("Validation Passed", "Dataset validation passed successfully!")
            
        except Exception as e:
            messagebox.showerror("Validation Error", f"Error validating dataset: {e}")
    
    def show_about(self):
        """Show about dialog."""
        about_text = """NOTEARS Causal Discovery Tool
        
A graphical interface for running the NOTEARS (No TEArs Recursive Structure learning) algorithm for learning nonlinear causal DAGs from observational data.

Features:
• Interactive algorithm configuration
• Dataset import and management
• Real-time result visualization
• Performance evaluation metrics
• Export functionality

Version: 1.0
Framework: Python/Tkinter"""
        
        messagebox.showinfo("About", about_text)
    
    def show_help(self):
        """Show help dialog."""
        help_text = """NOTEARS GUI User Guide

1. DATASET TAB:
   • Select from built-in synthetic datasets
   • Import custom CSV files
   • Import ground truth BIF files for evaluation
   • Preview dataset contents

2. ALGORITHM TAB:
   • Configure algorithm parameters
   • Lambda1: L1 regularization (sparsity)
   • Lambda2: L2 regularization (stability)
   • Max Iterations: Optimization steps
   • Hidden Units: Neural network width
   • Threshold: Edge detection cutoff
   • Run algorithm and monitor progress

3. RESULTS TAB:
   • View results summary and metrics
   • Interactive graph visualizations
   • Heatmap comparisons
   • Export results to files

Tips:
• Start with small datasets for parameter tuning
• Use threshold to control edge sensitivity
• Monitor the execution log for progress
• Compare with ground truth when available"""
        
        messagebox.showinfo("Help", help_text)
    
    def log(self, message: str):
        """Add message to the execution log."""
        timestamp = time.strftime("%H:%M:%S")
        log_message = f"[{timestamp}] {message}\n"
        
        self.log_text.insert(tk.END, log_message)
        self.log_text.see(tk.END)
        self.root.update_idletasks()


def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = NOTEARSGuiApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()