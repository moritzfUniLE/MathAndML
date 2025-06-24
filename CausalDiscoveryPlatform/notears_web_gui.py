"""
NOTEARS Web-based Graphical User Interface
A comprehensive web GUI for running NOTEARS causal discovery algorithm with interactive controls.
"""
import os
import json
import io
import base64
import threading
import time
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import networkx as nx
from flask import Flask, render_template, request, jsonify, send_from_directory, send_file
from flask_socketio import SocketIO, emit
import plotly.graph_objects as go
import plotly.express as px
from plotly.utils import PlotlyJSONEncoder

# Import NOTEARS components
from notears_utils import load_ground_truth_from_bif, compute_metrics, apply_threshold, create_artifical_dataset, adjacency_matrix_to_bif
from algorithms import get_available_algorithms, get_algorithm


class NOTEARSWebApp:
    """Web-based GUI application for NOTEARS causal discovery."""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'notears_secret_key'
        self.socketio = SocketIO(self.app, cors_allowed_origins="*")
        
        # Application state
        self.current_dataset = None
        self.current_data = None
        self.current_ground_truth = None
        self.current_node_names = None
        self.last_result = None
        # Session-based algorithm execution tracking
        self.running_sessions = {}  # Track running algorithms per session
        self.algorithm_threads = {}  # Track threads per session
        
        # Fix: Add is_running attribute initialization
        self.is_running = True

        # Setup routes and socket handlers
        self.setup_routes()
        self.setup_socket_handlers()
    
    def setup_routes(self):
        """Setup Flask routes."""
        
        @self.app.route('/')
        def index():
            return render_template('index.html')
        
        @self.app.route('/api/algorithms')
        def get_algorithms():
            """Get list of available algorithms."""
            try:
                algorithms = get_available_algorithms()
                return jsonify(algorithms)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/datasets')
        def get_datasets():
            """Get list of available datasets."""
            datasets = []
            datasets_dir = "datasets"
            
            if os.path.exists(datasets_dir):
                for item in os.listdir(datasets_dir):
                    dataset_path = os.path.join(datasets_dir, item)
                    if os.path.isdir(dataset_path):
                        info_file = os.path.join(dataset_path, "info.json")
                        if os.path.exists(info_file):
                            try:
                                with open(info_file, 'r') as f:
                                    info = json.load(f)
                                datasets.append({
                                    'name': item,
                                    'description': info.get('description', ''),
                                    'nodes': info.get('nodes', 0),
                                    'edges': info.get('edges', 0),
                                    'samples': info.get('samples', 0)
                                })
                            except:
                                pass
            
            return jsonify(datasets)
        
        @self.app.route('/api/load_dataset/<dataset_name>')
        def load_dataset(dataset_name):
            """Load a specific dataset."""
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
                    
                    # Load dataset info
                    info_file = os.path.join("datasets", dataset_name, "info.json")
                    info = {}
                    if os.path.exists(info_file):
                        with open(info_file, 'r') as f:
                            info = json.load(f)
                    
                    return jsonify({
                        'success': True,
                        'dataset_name': dataset_name,
                        'shape': self.current_data.shape,
                        'columns': list(self.current_data.columns),
                        'info': info,
                        'preview': self.current_data.head(10).to_dict('records'),
                        'has_ground_truth': self.current_ground_truth is not None
                    })
                else:
                    return jsonify({'success': False, 'error': f'Data file not found: {data_file}'})
                    
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/upload_dataset', methods=['POST'])
        def upload_dataset():
            """Upload a custom CSV dataset with validation."""
            try:
                file = request.files.get('file')
                if not file:
                    return jsonify({'success': False, 'error': 'No file provided'})
                
                # Read and validate CSV data
                data = pd.read_csv(file)
                validation_result = self._validate_dataset(data)
                
                if not validation_result['valid']:
                    return jsonify({'success': False, 'error': validation_result['error']})
                
                # Store temporarily for preview
                self.current_data = data
                self.current_dataset = file.filename
                self.current_ground_truth = None
                self.current_node_names = list(data.columns)
                
                return jsonify({
                    'success': True,
                    'dataset_name': self.current_dataset,
                    'shape': data.shape,
                    'columns': list(data.columns),
                    'preview': data.head(10).to_dict('records'),
                    'has_ground_truth': False,
                    'validation': validation_result
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/validate_dataset', methods=['POST'])
        def validate_dataset():
            """Validate dataset format for NOTEARS compatibility."""
            try:
                file = request.files.get('file')
                if not file:
                    return jsonify({'success': False, 'error': 'No file provided'})
                
                data = pd.read_csv(file)
                validation_result = self._validate_dataset(data)
                
                return jsonify({
                    'success': True,
                    'validation': validation_result,
                    'shape': data.shape,
                    'columns': list(data.columns),
                    'preview': data.head(5).to_dict('records')
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/save_dataset', methods=['POST'])
        def save_dataset():
            """Save uploaded dataset as a preset with custom name."""
            try:
                data = request.get_json()
                dataset_name = data.get('dataset_name', '').strip()
                description = data.get('description', '').strip()
                csv_data = data.get('csv_data')
                bif_content = data.get('bif_content', '')
                
                if not dataset_name:
                    return jsonify({'success': False, 'error': 'Dataset name is required'})
                
                if not csv_data:
                    return jsonify({'success': False, 'error': 'No CSV data provided'})
                
                # Create dataset directory
                dataset_dir = os.path.join('datasets', dataset_name)
                os.makedirs(dataset_dir, exist_ok=True)
                
                # Convert CSV data back to DataFrame and save
                df = pd.DataFrame(csv_data)
                csv_file = os.path.join(dataset_dir, f"{dataset_name}_data.csv")
                df.to_csv(csv_file, index=False)
                
                # Save BIF file if provided
                has_ground_truth = False
                if bif_content.strip():
                    bif_file = os.path.join(dataset_dir, f"{dataset_name}.bif")
                    with open(bif_file, 'w') as f:
                        f.write(bif_content)
                    has_ground_truth = True
                
                # Create info.json
                info = {
                    'name': dataset_name,
                    'description': description,
                    'nodes': len(df.columns),
                    'edges': 0,  # Will be updated if BIF is parsed successfully
                    'samples': len(df),
                    'created_date': pd.Timestamp.now().isoformat(),
                    'has_ground_truth': has_ground_truth
                }
                
                # Try to parse BIF for edge count
                if has_ground_truth:
                    try:
                        W_true, _, success = load_ground_truth_from_bif(bif_file)
                        if success and W_true is not None:
                            info['edges'] = int(np.sum(W_true != 0))
                    except:
                        pass
                
                info_file = os.path.join(dataset_dir, 'info.json')
                with open(info_file, 'w') as f:
                    json.dump(info, f, indent=2)
                
                return jsonify({
                    'success': True,
                    'message': f'Dataset "{dataset_name}" saved successfully',
                    'info': info
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/generate_synthetic_dataset', methods=['POST'])
        def generate_synthetic_dataset():
            """Generate a synthetic dataset with specified parameters."""
            try:
                data = request.get_json()
                
                # Extract parameters
                dataset_name = data.get('dataset_name', '').strip()
                description = data.get('description', '').strip()
                n_samples = int(data.get('n_samples', 1000))
                n_nodes = int(data.get('n_nodes', 10))
                n_edges = int(data.get('n_edges', 20))
                graph_type = data.get('graph_type', 'ER')
                sem_type = data.get('sem_type', 'gauss')
                
                # Validate parameters
                if not dataset_name:
                    return jsonify({'success': False, 'error': 'Dataset name is required'})
                
                if n_samples < 10 or n_samples > 10000:
                    return jsonify({'success': False, 'error': 'Number of samples must be between 10 and 10,000'})
                
                if n_nodes < 2 or n_nodes > 50:
                    return jsonify({'success': False, 'error': 'Number of nodes must be between 2 and 50'})
                
                if n_edges < 1 or n_edges > n_nodes * (n_nodes - 1):
                    return jsonify({'success': False, 'error': f'Number of edges must be between 1 and {n_nodes * (n_nodes - 1)}'})
                
                if graph_type not in ['ER', 'SF', 'BP']:
                    return jsonify({'success': False, 'error': 'Graph type must be ER, SF, or BP'})
                
                if sem_type not in ['gauss', 'exp', 'gumbel', 'uniform', 'logistic', 'poisson']:
                    return jsonify({'success': False, 'error': 'SEM type must be one of: gauss, exp, gumbel, uniform, logistic, poisson'})
                
                # Check if dataset name already exists
                dataset_dir = os.path.join('datasets', dataset_name)
                if os.path.exists(dataset_dir):
                    return jsonify({'success': False, 'error': f'Dataset "{dataset_name}" already exists'})
                
                # Generate synthetic dataset
                try:
                    X, W_true = create_artifical_dataset(n_samples, n_nodes, n_edges, graph_type, sem_type)
                except Exception as e:
                    return jsonify({'success': False, 'error': f'Failed to generate synthetic dataset: {str(e)}'})
                
                # Create dataset directory
                os.makedirs(dataset_dir, exist_ok=True)
                
                # Generate node names
                node_names = [f"X{i+1}" for i in range(n_nodes)]
                
                # Save CSV data
                df = pd.DataFrame(X, columns=node_names)
                csv_file = os.path.join(dataset_dir, f"{dataset_name}_data.csv")
                df.to_csv(csv_file, index=False)
                
                # Generate and save BIF file
                bif_content = adjacency_matrix_to_bif(W_true, node_names)
                bif_file = os.path.join(dataset_dir, f"{dataset_name}.bif")
                with open(bif_file, 'w') as f:
                    f.write(bif_content)
                
                # Calculate actual edge count from ground truth
                actual_edges = int(np.sum(W_true != 0))
                
                # Create info.json
                info = {
                    'name': dataset_name,
                    'description': description or f'Synthetic {graph_type} graph with {sem_type} noise',
                    'nodes': n_nodes,
                    'edges': actual_edges,
                    'samples': n_samples,
                    'created_date': pd.Timestamp.now().isoformat(),
                    'has_ground_truth': True,
                    'synthetic': True,
                    'generation_params': {
                        'graph_type': graph_type,
                        'sem_type': sem_type,
                        'expected_edges': n_edges,
                        'actual_edges': actual_edges
                    }
                }
                
                info_file = os.path.join(dataset_dir, 'info.json')
                with open(info_file, 'w') as f:
                    json.dump(info, f, indent=2)
                
                return jsonify({
                    'success': True,
                    'message': f'Synthetic dataset "{dataset_name}" generated successfully',
                    'info': info,
                    'preview': df.head(10).to_dict('records')
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/update_dataset/<dataset_name>', methods=['POST'])
        def update_dataset(dataset_name):
            """Update existing dataset with edited data."""
            try:
                data = request.get_json()
                csv_data = data.get('csv_data')
                
                if not csv_data:
                    return jsonify({'success': False, 'error': 'No CSV data provided'})
                
                # Validate the edited data
                df = pd.DataFrame(csv_data)
                validation_result = self._validate_dataset(df)
                
                if not validation_result['valid']:
                    return jsonify({'success': False, 'error': validation_result['error']})
                
                # Save updated CSV
                dataset_dir = os.path.join('datasets', dataset_name)
                csv_file = os.path.join(dataset_dir, f"{dataset_name}_data.csv")
                df.to_csv(csv_file, index=False)
                
                # Update info.json
                info_file = os.path.join(dataset_dir, 'info.json')
                if os.path.exists(info_file):
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    info['samples'] = len(df)
                    info['last_modified'] = pd.Timestamp.now().isoformat()
                    with open(info_file, 'w') as f:
                        json.dump(info, f, indent=2)
                
                return jsonify({
                    'success': True,
                    'message': f'Dataset "{dataset_name}" updated successfully'
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/update_dataset_metadata/<dataset_name>', methods=['POST'])
        def update_dataset_metadata(dataset_name):
            """Update dataset metadata (description and BIF file)."""
            try:
                data = request.get_json()
                description = data.get('description', '').strip()
                bif_content = data.get('bif_content', '')
                
                dataset_dir = os.path.join('datasets', dataset_name)
                if not os.path.exists(dataset_dir):
                    return jsonify({'success': False, 'error': f'Dataset "{dataset_name}" not found'})
                
                # Update BIF file if provided
                has_ground_truth = False
                if bif_content.strip():
                    bif_file = os.path.join(dataset_dir, f"{dataset_name}.bif")
                    with open(bif_file, 'w') as f:
                        f.write(bif_content)
                    has_ground_truth = True
                else:
                    # Remove BIF file if empty content provided
                    bif_file = os.path.join(dataset_dir, f"{dataset_name}.bif")
                    if os.path.exists(bif_file):
                        os.remove(bif_file)
                
                # Update info.json
                info_file = os.path.join(dataset_dir, 'info.json')
                info = {}
                if os.path.exists(info_file):
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                
                info['description'] = description
                info['has_ground_truth'] = has_ground_truth
                info['last_modified'] = pd.Timestamp.now().isoformat()
                
                # Try to parse BIF for edge count
                if has_ground_truth:
                    try:
                        W_true, _, success = load_ground_truth_from_bif(bif_file)
                        if success and W_true is not None:
                            info['edges'] = int(np.sum(W_true != 0))
                    except:
                        pass
                else:
                    info['edges'] = 0
                
                with open(info_file, 'w') as f:
                    json.dump(info, f, indent=2)
                
                return jsonify({
                    'success': True,
                    'message': f'Dataset "{dataset_name}" metadata updated successfully'
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/datasets/<dataset_name>/<filename>')
        def serve_dataset_file(dataset_name, filename):
            """Serve dataset files (like BIF files) for editing."""
            try:
                dataset_dir = os.path.join('datasets', dataset_name)
                file_path = os.path.join(dataset_dir, filename)
                
                if os.path.exists(file_path):
                    return send_file(file_path, as_attachment=False, mimetype='text/plain')
                else:
                    return jsonify({'error': 'File not found'}), 404
                    
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/delete_dataset/<dataset_name>', methods=['DELETE'])
        def delete_dataset(dataset_name):
            """Delete (archive) a dataset by moving it to trashcan."""
            try:
                dataset_dir = os.path.join('datasets', dataset_name)
                if not os.path.exists(dataset_dir):
                    return jsonify({'success': False, 'error': f'Dataset "{dataset_name}" not found'})
                
                # Create trashcan directory if it doesn't exist
                trashcan_dir = os.path.join('data', 'trashcan')
                os.makedirs(trashcan_dir, exist_ok=True)
                
                # Move dataset to trashcan
                import shutil
                import time
                
                trashcan_dataset_dir = os.path.join(trashcan_dir, dataset_name)
                
                # If a dataset with same name already exists in trashcan, add timestamp
                if os.path.exists(trashcan_dataset_dir):
                    timestamp = str(int(time.time()))
                    trashcan_dataset_dir = os.path.join(trashcan_dir, f"{dataset_name}_{timestamp}")
                
                shutil.move(dataset_dir, trashcan_dataset_dir)
                
                # Add deletion metadata to the trashcan copy
                info_file = os.path.join(trashcan_dataset_dir, 'info.json')
                if os.path.exists(info_file):
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    info['deleted_date'] = pd.Timestamp.now().isoformat()
                    info['original_location'] = os.path.join('datasets', dataset_name)
                    with open(info_file, 'w') as f:
                        json.dump(info, f, indent=2)
                
                return jsonify({
                    'success': True,
                    'message': f'Dataset "{dataset_name}" moved to trashcan successfully'
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/archived_datasets')
        def get_archived_datasets():
            """Get list of archived (deleted) datasets."""
            try:
                datasets = []
                trashcan_dir = os.path.join('data', 'trashcan')
                
                if os.path.exists(trashcan_dir):
                    for item in os.listdir(trashcan_dir):
                        dataset_path = os.path.join(trashcan_dir, item)
                        if os.path.isdir(dataset_path):
                            info_file = os.path.join(dataset_path, "info.json")
                            if os.path.exists(info_file):
                                try:
                                    with open(info_file, 'r') as f:
                                        info = json.load(f)
                                    
                                    # Extract original dataset name (remove timestamp suffix if present)
                                    original_name = info.get('name', item)
                                    if '_' in item and item.split('_')[-1].isdigit():
                                        original_name = '_'.join(item.split('_')[:-1])
                                    
                                    datasets.append({
                                        'name': original_name,
                                        'folder_name': item,
                                        'description': info.get('description', ''),
                                        'nodes': info.get('nodes', 0),
                                        'edges': info.get('edges', 0),
                                        'samples': info.get('samples', 0),
                                        'deleted_date': info.get('deleted_date', 'Unknown')
                                    })
                                except:
                                    pass
                
                return jsonify({'success': True, 'datasets': datasets})
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/restore_dataset/<dataset_name>', methods=['POST'])
        def restore_dataset(dataset_name):
            """Restore a dataset from the trashcan."""
            try:
                trashcan_dir = os.path.join('data', 'trashcan')
                
                # Find the dataset in trashcan (may have timestamp suffix)
                dataset_folder = None
                for item in os.listdir(trashcan_dir):
                    if item == dataset_name or item.startswith(f"{dataset_name}_"):
                        dataset_folder = item
                        break
                
                if not dataset_folder:
                    return jsonify({'success': False, 'error': f'Dataset "{dataset_name}" not found in trashcan'})
                
                trashcan_dataset_dir = os.path.join(trashcan_dir, dataset_folder)
                restore_target_dir = os.path.join('datasets', dataset_name)
                
                # Check if a dataset with the same name already exists
                if os.path.exists(restore_target_dir):
                    return jsonify({'success': False, 'error': f'Cannot restore: Dataset "{dataset_name}" already exists'})
                
                # Move back to datasets directory
                import shutil
                shutil.move(trashcan_dataset_dir, restore_target_dir)
                
                # Clean up deletion metadata
                info_file = os.path.join(restore_target_dir, 'info.json')
                if os.path.exists(info_file):
                    with open(info_file, 'r') as f:
                        info = json.load(f)
                    
                    # Remove deletion metadata
                    info.pop('deleted_date', None)
                    info.pop('original_location', None)
                    
                    with open(info_file, 'w') as f:
                        json.dump(info, f, indent=2)
                
                return jsonify({
                    'success': True,
                    'message': f'Dataset "{dataset_name}" restored successfully'
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/permanently_delete_dataset/<dataset_name>', methods=['DELETE'])
        def permanently_delete_dataset(dataset_name):
            """Permanently delete a dataset from the trashcan."""
            try:
                trashcan_dir = os.path.join('data', 'trashcan')
                
                # Find the dataset in trashcan (may have timestamp suffix)
                dataset_folder = None
                for item in os.listdir(trashcan_dir):
                    if item == dataset_name or item.startswith(f"{dataset_name}_"):
                        dataset_folder = item
                        break
                
                if not dataset_folder:
                    return jsonify({'success': False, 'error': f'Dataset "{dataset_name}" not found in trashcan'})
                
                trashcan_dataset_dir = os.path.join(trashcan_dir, dataset_folder)
                
                # Permanently delete the directory
                import shutil
                shutil.rmtree(trashcan_dataset_dir)
                
                return jsonify({
                    'success': True,
                    'message': f'Dataset "{dataset_name}" permanently deleted'
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/save_result', methods=['POST'])
        def save_result():
            """Save algorithm results for later viewing."""
            try:
                data = request.get_json()
                
                # Create saved_results directory if it doesn't exist
                results_dir = os.path.join('data', 'saved_results')
                os.makedirs(results_dir, exist_ok=True)
                
                # Generate unique ID and create result folder
                import uuid
                result_id = str(uuid.uuid4())[:8]  # Use shorter ID for folder names
                result_name = data.get('name', 'result').replace(' ', '_').replace('/', '_')
                folder_name = f"{result_name}_{result_id}"
                result_folder = os.path.join(results_dir, folder_name)
                os.makedirs(result_folder, exist_ok=True)
                
                # Prepare result data
                result_data = {
                    'id': result_id,
                    'folder_name': folder_name,
                    'name': data.get('name'),
                    'description': data.get('description', ''),
                    'result_data': data.get('result_data'),
                    'dataset_info': data.get('dataset_info'),
                    'saved_date': data.get('saved_date'),
                    'view_mode': data.get('view_mode', 'single'),
                    'visualization_type': data.get('visualization_type', 'graph')
                }
                
                # Save main result file
                result_file = os.path.join(result_folder, 'result.json')
                with open(result_file, 'w') as f:
                    json.dump(result_data, f, indent=2)
                
                # Save adjacency matrix as CSV
                result_data_obj = data.get('result_data', {})
                if 'adjacency_matrix' in result_data_obj and 'node_names' in result_data_obj:
                    adj_matrix = np.array(result_data_obj['adjacency_matrix'])
                    node_names = result_data_obj['node_names']
                    
                    # Save raw adjacency matrix
                    adj_df = pd.DataFrame(adj_matrix, index=node_names, columns=node_names)
                    adj_file = os.path.join(result_folder, 'adjacency_matrix.csv')
                    adj_df.to_csv(adj_file)
                    
                    # Save thresholded adjacency matrix if threshold available
                    threshold = result_data_obj.get('parameters', {}).get('threshold', 0.0)
                    if threshold > 0:
                        from notears_utils import apply_threshold
                        adj_thresh = apply_threshold(adj_matrix, threshold)
                        adj_thresh_df = pd.DataFrame(adj_thresh, index=node_names, columns=node_names)
                        thresh_file = os.path.join(result_folder, 'adjacency_matrix_thresholded.csv')
                        adj_thresh_df.to_csv(thresh_file)
                
                # Save ground truth matrix if available
                if 'ground_truth_matrix' in result_data_obj and result_data_obj['ground_truth_matrix']:
                    gt_matrix = np.array(result_data_obj['ground_truth_matrix'])
                    gt_df = pd.DataFrame(gt_matrix, index=node_names, columns=node_names)
                    gt_file = os.path.join(result_folder, 'ground_truth_matrix.csv')
                    gt_df.to_csv(gt_file)
                
                # Save parameters as separate file
                if 'parameters' in result_data_obj:
                    params_file = os.path.join(result_folder, 'parameters.json')
                    with open(params_file, 'w') as f:
                        json.dump(result_data_obj['parameters'], f, indent=2)
                
                # Save summary file
                summary = {
                    'name': data.get('name'),
                    'dataset': result_data_obj.get('dataset_name', 'Unknown'),
                    'algorithm': result_data_obj.get('algorithm', {}).get('name', 'Unknown'),
                    'runtime': result_data_obj.get('runtime', 0),
                    'learned_edges': result_data_obj.get('learned_edges', 0),
                    'nodes': len(result_data_obj.get('node_names', [])),
                    'metrics': result_data_obj.get('metrics', {}),
                    'saved_date': data.get('saved_date')
                }
                summary_file = os.path.join(result_folder, 'summary.json')
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
                
                return jsonify({
                    'success': True,
                    'message': f'Result "{data.get("name")}" saved successfully',
                    'result_id': result_id,
                    'folder_name': folder_name
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/saved_results')
        def get_saved_results():
            """Get list of all saved results."""
            try:
                results = []
                results_dir = os.path.join('data', 'saved_results')
                
                if os.path.exists(results_dir):
                    for folder_name in os.listdir(results_dir):
                        folder_path = os.path.join(results_dir, folder_name)
                        if os.path.isdir(folder_path):
                            try:
                                result_file = os.path.join(folder_path, 'result.json')
                                if os.path.exists(result_file):
                                    with open(result_file, 'r') as f:
                                        result_data = json.load(f)
                                    results.append(result_data)
                            except:
                                continue
                
                # Sort by saved date (newest first)
                results.sort(key=lambda x: x.get('saved_date', ''), reverse=True)
                
                return jsonify({'success': True, 'results': results})
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/saved_result/<result_id>')
        def get_saved_result(result_id):
            """Get a specific saved result."""
            try:
                results_dir = os.path.join('data', 'saved_results')
                
                # Find the folder containing this result ID
                result_folder = None
                for folder_name in os.listdir(results_dir):
                    folder_path = os.path.join(results_dir, folder_name)
                    if os.path.isdir(folder_path):
                        result_file = os.path.join(folder_path, 'result.json')
                        if os.path.exists(result_file):
                            with open(result_file, 'r') as f:
                                data = json.load(f)
                                if data.get('id') == result_id:
                                    result_folder = folder_path
                                    break
                
                if not result_folder:
                    return jsonify({'success': False, 'error': 'Result not found'})
                
                result_file = os.path.join(result_folder, 'result.json')
                with open(result_file, 'r') as f:
                    result_data = json.load(f)
                
                return jsonify({'success': True, 'data': result_data})
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/saved_result/<result_id>', methods=['DELETE'])
        def delete_saved_result(result_id):
            """Delete a specific saved result."""
            try:
                results_dir = os.path.join('data', 'saved_results')
                
                # Find the folder containing this result ID
                result_folder = None
                for folder_name in os.listdir(results_dir):
                    folder_path = os.path.join(results_dir, folder_name)
                    if os.path.isdir(folder_path):
                        result_file = os.path.join(folder_path, 'result.json')
                        if os.path.exists(result_file):
                            with open(result_file, 'r') as f:
                                data = json.load(f)
                                if data.get('id') == result_id:
                                    result_folder = folder_path
                                    break
                
                if not result_folder:
                    return jsonify({'success': False, 'error': 'Result not found'})
                
                # Delete the entire folder
                import shutil
                shutil.rmtree(result_folder)
                
                return jsonify({'success': True, 'message': 'Result deleted successfully'})
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/delete_multiple_results', methods=['DELETE'])
        def delete_multiple_results():
            """Delete multiple saved results."""
            try:
                data = request.get_json()
                result_ids = data.get('result_ids', [])
                
                results_dir = os.path.join('data', 'saved_results')
                deleted_count = 0
                
                if not os.path.exists(results_dir):
                    return jsonify({'success': False, 'error': 'Results directory not found'})
                
                for result_id in result_ids:
                    # Find the folder containing this result ID
                    result_folder = None
                    for folder_name in os.listdir(results_dir):
                        folder_path = os.path.join(results_dir, folder_name)
                        if os.path.isdir(folder_path):
                            result_file = os.path.join(folder_path, 'result.json')
                            if os.path.exists(result_file):
                                try:
                                    with open(result_file, 'r') as f:
                                        data_obj = json.load(f)
                                        if data_obj.get('id') == result_id:
                                            result_folder = folder_path
                                            break
                                except:
                                    continue
                    
                    if result_folder:
                        import shutil
                        shutil.rmtree(result_folder)
                        deleted_count += 1
                
                return jsonify({
                    'success': True,
                    'message': f'{deleted_count} result(s) deleted successfully'
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/duplicate_result/<result_id>', methods=['POST'])
        def duplicate_result(result_id):
            """Duplicate a saved result with a new ID."""
            try:
                results_dir = os.path.join('data', 'saved_results')
                
                if not os.path.exists(results_dir):
                    return jsonify({'success': False, 'error': 'Results directory not found'})
                
                # Find the source folder containing this result ID
                source_folder = None
                for folder_name in os.listdir(results_dir):
                    folder_path = os.path.join(results_dir, folder_name)
                    if os.path.isdir(folder_path):
                        result_file = os.path.join(folder_path, 'result.json')
                        if os.path.exists(result_file):
                            try:
                                with open(result_file, 'r') as f:
                                    data_obj = json.load(f)
                                    if data_obj.get('id') == result_id:
                                        source_folder = folder_path
                                        break
                            except:
                                continue
                
                if not source_folder:
                    return jsonify({'success': False, 'error': 'Source result not found'})
                
                # Load source data
                source_file = os.path.join(source_folder, 'result.json')
                with open(source_file, 'r') as f:
                    source_data = json.load(f)
                
                # Create duplicate with new ID
                import uuid
                new_id = str(uuid.uuid4())[:8]  # Use shorter ID for folder names
                
                duplicate_data = source_data.copy()
                duplicate_data['id'] = new_id
                duplicate_data['name'] = f"{source_data['name']} (Copy)"
                duplicate_data['saved_date'] = pd.Timestamp.now().isoformat()
                
                # Create duplicate folder
                result_name = duplicate_data['name'].replace(' ', '_').replace('/', '_')
                folder_name = f"{result_name}_{new_id}"
                duplicate_folder = os.path.join(results_dir, folder_name)
                
                import shutil
                shutil.copytree(source_folder, duplicate_folder)
                
                # Update the result.json in the duplicate folder
                duplicate_result_file = os.path.join(duplicate_folder, 'result.json')
                with open(duplicate_result_file, 'w') as f:
                    json.dump(duplicate_data, f, indent=2)
                
                return jsonify({
                    'success': True,
                    'message': 'Result duplicated successfully',
                    'new_id': new_id
                })
                
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
        
        @self.app.route('/api/export_results')
        def export_results():
            """Export results in various formats."""
            if self.last_result is None:
                return jsonify({'success': False, 'error': 'No results to export'})
            
            format_type = request.args.get('format', 'json')
            dataset_name = request.args.get('dataset', 'results')
            
            try:
                if format_type == 'csv':
                    # Export adjacency matrix as CSV
                    W_learned = np.array(self.last_result['adjacency_matrix'])
                    node_names = self.last_result['node_names']
                    threshold = self.last_result['parameters']['threshold']
                    
                    W_thresh = apply_threshold(W_learned, threshold)
                    df = pd.DataFrame(W_thresh, index=node_names, columns=node_names)
                    
                    output = io.StringIO()
                    df.to_csv(output)
                    
                    response = self.app.response_class(
                        output.getvalue(),
                        mimetype='text/csv',
                        headers={'Content-Disposition': f'attachment; filename={dataset_name}_adjacency.csv'}
                    )
                    return response
                    
                elif format_type == 'json':
                    # Export full results as JSON
                    response = self.app.response_class(
                        json.dumps(self.last_result, indent=2),
                        mimetype='application/json',
                        headers={'Content-Disposition': f'attachment; filename={dataset_name}_results.json'}
                    )
                    return response
                    
                else:
                    return jsonify({'success': False, 'error': f'Unsupported format: {format_type}'})
                    
            except Exception as e:
                return jsonify({'success': False, 'error': str(e)})
    
    def setup_socket_handlers(self):
        """Setup Socket.IO event handlers."""
        
        @self.socketio.on('run_algorithm')
        def handle_run_algorithm(data):
            """Handle algorithm execution request."""
            session_id = request.sid
            
            # Check if this session already has a running algorithm
            if session_id in self.running_sessions and self.running_sessions[session_id]:
                emit('algorithm_failed', {'error': 'Algorithm is already running in this session'})
                return
            
            if self.current_data is None:
                emit('algorithm_failed', {'error': 'No dataset loaded'})
                return
            
            # Start algorithm in separate thread for this session
            self.running_sessions[session_id] = True
            dataset_name = data.get('dataset', 'unknown')
            algorithm_id = data.get('algorithm', 'notears_nonlinear')
            parameters = data.get('parameters', {})
            
            self.algorithm_threads[session_id] = threading.Thread(
                target=self._run_algorithm_thread,
                args=(session_id, dataset_name, algorithm_id, parameters)
            )
            self.algorithm_threads[session_id].daemon = True
            self.algorithm_threads[session_id].start()
        
        @self.socketio.on('stop_algorithm')
        def handle_stop_algorithm():
            """Handle algorithm stop request."""
            session_id = request.sid
            if session_id in self.running_sessions:
                self.running_sessions[session_id] = False
            emit('status_update', {'message': 'Algorithm stopped by user', 'status': 'ready'})
    
    def _run_algorithm_thread(self, session_id: str, dataset_name: str, algorithm_id: str, parameters: Dict):
        """Execute the algorithm in a separate thread."""
        try:
            # Get algorithm instance
            try:
                algorithm = get_algorithm(algorithm_id)
                self.socketio.emit('status_update', {'message': f'Running {algorithm.name}...', 'status': 'running'}, room=session_id)
                self.socketio.emit('log_message', {'message': f'Starting {algorithm.name}...', 'level': 'info'}, room=session_id)
            except Exception as e:
                raise ValueError(f'Algorithm "{algorithm_id}" not found: {str(e)}')
            
            # Merge parameters with defaults
            default_params = algorithm.get_default_parameters()
            merged_params = {**default_params, **parameters}
            
            # Log parameters
            param_str = ', '.join([f'{k}={v}' for k, v in merged_params.items()])
            self.socketio.emit('log_message', {
                'message': f'Parameters: {param_str}',
                'level': 'info'
            }, room=session_id)
            
            # Prepare data
            X = self.current_data.values
            
            # Define progress callback for real-time updates (session-specific)
            def progress_callback(iteration, metric_value, additional_info=None):
                if session_id in self.running_sessions and self.running_sessions[session_id]:
                    self.socketio.emit('algorithm_progress', {
                        'iteration': iteration,
                        'metric_value': metric_value,
                        'additional_info': additional_info
                    }, room=session_id)
            
            # Run algorithm
            start_time = time.time()
            W_learned = algorithm.run(X, merged_params, progress_callback)
            runtime = time.time() - start_time
            
            if not self.is_running:
                return  # Algorithm was stopped
            
            # Get threshold for post-processing
            threshold = merged_params.get('threshold', 0.0)
            
            # Compute metrics if ground truth available
            metrics = None
            if self.current_ground_truth is not None:
                metrics = compute_metrics(W_learned, self.current_ground_truth, thresh=threshold)
            
            # Apply threshold for edge counting
            W_thresh = apply_threshold(W_learned, threshold)
            n_learned_edges = int(np.sum(W_thresh != 0))
            
            # Store results
            self.last_result = {
                'adjacency_matrix': W_learned.tolist(),
                'ground_truth_matrix': self.current_ground_truth.tolist() if self.current_ground_truth is not None else None,
                'node_names': self.current_node_names or [f"X{i}" for i in range(X.shape[1])],
                'runtime': runtime,
                'learned_edges': n_learned_edges,
                'metrics': metrics,
                'algorithm': {
                    'id': algorithm_id,
                    'name': algorithm.name
                },
                'parameters': merged_params,
                'dataset_name': dataset_name
            }
            
            # Notify completion
            self.running_sessions[session_id] = False
            self.socketio.emit('algorithm_completed', self.last_result, room=session_id)
            self.socketio.emit('status_update', {'message': 'Algorithm completed successfully', 'status': 'ready'}, room=session_id)
            
        except Exception as e:
            if session_id in self.running_sessions:
                self.running_sessions[session_id] = False
            error_msg = str(e)
            self.socketio.emit('algorithm_failed', {'error': error_msg}, room=session_id)
            self.socketio.emit('log_message', {'message': f'Algorithm failed: {error_msg}', 'level': 'error'}, room=session_id)
            self.socketio.emit('status_update', {'message': 'Algorithm failed', 'status': 'error'}, room=session_id)
        finally:
            # Clean up session tracking
            if session_id in self.running_sessions:
                del self.running_sessions[session_id]
            if session_id in self.algorithm_threads:
                del self.algorithm_threads[session_id]
    
    def _validate_dataset(self, data: pd.DataFrame) -> Dict:
        """Validate dataset for NOTEARS compatibility."""
        try:
            validation_result = {
                'valid': True,
                'warnings': [],
                'errors': [],
                'info': {}
            }
            
            # Check basic requirements
            if data.empty:
                validation_result['valid'] = False
                validation_result['errors'].append('Dataset is empty')
                return validation_result
            
            if len(data.columns) < 2:
                validation_result['valid'] = False
                validation_result['errors'].append('Dataset must have at least 2 variables')
                return validation_result
            
            # Check for numeric data
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            non_numeric_cols = data.select_dtypes(exclude=[np.number]).columns
            
            if len(non_numeric_cols) > 0:
                validation_result['errors'].append(f'Non-numeric columns found: {list(non_numeric_cols)}')
                validation_result['valid'] = False
            
            # Check for missing values
            missing_count = data.isnull().sum().sum()
            if missing_count > 0:
                validation_result['warnings'].append(f'Dataset contains {missing_count} missing values')
            
            # Check for sufficient samples
            n_samples, n_vars = data.shape
            if n_samples < n_vars * 10:
                validation_result['warnings'].append(f'Small sample size: {n_samples} samples for {n_vars} variables (recommended: ≥{n_vars * 10})')
            
            # Check for constant columns
            constant_cols = []
            for col in data.columns:
                if data[col].nunique() <= 1:
                    constant_cols.append(col)
            
            if constant_cols:
                validation_result['warnings'].append(f'Constant columns detected: {constant_cols}')
            
            # Check for extreme values
            for col in numeric_cols:
                if data[col].dtype in [np.float64, np.float32]:
                    if np.isinf(data[col]).any():
                        validation_result['errors'].append(f'Infinite values found in column: {col}')
                        validation_result['valid'] = False
                    
                    # Check for very large values that might cause overflow
                    max_val = data[col].abs().max()
                    if max_val > 1e6:
                        validation_result['warnings'].append(f'Large values in column {col} (max: {max_val:.2e})')
            
            # Add dataset statistics
            validation_result['info'] = {
                'shape': data.shape,
                'numeric_columns': len(numeric_cols),
                'missing_values': int(missing_count),
                'memory_usage': f"{data.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB"
            }
            
            return validation_result
            
        except Exception as e:
            return {
                'valid': False,
                'errors': [f'Validation error: {str(e)}'],
                'warnings': [],
                'info': {}
            }
    
    def _simulate_progress(self, max_iter):
        """Simulate progress updates for user feedback."""
        import time
        
        for i in range(1, max_iter + 1):
            if not self.is_running:
                break
            
            # Simulate decreasing h_value and increasing rho
            h_value = max(0.1 * (max_iter - i) / max_iter, 0.001)
            rho = min(2.0 ** i, 1e16)
            
            self.socketio.emit('algorithm_progress', {
                'iteration': i,
                'h_value': h_value,
                'rho': rho
            })
            
            # Sleep for a realistic interval (adjust based on typical runtime)
            time.sleep(2.0)  # 2 seconds per iteration simulation
    
    def run(self, host='localhost', port=5000, debug=False):
        """Run the Flask application."""
        print(f"Starting NOTEARS Web GUI at http://{host}:{port}")
        self.socketio.run(self.app, host=host, port=port, debug=debug, allow_unsafe_werkzeug=True)


def main():
    """Main function to run the web GUI."""
    app = NOTEARSWebApp()
    
    # Check if datasets directory exists
    if not os.path.exists('datasets'):
        print("Warning: 'datasets' directory not found. Please run 'python create_test_datasets.py' first.")
    
    print("NOTEARS Web GUI")
    print("=" * 40)
    print("Features:")
    print("• Interactive dataset selection and upload")
    print("• Real-time algorithm configuration")
    print("• Live progress monitoring")
    print("• Interactive result visualization")
    print("• Export functionality")
    print()
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        print("\nShutting down NOTEARS Web GUI...")
    except Exception as e:
        print(f"Error starting application: {e}")


if __name__ == "__main__":
    main()
