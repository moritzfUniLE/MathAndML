"""
Bnlearn-based Bayesian Network Structure Learning Algorithms
Provides integration with the bnlearn library for classical Bayesian network methods.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from .base_algorithm import BaseAlgorithm

try:
    import bnlearn as bn
    BNLEARN_AVAILABLE = True
except ImportError:
    BNLEARN_AVAILABLE = False
    bn = None

class BnlearnHillClimbing(BaseAlgorithm):
    """Hill Climbing structure learning algorithm using bnlearn."""
    
    def __init__(self):
        super().__init__(
            name="Hill Climbing (bnlearn)",
            description="Classical hill climbing algorithm for Bayesian network structure learning"
        )
    
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'scoring_method': 'bic',
            'max_iter': 100,
            'threshold': 0.3,
            'discretize': True,
            'discretize_nbins': 5
        }
    
    def get_parameter_definitions(self) -> Dict[str, Dict[str, Any]]:
        return {
            'scoring_method': {
                'type': 'choice',
                'choices': ['bic', 'k2', 'bdeu'],
                'default': 'bic',
                'description': 'Scoring function for evaluating network structures'
            },
            'max_iter': {
                'type': 'int',
                'min': 10,
                'max': 1000,
                'default': 100,
                'description': 'Maximum number of iterations for hill climbing'
            },
            'threshold': {
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                'step': 0.01,
                'default': 0.3,
                'description': 'Threshold for edge inclusion in final graph'
            },
            'discretize': {
                'type': 'bool',
                'default': True,
                'description': 'Automatically discretize continuous variables'
            },
            'discretize_nbins': {
                'type': 'int',
                'min': 2,
                'max': 20,
                'default': 5,
                'description': 'Number of bins for discretization'
            }
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters."""
        try:
            # Check all required parameters are valid
            scoring_method = parameters.get('scoring_method', 'bic')
            if scoring_method not in ['bic', 'k2', 'bdeu']:
                return False
            
            max_iter = int(parameters.get('max_iter', 100))
            if not (10 <= max_iter <= 1000):
                return False
            
            threshold = float(parameters.get('threshold', 0.3))
            if not (0.0 <= threshold <= 1.0):
                return False
            
            discretize_nbins = int(parameters.get('discretize_nbins', 5))
            if not (2 <= discretize_nbins <= 20):
                return False
            
            return True
        except (ValueError, TypeError):
            return False
    
    def run(self, X: np.ndarray, parameters: Dict[str, Any], 
            progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Run Hill Climbing structure learning."""
        
        if not BNLEARN_AVAILABLE:
            raise ImportError("bnlearn package is not available. Please install it: pip install bnlearn")
        
        # Convert to DataFrame
        n_vars = X.shape[1]
        column_names = [f"X{i}" for i in range(n_vars)]
        df = pd.DataFrame(X, columns=column_names)
        
        if progress_callback:
            progress_callback(0, 0.0, "Preparing data...")
        
        # Discretize if requested
        if parameters.get('discretize', True):
            if progress_callback:
                progress_callback(1, 0.0, "Discretizing continuous variables...")
            
            try:
                nbins = int(parameters.get('discretize_nbins', 5))
                df_discrete = bn.discretize(df, nbins=nbins)
            except:
                # Fallback: simple quantile-based discretization
                df_discrete = df.copy()
                for col in df.columns:
                    df_discrete[col] = pd.qcut(df[col], q=nbins, labels=False, duplicates='drop')
        else:
            df_discrete = df
        
        if progress_callback:
            progress_callback(2, 0.0, "Learning structure with Hill Climbing...")
        
        try:
            # Learn structure using Hill Climbing
            model = bn.structure_learning.fit(
                df_discrete, 
                methodtype='hc', 
                scoretype=parameters.get('scoring_method', 'bic'),
                max_iter=int(parameters.get('max_iter', 100))
            )
            
            if progress_callback:
                progress_callback(100, 0.0, "Structure learning completed")
            
            # Extract adjacency matrix
            W_learned = np.zeros((n_vars, n_vars))
            
            if 'adjmat' in model:
                W_learned = model['adjmat'].values.astype(float)
            elif 'edges' in model:
                # Create adjacency matrix from edges
                for edge in model['edges']:
                    try:
                        source_idx = column_names.index(edge[0])
                        target_idx = column_names.index(edge[1])
                        W_learned[source_idx, target_idx] = 1.0
                    except (ValueError, IndexError):
                        continue
            
            # Apply threshold
            threshold = float(parameters.get('threshold', 0.3))
            W_learned = np.where(np.abs(W_learned) >= threshold, W_learned, 0.0)
            
            return W_learned
            
        except Exception as e:
            error_msg = f"Hill Climbing failed: {str(e)}"
            if progress_callback:
                progress_callback(100, 0.0, error_msg)
            raise RuntimeError(error_msg)

class BnlearnPC(BaseAlgorithm):
    """PC algorithm implementation using bnlearn."""
    
    def __init__(self):
        super().__init__(
            name="PC Algorithm (bnlearn)",
            description="PC (Peter-Clark) constraint-based algorithm for causal discovery"
        )
    
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'alpha': 0.05,
            'ci_test': 'chi_square',
            'threshold': 0.3,
            'discretize': True,
            'discretize_nbins': 5
        }
    
    def get_parameter_definitions(self) -> Dict[str, Dict[str, Any]]:
        return {
            'alpha': {
                'type': 'float',
                'min': 0.001,
                'max': 0.2,
                'step': 0.001,
                'default': 0.05,
                'description': 'Significance level for independence tests'
            },
            'ci_test': {
                'type': 'choice',
                'choices': ['chi_square', 'pearsonr', 'g_sq', 'log_likelihood', 'freeman_tuckey', 'modified_log_likelihood', 'neyman', 'cressie_read', 'power_divergence'],
                'default': 'chi_square',
                'description': 'Conditional independence test to use'
            },
            'threshold': {
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                'step': 0.01,
                'default': 0.3,
                'description': 'Threshold for edge inclusion in final graph'
            },
            'discretize': {
                'type': 'bool',
                'default': True,
                'description': 'Automatically discretize continuous variables'
            },
            'discretize_nbins': {
                'type': 'int',
                'min': 2,
                'max': 20,
                'default': 5,
                'description': 'Number of bins for discretization'
            }
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters."""
        try:
            alpha = float(parameters.get('alpha', 0.05))
            if not (0.001 <= alpha <= 0.2):
                return False
            
            ci_test = parameters.get('ci_test', 'chi_square')
            valid_ci_tests = ['chi_square', 'pearsonr', 'g_sq', 'log_likelihood', 'freeman_tuckey', 'modified_log_likelihood', 'neyman', 'cressie_read', 'power_divergence']
            if ci_test not in valid_ci_tests:
                return False
            
            threshold = float(parameters.get('threshold', 0.3))
            if not (0.0 <= threshold <= 1.0):
                return False
            
            discretize_nbins = int(parameters.get('discretize_nbins', 5))
            if not (2 <= discretize_nbins <= 20):
                return False
            
            return True
        except (ValueError, TypeError):
            return False
    
    def run(self, X: np.ndarray, parameters: Dict[str, Any], 
            progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Run PC algorithm structure learning."""
        
        if not BNLEARN_AVAILABLE:
            raise ImportError("bnlearn package is not available. Please install it: pip install bnlearn")
        
        # Convert to DataFrame
        n_vars = X.shape[1]
        column_names = [f"X{i}" for i in range(n_vars)]
        df = pd.DataFrame(X, columns=column_names)
        
        if progress_callback:
            progress_callback(0, 0.0, "Preparing data...")
        
        # Discretize if requested
        if parameters.get('discretize', True):
            if progress_callback:
                progress_callback(1, 0.0, "Discretizing continuous variables...")
            
            try:
                nbins = int(parameters.get('discretize_nbins', 5))
                df_discrete = bn.discretize(df, nbins=nbins)
            except:
                # Fallback: simple quantile-based discretization
                df_discrete = df.copy()
                for col in df.columns:
                    df_discrete[col] = pd.qcut(df[col], q=nbins, labels=False, duplicates='drop')
        else:
            df_discrete = df
        
        if progress_callback:
            progress_callback(2, 0.0, "Learning structure with PC algorithm...")
        
        try:
            # Learn structure using PC algorithm
            model = bn.structure_learning.fit(
                df_discrete, 
                methodtype='pc',
                params_pc={
                    'alpha': float(parameters.get('alpha', 0.05)),
                    'ci_test': parameters.get('ci_test', 'chi_square')
                }
            )
            
            if progress_callback:
                progress_callback(10, 0.0, "Structure learning completed")
            
            # Extract adjacency matrix
            W_learned = np.zeros((n_vars, n_vars))
            
            if 'adjmat' in model:
                W_learned = model['adjmat'].values.astype(float)
            elif 'edges' in model:
                # Create adjacency matrix from edges
                for edge in model['edges']:
                    try:
                        source_idx = column_names.index(edge[0])
                        target_idx = column_names.index(edge[1])
                        W_learned[source_idx, target_idx] = 1.0
                    except (ValueError, IndexError):
                        continue
            
            # Apply threshold
            threshold = float(parameters.get('threshold', 0.3))
            W_learned = np.where(np.abs(W_learned) >= threshold, W_learned, 0.0)
            
            return W_learned
            
        except Exception as e:
            error_msg = f"PC algorithm failed: {str(e)}"
            if progress_callback:
                progress_callback(10, 0.0, error_msg)
            raise RuntimeError(error_msg)

class BnlearnGS(BaseAlgorithm):
    """Grow-Shrink algorithm implementation using bnlearn."""
    
    def __init__(self):
        super().__init__(
            name="Grow-Shrink (bnlearn)",
            description="Grow-Shrink constraint-based algorithm for Bayesian network structure learning"
        )
    
    def get_default_parameters(self) -> Dict[str, Any]:
        return {
            'alpha': 0.05,
            'ci_test': 'chi_square',
            'threshold': 0.3,
            'discretize': True,
            'discretize_nbins': 5
        }
    
    def get_parameter_definitions(self) -> Dict[str, Dict[str, Any]]:
        return {
            'alpha': {
                'type': 'float',
                'min': 0.001,
                'max': 0.2,
                'step': 0.001,
                'default': 0.05,
                'description': 'Significance level for independence tests'
            },
            'ci_test': {
                'type': 'choice',
                'choices': ['chi_square', 'pearsonr', 'g_sq', 'log_likelihood', 'freeman_tuckey', 'modified_log_likelihood', 'neyman', 'cressie_read', 'power_divergence'],
                'default': 'chi_square',
                'description': 'Conditional independence test to use'
            },
            'threshold': {
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                'step': 0.01,
                'default': 0.3,
                'description': 'Threshold for edge inclusion in final graph'
            },
            'discretize': {
                'type': 'bool',
                'default': True,
                'description': 'Automatically discretize continuous variables'
            },
            'discretize_nbins': {
                'type': 'int',
                'min': 2,
                'max': 20,
                'default': 5,
                'description': 'Number of bins for discretization'
            }
        }
    
    def validate_parameters(self, parameters: Dict[str, Any]) -> bool:
        """Validate parameters."""
        try:
            alpha = float(parameters.get('alpha', 0.05))
            if not (0.001 <= alpha <= 0.2):
                return False
            
            ci_test = parameters.get('ci_test', 'chi_square')
            valid_ci_tests = ['chi_square', 'pearsonr', 'g_sq', 'log_likelihood', 'freeman_tuckey', 'modified_log_likelihood', 'neyman', 'cressie_read', 'power_divergence']
            if ci_test not in valid_ci_tests:
                return False
            
            threshold = float(parameters.get('threshold', 0.3))
            if not (0.0 <= threshold <= 1.0):
                return False
            
            discretize_nbins = int(parameters.get('discretize_nbins', 5))
            if not (2 <= discretize_nbins <= 20):
                return False
            
            return True
        except (ValueError, TypeError):
            return False
    
    def run(self, X: np.ndarray, parameters: Dict[str, Any], 
            progress_callback: Optional[Callable] = None) -> np.ndarray:
        """Run Grow-Shrink algorithm structure learning."""
        
        if not BNLEARN_AVAILABLE:
            raise ImportError("bnlearn package is not available. Please install it: pip install bnlearn")
        
        # Convert to DataFrame
        n_vars = X.shape[1]
        column_names = [f"X{i}" for i in range(n_vars)]
        df = pd.DataFrame(X, columns=column_names)
        
        if progress_callback:
            progress_callback(0, 0.0, "Preparing data...")
        
        # Discretize if requested
        if parameters.get('discretize', True):
            if progress_callback:
                progress_callback(1, 0.0, "Discretizing continuous variables...")
            
            try:
                nbins = int(parameters.get('discretize_nbins', 5))
                df_discrete = bn.discretize(df, nbins=nbins)
            except:
                # Fallback: simple quantile-based discretization
                df_discrete = df.copy()
                for col in df.columns:
                    df_discrete[col] = pd.qcut(df[col], q=nbins, labels=False, duplicates='drop')
        else:
            df_discrete = df
        
        if progress_callback:
            progress_callback(2, 0.0, "Learning structure with Grow-Shrink algorithm...")
        
        try:
            # Learn structure using Grow-Shrink algorithm
            model = bn.structure_learning.fit(
                df_discrete, 
                methodtype='gs',
                params_pc={
                    'alpha': float(parameters.get('alpha', 0.05)),
                    'ci_test': parameters.get('ci_test', 'chi_square')
                }
            )
            
            if progress_callback:
                progress_callback(10, 0.0, "Structure learning completed")
            
            # Extract adjacency matrix
            W_learned = np.zeros((n_vars, n_vars))
            
            if 'adjmat' in model:
                W_learned = model['adjmat'].values.astype(float)
            elif 'edges' in model:
                # Create adjacency matrix from edges
                for edge in model['edges']:
                    try:
                        source_idx = column_names.index(edge[0])
                        target_idx = column_names.index(edge[1])
                        W_learned[source_idx, target_idx] = 1.0
                    except (ValueError, IndexError):
                        continue
            
            # Apply threshold
            threshold = float(parameters.get('threshold', 0.3))
            W_learned = np.where(np.abs(W_learned) >= threshold, W_learned, 0.0)
            
            return W_learned
            
        except Exception as e:
            error_msg = f"Grow-Shrink algorithm failed: {str(e)}"
            if progress_callback:
                progress_callback(10, 0.0, error_msg)
            raise RuntimeError(error_msg)