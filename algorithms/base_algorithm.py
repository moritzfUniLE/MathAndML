"""
Base algorithm interface for modular causal discovery algorithms.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Callable
import numpy as np


class BaseAlgorithm(ABC):
    """Abstract base class for causal discovery algorithms."""
    
    def __init__(self, name: str, description: str):
        """
        Initialize the algorithm.
        
        Args:
            name: Algorithm name (e.g., "NOTEARS Linear", "NOTEARS Nonlinear")
            description: Human-readable description of the algorithm
        """
        self.name = name
        self.description = description
        self._default_params = {}
        self._param_definitions = {}
    
    @abstractmethod
    def get_default_parameters(self) -> Dict[str, Any]:
        """
        Get the default parameters for this algorithm.
        
        Returns:
            Dictionary of parameter names and their default values
        """
        pass
    
    @abstractmethod
    def get_parameter_definitions(self) -> Dict[str, Dict[str, Any]]:
        """
        Get parameter definitions with metadata for UI generation.
        
        Returns:
            Dictionary where keys are parameter names and values are dicts with:
            - 'type': parameter type ('float', 'int', 'str', 'bool', 'choice')
            - 'min': minimum value (for numeric types)
            - 'max': maximum value (for numeric types)
            - 'step': step size for UI (for numeric types)
            - 'choices': list of valid choices (for 'choice' type)
            - 'description': human-readable description
            - 'default': default value
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """
        Validate algorithm parameters.
        
        Args:
            params: Dictionary of parameter names and values
            
        Returns:
            True if parameters are valid, False otherwise
        """
        pass
    
    @abstractmethod
    def run(self, X: np.ndarray, params: Dict[str, Any], 
            progress_callback: Optional[Callable] = None) -> np.ndarray:
        """
        Run the causal discovery algorithm.
        
        Args:
            X: Data matrix of shape (n_samples, n_variables)
            params: Algorithm parameters
            progress_callback: Optional callback function for progress updates
                             Signature: callback(iteration, metric_value, additional_info)
            
        Returns:
            Adjacency matrix of shape (n_variables, n_variables)
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get algorithm information for display.
        
        Returns:
            Dictionary with algorithm metadata
        """
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.get_parameter_definitions(),
            'defaults': self.get_default_parameters()
        }
    
    def prepare_data(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Prepare data for the algorithm (e.g., normalization, centering).
        Can be overridden by specific algorithms.
        
        Args:
            X: Raw data matrix
            params: Algorithm parameters
            
        Returns:
            Processed data matrix
        """
        # Default: return data as-is
        return X.astype(np.float32)
    
    def postprocess_result(self, W: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """
        Post-process the algorithm result (e.g., thresholding).
        Can be overridden by specific algorithms.
        
        Args:
            W: Raw adjacency matrix from algorithm
            params: Algorithm parameters
            
        Returns:
            Post-processed adjacency matrix
        """
        # Default: apply threshold if provided
        threshold = params.get('threshold', 0.0)
        if threshold > 0:
            W_processed = W.copy()
            W_processed[np.abs(W_processed) < threshold] = 0.0
            np.fill_diagonal(W_processed, 0.0)
            return W_processed
        return W