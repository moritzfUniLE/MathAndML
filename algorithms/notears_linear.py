"""
NOTEARS Linear algorithm implementation.
Adapted from the original NOTEARS linear implementation.
"""
import numpy as np
import scipy.linalg as slin
import scipy.optimize as sopt
from scipy.special import expit as sigmoid
from typing import Dict, Any, Optional, Callable

from .base_algorithm import BaseAlgorithm


class NOTEARSLinear(BaseAlgorithm):
    """NOTEARS Linear causal discovery algorithm."""
    
    def __init__(self):
        super().__init__(
            name="NOTEARS Linear",
            description="Linear structural equation model with continuous optimization for DAG learning"
        )
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for NOTEARS Linear."""
        return {
            'lambda1': 0.1,
            'loss_type': 'l2',
            'max_iter': 100,
            'h_tol': 1e-8,
            'rho_max': 1e+16,  # Keep original paper value for compatibility
            'threshold': 0.3
        }
    
    def get_parameter_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter definitions for UI generation."""
        return {
            'lambda1': {
                'type': 'float',
                'min': 0.001,
                'max': 1.0,
                'step': 0.001,
                'default': 0.1,
                'description': 'L1 regularization parameter (sparsity penalty)'
            },
            'loss_type': {
                'type': 'choice',
                'choices': ['l2', 'logistic', 'poisson'],
                'default': 'l2',
                'description': 'Loss function type'
            },
            'max_iter': {
                'type': 'int',
                'min': 10,
                'max': 500,
                'step': 10,
                'default': 100,
                'description': 'Maximum number of dual ascent steps'
            },
            'h_tol': {
                'type': 'float',
                'min': 1e-12,
                'max': 1e-4,
                'step': 1e-9,
                'default': 1e-8,
                'description': 'Tolerance for acyclicity constraint'
            },
            'rho_max': {
                'type': 'float',
                'min': 1e6,
                'max': 1e18,
                'step': 1e6,
                'default': 1e16,
                'description': 'Maximum penalty coefficient (high values may cause numerical issues)'
            },
            'threshold': {
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                'step': 0.01,
                'default': 0.3,
                'description': 'Edge weight threshold for sparsity'
            }
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate NOTEARS Linear parameters."""
        try:
            # Check required parameters
            required = ['lambda1', 'loss_type', 'max_iter', 'h_tol', 'rho_max', 'threshold']
            for param in required:
                if param not in params:
                    return False
            
            # Validate ranges
            if params['lambda1'] <= 0:
                return False
            if params['max_iter'] <= 0:
                return False
            if params['h_tol'] <= 0:
                return False
            if params['rho_max'] <= 0:
                return False
            if params['threshold'] < 0:
                return False
            if params['loss_type'] not in ['l2', 'logistic', 'poisson']:
                return False
                
            return True
        except (KeyError, TypeError, ValueError):
            return False
    
    def prepare_data(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Prepare data for NOTEARS Linear."""
        X_processed = X.astype(np.float64)  # Use float64 for numerical stability
        
        # Center data for L2 loss
        if params.get('loss_type', 'l2') == 'l2':
            X_processed = X_processed - np.mean(X_processed, axis=0, keepdims=True)
        
        return X_processed
    
    def run(self, X: np.ndarray, params: Dict[str, Any], 
            progress_callback: Optional[Callable] = None) -> np.ndarray:
        """
        Run NOTEARS Linear algorithm.
        
        Args:
            X: Data matrix of shape (n_samples, n_variables)
            params: Algorithm parameters
            progress_callback: Optional callback for progress updates
            
        Returns:
            Adjacency matrix of shape (n_variables, n_variables)
        """
        if not self.validate_parameters(params):
            raise ValueError("Invalid parameters for NOTEARS Linear")
        
        # Extract parameters
        lambda1 = params['lambda1']
        loss_type = params['loss_type']
        max_iter = params['max_iter']
        h_tol = params['h_tol']
        rho_max = params['rho_max']
        threshold = params['threshold']
        
        # Prepare data
        X = self.prepare_data(X, params)
        n, d = X.shape
        
        def _loss(W):
            """Evaluate value and gradient of loss."""
            M = X @ W
            if loss_type == 'l2':
                R = X - M
                loss = 0.5 / n * (R ** 2).sum()
                G_loss = - 1.0 / n * X.T @ R
            elif loss_type == 'logistic':
                loss = 1.0 / n * (np.logaddexp(0, M) - X * M).sum()
                G_loss = 1.0 / n * X.T @ (sigmoid(M) - X)
            elif loss_type == 'poisson':
                S = np.exp(M)
                loss = 1.0 / n * (S - X * M).sum()
                G_loss = 1.0 / n * X.T @ (S - X)
            else:
                raise ValueError('unknown loss type')
            return loss, G_loss

        def _h(W):
            """Evaluate value and gradient of acyclicity constraint."""
            E = slin.expm(W * W)  # (Zheng et al. 2018)
            h = np.trace(E) - d
            G_h = E.T * W * 2
            return h, G_h

        def _adj(w):
            """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
            return (w[:d * d] - w[d * d:]).reshape([d, d])

        def _func(w):
            """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
            W = _adj(w)
            loss, G_loss = _loss(W)
            h, G_h = _h(W)
            obj = loss + 0.5 * rho * h * h + alpha * h + lambda1 * w.sum()
            G_smooth = G_loss + (rho * h + alpha) * G_h
            g_obj = np.concatenate((G_smooth + lambda1, - G_smooth + lambda1), axis=None)
            return obj, g_obj

        # Initialize variables
        w_est, rho, alpha, h = np.zeros(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
        bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]
        
        # Main optimization loop
        for iteration in range(max_iter):
            w_new, h_new = None, None
            while rho < rho_max:
                sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
                w_new = sol.x
                h_new, _ = _h(_adj(w_new))
                if h_new > 0.25 * h:
                    rho *= 10
                else:
                    break
            
            w_est, h = w_new, h_new
            alpha += rho * h
            
            # Progress callback
            if progress_callback:
                progress_callback(iteration + 1, h, rho)
            else:
                print(f"[INFO] Iter {iteration + 1}: h={h:.8f}, rho={rho:.2e}")
            
            # Check convergence
            if h <= h_tol or rho >= rho_max:
                break
        
        # Extract final result
        W_est = _adj(w_est)
        W_est[np.abs(W_est) < threshold] = 0
        
        return W_est