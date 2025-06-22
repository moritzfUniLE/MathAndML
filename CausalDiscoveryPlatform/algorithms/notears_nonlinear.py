"""
NOTEARS Nonlinear algorithm implementation.
Refactored from the existing notears_core.py to fit the modular system.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import scipy.linalg as slin
import scipy.optimize as sopt
from typing import Dict, Any, Optional, Callable

from .base_algorithm import BaseAlgorithm


class TraceExpm(torch.autograd.Function):
    """
    Custom autograd for trace of matrix exponential of h(A)=trace(e^{A*A})
    Used in DAG acyclicity constraint.
    Modified with weight clipping to prevent numerical overflow.
    """
    @staticmethod
    def forward(ctx, input, max_norm=2.0):
        # Clip weights to prevent overflow in matrix exponential
        input_clipped = torch.clamp(input, -max_norm, max_norm)
        # Compute expm on CPU numpy array
        input_sq = (input_clipped * input_clipped).detach().cpu().numpy()
        
        # Additional safety check for large values
        if np.max(input_sq) > 10.0:
            print(f"[WARNING] Large values in matrix exp: max={np.max(input_sq):.3f}, clipping further")
            input_sq = np.clip(input_sq, 0, 10.0)
        
        try:
            E = slin.expm(input_sq)
            # Check for NaN or Inf in result
            if not np.isfinite(E).all():
                print("[WARNING] Non-finite values in matrix exponential, using approximation")
                # Use polynomial approximation for extreme cases
                E = np.eye(len(input_sq)) + input_sq + 0.5 * np.matmul(input_sq, input_sq)
        except Exception as e:
            print(f"[ERROR] Matrix exponential failed: {e}, using polynomial approximation")
            E = np.eye(len(input_sq)) + input_sq + 0.5 * np.matmul(input_sq, input_sq)
        
        ctx.save_for_backward(torch.from_numpy(E).to(input))
        return torch.tensor(E.trace(), dtype=input.dtype, device=input.device)

    @staticmethod
    def backward(ctx, grad_out):
        # Gradient is transpose of E times upstream grad
        (E,) = ctx.saved_tensors
        return grad_out * E.t(), None  # None for max_norm parameter


def trace_expm(input, max_norm=2.0):
    """Safe trace exponential with weight clipping"""
    return TraceExpm.apply(input, max_norm)


class LocallyConnected(nn.Module):
    """
    Per-node local layer: distinct weights for each input dimension.
    Used to map from hidden layer back to d-dimensional output.
    """
    def __init__(self, d, m_in, m_out, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d, m_in, m_out))
        self.bias = nn.Parameter(torch.empty(d, m_out)) if bias else None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.weight.shape[1]
        nn.init.uniform_(self.weight, -math.sqrt(k), math.sqrt(k))
        if self.bias is not None:
            nn.init.uniform_(self.bias, -math.sqrt(k), math.sqrt(k))

    def forward(self, x):
        # x: (n_samples, d, m_in)
        out = torch.matmul(x.unsqueeze(2), self.weight.unsqueeze(0)).squeeze(2)
        if self.bias is not None:
            out += self.bias
        return out


class LBFGSBScipy(torch.optim.Optimizer):
    """
    Wrap SciPy L-BFGS-B optimizer to work with PyTorch parameters.
    Splits parameters into a flat vector, handles bounds.
    """
    def __init__(self, params):
        super().__init__(params, {})
        self._params = self.param_groups[0]['params']
        self._numel = sum(p.numel() for p in self._params)
        for p in self._params:
            p.requires_grad_(True)

    def _gather_flat_grad(self):
        views=[]
        for p in self._params:
            view = p.grad.view(-1) if p.grad is not None else p.new_zeros(p.numel())
            views.append(view)
        return torch.cat(views,0)

    def _gather_flat(self):
        return torch.cat([p.data.view(-1) for p in self._params],0)

    def _distribute_flat(self,x):
        offset=0
        for p in self._params:
            numel=p.numel()
            p.data.copy_(x[offset:offset+numel].view_as(p))
            offset+=numel

    def step(self, closure):
        def wrapped_closure(x):
            x=torch.from_numpy(x).double()
            self._distribute_flat(x)
            loss=float(closure())
            flat_grad=self._gather_flat_grad().cpu().numpy()
            return loss, flat_grad

        x0=self._gather_flat().cpu().numpy()
        bounds=[(0,None) if i<self._numel//2 else (None,None) for i in range(self._numel)]
        res=sopt.minimize(wrapped_closure,x0,method='L-BFGS-B',jac=True,bounds=bounds,options={'maxiter':100})
        self._distribute_flat(torch.from_numpy(res.x))


class NotearsMLP(nn.Module):
    """
    Two-layer MLP for NOTEARS: parametrizes weighted adjacency via positive/negative weights.
    h(A)=trace(e^{A*A})-d enforces DAG constraint.
    """
    def __init__(self, d, m_hidden=10):
        super().__init__()
        self.d=d
        self.m=m_hidden
        # Positive and negative weight maps
        self.fc1_pos=nn.Linear(d,d*m_hidden)
        self.fc1_neg=nn.Linear(d,d*m_hidden)
        self.fc2=LocallyConnected(d,m_hidden,1)
        nn.init.uniform_(self.fc1_pos.weight,0,0.1)
        nn.init.uniform_(self.fc1_neg.weight,0,0.1)

    def forward(self,x):
        # x: (n,d)
        x=self.fc1_pos(x)-self.fc1_neg(x)
        x=torch.sigmoid(x)
        x=x.view(-1,self.d,self.m)
        x=self.fc2(x).squeeze(2)
        return x

    def h_func(self, max_norm=2.0):
        """Compute acyclicity constraint h(A) with weight clipping."""
        w=self.fc1_pos.weight-self.fc1_neg.weight
        w=w.view(self.d,self.m,self.d).permute(2,0,1)
        A=torch.sum(w*w,dim=2)
        return trace_expm(A, max_norm)-self.d

    def l2_reg(self):
        """Compute L2 regularisation term."""
        return torch.sum(self.fc1_pos.weight**2)+torch.sum(self.fc1_neg.weight**2)+torch.sum(self.fc2.weight**2)

    def fc1_l1(self):
        """Compute L1 norm on fc1 weights for sparsity."""
        return torch.sum(torch.abs(self.fc1_pos.weight))+torch.sum(torch.abs(self.fc1_neg.weight))

    @torch.no_grad()
    def fc1_to_adj(self):
        """Extract adjacency matrix from learned weights."""
        w=self.fc1_pos.weight-self.fc1_neg.weight
        w=w.view(self.d,self.m,self.d).permute(2,0,1)
        return torch.sqrt(torch.sum(w*w,dim=2)).cpu().numpy()


class NOTEARSNonlinear(BaseAlgorithm):
    """NOTEARS Nonlinear causal discovery algorithm."""
    
    def __init__(self):
        super().__init__(
            name="NOTEARS Nonlinear",
            description="Nonlinear structural equation model using neural networks with DAG constraint"
        )
    
    def get_default_parameters(self) -> Dict[str, Any]:
        """Get default parameters for NOTEARS Nonlinear."""
        return {
            'lambda1': 0.01,
            'lambda2': 0.01,
            'max_iter': 100,
            'hidden_units': 10,
            'h_tol': 1e-8,
            'rho_max': 1e32,
            'rho_mult': 2.0,
            'threshold': 0.1
        }
    
    def get_parameter_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Get parameter definitions for UI generation."""
        return {
            'lambda1': {
                'type': 'float',
                'min': 0.001,
                'max': 1.0,
                'step': 0.001,
                'default': 0.01,
                'description': 'L1 regularization parameter (sparsity penalty)'
            },
            'lambda2': {
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                'step': 0.001,
                'default': 0.01,
                'description': 'L2 regularization parameter (weight decay)'
            },
            'max_iter': {
                'type': 'int',
                'min': 10,
                'max': 500,
                'step': 10,
                'default': 100,
                'description': 'Maximum number of outer iterations'
            },
            'hidden_units': {
                'type': 'int',
                'min': 1,
                'max': 100,
                'step': 1,
                'default': 10,
                'description': 'Number of hidden units in neural network'
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
                'min': 1e20,
                'max': 1e40,
                'step': 1e20,
                'default': 1e32,
                'description': 'Maximum penalty coefficient'
            },
            'rho_mult': {
                'type': 'float',
                'min': 1.1,
                'max': 10.0,
                'step': 0.1,
                'default': 2.0,
                'description': 'Penalty multiplier on violation'
            },
            'threshold': {
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                'step': 0.01,
                'default': 0.1,
                'description': 'Edge detection threshold'
            }
        }
    
    def validate_parameters(self, params: Dict[str, Any]) -> bool:
        """Validate NOTEARS Nonlinear parameters."""
        try:
            # Check required parameters
            required = ['lambda1', 'lambda2', 'max_iter', 'hidden_units', 'h_tol', 'rho_max', 'rho_mult', 'threshold']
            for param in required:
                if param not in params:
                    return False
            
            # Validate ranges
            if params['lambda1'] <= 0:
                return False
            if params['lambda2'] < 0:
                return False
            if params['max_iter'] <= 0:
                return False
            if params['hidden_units'] <= 0:
                return False
            if params['h_tol'] <= 0:
                return False
            if params['rho_max'] <= 0:
                return False
            if params['rho_mult'] <= 1.0:
                return False
            if params['threshold'] < 0:
                return False
                
            return True
        except (KeyError, TypeError, ValueError):
            return False
    
    def prepare_data(self, X: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
        """Prepare data for NOTEARS Nonlinear."""
        # Convert to float32 for PyTorch compatibility
        return X.astype(np.float32)
    
    def run(self, X: np.ndarray, params: Dict[str, Any], 
            progress_callback: Optional[Callable] = None) -> np.ndarray:
        """
        Run NOTEARS Nonlinear algorithm.
        
        Args:
            X: Data matrix of shape (n_samples, n_variables)
            params: Algorithm parameters
            progress_callback: Optional callback for progress updates
            
        Returns:
            Adjacency matrix of shape (n_variables, n_variables)
        """
        if not self.validate_parameters(params):
            raise ValueError("Invalid parameters for NOTEARS Nonlinear")
        
        # Extract parameters
        lambda1 = params['lambda1']
        lambda2 = params['lambda2']
        max_iter = params['max_iter']
        hidden_units = params['hidden_units']
        h_tol = params['h_tol']
        rho_max = params['rho_max']
        rho_mult = params['rho_mult']
        
        # Prepare data
        X = self.prepare_data(X, params)
        n, d = X.shape
        
        # Initialize model and optimizer
        model = NotearsMLP(d, m_hidden=hidden_units)
        X_t = torch.from_numpy(X).float()
        opt = LBFGSBScipy(model.parameters())
        
        # Augmented Lagrangian optimization
        rho, alpha, h_val = 1.0, 0.0, np.inf

        for it in range(max_iter):
            def closure():
                opt.zero_grad()
                loss = 0.5 / n * torch.sum((model(X_t) - X_t)**2)
                h_curr = model.h_func()
                penalty = 0.5 * rho * h_curr * h_curr + alpha * h_curr
                obj = loss + penalty + lambda1 * model.fc1_l1() + 0.5 * lambda2 * model.l2_reg()
                obj.backward()
                return obj

            opt.step(closure)
            with torch.no_grad():
                h_val = model.h_func().item()
            
            # Call callback if provided, otherwise print to terminal
            if progress_callback:
                progress_callback(it+1, h_val, rho)
            else:
                print(f"[INFO] Iter {it+1}: h={h_val:.8f}, rho={rho:.2e}")
            
            if h_val <= h_tol: 
                break
            rho *= rho_mult
            alpha += rho * h_val
            if rho > rho_max: 
                break

        return model.fc1_to_adj()