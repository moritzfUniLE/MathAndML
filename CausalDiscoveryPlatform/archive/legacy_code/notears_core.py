"""
Core implementation of NOTEARS nonlinear causal discovery.
Defines the neural net, augmented Lagrangian optimizer, and utility functions.
"""
import math
import numpy as np
import torch
import torch.nn as nn
import scipy.linalg as slin
import scipy.optimize as sopt


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


def notears_nonlinear(
    model,
    X,
    lambda1=0.01,
    lambda2=0.01,
    max_iter=100,
    h_tol=1e-8,
    rho_max=1e32,
    rho_mult=2.0,
    progress_callback=None,
) -> np.ndarray:
    """
    Augmented Lagrangian optimization loop for NOTEARS nonlinear.

    Args:
        model: NotearsMLP instance.
        X: Data matrix (n x d).
        lambda1: L1 weight.
        lambda2: L2 weight.
        max_iter: Max outer iterations.
        h_tol: Acyclicity tolerance.
        rho_max: Max penalty coefficient.
        rho_mult: Penalty multiplier on violation.
        progress_callback: Optional callback function called after each iteration
                          with signature callback(iteration, h_value, rho).

    Returns:
        Learned adjacency matrix as numpy.ndarray.
    """
    rho, alpha, h_val = 1.0, 0.0, np.inf
    X_t=torch.from_numpy(X).float()  # Ensure float32 dtype
    opt=LBFGSBScipy(model.parameters())

    for it in range(max_iter):
        def closure():
            opt.zero_grad()
            loss=0.5/X.shape[0]*torch.sum((model(X_t)-X_t)**2)
            h_curr=model.h_func()
            penalty=0.5*rho*h_curr*h_curr+alpha*h_curr
            obj=loss+penalty+lambda1*model.fc1_l1()+0.5*lambda2*model.l2_reg()
            obj.backward()
            return obj

        opt.step(closure)
        with torch.no_grad():
            h_val=model.h_func().item()
        
        # Always print to terminal, and call callback if provided
        print(f"[INFO] Iter {it+1}: h={h_val:.8f}, rho={rho:.2e}")
        if progress_callback:
            progress_callback(it+1, h_val, rho)
        
        if h_val<=h_tol: break
        rho*=rho_mult
        alpha+=rho*h_val
        if rho>rho_max: break

    return model.fc1_to_adj()


def apply_threshold(W: np.ndarray, thresh: float | None) -> np.ndarray:
    """Identical helper to zero small entries and diagonal."""
    W_thr=W.copy()
    if thresh is not None: W_thr[np.abs(W_thr)<thresh]=0.0
    np.fill_diagonal(W_thr,0.0)
    return W_thr


def load_ground_truth_from_bif(bif_path: str) -> tuple[np.ndarray|None,list[str]|None,bool]:
    """
    Parse BIF file to extract ground-truth DAG and node names.

    Returns (W, node_names, success_flag).
    """
    import re
    try:
        text=open(bif_path).read()
        vars=re.findall(r'variable\s+(\w+)',text)
        if not vars: raise ValueError("No variables found in BIF")
        n=len(vars)
        idx={v:i for i,v in enumerate(vars)}
        W=np.zeros((n,n))
        # Extract parent-child relations
        for m in re.finditer(r'probability\s*\(\s*(\w+)\s*\|\s*([^\)]+)\)',text):
            child=m.group(1)
            parents=[p.strip() for p in m.group(2).split(',')]
            for p in parents:
                W[idx[p],idx[child]]=1
        print(f"[INFO] Ground Truth loaded: {n} nodes")
        return W,vars,True
    except Exception as e:
        print(f"[ERROR] BIF load failed: {e}")
        return None,None,False


def compute_metrics(W_learned: np.ndarray, W_true: np.ndarray, thresh: float | None=None) -> dict:
    """
    Compute Hamming distance, TP, FP, FN, precision, recall, and F1.
    Returns dict with metrics and any error message.
    """
    try:
        if W_learned.shape!=W_true.shape:
            raise ValueError(f"Shape mismatch: learned {W_learned.shape} vs true {W_true.shape}")
        Wl=(apply_threshold(W_learned,thresh)!=0).astype(int)
        Wt=(W_true!=0).astype(int)
        TP=int(np.sum((Wl==1)&(Wt==1)))
        FP=int(np.sum((Wl==1)&(Wt==0)))
        FN=int(np.sum((Wl==0)&(Wt==1)))
        precision=TP/(TP+FP) if TP+FP>0 else 0
        recall=TP/(TP+FN) if TP+FN>0 else 0
        f1=2*precision*recall/(precision+recall) if precision+recall>0 else 0
        return {
            'hamming_distance':int(np.sum(Wl!=Wt)),
            'true_positives':TP,
            'false_positives':FP,
            'false_negatives':FN,
            'precision':float(precision),
            'recall':float(recall),
            'f1_score':float(f1),
            'error':None
        }
    except Exception as e:
        return {k:None for k in ['hamming_distance','true_positives','false_positives','false_negatives','precision','recall','f1_score']} | {'error':str(e)}
