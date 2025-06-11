"""
notears_core.py
A self-contained implementation of nonlinear NOTEARS (MLP variant)
adapted from your original script.  No plotting, no test stubs.
MIT License – free to use on Universität Leipzig’s HPC cluster.
"""
# -------------------------- imports ---------------------------------------
import math, numpy as np, torch
import torch.nn as nn
import scipy.linalg as slin
import scipy.optimize as sopt

# --------------------- utility: trace expm --------------------------------
class TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        E = slin.expm((input * input).detach().cpu().numpy())
        ctx.save_for_backward(torch.from_numpy(E).to(input))
        return torch.tensor(E.trace(), dtype=input.dtype, device=input.device)

    @staticmethod
    def backward(ctx, grad_out):
        (E,) = ctx.saved_tensors
        return grad_out * E.t()

trace_expm = TraceExpm.apply

# --------------------- locally-connected layer ----------------------------
class LocallyConnected(nn.Module):
    def __init__(self, d, m_in, m_out, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d, m_in, m_out))
        self.bias   = nn.Parameter(torch.empty(d, m_out)) if bias else None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self):
        k = 1.0 / self.weight.shape[1]
        nn.init.uniform_(self.weight, -math.sqrt(k), math.sqrt(k))
        if self.bias is not None:
            nn.init.uniform_(self.bias, -math.sqrt(k), math.sqrt(k))

    def forward(self, x):                       # x: [n, d, m_in]
        out = torch.matmul(x.unsqueeze(2),      #   [n, d, 1, m_in]
                           self.weight.unsqueeze(0)) \
              .squeeze(2)                       # -> [n, d, m_out]
        if self.bias is not None:
            out += self.bias
        return out

# --------------------- custom LBFGS-B (box constraints) -------------------
class LBFGSBScipy(torch.optim.Optimizer):
    def __init__(self, params, bounds=None):
        super().__init__(params, {})
        ps = self.param_groups[0]["params"]
        self._numel  = sum(p.numel() for p in ps)
        self._bounds = bounds or [(None, None)] * self._numel   # <-- NEW
        for p in ps:
            p.requires_grad_(True)

    def _gather_flat(self, attr):
        return torch.cat([getattr(p, attr).view(-1) for p in self.param_groups[0]["params"]])

    def _distribute_flat(self, vec, attr):
        offset = 0
        for p in self.param_groups[0]["params"]:
            n = p.numel()
            getattr(p, attr).copy_(vec[offset:offset+n].view_as(p))
            offset += n

    def step(self, closure):
        flat0 = self._gather_flat("data").detach().cpu().double().numpy()
        # bounds = [(None, None)] * self._numel   # no explicit bounds, but could add
        bounds = self._bounds 
        def func(flat):
            flat_t = torch.as_tensor(flat, dtype=torch.get_default_dtype())
            self._distribute_flat(flat_t, "data")
            loss = closure().detach()
            grad = self._gather_flat("grad").detach().cpu().double().numpy()
            return loss.cpu().double().numpy(), grad

        sopt.minimize(func, flat0, method="L-BFGS-B", jac=True, bounds=bounds)

# --------------------- main model -----------------------------------------
class NotearsMLP(nn.Module):
    def __init__(self, d, m_hidden=10, bias=True):
        super().__init__()
        self.d, self.m = d, m_hidden
        # First layer is split into positive & negative weights → L1 via variable split
        self.fc1_pos = nn.Linear(d, d * m_hidden, bias=bias)
        self.fc1_neg = nn.Linear(d, d * m_hidden, bias=bias)
        # Subsequent local layers (here: one extra layer → output dim 1)
        self.fc2 = LocallyConnected(d, m_hidden, 1, bias=bias)

    # ---- core helpers -----------------------------------------------------
    def forward(self, x):                       # x: [n, d]
        x = self.fc1_pos(x) - self.fc1_neg(x)   # [n, d*m]
        x = torch.sigmoid(x).view(-1, self.d, self.m)
        x = self.fc2(x).squeeze(2)              # back to [n, d]
        return x

    def h_func(self):
        w = self.fc1_pos.weight - self.fc1_neg.weight       # [d*m, d]
        w = w.view(self.d, self.m, self.d).permute(2,0,1)   # [d, d, m]
        A = (w**2).sum(2)                                   # [d, d]
        return trace_expm(A) - self.d

    def l2_reg(self):
        reg = (self.fc1_pos.weight - self.fc1_neg.weight).pow(2).sum()
        reg += self.fc2.weight.pow(2).sum()
        return reg

    def fc1_l1(self):
        return (self.fc1_pos.weight + self.fc1_neg.weight).sum()

    @torch.no_grad()
    def fc1_to_adj(self):
        w = self.fc1_pos.weight - self.fc1_neg.weight
        w = w.view(self.d, self.m, self.d).permute(2,0,1)
        A = (w**2).sum(2).sqrt()
        return A.cpu().numpy()

# --------------------- solver (dual ascent) -------------------------------
def notears_nonlinear(model, X,
                      lambda1=0.1, lambda2=0.1,
                      max_iter=20, h_tol=1e-8, rho_max=1e16,
                      silent=False):
    rho, alpha, h = 1., 0., np.inf
    X_t = torch.from_numpy(X)
    named = list(model.named_parameters())
    params  = [p for _, p in named]
    bounds  = []
    for name, p in named:
        rng = (0.0, 1.5) if name in ("fc1_pos.weight", "fc1_neg.weight") else (None, None)
        bounds.extend([rng] * p.numel())          # one tuple per scalar entry

    opt = LBFGSBScipy(params, bounds=bounds)      # <-- pass bounds here
    for it in range(max_iter):
        def closure():
            opt.zero_grad()
            loss = 0.5/X.shape[0]*(model(X_t)-X_t).pow(2).sum()
            h_val = model.h_func()
            obj = loss + lambda1*model.fc1_l1() + 0.5*lambda2*model.l2_reg() \
                  + 0.5*rho*h_val*h_val + alpha*h_val
            obj.backward()
            return obj
        opt.step(closure)
        with torch.no_grad():
            h = model.h_func().item()
        if not silent:
            print(f"iter {it:02d}  h={h:.3e}  rho={rho:.1e}")
        if h <= h_tol or rho >= rho_max: break
        rho *= 10
        alpha += rho*h
    W = model.fc1_to_adj()
    #print("min/max W vor Cut-off:", W.min(), W.max())
    return W

