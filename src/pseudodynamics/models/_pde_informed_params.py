import os,sys,gc
import numpy as np
import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from typing import Any, Union
from ._PINN_base import PINN_base, PINN_base_sim
from .MLP_models import MLP_surrogate
from .Spline_models import MultiDim_CubicSpline, CubicSpline
from typing import Any, Union, Callable
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint 
# from TorchDiffEqPack import odesolve_adjoint_sym12
# from kan import KAN
import matplotlib.pyplot as plt

class pde_params_base(pl.LightningModule):
    def __init__(self, channels, collapse_D = True, collapse_v = False, g_channels=None, v_channels=None, D_channels=None, time_sensitive=True, lr=3e-4, ode_tol=1e-4, activation_fn:Union[str, list] = 'Tanh', deltax_weight = None, D_penalty = None, weight_intensity=None):
        """
        mlp u theta

        Arguments:
        -------------
        channel : the number of MLP channels of the Behavior function
        [g, v, D]_channel : the number of MLP channels of the Behavior function
        collapse_[D,v] : merge the multi-channel output into 1 channel, 
                         which controls the complexity of the pde term.
        
        kwargs 
        -------
        u_theta : the neural netowrk surrogate of u
        lr: float, the learning rate
        optim_class : str, the optimizer used
        D_penalty : float , default None the weight for penalizing D


        """
        super().__init__()
        self.save_hyperparameters()
        
        self.time_sensitive = time_sensitive
        self.lr = lr
        self.ode_tol = ode_tol 
        self.D_penalty = 0.1 if D_penalty is None else D_penalty
        self.deltax_weight = 0 if deltax_weight is None else deltax_weight

        self.weight_intensity = 1 if weight_intensity is None else weight_intensity

        self.GNLL_fn = nn.GaussianNLLLoss()                     # for population loss
        self.KLD_fn = torch.nn.KLDivLoss(reduction="none")

        self.g_channels = g_channels
        self.v_channels = v_channels
        self.D_channels = D_channels


    def loss_fn(self,x, x_hat, weight=None):
        """
        both x and x_hat are log transformed
        """
        # sanity check
        if x.shape != x_hat.shape:
            x = x.squeeze()
            x_hat = x_hat.squeeze()
        assert x.shape == x_hat.shape
 
        # if torch.all(x >0) :
        #     x  = torch.log(x + 1e-9)
        # if torch.all(x_hat > 0):
        #     x_hat  = torch.log(x_hat + 1e-9)
        
        # -24 is ~ log(1e-9)
        x = torch.clamp(x, min=-24) 
        x_hat = torch.clamp(x_hat, min=-24)

        if weight == None:
            weight = (24+x)**self.weight_intensity
            weight /= weight.sum()

        # compute loss
        loss = torch.sum(weight * (x - x_hat) ** 2)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            # [module.parameters() for module in  [self.g, self.v, self.D]],
            self.parameters(),
                lr=self.lr)
        return optimizer

    def trace_div(self, f, s):
        """
        Calculates the Divergence : which is the trace of the Jacobian df/ds.
        f :  f(s), the output of a function
        s :  s, the variable on which to calculating the derivitives

        Stolen from: https://github.com/rtqichen/ffjord/blob/master/lib/layers/odefunc.py#L13
        """
        sum_diag = 0.
        for i in range(s.shape[1]):
            sum_diag += torch.autograd.grad(f[:, i].sum(), s, create_graph=True , allow_unused=True)[0].contiguous()[:, i].contiguous()

        return sum_diag.contiguous()

    def mul(self, param, term):
        """
        own multiply function to deal with different dimension
        """

        # (bs, )  or (bs, n_dim)
        if param.shape == term.shape:
            prod = torch.mul(param, term)
    
        elif len(term.shape) == 1:
            prod = torch.mul(param, term.unsqueeze(1))
        
        elif len(param.shape) == 1:
            prod = torch.mul(param.unsqueeze(1), term)

        if len(prod.shape) != 1:
            prod = prod.sum(dim=1)

        return prod

    def gradient_of(self, out, variable):
        return torch.autograd.grad(out, variable, create_graph=True, allow_unused=True)[0]

    def D_forward(self, s, t):
        r"""
        Diffusion-net output, optionally hard-clamped into ``self.D_clip``.

        Opt-in via the ``--D_clip lo,hi`` flag (used for the GRN-3D experiment).
        When ``self.D_clip`` is None (the default) this returns the raw network
        output, so existing behaviour is unchanged.
        """
        out = self.D(s, t)
        clip = getattr(self, 'D_clip', None)
        if clip is not None:
            out = out.clamp(min=clip[0], max=clip[1])
        return out

    def equation(self, s, t) -> tuple:
        """
        Apply torch's auto grad to compute the dynamics
        
        based on the following equation:
            ∂u/∂t = ∂/∂s[ D* ∂u/∂s ] - ∂/∂s[ v*u ] + g*u
        
        we calcuate the left hand side (lhs) and the right hand side
        """
        u = self.get_u(s,t) # make sure it is u
        D = self.D_forward(s,t)
        v = self.v(s,t)
        g = self.g(s,t)
        
        # left : ∂u/∂t
        dudt = self.gradient_of(u.sum(), t) 
        
        
        # the first order deviritives of density u to time : ∂u/∂s
        duds = self.gradient_of(u.sum(), s) 
        
        # the first term:  a second order derivative
        Du = self.mul(D, duds)   # element-wise 
        
        # right hand side
        if len(Du.shape) == 1: # for one trajectory system
            # the second order deviritives of density u to cell state : ∂^2u/∂s^2
            #  ∂/∂s (D*∂u/∂s)
            d2Dds2 = self.gradient_of(Du.sum(), s)  

        else:   # for multi-dimensiona data
            # u_ss is different for multi dimension : ∂2u / ∂s_is_i 
            d2Dds2_ls  = []
            for i in range(v.shape[1]):
                du_dsisi = self.gradient_of(Du[:,i].sum(), s)[:, i:i+1]
                d2Dds2_ls.append(du_dsisi)
            d2Dds2 = torch.cat(d2Dds2_ls, dim=1)
        
        # the second term : ∂/∂s[ v*u ]
        vu = self.mul(v, u)
        dvuds = self.gradient_of(vu.sum(), s)  
        
        # right hand side
        diffuse = d2Dds2.sum(dim=1)
        drift = dvuds.sum(dim=1)
        growth = torch.mul(g, u)
        
        return dudt, growth, drift, diffuse

    def constrain_v(self, s, t, deltax):
        r"""
        regularize v to make it keep in the same direction as delta x
        
        Arguments
        ---------
        s : tensor (n_cell, n_dim)
        t : tensor (n_cell,)
        deltax : tensor (n_cell, n_dim), sampled from pseudotime / KNN 
        """
        if deltax is not None:
            v = self.v(s,t)
            v_loss = -1*torch.mean(nn.functional.cosine_similarity(deltax, v))
        else:
            v_loss = 0
        return v_loss    

    def restrict_D(self, s, t, exp=True):
        r"""
        penalize D to restrict instability

        REVIEW (suggestion: why exp-scale the diffusion penalty?).
        The suggestion is correct. With `exp=True`, the penalty is
        ||exp(D)||_2, which is one-sided: exp(D) -> 0 as D -> -inf, so the
        regularizer drives D toward large negative values rather than toward
        0. With `exp=False`, the symmetric L2 norm on D actually pulls D
        toward 0, which matches the intent of Eq. 9. The default here is
        `True` but every call site in this file passes `exp=False`
        explicitly, so the running behavior is fine. The default is
        misleading though -- decision pending on whether to flip it.
        """
        D = self.D(s,t)
        if exp:
            D = torch.exp(D)
        if getattr(self, 'd_penalty_mode', 'legacy') == 'mean':
            # Mean of squared D over all elements — genuinely batch-size invariant
            return D.pow(2).mean()
        D_L2 = torch.norm(D, p=2).sum()  # in case D is high dimensional
        return D_L2

    def diffusion_variance_loss(self, s, t, tp1, k=15):
        r"""
        Variance-matching loss for D(s,t): the learned diffusion coefficient
        should correlate with local expression variance across timepoints.

        Cells in regions with high density change (high |u(s,t+1) - u(s,t)|)
        should have higher D, as this indicates stochastic branching.

        Arguments
        ---------
        s : tensor (n, n_dim), cell states
        t : tensor (n,), time at t_k (already normalised)
        tp1 : tensor (n,), time at t_{k+1}
        k : int, neighbourhood size for local variance estimation
        """
        with torch.no_grad():
            # Estimate local variance from KNN in cell-state space
            # Use pairwise distances to find local spread
            dists = torch.cdist(s, s)                     # (n, n)
            _, knn_idx = dists.topk(k, dim=1, largest=False)  # (n, k)

            # Local variance: variance of cell states in k-neighbourhood
            neighbours = s[knn_idx]                       # (n, k, n_dim)
            local_var = neighbours.var(dim=1).mean(dim=1)  # (n,)

            # Normalise to [0, 1] for stable loss
            local_var = local_var / (local_var.max() + 1e-8)

        # D should be proportional to local variance
        D_pred = self.D(s, t.unsqueeze(1) if t.dim() == 1 else t)
        D_magnitude = torch.abs(D_pred).mean(dim=1) if D_pred.dim() > 1 else torch.abs(D_pred).squeeze()
        D_norm = D_magnitude / (D_magnitude.max().detach() + 1e-8)

        # Negative correlation loss: encourage D to correlate with local_var
        loss = -torch.mean(
            (D_norm - D_norm.mean()) * (local_var - local_var.mean())
        ) / (D_norm.std().detach() * local_var.std() + 1e-8)

        return loss

    def diffusion_entropy_loss(self, s, t, u_t, u_tp1):
        r"""
        Entropy regularization: D should be higher at branching points,
        identified by high entropy in density change between timepoints.

        Cells where the density ratio u(t+1)/u(t) varies most across
        neighbours are at fate decision points and need higher D.

        Arguments
        ---------
        s : tensor (n, n_dim), cell states
        t : tensor (n,), time
        u_t : tensor (n,), density at t
        u_tp1 : tensor (n,), density at t+1
        """
        with torch.no_grad():
            # Density ratio captures where mass is redistributing
            ratio = (u_tp1 + 1e-8) / (u_t + 1e-8)
            # Normalise into a probability-like quantity per cell
            p = ratio / (ratio.sum() + 1e-8)
            # Per-cell surprise (high = unexpected density change = branching)
            entropy_signal = -torch.log(p + 1e-8)
            entropy_signal = entropy_signal / (entropy_signal.max() + 1e-8)

        D_pred = self.D(s, t.unsqueeze(1) if t.dim() == 1 else t)
        D_magnitude = torch.abs(D_pred).mean(dim=1) if D_pred.dim() > 1 else torch.abs(D_pred).squeeze()
        D_norm = D_magnitude / (D_magnitude.max().detach() + 1e-8)

        # Encourage D to be high where entropy signal is high
        loss = -torch.mean(
            (D_norm - D_norm.mean()) * (entropy_signal - entropy_signal.mean())
        ) / (D_norm.std().detach() * entropy_signal.std() + 1e-8)

        return loss

    def area_loss(self, u, u_hat, var=None):
        r"""
        use the area under curve to compute loss

        Arguments
        ---------
        u : tensor (n_grid,)
        u_hat : tensor (n_grid,)
        """
        p_x = u / u.sum()
        p_hat = u_hat / u_hat.sum()

        area_x = torch.cumsum(p_x,dim=0)
        area_hat = torch.cumsum(p_hat, dim=0)

        if var is None:
            loss = torch.pow(area_x - area_hat,2).sum()
        else:
            # loss = self.GNLL_fn(input=p_hat, target=p_x, var=var.mean())
            loss = torch.pow(area_x - area_hat,2).mean() / var.mean()
            
        return loss


    def density_loss(self, u, u_hat, var=None):
        r"""
        use the density itself to compute

        Arguments
        ---------
        p_x : tensor (n_grid,)
        p_hat : tensor (n_grid,)
        """

        p_x = u / u.sum()
        p_hat = u_hat / u_hat.sum()

        if var is None:
            loss = torch.pow(p_x - p_hat,2).sum()
        else:
            GNLL_fn = nn.GaussianNLLLoss(eps=1e-9)
            L = GNLL_fn(input=p_hat, target=p_x, var=var)
            
            loss = self.GNLL_fn(input=p_hat, target=p_x, var=var)

            if loss < 0:
                eps = 1e-6
                loss = 0.5 * torch.pow(p_x - p_hat,2).mean() / max(var.mean(),eps)
            
        return loss

    def population_loss(self, u_pred, Mean, Var) -> torch.Tensor:
        r"""
        the loss term defined for population size, governed by Gaussian Negative Likelihood loss
            Gaussian NLL := 0.5 * log(var) + 0.5 * (input - target)**2/var  +const
        
        Arguments
        ---------
        u_pred : Tensor (t_obs, n_grid), the predicted density for all the cell states, self.u_theta(s_all, t_obs)
        Mean : Tensor (t_obs, 1), D['pop']['mean'], the mean of population size over repeat
        Var : Tensor (t_obs, 1), D['pop']['var'] / D['pop']['n_exp'] , the var of population size over repeat
        
        Return
        ---------
        L_pop : Tensor (1,), loss term summing all observed time point
        """
        
        # copying
        # assert u_pred.shape[1] == self.n_grid , "make sure the same grid is applied"
        
        # the estimated population size N_θ = ∫ u ds
        # N_theta = 0.5*(u_pred[:,1:]+u_pred[:,:-1]).sum(dim=1, keepdim = True) #/ h_inv   
        N_theta = u_pred[:-1].sum()
        # (t_obs, n_grid) -> (t_obs,1)
        
        # population 
        assert N_theta.shape == Mean.shape, 'input and target view not identical'
        
        # compute loss and sum for all observed time point
        L_pop = self.GNLL_fn(input=N_theta, target=Mean, var=Var)**0.5

        return L_pop

    # def growth_loss(self, u, s, t_list) -> torch.Tensor:
    #     r"""
    #     the loss term defined for population size, governed by Gaussian Negative Likelihood loss
    #         (T_j - T_i) ∑ p(x) ∫ exp ( ∫  )
        
    #     Arguments
    #     ---------
    #     u_pred : Tensor (t_obs, n_grid), the predicted density for all the cell states, self.u_theta(s_all, t_obs)
    #     Mean : Tensor (t_obs, 1), D['pop']['mean'], the mean of population size over repeat
    #     Var : Tensor (t_obs, 1), D['pop']['var'] / D['pop']['n_exp'] , the var of population size over repeat
        
    #     Return
    #     ---------
    #     L_pop : Tensor (1,), loss term summing all observed time point
    #     """
        
    #     # copying
    #     # assert u_pred.shape[1] == self.n_grid , "make sure the same grid is applied"
        
    #     # the estimated population size N_θ = ∫ u ds
    #     # N_theta = 0.5*(u_pred[:,1:]+u_pred[:,:-1]).sum(dim=1, keepdim = True) #/ h_inv   
        
    #     odeint(self.)

    #     return L_pop

    def distribution_loss(self, u_pred_b, u_b) -> torch.Tensor:
        r"""
        the loss defined as the kl divergence of the distribution, used to keep the shape
        """
        # from density to probability
        
        p_b = u_b/ u_b.sum()   
        p_pred_b = u_pred_b / u_pred_b.sum()
        
        # prediction should be a distribution in the log space
        #.         y pred  ,  y_true
        L_kld = self.KLD_fn(p_pred_b.squeeze().log(), p_b.squeeze())
        
        return L_kld.mean()

    def predict_nabla_v(self, train_DS, device=None):    
        r"""
        Given a DataSet Class, predict the param 
        """
        if device is None:
            device = next(self.parameters()).device

        # some variables
        n_timepoint = len(train_DS.T_b)
        v_dim = self.v.u_theta[-1].out_features
        

        s_ts = train_DS.s.float().to(device).requires_grad_()
        t_ts = train_DS.t_b.float().to(device).requires_grad_()
        chunk_size = 5000

        v_ls = []

        
        for i in range(0, len(t_ts), chunk_size):                              
            s_in = s_ts[i:i+chunk_size]
            t_in = t_ts[i:i+chunk_size]

            v_pred = self.v(s_in, t_in)
            nabla_v = self.trace_div(v_pred, s_in)

            v_ls.append(nabla_v.detach().cpu().numpy())

        v_pred_ay = np.concatenate(v_ls, axis=0).reshape(n_timepoint, -1) / self.time_scale_factor
        

        return v_pred_ay

    def predict_param(self, train_DS, device=None):    
        r"""
        Given a DataSet Class, predict the param 
        Return : g, v, D
        """
        
        if device is None:
            device = next(self.parameters()).device

        # some variables
        n_timepoint = len(train_DS.T_b)
        v_dim = self.v.u_theta[-1].out_features
        D_dim = self.D.u_theta[-1].out_features

        s_ts = train_DS.s.float().to(device)
        t_ts = train_DS.t_b.float().to(device)
        chunk_size = 5000

        u_pred_ls = []
        v_ls = []
        D_ls = []
        g_ls = []

        with torch.no_grad():
            for i in range(0, len(t_ts), chunk_size):                              
                s_in = s_ts[i:i+chunk_size]
                t_in = t_ts[i:i+chunk_size]

                v_pred = self.v(s_in, t_in)
                g_pred = self.g(s_in, t_in)
                D_pred = self.D_forward(s_in, t_in)
                
                v_ls.append(v_pred.detach().cpu().numpy())
                g_ls.append(g_pred.detach().cpu().numpy())
                D_ls.append(D_pred.detach().cpu().numpy())


        v_pred_ay = np.concatenate(v_ls, axis=0).reshape(n_timepoint, -1, v_dim) / self.time_scale_factor
        g_pred_ay = np.concatenate(g_ls, axis=0).reshape(n_timepoint,-1) / self.time_scale_factor
        D_pred_ay = np.concatenate(D_ls, axis=0).reshape(n_timepoint, -1, D_dim).squeeze() / self.time_scale_factor

        return  g_pred_ay, v_pred_ay, D_pred_ay

    def density_transfer(self, t, states):
        """
        states : initial with s and u of the last timepoint
        """
        s = states[0]
        u = states[1]
        s_n = states[2]
        u_n = states[3]
        device = s.device
        batch = s.shape[0]

        with torch.set_grad_enabled(True):
            s_t = s.clone().float().requires_grad_(True)   # get original s at t
            u_stn1 = u.clone().float().requires_grad_(True)
            s_next = s_n.clone().float().requires_grad_(True)   # get evolved s at t+1
            u_next = u_n.clone().float().requires_grad_(True)
            
            # get the params of the evolved s and t
            t_in = torch.full((batch,1), t.item()*self.time_scale_factor)
            t_in = t_in.to(device).requires_grad_(True)

            u_t = self.get_u(s_t, t_in) # get exp(logu)
            g_t = self.g(s_t, t_in)
            v_t = self.v(s_t, t_in)

            u_next = self.get_u(s_next, t_in) # get exp(logu)
            v_next = self.v(s_next, t_in)

            
            # The drift is sensing the global duds
            vu = self.mul(u_t, v_t).view(-1, 1)
            vu_next = self.mul(u_next, v_next).view(-1, 1)

            growth_local = g_t * u_stn1
            # u_stn1 += growth_local
            # global_drift = self.gradient_of(vu.sum(), s_t) 
            # global_drift = torch.autograd.grad(
            #     vu.sum(), s_t, create_graph=True, allow_unused=True)[0]
            
            global_drift = torch.div(torch.broadcast_to(vu_next - vu, s_t.shape), s_next-s_t)#.sum(dim=1)

            # the amout of mass flowing with the global drift
            # drift = torch.mul(global_drift.sum(dim=1) , torch.div(u_stn1,u_t))
            raw_drift = torch.mul(u_stn1 , torch.div(global_drift.sum(dim=1),u_t))
            # the cell only gives out 
            drift = nn.functional.relu(raw_drift)

            du = drift + self.g(s_next, t_in) * u_next
            ds = torch.zeros_like(s_t)       # cs doesn't change
            u_non = torch.zeros_like(u_stn1) # empty density
            
            return (ds, growth_local-drift, ds, du)

class _ZeroField(nn.Module):
    r"""R2.3 ablation stub: a parameterless growth field that returns g(s,t) == 0.

    Drop-in replacement for the growth MLP (``self.g``). Matches ``MLP_surrogate``'s
    output shape ``(B,)`` (it squeezes the trailing dim) so every g(s,t) call site
    (PDE source term, ODE/SDE forward, growth loss, predict_param) sees zeros with no
    other change; drift and diffusion are untouched. Has no parameters, so it adds
    nothing to the optimiser and an empty state_dict (load_from_checkpoint round-trips).
    """
    def __init__(self, time_sensitive=True):
        super().__init__()
        self.time_sensitive = time_sensitive

    def forward(self, s, t=None):
        if not isinstance(s, torch.Tensor):
            s = torch.as_tensor(s)
        return torch.zeros(s.shape[0], device=s.device, dtype=s.dtype)


class pde_params(pde_params_base):
    r"""
    Default model : PINN prediction + NeuralODE simulation to estimate parameters

    Args
    --------
    channel : list 
        the number of MLP channels of the Behavior function
    [g, v, D]_channel : list
        the number of MLP channels of the Behavior function
    collapse_[D,v] : bool
        merge the multi-channel output into 1 channel, which controls the complexity of the pde term.
    time_sensitive : bool
        whether to use time and state-dependent paramter (dynamic mode) or state-dependent paramter (constant mode)
    u_theta : torch.nn.Module
        the neural netowrk surrogate of u
    lr: float, 
        the learning rate
    optim_class : str, 
        the optimizer used
    activation_fn: str,
        activation function for the neural network
    weight_intensity: float,
        important ! the weight for emphasizing the denser cell. Lower values tend to weight each cell equally.
    R_weight : float,
        the weight for penalizing for the PINN residule loss
    pop_weight : float, 
        the weight for penalizing population
    deltax_weight : float,
        the weight for penalizing how v is similar to the sampled delta X
    D_penalty : float , 
        default None the weight for penalizing D


    Examples
    --------
    >>> import pseudodynamics as pdp
    >>> from pseudodynamics import models
    >>> config = pdp.ExperimentConfig(config=config_path)
    >>> pde_model = models.pde_params.load_from_checkpoint(
                    checkpoint_path = tompos_config.find_lastest_ckpt(), 
                    map_location='cpu')
    """
    def __init__(self, channels,
                    growth_weight=None,
                    collapse_D = True,
                    collapse_v = False,
                    g_channels=None,
                    v_channels=None,
                    D_channels=None,
                    time_sensitive=True,
                    lr=3e-4,
                    ode_tol=1e-4,
                    activation_fn:Union[str,list] = 'Tanh',
                    R_weight = None,
                    deltax_weight = None,
                    D_penalty = None,
                    weight_intensity=None,
                    time_scale_factor=None,
                    pop_weight=None,
                    cfm_weight=None,
                    D_var_weight=None,
                    neuralode_weight=None,
                    D_clip=None,
                    cfm_unbalanced_reg_m=None,
                    g_init_rate=None,
                    growth_loss_mode='legacy',   # default reproduces the published model; opt into 'logratio'/'massbalance'/'rcg' via config
                    growth_pop_ref='cellsum',
                    residual_mode='raw',
                    ema_decay=None,
                    rcg_warmup_steps=None,
                    rcg_clip_pct=None,
                    rcg_u_net_raw_weight=None,
                    n_timepoints=11,
                    cfm_loops=10,
                    d_penalty_mode='legacy',
                    zero_growth=False,
                ):

        super().__init__(channels=channels,collapse_D = collapse_D,collapse_v = collapse_v, g_channels=g_channels, v_channels=v_channels, D_channels=D_channels, 
                         time_sensitive=True,  lr=lr, ode_tol=ode_tol, activation_fn=activation_fn, 
                         D_penalty = D_penalty, weight_intensity=weight_intensity, deltax_weight=deltax_weight)
        self.save_hyperparameters()

        self.time_sensitive = time_sensitive
        self.lr = lr
        self.R_weight = 1 if R_weight is None else R_weight
        self.time_scale_factor = 5 if time_scale_factor is None else time_scale_factor
        self.D_penalty = 0.1 if D_penalty is None else D_penalty

        self.n_dim = channels[0] - 1 if time_sensitive  else channels[0]
        self.growth_weight = 0 if growth_weight is None else growth_weight
        self.log_transform = False
        self.pop_weight = pop_weight
        self.cfm_weight = 0 if cfm_weight is None else cfm_weight
        self.D_var_weight = 0 if D_var_weight is None else D_var_weight
        self.neuralode_weight = 2 if neuralode_weight is None else neuralode_weight
        # Growth-loss formulation (see training_step). 'logratio' (default): scale-free
        # d ln N/dt = E_p[g]. 'massbalance': N_t + ∫∫ g·u_int matched to N_{t+1} in log space
        # using the pop-scaled neural-ODE density. 'legacy': original .sum()/log-transform path.
        # 'growth_pop_ref' chooses the observed reference for N_{t+1}/N_t: 'cellsum' (batch cell
        # sums, self-normalising) or 'popmean' (the true pop_mean ratio carried in the batch).
        self.growth_loss_mode = (growth_loss_mode or 'logratio')
        self.growth_pop_ref = (growth_pop_ref or 'cellsum')
        assert self.growth_loss_mode in ('legacy', 'logratio', 'massbalance'), \
            f"growth_loss_mode must be legacy|logratio|massbalance, got {self.growth_loss_mode!r}"
        assert self.growth_pop_ref in ('cellsum', 'popmean'), \
            f"growth_pop_ref must be cellsum|popmean, got {self.growth_pop_ref!r}"
        # Opt-in: normalize the FP residual by u so it supervises g directly (the
        # g = rhs/u continuity inversion), scale-free, instead of the raw u**2-scaled
        # MSE. Default 'raw' preserves original behaviour.
        self.residual_mode = (residual_mode or 'raw')
        assert self.residual_mode in ('raw', 'ginv', 'rcg'), \
            f"residual_mode must be raw|ginv|rcg, got {self.residual_mode!r}"

        # RCG (Residual-Centered Growth) parameters
        self.ema_decay           = 0.95 if ema_decay is None else float(ema_decay)
        self.rcg_warmup_steps    = 300  if rcg_warmup_steps is None else int(rcg_warmup_steps)
        self.rcg_clip_pct        = 0.02 if rcg_clip_pct is None else float(rcg_clip_pct)
        self.rcg_u_net_raw_weight = 0.05 if rcg_u_net_raw_weight is None else float(rcg_u_net_raw_weight)
        self.n_timepoints        = int(n_timepoints)
        self.cfm_loops           = int(cfm_loops)
        self.d_penalty_mode      = str(d_penalty_mode)
        assert self.d_penalty_mode in ('legacy', 'mean'), \
            f"d_penalty_mode must be legacy|mean, got {self.d_penalty_mode!r}"
        # Opt-in: hard-clamp the diffusion field into (lo, hi) wherever D enters
        # the PDE dynamics (GRN-3D experiment). Default None preserves original
        # behaviour. Accepts "lo,hi" (CLI/config string) or a (lo, hi) sequence.
        if D_clip is None:
            self.D_clip = None
        else:
            parts = [p for p in D_clip.split(",") if p.strip() != ""] if isinstance(D_clip, str) else list(D_clip)
            if len(parts) != 2:
                raise ValueError(f"D_clip must be 'lo,hi' (exactly two values); got {D_clip!r}")
            _lo, _hi = float(parts[0]), float(parts[1])
            if _lo > _hi:
                raise ValueError(f"D_clip lo ({_lo}) must be <= hi ({_hi})")
            self.D_clip = (_lo, _hi)
        # Opt-in: use UNBALANCED OT (marginal relaxation reg_m) for the CFM velocity-loss
        # pairing instead of balanced OT. None (default) -> balanced (original behaviour).
        self.cfm_unbalanced_reg_m = None if cfm_unbalanced_reg_m is None else float(cfm_unbalanced_reg_m)

        # EMA buffers for RCG: per-timepoint smoothed density-weighted residual mean.
        # Stored in state_dict (survives checkpoint) but NOT in optimizer.
        # Initialized lazily (NaN) to avoid cold-start bias; first-batch value used directly.
        self.register_buffer('_Eg_resid_ema',
            torch.full((self.n_timepoints,), float('nan')))
        self.register_buffer('_ema_initialized',
            torch.zeros(self.n_timepoints, dtype=torch.bool))

        MLP_Module = MLP_surrogate
        # u_theta, density function
        self.u = MLP_surrogate(channels = channels, activation_fn=activation_fn, time_sensitive=True)

        # the output for growth is always 1
        if g_channels  is None:
            g_channels = channels[:-1] + [1]
        self.g = MLP_Module(channels = g_channels, activation_fn=activation_fn, time_sensitive=time_sensitive)
        # ABLATION (R2.3): force the growth field g(s,t) == 0 everywhere by swapping the
        # g-net for a parameterless zero stub. Every g(s,t) call site then returns 0 with
        # no further change; drift and diffusion are untouched. Captured by
        # save_hyperparameters() above so load_from_checkpoint restores the stub.
        self.zero_growth = bool(zero_growth)
        if self.zero_growth:
            self.g = _ZeroField(time_sensitive=time_sensitive)
        # Opt-in: warm-start the growth net to a population-derived mean rate so g does not
        # collapse toward 0. Sets the output-layer bias to g_init_rate and shrinks its weights,
        # so g(x,t) ~= g_init_rate at init; the (per-cell) growth loss then shapes the
        # spatial/temporal structure. Mirrors DeepRUOT's Phase-1 growth pretrain. Default None
        # = standard init (original behaviour).
        self.g_init_rate = None if g_init_rate is None else float(g_init_rate)
        if self.g_init_rate is not None and not self.zero_growth:
            with torch.no_grad():
                last_lin = [m for m in self.g.u_theta if isinstance(m, nn.Linear)][-1]
                last_lin.weight.mul_(0.01)
                last_lin.bias.fill_(self.g_init_rate)

        # if we choose to collapse v, that means the parameter is the same for all dimension
        if v_channels is None:
            v_channels = channels[:-1] + [1] if collapse_v else channels[:-1] + [self.n_dim]
        self.v = MLP_Module(channels = v_channels, activation_fn=activation_fn, time_sensitive=time_sensitive)

        # if we choose to collapse D, that means the parameter is the same for all dimension
        if D_channels is None:
            D_channels = channels[:-1] + [1] if collapse_D else channels[:-1] + [self.n_dim]
        self.D = MLP_Module(channels = D_channels, activation_fn=activation_fn, time_sensitive=time_sensitive)

    def get_u(self, s, t):
        logu = self.u(s, t) 
        u_pred = torch.exp(logu)
        return u_pred

    def forward(self, t, states):
        return self.ode_func(t, states)
    
    def ode_func(self, t, states):
        """
        the function used for odeint
        """
        s = states[1]
        device = s.device
        t_in = torch.full((s.shape[0],1), t.item()*self.time_scale_factor).float().to(device)

        with torch.set_grad_enabled(True):  

            s.requires_grad_(True)
            t_in.requires_grad_(True)

            # u = torch.exp(self.u(s, t_in)) # make sure it is u but not log u

            _, growth, drift, diffuse = self.equation(s, t_in)

            dudt = growth - drift + diffuse

            # set ds to zeros to fix cellstates
            ds = torch.zeros_like(s).float().to(device).requires_grad_(True)

            duds_by_time = ds  #self.gradient_of(dudt.sum(), s)

        return dudt, ds, duds_by_time, growth, drift, diffuse


    def forward_density_loss(self, s, t, ut):

        # loss 1 : boundary loss
        with torch.set_grad_enabled(True):
            s.requires_grad_(True)
            log_u_pred = self.u(s,t)

        # REVIEW (suggestion: use `log_u_pred + 1e-10`).
        # The suggestion does NOT make sense as written. `log_u_pred` is the
        # raw MLP output and is already in log-space (see `get_u`:
        # u = exp(self.u(s,t))). It is real-valued and unbounded; there is no
        # log(0) issue to guard against. The `+1e-10` is only needed inside
        # `torch.log(ut + 1e-10)` because `ut` is a linear-space density that
        # can be exactly 0.
        log_density_loss_t = self.loss_fn(torch.log(ut+1e-10), log_u_pred)

        return log_density_loss_t

    def forward_simulation(self, s, t, tp1, ut):

        # divided by 5 to reduce the integration time
        t0 = t[0].item() / self.time_scale_factor
        t1 = tp1[0].item() / self.time_scale_factor 

        device = s.device

        zeros = torch.zeros_like(ut)
        duds_init = torch.zeros_like(s)
        init_condition = (ut, s, duds_init, zeros.clone(), zeros.clone(), zeros.clone())

        step_size = np.around((t1 - t0)/15, decimals=1).item() 
        step_size = step_size if step_size > 0 else 0.05
        step_size = min(step_size, 0.4)

        u_int, s_t, duds, growth, drift, diffuse = odeint_adjoint(
                        self,
                        y0 = init_condition,
                        t = torch.tensor([t0, t1]).type(torch.float32).to(device),
                        atol=self.ode_tol,
                        rtol=self.ode_tol,
                        method='dopri5',
                        adjoint_options={'norm':'seminorm'},
                    )
        u_int = nn.functional.relu(u_int)
        
        return u_int, s_t, duds, growth, drift, diffuse

    def cfm_velocity_loss(self, x0, x1, t_k, t_kp1) -> torch.Tensor:
        r"""
        Conditional Flow Matching velocity loss.

        Given cell states x0 from timepoint t_k and x1 from t_{k+1},
        interpolates along straight paths and regresses the velocity network
        v_θ against the conditional velocity field u_t = (x1 - x0).

        Uses OT coupling via torchcfm when available, falling back to
        random pairing otherwise.

        Arguments
        ---------
        x0 : tensor (n, n_dim), cell states sampled from timepoint t_k
        x1 : tensor (n, n_dim), cell states sampled from timepoint t_{k+1}
        t_k : tensor (n,), normalised time at t_k
        t_kp1 : tensor (n,), normalised time at t_{k+1}
        """
        reg_m = getattr(self, 'cfm_unbalanced_reg_m', None)
        if reg_m is not None:
            # Opt-in: pair x0,x1 by UNBALANCED OT (marginal relaxation reg_m) instead of
            # balanced OT. With population growth the balanced coupling forces every t_k cell
            # into the growth-reshaped t_{k+1} cloud (apparent flow toward high-growth regions);
            # unbalanced coupling lets that extra mass be "created", so the sampled pairs carry
            # the true velocity. (See R1.1 divided-5D: balanced cos 0.72 -> unbalanced 0.90.)
            try:
                import ot as _ot
                x0n = x0.detach().cpu().numpy().astype('float64')
                x1n = x1.detach().cpu().numpy().astype('float64')
                n0, n1 = x0n.shape[0], x1n.shape[0]
                M = _ot.dist(x0n, x1n, metric='sqeuclidean'); M = M / (M.mean() + 1e-9)
                P = _ot.unbalanced.sinkhorn_unbalanced(
                    np.ones(n0) / n0, np.ones(n1) / n1, M, 0.05, float(reg_m), numItermax=100)
                Pf = P.flatten().astype('float64'); ssum = Pf.sum()
                assert np.isfinite(ssum) and ssum > 0
                sel = np.random.choice(Pf.shape[0], size=n0, p=Pf / ssum)
                idx0, idx1 = sel // n1, sel % n1
            except Exception:
                idx0 = np.arange(x0.shape[0])
                idx1 = np.random.randint(x1.shape[0], size=x0.shape[0])
            x0p, x1p = x0[idx0], x1[idx1]
            tau = torch.rand(x0p.shape[0], 1, device=x0.device)
            x_t = (1 - tau) * x0p + tau * x1p
            u_t = x1p - x0p
            tau = tau.squeeze(1)
        else:
            try:
                from torchcfm.conditional_flow_matching import (
                    ExactOptimalTransportConditionalFlowMatcher,
                )
                cfm = ExactOptimalTransportConditionalFlowMatcher(sigma=0.0)
                # Returns: interpolation time τ, x_τ, target velocity u_τ
                tau, x_t, u_t = cfm.sample_location_and_conditional_flow(x0, x1)
            except Exception:
                # Fallback: random pairing, uniform τ
                tau = torch.rand(x0.shape[0], 1, device=x0.device)
                x_t = (1 - tau) * x0 + tau * x1
                u_t = x1 - x0
                tau = tau.squeeze(1)

        # Map interpolation time τ ∈ [0,1] to actual normalised time
        t_actual = t_k[:x_t.shape[0]] + tau * (t_kp1[:x_t.shape[0]] - t_k[:x_t.shape[0]])
        t_in = t_actual.unsqueeze(1)  # (n, 1)

        # Predict velocity at interpolated point
        v_pred = self.v(x_t, t_in)

        # v is dx/dt in real-time; u_t = x1-x0 is per-τ-unit displacement, divide by Δt to convert to per-real-time-unit drift
        dt = (t_kp1[:x_t.shape[0]] - t_k[:x_t.shape[0]]).unsqueeze(1).clamp(min=1e-6)
        loss = torch.mean((v_pred - u_t / dt) ** 2)
        return loss

    def residual_loss(self, s, t) -> torch.Tensor:
        """
        calculate the loss for collocation points, this loss inject the pde into the neural network

        Input
        ------
        s: the cell state,
        t: experimental time
        """
        with torch.set_grad_enabled(True):
            s.requires_grad_(True)
            t.requires_grad_(True)

            dudt, growth, drift, diffuse = self.equation(s, t)
            if self.residual_mode == 'ginv':
                assert not self.log_transform, \
                    "residual_mode='ginv' requires a linear-density model (log_transform=False); " \
                    "for log-density models dudt and get_u are in different spaces."
                # Normalized residual = residual / u = g - g_target, with
                # g_target = (du/dt + div(vu) - div(D grad u)) / u, the continuity-
                # implied growth rate. Scale-free (O(g)), weights every cell equally,
                # and supervises g directly. Target detached so only g learns from it
                # (u, v, D stay fixed by their own losses -> protects the CFM velocity).
                # u floored to avoid 1/u blow-up in the lowest-density cells.
                u = self.get_u(s, t).reshape(-1)
                u_safe = u.clamp(min=(u.mean() * 1e-2).clamp(min=1e-9))
                g_target = ((dudt + drift - diffuse).reshape(-1) / u_safe).detach()
                g_pred = self.g(s, t).reshape(-1)
                return torch.mean((g_pred - g_target) ** 2)
            rhs = growth - drift + diffuse
        return self.loss_fn(rhs.squeeze(), dudt.squeeze())
        # return self.loss_fn(torch.log(rhs.squeeze()+1e-10), torch.log(dudt.squeeze()+1e-10))

    def residual_loss_raw(self, s, t):
        """Raw FP residual (u^2-weighted MSE). Provides a gradient path into u_net in RCG mode."""
        saved = self.residual_mode
        self.residual_mode = 'raw'
        loss = self.residual_loss(s, t)
        self.residual_mode = saved
        return loss

    def rcg_loss(self, s, t, tp1, ut, relmass, i_t):
        """
        Residual-Centered Growth (RCG) loss.

        Spatial signal: ginv target r_i approximates per-cell g at FP solution.
        Mean correction: subtract EMA-smoothed density-weighted batch mean Eg_resid,
                         which tracks the drifting intercept from u_net's temporal error.
        Magnitude anchor: add exact Eg_true = log(relmass)/dt_g (zero sampling variance).

        Only g_net receives gradient. All target computation is detached.
        """
        assert not self.log_transform, \
            "rcg_loss requires a linear-density model (log_transform=False)"

        with torch.set_grad_enabled(True):
            s = s.detach().requires_grad_(True)
            t = t.detach().requires_grad_(True)
            dudt, growth, drift, diffuse = self.equation(s, t)
            u = self.get_u(s, t).reshape(-1)
            u_safe = u.clamp(min=(u.mean() * 1e-2).clamp(min=1e-9))
            r_raw = ((dudt + drift - diffuse).reshape(-1) / u_safe).detach()

        r_raw = torch.nan_to_num(r_raw, nan=0.0)  # guard quantile against upstream NaN

        # Percentile clip: suppress 1/u_safe blow-up at low-density cells
        if self.rcg_clip_pct > 0:
            lo = torch.quantile(r_raw, self.rcg_clip_pct)
            hi = torch.quantile(r_raw, 1.0 - self.rcg_clip_pct)
            r_clip = r_raw.clamp(min=lo, max=hi)
        else:
            r_clip = r_raw

        # SNIS ratio estimator for density-weighted mean of the residual
        u0 = ut.reshape(-1).detach()
        Eg_resid_batch = (r_clip * u0).sum() / u0.sum().clamp(min=1e-12)

        # EMA update per timepoint; buffer lives in state_dict, not in optimizer
        with torch.no_grad():
            i_t_int = int(i_t)
            if not self._ema_initialized[i_t_int]:
                self._Eg_resid_ema[i_t_int] = Eg_resid_batch.item()
                self._ema_initialized[i_t_int] = True
            else:
                self._Eg_resid_ema[i_t_int] = (
                    self.ema_decay * self._Eg_resid_ema[i_t_int]
                    + (1.0 - self.ema_decay) * Eg_resid_batch.item()
                )
        Eg_resid_smooth = self._Eg_resid_ema[i_t_int].item()

        # Exact magnitude anchor from observed population ratio; zero sampling variance
        dt_g = (tp1 - t).abs().reshape(-1)[0].clamp(min=1e-6, max=10.0) / self.time_scale_factor
        Eg_true = torch.log(relmass.reshape(-1)[0].clamp(min=1e-9)) / dt_g

        # Per-cell corrected target: fully detached, no gradient anywhere in the target
        g_target_rcg = (r_clip - Eg_resid_smooth + Eg_true.item()).detach()

        # MSE: gradient flows only through g_pred (g_net parameters)
        g_pred = self.g(s.detach(), t.detach()).reshape(-1)
        return torch.mean((g_pred - g_target_rcg) ** 2), Eg_true.item(), Eg_resid_smooth


    def training_step(self, train_batch, index):
        
        # cellstate, t, t+1, u_t, u_{t+1}
        s = train_batch['s']
        t = train_batch['t']
        tp1 =train_batch['tp1'] 
        ut = train_batch['ut']
        utp1 = train_batch['utp1']
        deltax = train_batch['deltax']

        # loss 1 : boundary loss
        log_density_loss_t = self.forward_density_loss(s, t, ut)
        log_density_loss_tp1 = self.forward_density_loss(s, tp1, utp1)

        
        # loss 2 : dynamics 
        u_int, s_t, duds, growth, drift, diffuse = self.forward_simulation(s, t, tp1, ut)

        log_sim_loss_tp1 = self.loss_fn(torch.log(utp1+1e-10), torch.log(u_int[-1]+1e-10)) 


        # loss 3 : constrain related loss
        # REVIEW (suggestion: residual should use intermediate timepoints).
        # Valid suggestion. Here R_loss only evaluates collocation points at
        # the observed `t` of the batch, whereas `pde_params_fastmode` already
        # samples `t_rand ~ U(t, tp1)` for the same purpose. The residual does
        # NOT go through odeint, so adding intermediate t-samples is cheap;
        # the user's hypothesis about odeint runtime is the right reason
        # odeint itself is sparse, but it doesn't apply to residual_loss.
        R_loss = self.residual_loss(s, t)
        D_norm = self.restrict_D(s, t, exp=False)    # constrain D
        v_loss = self.constrain_v(s,t,deltax)        # constrain v by local velocity

        # RCG: replace R_loss with mean-corrected ginv target + relmass magnitude anchor
        _rcg_Eg_true = 0.0
        _rcg_Eg_smooth = 0.0
        raw_R_for_u = torch.tensor(0.0, device=s.device)
        if self.residual_mode == 'rcg':
            i_t = int(train_batch.get('i_t', 0))
            r_eff = self.R_weight * min(1.0, self.global_step / max(1, self.rcg_warmup_steps)) \
                    if self.rcg_warmup_steps > 0 else float(self.R_weight)
            if self.global_step >= self.rcg_warmup_steps:
                R_loss, _rcg_Eg_true, _rcg_Eg_smooth = self.rcg_loss(
                    s, t, tp1, ut, train_batch['relmass'], i_t)
                # Raw FP residual: intentionally trains BOTH u_net and g_net toward FP
                # consistency (the spec's u_net+g_net anchor). Do NOT detach g here.
                raw_R_for_u = self.residual_loss_raw(s, t)
            growth_weight_eff = 0.0
            growth_loss = torch.tensor(0.0, device=s.device)
        else:
            r_eff = float(self.R_weight)
            growth_weight_eff = self.growth_weight

        # constrain g by population size (skipped in RCG mode)
        if self.residual_mode == 'rcg':
            pass  # growth_loss already zeroed above
        elif growth_weight_eff == 0:
            # No population-size constraint (R2.3 --equal_mass C-arms, or any growth_weight=0
            # run): skip the growth loss. It is multiplied by 0 anyway, and the legacy form
            # divides by mass_gain, which -> 0 under equal-mass targets (NaN). Keep it at 0.
            growth_loss = torch.tensor(0.0, device=s.device)
        elif self.growth_loss_mode == 'logratio':
            # d ln N/dt = E_p[g]  =>  log(N_{t+1}/N_t) = ∫ E[g] dτ  (trapezoid over the
            # interval). Scale-free: depends only on g averaged over the batch cells, not on
            # the absolute scale of the free density net, so g can no longer hide behind a
            # mis-scaled u. Globally the transport terms integrate to zero (divergence
            # theorem), so total mass gain == total growth.
            # g_net = time_scale_factor * g_real (the ODE integrates in scaled time, and
            # predict_param divides g by time_scale_factor), so the real-time interval is
            # dt_real / time_scale_factor.
            dt_g = (tp1 - t).abs().reshape(-1)[0].clamp(min=1e-6) / self.time_scale_factor
            u0 = (torch.exp(ut) if self.log_transform else ut).reshape(-1)
            u1 = (torch.exp(utp1) if self.log_transform else utp1).reshape(-1)
            g_t = self.g(s, t).reshape(-1)
            g_tp1 = self.g(s, tp1).reshape(-1)
            # population-weighted mean E_p[g]: cells are sampled UNIFORMLY, so weight by the
            # observed density u to recover the density-weighted mean the identity requires.
            Eg_t = (g_t * u0).sum() / u0.sum().clamp(min=1e-12)
            Eg_tp1 = (g_tp1 * u1).sum() / u1.sum().clamp(min=1e-12)
            lnN_pred = 0.5 * (Eg_t + Eg_tp1) * dt_g
            if self.growth_pop_ref == 'popmean' and train_batch.get('relmass', None) is not None:
                lnN_true = torch.log(train_batch['relmass'].reshape(-1)[0].clamp(min=1e-9))
            else:
                lnN_true = torch.log(u1.sum().clamp(min=1e-9)) - torch.log(u0.sum().clamp(min=1e-9))
            growth_loss = (lnN_pred - lnN_true) ** 2
        elif self.growth_loss_mode == 'massbalance':
            # N_{t+1} = N_t + ∫∫ g·u dt, integrated with the POP-SCALED neural-ODE density
            # (u at t0 = ut data, u at t1 = u_int[-1] integrated) so the two sides share units;
            # compared in log space. drift/diffusion stay on u_net and vanish in the global sum.
            # g_net = time_scale_factor * g_real -> divide the interval by time_scale_factor.
            dt_g = (tp1 - t).abs().reshape(-1)[0].clamp(min=1e-6) / self.time_scale_factor
            u0 = (torch.exp(ut) if self.log_transform else ut).reshape(-1)
            u1 = (torch.exp(utp1) if self.log_transform else utp1).reshape(-1)
            u1_int = u_int[-1].reshape(-1)
            g_t = self.g(s, t).reshape(-1)
            g_tp1 = self.g(s, tp1).reshape(-1)
            integ_growth = 0.5 * (g_t * u0 + g_tp1 * u1_int) * dt_g
            pred_Ntp1 = u0.sum() + integ_growth.sum()
            if self.growth_pop_ref == 'popmean' and train_batch.get('relmass', None) is not None:
                true_Ntp1 = u0.sum() * train_batch['relmass'].reshape(-1)[0]
            else:
                true_Ntp1 = u1.sum()
            growth_loss = (torch.log(pred_Ntp1.clamp(min=1e-9)) - torch.log(true_Ntp1.clamp(min=1e-9))) ** 2
        elif self.log_transform:
            left = torch.exp(utp1).sum()
            right = torch.exp(ut + growth[-1]).sum()
            growth_loss = self.loss_fn(torch.log(left), torch.log(right))
        else:
            mass_gain = utp1.sum() -  ut.sum()
            predicted_gain = growth[-1].sum()
            # REVIEW (suggestion: divide by |mass_gain| or |mass_gain|+|predicted_gain|+1e-10).
            # The suggestion is correct -- this is a real sign-flip bug.
            # `loss_fn` returns a non-negative scalar; dividing by a signed
            # `mass_gain` flips the gradient sign when the population is
            # shrinking (mass_gain < 0), so SGD pushes `predicted_gain` away
            # from `mass_gain` instead of toward it. The
            # `|mass_gain|+|predicted_gain|+1e-10` form is also better
            # conditioned when both are near zero. Decision pending.
            growth_loss = self.loss_fn(mass_gain, predicted_gain, weight=2) / mass_gain
        

        # loss 4 (optional): Conditional Flow Matching velocity loss
        cfm_loss = torch.tensor(0.0, device=s.device)
        if self.cfm_weight > 0 and 'cfm_x0' in train_batch:
            for _ in range(self.cfm_loops):
                cfm_loss += self.cfm_velocity_loss(
                    train_batch['cfm_x0'], train_batch['cfm_x1'], t, tp1,
                )

        # loss 5 (optional): Diffusion field variance-matching + entropy losses
        D_var_loss = torch.tensor(0.0, device=s.device)
        if self.D_var_weight > 0:
            D_var_loss = (
                self.diffusion_variance_loss(s, t, tp1) +
                self.diffusion_entropy_loss(s, t, ut, utp1)
            )

        total_loss = log_density_loss_t + log_density_loss_tp1 + \
                    self.neuralode_weight * log_sim_loss_tp1 + \
                    self.D_penalty * D_norm + \
                    self.deltax_weight * v_loss + \
                    growth_weight_eff * growth_loss + \
                    self.cfm_weight * cfm_loss + \
                    self.D_var_weight * D_var_loss + \
                    r_eff * R_loss + \
                    self.rcg_u_net_raw_weight * raw_R_for_u


        with torch.no_grad():
            # self.log("residual_loss", Loss_r, on_epoch=True)
            # self.log("boundary_loss", Loss_b, on_epoch=True)
            self.log("population_loss", growth_loss.item(), on_epoch=True)
            if self.cfm_weight > 0:
                self.log("cfm_loss", cfm_loss.item(), on_epoch=True)
            if self.D_var_weight > 0:
                self.log("D_var_loss", D_var_loss.item(), on_epoch=True)
            self.log("boundary_loss",  log_density_loss_t.item(),  on_epoch=True)
            self.log("residual loss", R_loss.item(), on_epoch=True)
            self.log("integrat_loss", log_sim_loss_tp1.item(), on_epoch=True)
            self.log("total_loss", total_loss, on_epoch=True, prog_bar=True)
            if self.residual_mode == 'rcg':
                self.log("rcg_Eg_true",   float(_rcg_Eg_true),   on_epoch=True)
                self.log("rcg_Eg_smooth", float(_rcg_Eg_smooth), on_epoch=True)
                self.log("rcg_r_eff",     float(r_eff),          on_epoch=True)
                self.log("rcg_magnitude_correction",
                         float(_rcg_Eg_true) - float(_rcg_Eg_smooth), on_epoch=True)
                with torch.set_grad_enabled(False):
                    g_pred_mean = self.g(s.detach(), t.detach()).mean()
                self.log("g_pred_mean", g_pred_mean.item(), on_epoch=True)

        return total_loss


    def validation_step(self, val_batch, index):
        loss = self.training_step(val_batch, index)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        
