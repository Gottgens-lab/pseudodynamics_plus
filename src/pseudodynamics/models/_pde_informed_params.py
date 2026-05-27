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

    def equation(self, s, t) -> tuple:
        """
        Apply torch's auto grad to compute the dynamics
        
        based on the following equation:
            ∂u/∂t = ∂/∂s[ D* ∂u/∂s ] - ∂/∂s[ v*u ] + g*u
        
        we calcuate the left hand side (lhs) and the right hand side
        """
        u = self.get_u(s,t) # make sure it is u
        D = self.D(s,t)
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
        """
        D = self.D(s,t)
        if exp:
            D = torch.exp(D)
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
                D_pred = self.D(s_in, t_in)
                
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
                    neuralode_weight=None
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
       
        MLP_Module = MLP_surrogate
        # u_theta, density function
        self.u = MLP_surrogate(channels = channels, activation_fn=activation_fn, time_sensitive=True)

        # the output for growth is always 1
        if g_channels  is None:
            g_channels = channels[:-1] + [1]
        self.g = MLP_Module(channels = g_channels, activation_fn=activation_fn, time_sensitive=time_sensitive)

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

        # boundary u of the current timepoint
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
            rhs = growth - drift + diffuse

        return self.loss_fn(rhs.squeeze(), dudt.squeeze())
        # return self.loss_fn(torch.log(rhs.squeeze()+1e-10), torch.log(dudt.squeeze()+1e-10))


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
        R_loss = self.residual_loss(s, t)
        D_norm = self.restrict_D(s, t, exp=False)    # constrain D 
        v_loss = self.constrain_v(s,t,deltax)        # constrain v by local velocity   
        
        # duds_loss = -1 * nn.functional.cosine_similarity(duds[-1], duds_tp1)
        # constrain g by population size
        if self.log_transform:
            left = torch.exp(utp1).sum()
            right = torch.exp(ut + growth[-1]).sum()
            growth_loss = self.loss_fn(torch.log(left), torch.log(right))
        else:
            mass_gain = utp1.sum() -  ut.sum()
            predicted_gain = growth[-1].sum()
            growth_loss = self.loss_fn(mass_gain, predicted_gain, weight=2) / mass_gain
        

        # loss 4 (optional): Conditional Flow Matching velocity loss
        cfm_loss = torch.tensor(0.0, device=s.device)
        if self.cfm_weight > 0 and 'cfm_x0' in train_batch:
            for i in range(10):
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
                    self.growth_weight * growth_loss + \
                    self.cfm_weight * cfm_loss + \
                    self.D_var_weight * D_var_loss + \
                    self.R_weight * R_loss


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

        return total_loss


    def validation_step(self, val_batch, index):
        loss = self.training_step(val_batch, index)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        
    def stratified_ode(self, t, states):
        """
        the function used for odeint
        """
        s = states[1]
        device = s.device
        t_in = torch.full((s.shape[0],1), t.item()*5).float().to(device)

        with torch.set_grad_enabled(True):

            s.requires_grad_(True)
            t_in.requires_grad_(True)

            u = torch.exp(self.u(s, t_in)) # make sure it is u but not log u

            _, growth, drift, diffuse = self.equation(s, t_in)

            dudt = growth - drift + diffuse

            # set ds to zeros to fix cellstates
            ds = torch.zeros_like(s).float().to(device).requires_grad_(True)

        return (dudt, ds, growth, drift, diffuse)

    def statify_flow(self, train_DS, batch_size=1024, window_size=1):
        r"""
        stratify the constribution of cells

        Arguments:
        -----------
        train_DS: highdim_DS class
        batch_size : batch_size for looping the adataset
        window_size: the time window for integral, 1 for next timepoint

        Returns
        -----------
        dilution_flow : density gain from growth
        dirft_flow : density gain from differetiation
        diffuse_flow : density gain from random diffusion
        """
        # get device from module
        device = next(self.u.parameters()).device

        n_timepoint = train_DS.n_timepoint
        t_list = train_DS.T_b / self.time_scale_factor
        u_b = train_DS.u_b.reshape(n_timepoint, -1)
        cellstate = torch.from_numpy(train_DS.cellstate).float().requires_grad_()

    
        dilution_flow = []
        dirft_flow = []
        diffuse_flow = []
        
        for it in range(n_timepoint-window_size):
            
            t_i = t_list[it]
            t_ip1 = t_list[it + window_size]

            g_flow_ls = []
            v_flow_ls = []
            d_flow_ls = []

            for i in range(0, cellstate.shape[0], batch_size):

                u_t0 = u_b[it, i:i+batch_size].to(device)
                s = cellstate[i:i+batch_size].to(device)
                
                # (dudt, ds, growth, -1 * drift, diffuse)
                init_condition = (u_t0, s, 
                                  torch.zeros_like(u_t0).to(device), 
                                  torch.zeros_like(u_t0).to(device), 
                                  torch.zeros_like(u_t0).to(device))
                
                u_int, s_t, g_flow, v_flow, d_flow = odeint(
                                self.stratified_ode,
                                y0 = init_condition,
                                t = torch.tensor([t_i, t_ip1]).type(torch.float32).to(device),
                                atol = self.ode_tol,
                                rtol = self.ode_tol,
                                method='dopri5',
                            )

                # append for different batches
                g_flow_ls.append( g_flow[1].detach().cpu().numpy() )
                v_flow_ls.append( v_flow[1].detach().cpu().numpy() )
                d_flow_ls.append( d_flow[1].detach().cpu().numpy() )

                # torch.cuda.empty_cache()
            
            # append for different timepoints
            dilution_flow.append( np.concatenate(g_flow_ls, axis=0) )
            dirft_flow.append( np.concatenate(v_flow_ls, axis=0) )
            diffuse_flow.append( np.concatenate(d_flow_ls, axis=0) )
        
        
        return np.stack(dilution_flow), np.stack(dirft_flow), np.stack(diffuse_flow)

class pde_params_fastmode(pde_params):
    def __init__(self, channels, growth_weight=None, collapse_D = True, collapse_v = False, g_channels=None, v_channels=None, D_channels=None, time_sensitive=True, lr=3e-4, ode_tol=1e-4, activation_fn:Union[str, list] = 'Tanh', deltax_weight = None, D_penalty = None, weight_intensity=None, time_scale_factor=None, pop_weight=None):
        r"""
        pde model for fast mode data
        PINN prediction + NeuralODE simulation to estimate parameters

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
        super().__init__(channels=channels, collapse_D = collapse_D, collapse_v = collapse_v, g_channels=g_channels, v_channels=v_channels, D_channels=D_channels, time_sensitive=True, lr=lr, ode_tol=ode_tol, activation_fn=activation_fn, D_penalty = D_penalty, weight_intensity=weight_intensity, deltax_weight=deltax_weight)
        self.save_hyperparameters()

    def training_step(self, train_batch, index):
        
        # cellstate, t, t+1, u_t, u_{t+1}
        s = train_batch['s']
        t = train_batch['t']
        tp1 =train_batch['tp1'] 
        ut = train_batch['ut']
        utp1 = train_batch['utp1']
        deltax = train_batch['deltax']
        s_origin = train_batch['s_origin']
        # loss 1 : boundary loss
        log_density_loss_t = self.forward_density_loss(s, t, ut)
        log_density_loss_tp1 = self.forward_density_loss(s, tp1, utp1)

        
        # loss 2 : dynamics Neural ODE
        u_int, s_t, duds, growth, drift, diffuse = self.forward_simulation(s, t, tp1, ut)

        log_sim_loss_tp1 = self.loss_fn(torch.log(utp1+1e-10), torch.log(u_int[-1]+1e-10)) 


        # loss 3 : constrain related loss : use the original cell state
        t_rand = torch.Tensor(s.shape[0],1).uniform_(t[0].item(), tp1[0].item()).float().to(s.device)
        R_loss = self.residual_loss(s_origin, t_rand)       # constrain PINN
        D_norm = self.restrict_D(s_origin, t, exp=False)    # constrain D 
        v_loss = self.constrain_v(s_origin,t,deltax)        # constrain v by local velocity   
        
        # duds_loss = -1 * nn.functional.cosine_similarity(duds[-1], duds_tp1)
        # constrain g by population size
        if self.log_transform:
            left = torch.exp(utp1).sum()
            right = torch.exp(ut + growth[-1]).sum()
            growth_loss = self.loss_fn(torch.log(left), torch.log(right))
        else:
            mass_gain = utp1.sum() -  ut.sum()
            predicted_gain = growth[-1].sum()
            growth_loss = self.loss_fn(mass_gain, predicted_gain, weight=2) / mass_gain
        

        total_loss = log_density_loss_t + log_density_loss_tp1 + \
                    2 * log_sim_loss_tp1 + \
                    self.R_weight * R_loss +\
                    self.D_penalty * D_norm + \
                    self.deltax_weight * v_loss + \
                    self.growth_weight * growth_loss


        with torch.no_grad():
            # self.log("residual_loss", Loss_r, on_epoch=True)
            # self.log("boundary_loss", Loss_b, on_epoch=True)
            self.log("population_loss", growth_loss.item(), on_epoch=True)
            self.log("boundary_loss",  log_density_loss_t.item(),  on_epoch=True)
            self.log("residual loss", R_loss.item(), on_epoch=True)
            self.log("integrat_loss", log_sim_loss_tp1.item(), on_epoch=True)
            self.log("total_loss", total_loss, on_epoch=True, prog_bar=True)

        return total_loss
    
class log_pde_params(pde_params):
    def __init__(self, channels, growth_weight=None, collapse_D = True, collapse_v = False, g_channels=None, v_channels=None, D_channels=None, time_sensitive=True, lr=3e-4, ode_tol=1e-4, activation_fn:Union[str, list] = 'Tanh', deltax_weight = None, D_penalty = None, weight_intensity=None, time_scale_factor=None, pop_weight=None):
        r"""
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
        super().__init__(channels=channels, collapse_D = collapse_D, collapse_v = collapse_v, g_channels=g_channels, v_channels=v_channels, D_channels=D_channels, time_sensitive=True, lr=lr, ode_tol=ode_tol, activation_fn=activation_fn, D_penalty = D_penalty, weight_intensity=weight_intensity, deltax_weight=deltax_weight, time_scale_factor=time_scale_factor, pop_weight=pop_weight)
        self.save_hyperparameters()
        self.log_transform = True

    def equation(self, s, t) -> tuple:
        """
        the log Reaction-Advection Diffusion equation
        """
        logu = self.u(s,t)
        D = self.D(s,t)
        v = self.v(s,t)
        g = self.g(s,t)

        # first-order derivative
        dloguds = self.gradient_of(logu.sum(), s)
        dvds = self.gradient_of(v.sum(), s)
        dDds = self.gradient_of(D.sum(), s)

        # second-order derivative
        dlogudss = self.gradient_of(dloguds.sum(), s)
        if dlogudss is None:
            dlogudss = dloguds # becomes a scaler

        duDds = self.mul(dloguds, dDds)
        
        diffusion = self.mul(D, dlogudss) + self.mul(D, duDds)

        # drift : ∇v + (∇logu·v) 
        drift = dvds.sum(dim=1) + self.mul(dloguds, v)
        growth = g
        
        return None, growth, drift, diffusion



class pde_u_free(pde_params):
    def __init__(self, channels, step_size, growth_weight=None,collapse_D = True, collapse_v = False, g_channels=None, v_channels=None, D_channels=None, time_sensitive=True, lr=3e-4, ode_tol=1e-4, activation_fn:Union[str, list] = 'Tanh', deltax_weight = None, D_penalty = None, weight_intensity=None, time_scale_factor=None):
        r"""
        high dimensional pde without surrogate u

        Arguments:
        -------------
        channel : the number of MLP channels of the Behavior function
        step_size : important, the step size for rk4 odeint
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
        super().__init__(channels=channels, growth_weight=growth_weight, collapse_D = collapse_D, collapse_v = collapse_v, g_channels=g_channels, v_channels=v_channels, D_channels=D_channels, time_sensitive=True, lr=lr, ode_tol=ode_tol, activation_fn=activation_fn, D_penalty = D_penalty, weight_intensity=weight_intensity, deltax_weight=deltax_weight, time_scale_factor=time_scale_factor)
        self.save_hyperparameters()
        self.b1 = 35 / 384
        if step_size is None:
            self.solver = "dopri5"
        else:
            self.solver = 'rk4'
        self.step_size = step_size
        self.previous_t = None
        self.step_sizes = []

    def ode_func(self, t, states):
        """
        the function used for odeint
        """
        u = states[0]
        if not self.log_transform:
            u = nn.functional.relu(u)
        s = states[1]
        duds = states[2]
        device = s.device
        t_in = torch.full((s.shape[0],1), t.item()*5).float().to(device)

        # record step size
        # if self.previous_t is not None:
        #     step_size = t.item() - self.previous_t
        #     self.step_sizes.append(step_size)
        # self.previous_t = t.item()

        with torch.set_grad_enabled(True):

            s.requires_grad_(True)
            t_in.requires_grad_(True)
            u.requires_grad_(True)
            duds.requires_grad_(True)
        
            # the first order deviritives of density u to time : ∂u/∂s
            
            _, growth, drift, diffuse = self.equation(s, t_in, u, duds)

            dudt = growth - drift + diffuse

            # set ds to zeros to fix cellstates
            ds = torch.zeros_like(s).float().to(device).requires_grad_(True)
            
            duds_by_time = self.gradient_of(dudt.sum(), s)

        return (dudt, ds, duds_by_time, growth, drift, diffuse)

    def forward(self, t, states):
        return self.ode_func(t, states)

    def equation(self, s, t, u, duds) -> tuple:
        """
        Apply torch's auto grad to compute the dynamics
        
        based on the following equation:
            ∂u/∂t = ∂/∂s[ D* ∂u/∂s ] - ∂/∂s[ v*u ] + g*u
        
        we calcuate the left hand side (lhs) and the right hand side
        """
        D = self.D(s,t)
        v = self.v(s,t)
        g = self.g(s,t)
        
        
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

        if torch.isnan(diffuse).any():
            dudss = self.gradient_of(duds.sum(), s)
            dDds = self.gradient_of(D.sum(), s)
            if torch.isnan(dudss).any():
                diffuse = self.mul(dDds, duds)
            else:
                diffuse = self.mul(D, dudss) + self.mul(dDds, duds)
                
        return None, growth, drift, diffuse

    def training_step(self, train_batch, index):        
        # cellstate : [b, cellstate_dim], t, t+1, u_t, u_{t+1} : [b,], duds : [b, 2, cellstate_dim]
        s, t, tp1, ut, utp1, deltax, du_dDeltax = train_batch

        # if the input data contains any negative values
        # then 
        self.log_transform = torch.any(ut<0).item()

        # divided by 5 to reduce the integration time
        t0 = t[0].item() / self.time_scale_factor
        t1 = tp1[0].item() / self.time_scale_factor 

        device = s.device

        # loss 2 : dynamics 

        # init_condition 
        zeros = torch.zeros_like(ut)
        init_condition = (ut, s, du_dDeltax[:,0], zeros.clone(), zeros.clone(), zeros.clone())

        u_int, s_t, duds, growth, drift, diffuse = odeint_adjoint(
                        self,
                        y0 = init_condition,
                        t = torch.tensor([t0, t1]).type(torch.float32).to(device),
                        atol=self.ode_tol,
                        rtol=self.ode_tol,
                        method=self.solver,
                        options={'step_size':self.step_size},
                        adjoint_options={'norm':'seminorm'},
                    )

        # boundary u of  the next timepoint
        utp1_loss = self.loss_fn(u_int[-1], utp1)
        
        if not self.log_transform:
            u_int = nn.functional.relu(u_int)
            tp1_intput = torch.log(utp1+1e-30)
            tp1_sim_out = torch.log(u_int[-1]+1e-30)
        else:
            tp1_sim_out = u_int[-1]
            tp1_intput = utp1

        log_utp1_loss = self.loss_fn(tp1_intput, tp1_sim_out)

        D_norm = self.restrict_D(s, t, exp=False)
        v_loss = self.constrain_v(s, t, deltax)
        duds_loss = -1 * nn.functional.cosine_similarity(duds[-1],du_dDeltax[:,-1])

        
        if self.log_transform:
            left = torch.exp(utp1).sum()
            right = torch.exp(ut + growth[-1]).sum()
            growth_loss = self.loss_fn(torch.log(left), torch.log(right)) / left
        else:
            log_mass_gain = torch.log(nn.functional.relu(utp1.sum() -  ut.sum()) + 1e-30)
            log_predicted_gain = torch.log(nn.functional.relu(growth[-1].sum()) + 1e-30)
            growth_loss = self.loss_fn(log_mass_gain, log_predicted_gain) 


        total_loss =  2 * log_utp1_loss + self.D_penalty * D_norm + self.deltax_weight * v_loss + duds_loss.mean() + self.growth_weight * growth_loss


        with torch.no_grad():
            self.log("integrat_loss", utp1_loss.item(),  on_epoch=True)
            self.log("log_integrat_loss", log_utp1_loss.item(), on_epoch=True)
            self.log("total_loss", total_loss, on_epoch=True, prog_bar=True)

        return total_loss
        
class logrithmic_pde(pde_u_free):
    def __init__(self, channels, step_size, collapse_D = True, collapse_v = False, g_channels=None, v_channels=None, D_channels=None, time_sensitive=True, lr=3e-4, ode_tol=1e-4, activation_fn:Union[str, list] = 'Tanh', deltax_weight = None, D_penalty = None, weight_intensity=None, time_scale_factor=None):
        r"""
        compute ode in logrihmic space ,

        ∂logu/∂t = g 
                   - ∇v - (∇logu·v) 
                   + D(∇^2 logu) + D(∇logu·∇D)

        Arguments:
        -------------
        channel : the number of MLP channels of the Behavior function
        step_size : important, the step size for rk4 odeint
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
        super().__init__(channels=channels, step_size=step_size, collapse_D = collapse_D, collapse_v = collapse_v, g_channels=g_channels, v_channels=v_channels, D_channels=D_channels, time_sensitive=True, lr=lr, ode_tol=ode_tol, activation_fn=activation_fn, D_penalty = D_penalty, weight_intensity=weight_intensity, deltax_weight=deltax_weight, time_scale_factor=time_scale_factor)
        self.log_transform = True
        
    def equation(self, s, t, logu, dloguds) -> tuple:
        
        D = self.D(s,t)
        v = self.v(s,t)
        g = self.g(s,t)

        # first-order derivative
        dvds = self.gradient_of(v.sum(), s)
        dDds = self.gradient_of(D.sum(), s)

        # second-order derivative
        dlogudss = self.gradient_of(dloguds.sum(), s)
        if dlogudss is None:
            dlogudss = dloguds # becomes a scaler

        duDds = self.mul(dloguds, dDds)
        
        diffusion = self.mul(D, dlogudss) + self.mul(D, duDds)
        

        # drift : ∇v + (∇logu·v) 
        drift = dvds.sum(dim=1) + self.mul(dloguds, v)
        growth = g

        # # diffusion : 1/u·∇(D·u·∇logu)
        # u = torch.exp(logu)
        # Du = self.mul(D, u)
        # Dudloguds = self.mul(Du, dloguds)

        # # second order derivatives
        # ddDdss = self.gradient_of(Dudloguds.sum(), s).sum(dim=1)
        # diffusion = torch.div(ddDdss, u)
        
        return None, growth, drift, diffusion
    

class pde_singlebranch_twotimepoints(pde_params_base):
    def __init__(self, n_grid=300, channels=11, lr=3e-4,  D_penalty = None, weight_intensity=None, ode_tol=1e-4):
        """
        Use cubic spline to fit g,v,D for single branch dataset

        Arguments:
        -------------
        [g, v, D]_channels : the number of cubic spline knots of the Behavior function
        
        kwargs 
        -------
        lr: float, the learning rate
        optim_class : str, the optimizer used
        D_penalty : float , default None the weight for penalizing D
        """
        super().__init__(channels=channels, lr=lr, ode_tol=ode_tol, D_penalty = D_penalty,collapse_D = True, collapse_v = True, g_channels=channels, v_channels=channels, D_channels=channels, time_sensitive=True,  activation_fn='Tanh')
        
        self.weight_intensity = 0.5 if weight_intensity is None else weight_intensity
        self.n_grid = n_grid
        grid = np.linspace(0, 1, n_grid)
        self.h_inv = 1 / (grid[1] - grid[0])
        
        n_knot = channels

        # self.g = KAN(width=[1,1], grid=11, k=3, seed=42, grid_range=[1,1.5], symbolic_enabled=False)
        # self.v = KAN(width=[1,1], grid=11, k=3, seed=42, grid_range=[-4,-3], symbolic_enabled=False)
        # self.D = KAN(width=[1,1], grid=11, k=3, seed=42, grid_range=[-9,-6], symbolic_enabled=False)
        self.g = CubicSpline(y = torch.full((n_knot,), fill_value=1.05) , n_knot=n_knot)
        self.v = CubicSpline(y = torch.full((n_knot,), fill_value=-3.0) , n_knot=n_knot)
        self.D = CubicSpline(y = torch.full((n_knot,), fill_value=-9.0) , n_knot=n_knot)
    
    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(
        optimizer = torch.optim.RMSprop(
            [next(module.parameters()) for module in  [self.g, self.v, self.D]],
            # self.parameters(),
                lr=self.lr)
        return optimizer

    def plot_u_dt(self, t, s, u_t, g_hat, v_hat, D_hat):
        """
        plot three curves 
        """
        # : 300
        with torch.no_grad():
            u_t = u_t.clone().cpu().numpy()
            s = s.clone().cpu().numpy()
            g_hat = g_hat.clone().cpu().numpy()
            v_hat = v_hat.clone().cpu().numpy()
            D_hat = D_hat.clone().cpu().numpy()

        fig, axs = plt.subplots(1,4, figsize=(16,3))

        axs[0].plot(s, u_t)
        axs[0].set_title("u at {:.3f} at epoch {}".format(t, self.current_epoch))

        axs[1].plot(s, g_hat)
        axs[1].set_title(f"g^")

        axs[2].plot(s, v_hat)
        axs[2].set_title(f"v^")

        axs[3].plot(s, D_hat)
        axs[3].set_title(f"D^")

        try:
            dir_path = self.trainer.checkpoint_callback.dirpath
            version = dir_path.split("/")[-2]
        except:
            version = 'Eval'
        
        save_path = f"/home/wergillius/Project/PINN_dynamics/results/singlebranch_dudt_plots/{version}/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        fig.savefig(save_path+"t={:.3f}-e={}.png".format(t, self.current_epoch))


    def ode_func(self, t, states, plotting_prob=1e-2): 
        
        
        u_t = states[0]     # : 300
        u_t = nn.functional.relu(u_t)
        s = states[1]       # : 300

        # TO get some hyper,s
        device = s.device
        batch_size = s.shape[0]  # s: 300
        h2inv =  self.h_inv**2

        # infer the dynamics params with neural networks
        s_input = s.reshape(-1,1)
        g_hat = self.g(s,None)#.squeeze(1)
        D_hat = self.D(s,None)#.squeeze(1)
        v_hat = self.v(s,None)#.squeeze(1)

        v_hat = torch.exp(v_hat) # v can only be positive
        D_hat = torch.exp(D_hat)

        if np.random.random() < plotting_prob:
            self.plot_u_dt(t, s, u_t, g_hat, v_hat, D_hat)

        # assume u_near is a square
        assert g_hat.shape == u_t.shape , "u and g does not have the same shape"

        # growth is simply the product g*u
        growth = torch.mul(g_hat, u_t)    

        # discretized drift with boundary
        drift = torch.zeros_like(u_t)
        drift[0] = v_hat[0] * 0.5 * (u_t[1] - u_t[0]) * self.h_inv
        # drift[0] = 0.5 * (v_hat[1] * (u_t[1] + u_t[2]) - v_hat[0] * (u_t[0] + u_t[1]) ) * self.h_inv
        drift[-1] = -1 * v_hat[-1] * 0.5 * (u_t[-2] + u_t[-1]) * self.h_inv
        # vb1(end)*1/2*(x(n_grid-2)+x(n_grid-1))*sqrt(h2inv)

        for i in range(1, self.n_grid-1):
            # sqrt(h2inv)*1/2*(vb1(i-1)*(x(i-1)+x(i))-vb1(i)*(x(i)+x(i+1))) # remember to invert
            drift[i] = self.h_inv * 0.5 * (v_hat[i] * (u_t[i] + u_t[i + 1]) - v_hat[i - 1] * (u_t[i - 1] + u_t[i]) )
        
        # discretized diffusion with boundary
        diffusion = torch.zeros_like(u_t)
        diffusion[0] = -1 * h2inv * (D_hat[0] * (u_t[0] - u_t[1])) 
        # diffusion[0] = h2inv * (D_hat[0] * (u_t[0] - u_t[1]) - D_hat[1] * (u_t[1] - u_t[2]))
        diffusion[-1] = h2inv * (D_hat[-1] * (u_t[-2] - u_t[-1]))
        for i in range(1, self.n_grid-1):
            diffusion[i] = h2inv * (D_hat[i - 1] * (u_t[i - 1] - u_t[i]) - D_hat[i] * (u_t[i] - u_t[i + 1]))

            # h2inv*(Db1(i-1)*(x(i-1)-x(i))-Db1(i)*(x(i)-x(i+1)))


        assert growth.shape == drift.shape
        assert growth.shape == diffusion.shape

        # the equation
        dudt = growth - drift + diffusion

        dsdt = torch.zeros_like(s).to(device)

        return (dudt, dsdt)

    def forward(self,t, states):
        return self.ode_func(t, states, plotting_prob=0)

    def training_step(self, train_batch, index):
        
        # s : n_grid, n_dimension
        # t_b : n_grid, n_timepoints, 1
        # u_b : n_grid, n_timepoints, 1
        s, t_b, u_b, hist_var, area_var, mean, var, indexs = train_batch 

        it = indexs[0].item()
        itp1 = indexs[1].item()


        # divided by 5 to reduce the integration time
    
        t_list = t_b.detach().clone().flatten()

        device = s.device

        
        # loss 2 : dynamics 
        # init_condition 


        # it = np.random.choice(range(len(t_list)-1))
        t0 = t_list[it].item()
        t1 = t_list[itp1].item()

        step_size = np.around((t1 - t0)/5, decimals=2).item() 
        step_size = step_size if step_size > 0 else 0.05

        step_size = min(step_size, 0.4)

        ut0 = u_b[it,:]
        utp1 = u_b[itp1,:]
        u_var_t0 = hist_var[it,:]
        u_var_tp1 = hist_var[itp1,:]
        a_var_t0 = area_var[it,:]
        a_var_tp1 = area_var[itp1,:]

        # init condition : ut, s in a square, u in a square
        N =  ut0.sum()
        init_condition = (ut0, s)
        u_int, s_int = odeint_adjoint(
                    self,
                    init_condition,
                    torch.Tensor([t0,t1]).type(torch.float32).to(device),
                    atol=1e-6,
                    rtol=1e-6,
                    method='dopri5',
                    # options = {'step_size': step_size}
                )

        # boundary u of  the next timepoint
        utp1_loss = self.loss_fn(u_int[-1], utp1)
        u_int = nn.functional.relu(u_int)

        with torch.no_grad():
            weight = (torch.clamp(torch.log(utp1+1e-10), min=-24) + 24)**self.weight_intensity
            weight /= weight.sum()
        log_utp1_loss = self.loss_fn(torch.log(utp1+1e-10), torch.log(u_int[-1]+1e-10), weight=weight)
        
        
        area_loss = self.area_loss(utp1, u_int[-1], var=a_var_tp1)
        density_loss = self.density_loss(utp1, u_int[-1], var=u_var_tp1)
        distribution_loss = self.distribution_loss(utp1, u_int[-1])

        boundary_loss = self.density_loss(utp1[0], u_int[-1][0], var=u_var_tp1[0]) + \
                        self.density_loss(utp1[-1], u_int[-1][-1], var=u_var_tp1[-1])



        if var[itp1] < 1:
            var_base = torch.Tensor([1.0]).to(mean.device)
        else:
            var_base = var[itp1]
        pop_loss = self.population_loss(u_int[-1], mean[itp1], var_base)
        pop_loss = torch.clamp(pop_loss, max=1000)


        s_input = s.reshape(-1,1)
        D_norm = self.restrict_D(s, torch.full(utp1.shape, t1).to(device))

        # total_loss = log_utp1_loss/self.n_grid + self.D_penalty * D_norm

        total_loss = boundary_loss + density_loss +  pop_loss + self.D_penalty * D_norm
        # distribution_loss +


        with torch.no_grad():
            self.log("integrat_loss", utp1_loss.item(),  on_epoch=True)
            self.log("area_loss", area_loss.item(),  on_epoch=True)
            self.log("density_loss", density_loss.item(),  on_epoch=True)
            self.log("distribution_loss", distribution_loss.item(), on_epoch=True)
            self.log("total_loss", total_loss, on_epoch=True, prog_bar=True)

        return total_loss


    def validation_step(self, val_batch, index):
        
        # s : n_grid, n_dimension
        # t_b : n_grid, n_timepoints, 1
        # u_b : n_grid, n_timepoints, 1
        s, t_b, u_b, mean, var, indexs = val_batch 

        # divided by 5 to reduce the integration time
        device = s.device

        it = indexs[0].item()
        itp1 = indexs[1].item()

        # init_condition 
        ut0 = u_b[it,:]
        utp1 = u_b[itp1,:]

        
        # init_condition 
        t_list = t_b.detach().clone().flatten()
        t0 = t_list[it].item()
        t1 = t_list[itp1].item()

        step_size = np.around((t1 - t0)/10, decimals=2).item() 
        step_size = step_size if step_size > 0 else 0.05

        step_size = min(step_size, 0.4)

        # init condition : ut, s in a square, u in a square
        N =  ut0.sum()
        init_condition = (ut0, s)
        u_int, s_int = odeint_adjoint(
                    self.ode_func,
                    init_condition,
                    torch.Tensor([t0,t1]).type(torch.float32).to(device),
                    atol=1e-8,
                    rtol=1e-8,
                    method='midpoint',
                    options = {'step_size': step_size}
                )

        # boundary u of  the next timepoint
        u_int = nn.functional.relu(u_int)

        with torch.no_grad():

            density_loss = self.density_loss(utp1, u_int[-1])
            area_loss = self.area_loss(utp1, u_int[-1])
            distribution_loss = self.distribution_loss(utp1, u_int[-1])

            if var[itp1] < 1:
                var_base = torch.Tensor([1.0]).to(mean.device)
            else:
                var_base = var[itp1]
            pop_loss_t = self.population_loss(u_int[-1], mean[itp1], var_base)
            # print(i, self.population_loss(u_int[i], mean[i], var[i]))
            pop_loss = torch.clamp(pop_loss_t, max=1000)

            
            weight = (torch.clamp(torch.log(utp1+1e-10), min=-24) + 24)**self.weight_intensity
            weight /= weight.sum()
            log_utp1_loss = self.loss_fn(torch.log(utp1+1e-10), torch.log(u_int[-1]+1e-10), weight=weight)    

            total_loss = area_loss + pop_loss #+ self.D_penalty * D_norm
            self.plot_u_dt(self, t0, s, ut0, utp1, u_int[-1], self.g.y.clone().detach().cpu().numpy())


            self.log("area_loss", area_loss.item(),  on_epoch=True)
            self.log("density_loss", density_loss.item(),  on_epoch=True)
            self.log("log_density_loss", log_utp1_loss.item(),  on_epoch=True)
            self.log("pop_loss", pop_loss.item(), on_epoch=True)
            self.log("kld_loss", distribution_loss.item(), on_epoch=True)
            self.log("total_loss", total_loss, on_epoch=True, prog_bar=True)

        return total_loss

    def ode_func_old(self, t, states):
        u_t = states[0]     # : 300

        s = states[1]       # : 300

        # TO get some hyper,s
        device = s.device
        batch_size = s.shape[0]  # s: 300
        h2inv =  self.h_inv**2

        # infer the dynamics params with neural networks
        s_input = s.reshape(-1,1)
        g_hat = self.g(s,None)#.squeeze(1)
        D_hat = self.D(s,None)#.squeeze(1)
        v_hat = self.v(s,None)#.squeeze(1)

        v_hat = torch.exp(v_hat) # v can only be positive


        # assume u_near is a square
        assert g_hat.shape == u_t.shape , "u and g does not have the same shape"

        # growth is simply the product g*u
        growth = torch.mul(g_hat, u_t)    
        
        # discretize ∂vu/∂s -> 1/h*[v_{i+1} u_{i+1} - v_i u_i]
        vu = torch.mul(v_hat, u_t)
        dvuds_dim = self.h_inv * torch.diff(vu)
        dvuds_dim_end = self.h_inv * (vu[-2] + vu[-1])  # the boundary
        drift = torch.cat([dvuds_dim, dvuds_dim_end.reshape(1)])

            
        # discretize ∂(D∂u∂s)/∂s -> 1/h^2*[D_{i+1}(u_i+2 - u_i+1) - D_i(u_i+1 - u_i)]
        duds = h_inv * torch.diff(u_t)  # 299
        D_duds = torch.mul(D_hat[1:], duds) # 299
        diffusion_mid = h_inv * torch.diff(D_duds) # 298 
        
        # diffusion at the boundary
        diffusion_start = h_inv**2 * (D_hat[0] * (u_t[1] - u_t[0])).reshape(1)  
        diffusion_end = h_inv**2 * (-1 * D_hat[-1] * (u_t[-1] - u_t[-2])).reshape(1)  
    
        diffusion = torch.cat([diffusion_start, diffusion_mid, diffusion_end])

        assert growth.shape == drift.shape
        assert growth.shape == diffusion.shape

        # the equation
        dudt = growth - drift + diffusion

        dsdt = torch.zeros_like(s).to(device)

        return (dudt, dsdt)

class pde_params_meshgrid(pde_params_base):
    """
    mlp g,v,D for meshgrid dataset

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
    def __init__(self, channels, n_grid, collapse_D = True, collapse_v = False, g_channels=None, v_channels=None, D_channels=None, time_sensitive=True, lr=3e-4,  ode_tol=1e-4, activation_fn:Union[str, list] = 'Tanh', D_penalty = None):
        super().__init__(channels=channels, collapse_D=collapse_D, collapse_v=collapse_v, g_channels=g_channels, v_channels=v_channels, D_channels=D_channels, time_sensitive=time_sensitive, lr=lr, ode_tol=ode_tol, activation_fn=activation_fn, D_penalty=D_penalty)

        self.n_grid = n_grid
        self.h = 1 / n_grid
        
        
        self.n_dim = channels[0] - 1 if time_sensitive  else channels[0]
        Func_Module = MLP_surrogate
        
        # the output for growth is always 1
        if g_channels  is None:
            g_channels = channels + [1]
        self.g = Func_Module(channels = g_channels, activation_fn=activation_fn)

        # if we choose to collapse v, that means the parameter is the same for all dimension
        if v_channels is None:
            v_channels = channels + [1] if collapse_v else channels + [self.n_dim]
        self.v = Func_Module(channels = v_channels, activation_fn=activation_fn)

        # if we choose to collapse D, that means the parameter is the same for all dimension
        if D_channels is None:
            D_channels = channels + [1] if collapse_D else channels + [self.n_dim]
        self.D = Func_Module(channels = D_channels, activation_fn=activation_fn)

        self.g.time_sensitive = time_sensitive
        self.v.time_sensitive = time_sensitive
        self.D.time_sensitive = time_sensitive


    def mesh_grid_equation(self, s, t, u, h):
        dudt, growth, drift, diffuse = 0
        return dudt, growth, drift, diffuse

    def ode_func(self, t, states): 
        

        u_bt = states[1]
        s_nearby = states[1]
        u_nearby = states[2]

        # TO get some hyper,s
        device = s_nearby.device
        batch_size = s_nearby.shape[0]
        square_size = u_nearby.shape[1]
        n_dim = s_nearby.shape[-1]
        h_inv =  1/self.h

        # the v and D of each dimesion
        drift = 0
        diffusion = 0

        # infer the dynamics params with neural networks

        g_nearby = self.g(s_nearby, torch.broadcast_to(t, u_nearby.shape))
        D_nearby = self.D(s_nearby, torch.broadcast_to(t, u_nearby.shape))
        v_nearby = self.v(s_nearby, torch.broadcast_to(t, u_nearby.shape))


        # assume u_near is a square
        assert g_nearby.shape == u_nearby.shape , "u and g does not have the same shape"

        growth = torch.mul(g_nearby, u_nearby)
        drift = []
        diffusion = []

        for dim in range(n_dim):
            

            v_dim = v_nearby[..., dim]   # slice the last dimension
            assert v_dim.shape == u_nearby.shape , " u and v[:dim] does not have the same shape"
            vu = torch.mul(v_dim, u_nearby)

            # discretize : ∂vu / ∂s_d : ( v_{i+1} u_{i+1} - v_i u_i ) / h
            dvuds_dim = h_inv * torch.diff(vu, dim=dim+1) # batch is the first
            dvu_i_ds = 0.5 * dvuds_dim.sum(dim=dim+1, keepdim=True) #(v_{i+1}u_{i+1} - v_{i-1}u_{i-1})/ 2h
            
            drift_dim = torch.cat(
                [dvuds_dim.index_select( dim+1, torch.tensor([0]).to(device)), 
                dvu_i_ds,
                dvuds_dim.index_select( dim+1, torch.tensor([square_size-2]).to(device)), 
                ] , dim=dim+1
                )
            
            drift.append(drift_dim)

            
            #  ∂2u / ∂s_d ∂s_d  ( u_i + u_{i-1}}) - (u_{i+1} + u_i)
            assert D_nearby.shape == u_nearby.shape , "u and D does not have the same shape"
            u_ss =  h_inv**2 * torch.diff(u_nearby, n=2, dim=dim+1)
            diffusion_dim = torch.mul(D_nearby, torch.broadcast_to(u_ss, D_nearby.shape))

            diffusion.append(diffusion_dim)

            # # for the sample in the center of a square / cube

            # indices = torch.arange(0, square_size-1)
            # indices_next = torch.arange(1, square_size)
            # # to get ( u_i + u_{i-1}}), ((u_{i+1} + u_i)) in batch
            # u_i = u_nearby.index_select( dim+1, indices) + u_nearby.index_select( dim+1, indices_next)  # [b,2,3] fro dim 0
            # vu_i = torch.mul(v_dim.index_select( dim+1, indices), u_i)
            # dvu_i_ds = 0.5 * h_inv * torch.diff(vu_i) #

            # #  1/2h * [ v_{i-1}( u_i + u_{i-1}}) - v_i(u_{i+1} + u_i) ] 

            # drift_i = 0.5 * h_inv * ( self.mul(v_im1, u_im1+u_i) - self.mul(v_i, u_i + u_ip1) )
            # drift_ip1 = h_inv * ( self.mul(v_i, u_i) - self.mul(v_ip1, u_ip1) )    #  ( v_{i+1} u_{i+1} - v_i u_i ) / h
            # drift_im1 = h_inv * ( self.mul(v_im1, u_im1) - self.mul(v_i, u_i) )  #  ( v_i u_i - v_{i-1} u_{i-1}}) / h 

        drift = torch.stack(drift).sum(dim=0)
        diffusion = torch.stack(diffusion).sum(dim=0)

        dudt = growth - drift + diffusion

        dsdt = torch.zeros_like(s_nearby)

        posi = int((square_size - 1)/2)
        indices = [np.arange(batch_size).tolist()] + [posi]*n_dim
        duidt = dudt[indices]

        return (duidt, dsdt, dudt)

    def training_step(self, train_batch, index):
        
        # s : batch, n_dimension
        # t_b : batch, n_timepoints, 1
        # u_b : batch, n_timepoints, 1
        (s_bund,s_neighbor), t_b, (u_b, u_neighbor) = train_batch

        # divided by 5 to reduce the integration time

        t_list = t_b[0].detach().clone().flatten() / 5

        device = s_bund.device

        
        # loss 2 : dynamics 
        # init_condition 
        
        log_utp1_loss = 0
        D_norm = 0
        it = 0

        for t0, t1 in zip(t_list[:-1], t_list[1:]):
            t1 = t1.detach().cpu().item()
            t0 = t0.detach().cpu().item()

            step_size = np.around((t1 - t0)/15, decimals=1).item() 
            step_size = step_size if step_size > 0 else 0.05

            step_size = min(step_size, 0.4)

            ut0 = u_b[:,it]
            utp1 = u_b[:,it+1]
            u_neighbor_t0 = u_neighbor[:, it]

            # init condition : ut, s in a square, u in a square
            init_condition = (ut0, s_neighbor, u_neighbor_t0)
            u_int, s, u_neighbor_int = odeint_adjoint(
                        self.ode_func,
                        init_condition,
                        t_list.type(torch.float32).to(device),
                        atol=1e-8,
                        rtol=1e-8,
                        # method='midpoint',
                        method='dopri5',
                        # options = {'step_size': step_size}
                    )

            # boundary u of  the next timepoint
            utp1_loss = self.loss_fn(u_int[-1], utp1)
            u_int = nn.functional.relu(u_int)

            log_utp1_loss += self.loss_fn(torch.log(utp1+1e-10), torch.log(u_int[-1]+1e-10), weight=weight)

            it+=1

            D_norm += self.restrict_D(s_bund, torch.full(utp1.shape, t1).to(device))

        total_loss = log_utp1_loss + self.D_penalty * D_norm


        with torch.no_grad():
            # self.log("residual_loss", Loss_r, on_epoch=True)
            # self.log("boundary_loss", Loss_b, on_epoch=True)
            # self.log("population_loss", Loss_p, on_epoch=True)
            
            # self.log("boundary_loss", ub_loss.item(),  on_epoch=True)
            # self.log("log_boundary_loss",  log_density_loss_t.item(),  on_epoch=True)
            self.log("integrat_loss", utp1_loss.item(),  on_epoch=True)
            self.log("log_integrat_loss", log_utp1_loss.item(), on_epoch=True)
            self.log("total_loss", total_loss, on_epoch=True, prog_bar=True)

        return total_loss


class pde_neighborloss(pde_params_meshgrid):
    def __init__(self, channels,  n_grid, collapse_D = True, collapse_v = False, g_channels=None, v_channels=None, D_channels=None, time_sensitive=True, lr=3e-4, ode_tol=1e-4, activation_fn:Union[str, list] = 'Tanh', D_penalty = None, weight_intensity=None):
        super().__init__(channels, n_grid, collapse_D, collapse_v, g_channels, v_channels, D_channels, time_sensitive, lr, ode_tol, activation_fn, D_penalty)
        self.weight_intensity = 0.5 if weight_intensity is None else weight_intensity

    def ode_func(self,t, states):
                
        s_nearby = states[0]
        u_nearby = states[1]

        # TO get some hyper,s
        device = s_nearby.device
        batch_size = s_nearby.shape[0]
        square_size = u_nearby.shape[1]
        n_dim = s_nearby.shape[-1]
        h_inv =  1/self.h

        # the v and D of each dimesion
        drift = 0
        diffusion = 0

        if not isinstance(t, torch.Tensor):
            t = torch.Tensor([t]).float().to(s_nearby.device)

        # infer the dynamics params with neural networks
        g_nearby = self.g(s_nearby, torch.broadcast_to(t, u_nearby.shape))
        D_nearby = self.D(s_nearby, torch.broadcast_to(t, u_nearby.shape))
        v_nearby = self.v(s_nearby, torch.broadcast_to(t, u_nearby.shape))


        # assume u_near is a square
        assert g_nearby.shape == u_nearby.shape , "u and g does not have the same shape"

        growth = torch.mul(g_nearby, u_nearby)
        drift = []
        diffusion = []

        for dim in range(n_dim):
            

            v_dim = v_nearby[..., dim]   # slice the last dimension
            assert v_dim.shape == u_nearby.shape , " u and v[:dim] does not have the same shape"
            vu = torch.mul(v_dim, u_nearby)

            # discretize : ∂vu / ∂s_d : ( v_{i+1} u_{i+1} - v_i u_i ) / h
            dvuds_dim = h_inv * torch.diff(vu, dim=dim+1) # batch is the first
            dvu_i_ds = 0.5 * dvuds_dim.sum(dim=dim+1, keepdim=True) #(v_{i+1}u_{i+1} - v_{i-1}u_{i-1})/ 2h
            
            drift_dim = torch.cat(
                [dvuds_dim.index_select( dim+1, torch.tensor([0]).to(device)), 
                dvu_i_ds,
                dvuds_dim.index_select( dim+1, torch.tensor([square_size-2]).to(device)), 
                ] , dim=dim+1
                )
            
            drift.append(drift_dim)

            
            #  ∂2u / ∂s_d ∂s_d  ( u_i + u_{i-1}}) - (u_{i+1} + u_i)
            assert D_nearby.shape == u_nearby.shape , "u and D does not have the same shape"
            u_ss =  h_inv**2 * torch.diff(u_nearby, n=2, dim=dim+1)
            diffusion_dim = torch.mul(D_nearby, torch.broadcast_to(u_ss, D_nearby.shape))

            diffusion.append(diffusion_dim)

        drift = torch.stack(drift).sum(dim=0)
        diffusion = torch.stack(diffusion).sum(dim=0)

        dudt = growth - drift + diffusion

        dsdt = torch.zeros_like(s_nearby)

        return dsdt, dudt

    def forward(self, t, states):
        return self.ode_func(t, states)

    def training_step(self, train_batch, index):
        
        # s : batch, n_dimension
        # t_b : batch, n_timepoints, 1
        # u_b : batch, n_timepoints, 1
        (s_bund,s_neighbor), t_b, (u_b, u_neighbor) = train_batch

        # divided by 5 to reduce the integration time

        t_list = t_b[0].detach().clone().flatten() / 5

        device = s_bund.device

        
        # loss 2 : dynamics 
        # init_condition 
        
        it = np.random.choice(range(len(t_list)-1))
        t1 = t_list[it+1].item()
        t0 = t_list[it].item()

        # for t0, t1 in zip(t_list[:-1], t_list[1:]):

        step_size = np.around((t1 - t0)/15, decimals=1).item() 
        step_size = step_size if step_size > 0 else 0.05

        step_size = min(step_size, 0.4)

        ut0 = u_neighbor[:,it]
        utp1 = u_neighbor[:,it+1]
        u_neighbor_t0 = u_neighbor[:, it]

        # init condition : ut, s in a square, u in a square
        init_condition = (s_neighbor, u_neighbor_t0)
        # _, u_neighbor_int = odeint(
        #             self,
        #             init_condition,
        #             torch.tensor([t0,t1]).float().to(device),
        #             atol=1e-3,
        #             rtol=1e-3,
        #             method='dopri5',
        #             adjoint_options={'norm':'seminorm'},
        #             # options = {'step_size': step_size}
        #         )
        options = {
                'method': 'sym12async' , 
                'h': None , 
                't0': t0 , 
                't1': t1 , 
                'rtol': 1e-3 , 
                'atol': 1e-3 , 
                'print_neval': False , 
                'neval_max': 1e5 , 
                't_eval':None , 
                'interpolation_method':'cubic' , 
                'regenerate_graph':False , 
        }
        
        _, u_neighbor_int = odesolve_adjoint_sym12(
                    self,
                    init_condition,
                    options = options
                    # options = {'step_size': step_size}
                
        )

        # boundary u of  the next timepoint

        u_int = nn.functional.relu(u_neighbor_int[-1])
        log_utp1 = torch.log(utp1+1e-10).flatten()

        utp1_loss = nn.functional.mse_loss(u_int, utp1)
        
        with torch.no_grad():
            weight = (torch.clamp(log_utp1, min=-24) + 24)**self.weight_intensity
            weight /= weight.sum()

        log_utp1_loss = self.loss_fn(log_utp1, torch.log(u_int+1e-10).flatten(), weight=weight)
        D_norm = self.restrict_D(s_bund, t_b[:,it].to(device)) + self.restrict_D(s_bund, t_b[:,it+1].to(device))
            

        total_loss = log_utp1_loss + self.D_penalty * D_norm


        with torch.no_grad():

            self.log("integrat_loss", utp1_loss.item(),  on_epoch=True)
            self.log("D_norm", D_norm.item(), on_epoch=True)
            self.log("log_integrat_loss", log_utp1_loss.item(), on_epoch=True)
            self.log("total_loss", total_loss, on_epoch=True, prog_bar=True)

        return total_loss
