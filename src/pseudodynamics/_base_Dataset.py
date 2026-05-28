import torch
import numpy as np
import pandas as pd
from . import functions as myfn
from scipy.stats import gaussian_kde
from torch.utils.data import Dataset, DataLoader, TensorDataset

 
#TODO: complete AnnDataset
class AnnDataset(Dataset):
    
    def __init__(self, AnnData, cellstate_key='cellstate', timepoint_key='timepoint_tx_days', pop_dict=None, n_grid=300, collocation_points=600,  log_transform=False , norm_time=False, resampling_indensity=0.5, resampling_rate=0.5):
        """
        PINN-dynamics Dataset, extract 

        Arguments:
        -----------
        AnnData : annData, the scanpy 
        cellstate_key : str, the obsm key, the lower dimension representation on which we will use to compute density
        timepoint_key : str, the obs key that indicate the experimental time the cells are collected from
        pop_dict : dict, the dictionary we use to pass population statistics including collected timepoint, mean ,variation
        log_transform : bool, default False, whether the population size will be log transformed to reduce the magnitude of the data

        Returns:
        ----------
        minibatch: not defined
        """
        super().__init__()
        self.adata = AnnData
        self.cellstate_key = cellstate_key
        self.timepoint_key = timepoint_key
        self.log_transform = log_transform

        # check cell state
        assert cellstate_key in AnnData.obsm_keys(),  f'cellstate key `{cellstate_key}` not found in adata'
        assert timepoint_key in AnnData.obs_keys(),  f'timepoint key `{timepoint_key}` not found in adata'

        self.adata_tb = self.adata.obs[timepoint_key]          # pd.Series
        self.cellstate = self.adata.obsm[cellstate_key]        # np.values
        self.n_dim = self.cellstate.shape[1] # the dimension of the cell states

        self.resampling_rate = resampling_rate
        self.resampling_indensity = resampling_indensity

        # check poppulation
        if pop_dict is None:
            assert 'pop' in self.adata.uns, "please provide the cell population data"
            self.popD = self.adata.uns['pop'].copy()   # population Dict
        else:
            self.popD = pop_dict

        # if log_transform:
        #     mu = np.array(self.popD['mean'])
        #     self.popD['mean'] = np.log(mu)
        #     self.popD['var'] = self.popD['var']/ mu
        # else:
        # Backfill 'var' from 'std' when only 'std' is provided. The notebook
        # tutorial documents the {t, mean, std, n_lib} schema, so this lets
        # data prepared from the tutorial work without a separate 'var' key.
        if 'var' not in self.popD:
            if 'std' in self.popD:
                self.popD['var'] = np.asarray(self.popD['std']) ** 2
            else:
                raise KeyError(
                    "adata.uns['pop'] must contain 'var' or 'std' "
                    "(var will be inferred as std**2 if absent)."
                )
        N0 = self.popD['mean'][0]
        self.popD['mean'] = self.popD['mean'] / N0
        self.popD['var'] = self.popD['var']/ N0

        
        ###
        # set up params
        ### 
        self.N_coll = collocation_points

        ###
        # create grided cell state
        ###
        n_grid = n_grid
        self.n_grid = n_grid

    def resampling_by_density(self, n_samples, p=None):
        """
        sample meshes by the time-averaged density distribution
        """
        if p is None:

            try:  # detect density distribution 
                self.density_P
            except AttributeError:
                # create the distribution 
                ub_norm = self.u_b.sum(axis=1, keepdims=True)    # (t, n_grid**2)
                self.density_P = self.u_b/ub_norm

            p = np.mean(self.density_P, axis=0) # (n_grid**2, )

        
        # sample idx by their mean density over time

        a = np.arange(0, self.s.shape[0])
        idxs = np.random.choice(a, size=n_samples, p=p)
        return idxs
    


class OneBranch_AnnDS(AnnDataset):
    def __init__(self, *, n_repeat=10, **kwargs):
        """
        One branch system using one Dimensional cell state like pseudotime
        """
        super().__init__(**kwargs)
        self.n_repeat = n_repeat


class MeshGrid(Dataset):
    def __init__(self):
        super().__init__()
        self.s = None
        self.meshs = None
        self.n_grid = None
        self.nearby_cellstate =None
        self.u_b = None
        self.mesh_ub = None
        self.replicate_key = None

    def __len__(self):
        # repeat sampling for 10 times
        return self.s.shape[0] - self.nearby_cellstate

    def compute_grid_density(self):

        ub_ls = []
        tb_ls = []
        var_ls = []
        hist_var_ls = []
        area_var_ls = []
        
        for tb_idx, t_b in enumerate(self.popD['t']):
            
            # subset ad_t
            
            obs_t = self.adata.obs.query(f"`{self.timepoint_key}` == @t_b")
            rep_at_t = obs_t[self.replicate_key].unique()
            
            u_t = []
            for rep in rep_at_t:
                cb_t = obs_t.query(f"`{self.replicate_key}` == @rep").index
                ad_t = self.adata[cb_t].copy()
                cellstate_t = ad_t.obsm[self.cellstate_key]
                
                # assess density and return 
                density_fun = gaussian_kde(cellstate_t.T)
                u  = density_fun(self.grid_cellstate)
                n_exp = self.popD['n_lib'][tb_idx]
                u_t.append(u * self.h_inv )
            
            u_t = np.vstack(u_t)
            area_t = u_t.cumsum(axis=1)
                    
            ub_ls.append(u_t.mean(axis=0) * self.popD['mean'][tb_idx])
            hist_var_ls.append(u_t.var(axis=0)/n_exp)
            area_var_ls.append(area_t.var(axis=0)/n_exp)
            tb_ls.append(np.full_like(u, self.T_b[tb_idx])) # add norm t
            var_ls.append(self.popD['var'][tb_idx]**2 /n_exp)
        
        return ub_ls, hist_var_ls, area_var_ls, tb_ls, var_ls


    def resampling_by_density(self, n_samples, p=None):
        """
        sample meshes by the time-averaged density distribution
        """
        if p is None:

            try:  # detect density distribution 
                self.density_P
            except AttributeError:
                # create the distribution 
                ub_norm = self.u_b.sum(axis=1, keepdims=True)    # (t, n_grid**2)
                self.density_P = self.u_b/ub_norm

            p = np.mean(self.density_P, axis=0) # (n_grid**2, )

        
        # sample idx by their mean density over time

        a = np.arange(0, self.s.shape[0])
        idxs = np.random.choice(a, size=n_samples, p=p)
        return idxs
    
    def indexing_mesh(self, i):
        """
        given a sample index i in the flatten s, return the location in mesh-grid S
        """
        ix = i%self.n_grid
        iy = i//self.n_grid
        return ix, iy

    def indexing_flatten(self, ix, iy):
        """
        given a sample index i in the flatten s, return the location in mesh-grid S
        """
        return int(ix + iy*self.n_grid)

    def indexing_neighbormesh(self, i, neighborhood=None):
        """
        looking for the index of neighbor mesh in a square , given bottom left index 1.
        
        """
        if neighborhood is None:
            neighborhood = self.nearby_cellstate
        
        bound = self.n_grid - self.nearby_cellstate
        
        squares = []
        ix, iy = self.indexing_mesh(i)   # locate square in mesh s 

        ix = ix if ix < bound else bound            # left bound
        iy = iy if iy < bound else bound            # bottom
        ix_end = min(ix+neighborhood, self.n_grid)  # right bound
        iy_end = min(iy+neighborhood, self.n_grid)  # up bound
        

        for ixx in range(ix, ix_end):
            for iyy in range(iy, iy_end):
                squares.append(self.indexing_flatten(ixx, iyy))
        return squares
    
    def indexing_neighbormesh_center(self, i, neighborhood=None):
        """
        looking for the index of neighbor mesh in a square , given the center index i.
        
        """
        ix, iy = self.indexing_mesh(i)
        botten_left_i = self.indexing_flatten(max(ix-1, 0),  max(iy-1,0))

        squares = self.indexing_neighbormesh(botten_left_i, neighborhood)
        
        return squares
        

    def __getitem__(self, i):
        """
        there is no mini-batch, each item returns the full collocation points
        """ 

        square_idx = self.indexing_neighbormesh(i, self.nearby_cellstate)
        # s_range = slice(i, i + self.nearby_cellstate)

        # random collocation points across the s and t domain
        s_col = torch.Tensor(self.N_coll,self.n_dim).uniform_(0.01, 0.99).float()
        t_col = torch.Tensor(self.N_coll,1).uniform_(min(self.T_b), max(self.T_b)).float()
        
        t_b = torch.from_numpy(self.t_b).float()[:, square_idx].unsqueeze(-1)
        s_bund = self.s[square_idx, :].clone().detach().float()

        bc_shape = [t_b.shape[0]] + list(s_bund.shape)  # broadcast to
        s_bund = s_bund.broadcast_to(bc_shape).float()

        #
        u_b = torch.from_numpy(self.u_b[:, square_idx]).float()
        # u_b = np.where(u_b==0, u_b.min(), u_b)
        mean = 0.5*(u_b[:,1:]+u_b[:,:-1]).sum(dim=1, keepdim = True) #/ h_inv   

        # mean = torch.from_numpy(self.pop_mean).float()
        var = torch.from_numpy(self.pop_var).float()

        # for nearby_cellstate == 1
        if self.nearby_cellstate == 1:
            s_bund = s_bund.squeeze(1)
            u_b = u_b.squeeze(1)
        
        return s_col, t_col, s_bund, t_b, u_b, mean, var


class Processed_baseDS(Dataset):
    
    def __init__(self, Data_pt, n_grid=300, collocation_points=600, n_repeat=10, log_transform=True):
        """
        PINN-dynamics Dataset using the pre-extracted data.   
        This dataset returns full cell state (0-1) for each mini-batch
        
        Augment
        --------
        Data_pt : the output file path from script
        """
        super().__init__()
        # load the result
        D = torch.load(Data_pt)
        
        if log_transform:
            mu = np.array(D['pop']['mean'])
            D['pop']['mean'] = np.log(mu)
            D['pop']['var'] = D['pop']['var']/ mu

        ###
        # set up params
        ### 
        self.n_repeat = n_repeat
        self.N_coll = collocation_points

        ###
        # create grided cell state
        ###
        n_grid = n_grid
        self.n_grid = n_grid