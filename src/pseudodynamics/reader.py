import torch
import numpy as np
import pandas as pd
from . import functions as tl
from functools import partial
from tqdm.contrib.concurrent import process_map
from scipy.stats import gaussian_kde
from torch.utils.data import Dataset, DataLoader, TensorDataset
from ._base_Dataset import AnnDataset, MeshGrid, Processed_baseDS
# from Dataset_base 
 
                                ##################################
                                ## Trajectory Independent  DS   ##
                                ##################################



class HigDim_AnnDS(AnnDataset):
    r"""
    High Dimensional Cell state Dataset for trajectory indepdent modeling

    Args
    --------
    n_repeat :  int 
        the output file path from script
    nearby_cellstate : int
            the number of near (cell state)
    norm_Time : boolen 
        log-normalize the real timepoint 
    AnnData : annData, 
        the single cell object
    cellstate_key : str 
        the obsm key, the lower dimension representation on which we will use to compute density
    timepoint_key : str
        the obs key that indicate the experimental time the cells are collected from
    pop_dict : dict
        the dictionary we use to pass population statistics including collected timepoint, mean ,variation
    log_transform : bool
        default False, whether the population size will be log transformed to reduce the magnitude of the data
    base_cellstate : np.ndarray
        the space to evaluate the density

    """
    def __init__(self, AnnData, cellstate_key='cellstate', timepoint_key='timepoint_tx_days', timepoint_idx=None, n_dimension=5, knn_volume= False, nearby_cellstate=1, norm_time=False, deltax_key=None, density_funs=None, kde_kws={}, base_cellstate=None,  pop_dict=None, n_grid=300, collocation_points=600,  log_transform=False ,resampling_indensity=0.5, resampling_rate=0.5):
        
        super().__init__(AnnData, cellstate_key=cellstate_key, timepoint_key=timepoint_key, pop_dict=pop_dict, n_grid=n_grid, collocation_points=collocation_points,  log_transform=log_transform, norm_time=norm_time, resampling_indensity=resampling_indensity, resampling_rate=resampling_rate)
        
        self.knn_volume = knn_volume
        self.n_dimension = n_dimension
        self.nearby_cellstate = nearby_cellstate
        self.deltax_key = deltax_key
        self.kde_kws = kde_kws

        
        if isinstance(timepoint_idx, list):
            pop_idx = timepoint_idx
        else:
            pop_idx = slice(None, timepoint_idx)
        
        self.pop_idx = pop_idx

        self.popD = {k:v[pop_idx] for k,v in self.popD.items()}
        self.n_timepoint = len(self.popD['t'])

        # subset the adata
        if timepoint_idx is not None:
            t_max = self.popD['t'].max()
            cbs = self.adata.obs.query(f"`{self.timepoint_key}` <= @t_max").index
            self.adata = self.adata[cbs]

        # subset the density_funs
        if density_funs is not None and len(density_funs) != self.n_timepoint:
            density_funs = density_funs[pop_idx]

        # define the delta x for informing v
        if deltax_key is None:
            self.deltax = None
        elif deltax_key in self.adata.obsm_keys():
            self.deltax = self.adata.obsm[deltax_key].copy()[:, :n_dimension]
        elif deltax_key in self.adata.layers:
            self.deltax = self.adata.layers[deltax_key].copy()[:, :n_dimension]
        
        if norm_time == 'log':
            T_b =  np.log(np.where(self.popD['t']==0, 1, self.popD['t']))
            T_b = T_b / T_b.max()
            self.T_b = T_b
        elif norm_time == 'min_minus': 
            self.T_b = self.popD['t'] - min(self.popD['t'])
        elif norm_time == 'none':
            self.T_b = self.popD['t']
        else:
            T_b = self.popD['t']
            T_b = T_b / T_b.min() 
            self.T_b = T_b
        
        ###
        # set up boundary conditions 
        ### 
        self.cellstate_t_ls = []
        for tb_idx, t_b in enumerate(self.popD['t']):
            # subset ad_t
            cb_t = self.adata.obs.query(f"`{self.timepoint_key}` == @t_b").index
            ad_t = self.adata[cb_t].copy()
            cellstate_t = ad_t.obsm[self.cellstate_key][:, :n_dimension]
            self.cellstate_t_ls.append(cellstate_t)

        self.timepoint_mask = [self.adata.obs[self.timepoint_key] == t for t in self.popD['t']]

        ###
        #  IMPORTANT !
        ###
        # use the cell state key of the entire dataset 
        # as it tells what are the possible points of the entire cell state space 
        cellstate = self.adata.obsm[self.cellstate_key][:, :n_dimension] if base_cellstate is None else base_cellstate
        self.cellstate = cellstate
        self.s = torch.from_numpy(cellstate).float()
        self.s = torch.cat([self.s]*len(self.popD['t'])).float()

        
        # observeds
        self.pop_mean = self.popD['mean'] # (tb,)
        # self.T_b = T_b         # (tb,)
        var_ls = [self.popD['std'][i]**2 / self.popD['n_lib'][i] for i in range(len(self.popD['t']))]
        self.pop_var = np.array(var_ls)  # (tb,)
        
        ## compute the densities
        # pay attention to the definition of self.cellstate
        self.compute_density(density_funs)

    def compute_volume(self, dim=2, smooth=True):
        r"""
        Compute the volume of each cell from KNN distances, assume the inner dimension is 2 and the min distance to represent radius
        """
        print("Dataset : Use KNN distances to compute the volume and rescale density")
        
        dist = self.adata.obsp['distances'].copy()
        conn = self.adata.obsp['connectivities'].copy()
        s = self.cellstate

        Vol = []
        for i in range(dist.shape[0]):

            ### From the specified cell state space 
            DM_dist = np.sum((s[conn[i].indices] - s[[i]])**2,axis=1)*0.5
            min_DMdist = DM_dist[np.nonzero(DM_dist)].min()
            max_DMdist = DM_dist.max()

            # Vol.append(np.sqrt(min_DMdist*max_DMdist))
            Vol.append( DM_dist.mean() )

        vol = np.array(Vol)
        thres = np.quantile(Vol, 0.99) # simply to remove outlier
        vol_clip = np.clip(vol, 0, thres)
        
        ## volume smoothing
        if smooth:
            print("Dataset : smoothing KNN derived single-cell volume..")
            knn_idx = [conn[i].indices for i in range(s.shape[0])]
            vol_smooth = [vol_clip[knn_id].mean() for knn_id in knn_idx]
            vol_clip = np.array(vol_smooth)
        
        return vol_clip

    def compute_density(self, density_funs=None):
        r"""
        compute the density for the `self.cellstate`, if density functions not specified then we use the gaussian kde

        Returns
        -------
        self.u_b : Tensor, flatten,  (n_time * n_cell)
        self.t_b : Tensor, flatten,  (n_time * n_cell)
        self.density_funs : list of callable, [n_time]
        self.density_P : ndarray, average the total density into probability summing to 1
        self.s_std : the std of self.cellsate
        """
        ub_ls = []
        u_scale = []

        if density_funs is None:
            print("\n"+"="*20)
            print("Dataset : Computing density :")
            print("\t `density_funs` not specified, default estimator gaussian kde")
            use_gausian = True
            density_funs = []
        else:
            print("\n"+"="*20)
            print("Dataset : Computing density :")
            print(f"using pre-defined density fun `{type(density_funs[0])}`")
            use_gausian = False

        if self.knn_volume:
            self.volume = self.compute_volume()
        else:
            self.volume = np.ones_like(self.adata.shape[0])

        for tb_idx, t_b in enumerate(self.popD['t']):
            
            # subset ad_t
            
            cellstate_t = self.cellstate_t_ls[tb_idx]
            
            # assess density and return 
            if use_gausian:
                density_fun = gaussian_kde(cellstate_t.T, **self.kde_kws)
                density_funs.append(density_fun)

            try:
                u  = density_funs[tb_idx](self.cellstate)   # evaluate with the entire space
            except:
                u  = density_funs[tb_idx](self.cellstate.T)   # evaluate with the entire space
            n_exp = self.popD['n_lib'][tb_idx]

            # u_min = np.min(u[u!=0])
            if self.log_transform:
                threshold = np.quantile(u, q=[1e-3,1-1e-3])
                u = np.clip(u, *threshold) + np.log(self.volume)
                u_sum = np.exp(u).sum()
                scaler = np.log(self.popD['mean'][tb_idx] / u_sum)
                u = u - np.log(u_sum)
                ub_ls.append(u + np.log(self.popD['mean'][tb_idx]))
            else:
                u *= self.volume
                u_sum = u.sum()
                u = np.where(u!=0, u, 1e-10) # replace 0 with 0.1* u_min
                scaler = self.popD['mean'][tb_idx] / u_sum
                u = u / u.sum()
                # u = np.clip(u, a_min=1e-10, a_max=None) 
                ub_ls.append(u*self.popD['mean'][tb_idx])        
            
            u_scale.append(scaler)

        self.u_b = np.vstack(ub_ls)   # (tb, n_cell)
        self.density_funs = density_funs
        self.u_scale = u_scale

        
        # norm_p
        ub_norm = self.u_b.sum(axis=1, keepdims=True)    # (t, n_cell)
        self.density_P = self.u_b/ub_norm               
        scaled_P = self.density_P.flatten() ** self.resampling_indensity
        self.density_P = scaled_P / scaled_P.sum() 
        
        self.u_b = torch.from_numpy(self.u_b.flatten()).float()

        tb_ls = [np.full((self.cellstate.shape[0],), self.T_b[tb_idx]) for tb_idx, t_b in enumerate(self.popD['t'])]
        self.t_b = torch.from_numpy(np.vstack(tb_ls).flatten()).float()

        self.s_std = torch.from_numpy(self.cellstate.std(axis=0)).float()


    def __len__(self):
        return self.s.shape[0] 

    def __getitem__(self, i):
        
        # boundary points
        # resampling rate  is set to 0.5
        if np.random.random() <= self.resampling_rate: 
            i = self.resampling_by_density(1, p=self.density_P).item()
        s_bon = self.s[i]      
        t_bon = self.t_b[i]
        u_bon = self.u_b[i]

        # collocalization point
        err = np.random.randn()
        i_col = np.random.choice(range(len(self.cellstate)))
        s_col = self.s[i] + err * self.s_std
        # s_col = torch.from_numpy(s_col).float()

        t_col = np.random.uniform(self.t_b.min().item(), self.t_b.max().item())
        t_col = torch.tensor([t_col]).float()

        return  s_col, t_col, s_bon.squeeze(), t_bon, u_bon


class TwoTimpepoint_AnnDS(HigDim_AnnDS):
    r"""
    Dataset for high dimensional cellstate
    Each batch returns the cellstates, and their density in two consecutive timepoints
    
    Args
    -----
    AnnData : annData, 
        the single cell dataset 
    cellstate_key : str, 
        the obsm key, the lower dimension representation on which we will use to compute density
    timepoint_key : str, 
        the obs key that indicate the experimental time the cells are collected from
    log_transform : bool, 
        default True, whether the population size will be log transformed to reduce the magnitude of the data
    n_repeat : int
            the output file path from script
    nearby_cellstate : int
        the number of near (cell state)
    norm_Time : bool
        log-normalize the real timepoint 
    split : str, 
        train, val or test
    knn_volume : bool
        whether to use the volume of the knn graph to rescale the density


    Examples
    -----
    >>> import pseudodynamics as pdp
    >>> from pseudodynamics import reader
    >>> config = pdp.ExperimentConfig(config=config_path)
    >>> DS_sub = pdp.reader.TwoTimpepoint_AnnDS(AnnData=adata, split = 'train',
                            **config.dataset_config
                            )
    """
    def __init__(self, AnnData, split='train', cellstate_key='cellstate', timepoint_key='timepoint_tx_days', timepoint_idx=None, n_dimension=5, knn_volume= False,  batchsize=200,  norm_time=False, deltax_key=None, density_funs=None, kde_kws={}, nearby_cellstate=1, base_cellstate=None,  pop_dict=None, n_grid=300, collocation_points=600,  log_transform=False ,resampling_indensity=0.5, resampling_rate=0.5):
        
        super().__init__(AnnData, cellstate_key=cellstate_key, timepoint_key=timepoint_key, timepoint_idx=timepoint_idx, knn_volume=knn_volume, n_dimension=n_dimension, nearby_cellstate=nearby_cellstate, norm_time=norm_time, deltax_key=deltax_key, density_funs=density_funs, kde_kws=kde_kws, base_cellstate=base_cellstate,  pop_dict=pop_dict, n_grid=n_grid, collocation_points=collocation_points,  log_transform=log_transform ,resampling_indensity=resampling_indensity, resampling_rate=resampling_rate)
        self.batchsize = batchsize
        self.u_b = self.u_b.reshape(self.n_timepoint, -1)

        if split in ['train', 'val', 'test']:
            self.split = split
            if 'split' not in self.adata.obs_keys():
                self.random_train_val_test_split()
            
            # subset cells belonging to the given split
            self.subset_dataset()

        elif split is None: 
            self.split = None
            print("Dataset : all cells are used")

        else:
            raise ValueError("the input for kwarg `split` should be ['train', 'val', 'test' , None]")


    def random_train_val_test_split(self):
        r"""
        random train, val, test spliting with the ratio 0.8:0.1:0.1
        """
        
        print("random train, val, test spliting with the ratio 0.8:0.1:0.1")
        print("training label saved to obs under key `split`")

        n_cell = self.adata.shape[0]
        train_val_cells = np.random.choice(self.adata.obs_names, size=int(n_cell*0.9), replace=False)
        val_cells = np.random.choice(train_val_cells, size=int(len(train_val_cells)*0.1), replace=False)

        self.adata.obs['split'] = 'test'
        self.adata.obs.loc[train_val_cells, 'split'] = 'train'
        self.adata.obs.loc[val_cells, 'split'] = 'val'

    
    def subset_dataset(self):
        r"""
        when a valid data split is given, subset the adata, cell state and density
        """
        # self.adata_full = self.adata.copy() save some space ?
        in_split = self.adata.obs["split"].values == self.split
        self.adata = self.adata[in_split].copy()

        # subset
        self.cellstate = self.cellstate[in_split]
        self.deltax = self.deltax[in_split]
        self.u_b = self.u_b[:, in_split]
        self.s = torch.from_numpy(self.cellstate).float()
        self.s = torch.cat([self.s]*len(self.popD['t'])).float()

        # Per-timepoint cell indices (for CFM pairing)
        self._build_timepoint_indices()

    
    def _build_timepoint_indices(self):
        """Build per-timepoint cell index arrays for CFM pairing."""
        tp_key = self.timepoint_key
        self.cells_per_tp = []
        for t_val in self.popD['t']:
            mask = self.adata.obs[tp_key].values == t_val
            indices = np.where(mask)[0]
            self.cells_per_tp.append(indices)

    def __len__(self):
        return int(self.cellstate.shape[0] // self.batchsize)

    def __getitem__(self, i):


        # sample current t
        i_t = np.random.randint(0, self.n_timepoint-1)  # the i^th timepoint index
        i_tp1 = i_t + 1                                 # the index of next timepoint
        self.i_t = i_t                                  # update i_t

        # sample cellstates
        s_index = np.random.choice(np.arange(self.cellstate.shape[0]), size=(self.batchsize,), replace=False)
        s = torch.from_numpy(self.cellstate[s_index]).float()

        if self.deltax is not None:
            deltax = torch.from_numpy(self.deltax[s_index]).float()
        else:
            deltax = None

        # get time
        t = torch.full(size=(self.batchsize,), fill_value=self.T_b[i_t]).float()
        t_p1 = torch.full(size=(self.batchsize,), fill_value=self.T_b[i_tp1]).float()

        # the density of two consecutive
        u_t = self.u_b[i_t, s_index]
        u_tp1 = self.u_b[i_tp1, s_index]  # density of the t plus 1

        batch = {'s':s, 't':t, 'tp1':t_p1, 'ut':u_t, 'utp1':u_tp1, 'deltax':deltax}

        # CFM pairing: sample cells from consecutive timepoints
        if hasattr(self, 'cells_per_tp'):
            idx_t  = self.cells_per_tp[i_t]
            idx_tp1 = self.cells_per_tp[i_tp1]
            n_cfm = min(self.batchsize, len(idx_t), len(idx_tp1))
            x0_idx = np.random.choice(idx_t,  size=n_cfm, replace=len(idx_t)  < n_cfm)
            x1_idx = np.random.choice(idx_tp1, size=n_cfm, replace=len(idx_tp1) < n_cfm)
            batch['cfm_x0'] = torch.from_numpy(self.cellstate[x0_idx]).float()
            batch['cfm_x1'] = torch.from_numpy(self.cellstate[x1_idx]).float()

        # True population ratio N_{t+1}/N_t for the 'popmean' growth reference (open-system anchor)
        batch['relmass'] = torch.tensor(
            float(self.pop_mean[i_tp1]) / float(self.pop_mean[i_t]), dtype=torch.float32)

        return batch

class TwoTimpepoint_AnnDS_fastmode(TwoTimpepoint_AnnDS):
    def __init__(self, *args, pseudobulk_key='pseudo_bulk', resolution=None, n_pseudobulk=None, **kwargs):
        r"""
        Dataset for high dimensional cellstate
        Each batch returns the cellstates, and their density in two consecutive timepoints
        
        Args
        --------
        AnnData : annData, the scanpy 
        cellstate_key : str, the obsm key, the lower dimension representation on which we will use to compute density
        timepoint_key : str, the obs key that indicate the experimental time the cells are collected from
        pop_dict : dict, the dictionary we use to pass population statistics including collected timepoint, mean ,variation
        log_transform : bool, default True, whether the population size will be log transformed to reduce the magnitude of the data
        n_repeat : the output file path from script
        nearby_cellstate : the number of near (cell state)
        norm_Time : log-normalize the real timepoint 

        Fast Model Args
        -------
        n_pseudobulk : int, defult None -> adata.shape[0] / 20, the number of pseudo bulk to end with
        pseudobulk_key : str, default 'pseudo_bulk'
        resolution : int, default None -> 40, the resolution pass to leiden algorithmlao
        """
        super().__init__(*args,**kwargs)
        
        
        print("Using FAST mode")
        self.pseudobulk_key = pseudobulk_key
        self.resolution = resolution
        self.n_pseudobulk = n_pseudobulk
        self.pseudobulk_aggregation()

         ## compute the densities
        # pay attention to the definition of self.cellstate
        self.compute_density(self.density_funs)
        self.u_b = self.u_b.reshape(self.n_timepoint, -1)



    def pseudobulk_aggregation(self):

        # copy the original 
        self.cellstate_origin = self.cellstate.copy() # the original cellstate
        self.u_b_origin = self.u_b.clone()

        print("="*20)
        print("Definining cell-state space")
        print("Generating pseudobulk to represent cell-state")
        pseudobulk_key = self.pseudobulk_key

        if pseudobulk_key in self.adata.obs_keys():
            seed = None
        else:
            seed = np.random.randint(0,199)

        adata = tl.super_resolution_pseudobulk(self.adata, resolution=self.resolution, n_pseudobulk=self.n_pseudobulk, key_added=pseudobulk_key, seed=seed) # leiden clustering
        self.adata.uns[f'{pseudobulk_key}_settings'] = adata.uns[f'{pseudobulk_key}_settings']
        X_df = pd.DataFrame(adata.obsm[self.cellstate_key][:,:self.n_dimension], 
                            columns=['DM_%s'%i for i in range(self.n_dimension)])
        X_df[pseudobulk_key] = pd.Series(adata.obs[pseudobulk_key].values, dtype='str')
        pseudobulk_ay = X_df.groupby(pseudobulk_key).agg("mean")
        self.cellstate = pseudobulk_ay.values
        self.s = torch.from_numpy(self.cellstate).float()
        self.s = torch.cat([self.s]*len(self.popD['t'])).float()

       
    def __getitem__(self, i):
        data_dict = super().__getitem__(i)
        # {'s':s, 't':t, 'tp1':t_p1, 'ut':u_t, 'utp1':u_tp1, 'deltax':deltax}

        # fit delta x on the original cell state
        s_index = np.random.choice(np.arange(self.cellstate_origin.shape[0]), size=(self.batchsize,), replace=False)
        s_origin = torch.from_numpy(self.cellstate_origin[s_index]).float()
        
        u_t_origin = self.u_b_origin[self.i_t, s_index]
        u_tp1_origin = self.u_b_origin[self.i_t+1, s_index]

        if self.deltax is not None:
            # delta x is not aggregated
            deltax = torch.from_numpy(self.deltax[s_index]).float()

        data_dict.update({
            's_origin':s_origin, 'u_t_origin':u_t_origin, 'u_tp1_origin':u_tp1_origin,
            'deltax':deltax,
        })

        return  data_dict


    def get_pseudobulk_vector(self, agg_ad, x_key, pseudobulk_key):
        """
        aggregate multi-dimensional vector based on cell cluster label
        """

        # convert obsm to dataframe
        X_df = pd.DataFrame(agg_ad.obsm[x_key][:,:self.n_dimension], 
                            columns=['DM_%s'%i for i in range(self.n_dimension)])
        X_df[pseudobulk_key] = pd.Series(agg_ad.obs[pseudobulk_key].values, dtype='str')
        
        return X_df.groupby(pseudobulk_key).agg("mean").values # average by label




class Duds_AnnDS(TwoTimpepoint_AnnDS):
    def __init__(self, *args, precomputed_duds=None, **kwargs):
        r"""
        Mannually distrizitize the duds with pre-sampled ∆x as base
        Each batch returns the cellstates, and their density in two consecutive timepoints and density changes
        
        Args
        --------
        n_repeat : the output file path from script
        nearby_cellstate : the number of near (cell state)
        norm_Time : log-normalize the real timepoint 

        Other params from AnnDataset:
        --------
        AnnData : annData, the scanpy 
        cellstate_key : str, the obsm key, the lower dimension representation on which we will use to compute density
        timepoint_key : str, the obs key that indicate the experimental time the cells are collected from
        pop_dict : dict, the dictionary we use to pass population statistics including collected timepoint, mean ,variation
        log_transform : bool, default True, whether the population size will be log transformed to reduce the magnitude of the data
        """
        super().__init__(*args,**kwargs)
        self.delta_s = self.s_std * 0.01

        if precomputed_duds is None:
            self.compute_duds()
        else:
            self.duds = precomputed_duds
        
        if self.duds.shape[0] != self.T_b.shape[0]:
            self.duds = self.duds[self.pop_idx]

    def compute_duds(self):
        # computing duds
        dudcs_ls = []
        for i_t,t in enumerate(self.T_b):

            den_fn = self.density_funs[i_t]
            u_tb = self.u_b[i_t].numpy()

            cellstate = self.cellstate.copy()
            delta_s = self.delta_s.numpy().copy()
            scaler = self.u_scale[i_t]
            
            # important step : evaluate the density change of perturbing each dimension
            # wrap into a partial function for multi-process
            # iter_fn = partial(tl.evaluate_u_ds, cellstate=cellstate, u_tb=u_tb, delta_s=delta_s, den_fn=den_fn, scaler=scaler)
            # du_dcs_t = process_map(iter_fn, range(cellstate.shape[0]), max_workers=5, chunksize=1000) # (n_cell, n_dim)
            if isinstance(den_fn, partial):
                func = den_fn.func
                kwargs = den_fn.keywords
                if 'gradient' in dir(func):
                    du_dcs_t = func.gradient(cellstate, **kwargs)
            elif 'gradient' in dir(den_fn):
                du_dcs_t = den_fn.gradient(cellstate)
            else:
                du_dcs_t = []
                for dim in range(cellstate.shape[1]):
                    dcs = np.zeros_like(cellstate)
                    dcs[:,dim] = np.full((dcs.shape[0],), self.delta_s[dim])
                    s_prime = cellstate + dcs

                    dudcs_dim = (den_fn(s_prime.T) - den_fn(cellstate.T)) / self.delta_s[dim]
                    du_dcs_t.append(dudcs_dim)
                du_dcs_t = np.stack(du_dcs_t).T

            if self.log_transform:
                du_dcs_t += scaler
            else:
                du_dcs_t *= scaler
            dudcs_ls.append( du_dcs_t )

        self.duds = np.stack(dudcs_ls) # (n_time, n_cell, n_dim)
        assert not np.isinf(self.duds).any(), "infinit duds"
        assert not np.isnan(self.duds).any(), "Nan in duds"

    
    def __getitem__(self, i):
        # sample current t
        i_t = np.random.randint(0, self.n_timepoint-1)  # the i^th timepoint index
        i_tp1 = i_t + 1                                 # the index of next timepoint
        
        # density fun of time it
        den_fn = self.density_funs[i_t]

        # sample cellstates
        s_index = np.random.choice(np.arange(self.cellstate.shape[0]), size=(self.batchsize,), replace=False)
        s_ay = self.cellstate[s_index]
        s = torch.from_numpy(s_ay).float()

        if self.deltax is not None:
            deltax = torch.from_numpy(self.deltax[s_index]).float()
        else:
            raise ValueError("To use Duds AnnDS, 'deltax_key' can not be None")

        # get time
        t = torch.full(size=(self.batchsize,), fill_value=self.T_b[i_t]).float()
        t_p1 = torch.full(size=(self.batchsize,), fill_value=self.T_b[i_tp1]).float()
        

        # the density of two consecutive 
        u_t = self.u_b[i_t, s_index]
        u_tp1 = self.u_b[i_tp1, s_index]  # density of the t plus 1

        duds = torch.from_numpy(self.duds[[i_t, i_tp1]][:, s_index]).float()
        duds = torch.transpose(duds, 0, 1)
        return  s, t, t_p1, u_t, u_tp1, deltax, duds


class Duds_AnnDS_fastmode(Duds_AnnDS):
    def __init__(self, *args, n_pseudobulk=None, pseudobulk_key='pseudo_bulk', resolution=None, **kwargs):
        r"""
        FAST MODE : Mannually distrizitize the duds with pre-sampled ∆x as base
        Each batch returns the cellstates, and their density in two consecutive timepoints and density changes
        
        Args
        --------
        n_repeat : the output file path from script
        nearby_cellstate : the number of near (cell state)
        norm_Time : log-normalize the real timepoint 

        Other params from AnnDataset:
        --------
        AnnData : annData, the scanpy 
        cellstate_key : str, the obsm key, the lower dimension representation on which we will use to compute density
        timepoint_key : str, the obs key that indicate the experimental time the cells are collected from
        pop_dict : dict, the dictionary we use to pass population statistics including collected timepoint, mean ,variation
        
        Fast Model Args
        -------
        n_pseudobulk : int, defult None -> adata.shape[0] / 20, the number of pseudo bulk to end with
        pseudobulk_key : str, default 'pseudo_bulk'
        resolution : int, default None -> 40, the resolution pass to leiden algorithm
        """
        super().__init__(*args,**kwargs)

        print("Using FAST mode")
        print("="*20)
        print("Definining cell-state space")
        print("Generating pseudobulk to represent cell-state")

        adata = tl.super_resolution_pseudobulk(self.adata, resolution=resolution, n_pseudobulk=n_pseudobulk, key_added=pseudobulk_key) # leiden clustering
        self.adata.uns[f'{pseudobulk_key}_settings'] = adata.uns[f'{pseudobulk_key}_settings']

        X_df = pd.DataFrame(adata.obsm[self.cellstate_key][:,:self.n_dimension], 
                            columns=['DM_%s'%i for i in range(self.n_dimension)])
        X_df[pseudobulk_key] = pd.Series(adata.obs[pseudobulk_key].values, dtype='str')

        pdb_cellstate = X_df.groupby(pseudobulk_key).agg("mean").values
        self.cellstate = pdb_cellstate
        self.s = torch.from_numpy(self.cellstate).float()
        self.s = torch.cat([self.s]*len(self.popD['t'])).float()

        ## compute the densities
        # pay attention to the definition of self.cellstate
        self.compute_density(self.density_funs)
        self.u_b = self.u_b.reshape(self.n_timepoint, -1)

        self.delta_s = self.s_std * 0.01

        if (self.duds is not None) and (self.duds.shape[0] == self.cellstate.shape[0]):
            pass  # pre-specify duds is correct
        else:
            self.compute_duds()
        
        if self.duds.shape[0] != self.T_b.shape[0]:
            self.duds = self.duds[self.pop_idx]

class Syn_DS(Dataset):

    def __init__(self, cellstate, density, integrate_time, deltax=None, batchsize=200):
        r"""
        Synthsized data. The data is not time sensitive.
        Each time we will sample a batch of cells of the same timepoint

        Args
        --------
        cellstate : Tensor, [n_cell, n_dimension], the cellstate space
        density : Tensor, [n_time, n_cell] the cellular density
        integrate_time : Tensor, [n_time, ] the timepoints that are observed
        deltax : Tensor, [n_time,] the different of s sampled
        batchsize : int


        Returns
        -------
        s, t, t_p1, u_t, u_tp1, deltax
        """
        super().__init__()

        self.cellstate = cellstate
        self.u_b = density
        self.deltax = torch.zeros_like(cellstate) if deltax is None else deltax
        self.T_b = integrate_time

        self.batchsize = batchsize
        self.n_time = len(integrate_time)

    def __len__(self):
        return self.cellstate.shape[0] * self.n_time //self.batchsize

    def __getitem__(self, i):

        i_t = np.random.randint(0, self.n_time-1)  # the i^th timepoint index
        i_tp1 = i_t + 1 
        # sample cellstates
        s_index = np.random.choice(np.arange(self.cellstate.shape[0]), size=(self.batchsize,), replace=False)

        s = self.cellstate[s_index].float()

        t = torch.full(size=(self.batchsize,), fill_value=self.T_b[i_t]).float()
        t_p1 = torch.full(size=(self.batchsize,), fill_value=self.T_b[i_tp1]).float()

        # the density of two consecutive 
        u_t = self.u_b[i_t, s_index]
        u_tp1 = self.u_b[i_tp1, s_index]  # density of the t plus 1

        deltax = self.deltax[s_index].float()

        return s, t, t_p1, u_t, u_tp1, deltax

                                ################################
                                ## Trajectory Dependent  DS   ##
                                ################################
                                
class SingleBranch_AnnDS(AnnDataset, MeshGrid):
    def __init__(self, *,n_timepoint=None, n_repeat=10, nearby_cellstate=10, max_timespan = 3, replicate_key = 'batch', **kwargs):
        """
        Single branch system using pseudotime grid to span the all cell state space

        Args
        --------
        n_repeat : the output file path from script
        nearby_cellstate : the number of near (cell state)
        norm_Time : log-normalize the real timepoint 
        """
        super().__init__(**kwargs)
        self.n_repeat = n_repeat
        self.nearby_cellstate = nearby_cellstate
        self.h_inv = 1/self.n_grid
        self.replicate_key = replicate_key

        self.popD['t'] = self.popD['t'][:n_timepoint]
        self.n_timepoint = len(self.popD['t'])

        coords = np.linspace(0.01, 0.99, self.n_grid) # generate 1D uniform coord
        self.s = coords
        self.grid_cellstate = coords

        # density
        self.cellstate = self.cellstate.flatten()
        ub_ls, hist_var_ls,area_var_ls, tb_ls, var_ls = self.compute_grid_density() # this is from MeshGrid

        self.u_b = np.vstack(ub_ls) + 1e-30  # (tb, n_grid)
        # self.mesh_ub = self.u_b.reshape(-1, self.n_grid,self.n_grid) # (tb, n_grid, n_grid)
        self.hist_var = np.vstack(hist_var_ls)
        self.area_var = np.vstack(area_var_ls)
        self.t_b = np.vstack(tb_ls)
        
        # observeds
        self.pop_mean = self.popD['mean'] # (tb,)
        self.pop_var = np.array(var_ls)  # (tb,)

        # timepoint pair:
        self.timpoint_pairs_index = []
        
        if max_timespan is None:
            max_timespan = self.n_timepoint
        else:
            if max_timespan<=0:
                raise ValueError("max time span must larger than 0")
            max_timespan = max_timespan + 1

        for span in range(1,max_timespan):  # [1, N_t -1]
            for start in range(self.n_timepoint-span):
                self.timpoint_pairs_index.append( [start, start+span] )


    def __len__(self):
        return len(self.timpoint_pairs_index)

    def __getitem__(self, i):
        # no resampling

        indexs = torch.Tensor(self.timpoint_pairs_index[i]).long()

        s = torch.from_numpy(self.s).clone().float()
        t_b = torch.from_numpy(self.T_b).float()
        u_b = torch.from_numpy(self.u_b).float()
        hist_var = torch.from_numpy(self.hist_var).float()
        area_var = torch.from_numpy(self.area_var).float()

        mean = torch.from_numpy(self.pop_mean).float()
        var = torch.from_numpy(self.pop_var).float()
        return s, t_b, u_b, hist_var, area_var, mean, var, indexs

class MeshGrid_AnnDS(AnnDataset, MeshGrid):
    def __init__(self, *,n_timepoint=None, n_repeat=10, nearby_cellstate=10, norm_time=True, replicate_key='batch', **kwargs):
        """
        Two branch system using mesh grid to span the all cell state space

        Args
        --------
        n_repeat : the output file path from script
        nearby_cellstate : the number of near (cell state)
        norm_Time : log-normalize the real timepoint 
        """
        super().__init__(**kwargs)
        self.n_repeat = n_repeat
        self.nearby_cellstate = nearby_cellstate
        self.h = 1/self.n_grid
        self.replicate_key = replicate_key

        self.n_timepoint = n_timepoint
        self.popD['t'] = self.popD['t'][:n_timepoint]

        # subset the adata
        if n_timepoint is not None:
            t_max = self.popD['t'].max()
            cbs = self.adata.obs.query(f"`{self.timepoint_key}` <= @t_max").index
            self.adata = self.adata[cbs]

        ###
        # create grided cell state
        ###
        coords = [np.linspace(0.01, 0.99, self.n_grid) for i in range(self.n_dim)]  # generate 1D uniform coord
        meshgrid_flat = np.vstack([ay.flatten() for ay in  np.meshgrid(*coords)]).T

        self.s = meshgrid_flat
        self.grid_cellstate = meshgrid_flat.T
        # self.meshs = self.s.reshape(self.n_grid,self.n_grid, -1) # from flatten to squared high-dim

        self.h_inv = 1/np.prod([s[1] - s[0] for s in coords])
        
        ###
        # set up boundary conditions
        ### 
        ub_ls, hist_var_ls,area_var_ls, tb_ls, var_ls  = self.compute_grid_density()
        

        self.u_b = np.vstack(ub_ls) + 1e-30  # (tb, n_grid**2)
        # self.mesh_ub = self.u_b.reshape(-1, self.n_grid,self.n_grid) # (tb, n_grid, n_grid)
        self.t_b = np.vstack(tb_ls)

        # norm_p
        ub_norm = self.u_b.sum(axis=1, keepdims=True)    # (t, n_grid**2)
        self.density_P = self.u_b/ub_norm
        
        # observeds
        self.pop_var = np.array(var_ls)  # (tb,)
        self.pop_mean = self.popD['mean'] # (tb,)
        self.T_b = self.popD['t']         # (tb,)
    

class MeshGrid_Resample(MeshGrid_AnnDS):
    def __init__(self, *args, **kwargs):
        """
        the MeshGrid dataset with resampling trick
        """
        super().__init__(*args, **kwargs)

    def __len__(self):
        # repeat sampling for 10 times
        return self.s.shape[0] * (len(self.T_b)  + 1)

    def __getitem__(self, i):
        
        tb_i = i // self.s.shape[0] - 1
        tb_p = self.density_P[tb_i]

        if tb_i >= 0:
            resampled_i = self.resampling_by_density(1,p=self.density_P[tb_i]).item()
        else:
            resampled_i = i

        return super().__getitem__(resampled_i)

class TwoTimepoint_MeshGrid(MeshGrid_Resample):
    def __init__(self, *args, **kwargs):
        """
        the MeshGrid dataset returning the data of two consecutive timepoints
        """
        super().__init__(*args, **kwargs)

    def __len__(self):
        return self.s.shape[0] 

    def __getitem__(self, i):
        
        # sample current t
        i_t = np.random.randint(0, self.n_timepoint-1)  # the i^th timepoint index
        i_tp1 = i_t + 1                                 # the index of next timepoint

        if np.random.random() <= self.resampling_rate: 
            i = self.resampling_by_density(1, p=self.density_P[i_t]).item()
    
            
        # sample cellstates
        s = self.s[i].float()

        # get time
        t = torch.tensor([self.T_b[i_t]]).float()
        t_p1 = torch.tensor([self.T_b[i_tp1]]).float()
        

        # the density of two consecutive 
        u_t = torch.from_numpy(self.u_b[i_t, [i]]).float()
        u_tp1 = torch.from_numpy(self.u_b[i_tp1, [i]]).float()  # density of the t plus 1

        return  s, t, t_p1, u_t, u_tp1

class AllTimepoint_MeshGrid(MeshGrid_Resample):
    def __init__(self, *args, **kwargs):
        """
        the MeshGrid dataset returning the data of two consecutive timepoints
        """
        super().__init__(*args, **kwargs)

    def __len__(self):
        return self.s.shape[0] 

    def __getitem__(self, i):
        

        if np.random.random() <= self.resampling_rate: 
            tb_i = np.random.choice(range(self.density_P.shape[0]))
            i = self.resampling_by_density(1, p=self.density_P[tb_i]).item()

        s_bund = self.s[i,:].copy()
        s_bund = torch.from_numpy(s_bund).float()
        t_b = torch.from_numpy(self.t_b[:, i]).float().unsqueeze(-1)
        u_b = torch.from_numpy(self.u_b[:, i]).float()

        squre_indexes = self.indexing_neighbormesh_center(i, neighborhood=3)

        s_neighbor = self.s[squre_indexes].reshape(3,3,2)
        s_neighbor = torch.from_numpy(s_neighbor).float()
        u_neighbor = torch.from_numpy(self.u_b[:,squre_indexes]).reshape(-1, 3,3).float()
        return (s_bund,s_neighbor), t_b, (u_b, u_neighbor)


class MeshGrid_logDS(MeshGrid_Resample):
    def __init__(self, *args, **kwargs):
        """
        Two branch system using mesh grid to span the all cell state space

        Args
        --------
        n_repeat : the output file path from script
        nearby_cellstate : the number of near (cell state)
        norm_Time : log-normalize the real timepoint 
        """
        super().__init__(*args, **kwargs)
        self.u_b = np.log(self.u_b[:4])

        # self.mesh_ub = self.u_b.reshape(-1, self.n_grid,self.n_grid) # (tb, n_grid, n_grid)
        self.t_b = self.t_b[:4]

        # norm_p
        ub_norm = self.u_b.sum(axis=1, keepdims=True)    # (t, n_grid**2)
        self.density_P = self.u_b/ub_norm
        
        # observeds
        self.pop_var = self.pop_var[:4]  # (tb,)
        self.pop_mean = self.pop_mean[:4] # (tb,)
        self.T_b = self.T_b[:4]         # (tb,)

                                ########################
                                ##     Sim DataSet    ##
                                ########################

class Simple_DS(MeshGrid_AnnDS):
    
    def __init__(self, *, n_timepoint, **kwargs):
        """
        only use the a specific time point
        """
        super().__init__(**kwargs) 

        self.T_b = self.T_b[:n_timepoint]

        self.u_b = torch.from_numpy(self.u_b[:n_timepoint].flatten()).float()
        # self.u_b = torch.log(self.u_b)
        self.t_b = torch.from_numpy(self.t_b[:n_timepoint].flatten().reshape(-1,1)).float()
        self.s = torch.concat([self.s]*len(range(0, n_timepoint)), dim=0).float()
        scaled_P = self.density_P[:n_timepoint].flatten() ** 0.5
        self.density_P = scaled_P / scaled_P.sum() 

        for key in self.popD:
            self.popD[key] = self.popD[key][:n_timepoint]

    def __len__(self):
        return self.u_b.shape[0]

    def __getitem__(self, i):

        i = self.resampling_by_density(1, p=self.density_P)

        return self.s[i], self.t_b[i], self.u_b[i]


                                ########################
                                ##  Processed DataSet ##
                                ########################


class Pdyn_ExtractDataset(Processed_baseDS):
    
    def __init__(self, Data_pt, n_grid=300, collocation_points=600, n_repeat=10, log_transform=True):
        """
        PINN-dynamics Dataset using the pre-extracted data.   
        This dataset returns full cell state (0-1) for each mini-batch
        
        Args
        --------
        Data_pt : the output file path from script

        Returns:
        ----------
        s_col : tensor, collocation cellstate coords
        t_col : tensor, collocation time point
        s_bund : tensor, the boundary cellstate coords
        t_b : tensor, the observed boundary time point
        u_b : tensor, the observed density at each time point, evaluated at the s_bund
        mean : tensor, the mean population size of the cell
        var
        """
        super().__init__(Data_pt, n_grid=n_grid, collocation_points=collocation_points, n_repeat=n_repeat, log_transform=log_transform)
        # load the result
        D = torch.load(Data_pt)

        s = np.linspace(0, 1, n_grid)
        self.s = torch.from_numpy(s)
        h_inv = (1 / (s[1] - s[0]))
        
        ###
        # set up boundary conditions
        ### 
        ub_ls = []
        tb_ls = []
        var_ls = []
        
        for tb_idx, t_b in enumerate(D['pop']['t']):
            
            # quantify the cell densitied at grided s
            u, N, n_exp = tl.boundary_density_at(D, t_b, s)
                    
            ub_ls.append(u / h_inv)
            tb_ls.append(np.full_like(u, t_b))
            var_ls.append(D['pop']['var'][tb_idx] /n_exp)
        
        self.u_b = np.vstack(ub_ls)  # (tb, n_grid)
        self.t_b = np.vstack(tb_ls)
        
        # observeds
        self.pop_var = np.array(var_ls)  # (tb,)
        self.pop_mean = D['pop']['mean'] # (tb,)
        self.T_b = D['pop']['t']         # (tb,)
        
        
    def __len__(self):
        return self.n_repeat
    
        
    def __getitem__(self, i):
        """
        there is no mini-batch, each item returns the full collocation points
        """ 
        
        # random collocation points across the s and t domain
        s_col = torch.Tensor(self.N_coll,1).uniform_(0, 1).float()
        t_col = torch.Tensor(self.N_coll,1).uniform_(min(self.T_b), max(self.T_b)).float()
        
        
        t_b = torch.from_numpy(self.t_b).float()
        s_bund = self.s.clone().detach()
        s_bund = s_bund.broadcast_to(t_b.shape).float()

        #
        u_b = torch.from_numpy(self.u_b).float()
        mean = torch.from_numpy(self.pop_mean).float()
        var = torch.from_numpy(self.pop_var).float()
        
        return s_col, t_col, s_bund, t_b, u_b, mean, var


class Random_ExtractDataset(Pdyn_ExtractDataset):

    def __init__(self, Data_pt, nearby_cellstate=10, n_grid=300, collocation_points=600, n_repeat=10, log_transform=True):
        """
        Based on the pre-extracted dataset, 

        Args
        -----------
        Data_pt :
        nearby_cellstate : the range of cell state in a minibatch

        """
        super().__init__(Data_pt, n_grid=n_grid, collocation_points=collocation_points, n_repeat=n_repeat, log_transform=log_transform)       
        self.nearby_cellstate = nearby_cellstate
    
    def __len__(self):
        # repeat sampling for 10 times
        return self.n_grid - self.nearby_cellstate

    def __getitem__(self, i):
        """
        there is no mini-batch, each item returns the full collocation points
        """ 

        s_range = slice(i, i + self.nearby_cellstate)


        # random collocation points across the s and t domain
        s_col = self.s.clone().detach().float().view(-1,1)[s_range,:]
        t_col = torch.Tensor(self.nearby_cellstate,1).uniform_(min(self.T_b), max(self.T_b)).float()
        
        
        t_b = torch.from_numpy(self.t_b).float()[:, s_range]
        s_bund = self.s.clone().detach().float().view(1,-1)[:, s_range]
        s_bund = s_bund.broadcast_to(t_b.shape).float()

        #
        u_b = torch.from_numpy(self.u_b).float()[:, s_range]
        mean = 0.5*(u_b[:,1:]+u_b[:,:-1]).sum(dim=1, keepdim = True) #/ h_inv   

        # mean = torch.from_numpy(self.pop_mean).float()
        var = torch.from_numpy(self.pop_var).float()
        
        return s_col, t_col, s_bund, t_b, u_b, mean, var



class MeshGrid_DS(Processed_baseDS, MeshGrid):
    
    def __init__(self, Data_pt, nearby_cellstate=10, n_grid=300, collocation_points=600, n_repeat=10, log_transform=True, norm_time=True):
        """
        PINN-dynamics Dataset using the pre-extracted data.   
        This dataset returns full cell state (0-1) for each mini-batch
        
        Args
        --------
        Data_pt : the output file path from script
        nearby_cellstate : the number of near (cell state)
        """
        super().__init__(Data_pt, n_grid=n_grid, collocation_points=collocation_points, n_repeat=n_repeat, log_transform=log_transform)
        # load the result
        D = torch.load(Data_pt)
        self.n_dim = D['ind']['hist'][0].shape[1]

        ###
        # create grided cell state
        ###
        coords = [np.linspace(0.01, 0.99, n_grid) for i in range(self.n_dim)]  # generate 1D uniform coord
        meshgrid = np.vstack([ay.flatten() for ay in  np.meshgrid(*coords)]).T

        self.s = torch.from_numpy(meshgrid).float()
        h_inv = 1/np.prod([s[1] - s[0] for s in coords])

        if norm_time:
            T_b =  np.log(np.where(D['pop']['t']==0, 1, D['pop']['t']))
            T_b = T_b / T_b.max()
        
        ###
        # set up boundary conditions
        ### 
        ub_ls = []
        tb_ls = []
        var_ls = []
        
        for tb_idx, t_b in enumerate(D['pop']['t']):
            
            # quantify the cell densitied at grided s
            u, N, n_exp = tl.boundary_density_at(D, t_b, meshgrid)
                    
            ub_ls.append(u / h_inv)
            tb_ls.append(np.full_like(u, T_b[tb_idx])) # add norm t
            var_ls.append(D['pop']['var'][tb_idx] /n_exp)
        
        self.u_b = np.vstack(ub_ls)  # (tb, n_grid)
        self.t_b = np.vstack(tb_ls)
        
        # observeds
        self.pop_var = np.array(var_ls)  # (tb,)
        self.pop_mean = D['pop']['mean'] # (tb,)
        self.T_b = D['pop']['t']         # (tb,)
        self.T_b = T_b