import os,sys,gc
import numpy as np
import pandas as pd
import torch
from torch import nn
from typing import Any, Union
from typing import Any, Union, Callable
from torchdiffeq import odeint
from torchdiffeq import odeint_adjoint 
# from TorchDiffEqPack import odesolve_adjoint_sym12
# from kan import KAN
import matplotlib.pyplot as plt
import seaborn as sns
from ._pde_informed_params import *
from ..functions import eval_funs as efun
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from ..plotting_fns import density_plot, param_plot

class Density_Transfer(nn.Module):
    r"""
    Wrap up the density transfer function into a nn Module, for calling odeint_adjoint

    Args
    -----
    pde_model: nn.Modules from pdp.models
        the pde model (pde_params, log_pde_params)
    stochastic: bool, default False
        whether to add noise to the cell state simulation
    noise_schedule : callable, default None
        the noise schedule for the cell state
    n_repeat : int
        the number of times to repeat the cell state simulation

    Example
    -----
    >>> pde_model = models.pde_params.load_from_checkpoint(model_ckpt)
    >>> pde_model = pde_model.to(device)
    >>> DT = models.Density_Transfer(pde_model)
    >>> # prepare initial condition
    >>> integrate_time = np.linspace(t/t0, timepoint_tx_days[it+1]/t0 ,n_interval+1) / pde_model.time_scale_factor
    >>> cell_index = ?
    >>> u0 = Dataset.u_b[it, cell_index].float().to(device)
    >>> s0 = torch.from_numpy(Dataset.cellstate[cell_index]).float().to(device)
    >>> # perform simulation
    >>> S_trajectory = DT.cellstate_drift(s0, integrate_time)
    >>> # infer dynamic transport map
    >>> Tmaps_t, Tmaps_t_norm = DT.transition_by_batch(s0, u0, integrate_time, n_interval=n_interval, ncell=ncell)
    """
    def __init__(self, pde_model=None, stochastic=False, noise_schedule=None, n_repeat=30):
        super().__init__()
        self.model = pde_model
        self.relu = nn.ReLU()

        # for cell state drift 
        self.stochastic = stochastic
        self.n_repeat = n_repeat
        
        if noise_schedule is None:
            # the default noise schedule is a constant over time and state
            self.noise_schedule = lambda s, t: 1
        else:
            self.noise_schedule = noise_schedule
    
    def forward(self, t, states):
        return self.model.density_transfer(t, states)

    def velocity(self, t, s):
        device = s.device
        ncell  = s.shape[0]
        t_in = torch.full((s.shape[0],1), t.item()).to(device) *self.model.time_scale_factor

        if self.stochastic:
            noise =  torch.broadcast_to(torch.sqrt(self.model.D(s,t_in)*2).view(ncell,-1), s.shape)
            ds = self.model.v(s, t_in)  +  \
                self.noise_schedule(s, t_in) * noise * torch.randn((ncell,1)).to(device)
        else:
            # the state is deterministic and only decide by v
            ds = self.model.v(s, t_in)


        return ds

    def cellstate_drift(self, s0, integrate_time):
        """
        Integrate the drift field v from the initial cell states s0.

        integrate_time : 1-D array in *scaled* time, i.e. the dataset's normalized time
            (``DataSet.T_b``) divided by ``model.time_scale_factor`` -- the same unit that
            ``transition_by_batch`` and the training ODE use. The velocity function
            multiplies by ``time_scale_factor`` to obtain the network's time input.
        """
        
        # get device
        if isinstance(s0, np.ndarray):
            device = self.model.device
            s0 = torch.from_numpy(s0).float().requires_grad_().to(device)
        else:
            device = s0.device

        # integrate v
        with torch.no_grad():
            try:
                s_out = odeint(self.velocity, y0=s0, 
                        t=torch.tensor(integrate_time).to(device),   # scaled time (T_b / time_scale_factor), as in training and transition_by_batch
                        atol=self.model.ode_tol,
                        rtol=self.model.ode_tol,

                        )
            except AssertionError:  #underflow
                step_size = np.around((integrate_time[-1] - integrate_time[0]) / 100 , 2)
                s_out = odeint(self.velocity, y0=s0, 
                        t=torch.tensor(integrate_time).to(device),   # scaled time (T_b / time_scale_factor), as in training and transition_by_batch
                        atol=self.model.ode_tol,
                        rtol=self.model.ode_tol,
                        method='rk4',
                        options = {"step_size":step_size}
                        )
            
            s_traj = s_out.detach().cpu().numpy()
            del s_out 
        return s_traj
    
    def transition_by_batch(self, s0, u0, integrate_time, n_interval=10, ncell=200):
        """
        perform density transfer given initial cell state and density

        s0: tensor
        """
        # get device
        if isinstance(s0, np.ndarray):
            device = self.model.device
            s0 = torch.from_numpy(s0).float().requires_grad_().to(device)
        else:
            device = s0.device

        
        # step 1. cell state simulation
        s_traj_ay = self.cellstate_drift(s0, integrate_time)

        delta_t = integrate_time[1] - integrate_time[0]

        # step 2. density simulation
        u0_by_time = []
        for ic in range(0, s0.shape[0], ncell*4):
            s0_chunk = s0[ic:ic+ncell*4]
            u0_chunk = u0[ic:ic+ncell*4]

            zeros = torch.zeros_like(u0_chunk)
            duds_init = torch.zeros_like(s0_chunk)

            intout  = odeint_adjoint(
                            self.model,
                            y0 = (u0_chunk, s0_chunk, duds_init, zeros, zeros, zeros),
                            t = torch.tensor(integrate_time).type(torch.float32).to(device),
                            atol=self.model.ode_tol,
                            rtol=self.model.ode_tol,
                            method='dopri5',
                        )
            u0_by_time.append(self.relu(intout[0]).detach().cpu().numpy()) # (time, chunk)
            del intout

        if len(u0_by_time)>1:
            u0_by_time = np.concatenate(u0_by_time, axis=1) # (time, cell)
        else:
            u0_by_time = u0_by_time[0]

        # step 3. density transfer
        Tmaps_t = []
        Tmaps_norm_t = []

        for ic in range(0, s0.shape[0], ncell):
            u0_chunk = u0[ic:ic+ncell]
            leftover_u_global = [ np.zeros((n_interval, u0_chunk.shape[0])) ]
            diff_flow_u_global = []

            # step 3.1
            for offset in range(n_interval):   # the ith time of transition

                s_traj = torch.from_numpy(s_traj_ay[:,ic:ic+ncell,:]).float().to(device)
                
                tn1 = integrate_time[offset] # t_{n-1}
                diff_flow_u_local = [u0_by_time[offset, ic:ic+ncell]]
                leftover_u_local = []

                # step 3.2
                # for i, t in enumerate(integrate_time[1:n_interval-offset+1]):
                for i in range(n_interval-offset):

                    s_source = s_traj[i].requires_grad_().to(device) # s_i
                    s_target = s_traj[i+1].requires_grad_().to(device) # s_{i+1}
                    
                    left_u = torch.from_numpy(leftover_u_global[-1][i]).requires_grad_().to(device) # left from last round of diff trajactory
                    inflow_u = torch.from_numpy(diff_flow_u_local[-1]).requires_grad_().to(device)  # inflow from last step of the same round of diff trajectory
                    
                    # the 
                    u_source = left_u.float() + inflow_u.float()
                    u_target = torch.zeros_like(u_source).to(device)

                    # integrate the flow within small timespan
                    t = tn1 + delta_t
                    s_last, u_stay, s_next, u_flow = odeint_adjoint(self, 
                                    y0= (s_source, u_source, s_target, u_target), 
                                    t = torch.tensor([tn1, t]).float().to(device),
                                    rtol = self.model.ode_tol,
                                    atol = self.model.ode_tol,
                                    )
                    
                    with torch.no_grad():
                        diff_flow_u_local.append( self.relu(u_flow[-1]).detach().to('cpu').numpy() )
                        leftover_u_local.append( self.relu(u_stay[-1]).detach().to('cpu').numpy() )
                        tn1 = t

                    del s_source, s_target, u_source, u_target, left_u, inflow_u, s_last, u_stay, s_next, u_flow
                    # free_memory(to_delete)
                    torch.cuda.empty_cache()
                    self.model.zero_grad()

                ##
                #  the nth offset, closing step 3.2

                diff_flow_u_local = np.stack(diff_flow_u_local)[1:] # length : 100 - offset
                leftover_u_local = np.stack(leftover_u_local)#[1:]   # length : 100 - offset 

                diff_flow_u_global.append(diff_flow_u_local)
                leftover_u_global.append(leftover_u_local)

                gc.collect()
                torch.cuda.empty_cache()

                del diff_flow_u_local, leftover_u_local,  s_traj #, self.model
                
            ##
            #  place the flow and leftover density into a triangle matrix
            #  closing step 3.1
            Tmap_manual = []
            for icell in range(u0_chunk.shape[0]):
                T_M = np.zeros((n_interval,n_interval))
                
                for i,flow_u_t in enumerate(diff_flow_u_global):
                    nt = flow_u_t.shape[0]
                    
                    for j, u in enumerate(flow_u_t[:,icell]):
                        T_M[i+j, j] = u

                Tmap_manual.append(T_M)


            Tmap = np.stack(Tmap_manual)
            Tmaps_t.append(Tmap)

            Tmap_norm = Tmap / u0_chunk.to('cpu').numpy().reshape(-1,1,1)
            Tmaps_norm_t.append(Tmap_norm)

        # closing step 3
        Tmaps_t_all = np.concatenate(Tmaps_t, axis=0)
        Tmaps_norm_t_all = np.concatenate(Tmaps_norm_t, axis=0)
        
        return Tmaps_t_all, Tmaps_norm_t_all


class DT_analysis:
    r"""
    class to analyze the density transport result from saved files

    Args
    ------
    adata : anndata
        the single cell type object with annotaion
    result_dir : Path
        the abs path to the result directory where Density_Transfer results are saved
    celltype : str
        the cell type to analyze


    Example
    ------
    >>> dta_obj = DT_analysis(adata, result_dir, celltype)
    >>> TM = DT.TM_dict[time]
    >>> celltypes = ['Ery','HSC','Int prog','Meg']
    >>> ct_prop = dta_obj.density_by_celltype_step(celltypes)
    >>> agg_density = dta_obj.density_by_celltype_step(celltypes, step=-1, norm=False)
    >>> clusters, row_linkage, reorded_i = DT.hierarchical_clustering(density_matrix=laststep_TM,
                                                    n_clusters=n_clusters, 
                                                    log_normalize=log_normalize)
    >>> g = DT.clustermap(  
                density_matrix = laststep_TM,
                clusters = clusters, 
                row_linkage = row_linkage, 
                celltypes = tick_labels,
                cmap = 'viridis',
                cbar_pos = [1, 0.4, 0.03,0.2],
                figsize = (7,8),
                log_normalize=True
                )
    """
    def __init__(self, adata, result_dir, celltype=None):
        self.adata = adata
        self.result_dir = result_dir
        self.celltype = celltype
        self.load_result(result_dir)

    def load_result(self, result_dir):
        r"""
        load the result from saved files, this method returns 4 dictionary with timepoint as key
        
        Saved Properties:
        -----
        cb_dict: a dictionary of cell barcode at its corresponding time point
        TM_dict: a dictionary of *raw* transport map at its corresponding time point
        TM_norm_dict: a dictionary of *normalized* transport map at its corresponding time point
        trajectory_dict: a dictionary of trajectory at its corresponding time point
        """
        # load cell barcode dict
        
        

        # load density intermediate files
        self.TM_dict = {}                        # shape : [cell, step, step]
        self.TM_norm_dict = {}                   # shape : [cell, step, step]
        self.trajectory_dict = {}                # shape : [step+1, cell, n_dim ]
        self.cb_dict = {}
        ct_prop_ls = []

        # load trajectory and transport map

        for files in os.listdir(self.result_dir):

            if self.celltype is not None:
                assert type(self.celltype) == str
                if not files.startswith(self.celltype):
                    continue

            day = files.split('_')[1].replace('Day', '')

            if files.endswith('cellbarcode.npy'):
                self.cb_dict[day] = np.load(os.path.join(self.result_dir, files), allow_pickle=True)

            elif files.endswith('_Norm_TransportMap.npy'):
                self.TM_norm_dict[day] = np.load(os.path.join(self.result_dir, files))
            
            elif files.endswith('_TransportMap.npy'):
                self.TM_dict[day] = np.load(os.path.join(self.result_dir, files))

            elif files.endswith('sim_trajectory.npy'):
                self.trajectory_dict[day] = np.load(os.path.join(self.result_dir, files))
            
            elif files.endswith('ct_prop.csv'):
                ct_prop = pd.read_csv(os.path.join(self.result_dir, files))
                ct_prop_ls.append(ct_prop)
            else:
                print(f'{files} is not a valid file')
        
        if len(ct_prop_ls) >0:
            print("cell type proportion summary detected")
            print("adding to  ct_prop property")
            self.ct_prop = pd.concat(ct_prop_ls)

        # cb_list = np.concatenate(list(self.cb_dict.values))
        # self.adata = self.adata[cb_list].copy

    def summarize_cell_proportions(self, df, celltype_list):
        r"""
        Summarizes the proportion of cell types mapped from neighbors for each cell

        Inputs
        ------
        df : pd.DataFrame
            Input DataFrame where each column represents a cell and each row a sample.
        celltype_list : list
            List of cell types to include in the output as columns.
        """
        # Compute normalized value counts for each cell
        proportions = df.apply(lambda col: col.value_counts(normalize=True))
        proportions = proportions.fillna(0).T # cell as index

        # Reindex columns to match the given celltype_list, filling missing with 0
        proportions = proportions.reindex(columns=celltype_list, fill_value=0)

        return proportions
    
    def annotate_trajectory(self, cellstate_key, obs_key, copy=False):
        r"""
        Use the nearest celltype to annotate each step along the simulated trajectory

        Inputs:
        cellstate_key: which cellstate space for the simulation to map to
        obs_key: the key to store the annotation in adata

        Returns:
        celltype_trajectory: with the annotation added
        """
        self.celltype_key = obs_key
        self.cellstate_key = cellstate_key
        try:
            # the order matched with that in adata
            celltype_list = self.adata.obs[obs_key].cat.categories
        except:
            celltype_list = self.adata.obs[obs_key].unique()
        

        celltype_trajectory = {}

        for t, traj in self.trajectory_dict.items():

            # cb = self.cb_dict[t]
            print(t,traj.shape)

            ct_df = []
            for step in range(1, traj.shape[0]):

                annotations = efun.assign_nearest_cell(traj[step], self.adata, cellstate_key=cellstate_key, annotation=obs_key)
                proportions = self.summarize_cell_proportions(annotations, celltype_list)

                # assert proportions.shape[0] == len(cb)
                # proportions.index = cb

                proportions['Day'] = t
                proportions['step'] = step
                proportions['cell_index'] = proportions.index
                ct_df.append(proportions)
            
            celltype_trajectory[t] = pd.concat(ct_df).sort_values(['cell_index', 'Day'])
        
        if copy:
            self.celltype_trajectory = celltype_trajectory 
        else:
            self.ct_prop = pd.concat(celltype_trajectory.values())
            return celltype_trajectory
            
    def density_by_celltype_step(self, celltypes, step=-1, norm=True):
        
        if norm:
            DT_dict = self.TM_norm_dict
        else:
            DT_dict = self.TM_dict


        # step index in the df is 1-indexed but the matrix is 0-indexed
        step = sorted(self.ct_prop.step.unique())[step]
        step_index = step - 1

        ct_density_ls = []
        agg_density_ls = []

        for time in DT_dict:
            # e.g : '0-4'
            # density transport matrix at the step
            DT_M = DT_dict[time][:, step_index]
            ct_day_df = self.ct_prop.query("`Day` == @time").copy()

            assert DT_M.flatten().shape[0] == ct_day_df.shape[0], 'The number of cell types and the number of rows in the DT matrix should be equal'

            ct_day_df.loc[:,celltypes] = ct_day_df.loc[:,celltypes]* DT_M.flatten().reshape(-1,1)
            ct_day_df['Day'] = time
            ct_density_ls.append(ct_day_df)

            agg_density = ct_day_df.groupby('cell_index')[celltypes].agg('sum')
            agg_density['Day'] = time
            agg_density.index = self.cb_dict[time]
            agg_density_ls.append(agg_density)

            
        self.ct_density = pd.concat(ct_density_ls, axis=0)
        self.agg_density = pd.concat(agg_density_ls, axis=0)

        self.ct_density['start_day'] = self.ct_density['Day'].apply(lambda x : x.split('-')[0]).astype(int)
        self.ct_density = self.ct_density.sort_values('start_day', ascending=True)
        ncell_day = self.ct_density['Day'].value_counts().to_dict()

        return self.agg_density
    
    def select_data(self, value, step=None):

        if value == 'assignment_probability':
            try:
                df = self.ct_prop.query("`step`==@step")
            except:
                raise ValueError("Please run `get_celltype_prop` first")
            
        elif value == 'density':
            try:
                df = self.ct_density.query("`step`==@step")
            except:
                raise ValueError("Please run `density_by_celltype_step` first")
        
        elif value == 'agg_density':
            try:
                df = self.agg_density
            except:
                raise ValueError("Please run `density_by_celltype_step` first")
        
        else:
            raise ValueError("value must be 'assignment_probability' or 'density'")

        return df

    def heatmap_cell_proportions(self, celltypes, value = 'assignment_probability', step=-1, paletter=None,  log_normalize=False, **kwargs):
        r"""
        heatmap for  cell proportions with 

        Args:
        ------
        celltypes: list of cell types (the columns in self.ct_prop)
        value : 'assignment_probability', 'density' , 'agg_density'
        step : which step of the trajectory to plot
        paletter : color palette for day 
        cmap : matplotlib colormap
        **kwargs: kwargs for seaborn clustermap

        Returns:
        ------
        g : seaborn cluster grid
        """

        # sort dataframe
        step = sorted(self.ct_prop.step.unique())[step]

        # select data source
        df = self.select_data(value, step)
        
        df['start_day'] = df['Day'].apply(lambda x : x.split('-')[0]).astype(int)
        df = df.sort_values('start_day', ascending=True)
        ncell_day = df['Day'].value_counts().to_dict()

        # set up color
        nday = len(df['Day'].unique())
        if paletter is None:
            colors = sns.color_palette('Set2',nday)
            paletter = {day: colors[i] for i, day in enumerate(df['Day'].unique())}


        # MAIN Heatmap PLOTING FUNC
        if log_normalize:
            show_matrix = self.log_transform(df[celltypes].values)
        else:
            show_matrix = df[celltypes].values

        g = sns.clustermap(show_matrix, 
                    row_colors= df['Day'].map(paletter).to_numpy(), 
                    row_cluster=False, col_cluster=False,
                    rasterized=True,
                    **kwargs
                    )
        g.ax_heatmap.set_yticks([])
        g.ax_heatmap.set_xticklabels(celltypes)

        # Get the axis for row colors
        ax = g.ax_row_colors
        g.ax_cbar.set_title(
            ['','log '][log_normalize] + value
        )

        # Add labels next to each row color block
        total_cell = 0
        for i, label in enumerate(list(paletter.keys())[::-1]):
            # Calculate text position (x=1.05 places label to the right of the color block)

            ncell = ncell_day[label]

            ax.text(-0.5, (total_cell + ncell//2) / df.shape[0], label,
                    fontsize=16, ha='right', va='center',
                    transform=ax.transAxes)
            
            total_cell += ncell

        return g

    def log_transform(self, matrix):
        r"""
        log normalize and then scale to positive
        """
        matrix_raw = matrix.copy()
        matrix_log = np.log(matrix+1e-1)
        # transformed_matrix = np.where(matrix_raw == 0, 0, matrix_log-matrix_log.min())
        transformed_matrix = matrix_log-matrix_log.min()
        return transformed_matrix
        
    def hierarchical_clustering(self, density_matrix=None, value=None, celltypes=None, n_clusters=5, cluster_colors=None, log_normalize=False):
        r"""
        This function performs hierarchical clustering on the given density matrix.
        Two input mode, given: 
            1. density_matrix or 
            2. value and celltypes
        Args
        -----
        density_matrix: np.ndarray
        value : 'assignment_probability', 'density' , 'agg_density'
        celltypes: list of cell types (the columns in self.ct_prop)
        n_clusters: int
        cluster_colors: list

        Returns
        -------
        cluster_flat: np.ndarray
        row_linkage: np.ndarray
        reordered_indices: np.ndarray
        """
        from scipy.spatial.distance import pdist
        from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

        if density_matrix is None:
            df = self.select_data(value)
            density_matrix = df[celltypes].values
        elif (density_matrix is None) & (value is None):
            raise ValueError("no input given !!! provide density matrix or value_source and celltypes")
        

        # MAIN Heatmap PLOTING FUNC
        if log_normalize:
            show_matrix = self.log_transform(density_matrix)
        else:
            show_matrix = density_matrix

        row_distances = pdist(show_matrix, metric='minkowski')
        row_linkage = linkage(row_distances, method='ward')
        # Form clusters
        clusters = fcluster(row_linkage, n_clusters, criterion='maxclust')

        # Create a truncated dendrogram to get reordered indices
        dnd = dendrogram(row_linkage, truncate_mode="level", p=3, no_plot=True)
        reordered_indices = dnd['leaves']

        return clusters, row_linkage, reordered_indices
    

    def clustermap(self,
                        clusters, 
                        row_linkage,
                        celltypes,
                        density_matrix=None, 
                        value=None, 
                        log_normalize=False, 
                        cluster_colors=None,    
                        **kwargs
        ):
        r"""
        This function is a wrapper for seaborn.clustermap() that truncates the dendrogram and returns cluster assignments and reordered indices.

        Two input mode, given: 
            1. density_matrix or 
            2. value and celltypes

        """

        if density_matrix is None:
            df = self.select_data(value)
            density_matrix = df[celltypes].values
        elif (density_matrix is None) & (value is None):
            raise ValueError("no input given !!! provide density matrix or value_source and celltypes")
        

        # MAIN Heatmap PLOTING FUNC
        if log_normalize:
            show_matrix = self.log_transform(density_matrix)
        else:
            show_matrix = density_matrix

        from matplotlib.patches import Patch

        num_clusters = len(np.unique(clusters))

        if cluster_colors is None:
            cluster_colors = sns.color_palette("Set3", n_colors=num_clusters)
        row_colors = [cluster_colors[label - 1] for label in clusters]


        g = sns.clustermap(
                    show_matrix,
                    row_linkage=row_linkage,
                    colors_ratio = 0.03,
                    dendrogram_ratio = 0.2,
                    row_colors=row_colors,
                    col_cluster=False,
                    rasterized=True,
                    **kwargs
        )
        g.ax_heatmap.set_yticks([])
        g.ax_heatmap.set_xticklabels(celltypes)

        legend_patches = [
            Patch(color=cluster_colors[i], label=f"Cluster {i + 1}")
            for i in range(num_clusters)
            ]
        g.ax_heatmap.legend(
            handles = legend_patches,
            title = False,
            bbox_to_anchor = (1.05, 0.05, 0.25, 0.25),
            loc = 'lower left',
            borderaxespad = 0 ,
            fontsize=16,
            frameon=False,
        )

        g.ax_cbar.set_title(
            ['', 'log '][log_normalize] + "density"
        )

        return g
    
    def cluster_proportion(self, clusters, cb_list, ax=None, title=None):

        # cb_list = np.concatenate(list(self.cb_dict.values()))
        local_ad = self.adata.copy()
        if 'transport_cluster' in local_ad.obs.columns:
            local_ad.obs.pop('transport_cluster')
        named_cluster_flats = ['cluster%s'%c for c in clusters]
        local_ad.obs.loc[cb_list, 'transport_cluster'] = named_cluster_flats

        sns.set_theme(font_scale=1.3, style='ticks', palette='Set3')
        # with plt.rc_context({"figure.dpi":100, "figure.figsize":(6,4)}):
        fig, ax = density_plot.obs_composition(
                    local_ad[cb_list], 
                    'timepoint_tx_days', 'transport_cluster', 
                    figkws={'figsize':[4,3]},
                    # legend_kws={"handles":[], "frameon":False,  "fontsize":13 ,'ncol':3, "bbox_to_anchor":(0.05, 1.4), "loc":'upper left', },
                    ax=ax
                    )
        ax.set_title(title)
       
        ax.set_xlabel("timepoint")
        ax.set_ylabel("proportion of clusters")

        return fig, ax
    
    def vis_Tmap(self, TM, ax=None):

        trui = np.triu_indices_from(TM, k=1)
        TM[trui] = np.nan

        if ax is None:
            fig, ax = plt.subplots(figsize=(4,4))

        ms = ax.matshow(TM, cmap='viridis' )

        ax.set_yticks([0,2, 5, 9])
        ax.set_yticklabels([r"$t_1$", r"$t_3$", r"$\vdots$", r"$t_n$"])
        
        ax.tick_params(top=False, labeltop=False, bottom=True, labelbottom=True)
        
        ax.set_xticks([0,2, 5, 9])
        ax.set_xticklabels([r"$s_1$", r"$s_3$", r"$...$", r"$s_n$"])

        return ax
