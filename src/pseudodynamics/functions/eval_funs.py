import os, sys, re
import numpy as np
import pandas as pd
import scanpy as sc
import torch
import time

from tqdm.auto import tqdm
# from TorchDiffEqPack import odesolve
from torchdiffeq import odeint_adjoint as odeint
from scipy.stats import pearsonr, spearmanr
from scipy.special import kl_div
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from matplotlib.patches import Patch

def savefig(fig, name, path, dpi=200):
    fig.savefig(f'{path}/{name}', transparent=True, bbox_inches='tight', dpi=dpi)

def forward_get_params(pde_model, DataSet, t_ts=None, s_ts=None, timepoint_label=None):
    r"""
    Given the setup dataset, evalute the behavior functions

    Args:
    ----------
        pde_model : nn.Module, sub-class of pdp.models.pde_params_base
        DataSet : sub-class of `pdp.readers.HighdimAnnDS` 
        t_ts : None , time point tensor
        s_ts : None , cell state tensor
    """
    device = pde_model.device

    if timepoint_label is None:
        timepoint_label = DataSet.popD['t']

    # FORWARD 
    u_pred_ls = []
    v_ls = []
    D_ls = []
    g_ls = []
    chunk_size= 500

    s_ts = DataSet.s.float().to(device) if s_ts is None else s_ts

    t_ts = DataSet.t_b.float().to(device) if t_ts is None else t_ts

    with torch.no_grad():
        for i in tqdm(range(0, len(t_ts), chunk_size)):                              
            s_in = s_ts[i:i+chunk_size]
            t_in = t_ts[i:i+chunk_size]

            v_pred = pde_model.v(s_in, t_in)
            g_pred = pde_model.g(s_in, t_in)
            D_pred = pde_model.D(s_in, t_in)
            
            v_ls.append(v_pred.detach().cpu().numpy())
            g_ls.append(g_pred.detach().cpu().numpy())
            D_ls.append(D_pred.detach().cpu().numpy())

            if "u" in dir(pde_model):
                u_pred = torch.exp(pde_model.u(s_in, t_in))
                u_pred_ls.append(u_pred.detach().cpu().numpy())

        
    # # other params
    n_dimension = s_in.shape[1]
    n_timepoint = len(timepoint_label)

    # concate
    g_pred_ay = np.concatenate(g_ls, axis=0).reshape(n_timepoint,-1)
    v_pred_ay = np.concatenate(v_ls, axis=0).reshape(n_timepoint, -1, n_dimension)
    D_pred_ay = np.concatenate(D_ls, axis=0).reshape(n_timepoint,-1)


    return u_pred_ls, g_pred_ay, v_pred_ay, D_pred_ay


def agg_param(adata, param:np.ndarray, groupby_key='cell_type', timepoints=None, timepoint_key='timepoint_tx_days', cellcount_threshold = 10, agg_fn='mean'):
    r"""
    Aggregate dynamic parameters by specific cell state label

    Args:
    ------
        adata (AnnData): Annotated data matrix
        param (str): Parameter name
        groupby_key (str): Cell state label
        timepoints (list): Timepoints to aggregate
        timepoint_key (str): Anndata.obs key in which the time label is stored
        cellcount_threshold (int) : the min number of cells in the cell type to be considered

    Returns:
    ------
        (DataFrame): Aggregated parameters
    """
    if timepoints is None:
        timepoints = adata.uns['pop']['t']
    
    # sanity check
    assert len(timepoints) == param.shape[0], 'timepoints does not match with the given parameter'
    assert adata.shape[0] == param.shape[1], 'cell number does match with the given parameter'

    param_df = pd.DataFrame(param.T, columns=timepoints)
    param_df.index = adata.obs_names

    # copy cell state label
    param_df[groupby_key] = adata.obs[groupby_key].copy()

     # aggregation hre
    agg_param_df = param_df.groupby(groupby_key).agg(agg_fn)

    # count the number of cells per cell type
    ct_df = []
    for it,t in enumerate(timepoints):
        ad_t = adata[adata.obs[timepoint_key] == t]
        ct_df.append( ad_t.obs.groupby(groupby_key).agg({timepoint_key:'count'}) )

    # subsetting an AnnData drops unused categories, so align the count table with the
    # aggregated table (a category with no cell at all counts as below threshold)
    ct_df = pd.concat(ct_df, axis=1).reindex(agg_param_df.index)
    ct_count_thres_binary = ct_df.where(ct_df>=cellcount_threshold, np.nan) / ct_df.values

    assert ct_count_thres_binary.shape == agg_param_df.shape
    agg_param_df = agg_param_df * ct_count_thres_binary.loc[agg_param_df.index].values


    return agg_param_df.dropna(axis=0, how='all') # remove all nan index

@torch.no_grad
def continuous_params(pde_model, DataSet, 
                        param = 'g',
                        n_interval = 10,
                        groupby_key = None,
                        agg_fun = 'mean',
                        chunk_size = 1000,
                        device = 'cpu'
                        ):
    r"""
    Predict the continuous change of dynamic parameters.
    The cellstate observed at the last timepoint is used.

    Args:
    ----------
        pde_model : nn.Module, sub-class of pdp.models.pde_params_base
        DataSet : sub-class of `pdp.readers.HighdimAnnDS` 
        param : str, one of ['g', 'v', 'D', 'u']
        n_interval : number of intermediate point between two timepoints
        groupby_key : str, aggregate the predicted param according to cell type or cluster, one of the obs_key of the adata
        agg_fun : aggregtion function
        chunk_size : int, minibatch size
        device : str, on which device to compute the parameters i.e. cpu or cuda:0, cuda:1 ...
    """

    param_ls = []

    # <ForLoop1: iterate between timepoint and obsreved cellstate>
    for it, s_ts in tqdm(enumerate(DataSet.cellstate_t_ls[:-1])):
        t = DataSet.T_b[it]
        tp1 = DataSet.T_b[it+1]

        print("from" , t, "to",tp1)

        if groupby_key is not None:
            agg_col = []
            real_time = DataSet.popD['t'][it]
            # obs_t = DataSet.adata.obs.query(f"`{DataSet.timepoint_key}` == @real_time").copy()
            obs_t = DataSet.adata.obs.copy()
            obs_t['str_col'] = 'x'
            ct_count_dict = obs_t.query(f"`{DataSet.timepoint_key}` == @real_time").groupby([groupby_key]).agg({"str_col":'count'}).to_dict()['str_col']
        
        # <ForLoop2:iterate with differrent intemediate t>
        if it+1 == len(DataSet.cellstate_t_ls[:-1]):
            int_timepoint = np.linspace(t, tp1, n_interval+1)
        else:
            int_timepoint = np.linspace(t, tp1, n_interval+1)[:-1]
        for t_mid in int_timepoint:

            s_ts_gpu = torch.from_numpy(DataSet.cellstate).float().to(device)
            t_ts = torch.full((s_ts_gpu.shape[0],), t_mid).float().to(device)

            # <ForLoop3:forward by chunk>
            param_t = []
            for i in range(0, len(t_ts), chunk_size):                              
                s_in = s_ts_gpu[i:i+chunk_size]
                t_in = t_ts[i:i+chunk_size]

                # forward here
                param_pred = pde_model.__getattr__(param)(s_in, t_in) / pde_model.time_scale_factor
                param_t.append(param_pred.detach().cpu().numpy())
            
            param_catchunk = np.concatenate(param_t, axis=0)
            # <ForLoop3>

            if groupby_key is not None:
                obs_t[f"{np.around(t_mid*3, 2)}"] = param_catchunk
                agg_col.append(f"{np.around(t_mid*3, 2)}")
            else:
                param_ls.append(param_catchunk)
            # <ForLoop2>

        if groupby_key is not None:
            params = obs_t[agg_col+[groupby_key]].groupby(groupby_key).agg(agg_fun)
            params = pd.melt(params.reset_index(), id_vars=groupby_key, value_name=param)
            params['timepoint_tx_day'] = real_time
            params['ct_of_day'] = params[groupby_key].map(ct_count_dict)
            param_ls.append(params)
        # <ForLoop1>

    return param_ls

def density_shortterm_simulation(pde_model, DataSet, timepoint_idx=None, time_span=1, timepoints=None, cellstate=None, return_all=False):
    r"""
    simulate density for each cells for any two consecutive timepoints

    Args:
    ----------
        pde_model : nn.Module, sub-class of pdp.models.pde_params_base
        DataSet : sub-class of `pdp.readers.HighdimAnnDS` 
        timepoint_idx : list of index , default the full timepoints defined in DataSet
        time_span : int , how many step of the timepoint index
        return_all : bool, if return the other output
    
    Return:
    ---------
        u_int_all : np.ndarry [n_timepoints, n_cells]
    """

    device = pde_model.device
    model_name = type(pde_model).__name__

    if timepoints is None:
        timepoints = DataSet.popD['t']

    if cellstate is None:
        cellstate = torch.from_numpy(DataSet.cellstate).float()

    if timepoint_idx is None:
        timepoint_idx = np.arange(len(timepoints))
    
    u_b = DataSet.u_b.cpu().numpy().reshape(DataSet.T_b.shape[0], -1)

    if 'duds' in dir(DataSet):
        duds = DataSet.duds.copy()
    else:
        duds = np.zeros((u_b.shape[0], u_b.shape[1], cellstate.shape[1]))
    


    all_output = []
    chunk_size= 1000
    if timepoints[0] == 0:
        # mean minus
        t_list = timepoints
    else:
        t_list = timepoints/ timepoints[0] / pde_model.time_scale_factor

    for it, itp1 in tqdm(zip(timepoint_idx[:-1], timepoint_idx[1:])):

        out_t = []

        print("simulating from timepoint", timepoints[it], "to", timepoints[itp1])

        for i in range(0, len(cellstate), chunk_size):

            s0 = cellstate[i:i+chunk_size].to(device).requires_grad_()
            tompos_u0 = torch.from_numpy(u_b[it, i:i+chunk_size]).float().to(device).requires_grad_()
            duds_0 = torch.from_numpy(duds[it, i:i+chunk_size, :]).float().to(device).requires_grad_()
            
            y_0 = torch.zeros_like(tompos_u0)
            # init_condition = (tompos_u0, s0) if model_name == 'pde_params' else (tompos_u0, s0, duds_0, y_0.clone(), y_0.clone(), y_0.clone())
            init_condition = (tompos_u0, s0, duds_0, y_0.clone(), y_0.clone(), y_0.clone())

            int_out_raw = odeint(
                            pde_model,
                            y0 = init_condition,
                            t = torch.tensor(t_list[it:itp1+1]).type(torch.float32).to(device),
                            atol=pde_model.ode_tol,
                            rtol=pde_model.ode_tol,
                            method='dopri5',
                            adjoint_options={'norm':'seminorm'},
                        )
            int_out = []
            u_t = int_out_raw[0]
            if pde_model.log_transform:
                int_out.append(u_t[1:])
            else:
                int_out.append(torch.nn.functional.relu(u_t[1:]))

            del u_t

            int_out.extend(int_out_raw[1:])

            if len(out_t) == 0:
                out_t = [[] for i in range(len(int_out)) ]

            for i, o in enumerate(int_out):
                # add the i_th output
                out_t[i].append(o.detach().cpu().numpy())

            # torch.cuda.empty_cache()

        # concate at batch - wise

        for i, ith_out in enumerate(out_t):
            conate_axis = 1 if len(ith_out[0].shape) > 0 else 0 # 1 is sample wise if more than 1 evaluation timepoint
            ith_concate = np.concatenate(ith_out, axis=conate_axis)
            out_t[i] = ith_concate

       
        all_output.append(out_t)
            
        # it+=1

    # conate at timepoint-wise
    all_output = [np.concatenate([o[i] for o in all_output], axis=0) for i in range(len(out_t))]

    if return_all:
        return all_output
    else:
        return all_output[0]

def param_vs_score(adata, obs_key, param, timepoints=None,timepoint_key='timepoint_tx_days'):
    if timepoints is None:
        timepoints = adata.obs[timepoint_key].unique()

    assert len(timepoints) == param.shape[0], "the provided timepoints and the given params must the same dimension"
    spr = []
    pr = []
    score = adata.obs[obs_key].values
    for i,t in enumerate(timepoints):
        idx_t = adata.obs[timepoint_key] == t
        spr.append(spearmanr(param[i][idx_t], score[idx_t])[0] )
        pr.append( pearsonr(param[i][idx_t], score[idx_t])[0] )
    
    return np.array(spr), np.array(pr)


def project_params_to_pseudotime(adata, params, param_names='g v2', timepoints=None, pseudotime_key='pseudotime_scaled',nbins=100, return_y=True):
    r"""
    Project the parameters to the pseudotime and aggregate them by bins

    Args
    -------
        adata : AnnData object, the adata with pseudotime and obs
        params : list of np.ndarray, each array is the parameters for one timepoint
        param_names : str, the names of the parameters, e.g. 'g v2'
        timepoints : list of int, the timepoints for the parameters, if None, use the timepoints in adata.uns['pop']['t']
        pseudotime_key : str, the key of the pseudotime in adata.obs, default 'pseudotime_scaled'
        nbins : int, the number of bins to aggregate the pseudotime, default 100
        return_y : bool, if True, return the smoothed y values, otherwise return the figure and axes
    
    Return:
    -------
        fig, axs : matplotlib figure and axes, the figure with the aggregated parameters    
        
        if return_y : y_smooths

    Example
    -------
    >>> nbins =30
    >>> y =  pdp.tl.project_params_to_pseudotime(adata,
                                    params = adata.obs["vnorm_v1"].values.reshape(1,-1), 
                                    param_names = 'v_v1',
                                    timepoints = adata.uns['pop']['t'][[0]],
                                    pseudotime_key = 'pseudotime_scaled', 
                                    nbins = nbins+1, 
                                    return_y = True)

    """
    pdt_max = adata.obs[pseudotime_key].max()
    pdt_bins = np.linspace(0, pdt_max, nbins+1)
    pdt_label = (pdt_bins[1:] + pdt_bins[:-1])/2
    adata.obs['pseudotime_bin'] = pd.cut(adata.obs[pseudotime_key], bins=nbins,
                                                labels = pdt_label)


    timepoints = adata.uns['pop']['t'] if timepoints is None else timepoints 

    if not return_y:
        fig, axs = plt.subplots(1, len(timepoints), figsize=(4*len(timepoints)+1,3), gridspec_kw={"wspace":0.4, 'hspace':0.3})
        axs = axs.flatten() if len(timepoints)>1 else [axs]
    
    y_smooths = []
    for i,t in enumerate(timepoints):
        y_col = 'Day%s_'%t + param_names
        data = adata.obs[['pseudotime_bin']].copy()
        data[y_col] = params[i]
        gdata = data.groupby("pseudotime_bin").agg({y_col:"mean"}).reset_index()

        gdata[y_col] = gdata[y_col].interpolate(method='linear', limit_direction='both')
        gdata[y_col] = gdata[y_col].fillna(method='bfill').fillna(method='ffill')

        x = np.sort(gdata.pseudotime_bin.unique())
        y = gdata[y_col].values

        model2 = np.poly1d(np.polyfit(x, gdata[y_col], 7))

        x_smooth = (x[1:] + x[:-1] )/2
        y_smooth = (y[1:] + y[:-1] )/2
        if not return_y:
            axs[i].scatter(x_smooth, y_smooth, alpha=0.8)
            axs[i].plot(x, model2(x), color='red', alpha=0.4)

        y_smooths.append(y_smooth)

    if return_y:
        return y_smooths
    else:
        return fig, axs

def assign_nearest_cell(input_ay, adata, cellstate_key, n_dimension=None, n_trees=10,  n_neighbors=None, annotation=None, return_model=False, idx=None):
    """
    """
    cellstate = adata.obsm[cellstate_key][:,:n_dimension]
    assert input_ay.shape[1] == cellstate.shape[1], "the dimension of cellstate and the query don't match"
    # define number of neighbors
    if n_neighbors is None:
        try:
            n_neighbors = adata.uns['neighbors']['params']['n_neighbors']
        except:
            n_neighbors = n_trees
    # build index if not given: annoy (approximate, fast) when installed, exact scikit-learn otherwise
    try:
        import annoy
    except ImportError:
        annoy = None
    if idx is None:
        if annoy is not None:
            idx = annoy.AnnoyIndex(cellstate.shape[1], "euclidean")
            [idx.add_item(i, cellstate[i]) for i in range(len(adata))]
            idx.build(n_trees)
        else:
            from sklearn.neighbors import NearestNeighbors
            idx = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean").fit(cellstate)
    # find neighbors
    if annoy is not None and isinstance(idx, annoy.AnnoyIndex):
        nn = np.array(
            [idx.get_nns_by_vector(input_ay[i], n_neighbors) for i in range(len(input_ay))]
        )
    else:
        nn = idx.kneighbors(np.asarray(input_ay), n_neighbors=n_neighbors, return_distance=False)
    nn_return = nn

    # if annotation_key is given, return the annotation of the nearest cells
    if annotation is not None:
        nn_return = pd.DataFrame(adata[nn.flatten()].obs[annotation].to_numpy().reshape(-1, n_neighbors)).T

    if return_model:
        return nn_return, idx
    else:
        return nn_return



def W_distance(u_b, u_simulate, p=2, log_transform=False):
    r"""
    Normalize the density and compute the Wasserstein distance between observation and prediction.
    Log-density is supported, pass log_transform = True

    Args
    -------
        u_b : ndarray, [n_time, n_cell] , observed density
        u_simulate : ndarray, [n_time, n_cell], inferred desity
        p : int, degree of the distance, default W-2 distance
        sanity_check : bool, whether check shape and positivity

    Return 
    -------
        Wasserstein distance : ndarry, [n_time,]
    """
    u_b_local = u_b.copy()
    u_simulate_local = u_simulate.copy()

    # log_transform = True if np.any(u_b_local<0) else log_transform

    distance_ls = []
    for t in range(len(u_b_local)):

        if log_transform:
            u_b_local[t] = np.exp(u_b_local[t])
            u_simulate_local[t] = np.exp(u_simulate_local[t])

        # normalize
        p_b = u_b_local[t] / u_b_local[t].sum()

        if p==1:
            w = np.abs(u_b_local[t] - u_simulate_local[t])
        elif p > 1:
            w = np.power(u_b_local[t] - u_simulate_local[t], p)
            w = w**(1/p)
        distance_ls.append(np.sum(p_b*w))

    return np.array(distance_ls)


def W_log_distance(u_b, u_simulate, p=2, log_transform=False):
    r"""
    Normalize the density and compute the Wasserstein distance between observation and prediction.
    Log-density is supported, pass log_transform = True

    Args
    -------
        u_b : ndarray, [n_time, n_cell] , observed density
        u_simulate : ndarray, [n_time, n_cell], inferred desity
        p : int, degree of the distance, default W-2 distance
        sanity_check : bool, whether check shape and positivity

    Return 
    -------
        Wasserstein distance : ndarry, [n_time,]
    """
    u_b_local = u_b.copy()
    u_simulate_local = u_simulate.copy()

    # log_transform = True if np.any(u_b_local<0) else log_transform

    distance_ls = []
    for t in range(len(u_b_local)):

        if log_transform:
            u_b_local[t] = np.exp(u_b_local[t])
            u_simulate_local[t] = np.exp(u_simulate_local[t])

        # normalize
        p_b = u_b_local[t] / u_b_local[t].sum() + 1e-30
        p_int = u_simulate_local[t] / u_simulate_local[t].sum() + 1e-30

        if p==1:
            w = np.abs(np.log(p_b) - np.log(p_int))
        elif p > 1:
            w = np.power(np.log(p_b) - np.log(p_int), p)
            w = w**(1/p)
        distance_ls.append(np.sum(p_b*w))

    return np.array(distance_ls)

def KLD_density(u_b, u_simulate, sanity_check=True):
    r"""
    Normalize the density and compute the KL-divergence between observation and prediction

    Args
    -------
    u_b : ndarray, [n_time, n_cell] , observed density
    u_simulate : ndarray, [n_time, n_cell], inferred desity
    sanity_check : bool, whether check shape and positivity

    Return 
    -------
    KLD_ls : ndarray, [n_time, ]
    """
    u_b_local = u_b.copy()
    u_simulate_local = u_simulate.copy()

    if sanity_check:
        assert u_b_local.shape == u_simulate_local.shape, "observation and prediction must be the same"
        assert np.all(u_b_local>=0), "density must be positive"

    KLD_ls = []
    for t in range(len(u_b_local)):

        # normalize
        p_b = u_b_local[t] / u_b_local[t].sum()
        p_sim = u_simulate_local[t] / u_simulate_local[t].sum()
        p_b += 1e-34
        p_sim += 1e-34

        # kld
        KLD_ls.append(kl_div(p_b, p_sim).sum())
    KLD_ls= np.array(KLD_ls)

    return KLD_ls



def mmd_laplace(X, Y, gamma=None):
    r"""
    Compute MMD with Laplace kernel between two distributions.
    
    Args
    -------
        X, Y (array-like): Samples from the two distributions (shape: [n_samples, n_features]).
        gamma (float, optional): Bandwidth parameter. If None, uses median heuristic.
        
    Return 
    -------
        float: MMD distance.
    """
    X = np.asarray(X)
    Y = np.asarray(Y)
    
    # Ensure 2D arrays (samples, features)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if Y.ndim == 1:
        Y = Y.reshape(-1, 1)
    assert X.shape[1] == Y.shape[1], "X and Y must have the same number of features."
    
    # Median heuristic for gamma (if not provided)
    if gamma is None:
        Z = np.concatenate([X, Y], axis=0)
        pairwise_dist = np.sum(np.abs(Z[:, None, :] - Z[None, :, :]), axis=-1)
        gamma = 1.0 / np.median(pairwise_dist[np.triu_indices_from(pairwise_dist, k=1)])
        print("heuristic gamma", gamma)

    # Compute kernel matrices
    def laplace_kernel(a, b):
        dist = np.sum(np.abs(a[:, None, :] - b[None, :, :]), axis=-1)
        # dist = np.sum(np.abs(a[:,  :] - b[ :, :]), axis=-1)
        return np.exp(-gamma * dist)
    
    print('computing K<XX>')
    K_XX = laplace_kernel(X, X)
    print('computing K<YY>')
    K_YY = laplace_kernel(Y, Y)
    print('computing K<XY>')
    K_XY = laplace_kernel(X, Y)
    
    # Compute MMD² (unbiased estimator)
    m = X.shape[0]
    n = Y.shape[0]
    
    term_XX = (K_XX.sum() - m) / (m * (m - 1)) if m > 1 else 0.0
    term_YY = (K_YY.sum() - n) / (n * (n - 1)) if n > 1 else 0.0
    term_XY = K_XY.mean()
    
    mmd_squared = term_XX + term_YY - 2 * term_XY
    return np.sqrt(max(mmd_squared, 0.0))  # Ensure non-negative


import sklearn

def _read_process_F_hat(path, pickle=False, filter_nan=True):

    if pickle:
        F_hat = pd.read_pickle(path)[0]
    else:
        F_hat = pd.read_csv(path, index_col=0)
    F_hat_filt = F_hat.copy().drop("Undifferentiated", axis=1)
    F_hat_filt_norm = F_hat_filt.div(F_hat_filt.sum(1), axis=0).fillna(0)
    if filter_nan:
        F_hat_filt_norm = replace_undiff(F_hat_filt_norm)
    # F_hat_filt_norm.index = F_obs.index  # F_hat_filt_norm.index.astype(str)
    return F_hat_filt_norm


def replace_undiff(F_hat_filt_norm):

    # create a temp vector to add undiff label
    undiff = np.zeros(len(F_hat_filt_norm))
    replace_idx = np.where(F_hat_filt_norm.sum(1) == 0)
    #     print(f"{replace_idx[0].shape[0]} non-fates")
    undiff[replace_idx] = 1

    # add to normalized matrix
    F_hat_filt_norm["Undifferentiated"] = undiff

    return F_hat_filt_norm


def compute_accuracy(F_obs, F_hat):
    y_true, y_pred = F_obs.idxmax(1).tolist(), F_hat.idxmax(1).tolist()
    return sklearn.metrics.accuracy_score(y_true, y_pred)