import os
import torch
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
# import decoupler as dc
from tqdm import tqdm
from scipy.stats import gaussian_kde,entropy
# from scipy.integrate import trapz
from .. import models


def scale_dpt(dpt):
    """
    scale dpt array to 0 and 1
    """
    dpt_min = dpt.min()
    dpt_max = dpt.max()
    dpt_scaled = (dpt - dpt_min) / (dpt_max - dpt_min)
    
    return dpt_scaled

def load_model(ckptvx):
    r"""
    from a given ckpt , load the MLP model
    """
    model_vx = models.MLP(lr=1e-4, channels=[6,32,32,1], activation_fn='Tanh')
    ckpt = torch.load(ckptvx)
    model_vx.load_state_dict(ckpt['state_dict'])
    return model_vx

def pred_to_nday(x, n_timepoint=5, n_dim=5): 
    """
    use to reshape prediction
    """
    nday = x.detach().cpu().numpy().reshape(n_timepoint,-1)

    # if nday.shape[-1] != t5_ad.shape[0]:
    #     nday = nday.reshape(n_timepoint, -1, n_dim)
    return nday

######
#     Gaussian Density
######

def compute_guassian_u(Cellstate_ay, dimension=10):
    r"""
    Estimating the density high dimensional cell state coordinates and its change by experimental time.
    The density estimation function is 
    
    
    Args
    -------
        Cellstate_ay : ndarray of shape (N_cell, dimension), the high dimensional cell state representation, i.e. PC, Diffusion Map (DM) , SCVI-latent
        dimension : control the nubmer of the first few dimension to use for density esitmation

    Return
    -------
        gussian_kde_u : ndarray of shape (N_cell, 1), the density of each cell 

    Example
    ------
    >>> timepoints = sorted(ad.obs.timepoint_tx_days.unique()) # get timepoints
    >>> cbs_t = [ad.obs.query("`timepoint_tx_days` == @t").index for t in timepoints]
    >>> DM_ay = [ad[cb].obsm['DM_eigen'] for cb in cbs_t]
    >>> gussian_kde_u = guassian_u(DM_ay, dimension=5)
    >>> ad.obs['DM5_gaussisn_u']= gussian_kde_u

    """
    gussian_kde_u = []

    for m in Cellstate_ay:
        # dimension truncated matrix
        dm = m[:, :dimension].T  
        
        # take in [n_dim, n_sample]
        kde_fn = gaussian_kde(dm, bw_method='silverman')       
        gussian_kde_u.append(kde_fn(dm))

    gussian_kde_u = np.concatenate(gussian_kde_u, axis=0)

    return gussian_kde_u

def boundary_density_at(D, t_b, x, density_fn:callable=None):
    """
    Evaluate the cell density at time point `t_b` using the pre-defined density function

    
    Args
    -------
    D: the extracted data dictionary
    t_b : the boundary time point
    x : the grided cell state coordiante
    density_fn : the pre-defined density function
    
    Return
    -------
    u_t : smoothed cell density over s and adjusted by pop size
    """

    if density_fn is not None:
        raise NotImplementedError("the predefined func is not sett up")
        # u_t, Nt, n_exp = predefine_density(D, t_b, x, density_fn)

    elif len(x.shape) == 1:
        u_t, Nt, n_exp = evaluate_1d_density(D, t_b, x)

    elif len(x.shape) == 2:
        u_t, Nt, n_exp = evaluate_2d_mesh_density(D, t_b, x)

    else:
        # for higher dimensional data
        # use the data point itself but not sampling from the entire space
        # u_t, Nt, n_exp = evaluate_2d_mesh_density(D, t_b, x)
        raise NotImplementedError("higher dimensional func is under development")
   
    return u_t, Nt, n_exp


def evaluate_1d_density(D, t_b, x):
    """
    extract the cell density at time point `t_b`, this function works for 1 dimensional data
    
    
    Args
    -------
        D: the extracted data dictionary
        t_b : the boundary time point
        x : the grided cell state coordiante
    
    Return
    -------
        u_t : smoothed cell density over s and adjusted by pop size
    """
    
    assert t_b in D['pop']['t'], "`t_b` is not the observed time point"
    
    # find the batch that belong to the tb time point
    tp_index = [i for i, t in enumerate(D['ind']['tp']) if t==t_b]  
    
    n_lib = len(tp_index) 
    
    n_grid = x.shape[0]
    
    ut = np.zeros((n_lib, n_grid))
    Nt = np.zeros(n_lib)
    
    for i0 in range(n_lib):
        
        i_hist = tp_index[i0]
        
        # compute the density function based on the pdt coord
        density = gaussian_kde(D['ind']['hist'][i_hist])
        
        # evaluate and normalize at grided time x
        ut[i0, :] = density(x)
        ut[i0, :] /= trapz(ut[i0, :], x)
        Nt[i0] = len(D['ind']['hist'][i_hist]) # n cells


    tb_index = np.where(D['pop']['t']==t_b)[0].item()
    u_t = np.mean(ut, axis=0) * D['pop']['mean'][tb_index]
    # u_t = 0.5 * (u_t[:-1] + u_t[1:])

    # n_exp = 1  #TODO: change n_exp ?
    n_exp = n_lib
    
    return u_t, Nt, n_exp


def evaluate_2d_mesh_density(D, t_b, x):
    """
    extract the cell density at time point `t_b`. this function works for 2 dimensional mesh grid 
    
    Args
    -------
        D: the extracted data dictionary
        t_b : the boundary time point
        x : the mesh grided cell state s1, s2
    
    Return
    -------
        u_t : smoothed cell density over s and adjusted by pop size
    """
    
    assert t_b in D['pop']['t'], "`t_b` is not the observed time point"
    
    # find the batch that belong to the tb time point
    tp_index = [i for i, t in enumerate(D['ind']['tp']) if t==t_b]  
    
    n_lib = len(tp_index) 
    
    n_grid = x.shape[0]  # for 2 d, i.e 300 * 300

    space = (x[1,0] - x[0,0])**2
    
    ut = np.zeros((n_lib, n_grid))
    Nt = np.zeros(n_lib)
    
    for i0 in range(n_lib):
        
        i_hist = tp_index[i0]
        
        # compute the density function based on the pdt coord
        # the input data is (#dim , #samples)
        density = gaussian_kde(D['ind']['hist'][i_hist].T)
        
        # evaluate and normalize at grided time x
        ut[i0, :] = density(x.T)
        ut[i0, :] /= trapz(ut[i0, :], dx=space)
        Nt[i0] = len(D['ind']['hist'][i_hist]) # n cells


    tb_index = np.where(D['pop']['t']==t_b)[0].item()
    u_t = np.mean(ut, axis=0) * D['pop']['mean'][tb_index]
    # u_t = 0.5 * (u_t[:-1] + u_t[1:])

    # n_exp = 1  #TODO: change n_exp ?
    n_exp = n_lib
    
    return u_t, Nt, n_exp

#####
#    mellon density
#####

def compute_mellon_u(adata, cellstate_key, timepoint_key, n_dimension=None):
    try:
        model_t = mellon.DensityEstimator()
    except:
        import mellon
    
    adata_tb = adata.obs[timepoint_key]          # pd.Series
    X = adata.obsm[cellstate_key][:, :n_dimension]        # np.values
    
    density_funs = []
    log_u = []

    for i, t in enumerate(sorted(adata_tb.unique())):
        print('estimating density for time ', t)
        cb_t = adata.obs.query(f"`{timepoint_key}` == @t").index
        cs_t = adata[cb_t].obsm[cellstate_key][:, :n_dimension]   # cellstate t

        model_t = mellon.DensityEstimator()
        model_t.fit(cs_t)

        log_density = model_t.predict(X)
        
        log_u.append(log_density)
        density_funs.append(model_t)
    return np.stack(log_u), density_funs

def compute_mellon_timesense_u(adata, cellstate_key, timepoint_key, ls_time_estimate=None, n_dimension=None):
    r"""
    wrapping mellon time-sense density

    Args:
    -------
    adata : the cells from which for estimating density
    cellstate_key : the low dimensional representation
    timepoint_key : the obs key that record tiempoint, dtype should be int or float
    ls_time_estimate : float, the length scaler for time
    n_dimension : [int, None] ,
    
    Returns:
    -------
    log_u : np.array of shape (n_timepoints, n_cells), the log density
    density_predictor : trained `mellon.TimeSensitiveDensityEstimator.predict`


    Example:
    -------
    >>> import pseudodynamics as pdp
    >>> # a simple example for estimating density of a specific condition
    >>> adata_c1 = adata[adata.obs['condition'] == 'condition_1'].copy()
    >>> timepoints = adata_c1.obs['time'].unique()

    >>> # time ls for exploration
    >>> ls_time_estimate = 1.5 * np.mean(np.diff(np.sort(timepoints)))
    >>> # timesensitive estimator
    >>> log_u, estimator = pdp.tl.compute_mellon_timesense_u(adata_c1, 'DM_EigenVectors', 'time',   ls_time_estimate)
    """
    
    try:
        model_t = mellon.DensityEstimator()
    except:
        import mellon
    
    adata_tb = adata.obs[timepoint_key]          # pd.Series
    X = adata.obsm[cellstate_key][:, :n_dimension]        # np.values
    X_times = adata.obs[timepoint_key]          
    
    density_funs = []
    log_u = []

    if ls_time_estimate is None:
        ls_time_estimate = 1.5 * np.mean(np.diff(np.sort(X_times.unique())))

    print(ls_time_estimate)

    # Initialize the time-sensitive density estimator with an intrinsic dimensionality of 2
    try:
        t_est = mellon.TimeSensitiveDensityEstimator(d=2, ls_time=ls_time_estimate)
        # Fit the estimator to the data
        t_est.fit(X, X_times)

    except RuntimeError:
        os.environ['JAX_PLATFORMS'] = 'cpu'
        t_est = mellon.TimeSensitiveDensityEstimator(d=2, ls_time=ls_time_estimate)
        t_est.fit(X, X_times)


    for i, t in enumerate(sorted(adata_tb.unique())):
        print('estimating density for time ', t)
        
        # Save the predictor for later density evaluations
        density_predictor = t_est.predict

        log_density = density_predictor(X,t)
        
        log_u.append(log_density)

    return np.stack(log_u), density_predictor



def evaluate_u_ds(cid, cellstate, u_tb, delta_s, den_fn, scaler):
    r"""
    func for multi-process density estimate inside time-point loop

    
    Args
    ------
        cid: cell index , numerical index, not cell barcode
        cellstate : array [n_cell, n_dim] high-dimensional coordinate (representation) of all cells
        u_tb : array [n_cell,] , the density of each cell at a timepoint
        delta_s : [n_dim, ] the small perturbation add to cell state
        den_fn : callable([n_dim, n_cell]) ,  the density estimation function
        scaler : density scaler , normlized 

    Return
    ------
        the duds : [n_cell, n_dim] , the density chagne along each dimension
    """
    # cs = cellstate[cid]
    # u_t = u_tb[cid]
    du_dcs_cell = []

    for i in range(cellstate.shape[1]):
        ds = np.zeros_like(cs)
        ds[i] = delta_s[i]
        s_prime = cs + ds
        u_prime = den_fn(s_prime.reshape(-1,1)) * scaler
        u_t = den_fn(cs.reshape(-1,1)) * scaler
        dudcs = (u_prime - u_t)/delta_s[i]
        du_dcs_cell.append(dudcs.reshape(1,1))

    return np.concatenate(du_dcs_cell, axis=1)

def augment_cdf(x, x_a, y):
    r"""
    This function takes a CDF defined on a grid x and augments it to a finer grid x_a by duplicating the CDF values within intervals defined by x. zeros or ones accordingly were added for where x_a values are outside the range of x
    """
    
    i_shift = 0
    y_a = y.copy()
    for i in range(len(x) - 1):
        i1 = len(np.where(x_a > x[i])[0]) + len(np.where(x_a < x[i + 1])[0]) - len(x_a)
        if i1 > 0:
            ones_matrix = np.ones((i1, y_a.shape[1]))
            
            # If any x_a within the interval, 
            # duplicates the CDF values at the current interval
            y_a = np.concatenate(
                (y_a[:i + i_shift + 1, :], y_a[i + i_shift, :] * ones_matrix, y_a[i + i_shift + 1:, :]), 
                axis=0)
            i_shift += i1
            
    
    # boundary condition :  x_a  are smaller than the minimum
    ia = np.where(x_a < x[0])[0]
    y_a = np.concatenate((np.zeros((len(ia), y_a.shape[1])), y_a), axis=0)
    
    # boundary condition :  x_a  are larger than the maximum
    ie = np.where(x_a > x[-1])[0]
    y_a = np.concatenate((y_a, np.ones((len(ie), y_a.shape[1]))), axis=0)
    return y_a


def Lambda1(epoch, gamma=0.2):
    r"""
    lr scheduler rule 1, quickly decay then steady
    """
    if epoch >= 5 :
        factor = np.exp((3-epoch**0.4))**gamma + 0.1
        
    else:
        factor = np.exp((5-epoch**0.6))

    return 3 if factor > 3 else factor 


def Lambda2(epoch, gamma=0.15, changepoint=150):
    """
    lr scheduler rule 2, decay then grow
    """

    factor_decay = np.exp((3-epoch**0.7))**gamma + 0.01

    changepoint_f = np.exp((3-changepoint**0.7))**gamma

    factor_grow = changepoint_f * np.exp((epoch/changepoint)**2)**gamma

    return factor_decay if epoch <= changepoint else factor_grow 

def traverse_neighbor(connectivities, k, knn_idx):
    k = -1*k 
    # random walk
    next_degree_knn = []
    for i_d in knn_idx:
        neighbor = np.argpartition(connectivities[i_d].A, k)[k:].tolist()
        next_degree_knn.extend(neighbor)
    
    # all neighbor visited to the current degree

    knn_idx_d_plus1 = knn_idx + next_degree_knn
    # knn_idx_d_plus1 = np.unique(knn_idx_d_plus1).astype(int)
    knn_idx_d_plus1 = np.unique(next_degree_knn).astype(int).tolist()
    return knn_idx_d_plus1

def _sample_by_distance(dist_array, candidate_idx, alpha=None, repeat=1):
    r"""
    based on the knn distance, sample the closest cell 
        P ~ (1 - distance)

    Args
    ------
        dist_array : ndarray, distance from all other cells to cell i
        candidate_idx : the index of knn / or cansidered neighbor cells
        alpha: parameter to adjust uncertainty, the lower the more uncertrain
        repeat: the number of cells to sample each time
    """
    alpha = 1 if alpha is None else alpha

    dist = dist_array[candidate_idx]
    where_inf = np.isinf(dist)
    if np.any(where_inf):
        try:
            next_max = dist[~where_inf].max()
            dist = np.where(where_inf, next_max, dist)
        except ValueError:
            dist = np.zeros_like(dist)

    knn_p = dist.max() - dist + 0.1*dist.min()
    if knn_p.sum() > 0:
        knn_p = knn_p**alpha
        p = knn_p / knn_p.sum() # normalized
    else:
        p = None 

    neighbor_idx = [np.random.choice(candidate_idx, p=p) for i in range(repeat)]
    if repeat == 1:
        neighbor_idx = neighbor_idx[0]
    return neighbor_idx, p


def sample_deltax(adata,transition_matrix=None, max_degree=1, k=None, xkey=None, pseudotimekey='palantir_pseudotime', progressbar=True, temperature=1, repeat=1):
    """
    the Key function defines the noise sampling process 
    given the starting point i

    Return:
    -----
    delta_X, neighbor_ls
    """

    if transition_matrix is None:
        delta_X, neighbor_ls = sample_deltax_from_knn(adata, max_degree=max_degree, k=k, xkey=xkey, pseudotimekey=pseudotimekey, progressbar=progressbar, temperature=temperature, repeat=repeat)
    
    else:
        delta_X, neighbor_ls = sample_deltax_from_transition(adata, transition_matrix,xkey=xkey,repeat=repeat)

    return delta_X, neighbor_ls
        

def sample_deltax_from_transition(adata, transition_matrix,xkey=None, pseudotimekey='palantir_pseudotime', progressbar=False, repeat=1):
    r"""
    the Key function defines the delta-X sampling process 
    given the transition matrix, then sample the delta x

    
    Args:
    -----
        transition_matrix : nd array of shape (n_cell, n_cell), i.e : cell rank transition matrix
        xkey : obsm_key or layer_key of the adata, from which space we sample the delta x


    Return:
    -----
        delta_X, neighbor_ls

    Example:
    -----
    >>> ck = cr.kernels.ConnectivityKernel(adata)
    >>> ck.compute_transition_matrix()
    >>> pk = cr.kernels.PseudotimeKernel(adata, time_key="palantir_pseudotime")
    >>> pk.compute_transition_matrix()
    >>> combined_kernel = 0.8 * pk + 0.2 * ck
    >>> combined_kernel.compute_transition_matrix()
    >>> adata.obsp['combined_transition_matrix'] = combined_kernel.transition_matrix
    """

    def prograss_(x, turn_on=progressbar):
        if turn_on:
            return tqdm(x)
        else:
            return x
    
    pdt = adata.obs[pseudotimekey].values

    if xkey is None:
        X = adata.X
    elif xkey in adata.obsm_keys():
        X = adata.obsm[xkey].copy()
    elif xkey in adata.layers():
        X = adata.layers[xkey].copy()

    delta_X = []
    neighbor_ls = []

    T_M = transition_matrix 
    for i in prograss_(range(X.shape[0]), progressbar):
        nz_id = np.where(T_M[0].A[0]!=0)[0]   # non zero cell idx
        prob = T_M[0].A[0, nz_id].reshape(1,-1) # (1,knn)
        neighbor_ls.append(nz_id)

        weighted_dx =prob@(X[nz_id] - X[i]) #(1,nn) * (nn, n_dim)
        delta_X.append( weighted_dx.squeeze()) 
    
    return np.stack(delta_X), np.array(neighbor_ls)


def sample_deltax_from_knn(adata, max_degree=1, k=None, xkey=None, pseudotimekey='palantir_pseudotime', progressbar=False, temperature=1, repeat=1):
    """
    the Key function defines the noise sampling process 
    given the starting point i

    Return:
    -----
        delta_X, neighbor_ls
    """

    connectivities = adata.obsp['connectivities'].copy()
    distance = adata.obsp['distances'].copy()
    pdt = adata.obs[pseudotimekey].values

    if k is None:
        k  = adata.uns['neighbors']['params']['n_neighbors']
    
    if xkey is None:
        X = adata.X
    elif xkey in adata.obsm_keys():
        X = adata.obsm[xkey].copy()
    elif xkey in adata.layers():
        X = adata.layers[xkey].copy()

    def prograss_(x, turn_on=progressbar):
        if turn_on:
            return tqdm(x)
        else:
            return x
    
    delta_X = []
    neighbor_ls = []
    
    for i in prograss_(range(X.shape[0]), progressbar):

        knn_idx = np.array([i])
        t_i = pdt[i]

        # init
        pass_1 = 0
        pass_2 = 0
        n_degree = 0

        final_index = []
        
        # while pass_1*pass_2==0 and n_degree < max_degree:
        # for d in range(max_degree):  
            # if n_degree > self.free_search_degree:   # only under this degree can we expand knn without any constraints
            #     knn_idx = pass_2_idx
        
            # knn_idx = traverse_neighbor(connectivities, k, knn_idx) 
        try:
            nn_i = connectivities[i].A[0]
        except:
            nn_i = np.array(adata.obsp['connectivities'][i].todense())[0]

        knn_idx = np.argpartition(nn_i, -1*k)[-1*k:].tolist()

        # 1 : neighbor with the same condition
        pass_1_idx = knn_idx
        if len(pass_1_idx) > 0:
            pass_1 = 1
        else:
            final_index = [i]
            # continue

        # 2 : neighbor with bigger pseudo-time
        knn_t = pdt[pass_1_idx]
        if np.any(knn_t > t_i):
            pass_2_idx = np.array(pass_1_idx)[knn_t > t_i]
            final_index = pass_2_idx
            pass_2 = 1

        else:
            pass_2_idx = pass_1_idx
            final_index = [i]
            # continue
            # n_degree += 1
                    
        # sampled by distance

        
        if len(final_index) == 1:
            neighbor_idx = [final_index[0]] * repeat
        else:
            # neighbor_idx, p = _sample_by_distance(distance, final_index)
            knn_p = nn_i[final_index].flatten()**temperature
            p = knn_p / knn_p.sum() if knn_p.sum() != 0 else None

            neighbor_idx = np.random.choice(final_index,  p=p, size=repeat, replace=True)

        delta_X.append( X[neighbor_idx] - X[i] )
        neighbor_ls.append(neighbor_idx)

    return delta_X, neighbor_ls


def train_test_split_adata(adata, leaveout=[None], val_size=0.1, test_size=0.1, timepoint_key='timepoint_tx_days'):

    obs = adata.obs
    val_test_cbs = obs.query(f"`{timepoint_key}` not in @leaveout").sample(frac=0.2).index
    test_cb = np.random.choice(val_test_cbs, size=len(val_test_cbs)//2,replace=False)

    split_mapper = {cb:'test' for cb in test_cb}
    val_cbs = {cb:'val' for cb in val_test_cbs if cb not in test_cb}
    train_cbs = {cb:'train' for cb in adata.obs_names if cb not in val_test_cbs}

    split_mapper.update(val_cbs)
    split_mapper.update(train_cbs)

    return adata.obs.index.map(split_mapper)

def make_coord_adata(adata, cellstate_key, n_dimension, v = None):
    r"""
    construct adata based on cellstate coodinates from expression matrix based adata
    the new adata is mainly for visualizing v

    
    Args:
    -----------
        adata : anndata, the source anndata to extract info
        cellstate_key : which representation to use
        n_dimesion : the number of first dimeion of cellstate to use
        v : ndarray of shape [t, n_dim], default none
    """
    # create new adata with DM coordinate as X
    cellstate = adata.obsm[cellstate_key][:,:n_dimension]
    new_ad = ad.AnnData(
        X = cellstate,
        obs = adata.obs.copy(),
        var = pd.DataFrame(['DM_%d'%d for d in range(cellstate.shape[1])]).set_index(0)
    )

    # transfer other highdim matrix
    new_ad.obsp['connectivities'] = adata.obsp['connectivities'].copy()
    new_ad.obsp['distances'] = adata.obsp['distances'].copy()
    new_ad.layers['cellstate'] = new_ad.X.copy()
    for key in adata.obsm_keys():
        # if key.startswith('X_'):
        #     new_ad.obsm[key] = adata.obsm[key].copy()
        new_ad.obsm[key] = adata.obsm[key]
    
    # pop info
    new_ad.uns = adata.uns.copy()
    timepoints = adata.uns['pop']['t']
    n_timepionts = len(timepoints)

    if v is not None:
        if v.shape[0] == 1:
            # single timepoint
            new_ad.layers['v'] = v
        elif v.shape[0] > 1: 
            assert len(v.shape) == 3, "please put in the raw v"
            for i, t in enumerate(timepoints):
                new_ad.layers[f'Day{t} v'] = v[i]

    return new_ad

def super_resolution_pseudobulk(adata, resolution=200, n_pseudobulk=None, key_added='pseudo_bulk', seed=42, rapids_singlecell=False):
    r"""
    Use super-high resolution leiden algorithm to generate pseudo-bulk
    
    Augments
    --------
        n_pseudobulk : int, defult None -> adata.shape[0] / 20, the number of pseudo bulk to end with
        pseudobulk_key : str, default 'pseudo_bulk'
        resolution : int, default None -> 40, the resolution pass to leiden algorithm

    Return
    --------
        adata with pseudo bulk key
    """
    if n_pseudobulk is None:
        n_pseudobulk = adata.shape[0] / 20 
    if resolution is None:
        resolution = 200 

    ncell = adata.shape[0]
    if rapids_singlecell and (ncell > 1e4):
        try:
            import rapids_singlecell as rsc 
            leiden = rsc.tl.leiden
            print("rapids_singlecell detected, rsc leiden is used to accelarate")
        except:
            pass
    else:
        leiden = sc.tl.leiden
    
    magnitude_of = lambda x: int(np.log2(x))

    if key_added not in adata.obs_keys():
        leiden(adata, resolution=resolution, key_added=key_added, random_state=seed)

    while magnitude_of(adata.obs[key_added].nunique()) < magnitude_of(n_pseudobulk):
        resolution=resolution*5
        leiden(adata, resolution=resolution, key_added=key_added, random_state=seed)
        

    print(f"getting {adata.obs['pseudo_bulk'].nunique()} pseudobulk with resolution {resolution}")

    adata.uns[f'{key_added}_settings'] = {'resolution':resolution, 'n_pseudobulk':n_pseudobulk}

    return adata
    

def get_pseudobulk(adata, cellstate_key, n_dimension=None, pseudobulk_key='pseudo_bulk', keep_index=False):
    r"""
    generate pseudo-bulk (meta-cell) using super-high resolution clustering

    
    Args
    -----------
        adata_DM : adata with DM was X
        pseudobulk_key : str, a obs key to specify the pseudobulk_key to use 

    Return 
    -----------
        pdata : anndata with [n_pseudobulk, n_dimension]

    Example
    -----------
    >>> adata = super_resolution_pseudobulk(adata, resolution=400, key_added='pseudo_bulk') # leiden clustering
    >>> adata_DM = make_coord_adata(adata, cellstate_key='DM_scaled', n_dimesion=10)        # rebase data on low dimension
    >>> pdata = get_pseudobulk(adata_DM, 'pseudo_bulk')       # generate pseudobulk
    >>> pdata.shape
    """
    # adata_DM.obs = adata_DM.obs.dropna(axis=1, how='any')
    # pdata = dc.get_pseudobulk(
    #     adata_DM,
    #     sample_col=pseudobulk_key,
    #     groups_col=None,
    #     mode='mean',
    #     min_cells=1,
    #     min_counts=0,
    #     skip_checks = True
    # )

    # get X and dimension
    if cellstate_key in adata.obsm_keys():
        X = adata.obsm[cellstate_key][:,:n_dimension]
    elif cellstate_key in adata.obs_keys():
        X = adata.obs[cellstate_key].values[:,None]
    n_dimension = X.shape[1] if n_dimension is None else n_dimension

    # group
    X_df = pd.DataFrame(X, columns=['DM_%s'%i for i in range(n_dimension)])
    X_df[pseudobulk_key] = pd.Series(adata.obs[pseudobulk_key].values, dtype='str')
    grouped_cellstate = X_df.groupby(pseudobulk_key).agg("mean")

    if keep_index:
        return grouped_cellstate
    else:
        return grouped_cellstate.values