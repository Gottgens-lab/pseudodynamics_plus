import os
from scipy.interpolate import interp1d
from sklearn.linear_model import LinearRegression
from warnings import catch_warnings, simplefilter
from statsmodels.genmod.families.family import NegativeBinomial
from statsmodels.gam.generalized_additive_model import GLMGamResultsWrapper 
from statsmodels.gam.api import GLMGam, BSplines
from abc import ABC, abstractmethod
from collections import defaultdict
from itertools import combinations
from typing import Literal, Optional, Union, List
import numpy as np
import pandas as pd
from scipy.linalg import qr, solve_triangular
from scipy.stats import chi2
from scipy import sparse
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial

logger = logging.getLogger(__name__)



class _SmootherHolder:
    """Minimal stand-in for ``result.model`` so that ``result.model.smoother`` keeps working."""
    def __init__(self, smoother):
        self.smoother = smoother


class SlimGAMResult:
    """
    Lightweight replacement for a statsmodels ``GLMGamResults`` object.

    A full statsmodels result keeps the data, the spline basis and the residuals of
    every gene (about 1 MB per gene), which exhausts memory for tens of thousands of
    genes. The association tests only need the coefficients, their covariance and the
    smoother, so only those are kept. The smoother is shared by all genes fitted on the
    same pseudotime.
    """
    def __init__(self, res, smoother):
        self.params = np.asarray(res.params, dtype=float)
        self._cov_params = np.asarray(res.cov_params(), dtype=float)
        self.model = _SmootherHolder(smoother)

    def cov_params(self):
        return self._cov_params


def fitGAM(count_matrix, cell_time, n_knots=7, cell_weights=None, gene_symbol=None):
    r"""
    Fits a GAM model per gene using pseudotime.

    Args:
    ------
    count_matrix : np.ndarray
        Gene expression matrix (cells x genes).    
    cell_time : np.ndarray
        Pseudotime values for cells.  
    n_knots: int 
        Number of knots for spline fitting.  
    cell_weights :  np.ndarray
        Optional weights for each cell.  

    Returns:
    ------
    GAMResult: dict
        Dictionary containing fitted models and metadata.  
    """
    if cell_weights is None:
        cell_weights = np.ones(len(cell_time))

    cell_time = np.asarray(cell_time, dtype=float)
    models = {}
    design = np.linspace(0, 1, len(np.unique(cell_time)))   # identical for every gene

    # sorted pseudotime, shared by every pred_func (interp1d copies nothing then)
    order = np.argsort(cell_time)
    t_sorted = cell_time[order]
    zero_pred = interp1d(t_sorted, np.zeros(len(t_sorted), dtype=np.float32), assume_sorted=True,
                         copy=False, fill_value='extrapolate')

    # one B-spline basis on the pseudotime, shared by every gene
    try:
        bs = BSplines(cell_time.reshape(-1, 1), df=n_knots, degree=3)
    except ValueError:
        bs = None

    for i_gene in range(count_matrix.shape[1]):
        expr = np.asarray(count_matrix[:, i_gene], dtype=float)

        # constant expression, too few cells, or no basis: nothing to fit
        if bs is None or np.all(expr == expr[0]) or len(np.unique(cell_time)) < 2:
            models[i_gene] = {'model': None, 'design': design, 'pred_func': zero_pred}
            continue

        # Fit GAM using GLMGam with explicit endog and smoother
        try:
            gam = GLMGam(endog=expr, smoother=bs, alpha=1e-6, family=NegativeBinomial(alpha=1.0))  # Add small regularization

            # Suppress perfect separation warnings
            with catch_warnings():
                simplefilter("ignore")
                res = gam.fit()

            if res is None:
                models[i_gene] = {'model': None, 'design': design, 'pred_func': zero_pred}
            else:
                fitted = np.asarray(res.predict(), dtype=np.float32)[order]
                models[i_gene] = {
                    'model': SlimGAMResult(res, bs),      # coefficients, covariance, shared smoother
                    'design': design,
                    'pred_func': interp1d(t_sorted, fitted, assume_sorted=True, copy=False, fill_value='extrapolate'),
                }
        except:
            # Fallback to linear regression if GAM fails
            lr = LinearRegression()
            X = cell_time.reshape(-1, 1)
            lr.fit(X, expr)
            fitted = np.asarray(lr.predict(X), dtype=np.float32)[order]
            models[i_gene] = {
                'model': lr,
                'design': design,
                'pred_func': interp1d(t_sorted, fitted, assume_sorted=True, copy=False, fill_value='extrapolate'),
            }
    return {
        'models': models,
        'gene_symbol': gene_symbol,
        'gene_names' :np.arange(count_matrix.shape[1]),
        'pseudotime': cell_time
    }


def save_gamfit(gam_fit, save_dir):
    """
    Save a GAM fit to disk: metadata as .npy files, the slim models as one pickle and
    the fitted expression values (float32) of every gene as one array.
    """
    import pickle
    os.makedirs(save_dir, exist_ok=True)

    for k in ['gene_symbol', 'gene_names', 'pseudotime']:
        np.save(os.path.join(save_dir, f"{k}.npy"), gam_fit[k])

    models = gam_fit['models']
    first = next(iter(models.values()))
    np.save(os.path.join(save_dir, "design.npy"), first['design'])

    # fitted values on the sorted pseudotime, shared x
    t_sorted = np.sort(np.asarray(gam_fit['pseudotime'], dtype=float))
    fitted = np.zeros((len(models), len(t_sorted)), dtype=np.float32)
    for i, (gene_id, fit) in enumerate(models.items()):
        if fit['pred_func'] is not None:
            fitted[i] = fit['pred_func'](t_sorted)
    np.save(os.path.join(save_dir, "fitted.npy"), fitted)
    np.save(os.path.join(save_dir, "cell_time.npy"), t_sorted)

    with open(os.path.join(save_dir, "models.pkl"), "wb") as f:
        pickle.dump({gene_id: fit['model'] for gene_id, fit in models.items()}, f)


def load_gamfit(save_dir):
    """
    Load a GAM fit written by ``save_gamfit``.

    Returns:
        gam_fit (dict): same layout as the output of ``fitGAM``
    """
    import pickle
    assert os.path.exists(save_dir)

    gam_fit = {}
    for key in ['gene_symbol', 'gene_names', 'pseudotime']:
        gam_fit[key] = np.load(os.path.join(save_dir, f"{key}.npy"), allow_pickle=True)

    design = np.load(os.path.join(save_dir, "design.npy"), allow_pickle=True)
    t_sorted = np.load(os.path.join(save_dir, "cell_time.npy"), allow_pickle=True)
    fitted = np.load(os.path.join(save_dir, "fitted.npy"), allow_pickle=True)
    with open(os.path.join(save_dir, "models.pkl"), "rb") as f:
        slim_models = pickle.load(f)

    models = {}
    for i, gene_id in enumerate(gam_fit['gene_names']):
        models[gene_id] = {
            'model': slim_models.get(gene_id),
            'design': design,
            'pred_func': interp1d(t_sorted, fitted[i], assume_sorted=True, copy=False, fill_value='extrapolate'),
        }
    gam_fit['models'] = models
    return gam_fit

# Add a simple wrapper function for easier use

def fit_gam_simple(expr, time, n_knots=7):
    """
    Simple wrapper for GAM fitting

    Parameters:
        expr (np.ndarray): Gene expression values
        time (np.ndarray): Pseudotime values
        n_knots (int): Number of knots for spline fitting

    Returns:
        tuple: (predicted_values, model)
    """
    # Check for constant expression
    if np.all(expr == expr[0]) or len(np.unique(time)) < 2:
        return np.zeros_like(time), None
    
    # Create B-spline basis
    try:
        bs = BSplines(time.reshape(-1, 1), df=n_knots, degree=3)
    except ValueError as e:
        return np.zeros_like(time), None
    
    # Fit GAM using GLMGam with explicit endog and smoother
    gam = GLMGam(endog=expr, smoother=bs)
    
    # Suppress perfect separation warnings
    with catch_warnings():
        simplefilter("ignore")
        res = gam.fit()
        
    if res is None:
        return np.zeros_like(time), None
    else:
        return res.predict(), res



def process_gene_chunk(gene_indices, expression_matrix, cell_time, n_knots, gene_symbol):
    """
    Process a subset of genes using the fitGAM function
    
    Parameters:
    gene_indices: indices of genes to process
    expression_matrix: full gene expression matrix
    cell_time: pseudotime values
    n_knots: number of knots for spline fitting
    gene_symbol: list of gene names
    
    Returns: dict with results for processed genes
    """
    # `expression_matrix` holds only the columns of `gene_indices` (sliced by the caller),
    # so a worker never receives the full matrix.
    result = fitGAM(
        count_matrix=expression_matrix,
        cell_time=cell_time,
        n_knots=n_knots,
        gene_symbol=[gene_symbol[i] for i in gene_indices]
    )
    
    # Reindex models to match original indices
    reindexed_models = {}
    for local_idx, global_idx in enumerate(gene_indices):
        reindexed_models[global_idx] = result['models'][local_idx]
    
    return {
        'models': reindexed_models,
        'gene_symbol': result['gene_symbol'],
        'gene_names': [result['gene_names'][i] for i in range(len(gene_indices))],
        'pseudotime': result['pseudotime']
    }

def run_fitGAM_parallel(expression_matrix, cell_time, genes, n_knots=7, n_cores=20):
    """
    Run fitGAM in parallel across multiple cores using the original implementation
    
    Parameters:
    expression_matrix: gene expression matrix (cells x genes)
    cell_time: pseudotime values for cells
    genes: list of gene names
    n_knots: number of knots for spline fitting
    n_cores: number of CPU cores to use
    
    Returns: dict with all gene results
    """
    n_genes = expression_matrix.shape[1]
    
    # Create equal-sized chunks of gene indices
    gene_indices = np.array_split(np.arange(n_genes), n_cores)
    
    # Fixed parameters; the expression columns of each chunk are passed separately so
    # that every worker only receives (and holds) its own genes.
    worker_func = partial(process_gene_chunk, cell_time=cell_time, n_knots=n_knots, gene_symbol=genes)
    chunk_matrices = (np.ascontiguousarray(expression_matrix[:, idx]) for idx in gene_indices)

    # Run in parallel
    with ProcessPoolExecutor(max_workers=n_cores) as executor:
        chunk_results = list(executor.map(worker_func, gene_indices, chunk_matrices))
    
    # Combine results from all chunks
    combined_result = {
        'models': {},
        'gene_symbol': genes,
        'gene_names': np.arange(n_genes),
        'pseudotime': cell_time
    }
    
    # Merge all models into one dictionary
    for chunk in chunk_results:
        combined_result['models'].update(chunk['models'])
    
    return combined_result

class DifferentialExpressionTest(ABC):
    """Abstract base class for a DifferentialExpressionTest."""

    def __init__(self, model):
        """Initialize DifferentialExpressionTest class.

        Parameters
        ----------
        model
            Fitted GAM class.
        """
        self._model = model

    @abstractmethod
    def __call__(self, **kwargs):
        """Perform the DifferentialExpressionTest."""


def _wald_test(
    prediction: np.ndarray,
    contrast: np.ndarray,
    sigma: np.ndarray,
    method: Literal["qr", "pinv", "inv"] = "qr"
):
    """Perform Wald test with Python-native linear algebra."""
    # Find linearly independent rows
    q, r, piv = qr(contrast.T, pivoting=True, mode='economic')
    rank = np.sum(np.abs(np.diag(r)) > 1e-10)
    if rank == 0:
        return np.nan, np.nan, np.nan
    
    # Reduce to independent rows
    piv = piv[:rank]
    contrast_reduced = contrast[piv]
    prediction_reduced = prediction[piv]
    
    # Compute covariance of contrasts
    cov_contrast = contrast_reduced @ sigma @ contrast_reduced.T
    
    # Invert based on selected method
    if method == "qr":
        q_cov, r_cov = qr(cov_contrast, mode='economic')
        try:
            inv_cov = solve_triangular(r_cov, q_cov.T, lower=False)
        except:
            inv_cov = np.linalg.pinv(cov_contrast)
    elif method == "pinv":
        inv_cov = np.linalg.pinv(cov_contrast)
    elif method == "inv":
        inv_cov = np.linalg.inv(cov_contrast)
    else:
        raise ValueError("Invalid inversion method")
    
    # Compute Wald statistic
    wald = prediction_reduced @ inv_cov @ prediction_reduced
    wald = max(0, wald)  # Ensure non-negative
    
    # Degrees of freedom and p-value
    df = rank
    pval = chi2.sf(wald, df)
    
    return wald, df, pval


def _validate_shapes(beta, sigma, L, context=""):
    """Validate shapes of beta, sigma, and L matrices."""
    logger.debug(f"{context}: beta shape: {beta.shape}")
    logger.debug(f"{context}: sigma shape: {sigma.shape}")
    logger.debug(f"{context}: L shape: {L.shape}")
    
    # Ensure beta is a column vector
    if beta.ndim == 1:
        beta = beta.reshape(-1, 1)
        
    # Validate dimensions
    if sigma.shape[0] != sigma.shape[1]:
        raise ValueError(f"Sigma matrix must be square, got {sigma.shape}")
        
    if beta.shape[0] != sigma.shape[0]:
        raise ValueError(f"Beta and sigma dimension mismatch: beta={beta.shape}, sigma={sigma.shape}")
        
    if L.shape[1] != beta.shape[0]:
        raise ValueError(f"L matrix and beta dimension mismatch: L={L.shape}, beta={beta.shape}")
        
    return beta


def _linearly_independent_rows(matrix, tol=1e-8):
    """Find indices of linearly independent rows using QR decomposition."""
    Q, R, P = qr(matrix.T, pivoting=True, mode='economic')
    rank = np.sum(np.abs(np.diag(R)) > tol)
    return P[:rank] if rank > 0 else []


def _wald_test_fc(
    beta: np.ndarray,
    sigma: np.ndarray,
    L: np.ndarray,
    l2fc: float = 0,
    inverse: Literal["qr", "chol", "eigen", "generalized"] = "qr"
):
    """Python implementation of waldTestFC from tradeSeq"""
    # Ensure beta is column vector
    if beta.ndim == 1:
        beta = beta.reshape(-1, 1)  # Ensure column vector
        
    # Apply log fold change threshold
    log_fc_cutoff = np.log(2**l2fc) if l2fc != 0 else 0
    
    # Validate shapes
    beta = _validate_shapes(beta, sigma, L, "_wald_test_fc")
    
    # Compute contrast estimates
    est = L @ beta  # This will now work correctly
    
    # Apply fold change threshold
    if l2fc != 0:
        est = np.sign(est) * np.maximum(0, np.abs(est) - log_fc_cutoff)
    
    # Compute covariance matrix
    cov_contrast = L @ sigma @ L.T  # Updated to use correct matrix dimensions
    
    # Invert covariance matrix
    if inverse == "chol":
        try:
            # Cholesky decomposition
            L_chol = np.linalg.cholesky(cov_contrast)
            inv_cov = solve_triangular(
                L_chol, 
                solve_triangular(L_chol, np.eye(L_chol.shape[0]), lower=True, trans='T'),
                lower=True
            )
        except np.linalg.LinAlgError:
            logger.warning("Cholesky decomposition failed, using pseudo-inverse")
            inv_cov = np.linalg.pinv(cov_contrast)
    elif inverse == "qr":
        try:
            # QR decomposition
            Q_cov, R_cov = qr(cov_contrast, mode='economic')
            inv_cov = solve_triangular(R_cov, Q_cov.T)
        except np.linalg.LinAlgError:
            logger.warning("QR decomposition failed, using pseudo-inverse")
            inv_cov = np.linalg.pinv(cov_contrast)
    elif inverse == "eigen":
        # Eigen decomposition
        try:
            w, v = np.linalg.eigh(cov_contrast)
            inv_cov = v @ np.diag(1/w) @ v.T
        except np.linalg.LinAlgError:
            logger.warning("Eigen decomposition failed, using pseudo-inverse")
            inv_cov = np.linalg.pinv(cov_contrast)
    else:  # "generalized" or fallback
        inv_cov = np.linalg.pinv(cov_contrast)
    
    # Compute Wald statistic
    try:
        wald = est.T @ inv_cov @ est
        wald = max(0, wald[0, 0] if wald.shape[0] > 1 else wald[0])
    except Exception as e:
        logger.error(f"Wald computation failed: {str(e)}")
        logger.error(f"est shape: {est.shape}, inv_cov shape: {inv_cov.shape}")
        raise

    # Calculate rank for degrees of freedom
    pivot = _linearly_independent_rows(L)
    rank = len(pivot)
    
    # Degrees of freedom and p-value
    df = rank
    pval = chi2.sf(wald, df)
    
    return wald, df, pval


def _get_predict_custom_point_df(
    pseudotime: float, 
    lineage_id: int,
    n_lineages: int,
    conditions: Optional[np.ndarray] = None,
    condition_id: Optional[int] = None
) -> np.ndarray:
    """Create design row for a prediction point.
    
    Parameters
    ----------
    pseudotime : float
        Pseudotime value for this point
    lineage_id : int
        Index of the lineage
    n_lineages : int
        Total number of lineages
    conditions : Optional[np.ndarray]
        Array of unique conditions
    condition_id : Optional[int]
        Index of the condition if multiple conditions are present
    
    Returns
    -------
    np.ndarray
        Design matrix row for this prediction point
    """
    # Create base design row (all zeros)
    design_row = np.zeros(n_lineages * 2)  # t + l for each lineage
    
    # Set lineage indicator
    design_row[lineage_id - 1] = 1  # lineage indicator
    
    # Set pseudotime
    design_row[n_lineages + lineage_id - 1] = pseudotime
    
    # Handle conditions if present
    if conditions is not None and condition_id is not None:
        # Expand design for conditions
        expanded = np.zeros(n_lineages * (1 + len(np.unique(conditions))))
        # Set condition-specific lineage
        condition_offset = lineage_id * len(np.unique(conditions)) + condition_id - 1
        expanded[condition_offset] = 1
        # Set pseudotime
        expanded[len(expanded)//2 + lineage_id - 1] = pseudotime
        return expanded
    
    return design_row




class BetweenLineageTest(DifferentialExpressionTest):
    """Class for performing association tests between lineages using GAMs."""

    def __init__(self, model, lineage_names):
        super().__init__(model)
        self._model = model
        self.lineage_names = lineage_names

    def associationTest(
        self,
        pseudotimes: List[np.ndarray],
        lineages: Union[List[int], np.ndarray],
        pairwise_test: bool = False,
        global_test: bool = True,
        l2fc: float = 0,
        n_points: int = None
    ):
        """Perform association tests between lineages."""
        # If n_points not provided, default to 2 * number of knots
        if n_points is None:
            n_points = 2 * self._model.smoother.df
        
        result = defaultdict(dict)
        lineage_ids = np.concatenate([
            np.full(len(pseudotimes[i]), i) for i in lineages
        ])
        all_pseudotimes = np.concatenate(pseudotimes)
        
        for var_id in range(self._model.exog.shape[1]):
            var_name = f"Gene_{var_id}"
            try:
                # Get covariance matrix for this gene
                cov = self._model.cov_params()
                if cov.size == 0:
                    continue
                    
                # Get predictions and linear predictor matrix
                predictions = self._model.predict(exog=self._model.exog)
                lp_matrix = self._model.exog
                
                # Calculate differences between lineages
                pred_diffs = []
                lp_diffs = []
                for lineage_a, lineage_b in combinations(lineages, 2):
                    mask_a = (lineage_ids == lineage_a)
                    mask_b = (lineage_ids == lineage_b)
                    
                    pred_diff = predictions[mask_a] - predictions[mask_b]
                    lp_diff = lp_matrix[mask_a] - lp_matrix[mask_b]
                    
                    pred_diffs.append(pred_diff)
                    lp_diffs.append(lp_diff)
                
                # Apply fold change threshold
                for pred in pred_diffs:
                    log_fc_cutoff = l2fc / np.log2(np.e)
                    pred[np.abs(pred) < log_fc_cutoff] = 0
                
                # Pairwise tests
                if pairwise_test:
                    for (pred_diff, lp_diff, (lineage_a, lineage_b)) in zip(
                        pred_diffs, lp_diffs, combinations(lineages, 2)
                    ):
                        wald_stat, df, p_value = _wald_test(
                            pred_diff, lp_diff, cov
                        )
                        key = f"between {self.lineage_names[lineage_a]} and {self.lineage_names[lineage_b]}"
                        result[key][var_name] = (wald_stat, df, p_value, np.mean(pred_diff))
                
                # Global test
                if global_test and len(pred_diffs) > 0:
                    all_pred_diff = np.concatenate(pred_diffs)
                    all_lp_diff = np.vstack(lp_diffs)
                    wald_stat, df, p_value = _wald_test(all_pred_diff, all_lp_diff, cov)
                    result["globally"][var_name] = (
                        wald_stat, df, p_value, np.mean(np.concatenate(pred_diffs)))
                        
            except Exception as e:
                print(f"Error processing {var_name}: {str(e)}")
                continue
                
        return self._create_result_dataframe(result)

    def _create_result_dataframe(self, result_dict):
        dfs = []
        for test_type, gene_data in result_dict.items():
            df = pd.DataFrame.from_dict(gene_data, orient='index',
                                        columns=['wald_stat', 'df', 'p_value', 'log_fc'])
            df['test_type'] = test_type
            df['gene'] = df.index
            dfs.append(df)
        
        return pd.concat(dfs).reset_index(drop=True)

