import pandas as pd
import numpy as np

from ._base import fitGAM
from ._association_test import AssociationTest

from scipy.stats import pearsonr
from statsmodels.stats.multitest import multipletests
from scipy.interpolate import interp1d

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression


def calculate_mutual_information(
    parameter: np.ndarray,
    predicted_expression: np.ndarray,
    n_neighbors: int = 3,
    random_state: int = 42
) -> float:
    """
    Calculate mutual information between a parameter and predicted expression.
    
    Parameters:
        parameter (np.ndarray): Parameter values (e.g., pseudotime)
        predicted_expression (np.ndarray): Predicted expression values from GAM model
        n_neighbors (int): Number of neighbors to use for MI estimation
        random_state (int): Random state for reproducibility
        
    Returns:
        float: Mutual information value
    """
    # Ensure proper shape
    if parameter.ndim == 1:
        parameter = parameter.reshape(-1, 1)
        
    # Calculate mutual information
    mi = mutual_info_regression(
        parameter,
        predicted_expression,
        n_neighbors=n_neighbors,
        random_state=random_state
    )
    
    return float(mi[0])


def select_top_de_genes(results_df, qval_threshold=0.05):
    """
    Select top differentially expressed genes based on p-value threshold
    
    Parameters:
    results_df: DataFrame with association test results
    qval_threshold: p-value threshold for selecting DE genes
    
    Returns: list of gene indices passing threshold
    """
    # Filter significant genes
    significant = results_df[~results_df['pvalue'].isna()].copy()
    significant['fdr'] = multipletests(significant['pvalue'], method='fdr_bh')[1]
    
    # Sort by Wald statistic
    significant = significant.query("`fdr` <= @qval_threshold").sort_values('waldStat', ascending=False)
    
    return significant

def prepare_expression_data(gam_fit, pseudotime=None):
    """
    Prepare and scale gene expression data for plotting
    
    Parameters:
    gam_fit: fitted GAM models
    pseudotime: optional custom pseudotime points
    
    Returns: DataFrame with scaled expression values
    """
    if pseudotime is None:
        # Create prediction grid
        pseudotime = gam_fit['pseudotime']
        pred_grid = np.linspace(pseudotime.min(), pseudotime.max(), 100)
        pseudotime = pred_grid
    
    
    # Process each gene
    scaler = MinMaxScaler()
    
    Expr_list = []
    gene_symbol = []
    for i, gene_idx in enumerate(gam_fit['gene_names']):
        model_info = gam_fit['models'][gene_idx]
        
        # Get prediction function
        if model_info['model']:
            pred = model_info['pred_func'](pseudotime)
            scaled_pred = scaler.fit_transform(pred.reshape(-1, 1)).flatten()
            Expr_list.append(scaled_pred)
            gene_symbol.append(gam_fit['gene_symbol'][i])
        else:
            continue

    # Build the DataFrame once (bins x genes)
    expr_df = pd.DataFrame(np.stack(Expr_list).T, columns=gene_symbol)
    expr_df.index = pseudotime

    return expr_df