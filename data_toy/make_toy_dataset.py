"""
Build the toy dataset shipped with the repository: data_toy/tom_pos_toy.h5ad

- 5,000 cells sampled from data/tom_pos.h5ad in proportion to the observed time points (seed 0)
- genes: the highly variable genes of tom_pos plus every mouse transcription factor present in the
  raw matrix (SCENIC list, https://resources.aertslab.org/cistarget/tf_lists/allTFs_mm.txt)
- expression: log-normalized values from `adata.raw`, stored in X and raw
- the preprocessing of tutorial 1 (diffusion map, diffusion pseudotime, Palantir multiscale space,
  scaled diffusion map `DM_scaled`, local state changes `Delta_DM`, train/val/test split)

Run from the repository root:  python data_toy/make_toy_dataset.py
"""
import os, sys, time, urllib.request
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
for v in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[v] = '1'          # multi-threaded OpenBLAS crashes inside Palantir on macOS

import numpy as np, pandas as pd, scanpy as sc, anndata as ad, palantir, scipy.sparse as sp
import pseudodynamics as pdp

N_CELLS, SEED = 5000, 0
OUT = 'data_toy/tom_pos_toy.h5ad'
TF_URL = 'https://resources.aertslab.org/cistarget/tf_lists/allTFs_mm.txt'

t0 = time.time()
full = sc.read_h5ad('data/tom_pos.h5ad')
print('loaded', full.shape, flush=True)

# ---- 1. stratified subsample in proportion to the time points ----
rng = np.random.default_rng(SEED)
tp = full.obs['timepoint_tx_days'].astype(int)
counts = tp.value_counts().sort_index()
quota = (counts / counts.sum() * N_CELLS).round().astype(int)
quota.iloc[-1] += N_CELLS - quota.sum()
keep = np.sort(np.concatenate([rng.choice(np.where(tp.values == t)[0], size=int(q), replace=False) for t, q in quota.items()]))
print('cells per time point:', quota.to_dict(), flush=True)

# ---- 2. genes: highly variable genes + mouse TFs present in the raw matrix ----
tf_path = 'data_toy/allTFs_mm.txt'
if not os.path.exists(tf_path):
    urllib.request.urlretrieve(TF_URL, tf_path)
tfs = pd.read_csv(tf_path, sep='\t', names=['g'])['g']
raw_names = full.raw.var_names
genes = pd.Index(full.var_names).union(raw_names[raw_names.isin(tfs)])
gene_idx = raw_names.get_indexer(genes)
print('genes:', len(genes), '| highly variable', full.n_vars, '| TFs added', len(genes) - full.n_vars, flush=True)

# ---- 3. assemble the toy AnnData (log-normalized expression) ----
X = sp.csr_matrix(full.raw.X[keep][:, gene_idx]).astype(np.float32)
obs_cols = [c for c in ['anno_man', 'fine_anno', 'timepoint_tx_days', 'HSCscore', 'batch', 'biosample_id',
                        'S_score', 'G2M_score', 'phase', 'tom'] if c in full.obs]
toy = ad.AnnData(X=X, obs=full.obs.iloc[keep][obs_cols].copy(), var=full.raw.var.iloc[gene_idx][[]].copy())
toy.var['highly_variable'] = toy.var_names.isin(full.var_names)
toy.obsm['X_pca_harmony'] = full.obsm['X_pca_harmony'][keep].astype(np.float32)
toy.obsm['X_umap'] = full.obsm['X_umap'][keep].astype(np.float32)
toy.uns['pop'] = {k: np.asarray(v) for k, v in full.uns['pop'].items()}     # population size per time point
for k in ['anno_man_colors', 'fine_anno_colors']:
    if k in full.uns:
        toy.uns[k] = full.uns[k]
toy.raw = toy.copy()
del full

# ---- 4. tutorial-1 preprocessing on the subset ----
sc.pp.neighbors(toy, n_pcs=30, use_rep='X_pca_harmony')
sc.tl.diffmap(toy, n_comps=10)
toy.uns['iroot'] = int(np.argmax(toy.obs['HSCscore'].values))      # the most HSC-like cell is the root
sc.tl.dpt(toy)
palantir.utils.run_diffusion_maps(toy, pca_key='X_pca_harmony', n_components=30)
palantir.utils.determine_multiscale_space(toy)
DM = np.asarray(toy.obsm['DM_EigenVectors'], dtype=np.float32)[:, 1:]   # drop the trivial constant eigenvector
toy.obsm['DM_scaled'] = ((DM - DM.min(0, keepdims=True)) / (DM.max(0, keepdims=True) - DM.min(0, keepdims=True))).astype(np.float32)
toy.obsm['DM_EigenVectors_multiscaled'] = np.asarray(toy.obsm['DM_EigenVectors_multiscaled'], dtype=np.float32)

delta_DM, _ = pdp.tl.sample_deltax(toy, xkey='DM_scaled', pseudotimekey='dpt_pseudotime', repeat=10, progressbar=False)
toy.obsm['Delta_DM'] = np.stack(delta_DM).mean(axis=1).astype(np.float32)   # (cells, repeats, dims) -> mean over repeats
np.random.seed(SEED)
toy.obs['split'] = pdp.tl.train_test_split_adata(toy, leaveout=[3]).astype(str)

print(toy, flush=True)
toy.write_h5ad(OUT, compression='gzip')
print('written %s: %.1f MB in %.0f s' % (OUT, os.path.getsize(OUT) / 1e6, time.time() - t0))
