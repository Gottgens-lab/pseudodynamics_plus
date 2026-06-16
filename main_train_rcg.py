"""
main_train_rcg.py — clean training script for the RCG (Residual-Centered Growth) PINN.

Key differences from main_train.py:
  - No config-override foot-gun: --config is merged with CLI args; unspecified CLI args
    do NOT clobber config values.
  - --gpu_devices defaults to "0" instead of None.
  - CFM loop runs once (cfm_loops=1 passed to model constructor).
  - Exposes all RCG flags; defaults match the recommended launch config.
  - restrict_D uses D.pow(2).mean() (batch-size invariant) via d_penalty_mode='mean'.
"""

import ast
import os
import sys
import json
import argparse
import numpy as np
import scanpy as sc
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning import callbacks
from argparse import Namespace

import pseudodynamics
from pseudodynamics import models, reader

torch.set_float32_matmul_precision('medium')


# ─── argument parsing ──────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(
        "main_train_rcg: RCG-PINN training (Residual-Centered Growth)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset / IO
    p.add_argument("--dataset",        type=str, default="synthetic_FP_5D_divide_formal",
                   help="h5ad filename stem")
    p.add_argument("--data_dir",       type=str, default=None,
                   help="Explicit directory for h5ad")
    p.add_argument("--cellstate_key",  type=str, default="cellstate")
    p.add_argument("--deltax_key",     type=str, default="delta_x")
    p.add_argument("--log_name",       type=str, default=None)
    p.add_argument("--log_root",       type=str, default=None,
                   help="Root directory for logs (default: <repo>/logs)")
    p.add_argument("--config",         type=str, default=None,
                   help="JSON config; values merged UNDER CLI args (CLI wins on conflict)")
    p.add_argument("--resume_ckpt",    type=str, default=None)

    # GPU / infra
    p.add_argument("--gpu_devices",    type=str, default="0",
                   help="GPU index string, e.g. '0' or '1'")
    p.add_argument("--num_workers",    type=int, default=4)
    p.add_argument("--seed",           type=int, default=None)
    p.add_argument("--progress_bar",   action="store_true", default=True)

    # Data / density
    p.add_argument("--n_dimension",    type=int, default=5)
    p.add_argument("--n_timepoints",   type=int, default=None,
                   help="Set automatically from dataset; override only if needed")
    p.add_argument("--timepoint_idx",  type=str, default=None,
                   help="Slice timepoints; None = use all")
    p.add_argument("--batch_size",     type=int, default=512)
    p.add_argument("--norm_time",      type=str, default="min_minus",
                   choices=["min_minus", "log", "none", "False"])
    p.add_argument("--density_estimator", type=str, default="gmm",
                   choices=["kde", "gmm"])
    p.add_argument("--gmm_k_max",      type=int, default=5)
    p.add_argument("--bw",             type=float, default=None)
    p.add_argument("--knn_volume",     action="store_true", default=False)

    # Model architecture
    p.add_argument("--channels",       type=str, default="64,64",
                   help="Hidden layer widths, comma-separated")
    p.add_argument("--time_sensitive", action="store_true", default=True)
    p.add_argument("--activation_fn",  type=str, default="Tanh")

    # Optimiser
    p.add_argument("--lr",             type=float, default=3e-4)
    p.add_argument("--schedule_lr",    type=str,   default="CyclicLR")
    p.add_argument("--tol",            type=float, default=1e-4)
    p.add_argument("--max_epochs",     type=int, default=600)
    p.add_argument("--time_scale_factor", type=float, default=1.0)

    # Loss weights
    p.add_argument("--R_weight",           type=float, default=3.0,
                   help="RCG loss weight")
    p.add_argument("--growth_weight",      type=float, default=0.0,
                   help="Set to 0 in RCG mode (magnitude anchor is in the target)")
    p.add_argument("--neuralode_weight",   type=float, default=2.0)
    p.add_argument("--cfm_weight",         type=float, default=1.0,
                   help="CFM weight; loop runs once via cfm_loops=1")
    p.add_argument("--deltax_weight",      type=float, default=0.01)
    p.add_argument("--D_penalty",          type=float, default=0.1,
                   help="D regularisation; uses D.pow(2).mean() (batch-size invariant)")
    p.add_argument("--D_var_weight",       type=float, default=0.0)
    p.add_argument("--weight_intensity",   type=float, default=1.0)

    # Growth / residual mode
    p.add_argument("--residual_mode",      type=str, default="rcg",
                   choices=["raw", "ginv", "rcg"])
    p.add_argument("--growth_loss_mode",   type=str, default="logratio",
                   choices=["legacy", "logratio", "massbalance"])
    p.add_argument("--growth_pop_ref",     type=str, default="popmean",
                   choices=["cellsum", "popmean"])
    p.add_argument("--g_init_rate",        type=float, default=2.5,
                   help="Warm-start g_net bias to this value")

    # CFM pairing
    p.add_argument("--cfm_unbalanced_reg_m", type=float, default=0.1,
                   help="Unbalanced-OT reg_m for CFM pairing; 0 = balanced")
    p.add_argument("--D_clip",             type=str, default=None)

    # RCG-specific flags
    p.add_argument("--ema_decay",          type=float, default=0.95)
    p.add_argument("--rcg_warmup_steps",   type=int, default=300,
                   help="Steps before RCG engages; uses logratio fallback before this")
    p.add_argument("--rcg_clip_pct",       type=float, default=0.02,
                   help="Symmetric percentile to clip ginv residual tails")
    p.add_argument("--rcg_u_net_raw_weight", type=float, default=0.05,
                   help="Small raw FP residual weight; keeps u_net on FP in RCG mode")

    return p


def merge_config(args, config_path):
    """
    Load JSON config and fill in args attributes that were NOT provided on the CLI.
    CLI values always win; the config only supplies missing values.
    """
    with open(config_path) as f:
        raw = json.load(f)
    cfg = raw.get("raw_args", raw)
    for k, v in cfg.items():
        if k == "config":
            continue
        if not hasattr(args, k) or getattr(args, k) is None:
            setattr(args, k, v)
    return args


# ─── data loading ─────────────────────────────────────────────────────────────

def find_h5ad(dataset, data_dir=None):
    if data_dir is not None:
        path = os.path.join(data_dir, f"{dataset}.h5ad")
        if os.path.exists(path):
            return path
        raise FileNotFoundError(f"h5ad not found: {path}")
    candidates = [
        os.path.join(os.path.abspath("."), "data", f"{dataset}.h5ad"),
        os.path.join(os.path.abspath("."), f"{dataset}.h5ad"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f"Cannot find {dataset}.h5ad in: {candidates}")


def build_gmm_density_funs(adata, cellstate_key, n_dimension, timepoint_key,
                            pop_t, gmm_k_max):
    from sklearn.mixture import GaussianMixture
    coords = adata.obsm[cellstate_key][:, :n_dimension]
    tp_arr = adata.obs[timepoint_key].values
    density_funs = []
    for t_val in pop_t:
        ct = coords[tp_arr == t_val]
        best_bic, best_gmm = np.inf, None
        for k in range(1, gmm_k_max + 1):
            gmm = GaussianMixture(n_components=k, covariance_type='full',
                                  random_state=42, max_iter=500)
            try:
                gmm.fit(ct)
                bic = gmm.bic(ct)
                if bic < best_bic:
                    best_bic, best_gmm = bic, gmm
            except Exception:
                continue
        def _make_fn(g):
            def fn(query, **_):
                q = query.T if query.shape[0] == coords.shape[1] else query
                return np.exp(g.score_samples(q))
            return fn
        density_funs.append(_make_fn(best_gmm))
        print(f"  GMM t={t_val}: n_cells={ct.shape[0]}, "
              f"k={best_gmm.n_components}, BIC={best_bic:.1f}")
    return density_funs


# ─── model construction ────────────────────────────────────────────────────────

def build_model(args, n_timepoints):
    n_dim = args.n_dimension + 1 if args.time_sensitive else args.n_dimension
    hidden = [int(c) for c in args.channels.split(",")]
    channels = [args.n_dimension + 1] + hidden + [1]

    model_kws = dict(
        v_channels=[n_dim] + hidden + [args.n_dimension],
        g_channels=[n_dim] + hidden + [1],
        D_channels=[n_dim] + hidden + [1],
    )

    if args.residual_mode == 'rcg' and getattr(args, 'growth_pop_ref', 'cellsum') != 'popmean':
        print("[WARNING] residual_mode='rcg' requires growth_pop_ref='popmean'. Overriding.")
        args.growth_pop_ref = 'popmean'

    model = models.pde_params(
        lr=args.lr,
        channels=channels,
        activation_fn=args.activation_fn,
        ode_tol=args.tol,
        D_penalty=args.D_penalty,
        deltax_weight=args.deltax_weight,
        weight_intensity=args.weight_intensity,
        growth_weight=args.growth_weight,
        R_weight=args.R_weight,
        time_scale_factor=args.time_scale_factor,
        time_sensitive=args.time_sensitive,
        cfm_weight=args.cfm_weight,
        D_var_weight=args.D_var_weight,
        neuralode_weight=args.neuralode_weight,
        D_clip=args.D_clip,
        cfm_unbalanced_reg_m=args.cfm_unbalanced_reg_m,
        g_init_rate=args.g_init_rate,
        growth_loss_mode=args.growth_loss_mode,
        growth_pop_ref=args.growth_pop_ref,
        residual_mode=args.residual_mode,
        ema_decay=args.ema_decay,
        rcg_warmup_steps=args.rcg_warmup_steps,
        rcg_clip_pct=args.rcg_clip_pct,
        rcg_u_net_raw_weight=args.rcg_u_net_raw_weight,
        n_timepoints=n_timepoints,
        cfm_loops=1,          # single CFM sample per step (not 10x as in main_train.py)
        d_penalty_mode='mean', # D.pow(2).mean(): batch-size invariant
        **model_kws,
    )
    return model


# ─── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.config is not None:
        args = merge_config(args, args.config)

    if args.seed is not None:
        pl.seed_everything(args.seed, workers=True)

    # ── load data ──────────────────────────────────────────────────────────────
    h5ad_path = find_h5ad(args.dataset, getattr(args, 'data_dir', None))
    print(f"[data] loading {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)

    if args.timepoint_idx is None:
        args.timepoint_idx = len(adata.uns['pop']['t'])
    elif isinstance(args.timepoint_idx, str):
        args.timepoint_idx = ast.literal_eval(args.timepoint_idx)

    log_name = args.log_name or f"{args.dataset}-{args.cellstate_key}_n{args.timepoint_idx}"
    log_root = args.log_root or os.path.join(os.path.dirname(h5ad_path), "..", "logs")
    save_path = os.path.join(os.path.abspath(log_root), log_name, "pde_params_tsense")
    pseudodynamics.tl.make_dir(save_path)

    ds_kws = dict(
        timepoint_idx=args.timepoint_idx,
        n_dimension=args.n_dimension,
        cellstate_key=args.cellstate_key,
        knn_volume=args.knn_volume,
        log_transform=False,
        norm_time=args.norm_time,
        deltax_key=args.deltax_key,
        kde_kws={"bw_method": args.bw},
        batchsize=args.batch_size,
    )

    if args.density_estimator == 'gmm':
        pop_t = adata.uns['pop']['t']
        if args.timepoint_idx is not None and isinstance(args.timepoint_idx, int):
            pop_t = pop_t[:args.timepoint_idx]
        tp_key = 'timepoint_tx_days'
        if tp_key not in adata.obs:
            tp_key = [k for k in adata.obs.keys() if 'timepoint' in k.lower()][0]
        print(f"\n[data] building GMM density (BIC over k=1..{args.gmm_k_max})")
        ds_kws['density_funs'] = build_gmm_density_funs(
            adata, args.cellstate_key, args.n_dimension, tp_key, pop_t, args.gmm_k_max)

    train_DS = reader.TwoTimpepoint_AnnDS(AnnData=adata, split='train', **ds_kws)
    val_DS   = reader.TwoTimpepoint_AnnDS(AnnData=adata, split='val',   **ds_kws)

    n_tp_actual = train_DS.n_timepoint
    if args.n_timepoints is None:
        args.n_timepoints = n_tp_actual
    elif args.n_timepoints != n_tp_actual:
        print(f"[WARNING] --n_timepoints={args.n_timepoints} != dataset n_timepoint={n_tp_actual}. "
              f"Using dataset value.")
        args.n_timepoints = n_tp_actual

    train_DL = DataLoader(train_DS, batch_size=None, num_workers=args.num_workers)
    val_DL   = DataLoader(val_DS,   batch_size=None, num_workers=args.num_workers)

    # ── model ─────────────────────────────────────────────────────────────────
    model = build_model(args, args.n_timepoints)

    print(f"\n[model] residual_mode={model.residual_mode}, "
          f"rcg_warmup_steps={model.rcg_warmup_steps}, "
          f"ema_decay={model.ema_decay}, "
          f"n_timepoints={model.n_timepoints}")
    print(f"[model] R_weight={model.R_weight}, growth_weight={model.growth_weight}, "
          f"cfm_weight={model.cfm_weight}, cfm_loops={model.cfm_loops}, "
          f"D_penalty={model.D_penalty}, d_penalty_mode={model.d_penalty_mode}")

    # ── trainer ───────────────────────────────────────────────────────────────
    gpu_idx = [int(args.gpu_devices)]
    trainer = pl.Trainer(
        enable_progress_bar=args.progress_bar,
        accelerator='gpu',
        devices=gpu_idx,
        default_root_dir=save_path,
        max_epochs=args.max_epochs,
        callbacks=[
            callbacks.ModelCheckpoint(
                filename='{epoch}-{val_loss:.8f}',
                monitor="val_loss", mode="min",
                save_top_k=2, save_last=True,
            )
        ],
    )

    # ── save config ───────────────────────────────────────────────────────────
    version = trainer.logger.version
    config_run = pseudodynamics.ExperimentConfig(args=args, model=model)
    config_run.experiment_config['save_dir'] = save_path
    config_run.experiment_config['version'] = version
    config_run.experiment_config['checkpoint_dir'] = trainer.logger.log_dir
    config_run.save(os.path.join(save_path, f'V{version}_config.json'))

    # ── fit ───────────────────────────────────────────────────────────────────
    trainer.fit(model, train_dataloaders=train_DL, val_dataloaders=val_DL,
                ckpt_path=args.resume_ckpt)


if __name__ == '__main__':
    main()
