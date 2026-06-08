import os, argparse, typing, json
import numpy as np
import pandas as pd
import scanpy as sc

import torch 
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler

from functools import partial
from argparse import Namespace

import pytorch_lightning as pl
from pytorch_lightning import callbacks 
from pytorch_lightning import loggers as pl_loggers

import pseudodynamics
from pseudodynamics import models as models
from pseudodynamics import reader 
from pseudodynamics import functions as fns


torch.set_float32_matmul_precision('medium')



# Update argument parser setup
parser = argparse.ArgumentParser("Training PINN dynamics on mesh-free high dimensional cellstate")
parser.add_argument("--config", type=str, required=False, default=None,
                   help='Path to existing config JSON file (overrides all other arguments)')

# All other arguments made optional
optional_args = parser.add_argument_group('Optional arguments (ignored when using --config)')
optional_args.add_argument("-D", "--dataset", type=str, required=False, default="HSPC_clu7", help='the name of the dataset, can be found under folder data')
optional_args.add_argument("-K", "--cellstate_key", type=str, required=False, default="cellstate", help='the obsm key on which we represent cell and compute density')
optional_args.add_argument("-M", "--model", type=str, required=False, default="pde_params", help='the model class, defined in models.py')
optional_args.add_argument("-W", "--pretrained", type=str, required=False, default=None, help='the path of the pretrained weights')
optional_args.add_argument("-G", "--gpu_devices", required=False, default=None, help='select which gpu devices to use')
optional_args.add_argument("-L",  "--log_name", type=str, required=False, default=None, help='the name of the logging directory')
optional_args.add_argument("--lr", type=float, required=False, default=3e-4, help='the learning rate for training the model')
optional_args.add_argument("--schedule_lr", type=str, required=False, default="CyclicLR", help='LambdaLR if passing a lambda expression, else StepLR')
optional_args.add_argument("--n_grid", type=int, required=False, default=300, help='the number of grid or h to devid the cell state space')
optional_args.add_argument("--n_dimension", type=int, required=False, default=5, help='the number of dimension to used for estimating density')
optional_args.add_argument("--timepoint_idx", type=str, required=False, default=None, help='the number of time point to train the model')
optional_args.add_argument("--knn_volume", required=False, default=False, help='Whether to correct single-cell density estimate with KNN distance')
optional_args.add_argument("--batch_size", type=int, required=False, default=200, help='the number of nearby cell state to include within a minibatch')
optional_args.add_argument("--bw", type=float, required=False, default=None, help='the band width parameter , pass to bw_method for gaussian_kde')
optional_args.add_argument("--tol", type=float, required=False, default=1e-4, help='the tolerance of error , used to control the precision and speed of ode integral')
optional_args.add_argument("--channels", type=str, required=False, default="32,32", help='the depth and width of the hidden layers')
optional_args.add_argument("--D_penalty", type=float, required=False, default=None, help='the weight to regulate the level of D (Diffusion)')
optional_args.add_argument("--deltax_key", type=str, required=False, default="Delta_DM", help='the key to take deltax from adata')
optional_args.add_argument("--deltax_weight", type=float, required=False, default=1e-2, help='the weight used to regularize the similarity of deltax and v')
optional_args.add_argument("--weight_intensity", type=float, required=False, default=None, help='the weight to emphasize the high density cell, > 1 for weighting, <1 for unweighting')
optional_args.add_argument("--R_weight", type=float, required=False, default=None, help='the weight to balance PDE residue loss and the data-related loss')
optional_args.add_argument("--growth_weight", type=float, required=False, default=None, help='the weight to regularize the contribution of growth to overall density gain, greater means harder boundary')
optional_args.add_argument("--time_scale_factor", type=float, required=False, default=5, help='the scale the time for ode')
optional_args.add_argument("--norm_time", type=str, required=False, default=False, help='Ways to normlize the timepoint, [False, min_minus, log, none]')
optional_args.add_argument("--time_sensitive", action="store_true", required=False, help='Whether to include time in behavoir functions')
optional_args.add_argument("--cfm_weight", type=float, required=False, default=None, help='Weight for Conditional Flow Matching velocity loss (0 = disabled)')
optional_args.add_argument("--D_var_weight", type=float, required=False, default=None, help='Weight for diffusion variance-matching + entropy losses (0 = disabled)')
optional_args.add_argument("--neuralode_weight", type=float, required=False, default=None, help='Weight for the Neural-ODE simulation loss (default 2 to preserve previous behaviour)')
optional_args.add_argument("--D_clip", type=str, required=False, default=None, help='Hard-clamp the diffusion field into "lo,hi" (e.g. "-0.05,0.05") in the PDE dynamics; opt-in, default off')
optional_args.add_argument("--cfm_unbalanced_reg_m", type=float, required=False, default=None, help='If set, use unbalanced OT (marginal relaxation reg_m) for the CFM velocity-loss pairing instead of balanced OT; opt-in, default off (balanced)')
optional_args.add_argument("--g_init_rate", type=float, required=False, default=None, help='If set, warm-start the growth net so g(x,t) ~= this value at init (e.g. the population-derived mean rate ln(N_t+1/N_t)/dt); prevents g collapsing to 0. Opt-in, default off (standard init)')
optional_args.add_argument("--growth_loss_mode", type=str, required=False, default='legacy', choices=['legacy', 'logratio', 'massbalance'], help='Growth-loss formulation. "legacy" (DEFAULT) = original .sum()-based loss: loss_fn(mass_gain, predicted_gain)/mass_gain with mass_gain=sum(u_t+1)-sum(u_t) and predicted_gain=sum(g); absolute-scale and batch-size sensitive, and reproduces all runs trained before 2026-06 (caveat: it divides a non-negative loss by the *signed* mass_gain, which flips the gradient sign when the population shrinks). "logratio" (opt-in, scale-free) constrains d ln N/dt = E_p[g], i.e. log(N_t+1/N_t) = 0.5*(E_p[g_t]+E_p[g_t+1])*dt with the density-weighted mean E_p[g]=(g*u).sum()/u.sum(); it depends only on g averaged over cells, not on the absolute scale of the free density net, so g can no longer hide behind a mis-scaled u. "massbalance" (opt-in) matches N_t + integral(g*u_int) to N_t+1 in log space using the pop-scaled neural-ODE density. Pin "logratio"/"massbalance" only for new experiments; keep "legacy" to compare against the existing ablation arms.')
optional_args.add_argument("--growth_pop_ref", type=str, required=False, default='cellsum', choices=['cellsum', 'popmean'], help='Reference for observed N_t+1/N_t in the growth loss: "cellsum" (batch cell-sum ratio, default) or "popmean" (true pop ratio from uns, open-system anchor)')
optional_args.add_argument("--num_workers", type=int, required=False, default=10, help='DataLoader worker processes (default 10)')
optional_args.add_argument("--residual_mode", type=str, required=False, default='raw', choices=['raw', 'ginv'], help='FP residual formulation: "raw" (default, u**2-scaled MSE) or "ginv" (normalize by u -> supervises g via the continuity inversion g=(du/dt+div(vu)-div(D grad u))/u; scale-free, use a moderate R_weight)')
optional_args.add_argument("--density_estimator", type=str, required=False, default="kde", choices=["kde", "gmm"], help='Density estimator for u_obs: "kde" (default, scipy.stats.gaussian_kde) or "gmm" (sklearn GaussianMixture with BIC-selected k)')
optional_args.add_argument("--gmm_k_max", type=int, required=False, default=5, help='Max number of GMM components to test by BIC when --density_estimator=gmm (default 5)')
optional_args.add_argument("--seed", type=int, required=False, default=None, help='Random seed for pl.seed_everything; if None no seed is set')
optional_args.add_argument("--max_epochs", type=int, required=False, default=400, help='maximum training epochs (default 400)')
optional_args.add_argument("--progress_bar", type=str, required=False, default="True", help='whether show progress bar on screen, boolen value, default True')
optional_args.add_argument("--resume_ckpt", type=str, required=False, default=None, help='path to checkpoint to resume training from')

args = parser.parse_args()

# Configuration handling
if args.config:
    # Load configuration and override arguments
    config = pseudodynamics.ExperimentConfig(config=args.config)
    gpu_devices = args.gpu_devices
    seed_cli = args.seed
    log_name_cli = args.log_name
    progress_bar_cli = args.progress_bar
    max_epochs_cli = args.max_epochs
    args = Namespace(**config.raw_args)
    args.gpu_devices = gpu_devices    # otherwise covered by the configged gpu devices
    if seed_cli is not None:
        args.seed = seed_cli
    if log_name_cli is not None:
        args.log_name = log_name_cli
    args.progress_bar = progress_bar_cli
    args.max_epochs = max_epochs_cli
    # config.raw_args['config'] = args.config   # Preserve original config path
else:
    # Validate required arguments
    if args.gpu_devices is None:
        parser.error("The following arguments are required without --config: -G/--gpu_devices")

# ... [Rest of data loading code remains the same] ...

# Save path handling

if getattr(args, 'seed', None) is not None:
    pl.seed_everything(args.seed, workers=True)

path = os.path.abspath(".")
h5_path = os.path.join(path, f'{args.dataset}.h5ad')
# find adata path
if not os.path.exists(h5_path):
    main_path = path
    h5_path = os.path.join(path, f'data/{args.dataset}.h5ad')
else:
    main_path = os.path.dirname(path)

adata = sc.read_h5ad(h5_path)

if args.timepoint_idx is None:
    args.timepoint_idx = len(adata.uns['pop']['t'])
else:
    args.timepoint_idx = eval(args.timepoint_idx) if isinstance(args.timepoint_idx,str) else args.timepoint_idx


if args.log_name:
    log_name = args.log_name
else:
    log_name = f"{args.dataset}-{args.cellstate_key}_n{args.timepoint_idx}"

save_path = os.path.join(main_path, 'logs', log_name, args.model+['','_tsense'][args.time_sensitive])
pseudodynamics.tl.make_dir(save_path)

#####
##      define model
#####

#
model_class = eval(f"models.{args.model}")
hidden_channels = [int(c) for c in args.channels.split(",")]   

# for g v and D
if args.model == "pde_params": 
    n_dim = args.n_dimension + 1 if args.time_sensitive else args.n_dimension
    max_h = max(hidden_channels)
    model_kws = dict(v_channels = [n_dim] + hidden_channels + [args.n_dimension],
                    g_channels = [n_dim] + hidden_channels + [1],
                    D_channels = [n_dim] + hidden_channels + [1],#[args.n_dimension]
                    )
    channels = [args.n_dimension + 1 ] + hidden_channels + [1]
else:
    model_kws = {}

model = model_class(
        lr=args.lr,
        channels = channels,
        activation_fn='Tanh',
        ode_tol = args.tol,
        D_penalty = args.D_penalty,
        deltax_weight = args.deltax_weight,
        weight_intensity = args.weight_intensity,
        growth_weight = args.growth_weight,
        R_weight = args.R_weight,
        time_scale_factor = args.time_scale_factor,
        time_sensitive = args.time_sensitive,
        cfm_weight = getattr(args, 'cfm_weight', None),
        D_var_weight = getattr(args, 'D_var_weight', None),
        neuralode_weight = getattr(args, 'neuralode_weight', None),
        D_clip = getattr(args, 'D_clip', None),
        cfm_unbalanced_reg_m = getattr(args, 'cfm_unbalanced_reg_m', None),
        g_init_rate = getattr(args, 'g_init_rate', None),
        growth_loss_mode = getattr(args, 'growth_loss_mode', 'legacy'),
        growth_pop_ref = getattr(args, 'growth_pop_ref', 'cellsum'),
        residual_mode = getattr(args, 'residual_mode', 'raw'),
        **model_kws
    )

if args.pretrained is not None:
    Pretrain_class = args.pretrained.split("/")[2].replace("_tsense","")
    # Pretrain_class = "u_dt_weight"
    if Pretrain_class == args.model:
        model = model_class.load_from_checkpoint(args.pretrained)
    else:
        # then only u is use
        Pretrain_class = eval(f"models.{Pretrain_class}")
        Pretain_model = Pretrain_class.load_from_checkpoint(args.pretrained, map_location='cpu')
        # inherit the statedict
        state_dict = Pretain_model.model.state_dict()
        model.u.load_state_dict(state_dict)



##############################
##      define dataset      ##
##############################

ds_kws = dict(  timepoint_idx = args.timepoint_idx,
                n_dimension = args.n_dimension,
                cellstate_key=args.cellstate_key,  #'DM_EigenVector'
                knn_volume = eval(args.knn_volume) if isinstance(args.knn_volume, str) else args.knn_volume,
                log_transform=False,
                norm_time=args.norm_time,
                deltax_key=args.deltax_key,
                kde_kws = {"bw_method":args.bw},
                batchsize=args.batch_size
            )

# Optional GMM density (opt-in via --density_estimator=gmm; default kde is unchanged)
if getattr(args, 'density_estimator', 'kde') == 'gmm':
    from sklearn.mixture import GaussianMixture
    import numpy as _np
    print(f"\n[main_train] Building GMM density estimators (BIC over k=1..{args.gmm_k_max})")
    _coords = adata.obsm[args.cellstate_key][:, :args.n_dimension]
    _tp_arr = adata.obs[ds_kws.get('timepoint_key', 'timepoint_tx_days')].values \
              if 'timepoint_key' in ds_kws else adata.obs['timepoint_tx_days'].values
    _density_funs = []
    for _t in sorted(set(_tp_arr)):
        _ct = _coords[_tp_arr == _t]
        _best_bic = _np.inf; _best_gmm = None
        for _k in range(1, args.gmm_k_max + 1):
            _gmm = GaussianMixture(n_components=_k, covariance_type='full',
                                    random_state=42, max_iter=500)
            try:
                _gmm.fit(_ct)
                _bic = _gmm.bic(_ct)
                if _bic < _best_bic:
                    _best_bic = _bic; _best_gmm = _gmm
            except Exception:
                continue
        def _make_fn(_g):
            def _fn(query, **_):
                q = query.T if query.shape[0] == _coords.shape[1] else query
                return _np.exp(_g.score_samples(q))
            return _fn
        _density_funs.append(_make_fn(_best_gmm))
        print(f"  t={_t}: n_cells={(_tp_arr == _t).sum()}, BIC-selected k={_best_gmm.n_components}, BIC={_best_bic:.1f}")
    ds_kws['density_funs'] = _density_funs

train_DS = reader.TwoTimpepoint_AnnDS(AnnData=adata, split='train', **ds_kws)
val_DS = reader.TwoTimpepoint_AnnDS(AnnData=adata, split='val', **ds_kws)
_nw = getattr(args, 'num_workers', 10)
train_DL = DataLoader(train_DS, batch_size=None, num_workers=_nw)
val_DL = DataLoader(val_DS, batch_size=None, num_workers=_nw)
##############################
##      set up trainer      ##
##############################

device = 'gpu' if torch.cuda.is_available() else 'cpu'
device = 'cpu' if args.gpu_devices == None else 'gpu'
gpu_device = args.gpu_devices if args.gpu_devices == None else [int(args.gpu_devices)]

trainer = pl.Trainer(
                    #auto_lr_find=True,
                    enable_progress_bar=args.progress_bar,
                    accelerator=device,
                    # fast_dev_run=True,
                    # gradient_clip_val=0.5,
                    default_root_dir=save_path,
                    devices = gpu_device,
                    max_epochs=getattr(args, 'max_epochs', 400),
                    # save_last=True guarantees a checkpoint even when val_dataloader
                    # is empty (cord-blood PCA: val cells × 30 dims don't fill a
                    # batch_size=1024 batch, so val_loss is never logged and
                    # monitor="val_loss" would save nothing).  Klein is unaffected —
                    # save_last adds a "last.ckpt" alongside the two best-val checkpoints.
                    callbacks=[callbacks.ModelCheckpoint(filename='{epoch}-{val_loss:.8f}',
                                                monitor="val_loss", mode="min", save_top_k=2,
                                                save_last=True)]
                    )


##############################
##      save config         ##
##############################

# Create and save config if not loading from existing
version = trainer.logger.version

config_run = pseudodynamics.ExperimentConfig(args=args, model=model)
config_run.experiment_config['save_dir'] = save_path
config_run.experiment_config['version'] = version
config_run.experiment_config['checkpoint_dir'] = trainer.logger.log_dir
config_run.save(os.path.join(save_path, f'V{version}_config.json'))



trainer.fit(model, train_dataloaders=train_DL, val_dataloaders=val_DL,
            ckpt_path=getattr(args, 'resume_ckpt', None))
