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

train_DS = reader.TwoTimpepoint_AnnDS(AnnData=adata, split='train', **ds_kws)
val_DS = reader.TwoTimpepoint_AnnDS(AnnData=adata, split='val', **ds_kws)
train_DL = DataLoader(train_DS, batch_size=None, num_workers=10)
val_DL = DataLoader(val_DS, batch_size=None, num_workers=10)
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
                    callbacks=[callbacks.ModelCheckpoint(filename='{epoch}-{val_loss:.8f}',
                                                monitor="val_loss", mode="min", save_top_k=2)]
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
