import os
import re
import numpy as np
import datetime
import json
from argparse import Namespace
from typing import Dict, Any
from pathlib import Path

# We get the directory containing this file (the 'PINN' package directory).
PACKAGE_DIR = Path(__file__).resolve().parent

# The project root is two levels up from __init__.py
PROJECT_ROOT_DIR = PACKAGE_DIR.parent.parent

# Expose the variable for the user
main_dir = str(PROJECT_ROOT_DIR) if str(PROJECT_ROOT_DIR).endswith('/') else str(PROJECT_ROOT_DIR) + "/"

class ExperimentConfig:
    """Records and manages experiment configuration settings."""
    
    def __init__(self,  config: str=None,  args: Namespace = None, model = None):
        """
        Args:
            args: Parsed command-line arguments
            config: Experiment directory path
            model: Initialized model instance
        """
    

        if (args is not None) and (model is not None):
            self.run_date = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.experiment_config = self._get_experiment_config(args)
            self.dataset_config = self._get_dataset_config(args)
            self.model_config = self._get_model_config(model)
            self.training_config = self._get_training_config(args)
            self.raw_args = vars(args)

            

        elif os.path.exists(config) and config.endswith('.json'):
            try:
                abs_path = os.path.abspath(config)
                main_dir = abs_path.split("logs/")[0]
            except:
                pass
            self.from_json(config)

        elif config is None:
            pass # do nothing

        else:
            raise ValueError(f"{config} config file not Found, args and model can not be empty for new experiment")


    def _get_experiment_config(self, args: Namespace) -> Dict[str, Any]:
        return {
            'dataset': args.dataset,
            'gpu_devices': args.gpu_devices,
            'progress_bar': args.progress_bar,
        }

    def _get_dataset_config(self, args: Namespace) -> Dict[str, Any]:
        return {
            'cellstate_key': args.cellstate_key,
            'n_dimension': args.n_dimension,
            "kde_kws": {"bw_method":None},
            'timepoint_idx': args.timepoint_idx,
            'deltax_key': args.deltax_key,
            'norm_time': args.norm_time,
            'knn_volume' : args.knn_volume,
        } 

    def _get_model_config(self, model=None) -> Dict[str, Any]:

        if (model is None) and isinstance(self.raw_args, dict):
            config = {
                'model_class': getattr(self.args , 'model', None),
                'channels': getattr(self.args, 'channels', None),
                'activation_fn': getattr(self.args, 'activation_fn', None),
                'ode_tol': getattr(self.args, 'tol', None),
                'growth_weight': getattr(self.args, 'growth_weight', None),
                'R_weight': getattr(self.args, 'R_weight', None),
                'D_penalty': getattr(self.args, 'D_penalty', None),
                'deltax_weight': getattr(self.args, 'deltax_weight', None),
                'weight_intensity': getattr(self.args, 'weight_intensity', None),
                'time_scale_factor': getattr(self.args, 'time_scale_factor', None),
                'time_sensitive': getattr(self.args, 'time_sensitive', None),
                'v_channels': getattr(self.args, 'v_channels', None),
                'g_channels': getattr(self.args, 'g_channels', None),
                'D_channels': getattr(self.args, 'D_channels', None),
            
            }
        elif model is not None:
            config = {
                'model_class': model.__class__.__name__,
                'channels': getattr(model, 'channels', None),
                'activation_fn': getattr(model, 'activation_fn', None),
                'ode_tol': getattr(model, 'ode_tol', None),
                'growth_weight': getattr(model, 'growth_weight', None),
                'R_weight': getattr(model, 'R_weight', None),
                'D_penalty': getattr(model, 'D_penalty', None),
                'deltax_weight': getattr(model, 'deltax_weight', None),
                'weight_intensity': getattr(model, 'weight_intensity', None),
                'time_scale_factor': getattr(model, 'time_scale_factor', None),
                'time_sensitive': getattr(model, 'time_sensitive', None),
                'cfm_weight': getattr(model, 'cfm_weight', None),
                'D_var_weight': getattr(model, 'D_var_weight', None),
                'neuralode_weight': getattr(model, 'neuralode_weight', None),
                'v_channels': getattr(model, 'v_channels', None),
                'g_channels': getattr(model, 'g_channels', None),
                'D_channels': getattr(model, 'D_channels', None),
            }
        
        return config

    def _get_training_config(self, args: Namespace) -> Dict[str, Any]:
        return {
            'batch_size': args.batch_size,
            'schedule_lr': args.schedule_lr,
            'lr': args.lr,
            'max_epochs': 300,
            'optimizer': 'Adam',
        }

    def to_dict(self) -> Dict[str, Any]:
        """Returns all configurations as a dictionary."""
        return {
            'run_date': self.run_date,
            'experiment_config': self.experiment_config,
            'dataset_config': self.dataset_config,
            'model_config': self.model_config,
            'training_config': self.training_config,
            'raw_args': self.raw_args,
        }

    def save(self, path: str):
        """Saves configuration to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    def store_attr(self, json_load):
        for k, v in json_load.items():
            self.__setattr__(k, v)
            if v is None:
                self.__setattr__(k, None)
    
    def from_json(self, file_path: str, main_dir: str = main_dir) -> 'ExperimentConfig':
        """Load a saved experiment configuration from JSON file.
        
        Args:
            file_path: Path to the saved JSON configuration file
            
        Returns:
            ExperimentConfig instance with loaded parameters
        """
        with open(file_path, 'r') as f:
            data = json.load(f)

        self.store_attr(data)
        self.raw_args['config'] = file_path

        self.args = Namespace(**self.raw_args)

        if main_dir is not None:
            try:
                for key in ['checkpoint_dir', 'save_dir']:
                    if key in self.experiment_config and "logs/" in self.experiment_config[key]:
                        old_main = self.experiment_config[key].split("logs/")[0]
                        if old_main != "":
                            self.experiment_config[key] = self.experiment_config[key].replace(old_main, main_dir)
                        else:
                            # If the path already starts with logs/, just prepend main_dir
                            # os.path.join or simple concatenation will work since main_dir ends with /
                            self.experiment_config[key] = os.path.join(main_dir, self.experiment_config[key])
            except:
                pass

    def find_lastest_ckpt(self):
        """
        Find the ckpt file with the minimum val_loss.

        Fallback: when val_loss was never logged (e.g. cord-blood PCA whose
        val_dataloader is empty), the ModelCheckpoint monitor fires no match and
        only ``last.ckpt`` exists (saved via ``save_last=True``).  In that case
        we return ``last.ckpt`` so eval does not fail with FileNotFoundError.
        """
        # look for ckpt files
        log_dir = os.path.join(self.experiment_config['checkpoint_dir'], 'checkpoints')
        ckpts = [f for f in os.listdir(log_dir) if f.endswith('.ckpt')]

        # Try to match val-loss ckpt names; fall back to last.ckpt
        _pat = re.compile(r"epoch=\d{1,3}-\w{3,10}_loss=([-, \.,\d]{1,30}).ckpt")
        scored = [(f, float(m.group(1))) for f in ckpts if (m := _pat.match(f))]
        if scored:
            # look for min loss
            ckpt_path = os.path.join(log_dir, min(scored, key=lambda x: x[1])[0])
        elif "last.ckpt" in ckpts:
            # val_loss not logged (empty val_dataloader) — use last saved epoch
            ckpt_path = os.path.join(log_dir, "last.ckpt")
        else:
            raise FileNotFoundError(
                f"No val-loss checkpoints and no last.ckpt found in {log_dir}. "
                f"Files: {ckpts}"
            )
        return ckpt_path
    
    def get_args(self):
        return Namespace(**self.raw_args)
            


