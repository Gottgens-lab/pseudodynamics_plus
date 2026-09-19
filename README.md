
# Pseudodynamics+
Physics Informed Neural Network based method for solving the single-cell population dynamics.
For each cell, we estimate the dynamic parameter of the cell proliferation, differentiation and diffusion.

<img src="https://raw.githubusercontent.com/Gottgens-lab/pseudodynamics_plus/main/.github/images/pdyn_logo.jpg" alt="pseudodynamics+" width="600"/>


## Documentation and tutorials

Full documentation, API reference, and step-by-step tutorials live on Read the Docs:

- **Website**: https://pseudodynamics-plus.readthedocs.io
- **Tutorials** (data preparation, config setup, downstream analysis): https://pseudodynamics-plus.readthedocs.io/en/latest/tutorials.html
- **Config example**: https://pseudodynamics-plus.readthedocs.io/en/latest/notebooks/setup_config.html

If you are getting started, read the tutorials first — they cover the input format and the most common pitfalls.


## Installation
```bash
git clone https://github.com/Gottgens-lab/pseudodynamics_plus.git
cd pseudodynamics_plus
pip install -e .
```


## Training
Store the population size information in `AnnData.uns['pop']` and save as `h5ad`. Configure the training settings in `config.json` (see the [config tutorial](https://pseudodynamics-plus.readthedocs.io/en/latest/notebooks/setup_config.html) or any `V0_config.json` under `logs/`). Then run:

```bash
# with GPU
python main_train.py --config config.json -G 0

# without GPU
python main_train.py --config config.json -G None
```

All opt-in training flags and their defaults are documented on the [Training options](https://pseudodynamics-plus.readthedocs.io/en/latest/training_options.html) page.

### Toy dataset
`data_toy/tom_pos_toy.h5ad` is a 5,000-cell subsample of the Tom+ haematopoiesis dataset (6,157 genes: the
highly variable genes plus all mouse transcription factors) with every field that the tutorials need. It
trains on a laptop CPU in about 14 minutes, and tutorials 2-5 run on it end to end using the checkpoint
shipped under `logs/tom_pos_toy/`:

```bash
python main_train.py -D data_toy/tom_pos_toy -K DM_scaled --n_dimension 10 --channels 64,64 \
    --batch_size 50 --bw 0.5 --timepoint_idx "[0,1,2,3,4,6,8]" --time_sensitive \
    --max_epochs 200 --seed 0 -G None -L tom_pos_toy
```

`data_toy/make_toy_dataset.py` rebuilds the toy file from the full `data/tom_pos.h5ad`.


## Branches

- `main` — stable release used by the docs and `pip install`.
- `diffusion_improvement` — active development branch with the latest improvements to the diffusion term, new training scripts, and ongoing experiments. Check this branch if you want the most recent features or want to reproduce results from the current manuscript revision.

```bash
git checkout diffusion_improvement
```


## Updates

Reverse-chronological log of the latest functionality on `diffusion_improvement`.

### 2026-06-16 — `main_train_rcg.py`: clean RCG training entry point

A dedicated training script for the **RCG (Residual-Centered Growth)** PINN, kept separate from
`main_train.py` so RCG runs launch without the original script's config-override foot-guns:

- `--config` is **merged** with CLI args — unspecified CLI flags no longer clobber config values.
- Exposes every RCG flag and ships defaults that match the recommended launch config
  (`--residual_mode` defaults to `rcg`, `--growth_loss_mode` to `logratio`).
- Runs the CFM loop once (`cfm_loops=1`) and uses the batch-size-invariant diffusion penalty
  (`d_penalty_mode='mean'`, i.e. `D.pow(2).mean()`).
- `--gpu_devices` defaults to `"0"`.

```bash
python main_train_rcg.py --dataset <h5ad_stem> --config logs/<exp>/V0_config.json
```

The RCG residual mode itself (`residual_mode='rcg'` — mean-corrected `ginv` target plus a
`relmass` magnitude anchor) is documented in the *Constructor-only knobs* table on the [Training options](https://pseudodynamics-plus.readthedocs.io/en/latest/training_options.html) page.


## Reproducible configs and checkpoints

The `logs/` directory contains the exact configs and trained checkpoints for every experiment in the manuscript and rebuttals (Klein, Tom, cord blood, synthetic Fokker–Planck, ablations, etc.). Each experiment directory follows the same layout:

```
logs/<experiment_name>/
├── V0_config.json         # training config used to reproduce the run
└── lightning_logs/        # PyTorch Lightning checkpoints and metrics
```

To reproduce a run, point `main_train.py` at the stored config:

```bash
python main_train.py --config logs/<experiment_name>/V0_config.json -G 0
```
