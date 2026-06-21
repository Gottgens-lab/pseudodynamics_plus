
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
python main_train.py --config_path config.json -G 0

# without GPU
python main_train.py --config_path config.json -G None
```


## Training options (diffusion_improvement features)

The `diffusion_improvement` branch adds several opt-in training features on top of the base
PINN loss (log-density reconstruction + Fokker–Planck residual + Neural-ODE simulation). Every
flag below defaults to the **pre-2026-06 behaviour**, so a run launched without any of them
reproduces the original model. Pass a flag only to turn its feature on.

### Growth loss

| Flag | Default | Choices / type | What it does |
| --- | --- | --- | --- |
| `--growth_loss_mode` | `legacy` | `legacy` \| `logratio` \| `massbalance` | How the growth field `g` is constrained against observed population change. **`legacy`** (default, reproduces earlier runs): `loss_fn(mass_gain, predicted_gain)/mass_gain` with `mass_gain = Σu_{t+1} − Σu_t`, `predicted_gain = Σg`; absolute-scale and batch-size sensitive. Known caveat: dividing a non-negative loss by the *signed* `mass_gain` flips the gradient sign when the population shrinks. **`logratio`** (scale-free): constrains `d ln N/dt = E_p[g]`, i.e. `log(N_{t+1}/N_t) = ½(E_p[g_t]+E_p[g_{t+1}])·dt` with density-weighted mean `E_p[g] = (g·u).sum()/u.sum()`; depends only on `g` averaged over cells, so `g` cannot hide behind a mis-scaled density net. **`massbalance`**: matches `N_t + ∫∫ g·u dt` to `N_{t+1}` in log space using the pop-scaled neural-ODE density. |
| `--growth_pop_ref` | `cellsum` | `cellsum` \| `popmean` | Reference for the observed `N_{t+1}/N_t` ratio used by the growth loss. `cellsum` = batch cell-sum ratio (self-normalising); `popmean` = true population ratio carried in `AnnData.uns['pop']`/`relmass` (open-system anchor). |
| `--g_init_rate` | off (`None`) | float | Warm-start the growth net so `g(x,t) ≈ this value` at init (e.g. `ln(N_{t+1}/N_t)/dt`); prevents `g` collapsing to 0. |

### Fokker–Planck residual

| Flag | Default | Choices / type | What it does |
| --- | --- | --- | --- |
| `--residual_mode` | `raw` | `raw` \| `ginv` | FP residual formulation. `raw` = `u²`-scaled MSE (original). `ginv` = normalize the residual by `u` so it supervises `g` directly via the continuity inversion `g = (∂u/∂t + ∇·(vu) − ∇·(D∇u))/u` (scale-free; use a moderate `--R_weight`). |
| `--R_weight` | model default `1` | float | Weight balancing the PDE residual loss against the data-fit loss. |

### Diffusion field `D`

| Flag | Default | Choices / type | What it does |
| --- | --- | --- | --- |
| `--D_penalty` | model default `0.1` | float | Weight (λ_D) of the L2 penalty on `D`; larger → smaller `|D|`. Drives the magnitude of the learned diffusion field (measured `|D| ∝ λ_D^≈−1.2` on Klein). |
| `--D_var_weight` | off (`0`) | float | Weight for the diffusion variance-matching + entropy losses (encourage `D` to track local expression variance / density-change entropy). `0` disables both. |
| `--D_clip` | off (`None`) | `"lo,hi"` | Hard-clamp the diffusion field into `(lo, hi)` wherever `D` enters the PDE dynamics (e.g. `"-0.05,0.05"`). |

### Velocity / optimal transport (CFM)

| Flag | Default | Choices / type | What it does |
| --- | --- | --- | --- |
| `--cfm_weight` | off (`0`) | float | Weight for the Conditional Flow Matching velocity loss (supervises `v` by OT-paired endpoints). |
| `--cfm_unbalanced_reg_m` | off (`None` = balanced) | float | If set, use **unbalanced** OT (marginal-relaxation `reg_m`) for the CFM pairing instead of balanced OT. Helps when population growth makes balanced coupling force spurious pairings. |
| `--neuralode_weight` | model default `2` | float | Weight of the Neural-ODE density-simulation loss. |
| `--deltax_weight` | `1e-2` | float | Weight (λ_v) supervising `v` against the RNA-velocity-style `deltax` (set to 0 under the OT assumption). |

### Density estimation

| Flag | Default | Choices / type | What it does |
| --- | --- | --- | --- |
| `--density_estimator` | `kde` | `kde` \| `gmm` | Estimator for the observed density `u_obs`. `kde` = `scipy.stats.gaussian_kde`; `gmm` = `sklearn` GaussianMixture with BIC-selected components. |
| `--gmm_k_max` | `5` | int | Max number of GMM components tested by BIC when `--density_estimator gmm`. |

### Constructor-only knobs (not yet exposed on the CLI)

These are `pde_params(...)` arguments with no `main_train.py` flag; they take their class default
unless the model is constructed directly. Listed here because they affect training and are **not**
recorded in `V0_config.json`:

| Argument | Default | Choices / type | What it does |
| --- | --- | --- | --- |
| `d_penalty_mode` | `legacy` | `legacy` \| `mean` | `legacy` = `‖D‖₂.sum()` (batch-size dependent); `mean` = `D².mean()` (batch-size invariant). |
| `residual_mode='rcg'` | — | — | Residual-Centered Growth: replaces the R-loss with a mean-corrected `ginv` target + a `relmass` magnitude anchor. The class accepts `rcg` even though `--residual_mode` only lists `raw`/`ginv`. |
| `ema_decay`, `rcg_warmup_steps`, `rcg_clip_pct`, `rcg_u_net_raw_weight` | `0.95`, `300`, `0.02`, `0.05` | float/int | RCG hyperparameters; active only when `residual_mode='rcg'`. The EMA buffers (`_Eg_resid_ema`, `_ema_initialized`) are registered unconditionally, so checkpoints trained before this feature require `strict=False` to load. |

> **Reproducibility note.** Training flags above are not all persisted into `logs/<exp>/V0_config.json`
> (only the historical weight knobs are). To reproduce a run exactly, record the full command line, or
> pin the flags explicitly. For a clean ablation that varies only one parameter, fix every other flag
> to the same value across arms.

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
`relmass` magnitude anchor) is documented in the *Constructor-only knobs* table above.


## Reproducible configs and checkpoints

The `logs/` directory contains the exact configs and trained checkpoints for every experiment in the manuscript and rebuttals (Klein, Tom, cord blood, synthetic Fokker–Planck, ablations, etc.). Each experiment directory follows the same layout:

```
logs/<experiment_name>/
├── V0_config.json         # training config used to reproduce the run
└── lightning_logs/        # PyTorch Lightning checkpoints and metrics
```

To reproduce a run, point `main_train.py` at the stored config:

```bash
python main_train.py --config_path logs/<experiment_name>/V0_config.json -G 0
```
