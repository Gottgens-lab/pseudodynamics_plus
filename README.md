
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


## Branches

- `main` — stable release used by the docs and `pip install`.
- `diffusion_improvement` — active development branch with the latest improvements to the diffusion term, new training scripts, and ongoing experiments. Check this branch if you want the most recent features or want to reproduce results from the current manuscript revision.

```bash
git checkout diffusion_improvement
```


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
