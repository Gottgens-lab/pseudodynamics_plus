"""
fate_eval_pipeline.py — Method-agnostic per-cell fate accuracy evaluation
==========================================================================

A general pipeline for evaluating how well any trajectory inference model
predicts the dominant terminal fate of individual starting cells, using
clone-traced ground-truth proportions (F_obs).

Design
------
The two core functions accept `model` and `simulation_func` as arguments,
making the pipeline method-agnostic.  `simulation_func` acts as a decorator
that wraps method-specific forward simulation behind a common interface:

    simulation_func(start_cells, model, n_sims, device) -> np.ndarray

         start_cells : ndarray (n_cells, n_dims)
         model       : any — passed through unchanged from the caller
         n_sims      : int  — trajectories per start cell (stochastic methods)
         device      : str  — torch device string, e.g. 'cuda:0'
         returns     : ndarray
                         (n_cells, n_sims, n_dims)  stochastic — one row of
                                                    n_sims endpoint positions
                                                    per start cell
                         (n_cells, n_dims)           deterministic — a single
                                                    endpoint per start cell;
                                                    n_sims is ignored

The same simulation_func factory pattern lets each 03_evaluate.py build its
own function once (capturing model-specific state) and pass it in:

    sim_fn = make_prescient_sim_fn()
    result = run_fate_evaluation(..., model=(net, config), simulation_func=sim_fn)

Public API
----------
  predict_fates_per_cell(start_cells, model, simulation_func,
                         knn, cell_types, n_sims, device)
      -> np.ndarray (n_cells, n_cell_types)

  compute_fate_accuracy(F_obs_df, F_hat_df)
      -> dict(accuracy, pearson_r, pearson_p, y_true, y_pred)

  run_fate_evaluation(start_cells, start_cell_ids, model, simulation_func,
                      F_obs, x_ref, y_ref,
                      cell_types=None, n_sims=100, k=20, device='cpu')
      -> dict(accuracy, pearson_r, pearson_p, n_start_cells,
              F_hat, F_obs_aligned, y_true, y_pred)

Provided simulation_func factories
------------------------------------
  make_prescient_sim_fn(num_steps=None)
      model = (net, config)           stochastic SDE (PRESCIENT)

  make_tigon_sim_fn(t_start, t_end)
      model = func                    ODE via TorchDiffEqPack (TIGON)

  make_trajectorynet_sim_fn(int_tp_start, int_tp_end)
      model = cnf_model               CNF (TrajectoryNet)

Quick example
-------------
  from scripts.fate_eval_pipeline import run_fate_evaluation, make_prescient_sim_fn
  import pandas as pd

  F_obs = pd.read_csv("data/klein/F_obs.csv", index_col=0)
  F_obs.index = F_obs.index.astype(str)

  result = run_fate_evaluation(
      start_cells     = x_start,          # (n_cells, n_dims) float32
      start_cell_ids  = start_barcodes,   # list[str]
      model           = (net, config),
      simulation_func = make_prescient_sim_fn(),
      F_obs           = F_obs,
      x_ref           = x_train,          # (n_ref, n_dims) float32
      y_ref           = y_train,          # (n_ref,) str
      n_sims          = 100,
      k               = 20,
      device          = "cuda:0",
  )
  print(f"Accuracy : {result['accuracy']:.4f}")
  print(f"Pearson r: {result['pearson_r']:.4f}")
  # result['F_hat']         DataFrame (n_cells × n_cell_types)
  # result['F_obs_aligned'] DataFrame (n_cells × n_cell_types)
"""

import logging
import numpy as np
import pandas as pd
import sklearn.metrics
from scipy.stats import pearsonr
from sklearn.neighbors import KNeighborsClassifier

log = logging.getLogger(__name__)


# ─── Core: predict fates ─────────────────────────────────────────────────────

def predict_fates_per_cell(
    start_cells: np.ndarray,
    model,
    simulation_func,
    knn: KNeighborsClassifier,
    cell_types: list,
    n_sims: int = 100,
    device: str = "cpu",
) -> np.ndarray:
    """
    Predict fate distribution for each start cell via simulation + KNN.

    Parameters
    ----------
    start_cells     : ndarray (n_cells, n_dims) — initial cell positions.
    model           : any model object, passed through to simulation_func.
    simulation_func : callable with signature
                        (start_cells, model, n_sims, device) -> ndarray
                      Returns (n_cells, n_sims, n_dims) for stochastic models,
                      or (n_cells, n_dims) for deterministic models.
    knn             : fitted KNeighborsClassifier mapping embedding → cell type.
    cell_types      : ordered list of cell-type strings (defines column order).
    n_sims          : trajectories per start cell (passed to simulation_func).
    device          : torch device string.

    Returns
    -------
    F_hat_arr : ndarray (n_cells, n_cell_types), rows sum to 1.
    """
    log.info(
        f"  simulation_func: {len(start_cells)} cells"
        f" × {n_sims} sims ..."
    )
    endpoints = np.asarray(simulation_func(start_cells, model, n_sims, device))

    n_cells  = len(start_cells)
    ct_index = {ct: i for i, ct in enumerate(cell_types)}

    # Deterministic: (n_cells, n_dims) → treat as a single sim per cell
    if endpoints.ndim == 2:
        endpoints = endpoints[:, np.newaxis, :]   # (n_cells, 1, n_dims)
    # Now guaranteed: (n_cells, n_sims_actual, n_dims)
    actual_sims = endpoints.shape[1]

    # Batch KNN: flatten → predict → reshape
    flat_pts   = endpoints.reshape(-1, endpoints.shape[-1])  # (n_cells*sims, n_dims)
    all_labels = knn.predict(flat_pts).reshape(n_cells, actual_sims)

    F_hat_arr = np.zeros((n_cells, len(cell_types)), dtype=np.float32)
    for i in range(n_cells):
        uniq, counts = np.unique(all_labels[i], return_counts=True)
        for ct, cnt in zip(uniq, counts):
            if ct in ct_index:
                F_hat_arr[i, ct_index[ct]] = cnt / actual_sims

    return F_hat_arr


# ─── Accuracy metric ─────────────────────────────────────────────────────────

def compute_fate_accuracy(
    F_obs_df: pd.DataFrame,
    F_hat_df: pd.DataFrame,
) -> dict:
    """
    Compare predicted vs. ground-truth per-cell fate distributions.

    Accuracy is the fraction of cells where the predicted dominant fate
    (argmax of F_hat row) matches the ground-truth dominant fate
    (argmax of F_obs row).  Cells with all-zero F_hat are assigned
    "Undifferentiated".

    Pearson r is computed over per-cell-type mean proportions
    (population-level distribution similarity).

    Parameters
    ----------
    F_obs_df : DataFrame (n_cells, n_cell_types) — ground-truth proportions.
    F_hat_df : DataFrame (n_cells, n_cell_types) — predicted proportions.
               Must share the same index as F_obs_df.

    Returns
    -------
    dict with keys: accuracy, pearson_r, pearson_p, y_true, y_pred.
    """
    shared_cols = sorted(set(F_obs_df.columns) & set(F_hat_df.columns))
    if not shared_cols:
        raise ValueError("F_obs and F_hat share no cell-type columns.")

    obs = F_obs_df[shared_cols]
    hat = F_hat_df[shared_cols]

    y_true = obs.idxmax(axis=1).tolist()
    y_pred = hat.idxmax(axis=1).tolist()

    # Assign "Undifferentiated" to cells where no fate was predicted
    zero_idx = np.where(hat.sum(axis=1).values == 0)[0]
    for idx in zero_idx:
        y_pred[idx] = "Undifferentiated"

    accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)

    obs_mean = obs.mean(axis=0).values
    hat_mean = hat.mean(axis=0).values
    if obs_mean.std() < 1e-10 or hat_mean.std() < 1e-10:
        r, p = 0.0, 1.0
    else:
        r, p = pearsonr(obs_mean, hat_mean)

    return dict(accuracy=accuracy, pearson_r=r, pearson_p=p,
                y_true=y_true, y_pred=y_pred)


# ─── End-to-end pipeline ─────────────────────────────────────────────────────

def run_fate_evaluation(
    start_cells: np.ndarray,
    start_cell_ids,
    model,
    simulation_func,
    F_obs: pd.DataFrame,
    x_ref: np.ndarray,
    y_ref: np.ndarray,
    cell_types: list = None,
    n_sims: int = 100,
    k: int = 20,
    device: str = "cpu",
) -> dict:
    """
    End-to-end fate accuracy evaluation for any trajectory model.

    Ties together: (1) alignment of start cells to F_obs ground truth,
    (2) KNN fitting on reference data, (3) fate prediction via
    predict_fates_per_cell, (4) accuracy/Pearson-r via compute_fate_accuracy.

    Parameters
    ----------
    start_cells     : ndarray (n_cells, n_dims) — embedding of start cells.
    start_cell_ids  : array-like of str — barcodes for each start cell,
                      must align row-for-row with start_cells.
    model           : model object, passed through to simulation_func.
    simulation_func : callable — see predict_fates_per_cell docstring.
    F_obs           : DataFrame (cells × cell_types) — ground-truth fate
                      proportions indexed by cell barcode.
    x_ref           : ndarray (n_ref, n_dims) — reference embedding for KNN.
    y_ref           : ndarray (n_ref,) str — reference cell-type labels.
    cell_types      : list of cell-type names; defaults to sorted(F_obs.columns).
    n_sims          : trajectories per start cell.
    k               : KNN neighbours for cell-type assignment.
    device          : torch device string.

    Returns
    -------
    dict with keys:
        accuracy        float
        pearson_r       float
        pearson_p       float
        n_start_cells   int — cells actually evaluated (intersection with F_obs)
        F_hat           DataFrame (n_cells × n_cell_types)
        F_obs_aligned   DataFrame (n_cells × n_cell_types, F_obs subset)
        y_true          list[str]
        y_pred          list[str]
    """
    if cell_types is None:
        cell_types = sorted(F_obs.columns.tolist())

    # ── Align start cells to F_obs ────────────────────────────────────────
    ids_str  = np.array([str(i) for i in start_cell_ids])
    fobs_str = F_obs.index.astype(str)
    in_fobs  = pd.Series(ids_str).isin(fobs_str).values

    n_dropped = int((~in_fobs).sum())
    if n_dropped:
        log.warning(
            f"  {n_dropped}/{len(ids_str)} start cells not found in F_obs "
            "— dropped."
        )
    if not in_fobs.any():
        raise ValueError(
            "None of start_cell_ids found in F_obs.index.\n"
            f"  start_cell_ids sample : {ids_str[:5].tolist()}\n"
            f"  F_obs.index sample    : {list(fobs_str[:5])}"
        )

    cells_filt = start_cells[in_fobs]
    ids_filt   = ids_str[in_fobs]
    try:
        F_obs_aln  = F_obs.loc[ids_filt].reindex(columns=cell_types, fill_value=0.0)
    except KeyError:
        F_obs.index = ids_str
        F_obs_aln  = F_obs.loc[ids_filt].reindex(columns=cell_types, fill_value=0.0)
    log.info(f"Start cells matched to F_obs: {len(cells_filt)}")

    # ── KNN on reference data ─────────────────────────────────────────────
    knn = KNeighborsClassifier(n_neighbors=k, metric="euclidean")
    knn.fit(x_ref, y_ref)
    log.info(f"KNN fitted: {len(x_ref)} ref cells, k={k}")

    # ── Predict F_hat ─────────────────────────────────────────────────────
    F_hat_arr = predict_fates_per_cell(
        start_cells     = cells_filt,
        model           = model,
        simulation_func = simulation_func,
        knn             = knn,
        cell_types      = cell_types,
        n_sims          = n_sims,
        device          = device,
    )
    F_hat = pd.DataFrame(F_hat_arr, index=ids_filt, columns=cell_types)

    # ── Compute accuracy ──────────────────────────────────────────────────
    result = compute_fate_accuracy(F_obs_aln, F_hat)
    result.update(
        n_start_cells = len(cells_filt),
        F_hat         = F_hat,
        F_obs_aligned = F_obs_aln,
    )
    log.info(
        f"Accuracy: {result['accuracy']:.4f}   "
        f"Pearson r: {result['pearson_r']:.4f}"
    )
    return result


# ─── simulation_func factories ────────────────────────────────────────────────

def make_prescient_sim_fn(num_steps: int = None):
    """
    Return a simulation_func for PRESCIENT.

    Usage
    -----
        model = (net, config)   # net: AutoGenerator, config: SimpleNamespace
        sim_fn = make_prescient_sim_fn()
        # or override steps: make_prescient_sim_fn(num_steps=20)

        endpoints = sim_fn(start_cells, model, n_sims=100, device="cuda:0")
        # returns ndarray (n_cells, n_sims, n_dims)

    Notes
    -----
    All n_cells × n_sims copies are batched into one forward pass for
    efficiency.  num_steps is auto-calculated from config if not given:
        (config.train_t[-1] - config.start_t) / config.train_dt
    """
    import torch

    def _simulate(start_cells, model, n_sims, device):
        net, config = model
        steps = num_steps or int(
            (config.train_t[-1] - config.start_t) / config.train_dt
        )
        n_cells, n_dims = start_cells.shape

        # Expand each cell n_sims times → (n_cells * n_sims, n_dims)
        x = torch.tensor(
            np.repeat(start_cells, n_sims, axis=0), dtype=torch.float32
        ).to(device)

        net.eval()
        with torch.no_grad():
            for _ in range(steps):
                z = torch.randn_like(x) * config.train_sd
                x = net._step(x, dt=config.train_dt, z=z)

        return x.cpu().numpy().reshape(n_cells, n_sims, n_dims)

    return _simulate


def make_tigon_sim_fn(t_start: float, t_end: float):
    """
    Return a simulation_func for TIGON.

    Usage
    -----
        model = func           # TorchDiffEqPack UOT model
        sim_fn = make_tigon_sim_fn(t_start=0.0, t_end=1.0)

        endpoints = sim_fn(start_cells, model, n_sims=1, device="cuda:0")
        # returns ndarray (n_cells, n_dims)  — deterministic ODE

    Notes
    -----
    TIGON is a deterministic ODE so n_sims is ignored (returns one endpoint
    per start cell).  Run run_fate_evaluation with n_sims=1 or any value;
    the pipeline handles the (n_cells, n_dims) output automatically.
    """
    import torch
    from TorchDiffEqPack import odesolve

    def _simulate(start_cells, model, n_sims, device):
        func = model
        z = torch.tensor(start_cells, dtype=torch.float32,
                         requires_grad=True).to(device)
        g0    = torch.zeros(z.shape[0], 1, dtype=torch.float32, device=device)
        logp0 = torch.zeros(z.shape[0], 1, dtype=torch.float32, device=device)

        options = {
            "method": "Dopri5", "h": None,
            "rtol": 1e-3, "atol": 1e-5,
            "print_neval": False, "neval_max": 1_000_000,
            "safety": None, "t0": t_start, "t1": t_end,
        }
        with torch.no_grad():
            z_out, _, _ = odesolve(func, y0=(z, g0, logp0), options=options)

        return z_out.detach().cpu().numpy()   # (n_cells, n_dims)

    return _simulate


def make_pseudodynamics_sim_fn(
    t_start_norm: float,
    t_end_norm: float,
    tol: float = 1e-5,
):
    """
    Return a simulation_func for pseudodynamics+.

    Usage
    -----
        model = pde_model      # pseudodynamics PDE model with
                               #   .v(state, t_in) and .time_scale_factor
        sim_fn = make_pseudodynamics_sim_fn(t_start_norm=0.0, t_end_norm=4.0)

        endpoints = sim_fn(start_cells, model, n_sims=1, device="cuda:0")
        # returns ndarray (n_cells, n_dims)  — deterministic ODE

    Notes
    -----
    Pseudodynamics+ uses a deterministic ODE; n_sims is ignored.

    t_start_norm / t_end_norm are in the *human-readable* normalised time
    domain (e.g. for Klein with norm_time='min_minus': tp2=0, tp4=2, tp6=4).
    Division by time_scale_factor is handled internally, matching the
    convention in pseudodynamics_plus/scripts/klein_eval.py::propagate().
    """
    import torch
    from torchdiffeq import odeint

    def _simulate(start_cells, model, n_sims, device):
        pde_model = model
        tsf = pde_model.time_scale_factor

        def v_ode(t, state):
            t_in = torch.full(
                (state.shape[0], 1), t.item() * tsf, dtype=torch.float32
            ).to(device)
            return pde_model.v(state, t_in)

        s0     = torch.from_numpy(start_cells.astype(np.float32)).to(device)
        t_eval = torch.tensor(
            [t_start_norm / tsf, t_end_norm / tsf], dtype=torch.float32
        ).to(device)

        with torch.no_grad():
            traj = odeint(v_ode, s0, t_eval, method="dopri5", atol=tol, rtol=tol)

        return traj[-1].cpu().numpy()   # (n_cells, n_dims)

    return _simulate


def make_pseudodynamics_sde_sim_fn(
    t_start_norm: float,
    t_end_norm: float,
    n_steps: int = 200,
    noise_scale: float = 1.0,
):
    """
    Return a *stochastic* simulation_func for pseudodynamics+.

    Uses Euler-Maruyama integration with the learned velocity field v(s,t)
    and diffusion field D(s,t):

        dX = v(X,t)·dt + sqrt(2·|D(X,t)|·dt) · noise_scale · dW

    Running n_sims independent trajectories per cell produces diverse
    endpoints, giving a non-degenerate fate distribution F_hat.

    Usage
    -----
        sim_fn = make_pseudodynamics_sde_sim_fn(
            t_start_norm=0.0, t_end_norm=4.0, noise_scale=1.0,
        )
        endpoints = sim_fn(start_cells, pde_model, n_sims=100, device="cuda:0")
        # returns ndarray (n_cells, n_sims, n_dims)

    Parameters
    ----------
    t_start_norm / t_end_norm : float
        Normalised time domain (same convention as the deterministic sim).
    n_steps : int
        Euler-Maruyama discretisation steps.
    noise_scale : float
        Global multiplier on the stochastic term.  Set to 0 for the
        deterministic limit; sweep [0.1, 0.5, 1.0, 2.0] to tune.
    """
    import torch

    def _simulate(start_cells, model, n_sims, device):
        pde_model = model
        tsf = pde_model.time_scale_factor

        t0 = t_start_norm / tsf
        t1 = t_end_norm / tsf
        dt = (t1 - t0) / n_steps

        n_cells, n_dims = start_cells.shape
        sqrt_dt = np.sqrt(abs(dt))

        # Expand each cell n_sims times → (n_cells * n_sims, n_dims)
        s0 = torch.from_numpy(
            np.repeat(start_cells.astype(np.float32), n_sims, axis=0)
        ).to(device)

        pde_model.eval()
        s = s0.clone()
        with torch.no_grad():
            for step in range(n_steps):
                t_val = t0 + step * dt
                t_in = torch.full(
                    (s.shape[0], 1), t_val * tsf, dtype=torch.float32
                ).to(device)

                v = pde_model.v(s, t_in)                   # (N, n_dims)
                D = pde_model.D(s, t_in)                   # (N, 1) or (N, n_dims)

                # Euler-Maruyama step
                dW = torch.randn_like(s) * sqrt_dt         # (N, n_dims)
                diffusion_coeff = torch.sqrt(
                    2.0 * torch.abs(D)
                )                                           # (N, 1 or n_dims)
                # Broadcast D to match s shape if collapsed
                s = s + v * dt + noise_scale * diffusion_coeff * dW

        endpoints = s.cpu().numpy().reshape(n_cells, n_sims, n_dims)
        return endpoints

    return _simulate


def make_pseudodynamics_sb_sim_fn(
    t_start_norm: float,
    t_end_norm: float,
    n_steps: int = 200,
    noise_scale: float = 1.0,
):
    """
    Return a score-guided SDE (Schrödinger Bridge) simulation_func.

    Augments the velocity field with the score function ∇_s log u(s,t)
    computed via autograd on the density surrogate:

        dX = [v(X,t) + noise_scale² · ∇_s log u(X,t)] dt
             + noise_scale · sqrt(2) · dW

    This concentrates trajectories in high-density regions while allowing
    stochastic branching — the Schrödinger Bridge formulation.

    Usage
    -----
        sim_fn = make_pseudodynamics_sb_sim_fn(
            t_start_norm=0.0, t_end_norm=4.0, noise_scale=1.0,
        )
        endpoints = sim_fn(start_cells, pde_model, n_sims=100, device="cuda:0")
        # returns ndarray (n_cells, n_sims, n_dims)

    Parameters
    ----------
    t_start_norm / t_end_norm : float
        Normalised time domain.
    n_steps : int
        Euler-Maruyama discretisation steps.
    noise_scale : float
        Controls diffusion magnitude.  The score drift scales as σ² and
        the noise scales as σ, so larger values give more stochasticity
        while keeping the SB balance.
    """
    import torch

    def _simulate(start_cells, model, n_sims, device):
        pde_model = model
        tsf = pde_model.time_scale_factor

        t0 = t_start_norm / tsf
        t1 = t_end_norm / tsf
        dt = (t1 - t0) / n_steps
        sigma2 = noise_scale ** 2
        sqrt_dt = np.sqrt(abs(dt))

        n_cells, n_dims = start_cells.shape

        # Expand each cell n_sims times → (n_cells * n_sims, n_dims)
        s0 = torch.from_numpy(
            np.repeat(start_cells.astype(np.float32), n_sims, axis=0)
        ).to(device)

        pde_model.eval()
        s = s0.clone()

        # Score computation requires grad; we enable it selectively
        for step in range(n_steps):
            t_val = t0 + step * dt

            # -- compute score ∇_s log u(s,t) with autograd --
            s_grad = s.detach().requires_grad_(True)
            t_in = torch.full(
                (s_grad.shape[0], 1), t_val * tsf, dtype=torch.float32
            ).to(device)
            t_in.requires_grad_(False)

            log_u = pde_model.u(s_grad, t_in)     # log u(s,t)
            score = torch.autograd.grad(
                log_u.sum(), s_grad, create_graph=False
            )[0]                                   # ∇_s log u

            with torch.no_grad():
                v = pde_model.v(s, t_in)           # (N, n_dims)
                dW = torch.randn_like(s) * sqrt_dt

                # SB drift: v + σ² ∇log u
                drift = v + sigma2 * score.detach()
                s = s + drift * dt + noise_scale * np.sqrt(2.0) * dW

        endpoints = s.detach().cpu().numpy().reshape(n_cells, n_sims, n_dims)
        return endpoints

    return _simulate


def make_trajectorynet_sim_fn(int_tp_start: float, int_tp_end: float):
    """
    Return a simulation_func for TrajectoryNet.

    Usage
    -----
        model = cnf_model      # TrajectoryNet SequentialFlow
        sim_fn = make_trajectorynet_sim_fn(int_tp_start=1.0, int_tp_end=2.0)

        endpoints = sim_fn(start_cells, model, n_sims=1, device="cuda:0")
        # returns ndarray (n_cells, n_dims)  — deterministic CNF

    Notes
    -----
    TrajectoryNet's CNF is deterministic; n_sims is ignored.
    int_tp_start / int_tp_end come from tjn_args.int_tps.
    """
    import torch

    def _simulate(start_cells, model, n_sims, device):
        cnf_model = model
        z    = torch.tensor(start_cells, dtype=torch.float32).to(device)
        zero = torch.zeros(z.shape[0], 1, device=device)
        int_times = torch.tensor(
            [int_tp_start, int_tp_end], dtype=torch.float32
        ).to(device)

        cnf = cnf_model.chain[0]
        # with torch.no_grad():
        z_out, _ = cnf(z, zero, integration_times=int_times, reverse=False)

        return z_out.detach().cpu().numpy()   # (n_cells, n_dims)

    return _simulate
