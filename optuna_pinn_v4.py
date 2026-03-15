from __future__ import annotations

import argparse
import csv
import gc
import json
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
from optuna.exceptions import TrialPruned
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from optuna.visualization.matplotlib import plot_contour
from optuna.visualization.matplotlib import plot_edf
from optuna.visualization.matplotlib import plot_intermediate_values
from optuna.visualization.matplotlib import plot_optimization_history
from optuna.visualization.matplotlib import plot_parallel_coordinate
from optuna.visualization.matplotlib import plot_param_importances
from optuna.visualization.matplotlib import plot_slice
from optuna.visualization.matplotlib import plot_timeline
from optuna_dashboard import run_server
import scipy.io
import torch
import torch.nn as nn


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_RANDOM_SEED = 1234
DEFAULT_DATA_PATH = Path("plate_data.mat")
DEFAULT_PROXY_EPOCHS = 5000
DEFAULT_REPORT_EVERY = 500
DEFAULT_N_TRIALS = 25
DEFAULT_MEASUREMENT_POINT_COUNT = 50
DEFAULT_FIGURE_DPI = 300

STUDY_NAME_BY_MODE = {
    "baseline": "pinn_v4_baseline",
    "part_e": "pinn_v4_part_e",
}

# The tuning objective is displacement relative L2 error against the FEM field.
OBJECTIVE_NAME = "disp_rel_l2"

# Keep the PDE itself fixed. These are physical parameters, not tuning parameters.
YOUNGS_MODULUS = 10.0
POISSON_RATIO = 0.3
RIGHT_TRACTION = 0.1
TOP_TRACTION = 0.0

# Mirror the notebook defaults where possible.
ACTIVATION = nn.Tanh
LOSS_FUNCTION = nn.MSELoss()

# Explicit Optuna search ranges.
SEARCH_SPACE_BASE: dict[str, dict[str, Any]] = {
    "learning_rate": {
        "type": "float",
        "low": 1.0e-4,
        "high": 3.0e-3,
        "log": True,
        "description": "Adam learning rate.",
    },
    "disp_width": {
        "type": "categorical",
        "choices": [200, 300, 400],
        "description": "Hidden width for the displacement net with 2 hidden layers.",
    },
    "stress_width": {
        "type": "categorical",
        "choices": [300, 400, 500],
        "description": "Hidden width for the stress net with 2 hidden layers.",
    },
    "scheduler": {
        "type": "categorical",
        "choices": ["none", "step"],
        "description": "Use no scheduler or StepLR.",
    },
    "lr_gamma": {
        "type": "float",
        "low": 0.30,
        "high": 0.70,
        "description": "StepLR multiplicative decay factor when scheduler='step'.",
    },
    "lr_step_size": {
        "type": "categorical",
        "choices": [2000, 4000, 6000],
        "description": "StepLR step size for proxy runs when scheduler='step'.",
    },
    "w_bc": {
        "type": "categorical",
        "choices": [0.5, 1.0, 2.0, 5.0],
        "description": "Weight on the total boundary-condition loss.",
    },
    "w_cons_bc": {
        "type": "categorical",
        "choices": [0.5, 1.0, 2.0],
        "description": "Weight on the constitutive boundary-consistency loss.",
    },
}

SEARCH_SPACE_PART_E_EXTRA: dict[str, dict[str, Any]] = {
    "w_data": {
        "type": "float",
        "low": 10.0,
        "high": 1000.0,
        "log": True,
        "description": "Weight on the 50-point displacement data loss for part (e).",
    }
}


# =============================================================================
# Utilities
# =============================================================================


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_serializable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, nn.Module):
        return value.__class__.__name__
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(to_serializable(payload), f, indent=2)


def build_storage_url(storage_path: Path) -> str:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{storage_path.as_posix()}"


def get_preferred_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def device_supports_float64(device: torch.device) -> bool:
    try:
        test_tensor = torch.ones((100, 100), dtype=torch.float64, device=device)
        _ = test_tensor @ test_tensor
        return True
    except Exception:
        return False


def select_device(force_device: bool) -> tuple[torch.device, torch.dtype]:
    requested_device = get_preferred_device()

    if device_supports_float64(requested_device):
        device = requested_device
        dtype = torch.float64
    else:
        if force_device and requested_device.type != "cpu":
            device = requested_device
            dtype = torch.float32
        else:
            device = torch.device("cpu")
            dtype = torch.float64

    torch.set_default_dtype(dtype)
    return device, dtype


def select_device_for_dtype(target_dtype: torch.dtype) -> tuple[torch.device, torch.dtype]:
    requested_device = get_preferred_device()

    if target_dtype == torch.float32:
        device = requested_device
        dtype = torch.float32
    else:
        if device_supports_float64(requested_device):
            device = requested_device
        else:
            device = torch.device("cpu")
        dtype = torch.float64

    torch.set_default_dtype(dtype)
    return device, dtype


def maybe_empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def build_layers(input_dim: int, width: int, output_dim: int) -> list[int]:
    return [input_dim, width, width, output_dim]


def make_stiffness_matrix(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    stiff = YOUNGS_MODULUS / (1 - POISSON_RATIO**2) * torch.tensor(
        [
            [1, POISSON_RATIO, 0],
            [POISSON_RATIO, 1, 0],
            [0, 0, (1 - POISSON_RATIO) / 2],
        ],
        dtype=dtype,
        device=device,
    )
    return stiff.unsqueeze(0)


def get_search_space(study_mode: str) -> dict[str, dict[str, Any]]:
    search_space = dict(SEARCH_SPACE_BASE)
    if study_mode == "part_e":
        search_space.update(SEARCH_SPACE_PART_E_EXTRA)
    return search_space


def get_fixed_problem_config() -> dict[str, Any]:
    return {
        "youngs_modulus": YOUNGS_MODULUS,
        "poisson_ratio": POISSON_RATIO,
        "right_traction": RIGHT_TRACTION,
        "top_traction": TOP_TRACTION,
        "activation": ACTIVATION.__name__,
        "loss_function": LOSS_FUNCTION.__class__.__name__,
        "objective_name": OBJECTIVE_NAME,
        "constitutive_model": "Plane stress with engineering shear gamma_xy = du/dy + dv/dx.",
        "equilibrium_equations": [
            "d sigma_11 / dx + d sigma_12 / dy = 0",
            "d sigma_12 / dx + d sigma_22 / dy = 0",
        ],
        "boundary_conditions": {
            "left": "u_x = 0",
            "bottom": "u_y = 0",
            "right": f"sigma_xx = {RIGHT_TRACTION}, sigma_xy = 0",
            "top": f"sigma_yy = {TOP_TRACTION}, sigma_xy = 0",
            "hole": "sigma * n = 0",
        },
    }


def get_pinn_v4_alignment_summary() -> dict[str, Any]:
    return {
        "reference_notebook": "PINN_v4.ipynb",
        "architecture": {
            "displacement_net": "DenseNet [2, disp_width, disp_width, 2]",
            "stress_net": "DenseNet [2, stress_width, stress_width, 3]",
            "activation": ACTIVATION,
        },
        "training_loss_structure": (
            "w_eq * loss_eq + w_cons * loss_cons + w_cons_bc * loss_cons_bc + "
            "w_bc * loss_bc + w_data * loss_data"
        ),
        "fixed_problem_config": get_fixed_problem_config(),
        "tuned_parameters": list(SEARCH_SPACE_BASE.keys()) + list(SEARCH_SPACE_PART_E_EXTRA.keys()),
    }


# =============================================================================
# Data processing
# =============================================================================


@dataclass
class ProblemData:
    L_boundary: torch.Tensor
    R_boundary: torch.Tensor
    T_boundary: torch.Tensor
    B_boundary: torch.Tensor
    C_boundary: torch.Tensor
    Boundary: torch.Tensor
    disp_truth: torch.Tensor
    t_connect: torch.Tensor
    x_full: torch.Tensor
    x_full_eval: torch.Tensor
    x: torch.Tensor
    x_fix: torch.Tensor
    disp_fix: torch.Tensor
    coords_full_np: np.ndarray
    conn_full_np: np.ndarray
    disp_truth_np: np.ndarray
    fem_stress_elem_np: np.ndarray
    elem_centroids_np: np.ndarray


def compute_cst_element_stress(
    coords_np: np.ndarray,
    conn_np: np.ndarray,
    nodal_disp_np: np.ndarray,
) -> np.ndarray:
    elem_coords = coords_np[conn_np]
    n_elem = elem_coords.shape[0]

    me = np.concatenate(
        [
            np.ones((n_elem, 3, 1), dtype=coords_np.dtype),
            elem_coords,
        ],
        axis=2,
    )
    me_inv = np.linalg.inv(me)
    dndx = me_inv[:, 1, :]
    dndy = me_inv[:, 2, :]

    b_mats = np.zeros((n_elem, 3, 6), dtype=coords_np.dtype)
    b_mats[:, 0, 0::2] = dndx
    b_mats[:, 1, 1::2] = dndy
    b_mats[:, 2, 0::2] = dndy
    b_mats[:, 2, 1::2] = dndx

    elem_disp = nodal_disp_np[conn_np].reshape(n_elem, 6)
    elem_strain = np.einsum("eij,ej->ei", b_mats, elem_disp)

    constitutive_np = (YOUNGS_MODULUS / (1 - POISSON_RATIO**2)) * np.array(
        [
            [1, POISSON_RATIO, 0],
            [POISSON_RATIO, 1, 0],
            [0, 0, (1 - POISSON_RATIO) / 2],
        ],
        dtype=coords_np.dtype,
    )
    return elem_strain @ constitutive_np.T


def load_problem_data(
    data_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    measurement_point_count: int,
    random_seed: int,
) -> ProblemData:
    data = scipy.io.loadmat(data_path)

    L_boundary = torch.tensor(data["L_boundary"], dtype=dtype, device=device)
    R_boundary = torch.tensor(data["R_boundary"], dtype=dtype, device=device)
    T_boundary = torch.tensor(data["T_boundary"], dtype=dtype, device=device)
    B_boundary = torch.tensor(data["B_boundary"], dtype=dtype, device=device)
    C_boundary = torch.tensor(data["C_boundary"], dtype=dtype, device=device)
    Boundary = torch.tensor(data["Boundary"], dtype=dtype, device=device, requires_grad=True)

    disp_truth = torch.tensor(data["disp_data"], dtype=dtype, device=device)
    t_connect = torch.tensor(data["t"].astype(float), dtype=dtype, device=device)
    x_full = torch.tensor(data["p_full"], dtype=dtype, device=device, requires_grad=True)
    x_full_eval = torch.tensor(data["p_full"], dtype=dtype, device=device)
    x = torch.tensor(data["p"], dtype=dtype, device=device, requires_grad=True)

    rng = np.random.default_rng(random_seed)
    rand_index_np = rng.choice(len(data["p_full"]), size=measurement_point_count, replace=False)
    rand_index = torch.tensor(rand_index_np, dtype=torch.long, device=device)
    # Part (e) uses these as supervised measurement points only, so they should
    # not carry the autograd graph from x_full across training iterations.
    x_fix = x_full_eval.index_select(0, rand_index).detach()
    disp_fix = disp_truth.index_select(0, rand_index)

    coords_full_np = np.asarray(data["p_full"], dtype=np.float64)
    conn_full_np = np.asarray(data["t"], dtype=np.int64) - 1
    disp_truth_np = np.asarray(data["disp_data"], dtype=np.float64)
    fem_stress_elem_np = compute_cst_element_stress(coords_full_np, conn_full_np, disp_truth_np)
    elem_centroids_np = coords_full_np[conn_full_np].mean(axis=1)

    return ProblemData(
        L_boundary=L_boundary,
        R_boundary=R_boundary,
        T_boundary=T_boundary,
        B_boundary=B_boundary,
        C_boundary=C_boundary,
        Boundary=Boundary,
        disp_truth=disp_truth,
        t_connect=t_connect,
        x_full=x_full,
        x_full_eval=x_full_eval,
        x=x,
        x_fix=x_fix,
        disp_fix=disp_fix,
        coords_full_np=coords_full_np,
        conn_full_np=conn_full_np,
        disp_truth_np=disp_truth_np,
        fem_stress_elem_np=fem_stress_elem_np,
        elem_centroids_np=elem_centroids_np,
    )


# =============================================================================
# Define Neural Network
# =============================================================================


class DenseNet(nn.Module):
    def __init__(self, layers: list[int], nonlinearity: type[nn.Module]) -> None:
        super().__init__()

        self.n_layers = len(layers) - 1
        assert self.n_layers >= 1

        self.layers = nn.ModuleList()
        for layer_index in range(self.n_layers):
            self.layers.append(nn.Linear(layers[layer_index], layers[layer_index + 1]))
            if layer_index != self.n_layers - 1:
                self.layers.append(nonlinearity())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# =============================================================================
# Metrics and post-processing
# =============================================================================


def compute_displacement_metrics(disp_net: nn.Module, problem_data: ProblemData) -> dict[str, float]:
    with torch.no_grad():
        disp_pred_full = disp_net(problem_data.x_full_eval)
        disp_error_full = disp_pred_full - problem_data.disp_truth

        u_true_norm = torch.linalg.norm(problem_data.disp_truth[:, 0]).detach().cpu().item()
        v_true_norm = torch.linalg.norm(problem_data.disp_truth[:, 1]).detach().cpu().item()
        disp_true_norm = torch.linalg.norm(problem_data.disp_truth).detach().cpu().item()

        u_pred_np = disp_pred_full[:, 0].detach().cpu().numpy()
        v_pred_np = disp_pred_full[:, 1].detach().cpu().numpy()
        u_true_np = problem_data.disp_truth[:, 0].detach().cpu().numpy()
        v_true_np = problem_data.disp_truth[:, 1].detach().cpu().numpy()

    u_abs_err_np = np.abs(u_pred_np - u_true_np)
    v_abs_err_np = np.abs(v_pred_np - v_true_np)

    return {
        "u_rmse": float(torch.sqrt(torch.mean((disp_error_full[:, 0]) ** 2)).detach().cpu().item()),
        "v_rmse": float(torch.sqrt(torch.mean((disp_error_full[:, 1]) ** 2)).detach().cpu().item()),
        "disp_rmse": float(torch.sqrt(torch.mean(disp_error_full**2)).detach().cpu().item()),
        "u_rel_l2": float(torch.linalg.norm(disp_error_full[:, 0]).detach().cpu().item() / max(u_true_norm, 1e-16)),
        "v_rel_l2": float(torch.linalg.norm(disp_error_full[:, 1]).detach().cpu().item() / max(v_true_norm, 1e-16)),
        "disp_rel_l2": float(torch.linalg.norm(disp_error_full).detach().cpu().item() / max(disp_true_norm, 1e-16)),
        "u_max_abs": float(np.max(u_abs_err_np)),
        "v_max_abs": float(np.max(v_abs_err_np)),
    }


def compute_pinn_constitutive_stress(
    disp_net: nn.Module,
    points_np: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    query_points = torch.tensor(points_np, dtype=dtype, device=device, requires_grad=True)
    disp_query = disp_net(query_points)

    u_query = disp_query[:, 0]
    v_query = disp_query[:, 1]
    dudx_query = torch.autograd.grad(
        u_query,
        query_points,
        grad_outputs=torch.ones_like(u_query),
        create_graph=False,
        retain_graph=True,
    )[0]
    dvdx_query = torch.autograd.grad(
        v_query,
        query_points,
        grad_outputs=torch.ones_like(v_query),
        create_graph=False,
    )[0]

    strain_query = torch.stack(
        [
            dudx_query[:, 0],
            dvdx_query[:, 1],
            dudx_query[:, 1] + dvdx_query[:, 0],
        ],
        dim=1,
    )

    constitutive = make_stiffness_matrix(dtype, device).squeeze(0)
    stress_query = strain_query @ constitutive.T
    return stress_query.detach().cpu().numpy()


def compute_stress_metrics(
    disp_net: nn.Module,
    problem_data: ProblemData,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, dict[str, float]]:
    pinn_stress_elem_np = compute_pinn_constitutive_stress(
        disp_net=disp_net,
        points_np=problem_data.elem_centroids_np,
        device=device,
        dtype=dtype,
    )
    stress_err_elem_np = pinn_stress_elem_np - problem_data.fem_stress_elem_np

    metrics: dict[str, dict[str, float]] = {}
    for index, name in enumerate(["sigma11", "sigma22", "sigma12"]):
        diff = stress_err_elem_np[:, index]
        ref = problem_data.fem_stress_elem_np[:, index]
        metrics[name] = {
            "rmse": float(np.sqrt(np.mean(diff**2))),
            "rel_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(ref), 1e-16)),
            "max_abs": float(np.max(np.abs(diff))),
        }
    return metrics


# =============================================================================
# Trial configuration and training
# =============================================================================


@dataclass
class TrialConfig:
    learning_rate: float
    disp_width: int
    stress_width: int
    scheduler: str
    lr_gamma: float
    lr_step_size: int
    w_eq: float
    w_cons: float
    w_cons_bc: float
    w_bc: float
    w_data: float
    disp_layers: list[int]
    stress_layers: list[int]


@dataclass
class TrialResult:
    objective_value: float
    displacement_metrics: dict[str, float]
    stress_metrics: dict[str, dict[str, float]]
    final_loss: float
    best_reported_value: float
    history: dict[str, list[float]]


def suggest_trial_config(trial: optuna.Trial, study_mode: str) -> TrialConfig:
    learning_rate = trial.suggest_float("learning_rate", 1.0e-4, 3.0e-3, log=True)
    disp_width = trial.suggest_categorical("disp_width", [200, 300, 400])
    stress_width = trial.suggest_categorical("stress_width", [300, 400, 500])
    scheduler_name = trial.suggest_categorical("scheduler", ["none", "step"])
    w_bc = trial.suggest_categorical("w_bc", [0.5, 1.0, 2.0, 5.0])
    w_cons_bc = trial.suggest_categorical("w_cons_bc", [0.5, 1.0, 2.0])

    if scheduler_name == "step":
        lr_gamma = trial.suggest_float("lr_gamma", 0.30, 0.70)
        lr_step_size = trial.suggest_categorical("lr_step_size", [2000, 4000, 6000])
    else:
        lr_gamma = 1.0
        lr_step_size = 10**9

    if study_mode == "part_e":
        w_data = trial.suggest_float("w_data", 10.0, 1000.0, log=True)
    else:
        w_data = 0.0

    return TrialConfig(
        learning_rate=learning_rate,
        disp_width=disp_width,
        stress_width=stress_width,
        scheduler=scheduler_name,
        lr_gamma=lr_gamma,
        lr_step_size=lr_step_size,
        w_eq=1.0,
        w_cons=1.0,
        w_cons_bc=w_cons_bc,
        w_bc=w_bc,
        w_data=w_data,
        disp_layers=build_layers(2, disp_width, 2),
        stress_layers=build_layers(2, stress_width, 3),
    )


def trial_config_from_params(params: dict[str, Any], study_mode: str) -> TrialConfig:
    scheduler_name = str(params["scheduler"])
    if scheduler_name == "step":
        lr_gamma = float(params["lr_gamma"])
        lr_step_size = int(params["lr_step_size"])
    else:
        lr_gamma = 1.0
        lr_step_size = 10**9

    if study_mode == "part_e":
        w_data = float(params["w_data"])
    else:
        w_data = 0.0

    disp_width = int(params["disp_width"])
    stress_width = int(params["stress_width"])

    return TrialConfig(
        learning_rate=float(params["learning_rate"]),
        disp_width=disp_width,
        stress_width=stress_width,
        scheduler=scheduler_name,
        lr_gamma=lr_gamma,
        lr_step_size=lr_step_size,
        w_eq=1.0,
        w_cons=1.0,
        w_cons_bc=float(params["w_cons_bc"]),
        w_bc=float(params["w_bc"]),
        w_data=w_data,
        disp_layers=build_layers(2, disp_width, 2),
        stress_layers=build_layers(2, stress_width, 3),
    )


def save_loss_history_npz(path: Path, history: dict[str, list[float]]) -> None:
    np.savez(path, **{key: np.asarray(values) for key, values in history.items()})


def run_single_trial(
    *,
    trial: optuna.Trial | None,
    trial_config: TrialConfig,
    problem_data: ProblemData,
    device: torch.device,
    dtype: torch.dtype,
    epochs: int,
    report_every: int,
    use_data_part_e: bool,
    trial_dir: Path,
) -> TrialResult:
    torch.set_default_dtype(dtype)
    stress_net = DenseNet(trial_config.stress_layers, ACTIVATION).to(device=device, dtype=dtype)
    disp_net = DenseNet(trial_config.disp_layers, ACTIVATION).to(device=device, dtype=dtype)

    params = list(stress_net.parameters()) + list(disp_net.parameters())
    optimizer = torch.optim.Adam(params, lr=trial_config.learning_rate)
    scheduler = None
    if trial_config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=trial_config.lr_step_size,
            gamma=trial_config.lr_gamma,
        )

    stiff = make_stiffness_matrix(dtype, device)
    stiff_interior = torch.broadcast_to(stiff, (len(problem_data.x), 3, 3))
    stiff_boundary = torch.broadcast_to(stiff, (len(problem_data.Boundary), 3, 3))

    history = {
        "total": [],
        "eq_total": [],
        "cons": [],
        "cons_bc": [],
        "bc_total": [],
        "data": [],
        "lr": [],
        OBJECTIVE_NAME: [],
    }

    best_reported_value = float("inf")

    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)

        sigma = stress_net(problem_data.x)
        disp = disp_net(problem_data.x)

        u = disp[:, 0]
        v = disp[:, 1]
        dudx = torch.autograd.grad(u, problem_data.x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        dvdx = torch.autograd.grad(v, problem_data.x, grad_outputs=torch.ones_like(v), create_graph=True)[0]

        e_11 = dudx[:, 0].unsqueeze(1)
        e_22 = dvdx[:, 1].unsqueeze(1)
        e_12 = (dudx[:, 1] + dvdx[:, 0]).unsqueeze(1)
        strain = torch.cat((e_11, e_22, e_12), dim=1).unsqueeze(2)
        sig_aug = torch.bmm(stiff_interior, strain).squeeze(2)
        loss_cons = LOSS_FUNCTION(sig_aug, sigma)

        disp_bc = disp_net(problem_data.Boundary)
        sigma_bc = stress_net(problem_data.Boundary)
        u_bc = disp_bc[:, 0]
        v_bc = disp_bc[:, 1]
        dudx_bc = torch.autograd.grad(
            u_bc,
            problem_data.Boundary,
            grad_outputs=torch.ones_like(u_bc),
            create_graph=True,
        )[0]
        dvdx_bc = torch.autograd.grad(
            v_bc,
            problem_data.Boundary,
            grad_outputs=torch.ones_like(v_bc),
            create_graph=True,
        )[0]

        e_11_bc = dudx_bc[:, 0].unsqueeze(1)
        e_22_bc = dvdx_bc[:, 1].unsqueeze(1)
        e_12_bc = (dudx_bc[:, 1] + dvdx_bc[:, 0]).unsqueeze(1)
        strain_bc = torch.cat((e_11_bc, e_22_bc, e_12_bc), dim=1).unsqueeze(2)
        sig_aug_bc = torch.bmm(stiff_boundary, strain_bc).squeeze(2)
        loss_cons_bc = LOSS_FUNCTION(sig_aug_bc, sigma_bc)

        sig_11 = sigma[:, 0]
        sig_22 = sigma[:, 1]
        sig_12 = sigma[:, 2]
        dsig11dx = torch.autograd.grad(sig_11, problem_data.x, grad_outputs=torch.ones_like(sig_11), create_graph=True)[0]
        dsig22dx = torch.autograd.grad(sig_22, problem_data.x, grad_outputs=torch.ones_like(sig_22), create_graph=True)[0]
        dsig12dx = torch.autograd.grad(sig_12, problem_data.x, grad_outputs=torch.ones_like(sig_12), create_graph=True)[0]

        eq_x1 = dsig11dx[:, 0] + dsig12dx[:, 1]
        eq_x2 = dsig12dx[:, 0] + dsig22dx[:, 1]
        loss_eq1 = LOSS_FUNCTION(eq_x1, torch.zeros_like(eq_x1))
        loss_eq2 = LOSS_FUNCTION(eq_x2, torch.zeros_like(eq_x2))
        loss_eq = loss_eq1 + loss_eq2

        u_L = disp_net(problem_data.L_boundary)
        u_B = disp_net(problem_data.B_boundary)
        sig_R = stress_net(problem_data.R_boundary)
        sig_T = stress_net(problem_data.T_boundary)
        sig_C = stress_net(problem_data.C_boundary)

        loss_BC_L = LOSS_FUNCTION(u_L[:, 0], torch.zeros_like(u_L[:, 0]))
        loss_BC_B = LOSS_FUNCTION(u_B[:, 1], torch.zeros_like(u_B[:, 1]))
        loss_BC_R = LOSS_FUNCTION(sig_R[:, 0], RIGHT_TRACTION * torch.ones_like(sig_R[:, 0])) + LOSS_FUNCTION(
            sig_R[:, 2], torch.zeros_like(sig_R[:, 2])
        )
        loss_BC_T = LOSS_FUNCTION(sig_T[:, 1], TOP_TRACTION * torch.ones_like(sig_T[:, 1])) + LOSS_FUNCTION(
            sig_T[:, 2], torch.zeros_like(sig_T[:, 2])
        )
        loss_BC_C = LOSS_FUNCTION(
            sig_C[:, 0] * problem_data.C_boundary[:, 0] + sig_C[:, 2] * problem_data.C_boundary[:, 1],
            torch.zeros_like(sig_C[:, 0]),
        ) + LOSS_FUNCTION(
            sig_C[:, 2] * problem_data.C_boundary[:, 0] + sig_C[:, 1] * problem_data.C_boundary[:, 1],
            torch.zeros_like(sig_C[:, 0]),
        )
        loss_bc = loss_BC_L + loss_BC_B + loss_BC_R + loss_BC_T + loss_BC_C

        loss_data = torch.zeros((), dtype=dtype, device=device)
        loss = (
            trial_config.w_eq * loss_eq
            + trial_config.w_cons * loss_cons
            + trial_config.w_cons_bc * loss_cons_bc
            + trial_config.w_bc * loss_bc
        )
        if use_data_part_e:
            u_fix = disp_net(problem_data.x_fix)
            loss_data = LOSS_FUNCTION(u_fix, problem_data.disp_fix)
            loss = loss + trial_config.w_data * loss_data

        if not torch.isfinite(loss):
            if trial is None:
                raise RuntimeError("Encountered non-finite loss during rerun.")
            raise TrialPruned("Encountered non-finite loss.")

        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        history["total"].append(float(loss.detach().cpu().item()))
        history["eq_total"].append(float(loss_eq.detach().cpu().item()))
        history["cons"].append(float(loss_cons.detach().cpu().item()))
        history["cons_bc"].append(float(loss_cons_bc.detach().cpu().item()))
        history["bc_total"].append(float(loss_bc.detach().cpu().item()))
        history["data"].append(float(loss_data.detach().cpu().item()))
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        should_report = (epoch + 1) % report_every == 0 or epoch == epochs - 1
        if should_report:
            displacement_metrics = compute_displacement_metrics(disp_net, problem_data)
            current_value = displacement_metrics[OBJECTIVE_NAME]
            history[OBJECTIVE_NAME].append(current_value)
            best_reported_value = min(best_reported_value, current_value)

            if trial is not None:
                trial.report(current_value, step=epoch + 1)
                if trial.should_prune():
                    save_loss_history_npz(trial_dir / "loss_history_pruned.npz", history)
                    raise TrialPruned(f"Pruned at epoch {epoch + 1} with {OBJECTIVE_NAME}={current_value:.6e}")

    displacement_metrics = compute_displacement_metrics(disp_net, problem_data)
    stress_metrics = compute_stress_metrics(disp_net, problem_data, device=device, dtype=dtype)
    final_loss = history["total"][-1]
    objective_value = displacement_metrics[OBJECTIVE_NAME]

    save_loss_history_npz(trial_dir / "loss_history.npz", history)

    return TrialResult(
        objective_value=objective_value,
        displacement_metrics=displacement_metrics,
        stress_metrics=stress_metrics,
        final_loss=final_loss,
        best_reported_value=best_reported_value,
        history=history,
    )


# =============================================================================
# Study orchestration and visualization
# =============================================================================


@dataclass
class StudyPaths:
    study_root: Path
    figures_dir: Path
    trials_dir: Path
    reports_dir: Path
    reruns_dir: Path


def build_study_paths(study_name: str) -> StudyPaths:
    study_root = Path("problem_1_experiments") / "optuna_studies" / study_name
    figures_dir = study_root / "figures"
    trials_dir = study_root / "trials"
    reports_dir = study_root / "reports"
    reruns_dir = study_root / "reruns"
    for directory in [study_root, figures_dir, trials_dir, reports_dir, reruns_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    return StudyPaths(
        study_root=study_root,
        figures_dir=figures_dir,
        trials_dir=trials_dir,
        reports_dir=reports_dir,
        reruns_dir=reruns_dir,
    )


def save_matplotlib_figure(filename: Path, figure: plt.Figure) -> None:
    figure.tight_layout()
    figure.savefig(filename, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def save_study_visualizations(study: optuna.Study, study_paths: StudyPaths) -> None:
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        return

    plot_builders = {
        "optimization_history.png": lambda: plot_optimization_history(study).figure,
        "intermediate_values.png": lambda: plot_intermediate_values(study).figure,
        "edf.png": lambda: plot_edf(study).figure,
        "timeline.png": lambda: plot_timeline(study).figure,
    }

    if len(completed_trials) >= 2:
        plot_builders["param_importances.png"] = lambda: plot_param_importances(study).figure
        params = list(study.best_params.keys())
        if params:
            plot_builders["parallel_coordinate.png"] = lambda: plot_parallel_coordinate(study, params=params).figure
            plot_builders["slice.png"] = lambda: plot_slice(study, params=params).figure
        contour_params = [param for param in ["learning_rate", "w_bc", "w_data"] if param in study.best_params]
        if len(contour_params) >= 2:
            plot_builders["contour.png"] = lambda: plot_contour(study, params=contour_params[:2]).figure

    for filename, builder in plot_builders.items():
        try:
            figure = builder()
            save_matplotlib_figure(study_paths.figures_dir / filename, figure)
        except Exception as exc:
            print(f"Skipping study plot {filename}: {exc}")


def get_best_trial_or_none(study: optuna.Study) -> optuna.trial.FrozenTrial | None:
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        return None
    return study.best_trial


def save_study_reports(
    study: optuna.Study,
    study_paths: StudyPaths,
    storage_url: str,
    study_mode: str,
    use_data_part_e: bool,
    search_space: dict[str, dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    trials_df = study.trials_dataframe()
    trials_df.to_csv(study_paths.reports_dir / "trials.csv", index=False)

    write_json(
        study_paths.reports_dir / "study_config.json",
        {
            "study_name": study.study_name,
            "study_mode": study_mode,
            "use_data_part_e": use_data_part_e,
            "objective_name": OBJECTIVE_NAME,
            "storage_url": storage_url,
            "search_space": search_space,
            "fixed_problem_config": get_fixed_problem_config(),
            "pinn_v4_alignment": get_pinn_v4_alignment_summary(),
            "args": vars(args),
        },
    )

    best_trial = get_best_trial_or_none(study)
    if best_trial is not None:
        write_json(
            study_paths.reports_dir / "best_trial.json",
            {
                "number": best_trial.number,
                "value": best_trial.value,
                "params": best_trial.params,
                "user_attrs": best_trial.user_attrs,
            },
        )


def run_study(
    *,
    study_mode: str,
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    storage_url: str,
) -> None:
    use_data_part_e = study_mode == "part_e"
    study_name = STUDY_NAME_BY_MODE[study_mode]
    study_paths = build_study_paths(study_name)
    search_space = get_search_space(study_mode)

    write_json(study_paths.reports_dir / "search_space.json", search_space)

    pruner: optuna.pruners.BasePruner | None
    if args.disable_pruning:
        pruner = None
    else:
        pruner = MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=args.report_every,
            interval_steps=args.report_every,
        )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",
        sampler=TPESampler(seed=args.seed, multivariate=True),
        pruner=pruner,
    )

    problem_data = load_problem_data(
        data_path=args.data_path,
        device=device,
        dtype=dtype,
        measurement_point_count=args.measurement_point_count,
        random_seed=args.seed,
    )

    def objective(trial: optuna.Trial) -> float:
        set_random_seed(args.seed + trial.number)
        trial_config = suggest_trial_config(trial, study_mode=study_mode)
        trial_dir = study_paths.trials_dir / f"trial_{trial.number:04d}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        write_json(
            trial_dir / "trial_config.json",
            {
                "study_mode": study_mode,
                "use_data_part_e": use_data_part_e,
                "trial_number": trial.number,
                "trial_config": asdict(trial_config),
                "fixed_problem_config": get_fixed_problem_config(),
                "pinn_v4_alignment": get_pinn_v4_alignment_summary(),
                "device": device,
                "dtype": dtype,
                "epochs": args.epochs,
                "report_every": args.report_every,
            },
        )

        result = run_single_trial(
            trial=trial,
            trial_config=trial_config,
            problem_data=problem_data,
            device=device,
            dtype=dtype,
            epochs=args.epochs,
            report_every=args.report_every,
            use_data_part_e=use_data_part_e,
            trial_dir=trial_dir,
        )

        write_json(
            trial_dir / "trial_metrics.json",
            {
                "objective_name": OBJECTIVE_NAME,
                "objective_value": result.objective_value,
                "best_reported_value": result.best_reported_value,
                "final_loss": result.final_loss,
                "displacement_metrics": result.displacement_metrics,
                "stress_metrics": result.stress_metrics,
            },
        )

        trial.set_user_attr("study_mode", study_mode)
        trial.set_user_attr("use_data_part_e", use_data_part_e)
        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("final_loss", result.final_loss)
        trial.set_user_attr("best_reported_value", result.best_reported_value)
        for key, value in result.displacement_metrics.items():
            trial.set_user_attr(key, value)
        for stress_name, metrics in result.stress_metrics.items():
            for metric_name, metric_value in metrics.items():
                trial.set_user_attr(f"{stress_name}_{metric_name}", metric_value)

        return result.objective_value

    study.optimize(
        objective,
        n_trials=args.n_trials,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    save_study_visualizations(study, study_paths)
    save_study_reports(
        study=study,
        study_paths=study_paths,
        storage_url=storage_url,
        study_mode=study_mode,
        use_data_part_e=use_data_part_e,
        search_space=search_space,
        args=args,
    )

    maybe_empty_cache(device)
    gc.collect()

    best_trial = get_best_trial_or_none(study)
    if best_trial is not None:
        print(f"[{study_name}] best {OBJECTIVE_NAME} = {best_trial.value:.6e}")
        print(f"[{study_name}] best params = {best_trial.params}")
    print(f"[{study_name}] dashboard storage = {storage_url}")
    print(f"[{study_name}] saved reports under {study_paths.study_root}")


def get_completed_trials_sorted(study: optuna.Study) -> list[optuna.trial.FrozenTrial]:
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    return sorted(completed_trials, key=lambda trial: float(trial.value))


def flatten_rerun_result(label: str, device: torch.device, dtype: torch.dtype, result: TrialResult) -> dict[str, Any]:
    return {
        f"{label}_device": str(device),
        f"{label}_dtype": str(dtype),
        f"{label}_objective_value": result.objective_value,
        f"{label}_final_loss": result.final_loss,
        f"{label}_best_reported_value": result.best_reported_value,
        f"{label}_disp_rel_l2": result.displacement_metrics["disp_rel_l2"],
        f"{label}_u_rel_l2": result.displacement_metrics["u_rel_l2"],
        f"{label}_v_rel_l2": result.displacement_metrics["v_rel_l2"],
        f"{label}_sigma11_rel_l2": result.stress_metrics["sigma11"]["rel_l2"],
        f"{label}_sigma22_rel_l2": result.stress_metrics["sigma22"]["rel_l2"],
        f"{label}_sigma12_rel_l2": result.stress_metrics["sigma12"]["rel_l2"],
    }


def save_rerun_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_rerun_comparison_figure(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    valid_rows = [
        row
        for row in rows
        if row.get("float32_objective_value") is not None and row.get("float64_objective_value") is not None
    ]
    if not valid_rows:
        return

    labels = [f"trial {int(row['trial_number'])}" for row in valid_rows]
    positions = np.arange(len(valid_rows))
    width = 0.35

    fig, axes = plt.subplots(3, 1, figsize=(12, 12))

    float32_disp = [float(row["float32_disp_rel_l2"]) for row in valid_rows]
    float64_disp = [float(row["float64_disp_rel_l2"]) for row in valid_rows]
    axes[0].bar(positions - width / 2, float32_disp, width=width, label="float32")
    axes[0].bar(positions + width / 2, float64_disp, width=width, label="float64")
    axes[0].set_ylabel("disp_rel_l2")
    axes[0].set_title("Displacement Relative L2 Error")
    axes[0].set_xticks(positions, labels, rotation=20)
    axes[0].legend()

    float32_sigma11 = [float(row["float32_sigma11_rel_l2"]) for row in valid_rows]
    float64_sigma11 = [float(row["float64_sigma11_rel_l2"]) for row in valid_rows]
    axes[1].bar(positions - width / 2, float32_sigma11, width=width, label="float32")
    axes[1].bar(positions + width / 2, float64_sigma11, width=width, label="float64")
    axes[1].set_ylabel("sigma11_rel_l2")
    axes[1].set_title("Stress Relative L2 Error")
    axes[1].set_xticks(positions, labels, rotation=20)
    axes[1].legend()

    objective_delta = [float(row["float32_minus_float64_objective"]) for row in valid_rows]
    axes[2].bar(positions, objective_delta, color="tab:gray")
    axes[2].axhline(0.0, color="black", linewidth=1)
    axes[2].set_ylabel("float32 - float64")
    axes[2].set_title("Objective Difference")
    axes[2].set_xticks(positions, labels, rotation=20)

    save_matplotlib_figure(path, fig)


def rerun_single_frozen_trial(
    *,
    frozen_trial: optuna.trial.FrozenTrial,
    study_mode: str,
    problem_data: ProblemData,
    device: torch.device,
    dtype: torch.dtype,
    epochs: int,
    report_every: int,
    use_data_part_e: bool,
    output_dir: Path,
    seed: int,
) -> TrialResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    trial_config = trial_config_from_params(frozen_trial.params, study_mode=study_mode)

    write_json(
        output_dir / "trial_config.json",
        {
            "study_mode": study_mode,
            "use_data_part_e": use_data_part_e,
            "source_trial_number": frozen_trial.number,
            "source_trial_value": frozen_trial.value,
            "trial_config": asdict(trial_config),
            "fixed_problem_config": get_fixed_problem_config(),
            "device": device,
            "dtype": dtype,
            "epochs": epochs,
            "report_every": report_every,
            "seed": seed,
        },
    )

    set_random_seed(seed)
    result = run_single_trial(
        trial=None,
        trial_config=trial_config,
        problem_data=problem_data,
        device=device,
        dtype=dtype,
        epochs=epochs,
        report_every=report_every,
        use_data_part_e=use_data_part_e,
        trial_dir=output_dir,
    )

    write_json(
        output_dir / "trial_metrics.json",
        {
            "objective_name": OBJECTIVE_NAME,
            "objective_value": result.objective_value,
            "best_reported_value": result.best_reported_value,
            "final_loss": result.final_loss,
            "displacement_metrics": result.displacement_metrics,
            "stress_metrics": result.stress_metrics,
        },
    )
    return result


def rerun_best_trials(
    *,
    study_mode: str,
    args: argparse.Namespace,
    storage_url: str,
) -> None:
    study_name = STUDY_NAME_BY_MODE[study_mode]
    study_paths = build_study_paths(study_name)
    use_data_part_e = study_mode == "part_e"

    try:
        study = optuna.load_study(study_name=study_name, storage=storage_url)
    except KeyError:
        print(f"[{study_name}] study not found in storage {storage_url}")
        return

    top_trials = get_completed_trials_sorted(study)[: args.top_k]
    if not top_trials:
        print(f"[{study_name}] no completed trials found to rerun.")
        return

    precision_modes = {
        "float32": select_device_for_dtype(torch.float32),
        "float64": select_device_for_dtype(torch.float64),
    }
    problem_data_by_label = {
        label: load_problem_data(
            data_path=args.data_path,
            device=device,
            dtype=dtype,
            measurement_point_count=args.measurement_point_count,
            random_seed=args.seed,
        )
        for label, (device, dtype) in precision_modes.items()
    }

    rows: list[dict[str, Any]] = []
    for frozen_trial in top_trials:
        trial_root = study_paths.reruns_dir / f"trial_{frozen_trial.number:04d}"
        write_json(
            trial_root / "source_trial.json",
            {
                "study_name": study_name,
                "study_mode": study_mode,
                "objective_name": OBJECTIVE_NAME,
                "source_trial_number": frozen_trial.number,
                "source_trial_value": frozen_trial.value,
                "source_trial_params": frozen_trial.params,
                "source_trial_user_attrs": frozen_trial.user_attrs,
                "fixed_problem_config": get_fixed_problem_config(),
                "pinn_v4_alignment": get_pinn_v4_alignment_summary(),
            },
        )

        row: dict[str, Any] = {
            "trial_number": frozen_trial.number,
            "source_objective_value": frozen_trial.value,
            "study_name": study_name,
            "study_mode": study_mode,
        }
        row.update({f"param_{key}": value for key, value in frozen_trial.params.items()})

        for label, (device, dtype) in precision_modes.items():
            rerun_seed = args.seed + frozen_trial.number
            try:
                result = rerun_single_frozen_trial(
                    frozen_trial=frozen_trial,
                    study_mode=study_mode,
                    problem_data=problem_data_by_label[label],
                    device=device,
                    dtype=dtype,
                    epochs=args.epochs,
                    report_every=args.report_every,
                    use_data_part_e=use_data_part_e,
                    output_dir=trial_root / label,
                    seed=rerun_seed,
                )
                row.update(flatten_rerun_result(label, device, dtype, result))
                row[f"{label}_status"] = "ok"
            except Exception as exc:
                row[f"{label}_status"] = "failed"
                row[f"{label}_error"] = str(exc)
                write_json(
                    trial_root / label / "error.json",
                    {
                        "error": str(exc),
                        "device": device,
                        "dtype": dtype,
                    },
                )
            finally:
                maybe_empty_cache(device)
                gc.collect()

        if row.get("float32_objective_value") is not None and row.get("float64_objective_value") is not None:
            diff = float(row["float32_objective_value"]) - float(row["float64_objective_value"])
            row["float32_minus_float64_objective"] = diff
            row["abs_objective_gap"] = abs(diff)
            row["relative_objective_gap_vs_float64"] = abs(diff) / max(float(row["float64_objective_value"]), 1e-16)
        else:
            row["float32_minus_float64_objective"] = None
            row["abs_objective_gap"] = None
            row["relative_objective_gap_vs_float64"] = None

        rows.append(row)

    summary_payload = {
        "study_name": study_name,
        "study_mode": study_mode,
        "objective_name": OBJECTIVE_NAME,
        "epochs": args.epochs,
        "report_every": args.report_every,
        "top_k": args.top_k,
        "fixed_problem_config": get_fixed_problem_config(),
        "pinn_v4_alignment": get_pinn_v4_alignment_summary(),
        "precision_modes": {
            label: {"device": device, "dtype": dtype}
            for label, (device, dtype) in precision_modes.items()
        },
        "rows": rows,
    }
    write_json(study_paths.reports_dir / "rerun_comparison_summary.json", summary_payload)
    save_rerun_summary_csv(study_paths.reports_dir / "rerun_comparison_summary.csv", rows)
    save_rerun_comparison_figure(study_paths.figures_dir / "rerun_float32_vs_float64.png", rows)

    print(f"[{study_name}] reran top {len(rows)} completed trials.")
    print(f"[{study_name}] saved rerun comparison under {study_paths.study_root}")


# =============================================================================
# Dashboard
# =============================================================================


def launch_dashboard(storage_url: str, host: str, port: int) -> None:
    print(f"Starting Optuna dashboard at http://{host}:{port}")
    print(f"Using storage: {storage_url}")
    run_server(storage=storage_url, host=host, port=port)


# =============================================================================
# CLI
# =============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Optuna studies for the PINN_v4 Problem 1 notebook.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Run Optuna tuning for the baseline study, the part (e) study, or both.",
    )
    optimize_parser.add_argument(
        "--study-mode",
        choices=["baseline", "part_e", "both"],
        default="baseline",
        help="baseline uses USE_DATA_PART_E=False, part_e uses USE_DATA_PART_E=True.",
    )
    optimize_parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    optimize_parser.add_argument("--epochs", type=int, default=DEFAULT_PROXY_EPOCHS)
    optimize_parser.add_argument("--report-every", type=int, default=DEFAULT_REPORT_EVERY)
    optimize_parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    optimize_parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    optimize_parser.add_argument(
        "--measurement-point-count",
        type=int,
        default=DEFAULT_MEASUREMENT_POINT_COUNT,
    )
    optimize_parser.add_argument(
        "--storage-path",
        type=Path,
        default=Path("problem_1_experiments") / "optuna_studies" / "pinn_v4_optuna.db",
    )
    optimize_parser.add_argument(
        "--force-device",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If the selected accelerator does not support float64, keep it and use float32.",
    )
    optimize_parser.add_argument(
        "--disable-pruning",
        action="store_true",
        help="Disable Optuna pruning and run every trial to completion.",
    )

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Launch the Optuna dashboard for the shared SQLite study storage.",
    )
    dashboard_parser.add_argument(
        "--storage-path",
        type=Path,
        default=Path("problem_1_experiments") / "optuna_studies" / "pinn_v4_optuna.db",
    )
    dashboard_parser.add_argument("--host", default="127.0.0.1")
    dashboard_parser.add_argument("--port", type=int, default=8080)

    rerun_parser = subparsers.add_parser(
        "rerun-best",
        help="Rerun the top completed Optuna trials in float32 and float64 for comparison.",
    )
    rerun_parser.add_argument(
        "--study-mode",
        choices=["baseline", "part_e", "both"],
        default="baseline",
        help="baseline uses USE_DATA_PART_E=False, part_e uses USE_DATA_PART_E=True.",
    )
    rerun_parser.add_argument("--top-k", type=int, default=3)
    rerun_parser.add_argument("--epochs", type=int, default=DEFAULT_PROXY_EPOCHS)
    rerun_parser.add_argument("--report-every", type=int, default=DEFAULT_REPORT_EVERY)
    rerun_parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED)
    rerun_parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    rerun_parser.add_argument(
        "--measurement-point-count",
        type=int,
        default=DEFAULT_MEASUREMENT_POINT_COUNT,
    )
    rerun_parser.add_argument(
        "--storage-path",
        type=Path,
        default=Path("problem_1_experiments") / "optuna_studies" / "pinn_v4_optuna.db",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    storage_url = build_storage_url(args.storage_path)

    if args.command == "dashboard":
        launch_dashboard(storage_url=storage_url, host=args.host, port=args.port)
        return

    if args.command == "rerun-best":
        if args.study_mode == "both":
            modes = ["baseline", "part_e"]
        else:
            modes = [args.study_mode]

        for study_mode in modes:
            rerun_best_trials(
                study_mode=study_mode,
                args=args,
                storage_url=storage_url,
            )
        return

    set_random_seed(args.seed)
    device, dtype = select_device(force_device=args.force_device)
    print(f"Selected device={device}, dtype={dtype}, force_device={args.force_device}")

    if args.study_mode == "both":
        modes = ["baseline", "part_e"]
    else:
        modes = [args.study_mode]

    for study_mode in modes:
        run_study(
            study_mode=study_mode,
            args=args,
            device=device,
            dtype=dtype,
            storage_url=storage_url,
        )

    print("Dashboard command:")
    print(f"  uv run python optuna_pinn_v4.py dashboard --storage-path {args.storage_path}")


if __name__ == "__main__":
    main()
