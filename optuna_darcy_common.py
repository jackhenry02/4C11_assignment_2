from __future__ import annotations

import argparse
import copy
import datetime as dt
import gc
import json
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Any

import h5py
import matplotlib
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.utils.data as data
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DEFAULT_TRAIN_PATH = Path("Coursework2/Coursework2_Problem_2/Darcy_2D_data_train.mat")
DEFAULT_TEST_PATH = Path("Coursework2/Coursework2_Problem_2/Darcy_2D_data_test.mat")
DEFAULT_VAL_FRACTION = 0.15
DEFAULT_SEED = 1234
DEFAULT_N_TRIALS = 20
DEFAULT_EPOCHS = 100
DEFAULT_REPORT_EVERY = 10
DEFAULT_TEST_SAMPLE_INDEX = 0
DEFAULT_FIGURE_DPI = 200
OBJECTIVE_NAME = "validation_relative_l2"


@dataclass(frozen=True)
class StudyPaths:
    study_root: Path
    figures_dir: Path
    trials_dir: Path
    reports_dir: Path


@dataclass
class DarcyStudyData:
    a_train: torch.Tensor
    u_train: torch.Tensor
    a_val: torch.Tensor
    u_val: torch.Tensor
    a_test: torch.Tensor
    u_test: torch.Tensor
    u_normalizer: "UnitGaussianNormalizer"


MODEL_SPECS: dict[str, dict[str, Any]] = {
    "cnn_simple": {
        "display_name": "Darcy Simple CNN",
        "study_name": "darcy_cnn_simple_optuna",
        "model_name": "darcy_cnn_simple",
        "default_device_policy": "accelerator",
        "search_space": {
            "base_channels": {
                "type": "categorical",
                "choices": [16, 24, 32, 48],
                "description": "Base channel width for the encoder-decoder CNN.",
            },
            "learning_rate": {
                "type": "float",
                "low": 3.0e-4,
                "high": 3.0e-3,
                "log": True,
                "description": "Adam learning rate.",
            },
            "weight_decay": {
                "type": "categorical",
                "choices": [0.0, 1.0e-6, 1.0e-5, 1.0e-4],
                "description": "Adam weight decay.",
            },
            "batch_size": {
                "type": "categorical",
                "choices": [10, 20, 40],
                "description": "Mini-batch size.",
            },
            "scheduler": {
                "type": "categorical",
                "choices": ["none", "step"],
                "description": "Use no scheduler or StepLR.",
            },
            "lr_gamma": {
                "type": "float",
                "low": 0.3,
                "high": 0.8,
                "description": "StepLR multiplicative decay factor.",
            },
            "lr_step_size": {
                "type": "categorical",
                "choices": [25, 50, 75],
                "description": "StepLR step size for 100-epoch trials.",
            },
        },
        "contour_params": ["learning_rate", "base_channels"],
    },
    "unet": {
        "display_name": "Darcy U-Net",
        "study_name": "darcy_unet_optuna",
        "model_name": "darcy_cnn_unet",
        "default_device_policy": "accelerator",
        "search_space": {
            "base_channels": {
                "type": "categorical",
                "choices": [16, 24, 32, 48],
                "description": "Base channel width for the U-Net.",
            },
            "learning_rate": {
                "type": "float",
                "low": 3.0e-4,
                "high": 2.0e-3,
                "log": True,
                "description": "Adam learning rate.",
            },
            "weight_decay": {
                "type": "categorical",
                "choices": [0.0, 1.0e-6, 1.0e-5, 1.0e-4],
                "description": "Adam weight decay.",
            },
            "batch_size": {
                "type": "categorical",
                "choices": [10, 20, 40],
                "description": "Mini-batch size.",
            },
            "scheduler": {
                "type": "categorical",
                "choices": ["none", "step"],
                "description": "Use no scheduler or StepLR.",
            },
            "lr_gamma": {
                "type": "float",
                "low": 0.3,
                "high": 0.8,
                "description": "StepLR multiplicative decay factor.",
            },
            "lr_step_size": {
                "type": "categorical",
                "choices": [25, 50, 75],
                "description": "StepLR step size for 100-epoch trials.",
            },
        },
        "contour_params": ["learning_rate", "base_channels"],
    },
    "fno": {
        "display_name": "Darcy FNO",
        "study_name": "darcy_fno_optuna",
        "model_name": "darcy_fno",
        "default_device_policy": "safe_fft",
        "search_space": {
            "modes": {
                "type": "categorical",
                "choices": [6, 8, 10, 12],
                "description": "Number of retained Fourier modes in each spatial direction.",
            },
            "width": {
                "type": "categorical",
                "choices": [16, 24, 32, 40],
                "description": "Latent channel width for the FNO.",
            },
            "learning_rate": {
                "type": "float",
                "low": 1.0e-4,
                "high": 1.0e-3,
                "log": True,
                "description": "Adam learning rate.",
            },
            "weight_decay": {
                "type": "categorical",
                "choices": [0.0, 1.0e-6, 1.0e-5, 1.0e-4],
                "description": "Adam weight decay.",
            },
            "batch_size": {
                "type": "categorical",
                "choices": [10, 20],
                "description": "Mini-batch size.",
            },
            "scheduler": {
                "type": "categorical",
                "choices": ["none", "step"],
                "description": "Use no scheduler or StepLR.",
            },
            "lr_gamma": {
                "type": "float",
                "low": 0.3,
                "high": 0.8,
                "description": "StepLR multiplicative decay factor.",
            },
            "lr_step_size": {
                "type": "categorical",
                "choices": [25, 50, 75],
                "description": "StepLR step size for 100-epoch trials.",
            },
        },
        "contour_params": ["learning_rate", "modes"],
    },
}


class LpLoss:
    def __init__(self, d: int = 2, p: int = 2, size_average: bool = True, reduction: bool = True) -> None:
        assert d > 0 and p > 0
        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples, -1) - y.reshape(num_examples, -1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples, -1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms / y_norms)
            return torch.sum(diff_norms / y_norms)

        return diff_norms / y_norms

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.rel(x, y)


class UnitGaussianNormalizer:
    def __init__(self, x: torch.Tensor, eps: float = 1.0e-5) -> None:
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.std + self.eps)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return (x * (self.std + self.eps)) + self.mean

    def to(self, device: torch.device) -> "UnitGaussianNormalizer":
        self.mean = self.mean.to(device)
        self.std = self.std.to(device)
        return self


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SimpleCNN(nn.Module):
    def __init__(self, channel_width: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, channel_width)
        self.enc2 = ConvBlock(channel_width, 2 * channel_width)
        self.bottleneck = ConvBlock(2 * channel_width, 4 * channel_width)
        self.dec1 = ConvBlock(4 * channel_width, 2 * channel_width)
        self.dec2 = ConvBlock(2 * channel_width, channel_width)
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.out = nn.Conv2d(channel_width, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.enc1(x)
        x = self.pool(x)
        x = self.enc2(x)
        x = self.pool(x)
        x = self.bottleneck(x)
        x = self.up1(x)
        x = self.dec1(x)
        x = self.up2(x)
        x = self.dec2(x)
        return self.out(x).squeeze(1)


class UNet(nn.Module):
    def __init__(self, channel_width: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, channel_width)
        self.enc2 = ConvBlock(channel_width, 2 * channel_width)
        self.bottleneck = ConvBlock(2 * channel_width, 4 * channel_width)
        self.dec1 = ConvBlock(4 * channel_width + 2 * channel_width, 2 * channel_width)
        self.dec2 = ConvBlock(2 * channel_width + channel_width, channel_width)
        self.pool = nn.MaxPool2d(2)
        self.up1 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.up2 = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.out = nn.Conv2d(channel_width, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.bottleneck(self.pool(x2))
        x = self.up1(x3)
        x = torch.cat((x, x2), dim=1)
        x = self.dec1(x)
        x = self.up2(x)
        x = torch.cat((x, x1), dim=1)
        x = self.dec2(x)
        return self.out(x).squeeze(1)


class SpectralConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes1: int, modes2: int) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2

        scale = 1.0 / (in_channels * out_channels)
        self.weights1 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.rand(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def compl_mul2d(self, input_tensor: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bixy,ioxy->boxy", input_tensor, weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        x_ft = torch.fft.rfft2(x)
        out_ft = torch.zeros(
            batch_size,
            self.out_channels,
            x.size(-2),
            x.size(-1) // 2 + 1,
            dtype=x_ft.dtype,
            device=x.device,
        )
        out_ft[:, :, : self.modes1, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, : self.modes1, : self.modes2],
            self.weights1,
        )
        out_ft[:, :, -self.modes1 :, : self.modes2] = self.compl_mul2d(
            x_ft[:, :, -self.modes1 :, : self.modes2],
            self.weights2,
        )
        return torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))


class PointwiseMLP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, mid_channels: int) -> None:
        super().__init__()
        self.mlp1 = nn.Conv2d(in_channels, mid_channels, 1)
        self.mlp2 = nn.Conv2d(mid_channels, out_channels, 1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mlp1(x)
        x = self.act(x)
        x = self.mlp2(x)
        return x


class FNO(nn.Module):
    def __init__(self, modes1: int, modes2: int, width: int) -> None:
        super().__init__()
        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.p = nn.Linear(3, width)
        self.conv0 = SpectralConv2d(width, width, modes1, modes2)
        self.conv1 = SpectralConv2d(width, width, modes1, modes2)
        self.conv2 = SpectralConv2d(width, width, modes1, modes2)
        self.conv3 = SpectralConv2d(width, width, modes1, modes2)
        self.mlp0 = PointwiseMLP(width, width, width)
        self.mlp1 = PointwiseMLP(width, width, width)
        self.mlp2 = PointwiseMLP(width, width, width)
        self.mlp3 = PointwiseMLP(width, width, width)
        self.w0 = nn.Conv2d(width, width, 1)
        self.w1 = nn.Conv2d(width, width, 1)
        self.w2 = nn.Conv2d(width, width, 1)
        self.w3 = nn.Conv2d(width, width, 1)
        self.act0 = nn.GELU()
        self.act1 = nn.GELU()
        self.act2 = nn.GELU()
        self.q = PointwiseMLP(width, 1, width * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x.unsqueeze(-1), grid), dim=-1)
        x = self.p(x)
        x = x.permute(0, 3, 1, 2)

        x1 = self.mlp0(self.conv0(x))
        x = self.act0(x1 + self.w0(x))

        x1 = self.mlp1(self.conv1(x))
        x = self.act1(x1 + self.w1(x))

        x1 = self.mlp2(self.conv2(x))
        x = self.act2(x1 + self.w2(x))

        x1 = self.mlp3(self.conv3(x))
        x = x1 + self.w3(x)

        x = self.q(x)
        return x.squeeze(1)

    @staticmethod
    def get_grid(shape: torch.Size, device: torch.device) -> torch.Tensor:
        batch_size, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.linspace(0, 1, size_x, dtype=torch.float32, device=device)
        gridx = gridx.view(1, size_x, 1, 1).repeat(batch_size, 1, size_y, 1)
        gridy = torch.linspace(0, 1, size_y, dtype=torch.float32, device=device)
        gridy = gridy.view(1, 1, size_y, 1).repeat(batch_size, size_x, 1, 1)
        return torch.cat((gridx, gridy), dim=-1)


def set_random_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def maybe_empty_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
    elif device.type == "mps":
        torch.mps.empty_cache()


def make_default_storage_path(model_key: str) -> Path:
    spec = MODEL_SPECS[model_key]
    return Path("problem_2_experiments") / "optuna_studies" / spec["study_name"] / f"{spec['study_name']}.db"


def build_storage_url(storage_path: Path) -> str:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{storage_path.as_posix()}"


def build_study_paths(model_key: str) -> StudyPaths:
    spec = MODEL_SPECS[model_key]
    study_root = Path("problem_2_experiments") / "optuna_studies" / spec["study_name"]
    figures_dir = study_root / "figures"
    trials_dir = study_root / "trials"
    reports_dir = study_root / "reports"
    for directory in [study_root, figures_dir, trials_dir, reports_dir]:
        directory.mkdir(parents=True, exist_ok=True)
    return StudyPaths(
        study_root=study_root,
        figures_dir=figures_dir,
        trials_dir=trials_dir,
        reports_dir=reports_dir,
    )


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def save_matplotlib_figure(path: Path, figure: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path, dpi=DEFAULT_FIGURE_DPI, bbox_inches="tight")
    plt.close(figure)


def load_field_from_mat(path: Path, dataset_name: str) -> torch.Tensor:
    with h5py.File(path, "r") as handle:
        array = np.array(handle[dataset_name]).T
    return torch.tensor(array, dtype=torch.float32)


def split_train_validation(
    a_train: torch.Tensor,
    u_train: torch.Tensor,
    *,
    val_fraction: float,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_samples = a_train.shape[0]
    val_count = max(1, int(num_samples * val_fraction))
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator)
    val_index = permutation[:val_count]
    train_index = permutation[val_count:]
    return a_train[train_index], u_train[train_index], a_train[val_index], u_train[val_index]


def prepare_darcy_data(
    *,
    train_path: Path,
    test_path: Path,
    val_fraction: float,
    seed: int,
    device: torch.device,
) -> DarcyStudyData:
    a_train_raw = load_field_from_mat(train_path, "a_field")
    u_train = load_field_from_mat(train_path, "u_field")
    a_test_raw = load_field_from_mat(test_path, "a_field")
    u_test = load_field_from_mat(test_path, "u_field")

    a_normalizer = UnitGaussianNormalizer(a_train_raw)
    a_train = a_normalizer.encode(a_train_raw)
    a_test = a_normalizer.encode(a_test_raw)

    u_normalizer = UnitGaussianNormalizer(u_train).to(device)
    a_train_split, u_train_split, a_val, u_val = split_train_validation(
        a_train,
        u_train,
        val_fraction=val_fraction,
        seed=seed,
    )
    return DarcyStudyData(
        a_train=a_train_split,
        u_train=u_train_split,
        a_val=a_val,
        u_val=u_val,
        a_test=a_test,
        u_test=u_test,
        u_normalizer=u_normalizer,
    )


def select_device(model_key: str, requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)

    policy = MODEL_SPECS[model_key]["default_device_policy"]
    if policy == "safe_fft":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(model_key: str, config: dict[str, Any], device: torch.device) -> nn.Module:
    if model_key == "cnn_simple":
        model = SimpleCNN(channel_width=int(config["base_channels"]))
    elif model_key == "unet":
        model = UNet(channel_width=int(config["base_channels"]))
    elif model_key == "fno":
        model = FNO(int(config["modes"]), int(config["modes"]), int(config["width"]))
    else:
        raise ValueError(f"Unknown model_key={model_key}")
    return model.to(device)


def evaluate_model(
    model: nn.Module,
    *,
    input_tensor: torch.Tensor,
    target_tensor: torch.Tensor,
    device: torch.device,
    u_normalizer: UnitGaussianNormalizer,
    loss_func: LpLoss,
) -> float:
    model.eval()
    with torch.no_grad():
        output = model(input_tensor.to(device))
        output = u_normalizer.decode(output)
        loss = loss_func(output, target_tensor.to(device)).item()
    return float(loss)


def build_optimizer_and_scheduler(
    model: nn.Module,
    config: dict[str, Any],
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler | None]:
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["learning_rate"]),
        weight_decay=float(config["weight_decay"]),
    )
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None
    if config["scheduler"] == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(config["lr_step_size"]),
            gamma=float(config["lr_gamma"]),
        )
    return optimizer, scheduler


def save_trial_loss_curve(path: Path, epochs: list[int], train_losses: list[float], val_losses: list[float]) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(epochs, train_losses, label="train")
    axis.plot(epochs, val_losses, label="validation")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Relative $L_2$")
    axis.set_title("Trial loss history")
    axis.grid(True, alpha=0.25)
    axis.legend()
    save_matplotlib_figure(path, figure)


def save_prediction_figure(
    *,
    path: Path,
    model: nn.Module,
    data_bundle: DarcyStudyData,
    device: torch.device,
    test_sample_index: int,
) -> None:
    sample_count = int(data_bundle.a_test.shape[0])
    safe_index = test_sample_index % sample_count
    model.eval()
    with torch.no_grad():
        sample_input = data_bundle.a_test[safe_index : safe_index + 1].to(device)
        prediction = model(sample_input)
        prediction = data_bundle.u_normalizer.decode(prediction).cpu().squeeze(0)
    truth = data_bundle.u_test[safe_index].cpu()
    error = torch.abs(prediction - truth)

    figure, axes = plt.subplots(1, 3, figsize=(13, 4))
    plots = [
        (truth, "Truth"),
        (prediction, "Prediction"),
        (error, "Absolute error"),
    ]
    for axis, (tensor, title) in zip(axes, plots, strict=True):
        image = axis.imshow(tensor.numpy(), origin="lower", cmap="viridis")
        axis.set_title(title)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    save_matplotlib_figure(path, figure)


def suggest_trial_config(model_key: str, trial: optuna.Trial) -> dict[str, Any]:
    config: dict[str, Any] = {}
    if model_key in {"cnn_simple", "unet"}:
        config["base_channels"] = trial.suggest_categorical("base_channels", [16, 24, 32, 48])
    else:
        config["modes"] = trial.suggest_categorical("modes", [6, 8, 10, 12])
        config["width"] = trial.suggest_categorical("width", [16, 24, 32, 40])

    if model_key == "cnn_simple":
        config["learning_rate"] = trial.suggest_float("learning_rate", 3.0e-4, 3.0e-3, log=True)
        config["batch_size"] = trial.suggest_categorical("batch_size", [10, 20, 40])
    elif model_key == "unet":
        config["learning_rate"] = trial.suggest_float("learning_rate", 3.0e-4, 2.0e-3, log=True)
        config["batch_size"] = trial.suggest_categorical("batch_size", [10, 20, 40])
    else:
        config["learning_rate"] = trial.suggest_float("learning_rate", 1.0e-4, 1.0e-3, log=True)
        config["batch_size"] = trial.suggest_categorical("batch_size", [10, 20])

    config["weight_decay"] = trial.suggest_categorical("weight_decay", [0.0, 1.0e-6, 1.0e-5, 1.0e-4])
    config["scheduler"] = trial.suggest_categorical("scheduler", ["none", "step"])
    if config["scheduler"] == "step":
        config["lr_gamma"] = trial.suggest_float("lr_gamma", 0.3, 0.8)
        config["lr_step_size"] = trial.suggest_categorical("lr_step_size", [25, 50, 75])
    else:
        config["lr_gamma"] = 1.0
        config["lr_step_size"] = 10**9
    return config


def run_single_trial(
    *,
    model_key: str,
    config: dict[str, Any],
    trial: optuna.Trial,
    trial_dir: Path,
    data_bundle: DarcyStudyData,
    device: torch.device,
    epochs: int,
    report_every: int,
    test_sample_index: int,
) -> dict[str, Any]:
    loss_func = LpLoss()
    model = build_model(model_key, config, device)
    optimizer, scheduler = build_optimizer_and_scheduler(model, config)

    train_set = data.TensorDataset(data_bundle.a_train, data_bundle.u_train)
    train_loader = data.DataLoader(
        train_set,
        batch_size=int(config["batch_size"]),
        shuffle=True,
    )

    best_val_loss = float("inf")
    best_state_dict: dict[str, torch.Tensor] | None = None
    train_losses: list[float] = []
    val_losses: list[float] = []
    epoch_indices: list[int] = []
    start_time = time()

    for epoch in range(epochs):
        model.train(True)
        train_loss = 0.0
        for input_tensor, target_tensor in train_loader:
            input_tensor = input_tensor.to(device)
            target_tensor = target_tensor.to(device)

            output = model(input_tensor)
            output = data_bundle.u_normalizer.decode(output)
            loss = loss_func(output, target_tensor)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)
        if scheduler is not None:
            scheduler.step()

        val_loss = evaluate_model(
            model,
            input_tensor=data_bundle.a_val,
            target_tensor=data_bundle.u_val,
            device=device,
            u_normalizer=data_bundle.u_normalizer,
            loss_func=loss_func,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = copy.deepcopy(model.state_dict())

        epoch_number = epoch + 1
        train_losses.append(float(train_loss))
        val_losses.append(float(val_loss))
        epoch_indices.append(epoch_number)

        if epoch_number % report_every == 0 or epoch_number == epochs:
            trial.report(val_loss, epoch_number)
            if trial.should_prune():
                save_trial_loss_curve(trial_dir / "figures" / "loss_curve.png", epoch_indices, train_losses, val_losses)
                np.savez(
                    trial_dir / "metrics" / "loss_history.npz",
                    epochs=np.array(epoch_indices),
                    train=np.array(train_losses),
                    validation=np.array(val_losses),
                )
                write_json(
                    trial_dir / "metrics" / "trial_metrics.json",
                    {
                        "objective_name": OBJECTIVE_NAME,
                        "state": "pruned",
                        "best_validation_loss": best_val_loss,
                        "last_validation_loss": val_loss,
                        "epochs_completed": epoch_number,
                    },
                )
                raise optuna.TrialPruned(f"Pruned at epoch {epoch_number}")

    final_state_dict = copy.deepcopy(model.state_dict())
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    validation_loss = evaluate_model(
        model,
        input_tensor=data_bundle.a_val,
        target_tensor=data_bundle.u_val,
        device=device,
        u_normalizer=data_bundle.u_normalizer,
        loss_func=loss_func,
    )
    test_loss = evaluate_model(
        model,
        input_tensor=data_bundle.a_test,
        target_tensor=data_bundle.u_test,
        device=device,
        u_normalizer=data_bundle.u_normalizer,
        loss_func=loss_func,
    )
    total_seconds = time() - start_time
    n_params = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    (trial_dir / "weights").mkdir(parents=True, exist_ok=True)
    (trial_dir / "metrics").mkdir(parents=True, exist_ok=True)
    (trial_dir / "figures").mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), trial_dir / "weights" / "best_validation_model.pt")
    torch.save(final_state_dict, trial_dir / "weights" / "final_model.pt")
    np.savez(
        trial_dir / "metrics" / "loss_history.npz",
        epochs=np.array(epoch_indices),
        train=np.array(train_losses),
        validation=np.array(val_losses),
    )
    save_trial_loss_curve(trial_dir / "figures" / "loss_curve.png", epoch_indices, train_losses, val_losses)
    save_prediction_figure(
        path=trial_dir / "figures" / "prediction_comparison.png",
        model=model,
        data_bundle=data_bundle,
        device=device,
        test_sample_index=test_sample_index,
    )
    metrics = {
        "objective_name": OBJECTIVE_NAME,
        "state": "complete",
        "best_validation_loss": validation_loss,
        "final_validation_loss_history_value": val_losses[-1],
        "final_training_loss": train_losses[-1],
        "test_loss": test_loss,
        "epochs": epochs,
        "seconds": total_seconds,
        "training_time": str(dt.timedelta(seconds=int(total_seconds))),
        "n_params": n_params,
    }
    write_json(trial_dir / "metrics" / "trial_metrics.json", metrics)
    return metrics


def save_study_visualizations(model_key: str, study: optuna.Study, study_paths: StudyPaths) -> None:
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        return

    plot_builders: dict[str, Any] = {
        "optimization_history.png": lambda: plot_optimization_history(study).figure,
        "intermediate_values.png": lambda: plot_intermediate_values(study).figure,
        "edf.png": lambda: plot_edf(study).figure,
        "timeline.png": lambda: plot_timeline(study).figure,
    }

    if len(completed_trials) >= 2:
        plot_builders["param_importances.png"] = lambda: plot_param_importances(study).figure

    params = list(study.best_params.keys())
    if len(params) >= 2:
        plot_builders["parallel_coordinate.png"] = lambda: plot_parallel_coordinate(study, params=params).figure
        plot_builders["slice.png"] = lambda: plot_slice(study, params=params).figure

    contour_candidates = [
        parameter
        for parameter in MODEL_SPECS[model_key]["contour_params"]
        if parameter in study.best_params
    ]
    if len(contour_candidates) >= 2:
        plot_builders["contour.png"] = lambda: plot_contour(study, params=contour_candidates[:2]).figure

    for filename, builder in plot_builders.items():
        try:
            figure = builder()
            save_matplotlib_figure(study_paths.figures_dir / filename, figure)
        except Exception as exc:  # pragma: no cover - visualization errors should not kill the study
            print(f"Skipping study plot {filename}: {exc}")


def get_best_trial_or_none(study: optuna.Study) -> optuna.trial.FrozenTrial | None:
    completed_trials = [trial for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE]
    if not completed_trials:
        return None
    return study.best_trial


def save_study_reports(
    *,
    model_key: str,
    study: optuna.Study,
    study_paths: StudyPaths,
    storage_url: str,
    args: argparse.Namespace,
) -> None:
    trials_df = study.trials_dataframe()
    trials_df.to_csv(study_paths.reports_dir / "trials.csv", index=False)
    write_json(
        study_paths.reports_dir / "study_config.json",
        {
            "study_name": study.study_name,
            "display_name": MODEL_SPECS[model_key]["display_name"],
            "model_key": model_key,
            "objective_name": OBJECTIVE_NAME,
            "storage_url": storage_url,
            "device": args.device,
            "epochs_per_trial": args.epochs,
            "n_trials": args.n_trials,
            "val_fraction": args.val_fraction,
            "seed": args.seed,
            "train_path": args.train_path,
            "test_path": args.test_path,
        },
    )
    write_json(study_paths.reports_dir / "search_space.json", MODEL_SPECS[model_key]["search_space"])

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
    model_key: str,
    args: argparse.Namespace,
    device: torch.device,
    storage_url: str,
) -> None:
    spec = MODEL_SPECS[model_key]
    study_paths = build_study_paths(model_key)
    write_json(study_paths.reports_dir / "search_space.json", spec["search_space"])

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
        study_name=spec["study_name"],
        storage=storage_url,
        load_if_exists=True,
        direction="minimize",
        sampler=TPESampler(seed=args.seed, multivariate=True),
        pruner=pruner,
    )

    data_bundle = prepare_darcy_data(
        train_path=args.train_path,
        test_path=args.test_path,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=device,
    )

    def objective(trial: optuna.Trial) -> float:
        set_random_seed(args.seed + trial.number)
        config = suggest_trial_config(model_key, trial)
        trial_dir = study_paths.trials_dir / f"trial_{trial.number:04d}"
        (trial_dir / "figures").mkdir(parents=True, exist_ok=True)
        (trial_dir / "weights").mkdir(parents=True, exist_ok=True)
        (trial_dir / "metrics").mkdir(parents=True, exist_ok=True)

        write_json(
            trial_dir / "trial_config.json",
            {
                "model_key": model_key,
                "display_name": spec["display_name"],
                "trial_number": trial.number,
                "config": config,
                "epochs": args.epochs,
                "report_every": args.report_every,
                "seed": args.seed + trial.number,
                "device": device,
                "train_path": args.train_path,
                "test_path": args.test_path,
                "val_fraction": args.val_fraction,
            },
        )

        metrics = run_single_trial(
            model_key=model_key,
            config=config,
            trial=trial,
            trial_dir=trial_dir,
            data_bundle=data_bundle,
            device=device,
            epochs=args.epochs,
            report_every=args.report_every,
            test_sample_index=args.test_sample_index,
        )

        trial.set_user_attr("trial_dir", str(trial_dir))
        trial.set_user_attr("test_loss", float(metrics["test_loss"]))
        trial.set_user_attr("n_params", int(metrics["n_params"]))
        trial.set_user_attr("training_time_seconds", float(metrics["seconds"]))
        return float(metrics["best_validation_loss"])

    study.optimize(
        objective,
        n_trials=args.n_trials,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    save_study_visualizations(model_key, study, study_paths)
    save_study_reports(
        model_key=model_key,
        study=study,
        study_paths=study_paths,
        storage_url=storage_url,
        args=args,
    )
    maybe_empty_cache(device)
    gc.collect()

    best_trial = get_best_trial_or_none(study)
    if best_trial is not None:
        print(f"[{spec['study_name']}] best {OBJECTIVE_NAME} = {best_trial.value:.6e}")
        print(f"[{spec['study_name']}] best params = {best_trial.params}")
    print(f"[{spec['study_name']}] dashboard storage = {storage_url}")
    print(f"[{spec['study_name']}] saved reports under {study_paths.study_root}")


def build_parser(model_key: str) -> argparse.ArgumentParser:
    spec = MODEL_SPECS[model_key]
    default_storage_path = make_default_storage_path(model_key)

    parser = argparse.ArgumentParser(
        description=f"Optuna study runner for {spec['display_name']}.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    optimize_parser = subparsers.add_parser(
        "optimize",
        help="Run the Optuna hyperparameter study.",
    )
    optimize_parser.add_argument("--n-trials", type=int, default=DEFAULT_N_TRIALS)
    optimize_parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    optimize_parser.add_argument("--report-every", type=int, default=DEFAULT_REPORT_EVERY)
    optimize_parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    optimize_parser.add_argument("--train-path", type=Path, default=DEFAULT_TRAIN_PATH)
    optimize_parser.add_argument("--test-path", type=Path, default=DEFAULT_TEST_PATH)
    optimize_parser.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    optimize_parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    optimize_parser.add_argument("--storage-path", type=Path, default=default_storage_path)
    optimize_parser.add_argument("--test-sample-index", type=int, default=DEFAULT_TEST_SAMPLE_INDEX)
    optimize_parser.add_argument(
        "--disable-pruning",
        action="store_true",
        help="Disable Optuna pruning and run every trial to completion.",
    )

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Launch the Optuna dashboard for this study's SQLite storage.",
    )
    dashboard_parser.add_argument("--storage-path", type=Path, default=default_storage_path)
    dashboard_parser.add_argument("--host", default="127.0.0.1")
    dashboard_parser.add_argument("--port", type=int, default=8080)

    return parser


def main_for_model(model_key: str) -> None:
    parser = build_parser(model_key)
    args = parser.parse_args()
    storage_url = build_storage_url(args.storage_path)

    if args.command == "dashboard":
        print(f"Starting Optuna dashboard at http://{args.host}:{args.port}")
        print(f"Using storage: {storage_url}")
        run_server(storage=storage_url, host=args.host, port=args.port)
        return

    set_random_seed(args.seed)
    device = select_device(model_key, args.device)
    print(f"Selected device={device} for model_key={model_key}")
    run_study(
        model_key=model_key,
        args=args,
        device=device,
        storage_url=storage_url,
    )
    print("Dashboard command:")
    print(
        "  "
        f"uv run python optuna_darcy_{'cnn_simple' if model_key == 'cnn_simple' else model_key}.py "
        f"dashboard --storage-path {args.storage_path}"
    )
