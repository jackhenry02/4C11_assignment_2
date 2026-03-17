from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch


def _canonicalize_field_batch(field_batch: torch.Tensor) -> torch.Tensor:
    tensor = field_batch.detach().cpu()
    if tensor.ndim == 4 and tensor.shape[-1] == 1:
        tensor = tensor[..., 0]
    elif tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]

    if tensor.ndim != 3:
        raise ValueError(f"Expected decoded fields with shape [N, H, W], got {tuple(tensor.shape)}")
    return tensor


def _predict_decoded_fields(
    net: torch.nn.Module,
    input_tensor: torch.Tensor,
    u_normalizer: Any,
    device: torch.device | str,
    batch_size: int,
) -> torch.Tensor:
    was_training = net.training
    net.eval()

    predictions = []
    effective_batch_size = max(1, int(batch_size))
    with torch.no_grad():
        for start in range(0, input_tensor.shape[0], effective_batch_size):
            input_batch = input_tensor[start : start + effective_batch_size].to(device)
            output_batch = net(input_batch)
            output_batch = u_normalizer.decode(output_batch)
            predictions.append(_canonicalize_field_batch(output_batch))

    if was_training:
        net.train(True)

    return torch.cat(predictions, dim=0)


def _extract_boundary_values(field_batch: torch.Tensor) -> torch.Tensor:
    top = field_batch[:, 0, :]
    bottom = field_batch[:, -1, :]
    left = field_batch[:, 1:-1, 0]
    right = field_batch[:, 1:-1, -1]
    return torch.cat([top, bottom, left, right], dim=1)


def _safe_levels(field: np.ndarray, num_levels: int = 40) -> np.ndarray:
    field_min = float(np.min(field))
    field_max = float(np.max(field))
    if np.isclose(field_min, field_max):
        field_max = field_min + 1e-12
    return np.linspace(field_min, field_max, num_levels)


def _save_figure(fig: plt.Figure, path: Path, save_figures: bool, figure_dpi: int) -> None:
    if save_figures:
        fig.savefig(path, dpi=figure_dpi, bbox_inches="tight")


def run_darcy_posthoc_analysis(
    net: torch.nn.Module,
    a_test: torch.Tensor,
    u_test: torch.Tensor,
    *,
    device: torch.device | str,
    u_normalizer: Any,
    metrics: dict[str, Any],
    metrics_path: str | Path,
    figure_dir: str | Path,
    save_figures: bool = True,
    figure_dpi: int = 200,
    batch_size: int = 20,
    model_name: str = "Darcy model",
) -> dict[str, Any]:
    metrics_path = Path(metrics_path)
    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    error_distribution_path = figure_dir / "posthoc_test_error_distribution.png"
    spatial_error_path = figure_dir / "posthoc_mean_spatial_error.png"
    boundary_violation_path = figure_dir / "posthoc_boundary_violation.png"
    representative_samples_path = figure_dir / "posthoc_representative_samples.png"
    statistics_path = metrics_path.parent / "posthoc_test_statistics.npz"

    truth = _canonicalize_field_batch(u_test)
    prediction = _predict_decoded_fields(net, a_test, u_normalizer, device, batch_size)
    error = prediction - truth
    absolute_error = error.abs()

    num_test_samples = int(truth.shape[0])
    grid = np.linspace(0.0, 1.0, truth.shape[-1])
    X, Y = np.meshgrid(grid, grid, indexing="ij")

    error_flat = error.reshape(num_test_samples, -1)
    truth_flat = truth.reshape(num_test_samples, -1)
    truth_norm = torch.linalg.vector_norm(truth_flat, ord=2, dim=1).clamp_min(1e-12)
    sample_relative_l2 = torch.linalg.vector_norm(error_flat, ord=2, dim=1) / truth_norm

    boundary_values = _extract_boundary_values(prediction)
    boundary_violation = boundary_values.abs().mean(dim=1)

    mean_absolute_error_field = absolute_error.mean(dim=0)
    std_absolute_error_field = absolute_error.std(dim=0, unbiased=False)

    sample_relative_l2_np = sample_relative_l2.numpy()
    boundary_violation_np = boundary_violation.numpy()
    mean_absolute_error_field_np = mean_absolute_error_field.numpy()
    std_absolute_error_field_np = std_absolute_error_field.numpy()

    np.savez(
        statistics_path,
        sample_relative_l2=sample_relative_l2_np,
        boundary_violation=boundary_violation_np,
        mean_absolute_error_field=mean_absolute_error_field_np,
        std_absolute_error_field=std_absolute_error_field_np,
    )

    sorted_sample_errors = np.sort(sample_relative_l2_np)
    empirical_cdf = np.arange(1, len(sorted_sample_errors) + 1, dtype=np.float64) / len(sorted_sample_errors)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].hist(sample_relative_l2_np, bins=min(20, max(10, num_test_samples // 5)), color="tab:blue", edgecolor="black")
    axes[0].axvline(sample_relative_l2_np.mean(), color="crimson", linestyle="--", linewidth=2)
    axes[0].set_title("Per-sample relative L2 error")
    axes[0].set_xlabel("Relative L2 error")
    axes[0].set_ylabel("Count")

    axes[1].plot(sorted_sample_errors, empirical_cdf, color="tab:orange", linewidth=2)
    axes[1].set_title("Empirical CDF of test error")
    axes[1].set_xlabel("Relative L2 error")
    axes[1].set_ylabel("Fraction of test set")
    axes[1].set_ylim(0.0, 1.0)

    fig.suptitle(f"{model_name}: test-set error distribution", fontsize=12)
    fig.tight_layout()
    _save_figure(fig, error_distribution_path, save_figures, figure_dpi)
    plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    mean_plot = axes[0].contourf(
        X,
        Y,
        mean_absolute_error_field_np,
        levels=_safe_levels(mean_absolute_error_field_np),
        cmap="magma",
    )
    axes[0].set_title("Mean absolute error")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.colorbar(mean_plot, ax=axes[0])

    std_plot = axes[1].contourf(
        X,
        Y,
        std_absolute_error_field_np,
        levels=_safe_levels(std_absolute_error_field_np),
        cmap="magma",
    )
    axes[1].set_title("Std. absolute error")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    fig.colorbar(std_plot, ax=axes[1])

    fig.suptitle(f"{model_name}: aggregate spatial error", fontsize=12)
    fig.tight_layout()
    _save_figure(fig, spatial_error_path, save_figures, figure_dpi)
    plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].hist(boundary_violation_np, bins=min(20, max(10, num_test_samples // 5)), color="tab:green", edgecolor="black")
    axes[0].set_title("Boundary violation histogram")
    axes[0].set_xlabel("Mean |u| on boundary")
    axes[0].set_ylabel("Count")

    axes[1].boxplot(boundary_violation_np, vert=False)
    axes[1].set_title("Boundary violation boxplot")
    axes[1].set_xlabel("Mean |u| on boundary")

    fig.suptitle(f"{model_name}: boundary-condition check", fontsize=12)
    fig.tight_layout()
    _save_figure(fig, boundary_violation_path, save_figures, figure_dpi)
    plt.show()
    plt.close(fig)

    sorted_indices = torch.argsort(sample_relative_l2)
    representative_positions = [0, len(sorted_indices) // 2, len(sorted_indices) - 1]
    representative_labels = ["Best", "Median", "Worst"]
    representative_indices = [int(sorted_indices[position].item()) for position in representative_positions]

    fig, axes = plt.subplots(3, 3, figsize=(12, 11))
    for row, (label, sample_index) in enumerate(zip(representative_labels, representative_indices)):
        truth_field = truth[sample_index].numpy()
        prediction_field = prediction[sample_index].numpy()
        absolute_error_field = absolute_error[sample_index].numpy()

        field_min = min(float(np.min(truth_field)), float(np.min(prediction_field)))
        field_max = max(float(np.max(truth_field)), float(np.max(prediction_field)))
        if np.isclose(field_min, field_max):
            field_max = field_min + 1e-12
        solution_levels = np.linspace(field_min, field_max, 40)

        truth_plot = axes[row, 0].contourf(X, Y, truth_field, levels=solution_levels, cmap="viridis")
        axes[row, 0].set_title(f"{label}: truth")
        axes[row, 0].set_xlabel("x")
        axes[row, 0].set_ylabel("y")
        fig.colorbar(truth_plot, ax=axes[row, 0])

        prediction_plot = axes[row, 1].contourf(X, Y, prediction_field, levels=solution_levels, cmap="viridis")
        axes[row, 1].set_title(f"{label}: prediction")
        axes[row, 1].set_xlabel("x")
        axes[row, 1].set_ylabel("y")
        fig.colorbar(prediction_plot, ax=axes[row, 1])

        error_plot = axes[row, 2].contourf(
            X,
            Y,
            absolute_error_field,
            levels=_safe_levels(absolute_error_field),
            cmap="magma",
        )
        axes[row, 2].set_title(f"{label}: |error|, rel. L2 = {sample_relative_l2_np[sample_index]:.4f}")
        axes[row, 2].set_xlabel("x")
        axes[row, 2].set_ylabel("y")
        fig.colorbar(error_plot, ax=axes[row, 2])

    fig.suptitle(f"{model_name}: representative test samples", fontsize=12)
    fig.tight_layout()
    _save_figure(fig, representative_samples_path, save_figures, figure_dpi)
    plt.show()
    plt.close(fig)

    posthoc_metrics = {
        "sample_relative_l2": {
            "mean": float(sample_relative_l2.mean().item()),
            "median": float(sample_relative_l2.median().item()),
            "std": float(sample_relative_l2.std(unbiased=False).item()),
            "min": float(sample_relative_l2.min().item()),
            "max": float(sample_relative_l2.max().item()),
        },
        "boundary_violation": {
            "mean": float(boundary_violation.mean().item()),
            "median": float(boundary_violation.median().item()),
            "std": float(boundary_violation.std(unbiased=False).item()),
            "max": float(boundary_violation.max().item()),
        },
        "mean_absolute_error_field": {
            "overall_mean": float(mean_absolute_error_field.mean().item()),
            "max": float(mean_absolute_error_field.max().item()),
        },
        "representative_sample_indices": {
            "best": representative_indices[0],
            "median": representative_indices[1],
            "worst": representative_indices[2],
        },
        "artifact_paths": {
            "statistics_path": str(statistics_path),
            "error_distribution_figure_path": str(error_distribution_path) if save_figures else None,
            "spatial_error_figure_path": str(spatial_error_path) if save_figures else None,
            "boundary_violation_figure_path": str(boundary_violation_path) if save_figures else None,
            "representative_samples_figure_path": str(representative_samples_path) if save_figures else None,
        },
    }

    metrics["posthoc_analysis"] = posthoc_metrics
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Mean per-sample relative L2 error: {posthoc_metrics['sample_relative_l2']['mean']:.6f}")
    print(f"Median per-sample relative L2 error: {posthoc_metrics['sample_relative_l2']['median']:.6f}")
    print(f"Mean boundary violation: {posthoc_metrics['boundary_violation']['mean']:.6f}")
    print(f"Saved post-hoc statistics to: {statistics_path}")

    return posthoc_metrics
