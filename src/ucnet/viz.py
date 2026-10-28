"""
Scientific visualization utilities for UCNet:
- Coverage & reliability diagrams
- PIT (Probability Integral Transform) histograms
- Predictive mean and uncertainty breakdown plots (aleatoric vs epistemic)
- Training convergence curves
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Union

import matplotlib.pyplot as plt
import numpy as np


def plot_calibration_curve(
    pit_values: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Regression Reliability Diagram (Calibration Curve)",
    num_bins: int = 50,
) -> plt.Figure:
    """Plots empirical coverage vs nominal probability level."""
    u = np.sort(np.asarray(pit_values).ravel())
    N = len(u)
    p_levels = np.linspace(0.0, 1.0, num_bins)
    p_hat = np.searchsorted(u, p_levels, side="right") / N

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (Ideal)", alpha=0.7)
    ax.plot(p_levels, p_hat, "b-", linewidth=2, label="Model Empirical Coverage")

    ax.set_xlabel("Nominal Confidence Level $p$")
    ax.set_ylabel("Empirical Coverage $\\hat{p}$")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig


def plot_pit_histogram(
    pit_values: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Probability Integral Transform (PIT) Histogram",
    bins: int = 20,
) -> plt.Figure:
    """
    Plots PIT histogram.
    Uniform: well-calibrated.
    U-shaped: overconfident (underdispersed).
    Inverted-U: underconfident (overdispersed).
    """
    u = np.asarray(pit_values).ravel()
    fig, ax = plt.subplots(figsize=(6, 4))
    counts, bin_edges, _ = ax.hist(u, bins=bins, density=True, color="#4C72B0", edgecolor="white", alpha=0.8)

    # Reference uniform density = 1.0
    ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Ideal Uniform U(0,1)")

    ax.set_xlabel("PIT Value $u = F(y)$")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.set_xlim([0.0, 1.0])
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper center")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig


def plot_1d_uncertainty(
    x_test: np.ndarray,
    y_test: np.ndarray,
    mu: np.ndarray,
    total_std: np.ndarray,
    aleatoric_std: Optional[np.ndarray] = None,
    x_train: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Predictive Mean and Uncertainty Bounds",
) -> plt.Figure:
    """Plots 1D regression with predictive mean and aleatoric/epistemic uncertainty ribbons."""
    idx = np.argsort(x_test.ravel())
    x_s = x_test.ravel()[idx]
    y_s = y_test.ravel()[idx]
    mu_s = mu.ravel()[idx]
    tot_s = total_std.ravel()[idx]

    fig, ax = plt.subplots(figsize=(9, 5))

    # Scatter test points
    ax.scatter(x_s, y_s, color="black", s=15, alpha=0.6, label="Test Observations")

    if x_train is not None and y_train is not None:
        ax.scatter(x_train.ravel(), y_train.ravel(), color="gray", s=10, alpha=0.3, label="Train Points")

    # Predictive mean
    ax.plot(x_s, mu_s, "b-", linewidth=2, label="Predictive Mean $\\mu(x)$")

    # Total uncertainty (2 sigma): filled, with explicit boundary lines so it stays visible even when
    # the epistemic share is small and the aleatoric ribbon drawn on top almost coincides with it.
    ax.fill_between(
        x_s,
        mu_s - 2.0 * tot_s,
        mu_s + 2.0 * tot_s,
        color="royalblue",
        alpha=0.2,
        label="Total Uncertainty ($\\pm 2\\sigma$)",
    )
    ax.plot(x_s, mu_s + 2.0 * tot_s, color="royalblue", linestyle="--", linewidth=1.2)
    ax.plot(x_s, mu_s - 2.0 * tot_s, color="royalblue", linestyle="--", linewidth=1.2)

    # Aleatoric uncertainty (2 sigma) if provided
    if aleatoric_std is not None:
        ale_s = aleatoric_std.ravel()[idx]
        ax.fill_between(
            x_s,
            mu_s - 2.0 * ale_s,
            mu_s + 2.0 * ale_s,
            color="orange",
            alpha=0.25,
            label="Aleatoric Noise ($\\pm 2\\sigma_{ale}$)",
        )

    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper left")
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig


def plot_training_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float],
    save_path: Optional[Union[str, Path]] = None,
    title: str = "Training and Validation Loss",
) -> plt.Figure:
    """Plots training and validation loss progression over epochs."""
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label="Train Loss", linewidth=1.5)
    ax.plot(epochs, val_losses, label="Val Loss", linewidth=1.5, linestyle="--")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend()
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300)

    return fig
