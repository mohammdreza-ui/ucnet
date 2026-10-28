"""Shared helpers for the experiment scripts: training, validation-only tuning and evaluation."""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np

from ucnet.baselines import BayesianLinearRegression, HomoscedasticBaseline
from ucnet.data import StandardScaler
from ucnet.ensemble import DeepEnsemble, ProbabilisticMLP
from ucnet.losses import GaussianNLLLoss, MSELoss
from ucnet.metrics import (
    CalibrationReport,
    compute_nll,
    evaluate_calibration,
    make_original_scale_cdf,
)
from ucnet.nn.activations import ReLU, Tanh
from ucnet.nn.containers import Sequential
from ucnet.nn.linear import Linear
from ucnet.optim import Adam
from ucnet.trainer import fit


def train_probabilistic_member(
    in_dim: int,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    hidden: int = 32,
    activation: str = "relu",
    lr: float = 0.01,
    weight_decay: float = 0.0,
    epochs: int = 100,
    patience: int = 15,
) -> tuple[ProbabilisticMLP, float]:
    """Train one heteroscedastic MLP (init and batch order fully determined by `seed`).
    Returns the model (best-validation weights) and its best validation NLL."""
    np.random.seed(seed)
    model = ProbabilisticMLP(in_features=in_dim, hidden_dims=(hidden,), activation=activation)
    hist = fit(
        model, Adam(model.parameters(), lr=lr, weight_decay=weight_decay), GaussianNLLLoss(),
        X_tr, y_tr, X_val, y_val, epochs=epochs, patience=patience, seed=seed,
    )
    return model, float(min(hist.val_losses))


def train_homoscedastic(
    in_dim: int,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    hidden: int = 32,
    activation: str = "relu",
    lr: float = 0.01,
    weight_decay: float = 0.0,
    epochs: int = 100,
    patience: int = 15,
) -> HomoscedasticBaseline:
    """M0: MSE-trained MLP with a constant sigma estimated from validation residuals."""
    np.random.seed(seed)
    act = ReLU() if activation == "relu" else Tanh()
    net = Sequential(Linear(in_dim, hidden), act, Linear(hidden, 1))
    fit(
        net, Adam(net.parameters(), lr=lr, weight_decay=weight_decay), MSELoss(),
        X_tr, y_tr, X_val, y_val, epochs=epochs, patience=patience, seed=seed,
    )
    m0 = HomoscedasticBaseline(net)
    m0.fit_variance(X_val, y_val)
    return m0


def tune_mlp_hyperparams(
    in_dim: int,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    seed: int,
    activation: str = "relu",
    lrs: tuple[float, ...] = (0.003, 0.01),
    weight_decays: tuple[float, ...] = (0.0, 1e-4, 1e-2),
    epochs: int = 100,
    patience: int = 15,
) -> dict[str, float]:
    """Small grid search for the single heteroscedastic model, scored on validation NLL only.
    The same (lr, weight_decay) is then used for M0, M1 and every M2 member: equal tuning budget."""
    best: Optional[tuple[float, float, float]] = None
    for lr, wd in itertools.product(lrs, weight_decays):
        _, val_nll = train_probabilistic_member(
            in_dim, X_tr, y_tr, X_val, y_val, seed=seed, activation=activation,
            lr=lr, weight_decay=wd, epochs=epochs, patience=patience,
        )
        if best is None or val_nll < best[0]:
            best = (val_nll, lr, wd)
    assert best is not None
    return {"lr": best[1], "weight_decay": best[2], "val_nll": best[0]}


def tune_blr(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    basis: str = "linear",
    alphas: tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    betas: tuple[float, ...] = (0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 400.0, 1600.0),
    **blr_kwargs,
) -> BayesianLinearRegression:
    """Pick the BLR prior precision alpha and noise precision beta by validation NLL (standardized y)."""
    y_tr_flat, y_val_flat = np.asarray(y_tr).ravel(), np.asarray(y_val).ravel()
    best: Optional[tuple[float, float, float]] = None
    for alpha, beta in itertools.product(alphas, betas):
        blr = BayesianLinearRegression(alpha=alpha, beta=beta, basis=basis, **blr_kwargs).fit(X_tr, y_tr_flat)
        pred = blr.predict(X_val)
        nll = compute_nll(y_val_flat, pred.mean, pred.std)
        if best is None or nll < best[0]:
            best = (nll, alpha, beta)
    assert best is not None
    return BayesianLinearRegression(alpha=best[1], beta=best[2], basis=basis, **blr_kwargs).fit(X_tr, y_tr_flat)


def evaluate_predictor(
    model,
    X_norm: np.ndarray,
    y_orig: np.ndarray,
    scaler_y: StandardScaler,
) -> tuple[CalibrationReport, np.ndarray, np.ndarray]:
    """Evaluate in original units. PIT uses the model's true predictive CDF (mixture CDF for
    ensembles); NLL/CRPS/coverage use the moment-matched Gaussian (mean, total std)."""
    assert scaler_y.mean_ is not None and scaler_y.scale_ is not None
    y_mean, y_scale = float(scaler_y.mean_[0, 0]), float(scaler_y.scale_[0, 0])
    pred = model.predict(X_norm)
    mu = scaler_y.inverse_transform(pred.mean)
    std = pred.std * y_scale
    cdf = make_original_scale_cdf(model.mixture_cdf, X_norm, y_mean, y_scale)
    return evaluate_calibration(y_orig, mu, std, cdf_fn=cdf), mu, std


def build_ensemble(members: list[ProbabilisticMLP]) -> DeepEnsemble:
    return DeepEnsemble(members)
