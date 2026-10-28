"""
Exact closed-form Bayesian Linear Regression and reference baseline models.
Provides exact analytical ground truth for predictive mean and epistemic/aleatoric uncertainty.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
from scipy.special import ndtr

from ucnet.ensemble import EnsemblePrediction, align_targets
from ucnet.nn.module import Module
from ucnet.ops import to_tensor
from ucnet.tensor import Tensor


class BayesianLinearRegression:
    """
    Bayesian Linear Regression with conjugate Gaussian prior (Bishop §3.3).
    Prior: p(w) = N(w | 0, alpha^-1 * I)
    Likelihood: p(y | x, w) = N(y | w^T phi(x), beta^-1)

    Exact posterior:
    S_N^-1 = alpha * I + beta * Phi^T * Phi
    m_N = beta * S_N * Phi^T * y

    Exact predictive distribution:
    p(y | x) = N(y | m_N^T phi(x), beta^-1 + phi(x)^T S_N phi(x))
    where beta^-1 is aleatoric noise variance, and phi(x)^T S_N phi(x) is epistemic variance.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        beta: float = 25.0,
        basis: str = "linear",
        degree: int = 3,
        rbf_centers: Optional[np.ndarray] = None,
        rbf_gamma: float = 1.0,
    ) -> None:
        self.alpha = alpha  # Prior precision
        self.beta = beta  # Observation noise precision
        self.basis = basis
        self.degree = degree
        self.rbf_centers = rbf_centers
        self.rbf_gamma = rbf_gamma

        self.m_N: Optional[np.ndarray] = None  # Posterior mean (D,)
        self.S_N: Optional[np.ndarray] = None  # Posterior covariance (D, D)

    def _transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms raw input features into design matrix Phi."""
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        N, D = X.shape

        if self.basis == "linear":
            # Add bias column
            return np.hstack([np.ones((N, 1)), X])
        elif self.basis == "poly":
            # Polynomial expansion
            cols = [np.ones((N, 1))]
            for p in range(1, self.degree + 1):
                cols.append(X ** p)
            return np.hstack(cols)
        elif self.basis == "rbf":
            if self.rbf_centers is None:
                # Default 10 evenly spaced centers
                self.rbf_centers = np.linspace(np.min(X), np.max(X), 10).reshape(-1, D)
            # RBF expansion: exp(-gamma * ||x - c||^2)
            dists = np.sum((X[:, None, :] - self.rbf_centers[None, :, :]) ** 2, axis=2)
            phi_rbf = np.exp(-self.rbf_gamma * dists)
            return np.hstack([np.ones((N, 1)), phi_rbf])
        else:
            return np.hstack([np.ones((N, 1)), X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BayesianLinearRegression":
        """Compute exact posterior parameters m_N and S_N."""
        if y.ndim > 1 and y.shape[1] == 1:
            y = y.ravel()

        Phi = self._transform(X)
        N, M = Phi.shape

        # S_N^-1 = alpha * I + beta * Phi^T * Phi
        S_N_inv = self.alpha * np.eye(M) + self.beta * (Phi.T @ Phi)
        self.S_N = np.linalg.inv(S_N_inv)

        # m_N = beta * S_N * Phi^T * y
        self.m_N = self.beta * (self.S_N @ (Phi.T @ y))
        return self

    def predict(self, X: np.ndarray) -> EnsemblePrediction:
        """
        Compute exact posterior predictive distribution.
        Returns EnsemblePrediction with analytical aleatoric and epistemic breakdown.
        """
        if self.m_N is None or self.S_N is None:
            raise RuntimeError("Model has not been fitted yet")

        Phi = self._transform(X)
        # Predictive mean: Phi @ m_N
        mean = (Phi @ self.m_N).reshape(-1, 1)

        # Aleatoric variance: 1 / beta
        aleatoric = np.full_like(mean, 1.0 / self.beta)

        # Epistemic variance: diag(Phi @ S_N @ Phi^T)
        epistemic = np.sum((Phi @ self.S_N) * Phi, axis=1, keepdims=True)

        total_variance = aleatoric + epistemic

        return EnsemblePrediction(
            mean=mean,
            variance=total_variance,
            aleatoric=aleatoric,
            epistemic=epistemic,
            member_means=mean[None, ...],
            member_vars=total_variance[None, ...],
        )

    def mixture_cdf(self, y: Union[np.ndarray, float], X: np.ndarray) -> np.ndarray:
        """Evaluate exact Gaussian predictive CDF."""
        pred = self.predict(X)
        y_arr = align_targets(y, pred.mean[None, ...])
        z = (y_arr - pred.mean) / np.maximum(pred.std, 1e-12)
        return ndtr(z)


class HomoscedasticBaseline:
    """
    Model M0: Standard neural network or regression model trained with MSE loss,
    where predictive variance is assumed constant (homoscedastic) and estimated
    from validation set residual variance:
    sigma^2 = (1 / N_val) * sum_i (y_val_i - y_hat_val_i)^2
    """

    def __init__(self, model: Module) -> None:
        self.model = model
        self.constant_variance: float = 1.0

    def fit_variance(self, x_val: Union[np.ndarray, Tensor], y_val: Union[np.ndarray, Tensor]) -> float:
        self.model.eval()
        t_x = to_tensor(x_val)
        y_val_np = y_val.data if isinstance(y_val, Tensor) else np.asarray(y_val)

        pred = self.model(t_x).data
        residuals = y_val_np - pred
        self.constant_variance = float(np.mean(residuals ** 2.0))
        return self.constant_variance

    def predict(self, x: Union[np.ndarray, Tensor]) -> EnsemblePrediction:
        self.model.eval()
        t_x = to_tensor(x)
        mean = self.model(t_x).data
        if mean.ndim == 1:
            mean = mean.reshape(-1, 1)

        N = len(mean)
        var = np.full((N, 1), self.constant_variance)
        zero_epistemic = np.zeros_like(var)

        return EnsemblePrediction(
            mean=mean,
            variance=var,
            aleatoric=var,
            epistemic=zero_epistemic,
            member_means=mean[None, ...],
            member_vars=var[None, ...],
        )

    def mixture_cdf(self, y: Union[np.ndarray, float], x: Union[np.ndarray, Tensor]) -> np.ndarray:
        pred = self.predict(x)
        y_arr = align_targets(y, pred.mean[None, ...])
        z = (y_arr - pred.mean) / np.maximum(pred.std, 1e-12)
        return ndtr(z)
