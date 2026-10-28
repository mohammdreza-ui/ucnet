"""
Loss functions for UCNet:
- MSELoss (Mean Squared Error)
- SoftmaxCrossEntropyLoss (Multi-class classification)
- GaussianNLLLoss (Heteroscedastic regression with aleatoric uncertainty)
"""

from __future__ import annotations

from typing import Union

import numpy as np

from ucnet import ops
from ucnet.ops import to_tensor
from ucnet.tensor import Tensor

# Log-variance range shared by the training loss and by every predictor (see DeepEnsemble.predict).
# Training and prediction must clamp identically, otherwise out-of-range outputs (which receive zero
# gradient from the clamp) turn into astronomically large or small variances at test time.
DEFAULT_MIN_LOG_VAR = -10.0
DEFAULT_MAX_LOG_VAR = 5.0


class MSELoss:
    """Mean Squared Error Loss: L = mean((pred - target)^2)."""

    def __init__(self, reduction: str = "mean") -> None:
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction '{reduction}', must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def __call__(self, pred: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
        t_target = to_tensor(target)
        diff = pred - t_target
        sq_err = diff ** 2.0

        if self.reduction == "mean":
            return sq_err.mean()
        elif self.reduction == "sum":
            return sq_err.sum()
        return sq_err


class SoftmaxCrossEntropyLoss:
    """
    Computes cross-entropy loss between logits and target class labels or one-hot distributions:
    L_i = -log(softmax(logits)_yi) = -logits[y_i] + logsumexp(logits)
    Numerically stable via logsumexp.
    """

    def __init__(self, reduction: str = "mean") -> None:
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction '{reduction}', must be 'mean', 'sum', or 'none'")
        self.reduction = reduction

    def __call__(self, logits: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
        """
        logits: (N, C)
        target: (N,) integer class indices or (N, C) class probabilities/one-hot
        """
        N, C = logits.shape
        t_target = target.data if isinstance(target, Tensor) else np.asarray(target)

        # logsumexp over classes: (N, 1)
        lse = ops.logsumexp(logits, axis=-1, keepdims=True)

        if t_target.ndim == 1:
            # Integer indices
            rows = np.arange(N)
            correct_logits = logits[rows, t_target.astype(np.int64)].reshape(N, 1)
            loss = lse - correct_logits
        else:
            # One-hot or soft labels: (N, C)
            target_t = to_tensor(t_target)
            loss = lse - (logits * target_t).sum(axis=-1, keepdims=True)

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class GaussianNLLLoss:
    """
    Heteroscedastic Gaussian Negative Log-Likelihood Loss for regression:
    L(y, mu, s) = 0.5 * s + 0.5 * (y - mu)^2 * exp(-s) + 0.5 * ln(2*pi)
    where s = log(sigma^2) is the predicted log-variance.
    Clamps log-variance to [min_log_var, max_log_var] for numerical stability.
    """

    def __init__(
        self,
        min_log_var: float = DEFAULT_MIN_LOG_VAR,
        max_log_var: float = DEFAULT_MAX_LOG_VAR,
        reduction: str = "mean",
    ) -> None:
        if reduction not in ("mean", "sum", "none"):
            raise ValueError(f"Invalid reduction '{reduction}', must be 'mean', 'sum', or 'none'")
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var
        self.reduction = reduction
        self.const = 0.5 * np.log(2.0 * np.pi)

    def __call__(
        self,
        pred_mean: Tensor,
        pred_log_var: Tensor,
        target: Union[Tensor, np.ndarray],
    ) -> Tensor:
        t_target = to_tensor(target)

        # Numerical clamping on log-variance
        clamped_s = pred_log_var.clip(min_val=self.min_log_var, max_val=self.max_log_var)

        diff = t_target - pred_mean
        precision = (-clamped_s).exp()

        loss = 0.5 * clamped_s + 0.5 * (diff ** 2.0) * precision + self.const

        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss
