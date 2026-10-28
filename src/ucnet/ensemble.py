"""
Deep Ensemble implementation and Gaussian mixture combination for uncertainty quantification.
Decomposes total predictive variance into aleatoric and epistemic components via the Law of Total Variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
from scipy.special import ndtr

from ucnet.losses import DEFAULT_MAX_LOG_VAR, DEFAULT_MIN_LOG_VAR
from ucnet.nn.module import Module
from ucnet.ops import to_tensor
from ucnet.tensor import Tensor


@dataclass
class EnsemblePrediction:
    """Container for deep ensemble probabilistic predictions."""

    mean: np.ndarray  # (N, D) overall mean mu*
    variance: np.ndarray  # (N, D) overall variance sigma*^2
    aleatoric: np.ndarray  # (N, D) expected observation noise
    epistemic: np.ndarray  # (N, D) variance between model predictions
    member_means: np.ndarray  # (M, N, D) individual model means
    member_vars: np.ndarray  # (M, N, D) individual model variances

    @property
    def std(self) -> np.ndarray:
        return np.sqrt(self.variance)


def align_targets(y: Union[np.ndarray, float], member_means: np.ndarray) -> np.ndarray:
    """
    Reshape targets so they broadcast against member predictions of shape (M, N, D) without creating
    an (N, N) cross-product: a flat (N,) vector of targets is treated as one target per sample.
    """
    y_arr = np.asarray(y, dtype=np.float64)
    if y_arr.ndim == 1:
        if member_means.shape[-1] != 1:
            raise ValueError("1-D targets are only supported for single-output predictors")
        y_arr = y_arr.reshape(-1, 1)
    return y_arr


class DeepEnsemble:
    """
    Deep Ensemble of probabilistic models (Lakshminarayanan et al., 2017).
    Combines M neural networks trained with different random initializations and data shuffles.
    Each member outputs (mean, log_variance).
    """

    def __init__(
        self,
        members: Optional[Sequence[Module]] = None,
        min_log_var: float = DEFAULT_MIN_LOG_VAR,
        max_log_var: float = DEFAULT_MAX_LOG_VAR,
    ) -> None:
        self.members: list[Module] = list(members) if members is not None else []
        # Same clamp as GaussianNLLLoss, so predicted variances match what the loss trained on.
        self.min_log_var = min_log_var
        self.max_log_var = max_log_var

    def add_member(self, member: Module) -> None:
        self.members.append(member)

    def __len__(self) -> int:
        return len(self.members)

    def predict(self, x: Union[np.ndarray, Tensor]) -> EnsemblePrediction:
        """
        Computes the ensemble mixture distribution over input x.
        Applies the Law of Total Variance:
        Var[Y|x] = E[Var[Y|x, m]] + Var[E[Y|x, m]]
                 = (1/M) sum_m sigma_m^2 + (1/M) sum_m (mu_m - mu*)^2
        """
        if len(self.members) == 0:
            raise RuntimeError("Ensemble has no trained members")

        t_x = to_tensor(x)

        means_list: list[np.ndarray] = []
        vars_list: list[np.ndarray] = []

        for member in self.members:
            member.eval()
            out = member(t_x)

            # Member forward can return either (mu, log_var) tuple or a single tensor with 2 outputs
            if isinstance(out, (tuple, list)):
                mu, log_var = out[0], out[1]
            elif isinstance(out, Tensor) and out.shape[-1] == 2:
                mu = out[..., :1]
                log_var = out[..., 1:]
            else:
                raise ValueError("Ensemble member must output (mean, log_var) tuple or shape (..., 2)")

            means_list.append(mu.data.copy())
            # Convert (clamped) log-variance to variance: sigma^2 = exp(clip(s))
            vars_list.append(np.exp(np.clip(log_var.data, self.min_log_var, self.max_log_var)))

        # member_means: (M, N, D), member_vars: (M, N, D)
        member_means = np.stack(means_list, axis=0)
        member_vars = np.stack(vars_list, axis=0)

        # 1. Overall ensemble mean: mu* = (1/M) sum_m mu_m
        ens_mean = np.mean(member_means, axis=0)

        # 2. Aleatoric variance: average observation noise (1/M) sum_m sigma_m^2
        aleatoric_var = np.mean(member_vars, axis=0)

        # 3. Epistemic variance: model disagreement (1/M) sum_m (mu_m - mu*)^2
        epistemic_var = np.mean((member_means - ens_mean) ** 2, axis=0)

        # 4. Total predictive variance
        total_var = aleatoric_var + epistemic_var

        return EnsemblePrediction(
            mean=ens_mean,
            variance=total_var,
            aleatoric=aleatoric_var,
            epistemic=epistemic_var,
            member_means=member_means,
            member_vars=member_vars,
        )

    def mixture_cdf(self, y: Union[np.ndarray, float], x: Union[np.ndarray, Tensor]) -> np.ndarray:
        """
        Computes the exact Cumulative Distribution Function of the Gaussian Mixture:
        F(y|x) = (1/M) sum_m Phi((y - mu_m(x)) / sigma_m(x))
        Crucial for exact Probability Integral Transform (PIT) and calibration testing.
        """
        pred = self.predict(x)
        y_arr = align_targets(y, pred.member_means)

        # Broadcast y against (M, N, D)
        diff = y_arr - pred.member_means
        std = np.sqrt(pred.member_vars)
        z = diff / np.maximum(std, 1e-12)

        # Evaluate standard normal CDF: Phi(z)
        cdf_members = ndtr(z)
        # Average across members: (N, D)
        return np.mean(cdf_members, axis=0)


class ProbabilisticMLP(Module):
    """
    Standard neural network that predicts both mean and log-variance for heteroscedastic regression:
    Outputs: [mu, log_var] of shape (batch_size, 2).
    """

    def __init__(
        self,
        in_features: int,
        hidden_dims: Sequence[int] = (64, 64),
        activation: str = "relu",
        out_features: int = 1,
    ) -> None:
        super().__init__()
        from ucnet.nn.activations import ReLU, Tanh
        from ucnet.nn.linear import Linear

        layers: list[Module] = []
        curr_dim = in_features

        for h_dim in hidden_dims:
            layers.append(Linear(curr_dim, h_dim))
            if activation == "relu":
                layers.append(ReLU())
            elif activation == "tanh":
                layers.append(Tanh())
            curr_dim = h_dim

        # Final output layer: outputs mean and log-variance (out_features * 2)
        layers.append(Linear(curr_dim, out_features * 2))

        from ucnet.nn.containers import Sequential
        self.net = Sequential(*layers)
        self.out_features = out_features

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        out = self.net(x)
        mu = out[..., :self.out_features]
        log_var = out[..., self.out_features:]
        return mu, log_var
