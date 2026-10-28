"""
Activation and functional modules for UCNet.
"""

from __future__ import annotations

import numpy as np

from ucnet.nn.module import Module
from ucnet.tensor import Tensor


class ReLU(Module):
    """Applies element-wise rectified linear unit: ReLU(x) = max(0, x)."""

    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def __repr__(self) -> str:
        return "ReLU()"


class Tanh(Module):
    """Applies element-wise hyperbolic tangent: Tanh(x) = tanh(x)."""

    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()

    def __repr__(self) -> str:
        return "Tanh()"


class Sigmoid(Module):
    """Applies element-wise sigmoid: Sigmoid(x) = 1 / (1 + exp(-x))."""

    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

    def __repr__(self) -> str:
        return "Sigmoid()"


class Softplus(Module):
    """Applies element-wise softplus: Softplus(x) = log(1 + exp(x))."""

    def forward(self, x: Tensor) -> Tensor:
        return x.softplus()

    def __repr__(self) -> str:
        return "Softplus()"


class GELU(Module):
    """
    Applies Gaussian Error Linear Unit activation:
    GELU(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
    """

    def forward(self, x: Tensor) -> Tensor:
        # Standard Hendrycks & Gimpel approximation used in BERT/GPT
        c1 = np.sqrt(2.0 / np.pi)
        inner = c1 * (x + (x ** 3.0) * 0.044715)
        return x * 0.5 * (inner.tanh() + 1.0)

    def __repr__(self) -> str:
        return "GELU()"


class Dropout(Module):
    """
    During training, randomly zeroes some of the elements of the input tensor
    with probability p using samples from a Bernoulli distribution.
    Scales remaining activations by 1 / (1 - p).
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if p < 0.0 or p >= 1.0:
            raise ValueError(f"Dropout probability p must be in [0, 1), got {p}")
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        mask_data = (np.random.rand(*x.shape) >= self.p).astype(x.dtype) / (1.0 - self.p)
        mask = Tensor(mask_data, requires_grad=False)
        return x * mask

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


class Flatten(Module):
    """Flattens a contiguous range of dims into a single dimension."""

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        return x.flatten(start_dim=self.start_dim, end_dim=self.end_dim)

    def __repr__(self) -> str:
        return f"Flatten(start_dim={self.start_dim}, end_dim={self.end_dim})"


class GlobalAvgPool1D(Module):
    """Reduces (N, C, L) to (N, C) by averaging along the sequence/spectral axis."""

    def forward(self, x: Tensor) -> Tensor:
        return x.mean(axis=2)

    def __repr__(self) -> str:
        return "GlobalAvgPool1D()"
