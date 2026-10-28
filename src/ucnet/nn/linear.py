"""
Fully-connected Linear layer implementation.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ucnet.nn.module import Module, Parameter
from ucnet.tensor import Tensor


class Linear(Module):
    """
    Applies an affine linear transformation to incoming data: y = x @ W + b.
    Weight shape: (in_features, out_features)
    Bias shape: (out_features,)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        init: str = "he",
        dtype: Optional[np.dtype] = np.float64,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weight matrix according to standard variance preservation principles
        if init in ("he", "kaiming"):
            # Preserves variance under ReLU activation
            std = np.sqrt(2.0 / in_features)
            w_data = np.random.randn(in_features, out_features) * std
        elif init in ("xavier", "glorot"):
            # Preserves variance across linear/tanh activations
            std = np.sqrt(2.0 / (in_features + out_features))
            w_data = np.random.randn(in_features, out_features) * std
        elif init == "uniform":
            bound = 1.0 / np.sqrt(in_features)
            w_data = np.random.uniform(-bound, bound, size=(in_features, out_features))
        else:
            w_data = np.random.randn(in_features, out_features) * 0.01

        self.weight = Parameter(w_data, dtype=dtype)

        if bias:
            b_data = np.zeros(out_features)
            self.bias: Optional[Parameter] = Parameter(b_data, dtype=dtype)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self) -> str:
        bias_str = f", bias={self.bias is not None}"
        return f"Linear(in_features={self.in_features}, out_features={self.out_features}{bias_str})"
