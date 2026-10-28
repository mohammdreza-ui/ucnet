"""
Layer Normalization implementation with exact analytical backpropagation.
"""

from __future__ import annotations

from typing import Optional, Union

import numpy as np

from ucnet.nn.module import Module, Parameter
from ucnet.tensor import Tensor


def layernorm_forward_backward(
    x: Tensor,
    weight: Optional[Parameter],
    bias: Optional[Parameter],
    normalized_shape: tuple[int, ...],
    eps: float = 1e-5,
) -> Tensor:
    """
    Computes LayerNorm forward pass and attaches exact analytical backward pass:
    y = gamma * (x - mu) / sqrt(var + eps) + beta
    """
    ndim_norm = len(normalized_shape)
    norm_axes = tuple(range(x.ndim - ndim_norm, x.ndim))
    batch_axes = tuple(range(0, x.ndim - ndim_norm))

    mu = np.mean(x.data, axis=norm_axes, keepdims=True)
    var = np.var(x.data, axis=norm_axes, keepdims=True)
    std_inv = 1.0 / np.sqrt(var + eps)
    x_hat = (x.data - mu) * std_inv

    y_data = x_hat
    if weight is not None:
        y_data = y_data * weight.data
    if bias is not None:
        y_data = y_data + bias.data

    prev_list = [x]
    if weight is not None:
        prev_list.append(weight)
    if bias is not None:
        prev_list.append(bias)

    requires_grad = x.requires_grad or (weight is not None and weight.requires_grad) or (bias is not None and bias.requires_grad)
    out = Tensor(y_data, requires_grad=requires_grad, _prev=prev_list, _op="layernorm")

    if requires_grad:
        def _backward() -> None:
            if out.grad is None:
                return
            ybar = out.grad

            # Gradients w.r.t gamma and beta
            if weight is not None and weight.requires_grad:
                dgamma = np.sum(ybar * x_hat, axis=batch_axes)
                weight.grad = dgamma if weight.grad is None else weight.grad + dgamma

            if bias is not None and bias.requires_grad:
                dbeta = np.sum(ybar, axis=batch_axes)
                bias.grad = dbeta if bias.grad is None else bias.grad + dbeta

            # Gradient w.r.t input x
            if x.requires_grad:
                gamma = weight.data if weight is not None else 1.0
                g = ybar * gamma
                g_mean = np.mean(g, axis=norm_axes, keepdims=True)
                g_xhat_mean = np.mean(g * x_hat, axis=norm_axes, keepdims=True)
                dx = std_inv * (g - g_mean - x_hat * g_xhat_mean)
                x.grad = dx if x.grad is None else x.grad + dx

        out._backward = _backward

    return out


class LayerNorm(Module):
    """
    Applies Layer Normalization over the last dimensions specified by normalized_shape:
    y = (x - E[x]) / sqrt(Var[x] + eps) * gamma + beta
    """

    def __init__(
        self,
        normalized_shape: Union[int, tuple[int, ...], list[int]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        dtype: Optional[np.dtype] = np.float64,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)

        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight: Optional[Parameter] = Parameter(np.ones(self.normalized_shape), dtype=dtype)
            self.bias: Optional[Parameter] = Parameter(np.zeros(self.normalized_shape), dtype=dtype)
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        return layernorm_forward_backward(
            x,
            self.weight,
            self.bias,
            self.normalized_shape,
            eps=self.eps,
        )

    def __repr__(self) -> str:
        return f"LayerNorm(normalized_shape={self.normalized_shape}, eps={self.eps}, elementwise_affine={self.elementwise_affine})"
