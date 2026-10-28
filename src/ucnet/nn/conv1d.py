"""
1D Convolutional layer implementation for signal and spectral data.
Optimized via sliding window views and einsum, with exact reverse-mode autodiff.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ucnet.nn.module import Module, Parameter
from ucnet.tensor import Tensor


def conv1d_forward_backward(
    x: Tensor,
    weight: Parameter,
    bias: Optional[Parameter],
    stride: int = 1,
    padding: int = 0,
) -> Tensor:
    """
    Computes 1D cross-correlation convolution and attaches analytical backward pass:
    x: (N, C_in, L)
    weight: (C_out, C_in, K)
    bias: (C_out,) or None
    stride: s
    padding: p
    """
    N, C_in, L = x.shape
    C_out, w_Cin, K = weight.shape
    assert C_in == w_Cin, f"Input channels {C_in} != weight channels {w_Cin}"

    # Pad input along sequence length if padding > 0
    if padding > 0:
        x_pad = np.pad(x.data, ((0, 0), (0, 0), (padding, padding)), mode="constant")
    else:
        x_pad = x.data

    L_pad = x_pad.shape[2]
    T = (L_pad - K) // stride + 1

    # Extract windows of shape (N, C_in, T, K)
    all_windows = np.lib.stride_tricks.sliding_window_view(x_pad, window_shape=K, axis=2)
    windows = all_windows[:, :, ::stride, :]
    assert windows.shape == (N, C_in, T, K)

    # Compute convolution via einsum: (N, C_in, T, K) * (C_out, C_in, K) -> (N, C_out, T)
    y_data = np.einsum("nctk,ock->not", windows, weight.data)

    if bias is not None:
        y_data = y_data + bias.data[None, :, None]

    prev = (x, weight, bias) if bias is not None else (x, weight)
    requires_grad = x.requires_grad or weight.requires_grad or (bias is not None and bias.requires_grad)
    out = Tensor(y_data, requires_grad=requires_grad, _prev=prev, _op="conv1d")

    if requires_grad:
        def _backward() -> None:
            if out.grad is None:
                return
            ybar = out.grad

            # 1. Gradient w.r.t bias: sum over batch and sequence length
            if bias is not None and bias.requires_grad:
                gb = ybar.sum(axis=(0, 2))
                bias.grad = gb if bias.grad is None else bias.grad + gb

            # 2. Gradient w.r.t weight
            if weight.requires_grad:
                gw = np.zeros_like(weight.data)
                for k in range(K):
                    x_slice = x_pad[:, :, k : k + stride * T : stride]
                    gw[:, :, k] = np.einsum("not,nct->oc", ybar, x_slice)
                weight.grad = gw if weight.grad is None else weight.grad + gw

            # 3. Gradient w.r.t input x
            if x.requires_grad:
                xbar_pad = np.zeros_like(x_pad)
                for k in range(K):
                    term = np.einsum("not,oc->nct", ybar, weight.data[:, :, k])
                    xbar_pad[:, :, k : k + stride * T : stride] += term

                if padding > 0:
                    gx = xbar_pad[:, :, padding : L_pad - padding]
                else:
                    gx = xbar_pad

                x.grad = gx if x.grad is None else x.grad + gx

        out._backward = _backward

    return out


class Conv1D(Module):
    """
    1D Convolutional layer:
    input shape: (batch_size, in_channels, length)
    output shape: (batch_size, out_channels, out_length)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
        dtype: Optional[np.dtype] = np.float64,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Kaiming/He initialization for convolutional weights
        std = np.sqrt(2.0 / (in_channels * kernel_size))
        w_data = np.random.randn(out_channels, in_channels, kernel_size) * std
        self.weight = Parameter(w_data, dtype=dtype)

        if bias:
            b_data = np.zeros(out_channels)
            self.bias: Optional[Parameter] = Parameter(b_data, dtype=dtype)
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        return conv1d_forward_backward(
            x,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=self.padding,
        )

    def __repr__(self) -> str:
        return (
            f"Conv1D(in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, padding={self.padding}, "
            f"bias={self.bias is not None})"
        )
