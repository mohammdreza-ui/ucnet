"""
Differentiable mathematical and tensor operations for UCNet.
Every operation implements both forward pass and numerical reverse-mode backward adjoints.
"""

from __future__ import annotations

from typing import Any, Optional, Union

import numpy as np

from ucnet.tensor import Tensor, unbroadcast


def to_tensor(x: Any, dtype: Optional[np.dtype] = None) -> Tensor:
    """Ensure input is a Tensor instance."""
    if isinstance(x, Tensor):
        return x
    return Tensor(x, dtype=dtype)


def add(a: Any, b: Any) -> Tensor:
    """Element-wise addition: out = a + b."""
    t_a = to_tensor(a)
    t_b = to_tensor(b)
    requires_grad = t_a.requires_grad or t_b.requires_grad
    out = Tensor(t_a.data + t_b.data, requires_grad=requires_grad, _prev=(t_a, t_b), _op="add")

    if requires_grad:
        def _backward() -> None:
            if out.grad is None:
                return
            if t_a.requires_grad:
                g_a = unbroadcast(out.grad, t_a.shape)
                t_a.grad = g_a if t_a.grad is None else t_a.grad + g_a
            if t_b.requires_grad:
                g_b = unbroadcast(out.grad, t_b.shape)
                t_b.grad = g_b if t_b.grad is None else t_b.grad + g_b

        out._backward = _backward

    return out


def sub(a: Any, b: Any) -> Tensor:
    """Element-wise subtraction: out = a - b."""
    t_a = to_tensor(a)
    t_b = to_tensor(b)
    requires_grad = t_a.requires_grad or t_b.requires_grad
    out = Tensor(t_a.data - t_b.data, requires_grad=requires_grad, _prev=(t_a, t_b), _op="sub")

    if requires_grad:
        def _backward() -> None:
            if out.grad is None:
                return
            if t_a.requires_grad:
                g_a = unbroadcast(out.grad, t_a.shape)
                t_a.grad = g_a if t_a.grad is None else t_a.grad + g_a
            if t_b.requires_grad:
                g_b = unbroadcast(-out.grad, t_b.shape)
                t_b.grad = g_b if t_b.grad is None else t_b.grad + g_b

        out._backward = _backward

    return out


def mul(a: Any, b: Any) -> Tensor:
    """Element-wise multiplication: out = a * b."""
    t_a = to_tensor(a)
    t_b = to_tensor(b)
    requires_grad = t_a.requires_grad or t_b.requires_grad
    out = Tensor(t_a.data * t_b.data, requires_grad=requires_grad, _prev=(t_a, t_b), _op="mul")

    if requires_grad:
        def _backward() -> None:
            if out.grad is None:
                return
            if t_a.requires_grad:
                g_a = unbroadcast(out.grad * t_b.data, t_a.shape)
                t_a.grad = g_a if t_a.grad is None else t_a.grad + g_a
            if t_b.requires_grad:
                g_b = unbroadcast(out.grad * t_a.data, t_b.shape)
                t_b.grad = g_b if t_b.grad is None else t_b.grad + g_b

        out._backward = _backward

    return out


def div(a: Any, b: Any) -> Tensor:
    """Element-wise division: out = a / b."""
    t_a = to_tensor(a)
    t_b = to_tensor(b)
    requires_grad = t_a.requires_grad or t_b.requires_grad
    out = Tensor(t_a.data / t_b.data, requires_grad=requires_grad, _prev=(t_a, t_b), _op="div")

    if requires_grad:
        def _backward() -> None:
            if out.grad is None:
                return
            if t_a.requires_grad:
                g_a = unbroadcast(out.grad / t_b.data, t_a.shape)
                t_a.grad = g_a if t_a.grad is None else t_a.grad + g_a
            if t_b.requires_grad:
                g_b = unbroadcast(-out.grad * t_a.data / (t_b.data ** 2), t_b.shape)
                t_b.grad = g_b if t_b.grad is None else t_b.grad + g_b

        out._backward = _backward

    return out


def pow_op(a: Any, power: Union[int, float, Tensor]) -> Tensor:
    """Element-wise power: out = a ** power."""
    t_a = to_tensor(a)
    is_tensor_p = isinstance(power, Tensor)
    t_p = power if is_tensor_p else to_tensor(power)
    requires_grad = t_a.requires_grad or (is_tensor_p and t_p.requires_grad)

    out_data = t_a.data ** t_p.data
    prev = (t_a, t_p) if is_tensor_p else (t_a,)
    out = Tensor(out_data, requires_grad=requires_grad, _prev=prev, _op="pow")

    if requires_grad:
        def _backward() -> None:
            if out.grad is None:
                return
            if t_a.requires_grad:
                # d/da [a^p] = p * a^(p - 1)
                g_a = unbroadcast(out.grad * t_p.data * (t_a.data ** (t_p.data - 1)), t_a.shape)
                t_a.grad = g_a if t_a.grad is None else t_a.grad + g_a
            if is_tensor_p and t_p.requires_grad:
                # d/dp [a^p] = a^p * ln(a)
                # handle non-positive safe mask
                safe_a = np.where(t_a.data > 0, t_a.data, 1.0)
                g_p = unbroadcast(out.grad * out.data * np.log(safe_a), t_p.shape)
                t_p.grad = g_p if t_p.grad is None else t_p.grad + g_p

        out._backward = _backward

    return out


def neg(a: Any) -> Tensor:
    """Element-wise negation: out = -a."""
    t_a = to_tensor(a)
    out = Tensor(-t_a.data, requires_grad=t_a.requires_grad, _prev=(t_a,), _op="neg")

    if t_a.requires_grad:
        def _backward() -> None:
            if out.grad is None:
                return
            t_a.grad = -out.grad if t_a.grad is None else t_a.grad - out.grad

        out._backward = _backward

    return out


def matmul(a: Any, b: Any) -> Tensor:
    """
    Matrix multiplication with full analytical adjoints:
    C = A @ B
    d/dA = dC @ B^T
    d/dB = A^T @ dC
    Supports 2D, 1D vectors, and arbitrary N-D batched matrices with broadcasting.
    """
    t_a = to_tensor(a)
    t_b = to_tensor(b)
    requires_grad = t_a.requires_grad or t_b.requires_grad

    A = t_a.data
    B = t_b.data
    C = A @ B
    out = Tensor(C, requires_grad=requires_grad, _prev=(t_a, t_b), _op="matmul")

    if requires_grad:
        def _backward() -> None:
            if out.grad is None:
                return
            dC = out.grad

            # 1. 1D @ 1D (dot product -> scalar)
            if A.ndim == 1 and B.ndim == 1:
                if t_a.requires_grad:
                    g_a = dC * B
                    t_a.grad = g_a if t_a.grad is None else t_a.grad + g_a
                if t_b.requires_grad:
                    g_b = dC * A
                    t_b.grad = g_b if t_b.grad is None else t_b.grad + g_b
                return

            # 2. 1D @ 2D: A is (K,), B is (K, N), C is (N,)
            if A.ndim == 1 and B.ndim == 2:
                if t_a.requires_grad:
                    g_a = dC @ B.T
                    t_a.grad = g_a if t_a.grad is None else t_a.grad + g_a
                if t_b.requires_grad:
                    g_b = A[:, None] @ dC[None, :]
                    t_b.grad = g_b if t_b.grad is None else t_b.grad + g_b
                return

            # 3. 2D @ 1D: A is (M, K), B is (K,), C is (M,)
            if A.ndim == 2 and B.ndim == 1:
                if t_a.requires_grad:
                    g_a = dC[:, None] @ B[None, :]
                    t_a.grad = g_a if t_a.grad is None else t_a.grad + g_a
                if t_b.requires_grad:
                    g_b = A.T @ dC
                    t_b.grad = g_b if t_b.grad is None else t_b.grad + g_b
                return

            # 4. Standard 2D and N-D Batched Matrix Multiplication
            # For A (..., M, K) and B (..., K, N) -> C (..., M, N)
            # dA = dC @ B^T
            # dB = A^T @ dC
            if t_a.requires_grad:
                B_transposed = np.swapaxes(B, -1, -2)
                dA = dC @ B_transposed
                g_a = unbroadcast(dA, t_a.shape)
                t_a.grad = g_a if t_a.grad is None else t_a.grad + g_a

            if t_b.requires_grad:
                A_transposed = np.swapaxes(A, -1, -2)
                dB = A_transposed @ dC
                g_b = unbroadcast(dB, t_b.shape)
                t_b.grad = g_b if t_b.grad is None else t_b.grad + g_b

        out._backward = _backward

    return out


def exp_op(a: Any) -> Tensor:
    """Element-wise exponential: out = exp(a)."""
    t_a = to_tensor(a)
    out_data = np.exp(t_a.data)
    out = Tensor(out_data, requires_grad=t_a.requires_grad, _prev=(t_a,), _op="exp")

    if t_a.requires_grad:
        def _backward() -> None:
            if out.grad is None:
                return
            g = out.grad * out.data
            t_a.grad = g if t_a.grad is None else t_a.grad + g

        out._backward = _backward

    return out


def log_op(a: Any) -> Tensor:
    """Element-wise natural logarithm: out = log(a)."""
    t_a = to_tensor(a)
    out_data = np.log(t_a.data)
    out = Tensor(out_data, requires_grad=t_a.requires_grad, _prev=(t_a,), _op="log")

    if t_a.requires_grad:
        def _backward() -> None:
            if out.grad is None:
                return
            g = out.grad / t_a.data
            t_a.grad = g if t_a.grad is None else t_a.grad + g

        out._backward = _backward

    return out


def tanh_op(a: Any) -> Tensor:
    """Element-wise hyperbolic tangent: out = tanh(a)."""
    t_a = to_tensor(a)
    out_data = np.tanh(t_a.data)
    out = Tensor(out_data, requires_grad=t_a.requires_grad, _prev=(t_a,), _op="tanh")

    if t_a.requires_grad:
        def _backward() -> None:
            if out.grad is None:
                return
            g = out.grad * (1.0 - out.data ** 2)
            t_a.grad = g if t_a.grad is None else t_a.grad + g

        out._backward = _backward

    return out


def sigmoid_op(a: Any) -> Tensor:
    """
    Numerically stable sigmoid activation:
    out = 1 / (1 + exp(-a)) for a >= 0
    out = exp(a) / (1 + exp(a)) for a < 0
    """
    t_a = to_tensor(a)
    x = t_a.data
    pos_mask = x >= 0
    out_data = np.empty_like(x, dtype=np.float64 if t_a.dtype == np.float64 else np.float32)

    # Numerically stable evaluation
    out_data[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
    exp_neg = np.exp(x[~pos_mask])
    out_data[~pos_mask] = exp_neg / (1.0 + exp_neg)

    out = Tensor(out_data, requires_grad=t_a.requires_grad, _prev=(t_a,), _op="sigmoid")

    if t_a.requires_grad:
        def _backward() -> None:
            if out.grad is None:
                return
            g = out.grad * out.data * (1.0 - out.data)
            t_a.grad = g if t_a.grad is None else t_a.grad + g

        out._backward = _backward

    return out


def relu_op(a: Any) -> Tensor:
    """Rectified Linear Unit: out = max(a, 0)."""
    t_a = to_tensor(a)
    out_data = np.maximum(t_a.data, 0.0)
    out = Tensor(out_data, requires_grad=t_a.requires_grad, _prev=(t_a,), _op="relu")

    if t_a.requires_grad:
        def _backward() -> None:
            if out.grad is None:
                return
            g = out.grad * (t_a.data > 0.0)
            t_a.grad = g if t_a.grad is None else t_a.grad + g

        out._backward = _backward

    return out


def softplus_op(a: Any) -> Tensor:
    """
    Numerically stable softplus activation:
    out = max(a, 0) + log1p(exp(-|a|))
    Derivative is sigmoid(a).
    """
    t_a = to_tensor(a)
    x = t_a.data
    out_data = np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))
    out = Tensor(out_data, requires_grad=t_a.requires_grad, _prev=(t_a,), _op="softplus")

    if t_a.requires_grad:
        def _backward() -> None:
            if out.grad is None:
                return
            # sigmoid(x)
            pos_mask = x >= 0
            sig = np.empty_like(x)
            sig[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))
            exp_neg = np.exp(x[~pos_mask])
            sig[~pos_mask] = exp_neg / (1.0 + exp_neg)

            g = out.grad * sig
            t_a.grad = g if t_a.grad is None else t_a.grad + g

        out._backward = _backward

    return out


def clip_op(a: Any, min_val: Optional[float] = None, max_val: Optional[float] = None) -> Tensor:
    """Element-wise clipping."""
    t_a = to_tensor(a)
    out_data = np.clip(t_a.data, a_min=min_val, a_max=max_val)
    out = Tensor(out_data, requires_grad=t_a.requires_grad, _prev=(t_a,), _op="clip")

    if t_a.requires_grad:
        def _backward() -> None:
            if out.grad is None:
                return
            mask = np.ones_like(t_a.data, dtype=bool)
            if min_val is not None:
                mask = mask & (t_a.data >= min_val)
            if max_val is not None:
                mask = mask & (t_a.data <= max_val)
            g = out.grad * mask
            t_a.grad = g if t_a.grad is None else t_a.grad + g

        out._backward = _backward

    return out


def sum_op(
    a: Any,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Tensor:
    """Tensor summation over specified axes."""
    t_a = to_tensor(a)
    out_data = np.sum(t_a.data, axis=axis, keepdims=keepdims)
    out = Tensor(out_data, requires_grad=t_a.requires_grad, _prev=(t_a,), _op="sum")

    if t_a.requires_grad:
        def _backward() -> None:
            if out.grad is None:
                return
            g = out.grad
            if not keepdims and axis is not None:
                # Restore summed dimensions as size-1 axes for broadcasting
                axes = (axis,) if isinstance(axis, int) else tuple(sorted(axis))
                shape = list(t_a.shape)
                for ax in axes:
                    norm_ax = ax if ax >= 0 else t_a.ndim + ax
                    shape[norm_ax] = 1
                g = g.reshape(tuple(shape))

            expanded_g = np.broadcast_to(g, t_a.shape)
            t_a.grad = expanded_g if t_a.grad is None else t_a.grad + expanded_g

        out._backward = _backward

    return out


def mean_op(
    a: Any,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Tensor:
    """Tensor mean over specified axes."""
    t_a = to_tensor(a)
    out_data = np.mean(t_a.data, axis=axis, keepdims=keepdims)
    out = Tensor(out_data, requires_grad=t_a.requires_grad, _prev=(t_a,), _op="mean")

    if t_a.requires_grad:
        count = t_a.size / out.size

        def _backward() -> None:
            if out.grad is None:
                return
            g = out.grad
            if not keepdims and axis is not None:
                axes = (axis,) if isinstance(axis, int) else tuple(sorted(axis))
                shape = list(t_a.shape)
                for ax in axes:
                    norm_ax = ax if ax >= 0 else t_a.ndim + ax
                    shape[norm_ax] = 1
                g = g.reshape(tuple(shape))

            expanded_g = np.broadcast_to(g, t_a.shape) / count
            t_a.grad = expanded_g if t_a.grad is None else t_a.grad + expanded_g

        out._backward = _backward

    return out


def reshape(a: Any, target_shape: tuple[int, ...]) -> Tensor:
    """Reshape tensor to new shape."""
    t_a = to_tensor(a)
    out_data = t_a.data.reshape(target_shape)
    out = Tensor(out_data, requires_grad=t_a.requires_grad, _prev=(t_a,), _op="reshape")

    if t_a.requires_grad:
        def _backward() -> None:
            if out.grad is None:
                return
            g = out.grad.reshape(t_a.shape)
            t_a.grad = g if t_a.grad is None else t_a.grad + g

        out._backward = _backward

    return out


def transpose(a: Any, axes: Optional[tuple[int, ...]] = None) -> Tensor:
    """Permute the axes of a tensor."""
    t_a = to_tensor(a)
    out_data = np.transpose(t_a.data, axes=axes)
    out = Tensor(out_data, requires_grad=t_a.requires_grad, _prev=(t_a,), _op="transpose")

    if t_a.requires_grad:
        if axes is None:
            inv_axes = tuple(range(t_a.ndim)[::-1])
        else:
            inv_axes = tuple(np.argsort(axes))

        def _backward() -> None:
            if out.grad is None:
                return
            g = np.transpose(out.grad, axes=inv_axes)
            t_a.grad = g if t_a.grad is None else t_a.grad + g

        out._backward = _backward

    return out


def getitem(a: Any, idx: Any) -> Tensor:
    """Indexing and slicing operation."""
    t_a = to_tensor(a)
    out_data = t_a.data[idx]
    out = Tensor(out_data, requires_grad=t_a.requires_grad, _prev=(t_a,), _op="getitem")

    if t_a.requires_grad:
        def _backward() -> None:
            if out.grad is None:
                return
            g = np.zeros_like(t_a.data)
            np.add.at(g, idx, out.grad)
            t_a.grad = g if t_a.grad is None else t_a.grad + g

        out._backward = _backward

    return out


def logsumexp(z: Any, axis: int = -1, keepdims: bool = True) -> Tensor:
    """
    Numerically stable logsumexp operation with exact analytical derivative:
    LSE(z) = m + log(sum(exp(z - m))) where m = max(z).
    dLSE/dz_j = softmax(z)_j.
    """
    t_z = to_tensor(z)
    m = np.max(t_z.data, axis=axis, keepdims=True)
    exp_shifted = np.exp(t_z.data - m)
    sum_exp = np.sum(exp_shifted, axis=axis, keepdims=keepdims)
    out_data = m + np.log(sum_exp) if keepdims else (np.squeeze(m, axis=axis) + np.log(sum_exp))

    out = Tensor(out_data, requires_grad=t_z.requires_grad, _prev=(t_z,), _op="logsumexp")

    if t_z.requires_grad:
        # Precompute softmax for backprop
        softmax = exp_shifted / np.sum(exp_shifted, axis=axis, keepdims=True)

        def _backward() -> None:
            if out.grad is None:
                return
            g = out.grad
            if not keepdims:
                g = np.expand_dims(g, axis=axis)
            dz = g * softmax
            t_z.grad = dz if t_z.grad is None else t_z.grad + dz

        out._backward = _backward

    return out
