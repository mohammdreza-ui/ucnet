"""
Core Tensor class and automatic differentiation engine for UCNet.
Implemented from scratch in pure NumPy.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Sequence, Union

import numpy as np


def unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """
    Sum gradients across broadcasted dimensions to match target_shape exactly.

    Handles:
    1. Scalar targets: target_shape == ()
    2. Leading broadcast dimensions (e.g. grad is (2, 3, 4), target is (3, 4))
    3. Trailing/internal size-1 broadcast dimensions (e.g. grad is (2, 3, 4), target is (2, 1, 4))
    """
    if grad.shape == target_shape:
        return grad

    # 1. Target is a scalar
    if len(target_shape) == 0:
        return np.sum(grad).reshape(())

    # 2. Leading dimensions
    leading_dims = grad.ndim - len(target_shape)
    if leading_dims > 0:
        grad = grad.sum(axis=tuple(range(leading_dims)))

    # 3. Size-1 dimensions
    for ax, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape)):
        if t_dim == 1 and g_dim != 1:
            grad = grad.sum(axis=ax, keepdims=True)

    return grad.reshape(target_shape)


class Tensor:
    """
    A multi-dimensional array with automatic reverse-mode differentiation support.
    """

    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        dtype: Optional[np.dtype] = None,
        _prev: Sequence["Tensor"] = (),
        _op: str = "",
    ) -> None:
        if isinstance(data, Tensor):
            data = data.data

        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=dtype if dtype is not None else np.float64)
        elif dtype is not None and data.dtype != dtype:
            data = data.astype(dtype)
        elif data.dtype == np.float32 and dtype is None:
            # Keep as float32 if explicitly provided as float32 ndarray
            pass
        elif np.issubdtype(data.dtype, np.integer) and dtype is None:
            # Default integer arrays to float64 for differentiability unless specified
            data = data.astype(np.float64)

        self.data: np.ndarray = data
        self.requires_grad: bool = requires_grad
        self.grad: Optional[np.ndarray] = None
        self._prev: tuple[Tensor, ...] = tuple(_prev)
        self._backward: Optional[Callable[[], None]] = None
        self._op: str = _op

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def T(self) -> "Tensor":
        return self.transpose()

    def zero_grad(self) -> None:
        """Reset gradient to None."""
        self.grad = None

    def detach(self) -> "Tensor":
        """Return a new Tensor detached from the computation graph."""
        return Tensor(self.data.copy(), requires_grad=False)

    def numpy(self) -> np.ndarray:
        """Return a copy of the underlying numpy array."""
        return self.data.copy()

    def item(self) -> Union[float, int, bool]:
        """Return the standard Python scalar value of a 1-element tensor."""
        return self.data.item()

    def backward(self, gradient: Optional[Union[np.ndarray, "Tensor", float]] = None) -> None:
        """
        Compute gradients of this tensor with respect to all graph leaves via reverse-mode autodiff.
        """
        if not self.requires_grad:
            return

        if gradient is None:
            if self.shape == () or self.shape == (1,):
                gradient = np.ones_like(self.data, dtype=self.data.dtype)
            else:
                raise RuntimeError("grad can be implicitly created only for scalar outputs")
        elif isinstance(gradient, Tensor):
            gradient = gradient.data
        elif not isinstance(gradient, np.ndarray):
            gradient = np.array(gradient, dtype=self.data.dtype)

        if gradient.shape != self.shape:
            raise ValueError(f"Gradient shape {gradient.shape} does not match tensor shape {self.shape}")

        # Build topological graph ordering via Depth-First Search
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build_topo(v: Tensor) -> None:
            v_id = id(v)
            if v_id not in visited:
                visited.add(v_id)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Adjoints of non-leaf nodes are per-call quantities: clear any stale value left by an
        # earlier backward pass, otherwise a shared subgraph would re-propagate old adjoints.
        # Leaves (no _backward) keep accumulating across calls, as in standard autodiff APIs.
        for node in topo:
            if node._backward is not None:
                node.grad = None

        # Out-of-place accumulation: self.grad may be a read-only view or alias another tensor's grad.
        self.grad = gradient.copy() if self.grad is None else self.grad + gradient

        # Backpropagate adjoints in reverse topological order
        for node in reversed(topo):
            if node._backward is not None and node.requires_grad:
                node._backward()

    # Operator overloads
    def __add__(self, other: Any) -> "Tensor":
        from ucnet.ops import add
        return add(self, other)

    def __radd__(self, other: Any) -> "Tensor":
        from ucnet.ops import add
        return add(other, self)

    def __sub__(self, other: Any) -> "Tensor":
        from ucnet.ops import sub
        return sub(self, other)

    def __rsub__(self, other: Any) -> "Tensor":
        from ucnet.ops import sub
        return sub(other, self)

    def __mul__(self, other: Any) -> "Tensor":
        from ucnet.ops import mul
        return mul(self, other)

    def __rmul__(self, other: Any) -> "Tensor":
        from ucnet.ops import mul
        return mul(other, self)

    def __truediv__(self, other: Any) -> "Tensor":
        from ucnet.ops import div
        return div(self, other)

    def __rtruediv__(self, other: Any) -> "Tensor":
        from ucnet.ops import div
        return div(other, self)

    def __pow__(self, power: Union[int, float, "Tensor"]) -> "Tensor":
        from ucnet.ops import pow_op
        return pow_op(self, power)

    def __rpow__(self, base: Union[int, float, "Tensor"]) -> "Tensor":
        from ucnet.ops import pow_op
        return pow_op(base, self)

    def __neg__(self) -> "Tensor":
        from ucnet.ops import neg
        return neg(self)

    def __matmul__(self, other: Any) -> "Tensor":
        from ucnet.ops import matmul
        return matmul(self, other)

    def __rmatmul__(self, other: Any) -> "Tensor":
        from ucnet.ops import matmul
        return matmul(other, self)

    def __getitem__(self, idx: Any) -> "Tensor":
        from ucnet.ops import getitem
        return getitem(self, idx)

    def sum(self, axis: Optional[Union[int, tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        from ucnet.ops import sum_op
        return sum_op(self, axis=axis, keepdims=keepdims)

    def mean(self, axis: Optional[Union[int, tuple[int, ...]]] = None, keepdims: bool = False) -> "Tensor":
        from ucnet.ops import mean_op
        return mean_op(self, axis=axis, keepdims=keepdims)

    def reshape(self, *shape: Union[int, tuple[int, ...], list[int]]) -> "Tensor":
        from ucnet.ops import reshape
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            target_shape = tuple(shape[0])
        else:
            target_shape = tuple(shape)  # type: ignore[arg-type]
        return reshape(self, target_shape)

    def transpose(self, *axes: Union[int, tuple[int, ...], list[int]]) -> "Tensor":
        from ucnet.ops import transpose
        if len(axes) == 0:
            target_axes = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            target_axes = tuple(axes[0])
        else:
            target_axes = tuple(axes)  # type: ignore[arg-type]
        return transpose(self, target_axes)

    def exp(self) -> "Tensor":
        from ucnet.ops import exp_op
        return exp_op(self)

    def log(self) -> "Tensor":
        from ucnet.ops import log_op
        return log_op(self)

    def tanh(self) -> "Tensor":
        from ucnet.ops import tanh_op
        return tanh_op(self)

    def sigmoid(self) -> "Tensor":
        from ucnet.ops import sigmoid_op
        return sigmoid_op(self)

    def relu(self) -> "Tensor":
        from ucnet.ops import relu_op
        return relu_op(self)

    def softplus(self) -> "Tensor":
        from ucnet.ops import softplus_op
        return softplus_op(self)

    def clip(self, min_val: Optional[float] = None, max_val: Optional[float] = None) -> "Tensor":
        from ucnet.ops import clip_op
        return clip_op(self, min_val, max_val)

    def squeeze(self, axis: Optional[Union[int, tuple[int, ...]]] = None) -> "Tensor":
        new_shape = np.squeeze(self.data, axis=axis).shape
        return self.reshape(new_shape)

    def unsqueeze(self, axis: int) -> "Tensor":
        new_shape = list(self.shape)
        if axis < 0:
            axis = len(new_shape) + 1 + axis
        new_shape.insert(axis, 1)
        return self.reshape(tuple(new_shape))

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> "Tensor":
        if end_dim < 0:
            end_dim = self.ndim + end_dim
        dim_start = self.shape[:start_dim]
        dim_flat = int(np.prod(self.shape[start_dim : end_dim + 1]))
        dim_end = self.shape[end_dim + 1 :]
        target_shape = dim_start + (dim_flat,) + dim_end
        return self.reshape(target_shape)

    def __repr__(self) -> str:
        grad_str = f", requires_grad={self.requires_grad}" if self.requires_grad else ""
        op_str = f", op='{self._op}'" if self._op else ""
        return f"Tensor({self.data}{grad_str}{op_str})"
