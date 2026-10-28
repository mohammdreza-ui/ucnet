"""
First-principles numerical optimizers and learning rate schedulers for UCNet:
- SGD with momentum and weight decay
- Adam with analytical bias correction
- AdamW with decoupled weight decay
- Global norm gradient clipping
- Cosine Annealing learning rate schedule
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from ucnet.nn.module import Parameter


def clip_grad_norm(parameters: Iterable[Parameter], max_norm: float, eps: float = 1e-6) -> float:
    """
    Clips gradient norm of an iterable of parameters.
    The norm is computed over all gradients together, as if concatenated into a single vector.
    Returns the total L2 norm of the parameters before clipping.
    """
    params = [p for p in parameters if p.grad is not None]
    if len(params) == 0:
        return 0.0

    total_norm_sq = 0.0
    for p in params:
        assert p.grad is not None
        param_norm = np.linalg.norm(p.grad)
        total_norm_sq += float(param_norm ** 2)

    total_norm = float(np.sqrt(total_norm_sq))
    clip_coef = max_norm / (total_norm + eps)

    if clip_coef < 1.0:
        for p in params:
            assert p.grad is not None
            # Out-of-place: p.grad may be a read-only broadcast view or alias another parameter's grad.
            p.grad = p.grad * clip_coef

    return total_norm


class Optimizer:
    """Base class for all UCNet optimizers."""

    def __init__(self, params: Sequence[Parameter], lr: float, weight_decay: float = 0.0) -> None:
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.state: dict[int, dict[str, np.ndarray]] = {}

    def zero_grad(self) -> None:
        """Clear gradients for all managed parameters."""
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        """Perform a single optimization step."""
        raise NotImplementedError("Subclasses must implement step()")


class SGD(Optimizer):
    """
    Stochastic Gradient Descent with momentum and optional weight decay.
    v_t = momentum * v_{t-1} + g_t
    theta_t = theta_{t-1} - lr * v_t
    """

    def __init__(
        self,
        params: Sequence[Parameter],
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr, weight_decay)
        self.momentum = momentum

    def step(self) -> None:
        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.copy()

            if self.weight_decay != 0.0:
                grad += self.weight_decay * p.data

            p_id = id(p)
            if self.momentum != 0.0:
                if p_id not in self.state:
                    self.state[p_id] = {"momentum_buffer": grad.copy()}
                else:
                    buf = self.state[p_id]["momentum_buffer"]
                    buf[:] = self.momentum * buf + grad
                step_val = self.state[p_id]["momentum_buffer"]
            else:
                step_val = grad

            p.data -= self.lr * step_val


class Adam(Optimizer):
    """
    Adam optimizer (Kingma & Ba, 2015) with exact bias corrections:
    m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
    v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
    m_hat = m_t / (1 - beta1^t)
    v_hat = v_t / (1 - beta2^t)
    theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        params: Sequence[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        b1, b2 = self.beta1, self.beta2

        bias_correction1 = 1.0 - (b1 ** self.step_count)
        bias_correction2 = 1.0 - (b2 ** self.step_count)

        for p in self.params:
            if p.grad is None:
                continue

            grad = p.grad.copy()

            if self.weight_decay != 0.0:
                grad += self.weight_decay * p.data

            p_id = id(p)
            if p_id not in self.state:
                self.state[p_id] = {
                    "m": np.zeros_like(p.data),
                    "v": np.zeros_like(p.data),
                }

            st = self.state[p_id]
            m, v = st["m"], st["v"]

            m[:] = b1 * m + (1.0 - b1) * grad
            v[:] = b2 * v + (1.0 - b2) * (grad ** 2)

            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """
    AdamW optimizer (Loshchilov & Hutter, 2019) with decoupled weight decay:
    theta_t = theta_{t-1} * (1 - lr * weight_decay) - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(
        self,
        params: Sequence[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__(params, lr, weight_decay)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1
        b1, b2 = self.beta1, self.beta2

        bias_correction1 = 1.0 - (b1 ** self.step_count)
        bias_correction2 = 1.0 - (b2 ** self.step_count)

        for p in self.params:
            if p.grad is None:
                continue

            # 1. Decoupled weight decay update
            if self.weight_decay != 0.0:
                p.data -= self.lr * self.weight_decay * p.data

            grad = p.grad

            p_id = id(p)
            if p_id not in self.state:
                self.state[p_id] = {
                    "m": np.zeros_like(p.data),
                    "v": np.zeros_like(p.data),
                }

            st = self.state[p_id]
            m, v = st["m"], st["v"]

            m[:] = b1 * m + (1.0 - b1) * grad
            v[:] = b2 * v + (1.0 - b2) * (grad ** 2)

            m_hat = m / bias_correction1
            v_hat = v / bias_correction2

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class CosineAnnealingLR:
    """
    Cosine annealing learning rate schedule:
    eta_t = eta_min + 0.5 * (eta_max - eta_min) * (1 + cos(pi * t / T_max))
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
    ) -> None:
        self.optimizer = optimizer
        self.T_max = max(1, T_max)
        self.eta_min = eta_min
        self.eta_max = optimizer.lr
        self.current_step = 0

    def step(self) -> float:
        self.current_step += 1
        t = min(self.current_step, self.T_max)
        new_lr = self.eta_min + 0.5 * (self.eta_max - self.eta_min) * (1.0 + np.cos(np.pi * t / self.T_max))
        self.optimizer.lr = new_lr
        return new_lr
