"""
Training loop, early stopping, and training history management for UCNet.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Union

import numpy as np

from ucnet.nn.module import Module
from ucnet.ops import to_tensor
from ucnet.optim import Optimizer, clip_grad_norm
from ucnet.tensor import Tensor


@dataclass
class TrainHistory:
    epochs: list[int] = field(default_factory=list)
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)

    def save_csv(self, filepath: Union[str, Path]) -> None:
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "lr"])
            for ep, tr, val, lr in zip(self.epochs, self.train_losses, self.val_losses, self.learning_rates):
                writer.writerow([ep, tr, val, lr])


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False
        self.best_state: Optional[dict[str, np.ndarray]] = None

    def __call__(self, val_loss: float, model: Module) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.best_state = model.state_dict()
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def fit(
    model: Module,
    optimizer: Optimizer,
    loss_fn: Callable[..., Tensor],
    x_train: Union[np.ndarray, Tensor],
    y_train: Union[np.ndarray, Tensor],
    x_val: Optional[Union[np.ndarray, Tensor]] = None,
    y_val: Optional[Union[np.ndarray, Tensor]] = None,
    epochs: int = 100,
    batch_size: int = 32,
    max_grad_norm: Optional[float] = None,
    patience: Optional[int] = None,
    scheduler: Optional[Any] = None,
    verbose: bool = False,
    seed: Optional[int] = None,
) -> TrainHistory:
    """
    Standard mini-batch training loop with validation, early stopping, and history recording.

    When early stopping is active, the weights with the best validation loss are restored when the
    loop ends, whether it stopped early or simply ran out of epochs. Batch order is drawn from a
    dedicated generator: seeded with `seed`, or derived from the global NumPy state if `seed` is None
    (so `np.random.seed(...)` before the call still makes the run reproducible).
    """
    x_train_np = x_train.data if isinstance(x_train, Tensor) else np.asarray(x_train)
    y_train_np = y_train.data if isinstance(y_train, Tensor) else np.asarray(y_train)

    has_val = x_val is not None and y_val is not None
    if has_val:
        x_val_t = to_tensor(x_val)
        y_val_t = to_tensor(y_val)

    early_stopper = EarlyStopping(patience=patience) if patience is not None and has_val else None
    history = TrainHistory()
    N = len(x_train_np)
    rng = np.random.default_rng(seed if seed is not None else int(np.random.randint(0, 2**31 - 1)))

    for epoch in range(1, epochs + 1):
        model.train()
        indices = rng.permutation(N)
        epoch_losses: list[float] = []

        # Mini-batch iterations
        for start_idx in range(0, N, batch_size):
            batch_idx = indices[start_idx : start_idx + batch_size]
            xb = Tensor(x_train_np[batch_idx])
            yb = Tensor(y_train_np[batch_idx])

            optimizer.zero_grad()
            pred = model(xb)
            if isinstance(pred, tuple):
                loss = loss_fn(*pred, yb)
            else:
                loss = loss_fn(pred, yb)
            loss.backward()

            if max_grad_norm is not None:
                clip_grad_norm(model.parameters(), max_grad_norm)

            optimizer.step()
            epoch_losses.append(float(loss.data))

        train_loss = float(np.mean(epoch_losses))

        # Validation evaluation
        val_loss = train_loss
        if has_val:
            model.eval()
            val_pred = model(x_val_t)
            if isinstance(val_pred, tuple):
                val_loss = float(loss_fn(*val_pred, y_val_t).data)
            else:
                val_loss = float(loss_fn(val_pred, y_val_t).data)

        if scheduler is not None:
            scheduler.step()

        history.epochs.append(epoch)
        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)
        history.learning_rates.append(optimizer.lr)

        if verbose and (epoch % max(1, epochs // 10) == 0 or epoch == epochs):
            print(f"Epoch {epoch:4d}/{epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | LR: {optimizer.lr:.6f}")

        if early_stopper is not None and early_stopper(val_loss, model):
            if verbose:
                print(f"Early stopping triggered at epoch {epoch}")
            break

    if early_stopper is not None and early_stopper.best_state is not None:
        model.load_state_dict(early_stopper.best_state)

    return history
