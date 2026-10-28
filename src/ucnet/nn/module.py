"""
Base Module and Parameter classes for UCNet neural networks.
Provides parameter registration, state dict manipulation, and hierarchy management.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional

import numpy as np

from ucnet.tensor import Tensor


class Parameter(Tensor):
    """
    A Tensor that is automatically registered as a learnable parameter when assigned to a Module.
    Requires gradient by default.
    """

    def __init__(
        self,
        data: Any,
        requires_grad: bool = True,
        dtype: Optional[np.dtype] = None,
    ) -> None:
        super().__init__(data, requires_grad=requires_grad, dtype=dtype)


class Module:
    """
    Base class for all neural network modules in UCNet.
    """

    def __init__(self) -> None:
        self._parameters: dict[str, Optional[Parameter]] = {}
        self._modules: dict[str, Optional[Module]] = {}
        self.training: bool = True

    def __setattr__(self, name: str, value: Any) -> None:
        if isinstance(value, Parameter):
            if hasattr(self, "_parameters"):
                self._parameters[name] = value
        elif isinstance(value, Module):
            if hasattr(self, "_modules"):
                self._modules[name] = value
        super().__setattr__(name, value)

    def register_parameter(self, name: str, param: Optional[Parameter]) -> None:
        """Register a parameter with this module."""
        self._parameters[name] = param
        super().__setattr__(name, param)

    def parameters(self) -> list[Parameter]:
        """Return a list of all parameters in this module and submodules."""
        params: list[Parameter] = []
        for param in self._parameters.values():
            if param is not None:
                params.append(param)
        for module in self._modules.values():
            if module is not None:
                params.extend(module.parameters())
        return params

    def named_parameters(self, prefix: str = "") -> Iterator[tuple[str, Parameter]]:
        """Yield (name, parameter) tuples recursively."""
        for name, param in self._parameters.items():
            if param is not None:
                full_name = f"{prefix}.{name}" if prefix else name
                yield full_name, param
        for mod_name, module in self._modules.items():
            if module is not None:
                sub_prefix = f"{prefix}.{mod_name}" if prefix else mod_name
                yield from module.named_parameters(prefix=sub_prefix)

    def modules(self) -> list[Module]:
        """Return a list of all child modules recursively."""
        mods = [self]
        for m in self._modules.values():
            if m is not None:
                mods.extend(m.modules())
        return mods

    def zero_grad(self) -> None:
        """Reset gradients of all parameters to None."""
        for p in self.parameters():
            p.zero_grad()

    def train(self, mode: bool = True) -> "Module":
        """Set module training mode (affects dropout, etc.)."""
        self.training = mode
        for module in self._modules.values():
            if module is not None:
                module.train(mode)
        return self

    def eval(self) -> "Module":
        """Set module evaluation mode."""
        return self.train(False)

    def state_dict(self, prefix: str = "") -> dict[str, np.ndarray]:
        """Return a dictionary containing parameter numpy arrays."""
        state: dict[str, np.ndarray] = {}
        for name, param in self.named_parameters(prefix=prefix):
            state[name] = param.data.copy()
        return state

    def load_state_dict(self, state_dict: dict[str, np.ndarray], strict: bool = True) -> None:
        """Load parameter numpy arrays from a state dict."""
        named_params = dict(self.named_parameters())
        for name, param in named_params.items():
            if name in state_dict:
                param.data = np.array(state_dict[name], dtype=param.dtype).copy()
            elif strict:
                raise KeyError(f"Missing key '{name}' in state_dict")

        if strict:
            for key in state_dict:
                if key not in named_params:
                    raise KeyError(f"Unexpected key '{key}' in state_dict")

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Subclasses of Module must implement forward")

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)
