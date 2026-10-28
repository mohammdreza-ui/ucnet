"""
Container modules for UCNet.
"""

from __future__ import annotations

from typing import Iterator, Sequence, Union

from ucnet.nn.module import Module
from ucnet.tensor import Tensor


class Sequential(Module):
    """
    A sequential container that chains modules in the order they were passed.
    """

    def __init__(self, *args: Union[Module, Sequence[Module]]) -> None:
        super().__init__()
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            modules = args[0]
        else:
            modules = args  # type: ignore[assignment]

        for idx, module in enumerate(modules):
            if not isinstance(module, Module):
                raise TypeError(f"Sequential argument at index {idx} must be a Module, got {type(module)}")
            self.add_module(str(idx), module)

    def add_module(self, name: str, module: Module) -> None:
        self._modules[name] = module
        super().__setattr__(name, module)

    def __getitem__(self, idx: Union[int, slice]) -> Union[Module, "Sequential"]:
        if isinstance(idx, slice):
            return Sequential(list(self._modules.values())[idx])
        elif isinstance(idx, int):
            if idx < 0:
                idx = len(self._modules) + idx
            return list(self._modules.values())[idx]  # type: ignore[return-value]
        raise TypeError(f"Invalid index type: {type(idx)}")

    def __len__(self) -> int:
        return len(self._modules)

    def __iter__(self) -> Iterator[Module]:
        return iter(self._modules.values())  # type: ignore[return-value]

    def forward(self, x: Tensor) -> Tensor:
        for module in self._modules.values():
            if module is not None:
                x = module(x)
        return x

    def __repr__(self) -> str:
        lines = [f"  ({key}): {repr(mod)}" for key, mod in self._modules.items() if mod is not None]
        return "Sequential(\n" + "\n".join(lines) + "\n)"
