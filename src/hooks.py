from __future__ import annotations

import torch
from torch import nn


class ActivationCapture:
    def __init__(self, model: nn.Module, layer_names: list[str]) -> None:
        self.model = model
        self.layer_names = layer_names
        self.activations: dict[str, torch.Tensor] = {}
        self._handles: list = []

    def __enter__(self) -> 'ActivationCapture':
        named_modules = dict(self.model.named_modules())
        for name in self.layer_names:
            if name not in named_modules:
                raise KeyError(
                    f'Layer {name!r} not found in model. Available: {sorted(named_modules.keys())}'
                )
            handle = named_modules[name].register_forward_hook(self._make_hook(name))
            self._handles.append(handle)
        return self

    def __exit__(self, *args) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            self.activations[name] = out

        return hook
