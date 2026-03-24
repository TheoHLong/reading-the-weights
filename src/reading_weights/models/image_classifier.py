from __future__ import annotations

import torch
from einops import rearrange
from torch import Tensor, nn

from reading_weights.models.layers import Bilinear, Linear


class BilinearImageClassifier(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        d_output: int,
        n_layer: int = 1,
        bias: bool = False,
        residual: bool = False,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.residual = residual

        self.embed = Linear(d_input, d_hidden, bias=False)
        self.blocks = nn.ModuleList([
            Bilinear(d_hidden, d_hidden, bias=bias) for _ in range(n_layer)
        ])
        self.head = Linear(d_hidden, d_output, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embed(x.flatten(start_dim=1))
        for layer in self.blocks:
            x = x + layer(x) if self.residual else layer(x)
        return self.head(x)

    @property
    def embedding_weight(self) -> Tensor:
        return self.embed.weight.detach()

    @property
    def output_weight(self) -> Tensor:
        return self.head.weight.detach()

    @property
    def bilinear_weights(self) -> Tensor:
        return torch.stack([
            rearrange(layer.weight.detach(), '(side out) hidden -> side out hidden', side=2)
            for layer in self.blocks
        ])


def build_image_classifier(model_cfg: dict, seed: int) -> BilinearImageClassifier:
    return BilinearImageClassifier(
        d_input=int(model_cfg['d_input']),
        d_hidden=int(model_cfg['d_hidden']),
        d_output=int(model_cfg['d_output']),
        n_layer=int(model_cfg['n_layer']),
        bias=bool(model_cfg['bias']),
        residual=bool(model_cfg['residual']),
        seed=seed,
    )
