from __future__ import annotations

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn


class Bilinear(nn.Linear):
    """Bilinear layer implemented as two linear projections multiplied elementwise."""

    def __init__(self, d_in: int, d_out: int, bias: bool = False, gate: str | None = None) -> None:
        super().__init__(d_in, 2 * d_out, bias=bias)
        self.gate = {
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),
            'gelu': nn.GELU(),
            None: nn.Identity(),
        }[gate]

    def forward(self, x: Float[Tensor, '... d_in']) -> Float[Tensor, '... d_out']:
        left, right = super().forward(x).chunk(2, dim=-1)
        return self.gate(left) * right

    @property
    def w_l(self) -> Tensor:
        return self.weight.chunk(2, dim=0)[0]

    @property
    def w_r(self) -> Tensor:
        return self.weight.chunk(2, dim=0)[1]


class Linear(nn.Linear):
    def __init__(self, d_in: int, d_out: int, bias: bool = False, gate: str | None = None) -> None:
        super().__init__(d_in, d_out, bias=bias)
        self.gate = {
            'relu': nn.ReLU(),
            'silu': nn.SiLU(),
            'gelu': nn.GELU(),
            None: nn.Identity(),
        }[gate]

    def forward(self, x: Float[Tensor, '... d_in']) -> Float[Tensor, '... d_out']:
        return self.gate(super().forward(x))


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
