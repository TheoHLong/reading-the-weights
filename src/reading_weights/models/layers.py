from __future__ import annotations

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
