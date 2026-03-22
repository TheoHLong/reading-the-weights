from __future__ import annotations

import torch
from einops import einsum, rearrange
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
    def w_e(self) -> Tensor:
        return self.embed.weight.detach()

    @property
    def w_u(self) -> Tensor:
        return self.head.weight.detach()

    @property
    def w_lr(self) -> Tensor:
        return torch.stack([
            rearrange(layer.weight.detach(), '(s o) h -> s o h', s=2)
            for layer in self.blocks
        ])

    def decompose(self) -> tuple[Tensor, Tensor]:
        if len(self.blocks) != 1:
            raise ValueError('decompose() currently supports single-layer models only.')

        left, right = self.w_lr[0].unbind()
        bilinear_tensor = einsum(
            self.w_u,
            left,
            right,
            'cls out, out in1, out in2 -> cls in1 in2',
        )
        bilinear_tensor = 0.5 * (bilinear_tensor + bilinear_tensor.mT)

        eigenvalues, eigenvectors = torch.linalg.eigh(bilinear_tensor)
        eigenvectors = einsum(
            eigenvectors,
            self.w_e,
            'cls emb comp, emb inp -> cls comp inp',
        )
        return eigenvalues, eigenvectors
