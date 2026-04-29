from __future__ import annotations

import torch
from torch import Tensor


def _gram_centered(x: Tensor) -> Tensor:
    x = x - x.mean(dim=0, keepdim=True)
    return x @ x.T


def linear_cka(x: Tensor, y: Tensor, eps: float = 1e-8) -> Tensor:
    if x.shape[0] != y.shape[0]:
        raise ValueError(f'CKA requires matching batch sizes, got {x.shape[0]} vs {y.shape[0]}.')

    K = _gram_centered(x.float())
    L = _gram_centered(y.float())
    hsic_xy = (K * L).sum()
    hsic_xx = (K * K).sum()
    hsic_yy = (L * L).sum()
    return hsic_xy / (hsic_xx.sqrt() * hsic_yy.sqrt() + eps)


def cka_distance(x: Tensor, y: Tensor) -> Tensor:
    return 1.0 - linear_cka(x, y)
