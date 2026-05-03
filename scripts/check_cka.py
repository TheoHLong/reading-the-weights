#!/usr/bin/env python
from __future__ import annotations

import torch

from _bootstrap import ensure_project_on_path

ensure_project_on_path()

from src.cka import linear_cka


def main() -> None:
    atol = 1e-5
    x = torch.randn(64, 128)
    self_score = linear_cka(x, x)
    assert torch.isclose(self_score, torch.tensor(1.0), atol=atol)

    y = torch.randn(64, 128)
    score = linear_cka(x, y)
    assert -atol <= score.item() <= 1.0 + atol

    perm = torch.randperm(128)
    perm_score = linear_cka(x, x[:, perm])
    assert torch.isclose(perm_score, torch.tensor(1.0), atol=atol)

    scale_score = linear_cka(x, 5.0 * x)
    assert torch.isclose(scale_score, torch.tensor(1.0), atol=atol)

    print('CKA sanity checks passed.')


if __name__ == '__main__':
    main()
