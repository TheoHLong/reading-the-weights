from __future__ import annotations

import math

import torch
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn


class SignedQuadraticShrink(nn.Module):
    """Signed Quadratic Shrink gate with the paper's p=1 defaults."""

    def __init__(self, c: float = 0.01, lambd: float = 0.5) -> None:
        super().__init__()
        self.c = c
        self.lambd = lambd

    def forward(self, x: Tensor) -> Tensor:
        sign = torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))
        return (x - self.c * sign) / (1.0 + self.lambd * x * sign)


def _build_gate(gate: str | None) -> nn.Module:
    return {
        'relu': nn.ReLU(),
        'silu': nn.SiLU(),
        'gelu': nn.GELU(),
        'sqs': SignedQuadraticShrink(),
        None: nn.Identity(),
    }[gate]


class Bilinear(nn.Linear):
    """Bilinear layer implemented as two linear projections multiplied elementwise."""

    def __init__(self, d_in: int, d_out: int, bias: bool = False, gate: str | None = None) -> None:
        super().__init__(d_in, 2 * d_out, bias=bias)
        self.gate_name = gate
        self.gate = _build_gate(gate)

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
        self.gate = _build_gate(gate)

    def forward(self, x: Float[Tensor, '... d_in']) -> Float[Tensor, '... d_out']:
        return self.gate(super().forward(x))


def build_input_projection(
    *,
    d_input: int,
    preprocess: str | None,
    input_channels: int | None,
    image_size: int | None,
    pool_kernel: int | None,
    pool_stride: int | None,
) -> Tensor | None:
    if preprocess in (None, 'identity'):
        return None

    if preprocess != 'avg_pool':
        raise ValueError(f'Unsupported preprocess mode: {preprocess}')
    if input_channels is None or image_size is None:
        raise ValueError('avg_pool preprocess requires input_channels and image_size')

    input_channels = int(input_channels)
    image_size = int(image_size)
    pool_kernel = int(pool_kernel or 2)
    pool_stride = int(pool_stride or pool_kernel)
    pooled_size = math.floor((image_size - pool_kernel) / pool_stride) + 1
    if pooled_size <= 0:
        raise ValueError('avg_pool preprocess produced a non-positive spatial size')

    projection = torch.zeros(input_channels * pooled_size * pooled_size, d_input)
    patch_area = float(pool_kernel * pool_kernel)

    row_idx = 0
    for channel in range(input_channels):
        channel_offset = channel * image_size * image_size
        for out_row in range(pooled_size):
            row_start = out_row * pool_stride
            for out_col in range(pooled_size):
                col_start = out_col * pool_stride
                for k_row in range(pool_kernel):
                    for k_col in range(pool_kernel):
                        in_row = row_start + k_row
                        in_col = col_start + k_col
                        flat_idx = channel_offset + in_row * image_size + in_col
                        projection[row_idx, flat_idx] = 1.0 / patch_area
                row_idx += 1

    return projection


class BilinearImageClassifier(nn.Module):
    def __init__(
        self,
        d_input: int,
        d_hidden: int,
        d_output: int,
        n_layer: int = 1,
        bias: bool = False,
        residual: bool = False,
        gate: str | None = None,
        preprocess: str | None = None,
        input_channels: int | None = None,
        image_size: int | None = None,
        pool_kernel: int | None = None,
        pool_stride: int | None = None,
        seed: int = 42,
    ) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.residual = residual
        self.gate = gate
        self.raw_input_dim = d_input
        self.preprocess = preprocess or 'identity'

        input_projection = build_input_projection(
            d_input=d_input,
            preprocess=self.preprocess,
            input_channels=input_channels,
            image_size=image_size,
            pool_kernel=pool_kernel,
            pool_stride=pool_stride,
        )
        self.register_buffer(
            'input_projection',
            input_projection if input_projection is not None else torch.empty(0),
        )
        self.has_input_projection = input_projection is not None
        processed_input_dim = int(input_projection.shape[0]) if input_projection is not None else d_input

        self.embed = Linear(processed_input_dim, d_hidden, bias=False)
        self.blocks = nn.ModuleList([
            Bilinear(d_hidden, d_hidden, bias=bias, gate=gate) for _ in range(n_layer)
        ])
        self.head = Linear(d_hidden, d_output, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        x = x.flatten(start_dim=1)
        if self.has_input_projection:
            x = x @ self.input_projection.T
        x = self.embed(x)
        for layer in self.blocks:
            x = x + layer(x) if self.residual else layer(x)
        return self.head(x)

    @property
    def embedding_weight(self) -> Tensor:
        if not self.has_input_projection:
            return self.embed.weight.detach()
        return self.embed.weight.detach() @ self.input_projection.detach()

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
        gate=model_cfg.get('gate'),
        preprocess=model_cfg.get('preprocess'),
        input_channels=model_cfg.get('input_channels'),
        image_size=model_cfg.get('image_size'),
        pool_kernel=model_cfg.get('pool_kernel'),
        pool_stride=model_cfg.get('pool_stride'),
        seed=seed,
    )


def load_image_classifier_state(model: BilinearImageClassifier, state_dict: dict) -> None:
    load_result = model.load_state_dict(state_dict, strict=False)
    tolerated_missing = {'input_projection'}
    missing = set(load_result.missing_keys) - tolerated_missing
    if missing or load_result.unexpected_keys:
        raise RuntimeError(
            'Checkpoint state dict mismatch: '
            f'missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}'
        )
