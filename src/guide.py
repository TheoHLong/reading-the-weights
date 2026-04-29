from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models

from src.utils import load_checkpoint


GUIDE_ARCHITECTURE = 'cifar_resnet18'
MLP_GUIDE_ARCHITECTURE = 'image_mlp'


class ImageMLPGuide(nn.Module):
    def __init__(
        self,
        d_input: int,
        hidden_dim: int,
        depth: int,
        num_classes: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f'ImageMLPGuide depth must be >= 1, got {depth}')

        self.flatten = nn.Flatten()
        self.embed = nn.Sequential(
            nn.Linear(d_input, hidden_dim, bias=bias),
            nn.ReLU(),
        )
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim, bias=bias),
                nn.ReLU(),
            )
            for _ in range(max(depth - 1, 0))
        ])
        self.head = nn.Linear(hidden_dim, num_classes, bias=bias)

    def forward(self, x):
        x = self.flatten(x)
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def build_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_image_mlp_guide(config: dict[str, Any]) -> nn.Module:
    dataset_cfg = config['dataset']
    guide_cfg = config.get('guide', {})
    image_size = int(dataset_cfg['image_size'])
    channels = int(dataset_cfg['channels'])
    d_input = channels * image_size * image_size
    hidden_dim = int(guide_cfg.get('hidden_dim', 512))
    depth = int(guide_cfg.get('depth', 5))
    bias = bool(guide_cfg.get('bias', False))
    num_classes = int(dataset_cfg['num_classes'])
    return ImageMLPGuide(
        d_input=d_input,
        hidden_dim=hidden_dim,
        depth=depth,
        num_classes=num_classes,
        bias=bias,
    )


def build_guide(config: dict[str, Any]) -> nn.Module:
    architecture = config.get('guide', {}).get('architecture', GUIDE_ARCHITECTURE)
    if architecture == GUIDE_ARCHITECTURE:
        return build_cifar_resnet18(num_classes=int(config['dataset']['num_classes']))
    if architecture == MLP_GUIDE_ARCHITECTURE:
        return build_image_mlp_guide(config)
    raise ValueError(f'Unsupported guide architecture: {architecture}')


def load_frozen_teacher(checkpoint_path: str | Path, device) -> nn.Module:
    payload = load_checkpoint(checkpoint_path, map_location='cpu')
    if 'config' in payload:
        model = build_guide(payload['config'])
    else:
        model = build_cifar_resnet18(num_classes=int(payload.get('num_classes', 10)))

    model.load_state_dict(payload['model_state_dict'])
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def load_frozen_guide(config: dict[str, Any], transfer_cfg: dict[str, Any], device) -> nn.Module:
    teacher_source = str(transfer_cfg.get('teacher_source', 'checkpoint'))
    if teacher_source == 'checkpoint':
        return load_frozen_teacher(transfer_cfg['teacher_checkpoint'], device)
    if teacher_source != 'random':
        raise ValueError(f'Unsupported teacher_source: {teacher_source!r}')

    guide_seed = int(transfer_cfg.get('guide_seed', config.get('seed', 42)))
    guide_cfg = dict(config.get('guide', {}))
    override_architecture = transfer_cfg.get('guide_architecture')
    if override_architecture is not None:
        guide_cfg['architecture'] = override_architecture
    random_guide_config = dict(config)
    random_guide_config['guide'] = guide_cfg

    cpu_rng_state = torch.random.get_rng_state()
    try:
        torch.manual_seed(guide_seed)
        model = build_guide(random_guide_config)
    finally:
        torch.random.set_rng_state(cpu_rng_state)

    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model
