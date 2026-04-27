from __future__ import annotations

from pathlib import Path
from typing import Any

from torch import Tensor, nn
from torchvision import models

from reading_weights.utils import load_checkpoint


GUIDE_ARCHITECTURE = 'cifar_resnet18'


def build_cifar_resnet18(num_classes: int = 10) -> nn.Module:
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)

    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_guide(config: dict[str, Any]) -> nn.Module:
    architecture = config.get('guide', {}).get('architecture', GUIDE_ARCHITECTURE)
    if architecture != GUIDE_ARCHITECTURE:
        raise ValueError(f'Unsupported guide architecture: {architecture}')
    return build_cifar_resnet18(num_classes=int(config['dataset']['num_classes']))


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


def get_guide_features(model: nn.Module, x: Tensor, layer_names: list[str]) -> dict[str, Tensor]:
    raise NotImplementedError('Guide feature extraction is reserved for future CKA experiments.')
