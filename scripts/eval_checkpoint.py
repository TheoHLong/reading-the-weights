#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from torch import nn

from _bootstrap import ensure_project_on_path

ensure_project_on_path()

from src.data import build_image_dataloaders
from src.guide import GUIDE_ARCHITECTURE, build_guide
from src.model import build_image_classifier, load_image_classifier_state
from src.train import evaluate
from src.utils import load_checkpoint, resolve_device


def build_model_from_checkpoint(payload: dict, device) -> tuple[nn.Module, str]:
    if 'config' not in payload:
        raise ValueError('Checkpoint must include config to run evaluation.')

    if payload.get('model_type') == GUIDE_ARCHITECTURE:
        model = build_guide(payload['config'])
        model_label = GUIDE_ARCHITECTURE
    elif 'model' in payload['config']:
        model = build_image_classifier(payload['config']['model'], seed=int(payload['config']['seed']))
        model_label = 'bilinear_student'
    else:
        raise ValueError('Unsupported checkpoint type for evaluation.')

    if model_label == 'bilinear_student':
        load_image_classifier_state(model, payload['model_state_dict'])
    else:
        model.load_state_dict(payload['model_state_dict'])
    model.to(device)
    return model, model_label


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a saved checkpoint on val/test split.')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to a saved checkpoint.')
    parser.add_argument(
        '--split',
        choices=('val', 'test'),
        default='test',
        help='Which held-out split to evaluate.',
    )
    parser.add_argument(
        '--device',
        default='auto',
        help="Device override. Default 'auto' matches the repo's existing convention.",
    )
    parser.add_argument(
        '--max-batches',
        type=int,
        default=0,
        help='Optional debug limiter. Use 0 to evaluate the full split.',
    )
    args = parser.parse_args()

    payload = load_checkpoint(args.checkpoint, map_location='cpu')
    config = payload['config']
    config.setdefault('train', {})
    config['train'].setdefault('split_seed', int(config['seed']))

    dataset_bundle = build_image_dataloaders(config['dataset'], config['train'])
    loader = dataset_bundle.val_loader if args.split == 'val' else dataset_bundle.test_loader
    device = resolve_device(args.device)
    model, model_label = build_model_from_checkpoint(payload, device)
    criterion = nn.CrossEntropyLoss()
    max_batches = None if args.max_batches in (None, 0) else int(args.max_batches)
    loss, acc = evaluate(model, loader, criterion, device, max_batches=max_batches)

    print(f'checkpoint: {args.checkpoint}')
    print(f'model_type: {model_label}')
    print(f'split: {args.split}')
    print(f'loss: {loss:.6f}')
    print(f'acc: {acc:.4f}')


if __name__ == '__main__':
    main()
