#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from _bootstrap import ensure_project_on_path

ensure_project_on_path()

from src.decomposition import decompose_bilinear_model
from src.model import build_image_classifier
from src.utils import ensure_dir, load_checkpoint, write_json


def validate_checkpoint_for_decomposition(payload: dict) -> dict:
    config = payload.get('config')
    if config is None:
        raise ValueError('Checkpoint is missing config; cannot reconstruct a bilinear model for decomposition.')

    if 'model' not in config:
        model_type = payload.get('model_type', 'unknown')
        raise ValueError(
            'analyze_checkpoint.py only supports bilinear student checkpoints with config["model"]. '
            f'Got checkpoint type {model_type!r}. Teacher checkpoints are not decomposable; '
            'use scripts/eval_checkpoint.py for teacher evaluation instead.'
        )

    return config


def main() -> None:
    parser = argparse.ArgumentParser(description='Export Task A decomposition artifacts from a checkpoint.')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to a saved checkpoint.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/analysis'),
        help='Directory for decomposition artifacts.',
    )
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f'Checkpoint path must be a file, got {args.checkpoint!s}. '
            'If this came from a shell variable, check that the checkpoint glob matched.'
        )

    payload = load_checkpoint(args.checkpoint, map_location='cpu')
    config = validate_checkpoint_for_decomposition(payload)
    model = build_image_classifier(config['model'], seed=int(config['seed']))
    model.load_state_dict(payload['model_state_dict'])
    model.eval()

    artifacts = decompose_bilinear_model(model)
    output_dir = ensure_dir(args.output_dir / args.checkpoint.stem)
    torch.save(artifacts.to_payload(), output_dir / 'decomposition.pt')

    top_class = int(artifacts.eigenvalues[:, -1].argmax().item())
    write_json(
        output_dir / 'summary.json',
        {
            'checkpoint': str(args.checkpoint),
            'dataset': config['dataset']['name'],
            'epoch': int(payload['epoch']),
            'val_acc': float(payload['metrics']['val_acc']),
            'top_class': top_class,
            'top_eigenvalue': float(artifacts.eigenvalues[top_class, -1].item()),
            'tensor_shapes': {
                'bilinear_tensor': list(artifacts.bilinear_tensor.shape),
                'symmetrized_tensor': list(artifacts.symmetrized_tensor.shape),
                'eigenvalues': list(artifacts.eigenvalues.shape),
                'eigenvectors_hidden': list(artifacts.eigenvectors_hidden.shape),
                'eigenvectors_input': list(artifacts.eigenvectors_input.shape),
            },
        },
    )
    print(f'Decomposition artifacts saved to {output_dir}')


if __name__ == '__main__':
    main()
