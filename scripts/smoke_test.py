#!/usr/bin/env python
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import torch

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from reading_weights.analysis.decomposition import decompose_bilinear_model
from reading_weights.config import load_config
from reading_weights.models.image_classifier import build_image_classifier
from reading_weights.train import train_image_experiment
from reading_weights.utils import ensure_dir, load_checkpoint, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a tiny end-to-end Task A smoke test.')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/mnist_baseline.yaml'),
        help='Baseline config to derive the smoke test from.',
    )
    args = parser.parse_args()

    config = deepcopy(load_config(args.config))
    config['experiment_name'] = f"{config['experiment_name']}_smoke"
    config['train']['epochs'] = 2
    config['train']['batch_size'] = min(int(config['train']['batch_size']), 128)
    config['train']['num_workers'] = 0
    config['train']['pin_memory'] = False
    config['train']['max_train_batches'] = 2
    config['train']['max_eval_batches'] = 1

    artifacts = train_image_experiment(config)
    payload = load_checkpoint(artifacts['best_checkpoint_path'], map_location='cpu')

    model = build_image_classifier(payload['config']['model'], seed=int(payload['config']['seed']))
    model.load_state_dict(payload['model_state_dict'])
    model.eval()

    decomposition = decompose_bilinear_model(model)
    output_dir = ensure_dir(Path('results/analysis') / artifacts['best_checkpoint_path'].stem)
    torch.save(decomposition.to_payload(), output_dir / 'decomposition.pt')

    expected_classes = int(payload['config']['dataset']['num_classes'])
    expected_hidden = int(payload['config']['model']['d_hidden'])
    expected_input = int(payload['config']['model']['d_input'])

    assert decomposition.eigenvalues.shape == (expected_classes, expected_hidden)
    assert decomposition.eigenvectors_hidden.shape == (expected_classes, expected_hidden, expected_hidden)
    assert decomposition.eigenvectors_input.shape == (expected_classes, expected_hidden, expected_input)

    write_json(
        output_dir / 'summary.json',
        {
            'checkpoint': str(artifacts['best_checkpoint_path']),
            'dataset': payload['config']['dataset']['name'],
            'epoch': int(payload['epoch']),
            'val_acc': float(payload['metrics']['val_acc']),
            'tensor_shapes': {
                'bilinear_tensor': list(decomposition.bilinear_tensor.shape),
                'symmetrized_tensor': list(decomposition.symmetrized_tensor.shape),
                'eigenvalues': list(decomposition.eigenvalues.shape),
                'eigenvectors_hidden': list(decomposition.eigenvectors_hidden.shape),
                'eigenvectors_input': list(decomposition.eigenvectors_input.shape),
            },
        },
    )

    print('Smoke test passed.')
    print(f"metrics_path: {artifacts['metrics_path']}")
    print(f"checkpoint: {artifacts['best_checkpoint_path']}")
    print(f"analysis_dir: {output_dir}")


if __name__ == '__main__':
    main()
