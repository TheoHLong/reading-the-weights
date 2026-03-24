#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch

from reading_weights.analysis.decomposition import decompose_bilinear_model
from reading_weights.models.image_classifier import build_image_classifier
from reading_weights.utils import ensure_dir, load_checkpoint, write_json


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

    payload = load_checkpoint(args.checkpoint, map_location='cpu')
    config = payload['config']
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
