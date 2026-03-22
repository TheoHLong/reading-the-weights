#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from reading_weights.analysis.plots import save_eigenspectrum_html
from reading_weights.models.image_classifier import BilinearImageClassifier
from reading_weights.utils import ensure_dir, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze a trained MNIST bilinear checkpoint.')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to a saved checkpoint.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/figures'),
        help='Directory for analysis artifacts.',
    )
    args = parser.parse_args()

    payload = torch.load(args.checkpoint, map_location='cpu')
    config = payload['config']
    model = BilinearImageClassifier(
        d_input=int(config['model']['d_input']),
        d_hidden=int(config['model']['d_hidden']),
        d_output=int(config['model']['d_output']),
        n_layer=int(config['model']['n_layer']),
        bias=bool(config['model']['bias']),
        residual=bool(config['model']['residual']),
        seed=int(config['seed']),
    )
    model.load_state_dict(payload['model_state_dict'])
    model.eval()

    output_dir = ensure_dir(args.output_dir / args.checkpoint.stem)
    eigenvalues, eigenvectors = model.decompose()
    eigenvalues = eigenvalues.cpu()
    eigenvectors = eigenvectors.cpu()

    torch.save({'eigenvalues': eigenvalues, 'eigenvectors': eigenvectors}, output_dir / 'decomposition.pt')

    top_class = int(eigenvalues[:, -1].argmax().item())
    save_eigenspectrum_html(
        eigenvalues[top_class].tolist(),
        output_dir / 'top_class_eigenspectrum.html',
        title=f'Top-class eigenspectrum (class={top_class})',
    )
    write_json(
        output_dir / 'summary.json',
        {
            'checkpoint': str(args.checkpoint),
            'epoch': int(payload['epoch']),
            'top_class': top_class,
            'top_eigenvalue': float(eigenvalues[top_class, -1].item()),
        },
    )
    print(f'Analysis saved to {output_dir}')


if __name__ == '__main__':
    main()
