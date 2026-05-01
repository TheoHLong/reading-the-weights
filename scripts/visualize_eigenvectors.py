#!/usr/bin/env python
"""Visualize top eigenvectors per class from a decomposition.pt artifact.

Mirrors the SQS paper's Figure 2A (one row per class, top-K positive
eigenvectors) and the appendix figures (positive + negative eigenvectors
per class). Used to sanity-check that a trained model's interaction-matrix
eigenstructure has learned meaningful, image-shaped feature directions
before drawing conclusions from the cosine-similarity comparison.

Each eigenvector is reshaped to (channels, H, W) and shown with a diverging
colormap (RdBu_r) centered at zero so positive/negative weights are
visible. Eigenvalues are printed as titles for reference.

Examples
--------
MNIST/FMNIST (28x28 grayscale), top-3 positive + top-3 negative per class:
    python scripts/visualize_eigenvectors.py \\
        --decomposition results/analysis/mnist_paper_sqs_<TS>/decomposition.pt \\
        --image-size 28 --channels 1 \\
        --top-k 3 --bottom-k 3 \\
        --output results/figures/mnist_sqs_eigenvectors.png

CIFAR-10 (32x32 RGB), top-3 positive only:
    python scripts/visualize_eigenvectors.py \\
        --decomposition results/analysis/cifar10_baseline_n1_sqs_<TS>/decomposition.pt \\
        --image-size 32 --channels 3 \\
        --top-k 3 --bottom-k 0 \\
        --output results/figures/cifar_sqs_eigenvectors.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from _bootstrap import ensure_project_on_path

ensure_project_on_path()

from src.utils import ensure_dir, load_checkpoint


CIFAR10_LABELS = ['airplane', 'auto', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
FMNIST_LABELS = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                 'sandal', 'shirt', 'sneaker', 'bag', 'boot']


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Visualize top eigenvectors per class.')
    parser.add_argument('--decomposition', type=Path, required=True,
                        help='Path to a decomposition.pt produced by analyze_checkpoint.py.')
    parser.add_argument('--image-size', type=int, required=True,
                        help='Spatial size H=W of the input image (e.g., 28 or 32).')
    parser.add_argument('--channels', type=int, default=1,
                        help='Number of input channels (1 grayscale, 3 RGB). Default 1.')
    parser.add_argument('--top-k', type=int, default=3,
                        help='Number of largest-positive eigenvectors per class.')
    parser.add_argument('--bottom-k', type=int, default=0,
                        help='Number of largest-negative eigenvectors per class.')
    parser.add_argument('--classes', type=int, nargs='+', default=None,
                        help='Subset of class indices to plot (default: all classes).')
    parser.add_argument('--label-set', choices=['index', 'cifar10', 'fmnist', 'mnist'],
                        default='index',
                        help='How to label rows. Default uses raw class indices.')
    parser.add_argument('--output', type=Path, required=True,
                        help='Output image path (PNG/PDF).')
    parser.add_argument('--cell-size', type=float, default=1.4,
                        help='Inches per subplot cell.')
    return parser.parse_args()


def reshape_to_image(vec: Tensor, channels: int, size: int) -> np.ndarray:
    """Reshape (input_dim,) vector to (H, W) for grayscale, (H, W, 3) for RGB."""
    arr = vec.detach().cpu().numpy()
    if channels == 1:
        return arr.reshape(size, size)
    if channels == 3:
        # Stored as flat with channel-major (CHW) layout assumed; matches torchvision.ToTensor.
        return arr.reshape(channels, size, size).transpose(1, 2, 0)
    raise ValueError(f'Unsupported channels={channels}; only 1 or 3 are supported.')


def plot_grayscale(ax: plt.Axes, vec: Tensor, channels: int, size: int) -> None:
    img = reshape_to_image(vec, channels, size)
    abs_max = float(np.abs(img).max()) + 1e-12
    if channels == 1:
        ax.imshow(img, cmap='RdBu_r', vmin=-abs_max, vmax=abs_max, interpolation='nearest')
    else:
        # For RGB, normalize each channel symmetrically around 0 then shift to [0, 1]
        rgb = (img / (2 * abs_max)) + 0.5
        ax.imshow(np.clip(rgb, 0.0, 1.0), interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])


def get_label(label_set: str, c: int) -> str:
    if label_set == 'cifar10' and 0 <= c < len(CIFAR10_LABELS):
        return f'{c}: {CIFAR10_LABELS[c]}'
    if label_set == 'fmnist' and 0 <= c < len(FMNIST_LABELS):
        return f'{c}: {FMNIST_LABELS[c]}'
    if label_set == 'mnist':
        return f'digit {c}'
    return f'class {c}'


def main() -> None:
    args = parse_args()

    if args.top_k <= 0 and args.bottom_k <= 0:
        raise ValueError('At least one of --top-k or --bottom-k must be positive.')
    if not args.decomposition.is_file():
        raise FileNotFoundError(
            f'Decomposition path must be a file, got {args.decomposition!s}. '
            'Run scripts/analyze_checkpoint.py first and verify the analysis directory name.'
        )

    payload = load_checkpoint(args.decomposition, map_location='cpu')
    eigvals: Tensor = payload['eigenvalues']           # (cls, eig)
    eigvecs: Tensor = payload['eigenvectors_input']    # (cls, eig, input)

    n_cls, n_eig, n_input = eigvecs.shape
    expected = args.image_size * args.image_size * args.channels
    if n_input != expected:
        raise ValueError(
            f'Input dim {n_input} does not match image_size**2 * channels = '
            f'{args.image_size}**2 * {args.channels} = {expected}.'
        )

    classes = args.classes if args.classes is not None else list(range(n_cls))
    if any(c < 0 or c >= n_cls for c in classes):
        raise ValueError(f'class indices out of range [0, {n_cls}). Got {classes}.')

    # Sort orderings per class
    pos_order = eigvals.argsort(dim=-1, descending=True)   # large positive first
    neg_order = eigvals.argsort(dim=-1, descending=False)  # large negative first

    n_rows = len(classes)
    n_cols = args.top_k + args.bottom_k

    fig_w = max(args.cell_size * n_cols + 1.4, 4.0)
    fig_h = max(args.cell_size * n_rows + 0.5, 2.5)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    for row, c in enumerate(classes):
        # Positive (largest λ first)
        for k in range(args.top_k):
            idx = int(pos_order[c, k].item())
            ax = axes[row, k]
            plot_grayscale(ax, eigvecs[c, idx], args.channels, args.image_size)
            lam = float(eigvals[c, idx].item())
            ax.set_title(f'+{k+1}\nλ={lam:+.3f}', fontsize=7)
        # Negative (largest |λ| first, but with negative sign)
        for k in range(args.bottom_k):
            idx = int(neg_order[c, k].item())
            ax = axes[row, args.top_k + k]
            plot_grayscale(ax, eigvecs[c, idx], args.channels, args.image_size)
            lam = float(eigvals[c, idx].item())
            ax.set_title(f'-{k+1}\nλ={lam:+.3f}', fontsize=7)

        axes[row, 0].set_ylabel(get_label(args.label_set, c), fontsize=9, rotation=0,
                                labelpad=42, ha='right', va='center')

    fig.suptitle(args.decomposition.parent.name, fontsize=10)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    output_path = args.output
    ensure_dir(output_path.parent)
    fig.savefig(output_path, bbox_inches='tight')
    plt.close(fig)

    print(f'Saved {n_rows} classes × {n_cols} eigenvectors to {output_path}')
    if args.bottom_k > 0:
        print(f'  Layout: cols 1..{args.top_k} = top positive λ,  '
              f'cols {args.top_k+1}..{n_cols} = top negative λ')
    print(f'  Eigenvalue range: [{float(eigvals.min()):+.3f}, {float(eigvals.max()):+.3f}]')


if __name__ == '__main__':
    main()
