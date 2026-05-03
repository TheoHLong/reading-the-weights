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
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('report_assets/.mplconfig')))

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
CLASS_NAMES = {
    'mnist': [str(idx) for idx in range(10)],
    'fashion_mnist': [
        't-shirt',
        'trouser',
        'pullover',
        'dress',
        'coat',
        'sandal',
        'shirt',
        'sneaker',
        'bag',
        'ankle boot',
    ],
    'fmnist': FMNIST_LABELS,
    'cifar10': CIFAR10_LABELS,
}


def load_decomposition(path: Path) -> dict[str, Tensor]:
    payload = load_checkpoint(path, map_location='cpu')
    required = {'eigenvalues', 'eigenvectors_input'}
    missing = required - set(payload)
    if missing:
        raise KeyError(f'Decomposition file is missing keys: {sorted(missing)}')
    return payload


def select_component_indices(eigenvalues: Tensor, mode: str) -> Tensor:
    if mode == 'largest':
        return torch.argmax(eigenvalues, dim=-1)
    if mode == 'abs':
        return torch.argmax(eigenvalues.abs(), dim=-1)
    raise ValueError(f'Unsupported component selection mode: {mode}')


def orient_vector(vector: Tensor) -> Tensor:
    max_idx = torch.argmax(vector.abs())
    if vector.flatten()[max_idx] < 0:
        return -vector
    return vector


def reshape_vector(vector: Tensor, channels: int, image_size: int) -> Tensor:
    image = vector.reshape(channels, image_size, image_size)
    if channels == 1:
        return image.squeeze(0)
    return image.permute(1, 2, 0)


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
    parser.add_argument('--dataset', choices=sorted(CLASS_NAMES), default=None,
                        help='Panel-mode dataset label set for one eigenvector per class.')
    parser.add_argument('--class-indices', type=str, default='',
                        help='Panel mode: comma-separated class indices to plot.')
    parser.add_argument('--number-panels', action='store_true',
                        help='Panel mode: number panels and place class names below images.')
    parser.add_argument('--component-mode', choices=['largest', 'abs'], default='largest',
                        help='Panel mode: choose top positive or largest-absolute component.')
    parser.add_argument('--scale', choices=['global', 'per-image'], default='per-image',
                        help='Panel mode: color scaling strategy.')
    parser.add_argument('--show-eigenvalue', action='store_true',
                        help='Panel mode: include selected eigenvalue in each title.')
    parser.add_argument('--title', type=str, default='',
                        help='Panel mode: optional figure title.')
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


def _parse_panel_indices(raw: str, n_cls: int) -> list[int]:
    if not raw:
        return list(range(n_cls))
    return [int(part.strip()) for part in raw.split(',') if part.strip()]


def _imshow_signed(ax: plt.Axes, image: Tensor, channels: int, vmax: float) -> None:
    if channels == 1:
        ax.imshow(image, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='bilinear')
        return
    arr = image.detach().cpu().numpy()
    rgb = np.clip((arr / (2 * vmax)) + 0.5, 0.0, 1.0)
    ax.imshow(rgb, interpolation='bilinear')


def plot_one_eigenvector_per_class(args: argparse.Namespace) -> None:
    payload = load_decomposition(args.decomposition)
    eigvals: Tensor = payload['eigenvalues']
    eigvecs: Tensor = payload['eigenvectors_input']
    n_cls = int(eigvecs.shape[0])
    expected = args.image_size * args.image_size * args.channels
    if int(eigvecs.shape[-1]) != expected:
        raise ValueError(
            f'Input dim {eigvecs.shape[-1]} does not match image_size**2 * channels = '
            f'{args.image_size}**2 * {args.channels} = {expected}.'
        )

    dataset = args.dataset
    if dataset is None:
        dataset = 'fashion_mnist' if args.label_set == 'fmnist' else args.label_set
    class_names = CLASS_NAMES.get(dataset, [str(idx) for idx in range(n_cls)])
    if len(class_names) != n_cls:
        class_names = [str(idx) for idx in range(n_cls)]

    class_indices = _parse_panel_indices(args.class_indices, n_cls)
    if any(class_idx < 0 or class_idx >= n_cls for class_idx in class_indices):
        raise ValueError(f'class indices out of range [0, {n_cls}). Got {class_indices}.')

    component_indices = select_component_indices(eigvals, args.component_mode)
    selected_images: list[Tensor] = []
    selected_eigenvalues: list[float] = []
    selected_class_names: list[str] = []
    for class_idx in class_indices:
        component_idx = int(component_indices[class_idx].item())
        vector = orient_vector(eigvecs[class_idx, component_idx])
        selected_images.append(reshape_vector(vector, args.channels, args.image_size))
        selected_eigenvalues.append(float(eigvals[class_idx, component_idx].item()))
        selected_class_names.append(class_names[class_idx])

    if args.scale == 'global':
        all_values = torch.cat([image.flatten() for image in selected_images])
        global_vmax = float(torch.quantile(all_values.abs(), 0.995).item())
    else:
        global_vmax = None

    cols = min(5, max(1, len(selected_images)))
    rows = math.ceil(len(selected_images) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.45), squeeze=False)
    for panel_idx, ax in enumerate(axes.flatten()):
        ax.axis('off')
        if panel_idx >= len(selected_images):
            continue
        image = selected_images[panel_idx]
        vmax = global_vmax
        if vmax is None:
            vmax = float(torch.quantile(image.abs().flatten(), 0.995).item())
        vmax = max(vmax, 1e-12)
        _imshow_signed(ax, image, args.channels, vmax)

        title_text = str(panel_idx + 1) if args.number_panels else selected_class_names[panel_idx]
        if args.show_eigenvalue:
            title_text = f'{title_text}\nλ={selected_eigenvalues[panel_idx]:.3g}'
        ax.set_title(title_text, fontsize=10, fontweight='bold', color='#1c3557', pad=8)
        if args.number_panels:
            ax.text(
                0.5,
                -0.10,
                selected_class_names[panel_idx].lower(),
                ha='center',
                va='top',
                transform=ax.transAxes,
                fontsize=10,
                fontweight='bold',
                color='#1c3557',
            )

    if args.title:
        fig.suptitle(args.title, fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=1.8, w_pad=0.8)
    else:
        fig.tight_layout(h_pad=1.8, w_pad=0.8)
    if args.number_panels:
        fig.subplots_adjust(top=0.86, bottom=0.20)

    ensure_dir(args.output.parent)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)
    print(f'Eigenvector figure saved to {args.output}')


def main() -> None:
    args = parse_args()

    panel_mode = bool(args.dataset or args.class_indices or args.number_panels or args.show_eigenvalue or args.title)
    if panel_mode:
        plot_one_eigenvector_per_class(args)
        return

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
