#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('report_assets/.mplconfig')))

import matplotlib.pyplot as plt
import torch

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from reading_weights.utils import ensure_dir


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
    'cifar10': [
        'airplane',
        'automobile',
        'bird',
        'cat',
        'deer',
        'dog',
        'frog',
        'horse',
        'ship',
        'truck',
    ],
}


def load_decomposition(path: Path) -> dict[str, torch.Tensor]:
    try:
        payload = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location='cpu')
    required = {'eigenvalues', 'eigenvectors_input'}
    missing = required - set(payload)
    if missing:
        raise KeyError(f'Decomposition file is missing keys: {sorted(missing)}')
    return payload


def select_component_indices(eigenvalues: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == 'largest':
        return torch.argmax(eigenvalues, dim=-1)
    if mode == 'abs':
        return torch.argmax(eigenvalues.abs(), dim=-1)
    raise ValueError(f'Unsupported component selection mode: {mode}')


def orient_vector(vector: torch.Tensor) -> torch.Tensor:
    max_idx = torch.argmax(vector.abs())
    if vector.flatten()[max_idx] < 0:
        return -vector
    return vector


def reshape_vector(vector: torch.Tensor, channels: int, image_size: int) -> torch.Tensor:
    image = vector.reshape(channels, image_size, image_size)
    if channels == 1:
        return image.squeeze(0)
    return image.permute(1, 2, 0)


def plot_top_eigenvectors(
    *,
    decomposition_path: Path,
    output_path: Path,
    dataset: str,
    image_size: int,
    channels: int,
    component_mode: str,
    class_indices: list[int] | None,
    number_panels: bool,
    scale: str,
    show_eigenvalue: bool,
    title: str,
) -> None:
    payload = load_decomposition(decomposition_path)
    eigenvalues = payload['eigenvalues']
    eigenvectors_input = payload['eigenvectors_input']
    component_indices = select_component_indices(eigenvalues, component_mode)

    num_classes = int(eigenvectors_input.shape[0])
    class_names = CLASS_NAMES.get(dataset, [str(idx) for idx in range(num_classes)])
    if len(class_names) != num_classes:
        class_names = [str(idx) for idx in range(num_classes)]
    if class_indices is None:
        class_indices = list(range(num_classes))
    if any(class_idx < 0 or class_idx >= num_classes for class_idx in class_indices):
        raise ValueError(f'Class indices must be in [0, {num_classes - 1}], got {class_indices}')

    selected_images = []
    selected_eigenvalues = []
    selected_class_names = []
    for class_idx in class_indices:
        component_idx = int(component_indices[class_idx].item())
        vector = orient_vector(eigenvectors_input[class_idx, component_idx])
        selected_images.append(reshape_vector(vector, channels, image_size))
        selected_eigenvalues.append(float(eigenvalues[class_idx, component_idx].item()))
        selected_class_names.append(class_names[class_idx])

    if scale == 'global':
        all_values = torch.cat([image.flatten() for image in selected_images])
        global_vmax = float(torch.quantile(all_values.abs(), 0.995).item())
    elif scale == 'per-image':
        global_vmax = None
    else:
        raise ValueError(f'Unsupported scale mode: {scale}')

    cols = 5
    rows = math.ceil(len(class_indices) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.0, rows * 2.45))
    axes_flat = axes.flatten() if hasattr(axes, 'flatten') else [axes]

    for panel_idx, ax in enumerate(axes_flat):
        ax.axis('off')
        if panel_idx >= len(class_indices):
            continue
        image = selected_images[panel_idx]
        if global_vmax is None:
            vmax = float(torch.quantile(image.abs().flatten(), 0.995).item())
        else:
            vmax = global_vmax
        vmin = -vmax
        if channels == 1:
            ax.imshow(image, cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='bilinear')
        else:
            ax.imshow(image, cmap='RdBu_r', vmin=vmin, vmax=vmax, interpolation='bilinear')
        title_text = str(panel_idx + 1) if number_panels else selected_class_names[panel_idx]
        if show_eigenvalue:
            title_text = f'{title_text}\nlambda={selected_eigenvalues[panel_idx]:.3g}'
        ax.set_title(title_text, fontsize=10, fontweight='bold', color='#1c3557', pad=8)
        if number_panels:
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

    if title:
        fig.suptitle(title, fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.92), h_pad=1.8, w_pad=0.8)
    else:
        fig.tight_layout(h_pad=1.8, w_pad=0.8)
    if number_panels:
        fig.subplots_adjust(bottom=0.18)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Visualize top input-space eigenvectors per output class.')
    parser.add_argument('--decomposition', type=Path, required=True, help='Path to decomposition.pt.')
    parser.add_argument('--output', type=Path, required=True, help='Output PNG path.')
    parser.add_argument('--dataset', type=str, required=True, choices=sorted(CLASS_NAMES), help='Dataset label set.')
    parser.add_argument('--image-size', type=int, required=True, help='Input image width/height.')
    parser.add_argument('--channels', type=int, required=True, help='Input image channels.')
    parser.add_argument(
        '--class-indices',
        type=str,
        default='',
        help='Optional comma-separated class indices to plot, in the requested order.',
    )
    parser.add_argument(
        '--number-panels',
        action='store_true',
        help='Put panel numbers above images and class names below images.',
    )
    parser.add_argument(
        '--component-mode',
        choices=['largest', 'abs'],
        default='largest',
        help='Which eigencomponent to show for each class.',
    )
    parser.add_argument(
        '--scale',
        choices=['global', 'per-image'],
        default='per-image',
        help='Color scaling mode for the grid.',
    )
    parser.add_argument(
        '--show-eigenvalue',
        action='store_true',
        help='Include the selected eigenvalue in each subplot title.',
    )
    parser.add_argument('--title', type=str, default='', help='Optional figure title.')
    args = parser.parse_args()

    ensure_dir(args.output.parent)
    class_indices = None
    if args.class_indices:
        class_indices = [int(part.strip()) for part in args.class_indices.split(',') if part.strip()]
    plot_top_eigenvectors(
        decomposition_path=args.decomposition,
        output_path=args.output,
        dataset=args.dataset,
        image_size=args.image_size,
        channels=args.channels,
        component_mode=args.component_mode,
        class_indices=class_indices,
        number_panels=bool(args.number_panels),
        scale=args.scale,
        show_eigenvalue=args.show_eigenvalue,
        title=args.title,
    )
    print(f'Eigenvector figure saved to {args.output}')


if __name__ == '__main__':
    main()
