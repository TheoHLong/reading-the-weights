#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('report_assets/.mplconfig')))

import matplotlib.pyplot as plt
import torch

from _bootstrap import ensure_src_on_path
from visualize_eigenvectors import CLASS_NAMES, load_decomposition, orient_vector, reshape_vector, select_component_indices

ensure_src_on_path()


def parse_indices(raw: str) -> list[int]:
    return [int(value.strip()) for value in raw.split(',') if value.strip()]


def chunk(items: list[tuple[str, torch.Tensor]], size: int) -> list[list[tuple[str, torch.Tensor]]]:
    return [items[start:start + size] for start in range(0, len(items), size)]


def selected_images(
    *,
    decomposition_path: Path,
    dataset: str,
    class_indices: list[int],
    image_size: int,
    channels: int,
    component_mode: str,
) -> list[tuple[str, torch.Tensor]]:
    payload = load_decomposition(decomposition_path)
    eigenvalues = payload['eigenvalues']
    eigenvectors_input = payload['eigenvectors_input']
    component_indices = select_component_indices(eigenvalues, component_mode)
    class_names = CLASS_NAMES[dataset]

    images = []
    for class_idx in class_indices:
        component_idx = int(component_indices[class_idx].item())
        vector = orient_vector(eigenvectors_input[class_idx, component_idx])
        images.append((class_names[class_idx], reshape_vector(vector, channels, image_size)))
    return images


def plot_comparison(
    *,
    rows: list[list[tuple[str, torch.Tensor]]],
    output_path: Path,
    channels: int,
    quantile: float,
    columns: int,
) -> None:
    row_count = len(rows)
    col_count = columns
    fig, axes = plt.subplots(row_count, col_count, figsize=(col_count * 1.85, row_count * 1.85))
    if row_count == 1:
        axes = [axes]

    for row_idx, row in enumerate(rows):
        for col_idx in range(col_count):
            ax = axes[row_idx][col_idx]
            ax.axis('off')
            if col_idx >= len(row):
                continue
            label, image = row[col_idx]
            vmax = float(torch.quantile(image.abs().flatten(), quantile).item())
            if channels == 1:
                ax.imshow(image, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='bilinear')
            else:
                ax.imshow(image, cmap='RdBu_r', vmin=-vmax, vmax=vmax, interpolation='bilinear')
            ax.set_title(label, fontsize=10, fontweight='bold', color='#1c3557', pad=6)

    fig.tight_layout(h_pad=0.9, w_pad=0.45)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build compact Task C report eigenvector panels.')
    parser.add_argument('--mnist-decomposition', type=Path, required=True)
    parser.add_argument('--fmnist-decomposition', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('report_assets/task_c/task_c_eigenvector_panel.png'))
    parser.add_argument('--mnist-classes', type=str, default='0,1,2,3,4,5,6,7,8,9')
    parser.add_argument('--fmnist-classes', type=str, default='0,1,2,3,4,5,6,7,8,9')
    parser.add_argument('--columns', type=int, default=5, help='Number of panels per row.')
    parser.add_argument('--component-mode', choices=['largest', 'abs'], default='largest')
    parser.add_argument('--quantile', type=float, default=0.995)
    args = parser.parse_args()

    mnist_images = selected_images(
        decomposition_path=args.mnist_decomposition,
        dataset='mnist',
        class_indices=parse_indices(args.mnist_classes),
        image_size=28,
        channels=1,
        component_mode=args.component_mode,
    )
    fmnist_images = selected_images(
        decomposition_path=args.fmnist_decomposition,
        dataset='fashion_mnist',
        class_indices=parse_indices(args.fmnist_classes),
        image_size=28,
        channels=1,
        component_mode=args.component_mode,
    )
    rows = chunk(mnist_images, int(args.columns)) + chunk(fmnist_images, int(args.columns))
    plot_comparison(
        rows=rows,
        output_path=args.output,
        channels=1,
        quantile=float(args.quantile),
        columns=int(args.columns),
    )
    print(f'Task C report panel saved to {args.output}')


if __name__ == '__main__':
    main()
