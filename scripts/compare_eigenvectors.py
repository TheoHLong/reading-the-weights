#!/usr/bin/env python
"""Reproduce Figure 2B from Abohwo & Mosen (2025).

For each class, sort eigenvectors by |eigenvalue| descending (so rank-r is the
r-th most important feature direction). Compute |cos(v_r^bilinear, v_r^sqs)|
in input space, then average across classes and across (bilinear, SQS)
decomposition pairs.

Inputs are decomposition.pt artifacts produced by scripts/analyze_checkpoint.py
applied to a pure-bilinear baseline (gate=None) and an SQS baseline
(gate='sqs'). Both must share d_hidden and num_classes.

Example
-------
    python scripts/compare_eigenvectors.py \\
        --bilinear-decomps results/analysis/cifar10_baseline/decomposition.pt \\
        --sqs-decomps      results/analysis/cifar10_baseline_n1_sqs/decomposition.pt \\
        --output-dir       results/figures/figure_2b
"""
from __future__ import annotations

import argparse
import csv
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor

from _bootstrap import ensure_project_on_path

ensure_project_on_path()

from src.utils import ensure_dir, load_checkpoint, write_json


REQUIRED_KEYS = {'eigenvalues', 'eigenvectors_input'}


def load_decomposition(path: Path) -> dict[str, Tensor]:
    if not path.is_file():
        raise FileNotFoundError(
            f'Decomposition path must be a file, got {path!s}. '
            'Run scripts/analyze_checkpoint.py first and verify the analysis directory name.'
        )
    payload = load_checkpoint(path, map_location='cpu')
    missing = REQUIRED_KEYS - set(payload.keys())
    if missing:
        raise ValueError(
            f'{path} is missing required keys {sorted(missing)}. '
            'Run scripts/analyze_checkpoint.py to regenerate.'
        )
    return payload


def rank_sorted_eigenvectors(eigenvalues: Tensor, eigenvectors_input: Tensor) -> Tensor:
    """Sort eigenvectors per class by |eigenvalue| descending.

    Args:
        eigenvalues: (cls, eig)
        eigenvectors_input: (cls, eig, input)
    Returns:
        (cls, eig, input) sorted within each class by |eigenvalue| descending.
    """
    order = eigenvalues.abs().argsort(dim=-1, descending=True)
    cls, eig, inp = eigenvectors_input.shape
    index = order.unsqueeze(-1).expand(cls, eig, inp)
    return torch.gather(eigenvectors_input, dim=1, index=index)


def per_rank_abs_cosine(a: Tensor, b: Tensor, eps: float = 1e-12) -> Tensor:
    """Absolute cosine similarity per (class, rank).

    Eigenvectors are sign-ambiguous, so we take |cos|.

    Args:
        a, b: (cls, eig, input)
    Returns:
        (cls, eig)
    """
    a_n = a / (a.norm(dim=-1, keepdim=True) + eps)
    b_n = b / (b.norm(dim=-1, keepdim=True) + eps)
    return (a_n * b_n).sum(dim=-1).abs()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Reproduce Figure 2B (cosine similarity per eigenvector rank) '
                    'from the SQS paper.',
    )
    parser.add_argument(
        '--bilinear-decomps',
        type=Path,
        nargs='+',
        required=True,
        help='One or more pure-bilinear decomposition.pt files (different seeds).',
    )
    parser.add_argument(
        '--sqs-decomps',
        type=Path,
        nargs='+',
        required=True,
        help='One or more SQS decomposition.pt files (different seeds).',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/figures/figure_2b'),
        help='Where to save plot, CSV, and JSON summary.',
    )
    parser.add_argument(
        '--pair-mode',
        choices=['by-index', 'all-pairs'],
        default='by-index',
        help='by-index pairs bilinear[i] with sqs[i] (matched seeds, recommended). '
             'all-pairs averages over the Cartesian product.',
    )
    parser.add_argument(
        '--y-margin',
        type=float,
        default=0.05,
        help='Padding above and below the data for the plot y-axis.',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    bilinear_decomps = [load_decomposition(p) for p in args.bilinear_decomps]
    sqs_decomps = [load_decomposition(p) for p in args.sqs_decomps]

    if args.pair_mode == 'by-index':
        if len(bilinear_decomps) != len(sqs_decomps):
            raise ValueError(
                'pair-mode=by-index requires equal counts of bilinear and SQS decompositions, '
                f'got {len(bilinear_decomps)} vs {len(sqs_decomps)}.'
            )
        pairs = list(zip(bilinear_decomps, sqs_decomps, strict=True))
    else:
        pairs = list(itertools.product(bilinear_decomps, sqs_decomps))

    per_pair_cos: list[Tensor] = []
    for bil, sqs in pairs:
        bil_sorted = rank_sorted_eigenvectors(bil['eigenvalues'], bil['eigenvectors_input'])
        sqs_sorted = rank_sorted_eigenvectors(sqs['eigenvalues'], sqs['eigenvectors_input'])
        if bil_sorted.shape != sqs_sorted.shape:
            raise ValueError(
                f'Shape mismatch: bilinear {tuple(bil_sorted.shape)} vs SQS {tuple(sqs_sorted.shape)}. '
                'Both checkpoints must share num_classes and d_hidden.'
            )
        per_pair_cos.append(per_rank_abs_cosine(bil_sorted, sqs_sorted))

    stacked = torch.stack(per_pair_cos)  # (n_pairs, cls, eig)
    mean_over_pairs_classes = stacked.mean(dim=(0, 1))  # (eig,)
    std_over_pairs_classes = stacked.std(dim=(0, 1))  # (eig,)

    n_pairs, n_cls, n_eig = stacked.shape
    ranks = np.arange(1, n_eig + 1, dtype=int)

    output_dir = ensure_dir(args.output_dir)

    # ---- raw tensor (full breakdown) ----
    torch.save(
        {
            'cosine_per_pair_class_rank': stacked,
            'pair_mode': args.pair_mode,
            'bilinear_decomps': [str(p) for p in args.bilinear_decomps],
            'sqs_decomps': [str(p) for p in args.sqs_decomps],
        },
        output_dir / 'cosine_breakdown.pt',
    )

    # ---- CSV (rank, mean, std, per-class means) ----
    per_class_mean = stacked.mean(dim=0)  # (cls, eig)
    csv_path = output_dir / 'cosine_similarity_per_rank.csv'
    with csv_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        header = ['rank', 'mean', 'std'] + [f'class_{c}' for c in range(n_cls)]
        writer.writerow(header)
        for r in range(n_eig):
            row = [
                int(ranks[r]),
                float(mean_over_pairs_classes[r].item()),
                float(std_over_pairs_classes[r].item()),
            ]
            row.extend(float(per_class_mean[c, r].item()) for c in range(n_cls))
            writer.writerow(row)

    # ---- JSON summary ----
    summary = {
        'bilinear_decomps': [str(p) for p in args.bilinear_decomps],
        'sqs_decomps': [str(p) for p in args.sqs_decomps],
        'pair_mode': args.pair_mode,
        'n_pairs': n_pairs,
        'n_classes': n_cls,
        'n_eigenvectors': n_eig,
        'rank_1_mean': float(mean_over_pairs_classes[0].item()),
        'top_5_avg': float(mean_over_pairs_classes[:5].mean().item()),
        'top_10_avg': float(mean_over_pairs_classes[:10].mean().item()),
        'overall_mean': float(mean_over_pairs_classes.mean().item()),
        'overall_min': float(mean_over_pairs_classes.min().item()),
        'overall_max': float(mean_over_pairs_classes.max().item()),
        'paper_reference': {
            'top_eigenvectors_target': 0.95,
            'overall_floor_target': 0.50,
        },
    }
    write_json(output_dir / 'summary.json', summary)

    # ---- Figure 2B ----
    fig, ax = plt.subplots(figsize=(7.0, 4.5), dpi=140)
    mean_np = mean_over_pairs_classes.numpy()
    std_np = std_over_pairs_classes.numpy()
    ax.plot(ranks, mean_np, color='tab:blue', linewidth=1.0,
            label=f'mean over {n_pairs} pair(s) × {n_cls} classes')
    ax.fill_between(
        ranks,
        np.clip(mean_np - std_np, 0.0, 1.0),
        np.clip(mean_np + std_np, 0.0, 1.0),
        color='tab:blue', alpha=0.15, label='±1 std',
    )
    y_min = max(0.0, float(mean_np.min()) - args.y_margin)
    y_max = min(1.0, float(mean_np.max()) + args.y_margin)
    ax.set_ylim(y_min, y_max)
    ax.set_xlim(1, n_eig)
    ax.set_xlabel('Rank')
    ax.set_ylabel('Cosine similarity')
    ax.set_title('Cosine similarity between eigenvectors for each rank\n'
                 '(SQS-GLU vs Bilinear MLP)')
    ax.grid(alpha=0.3)
    ax.legend(loc='lower left')
    fig.tight_layout()
    fig.savefig(output_dir / 'figure_2b.png')
    fig.savefig(output_dir / 'figure_2b.pdf')
    plt.close(fig)

    print('Figure 2B reproduction done.')
    print(f'  Pairs           : {n_pairs}')
    print(f'  Classes         : {n_cls}')
    print(f'  Eigenvectors    : {n_eig}')
    print(f'  Rank-1 cosine   : {summary["rank_1_mean"]:.4f}  (paper top: ~0.95)')
    print(f'  Top-5 avg       : {summary["top_5_avg"]:.4f}')
    print(f'  Top-10 avg      : {summary["top_10_avg"]:.4f}')
    print(f'  Overall avg     : {summary["overall_mean"]:.4f}  (paper floor: ~0.50)')
    print(f'  Output dir      : {output_dir}')


if __name__ == '__main__':
    main()
