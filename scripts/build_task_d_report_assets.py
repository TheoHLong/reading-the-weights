#!/usr/bin/env python
from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path('report_assets/.mplconfig')))

import matplotlib.pyplot as plt
import pandas as pd

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from reading_weights.utils import ensure_dir


RUNS = [
    {
        'label': 'MNIST',
        'run_name': 'mnist_pilot_mps_20260412-223552',
        'metrics': 'results/metrics/mnist_pilot_mps_20260412-223552/summary.json',
        'truncation': 'results/truncation/mnist_pilot_mps_20260412-223552/test_truncation.csv',
        'kind': 'baseline',
    },
    {
        'label': 'CIFAR Raw 25e',
        'run_name': 'cifar10_headline_mps_20260412-223650',
        'metrics': 'results/metrics/cifar10_headline_mps_20260412-223650/summary.json',
        'truncation': 'results/truncation/cifar10_headline_mps_20260412-223650/test_truncation.csv',
        'kind': 'baseline',
    },
    {
        'label': 'CIFAR Raw 50e',
        'run_name': 'cifar10_completion_mps_20260412-225712',
        'metrics': 'results/metrics/cifar10_completion_mps_20260412-225712/summary.json',
        'truncation': 'results/truncation/cifar10_completion_mps_20260412-225712/test_truncation.csv',
        'kind': 'baseline',
    },
    {
        'label': 'CIFAR Local 2x2',
        'run_name': 'cifar10_locality_mps_20260412-230328',
        'metrics': 'results/metrics/cifar10_locality_mps_20260412-230328/summary.json',
        'truncation': 'results/truncation/cifar10_locality_mps_20260412-230328/test_truncation.csv',
        'kind': 'redesign',
    },
    {
        'label': 'CIFAR Local 4x4 s42',
        'run_name': 'cifar10_locality4_mps_20260412-231256',
        'metrics': 'results/metrics/cifar10_locality4_mps_20260412-231256/summary.json',
        'truncation': 'results/truncation/cifar10_locality4_mps_20260412-231256/test_truncation.csv',
        'kind': 'redesign',
    },
    {
        'label': 'CIFAR Local 4x4 s123',
        'run_name': 'cifar10_locality4_seed123_mps_20260412-231948',
        'metrics': 'results/metrics/cifar10_locality4_seed123_mps_20260412-231948/summary.json',
        'truncation': 'results/truncation/cifar10_locality4_seed123_mps_20260412-231948/test_truncation.csv',
        'kind': 'redesign',
    },
]


def build_summary_table(output_dir: Path) -> pd.DataFrame:
    rows = []
    for run in RUNS:
        metrics = pd.read_json(run['metrics'], typ='series')
        truncation = pd.read_csv(run['truncation'])
        lookup = {int(row.rank): float(row.accuracy) for row in truncation.itertuples(index=False)}
        rows.append(
            {
                'label': run['label'],
                'run_name': run['run_name'],
                'kind': run['kind'],
                'best_val_acc': float(metrics['best_val_acc']),
                'test_acc': float(metrics['test_acc_at_best_val']),
                'rank_16_acc': lookup.get(16),
                'rank_32_acc': lookup.get(32),
                'rank_64_acc': lookup.get(64),
                'full_rank_acc': float(truncation.iloc[-1]['accuracy']),
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / 'task_d_summary_table.csv', index=False)
    markdown_lines = [
        '| ' + ' | '.join(table.columns) + ' |',
        '| ' + ' | '.join(['---'] * len(table.columns)) + ' |',
    ]
    for row in table.itertuples(index=False):
        markdown_lines.append('| ' + ' | '.join(str(value) for value in row) + ' |')
    (output_dir / 'task_d_summary_table.md').write_text('\n'.join(markdown_lines) + '\n', encoding='utf-8')
    return table


def plot_truncation_curves(output_dir: Path) -> None:
    plt.figure(figsize=(8, 5))
    for run in RUNS:
        truncation = pd.read_csv(run['truncation'])
        plt.plot(truncation['rank'], truncation['accuracy'], marker='o', label=run['label'])
    plt.xscale('log', base=2)
    plt.xlabel('Truncation Rank')
    plt.ylabel('Test Accuracy')
    plt.title('Task D Truncation vs Accuracy')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'task_d_truncation_curves.png', dpi=200)
    plt.close()


def plot_cifar_comparison(output_dir: Path) -> None:
    selected = [run for run in RUNS if run['label'].startswith('CIFAR')]
    plt.figure(figsize=(8, 5))
    for run in selected:
        truncation = pd.read_csv(run['truncation'])
        plt.plot(truncation['rank'], truncation['accuracy'], marker='o', label=run['label'])
    plt.xscale('log', base=2)
    plt.xlabel('Truncation Rank')
    plt.ylabel('Test Accuracy')
    plt.title('CIFAR Raw vs Locality-Preserving Variants')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'task_d_cifar_truncation_curves.png', dpi=200)
    plt.close()


def write_narrative(summary: pd.DataFrame, output_dir: Path) -> None:
    raw_50 = summary.loc[summary['label'] == 'CIFAR Raw 50e'].iloc[0]
    local_42 = summary.loc[summary['label'] == 'CIFAR Local 4x4 s42'].iloc[0]
    local_123 = summary.loc[summary['label'] == 'CIFAR Local 4x4 s123'].iloc[0]

    narrative = f"""# Task D Narrative

## Main finding

Task D now supports a clean three-stage story:

1. MNIST is strongly low-rank under the bilinear decomposition framework.
2. Raw-pixel CIFAR-10 trains, but remains broad-spectrum even after longer training.
3. Adding locality-preserving preprocessing makes CIFAR-10 substantially more compressible under the same decomposition analysis.

## Raw-pixel completion

- Best raw-pixel completion run: `{raw_50['run_name']}`
- Best validation accuracy: `{raw_50['best_val_acc']:.4f}`
- Test accuracy: `{raw_50['test_acc']:.4f}`
- Rank-64 accuracy: `{raw_50['rank_64_acc']:.4f}`
- Full-rank accuracy: `{raw_50['full_rank_acc']:.4f}`

Interpretation: the raw-pixel CIFAR extension is complete, but the resulting spectrum remains broad.

## Locality-preserving redesign

- Seed 42 run: `{local_42['run_name']}`, test accuracy `{local_42['test_acc']:.4f}`, rank-64 accuracy `{local_42['rank_64_acc']:.4f}`
- Seed 123 run: `{local_123['run_name']}`, test accuracy `{local_123['test_acc']:.4f}`, rank-64 accuracy `{local_123['rank_64_acc']:.4f}`

Interpretation: the `4x4` locality-preserving redesign is stable across two seeds and reaches essentially full-rank performance by rank 64.

## Report recommendation

Frame Extension 1 as a negative-to-positive result:

- Raw-pixel CIFAR-10 shows that the MNIST result does not directly transfer.
- Locality-preserving CIFAR-10 shows that the weight-based analysis becomes much more compatible with natural images once basic spatial structure is restored.
"""
    (output_dir / 'task_d_narrative.md').write_text(narrative, encoding='utf-8')


def main() -> None:
    output_dir = ensure_dir(Path('report_assets/task_d'))
    ensure_dir(Path('report_assets/.mplconfig'))
    summary = build_summary_table(output_dir)
    plot_truncation_curves(output_dir)
    plot_cifar_comparison(output_dir)
    write_narrative(summary, output_dir)
    print(f'Task D report assets saved to {output_dir}')


if __name__ == '__main__':
    main()
