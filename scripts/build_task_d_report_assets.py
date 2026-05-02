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
    {
        'label': 'CIFAR Local 4x4 50e',
        'run_name': 'cifar10_locality4_50e_mps_20260427-224921',
        'metrics': 'results/metrics/cifar10_locality4_50e_mps_20260427-224921/summary.json',
        'truncation': 'results/truncation/cifar10_locality4_50e_mps_20260427-224921/test_truncation.csv',
        'kind': 'redesign',
    },
    {
        'label': 'CIFAR Local 4x4 50e wd=.01',
        'run_name': 'cifar10_locality4_wd001_50e_mps_20260427-225245',
        'metrics': 'results/metrics/cifar10_locality4_wd001_50e_mps_20260427-225245/summary.json',
        'truncation': 'results/truncation/cifar10_locality4_wd001_50e_mps_20260427-225245/test_truncation.csv',
        'kind': 'redesign',
    },
    {
        'label': 'CIFAR Local 2x2 50e',
        'run_name': 'cifar10_locality2_50e_mps_20260427-225510',
        'metrics': 'results/metrics/cifar10_locality2_50e_mps_20260427-225510/summary.json',
        'truncation': 'results/truncation/cifar10_locality2_50e_mps_20260427-225510/test_truncation.csv',
        'kind': 'redesign',
    },
    {
        'label': 'CIFAR Raw 25e + L1',
        'run_name': 'cifar10_l1tiny_mps_20260422-203929',
        'metrics': 'results/metrics/cifar10_l1tiny_mps_20260422-203929/summary.json',
        'truncation': 'results/truncation/cifar10_l1tiny_mps_20260422-203929/test_truncation.csv',
        'kind': 'control',
    },
    {
        'label': 'CIFAR Local 8x8',
        'run_name': 'cifar10_locality8_mps_20260422-204446',
        'metrics': 'results/metrics/cifar10_locality8_mps_20260422-204446/summary.json',
        'truncation': 'results/truncation/cifar10_locality8_mps_20260422-204446/test_truncation.csv',
        'kind': 'redesign',
    },
    {
        'label': 'CIFAR Local 16x16 s42',
        'run_name': 'cifar10_locality16_mps_20260422-204115',
        'metrics': 'results/metrics/cifar10_locality16_mps_20260422-204115/summary.json',
        'truncation': 'results/truncation/cifar10_locality16_mps_20260422-204115/test_truncation.csv',
        'kind': 'redesign',
    },
    {
        'label': 'CIFAR Local 16x16 s123',
        'run_name': 'cifar10_locality16_seed123_mps_20260422-204255',
        'metrics': 'results/metrics/cifar10_locality16_seed123_mps_20260422-204255/summary.json',
        'truncation': 'results/truncation/cifar10_locality16_seed123_mps_20260422-204255/test_truncation.csv',
        'kind': 'redesign',
    },
]

PRIMARY_FIGURE_LABELS = {
    'MNIST': 'MNIST baseline',
    'CIFAR Raw 50e': 'CIFAR raw pixels',
    'CIFAR Local 4x4 50e wd=.01': 'CIFAR 4x4 pooled',
}

CIFAR_FIGURE_LABELS = {
    'CIFAR Raw 50e': 'Raw pixels',
    'CIFAR Raw 25e + L1': 'Raw + tiny L1',
    'CIFAR Local 2x2 50e': '2x2 pooling',
    'CIFAR Local 4x4 50e wd=.01': '4x4 pooling',
    'CIFAR Local 8x8': '8x8 pooling',
    'CIFAR Local 16x16 s42': '16x16 pooling',
}


def build_summary_table(output_dir: Path) -> pd.DataFrame:
    rows = []
    for run in RUNS:
        metrics = pd.read_json(run['metrics'], typ='series')
        history = pd.read_csv(Path(run['metrics']).with_name('metrics.csv'))
        truncation = pd.read_csv(run['truncation'])
        truncation_summary = pd.read_json(
            Path(run['truncation']).with_name('test_summary.json'),
            typ='series',
        )
        rank_efficiency = truncation_summary['rank_efficiency']
        lookup = {int(row.rank): float(row.accuracy) for row in truncation.itertuples(index=False)}
        rows.append(
            {
                'label': run['label'],
                'run_name': run['run_name'],
                'kind': run['kind'],
                'best_val_acc': float(metrics['best_val_acc']),
                'best_val_loss': float(history['val_loss'].min()),
                'test_acc': float(metrics['test_acc_at_best_val']),
                'rank_16_acc': lookup.get(16),
                'rank_32_acc': lookup.get(32),
                'rank_64_acc': lookup.get(64),
                'full_rank_acc': float(truncation.iloc[-1]['accuracy']),
                'rank_for_90pct_full': rank_efficiency['rank_for_90pct_full'],
                'rank_for_95pct_full': rank_efficiency['rank_for_95pct_full'],
                'rank_for_99pct_full': rank_efficiency['rank_for_99pct_full'],
            }
        )

    table = pd.DataFrame(rows)
    table.to_csv(output_dir / 'task_d_summary_table.csv', index=False)

    report_columns = {
        'label': 'Run',
        'kind': 'Kind',
        'best_val_acc': 'Best Val Acc',
        'best_val_loss': 'Best Val Loss',
        'test_acc': 'Test Acc',
        'rank_32_acc': 'Rank-32 Acc',
        'rank_64_acc': 'Rank-64 Acc',
        'rank_for_90pct_full': 'Rank @ 90%',
        'rank_for_95pct_full': 'Rank @ 95%',
        'rank_for_99pct_full': 'Rank @ 99%',
    }
    report_table = table[list(report_columns)].rename(columns=report_columns)
    for column in ['Best Val Acc', 'Test Acc', 'Rank-32 Acc', 'Rank-64 Acc']:
        report_table[column] = report_table[column].map(lambda value: f'{100.0 * value:.2f}%')
    report_table['Best Val Loss'] = report_table['Best Val Loss'].map(lambda value: f'{value:.4f}')

    markdown_lines = [
        '| ' + ' | '.join(report_table.columns) + ' |',
        '| ' + ' | '.join(['---'] * len(report_table.columns)) + ' |',
    ]
    for row in report_table.itertuples(index=False):
        markdown_lines.append('| ' + ' | '.join(str(value) for value in row) + ' |')
    (output_dir / 'task_d_summary_table.md').write_text('\n'.join(markdown_lines) + '\n', encoding='utf-8')
    return table


def plot_truncation_curves(output_dir: Path) -> None:
    selected = [run for run in RUNS if run['label'] in PRIMARY_FIGURE_LABELS]
    plt.figure(figsize=(7.2, 4.8))
    for run in selected:
        truncation = pd.read_csv(run['truncation'])
        plt.plot(
            truncation['rank'],
            truncation['accuracy'],
            marker='o',
            linewidth=2.2,
            label=PRIMARY_FIGURE_LABELS[run['label']],
        )
    plt.xscale('log', base=2)
    plt.xlabel('Truncation Rank')
    plt.ylabel('Test Accuracy')
    plt.title('Bilinear Spectrum: MNIST vs CIFAR-10')
    plt.ylim(0.1, 1.02)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'task_d_truncation_curves.png', dpi=200)
    plt.close()


def plot_cifar_comparison(output_dir: Path) -> None:
    selected = [run for run in RUNS if run['label'] in CIFAR_FIGURE_LABELS]
    plt.figure(figsize=(7.6, 4.8))
    for run in selected:
        truncation = pd.read_csv(run['truncation'])
        plt.plot(
            truncation['rank'],
            truncation['accuracy'],
            marker='o',
            linewidth=2.0,
            label=CIFAR_FIGURE_LABELS[run['label']],
        )
    plt.xscale('log', base=2)
    plt.xlabel('Truncation Rank')
    plt.ylabel('Test Accuracy')
    plt.title('CIFAR-10 Locality and Control Runs')
    plt.ylim(0.1, 0.5)
    plt.legend(ncol=2, fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'task_d_cifar_truncation_curves.png', dpi=200)
    plt.close()


def write_narrative(summary: pd.DataFrame, output_dir: Path) -> None:
    raw_50 = summary.loc[summary['label'] == 'CIFAR Raw 50e'].iloc[0]
    local_42 = summary.loc[summary['label'] == 'CIFAR Local 4x4 s42'].iloc[0]
    local_123 = summary.loc[summary['label'] == 'CIFAR Local 4x4 s123'].iloc[0]
    local_best = summary.loc[summary['label'] == 'CIFAR Local 4x4 50e wd=.01'].iloc[0]

    narrative = f"""# Task D Narrative

## Main finding

The CIFAR extension now supports a clean compatibility story:

1. MNIST is strongly low-rank under the bilinear decomposition framework.
2. Raw-pixel CIFAR-10 trains, but remains broad-spectrum even after longer training.
3. Scalar controls such as width, weight decay, and tiny L1 do not fix that broad-spectrum behavior.
4. Adding locality-preserving preprocessing makes CIFAR-10 substantially more compressible under the same decomposition analysis.
5. The benefit is not monotone: `4x4` pooling is a strong regime, while `16x16` pooling overcompresses the task.

## Raw-pixel completion

- Best raw-pixel completion run: `{raw_50['run_name']}`
- Best validation accuracy: `{raw_50['best_val_acc']:.4f}`
- Test accuracy: `{raw_50['test_acc']:.4f}`
- Rank-64 accuracy: `{raw_50['rank_64_acc']:.4f}`
- Full-rank accuracy: `{raw_50['full_rank_acc']:.4f}`

Interpretation: the raw-pixel CIFAR extension is complete, but the resulting spectrum remains broad.

## Failed controls

- Matched-budget tiny-L1 control: `cifar10_l1tiny_mps_20260422-203929`, test accuracy `0.4180`, rank-64 accuracy `0.4100`

Interpretation: scalar regularization does not explain the pooled-model gain.

## Locality-preserving redesign

- Seed 42 run: `{local_42['run_name']}`, test accuracy `{local_42['test_acc']:.4f}`, rank-64 accuracy `{local_42['rank_64_acc']:.4f}`
- Seed 123 run: `{local_123['run_name']}`, test accuracy `{local_123['test_acc']:.4f}`, rank-64 accuracy `{local_123['rank_64_acc']:.4f}`
- Tuned 50-epoch run: `{local_best['run_name']}`, test accuracy `{local_best['test_acc']:.4f}`, rank-64 accuracy `{local_best['rank_64_acc']:.4f}`

Interpretation: the `4x4` locality-preserving redesign is stable across two seeds, clears the raw-student Task G baseline when trained for 50 epochs, and reaches essentially full-rank performance by rank 64.

## Locality-strength tradeoff

- Intermediate `8x8` pooling: `cifar10_locality8_mps_20260422-204446`, test accuracy `0.3812`, rank-32 accuracy `0.3805`
- Aggressive `16x16` pooling: `cifar10_locality16_mps_20260422-204115`, test accuracy `0.2556`, rank-8 accuracy `0.2493`
- Replication: `cifar10_locality16_seed123_mps_20260422-204255`, test accuracy `0.2570`, rank-8 accuracy `0.2543`

Interpretation: stronger pooling keeps making the spectrum more compressible, but beyond `4x4` it removes too much semantic content from CIFAR-10.

## Report recommendation

Frame Extension 1 / Task I as a compatibility result:

- Raw-pixel CIFAR-10 shows that the MNIST result does not directly transfer.
- Scalar controls do not fix the problem.
- Locality-preserving CIFAR-10 shows that the weight-based analysis becomes much more compatible with natural images once basic spatial structure is restored.
- The best regime is moderate locality bias, not maximal pooling.
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
