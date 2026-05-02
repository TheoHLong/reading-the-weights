#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import torch

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from reading_weights.data import build_image_dataloaders
from reading_weights.decomposition import decompose_bilinear_model
from reading_weights.model import build_image_classifier
from reading_weights.utils import ensure_dir, load_checkpoint, resolve_device, write_json


def parse_ranks(rank_arg: str, d_hidden: int) -> list[int]:
    if rank_arg == 'auto':
        ranks = [1, 2, 4, 8, 16, 32, 64, 128, 256, d_hidden]
        return sorted({rank for rank in ranks if rank <= d_hidden})
    return sorted({int(part.strip()) for part in rank_arg.split(',') if part.strip()})


def compute_rank_efficiency(results: list[dict[str, float | int]]) -> dict[str, float | int | None]:
    full_rank_acc = float(results[-1]['accuracy'])
    thresholds = {
        'rank_for_90pct_full': 0.90 * full_rank_acc,
        'rank_for_95pct_full': 0.95 * full_rank_acc,
        'rank_for_99pct_full': 0.99 * full_rank_acc,
    }
    rank_efficiency: dict[str, float | int | None] = {
        'full_rank_accuracy': full_rank_acc,
    }
    for key, target_acc in thresholds.items():
        reached_rank = next(
            (int(row['rank']) for row in results if float(row['accuracy']) >= target_acc),
            None,
        )
        rank_efficiency[key] = reached_rank
    return rank_efficiency


@torch.no_grad()
def evaluate_truncated_spectrum(
    model,
    loader,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    ranks: list[int],
    device: torch.device,
) -> list[dict[str, float | int]]:
    model.eval()
    embedding_weight = model.embedding_weight.to(device)
    sorted_idx = torch.argsort(eigenvalues.abs(), dim=-1, descending=True)
    truncated_components = {}
    for rank in ranks:
        top_idx = sorted_idx[:, :rank]
        truncated_components[rank] = (
            torch.gather(eigenvalues, dim=-1, index=top_idx).to(device),
            torch.gather(
                eigenvectors,
                dim=-1,
                index=top_idx.unsqueeze(1).expand(-1, eigenvectors.size(1), -1),
            ).to(device),
        )

    metrics = {
        rank: {
            'correct': 0,
            'count': 0,
        }
        for rank in ranks
    }

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        flat = x.flatten(start_dim=1)
        hidden = flat @ embedding_weight.T

        for rank in ranks:
            truncated_evals, top_evecs = truncated_components[rank]
            projections = torch.einsum('bi,cir->bcr', hidden, top_evecs)
            logits = (projections.square() * truncated_evals.unsqueeze(0)).sum(dim=-1)

            metrics[rank]['correct'] += (logits.argmax(dim=-1) == y).sum().item()
            metrics[rank]['count'] += y.size(0)

    results: list[dict[str, float | int]] = []
    for rank in ranks:
        count = metrics[rank]['count']
        results.append(
            {
                'rank': rank,
                'accuracy': metrics[rank]['correct'] / count,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate eigenvalue truncation vs. accuracy from a saved checkpoint.')
    parser.add_argument('--checkpoint', type=Path, required=True, help='Path to a saved checkpoint.')
    parser.add_argument(
        '--ranks',
        type=str,
        default='auto',
        help='Comma-separated truncation ranks, or "auto" for a default schedule.',
    )
    parser.add_argument(
        '--split',
        choices=['val', 'test'],
        default='test',
        help='Data split used for truncation evaluation.',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        help='Evaluation device. Defaults to auto.',
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/truncation'),
        help='Directory for truncation analysis outputs.',
    )
    args = parser.parse_args()

    payload = load_checkpoint(args.checkpoint, map_location='cpu')
    config = payload['config']
    config.setdefault('train', {})
    config['train']['device'] = args.device
    if resolve_device(args.device).type == 'cpu':
        config['train']['num_workers'] = 0
        config['train']['pin_memory'] = False

    dataset_bundle = build_image_dataloaders(config['dataset'], config['train'])
    loader = dataset_bundle.val_loader if args.split == 'val' else dataset_bundle.test_loader
    device = resolve_device(args.device)

    model_cpu = build_image_classifier(config['model'], seed=int(config['seed']))
    load_result = model_cpu.load_state_dict(payload['model_state_dict'], strict=False)
    tolerated_missing = {'input_projection'}
    if set(load_result.missing_keys) - tolerated_missing or load_result.unexpected_keys:
        raise RuntimeError(
            'Checkpoint state dict mismatch: '
            f'missing={load_result.missing_keys}, unexpected={load_result.unexpected_keys}'
        )
    model_cpu.eval()
    artifacts = decompose_bilinear_model(model_cpu)
    eigenvalues, eigenvectors = torch.linalg.eigh(artifacts.symmetrized_tensor)

    model = model_cpu.to(device)
    d_hidden = int(config['model']['d_hidden'])
    ranks = parse_ranks(args.ranks, d_hidden)
    results = evaluate_truncated_spectrum(
        model=model,
        loader=loader,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        ranks=ranks,
        device=device,
    )
    rank_efficiency = compute_rank_efficiency(results)

    output_dir = ensure_dir(args.output_dir / args.checkpoint.stem)
    results_path = output_dir / f'{args.split}_truncation.csv'
    pd.DataFrame(results).to_csv(results_path, index=False)

    write_json(
        output_dir / f'{args.split}_summary.json',
        {
            'checkpoint': str(args.checkpoint),
            'dataset': config['dataset']['name'],
            'split': args.split,
            'device': str(device),
            'ranks': ranks,
            'best_checkpoint_val_acc': float(payload['metrics']['val_acc']),
            'rank_efficiency': rank_efficiency,
            'results_path': str(results_path),
        },
    )
    print(f'Truncation results saved to {results_path}')


if __name__ == '__main__':
    main()
