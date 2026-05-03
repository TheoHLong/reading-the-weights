#!/usr/bin/env python
"""Measure stable rank and effective rank of frozen guide activations.

For each (guide_source, guide_arch, guide_seed, input_type) combination defined
by a wave-2/wave-3 transfer config, this script:

  1. Builds the frozen guide via load_frozen_guide() — the SAME path used by
     train_cka_experiment(), so the guide here is bit-for-bit identical to what
     produced our training results.
  2. Forwards a fixed CIFAR-10 batch (or noise of matching shape) through the
     guide, with ActivationCapture wrapping the configured teacher_layers.
  3. Computes per-layer:
        stable_rank  = ||A||_F^2 / ||A||_2^2
        effective_rank = exp( -sum_i p_i * log(p_i) )   with p_i = sigma_i^2 / sum_j sigma_j^2
     where A is the (B, D)-flattened activation matrix and sigma are its
     singular values. Stable rank gives a continuous proxy for "how many
     directions actually carry energy"; effective rank is the spectral entropy
     analogue and is more sensitive to long tails.
  4. Saves results/diagnostics/guide_rank/<config_name>.json.

This is the diagnostic that turns the wave-2 "low-rank covariance" hypothesis
from a story into evidence: stable / crashed seeds should show a measurable
gap in these numbers.

Usage:
    python scripts/measure_guide_rank.py --config configs/transfer/cifar10_cka_n4_random_cnn_noise_s44.yaml
    python scripts/measure_guide_rank.py --config-glob "configs/transfer/cifar10_cka_n4*.yaml"
    python scripts/measure_guide_rank.py --batch-size 256 --num-batches 4 \
        --config configs/transfer/cifar10_cka_n4_random_cnn_noise.yaml \
        --config configs/transfer/cifar10_cka_n4_random_cnn_noise_s43.yaml \
        --config configs/transfer/cifar10_cka_n4_random_cnn_noise_s44.yaml
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any

import torch

from _bootstrap import ensure_project_on_path

ensure_project_on_path()

from src.config import load_config
from src.data import build_image_dataloaders
from src.guide import load_frozen_guide
from src.hooks import ActivationCapture
from src.utils import ensure_dir, resolve_device, set_seed


def stable_rank(activations: torch.Tensor) -> float:
    """||A||_F^2 / ||A||_2^2 — a continuous lower bound on rank.

    Activations are reshaped to (B, D). For (B, C, H, W) we treat each example
    as one row of length C*H*W; this is exactly what cka.py does at training time.
    """
    matrix = activations.reshape(activations.size(0), -1).to(torch.float64)
    if matrix.numel() == 0:
        return 0.0
    fro_sq = matrix.pow(2).sum().item()
    spec = torch.linalg.svdvals(matrix)[0].pow(2).item()
    if spec <= 0:
        return 0.0
    return fro_sq / spec


def effective_rank(activations: torch.Tensor) -> float:
    """exp( spectral entropy of normalized singular values ).

    Roy & Vetterli (2007). For a uniform spectrum on k singular values this
    returns k; for a one-direction matrix it returns 1; degrades smoothly in
    between. Better than stable_rank at picking up long tails (heavy-tailed
    spectra collapse stable_rank but keep effective_rank high).
    """
    matrix = activations.reshape(activations.size(0), -1).to(torch.float64)
    if matrix.numel() == 0:
        return 0.0
    sigma = torch.linalg.svdvals(matrix)
    sigma_sq = sigma.pow(2)
    total = sigma_sq.sum()
    if total <= 0:
        return 0.0
    p = sigma_sq / total
    # avoid log(0): drop near-zero singular values from entropy sum
    p = p[p > 1e-12]
    entropy = -(p * p.log()).sum().item()
    return float(torch.tensor(entropy).exp())


def matrix_summary(activations: torch.Tensor) -> dict[str, float]:
    matrix = activations.reshape(activations.size(0), -1).to(torch.float64)
    return {
        'shape': list(activations.shape),
        'flattened_shape': [int(matrix.size(0)), int(matrix.size(1))],
        'l2_mean_per_row': float(matrix.norm(dim=1).mean().item()),
        'frobenius': float(matrix.norm().item()),
        'max_abs': float(matrix.abs().max().item()),
        'stable_rank': stable_rank(activations),
        'effective_rank': effective_rank(activations),
    }


def build_inputs_for_mode(reference_batch: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == 'student':
        return reference_batch
    if mode == 'noise':
        return torch.randn_like(reference_batch)
    raise ValueError(f"Unsupported teacher_input mode: {mode!r}")


def measure_one_config(
    config_path: Path,
    *,
    batch_size: int,
    num_batches: int,
    device: torch.device,
) -> dict[str, Any]:
    config = load_config(config_path)
    transfer_cfg = config.get('transfer', {})
    teacher_layers = list(transfer_cfg.get('teacher_layers', []))
    if not teacher_layers:
        raise ValueError(f'{config_path}: transfer.teacher_layers is required')

    # Force a deterministic seed for the input batch, independent of guide_seed.
    set_seed(int(config.get('seed', 42)))

    # Build a small CIFAR-10 loader. We override batch_size and use the train loader
    # because the val transforms are deterministic and CKA at train time uses train inputs.
    train_cfg_override = dict(config['train'])
    train_cfg_override['batch_size'] = batch_size
    train_cfg_override['num_workers'] = 0
    train_cfg_override['pin_memory'] = False
    bundle = build_image_dataloaders(config['dataset'], train_cfg_override)

    teacher_input_mode = str(transfer_cfg.get('teacher_input', 'student'))

    guide = load_frozen_guide(config, transfer_cfg, device)

    per_layer_records: dict[str, list[dict[str, float]]] = {name: [] for name in teacher_layers}

    fed_batches = 0
    for batch_idx, (x, _) in enumerate(bundle.train_loader):
        if fed_batches >= num_batches:
            break
        x = x.to(device, non_blocking=False)
        guide_inputs = build_inputs_for_mode(x, teacher_input_mode)
        with torch.no_grad():
            with ActivationCapture(guide, teacher_layers) as cap:
                _ = guide(guide_inputs)
            for name in teacher_layers:
                per_layer_records[name].append(matrix_summary(cap.activations[name].detach()))
        fed_batches += 1

    # Aggregate across batches
    summary: dict[str, dict[str, float]] = {}
    for name, records in per_layer_records.items():
        if not records:
            summary[name] = {}
            continue
        keys = ['stable_rank', 'effective_rank', 'l2_mean_per_row', 'frobenius', 'max_abs']
        agg: dict[str, float] = {}
        for k in keys:
            vals = [r[k] for r in records]
            agg[f'{k}_mean'] = float(sum(vals) / len(vals))
            agg[f'{k}_min'] = float(min(vals))
            agg[f'{k}_max'] = float(max(vals))
        agg['flattened_shape'] = records[0]['flattened_shape']
        summary[name] = agg

    return {
        'config_path': str(config_path),
        'experiment_name': config.get('experiment_name'),
        'teacher_source': str(transfer_cfg.get('teacher_source', 'checkpoint')),
        'teacher_input': teacher_input_mode,
        'guide_architecture': str(transfer_cfg.get('guide_architecture')
                                  or config.get('guide', {}).get('architecture')
                                  or 'cifar_resnet18'),
        'guide_seed': transfer_cfg.get('guide_seed'),
        'data_seed': int(config.get('seed', 42)),
        'batch_size': batch_size,
        'num_batches': num_batches,
        'teacher_layers': teacher_layers,
        'per_layer': summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Measure stable / effective rank of frozen guide activations.')
    parser.add_argument('--config', type=Path, action='append', default=[],
                        help='Config to measure (repeatable).')
    parser.add_argument('--config-glob', type=str, default=None,
                        help='Optional glob to expand into a list of configs.')
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--num-batches', type=int, default=4,
                        help='Number of batches to average across (more = smoother rank estimates).')
    parser.add_argument('--out-dir', type=Path, default=Path('results/diagnostics/guide_rank'))
    parser.add_argument('--device', type=str, default='auto')
    args = parser.parse_args()

    configs: list[Path] = list(args.config)
    if args.config_glob:
        configs.extend(Path(p) for p in sorted(glob.glob(args.config_glob)))
    configs = [p for p in configs if p.exists()]
    if not configs:
        raise SystemExit('No configs provided. Use --config or --config-glob.')

    device = resolve_device(args.device)
    out_dir = ensure_dir(args.out_dir)

    print(f'[measure_guide_rank] device = {device}')
    print(f'[measure_guide_rank] writing to {out_dir}')
    print(f'[measure_guide_rank] {len(configs)} configs to process')

    summary_index: list[dict[str, Any]] = []
    for cfg_path in configs:
        print(f'\n--- {cfg_path} ---')
        try:
            result = measure_one_config(
                cfg_path,
                batch_size=args.batch_size,
                num_batches=args.num_batches,
                device=device,
            )
        except Exception as exc:  # noqa: BLE001
            print(f'  [error] {exc}')
            continue
        out_path = out_dir / f'{cfg_path.stem}.json'
        with out_path.open('w') as f:
            json.dump(result, f, indent=2)
        # Print a one-line per-layer summary
        for layer, stats in result['per_layer'].items():
            sr = stats.get('stable_rank_mean')
            er = stats.get('effective_rank_mean')
            l2 = stats.get('l2_mean_per_row_mean')
            mx = stats.get('max_abs_mean')
            print(f'  {layer:>12s}  stable={sr:7.2f}  effective={er:7.2f}  '
                  f'rowL2={l2:8.2e}  max|A|={mx:8.2e}')
        summary_index.append({
            'config': str(cfg_path),
            'experiment_name': result['experiment_name'],
            'teacher_source': result['teacher_source'],
            'teacher_input': result['teacher_input'],
            'guide_architecture': result['guide_architecture'],
            'guide_seed': result['guide_seed'],
            'output': str(out_path),
        })

    index_path = out_dir / 'index.json'
    with index_path.open('w') as f:
        json.dump(summary_index, f, indent=2)
    print(f'\n[measure_guide_rank] wrote index → {index_path}')


if __name__ == '__main__':
    main()
