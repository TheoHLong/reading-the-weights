#!/usr/bin/env python
from __future__ import annotations

import argparse
from copy import deepcopy
import os
from pathlib import Path
import tempfile

_plot_cache_dir = Path(tempfile.gettempdir()) / 'reading_weights_matplotlib'
_plot_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR', str(_plot_cache_dir))
os.environ.setdefault('XDG_CACHE_HOME', str(_plot_cache_dir))

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from _bootstrap import ensure_project_on_path

ensure_project_on_path()

from src.config import load_config
from src.data import build_image_dataloaders
from src.decomposition import decompose_bilinear_model, spectral_effective_rank
from src.model import build_image_classifier
from src.train import evaluate
from src.utils import ensure_dir, load_checkpoint, resolve_device, set_seed, timestamp, write_json


CLASS_NAMES = {
    'mnist': [str(i) for i in range(10)],
    'fashion_mnist': [
        'T-shirt',
        'Trouser',
        'Pullover',
        'Dress',
        'Coat',
        'Sandal',
        'Shirt',
        'Sneaker',
        'Bag',
        'Boot',
    ],
}


def validate_decomposition_config(config: dict) -> None:
    model_cfg = config.get('model', {})
    n_layer = int(model_cfg.get('n_layer', 1))
    gate = model_cfg.get('gate')
    if n_layer != 1:
        raise ValueError(
            'train_with_dynamics currently supports n_layer=1 only '
            '(decomposition limitation).'
        )
    if gate not in (None, 'sqs'):
        raise ValueError(
            'train_with_dynamics requires a decomposable gate: None or "sqs"; '
            f'got gate={gate!r}.'
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train a bilinear model and track eigenspectrum dynamics from periodic checkpoints.'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/baselines/mnist_baseline.yaml'),
        help='Path to the YAML config file.',
    )
    parser.add_argument('--epochs', type=int, default=None, help='Override config train.epochs.')
    parser.add_argument(
        '--checkpoint-after',
        type=int,
        default=None,
        help='Save and analyze a periodic checkpoint every N epochs. Defaults to config value or 10.',
    )
    parser.add_argument(
        '--analysis-dir',
        type=Path,
        default=Path('results/analysis'),
        help='Base directory for training-dynamics artifacts.',
    )
    parser.add_argument('--max-train-batches', type=int, default=None, help='Debug limiter for training batches.')
    parser.add_argument('--max-eval-batches', type=int, default=None, help='Debug limiter for eval batches.')
    parser.add_argument('--num-workers', type=int, default=None, help='Override DataLoader workers.')
    parser.add_argument('--device', default=None, help='Override train.device, e.g. cpu, cuda, mps, or auto.')
    parser.add_argument(
        '--clamp-noisy-inputs',
        action='store_true',
        default=None,
        help=(
            'Clamp inputs to [0, 1] after train.input_noise_std is applied. '
            'Default: off (preserves the existing input_noise_std training-noise '
            'semantics used by SQS / paper-style configs). Turn this on to match '
            'the input-perturbation semantics used by scripts/noise_sweep_analysis.py, '
            'where noisy inputs are kept inside the valid image range. Only makes '
            'sense for unit-range datasets like MNIST / Fashion-MNIST.'
        ),
    )
    return parser.parse_args()


def train_with_periodic_checkpoints(config: dict) -> dict[str, Path | list[Path]]:
    set_seed(int(config['seed']))
    config.setdefault('train', {})
    config['train'].setdefault('split_seed', int(config['seed']))
    validate_decomposition_config(config)

    dataset_bundle = build_image_dataloaders(config['dataset'], config['train'])
    device = resolve_device(config['train'].get('device', 'auto'))
    max_train_batches = config['train'].get('max_train_batches')
    max_eval_batches = config['train'].get('max_eval_batches')
    max_train_batches = None if max_train_batches in (None, 0) else int(max_train_batches)
    max_eval_batches = None if max_eval_batches in (None, 0) else int(max_eval_batches)
    checkpoint_after = int(config['train'].get('checkpoint_after', 10) or 10)
    if checkpoint_after <= 0:
        raise ValueError('checkpoint_after must be positive.')

    model = build_image_classifier(config['model'], seed=int(config['seed'])).to(device)
    input_noise_std = float(config['train'].get('input_noise_std', 0.0))
    clamp_noisy_inputs = bool(config['train'].get('clamp_noisy_inputs', False))
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=float(config['train']['lr']),
        weight_decay=float(config['train']['wd']),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=int(config['train']['epochs']))

    run_name = f"{config['experiment_name']}_{timestamp()}"
    metrics_dir = ensure_dir(config.get('output_dir', 'results/metrics'))
    checkpoint_dir = ensure_dir(config.get('checkpoint_dir', 'checkpoints'))
    run_dir = ensure_dir(metrics_dir / run_name)

    history: list[dict[str, float | int]] = []
    best_val_acc = -1.0
    best_checkpoint_path = checkpoint_dir / f'{run_name}.pt'
    latest_checkpoint_path = checkpoint_dir / f'{run_name}_latest.pt'
    periodic_checkpoint_paths: list[Path] = []

    for epoch in tqdm(range(1, int(config['train']['epochs']) + 1), desc='training+dynamics'):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_examples = 0

        for batch_idx, (x, y) in enumerate(dataset_bundle.train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if input_noise_std > 0:
                x = x + input_noise_std * torch.randn_like(x)
                if clamp_noisy_inputs:
                    x = x.clamp(0.0, 1.0)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            running_correct += (logits.argmax(dim=-1) == y).sum().item()
            running_examples += y.size(0)

            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

        scheduler.step()

        train_loss = running_loss / running_examples
        train_acc = running_correct / running_examples
        val_loss, val_acc = evaluate(
            model,
            dataset_bundle.val_loader,
            criterion,
            device,
            max_batches=max_eval_batches,
        )

        row = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': scheduler.get_last_lr()[0],
        }
        history.append(row)

        checkpoint_payload = {
            'model_state_dict': model.state_dict(),
            'config': config,
            'epoch': epoch,
            'metrics': row,
        }
        torch.save(checkpoint_payload, latest_checkpoint_path)

        if epoch % checkpoint_after == 0:
            periodic_path = checkpoint_dir / f'{run_name}_epoch{epoch:04d}.pt'
            torch.save(checkpoint_payload, periodic_path)
            periodic_checkpoint_paths.append(periodic_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint_payload, best_checkpoint_path)

    metrics_path = run_dir / 'metrics.csv'
    pd.DataFrame(history).to_csv(metrics_path, index=False)
    write_json(run_dir / 'config.json', config)
    write_json(
        run_dir / 'summary.json',
        {
            'run_name': run_name,
            'dataset': config['dataset']['name'],
            'device': str(device),
            'best_val_acc': best_val_acc,
            'best_checkpoint': str(best_checkpoint_path),
            'latest_checkpoint': str(latest_checkpoint_path),
            'periodic_checkpoints': [str(path) for path in periodic_checkpoint_paths],
        },
    )

    return {
        'run_dir': run_dir,
        'metrics_path': metrics_path,
        'best_checkpoint_path': best_checkpoint_path,
        'latest_checkpoint_path': latest_checkpoint_path,
        'periodic_checkpoint_paths': periodic_checkpoint_paths,
    }


def analyze_checkpoint(checkpoint_path: Path, snapshots_dir: Path) -> dict[str, float | int | str]:
    payload = load_checkpoint(checkpoint_path, map_location='cpu')
    config = payload['config']
    model = build_image_classifier(config['model'], seed=int(config['seed']))
    model.load_state_dict(payload['model_state_dict'])
    model.eval()

    decomposition = decompose_bilinear_model(model)
    effective_rank = spectral_effective_rank(decomposition.eigenvalues)
    snapshot_path = snapshots_dir / f"epoch_{int(payload['epoch']):04d}_decomposition.pt"
    torch.save(decomposition.to_payload(), snapshot_path)

    top_class = int(decomposition.eigenvalues[:, -1].argmax().item())
    return {
        'epoch': int(payload['epoch']),
        'checkpoint': str(checkpoint_path),
        'snapshot_path': str(snapshot_path),
        'val_acc': float(payload['metrics']['val_acc']),
        'train_acc': float(payload['metrics']['train_acc']),
        'mean_effective_rank': float(effective_rank.mean().item()),
        'max_effective_rank': float(effective_rank.max().item()),
        'top_class': top_class,
        'top_eigenvalue': float(decomposition.eigenvalues[top_class, -1].item()),
        'top_class_effective_rank': float(effective_rank[top_class].item()),
    }


def resolve_analysis_checkpoints(artifacts: dict[str, Path | list[Path]]) -> list[Path]:
    paths = [Path(path) for path in artifacts['periodic_checkpoint_paths']]
    for key in ('best_checkpoint_path', 'latest_checkpoint_path'):
        path = Path(artifacts[key])
        if path.exists():
            paths.append(path)

    unique_by_epoch: dict[int, Path] = {}
    for path in paths:
        payload = load_checkpoint(path, map_location='cpu')
        epoch = int(payload['epoch'])
        if epoch not in unique_by_epoch:
            unique_by_epoch[epoch] = path
        elif path == Path(artifacts['best_checkpoint_path']):
            unique_by_epoch[epoch] = path

    return [unique_by_epoch[epoch] for epoch in sorted(unique_by_epoch)]


def build_eigenvector_evolution_plot(
    *,
    dynamics_df: pd.DataFrame,
    output_dir: Path,
    input_shape: tuple[int, int, int],
    dataset_name: str,
) -> list[str]:
    epochs = dynamics_df['epoch'].tolist()
    if not epochs:
        return []

    last_snapshot = torch.load(Path(dynamics_df.iloc[-1]['snapshot_path']), map_location='cpu', weights_only=False)
    num_classes = int(last_snapshot['eigenvalues'].shape[0])
    class_names = CLASS_NAMES.get(dataset_name, [f'class_{idx}' for idx in range(num_classes)])
    channels, height, width = input_shape
    plot_paths: list[str] = []

    for class_idx in range(num_classes):
        if channels == 1:
            fig, axes = plt.subplots(1, len(epochs), figsize=(3 * len(epochs), 3), squeeze=False)
        else:
            fig, axes = plt.subplots(
                channels,
                len(epochs),
                figsize=(3 * len(epochs), 2.4 * channels),
                squeeze=False,
            )

        for col_idx, (_, row) in enumerate(dynamics_df.iterrows()):
            snapshot = torch.load(Path(row['snapshot_path']), map_location='cpu', weights_only=False)
            eigenvector = snapshot['eigenvectors_input'][class_idx, -1].reshape(channels, height, width)
            eigenvalue = float(snapshot['eigenvalues'][class_idx, -1].item())
            vmax = float(eigenvector.abs().max().clamp_min(1e-8).item())

            for channel_idx in range(channels):
                ax = axes[channel_idx][col_idx]
                ax.imshow(eigenvector[channel_idx].numpy(), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                if channel_idx == 0:
                    ax.set_title(f"Epoch {int(row['epoch'])}\nEig {eigenvalue:.3f}")
                if channels > 1 and col_idx == 0:
                    ax.set_ylabel(f'channel {channel_idx}')
                ax.axis('off')

        fig.suptitle(f'Top eigenvector evolution: {class_names[class_idx]}', fontsize=12)
        fig.tight_layout()
        output_path = output_dir / f'eigenvector_formation_class_{class_idx:02d}.png'
        fig.savefig(output_path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        plot_paths.append(str(output_path))

    return plot_paths


def build_effective_rank_plot(dynamics_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(dynamics_df['epoch'], dynamics_df['mean_effective_rank'], marker='o', label='Mean effective rank')
    ax.plot(dynamics_df['epoch'], dynamics_df['max_effective_rank'], marker='s', label='Max effective rank')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Effective rank')
    ax.set_title('Effective rank evolution over training')
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = deepcopy(load_config(args.config))
    config.setdefault('train', {})
    if args.epochs is not None:
        config['train']['epochs'] = int(args.epochs)
    if args.checkpoint_after is not None:
        config['train']['checkpoint_after'] = int(args.checkpoint_after)
    else:
        config['train'].setdefault('checkpoint_after', 10)
    if args.max_train_batches is not None:
        config['train']['max_train_batches'] = int(args.max_train_batches)
    if args.max_eval_batches is not None:
        config['train']['max_eval_batches'] = int(args.max_eval_batches)
    if args.num_workers is not None:
        config['train']['num_workers'] = int(args.num_workers)
        config['train']['pin_memory'] = False
    if args.device is not None:
        config['train']['device'] = args.device
    if args.clamp_noisy_inputs is not None:
        config['train']['clamp_noisy_inputs'] = bool(args.clamp_noisy_inputs)

    artifacts = train_with_periodic_checkpoints(config)
    run_dir = Path(artifacts['run_dir'])
    dynamics_dir = ensure_dir(args.analysis_dir / run_dir.name / 'training_dynamics')
    snapshots_dir = ensure_dir(dynamics_dir / 'snapshots')

    checkpoint_paths = resolve_analysis_checkpoints(artifacts)

    rows = [analyze_checkpoint(path, snapshots_dir) for path in checkpoint_paths]
    dynamics_df = pd.DataFrame(rows).sort_values('epoch').reset_index(drop=True)
    history_path = dynamics_dir / 'effective_rank_history.csv'
    dynamics_df.to_csv(history_path, index=False)

    input_shape = (
        int(config['dataset']['channels']),
        int(config['dataset']['image_size']),
        int(config['dataset']['image_size']),
    )
    effective_rank_plot = dynamics_dir / 'effective_rank_evolution.png'
    build_effective_rank_plot(dynamics_df, effective_rank_plot)
    eigenvector_plot_paths = build_eigenvector_evolution_plot(
        dynamics_df=dynamics_df,
        output_dir=dynamics_dir,
        input_shape=input_shape,
        dataset_name=config['dataset']['name'],
    )

    summary_path = dynamics_dir / 'summary.json'
    write_json(
        summary_path,
        {
            'run_dir': str(run_dir),
            'metrics_path': str(artifacts['metrics_path']),
            'best_checkpoint_path': str(artifacts['best_checkpoint_path']),
            'latest_checkpoint_path': str(artifacts['latest_checkpoint_path']),
            'tracked_checkpoints': [str(path) for path in checkpoint_paths],
            'checkpoint_after': int(config['train']['checkpoint_after']),
            'history_path': str(history_path),
            'plots': {
                'effective_rank_evolution': str(effective_rank_plot),
                'eigenvector_formation': eigenvector_plot_paths,
            },
        },
    )

    print('Training dynamics complete.')
    print(f'run_dir: {run_dir}')
    print(f'dynamics_dir: {dynamics_dir}')
    print(f'history_csv: {history_path}')
    print(f'summary: {summary_path}')


if __name__ == '__main__':
    main()
