#!/usr/bin/env python
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from reading_weights.config import load_config
from reading_weights.data import build_image_dataloaders
from reading_weights.decomposition import decompose_bilinear_model, spectral_effective_rank
from reading_weights.model import build_image_classifier
from reading_weights.train import evaluate
from reading_weights.utils import ensure_dir, load_checkpoint, resolve_device, set_seed, timestamp, write_json
DIGIT_NAMES = {
    'mnist': [str(i) for i in range(10)],
    'fashion_mnist': ['T-shirt','Trouser','Pullover','Dress','Coat',
                      'Sandal','Shirt','Sneaker','Bag','Boot'],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Train the model and track eigendecomposition dynamics from checkpoints.'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/mnist_baseline.yaml'),
        help='Path to the YAML config file.',
    )
    parser.add_argument(
        '--checkpoint-after',
        type=int,
        default=None,
        help='Override the periodic checkpoint cadence in epochs.',
    )
    parser.add_argument(
        '--analysis-dir',
        type=Path,
        default=Path('results/analysis'),
        help='Base directory for training-dynamics artifacts.',
    )
    return parser.parse_args()


def train_with_periodic_checkpoints(config: dict) -> dict[str, Path | list[Path]]:
    set_seed(int(config['seed']))

    dataset_bundle = build_image_dataloaders(config['dataset'], config['train'])
    device = resolve_device(config['train'].get('device', 'auto'))
    max_train_batches = config['train'].get('max_train_batches')
    max_eval_batches = config['train'].get('max_eval_batches')
    max_train_batches = None if max_train_batches in (None, 0) else int(max_train_batches)
    max_eval_batches = None if max_eval_batches in (None, 0) else int(max_eval_batches)
    checkpoint_after = int(config['train'].get('checkpoint_after', 0) or 0)
    if checkpoint_after <= 0:
        raise ValueError('checkpoint after should be > 0.')

    model = build_image_classifier(config['model'], seed=int(config['seed'])).to(device)
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


def analyze_checkpoint(
    checkpoint_path: Path,
    snapshots_dir: Path,
) -> dict[str, float | int | str]:
    payload = load_checkpoint(checkpoint_path, map_location='cpu')
    config = payload['config']
    model = build_image_classifier(config['model'], seed=int(config['seed']))
    model.load_state_dict(payload['model_state_dict'])
    model.eval()

    decomposition = decompose_bilinear_model(model)
    effective_rank = spectral_effective_rank(decomposition.eigenvalues)
    snapshot_path = snapshots_dir / f"epoch_{int(payload['epoch']):04d}_decomposition.pt"
    torch.save(decomposition.to_payload(), snapshot_path)

    dominant_class = int(decomposition.eigenvalues[:, -1].argmax().item())
    top_effective_rank = float(effective_rank[dominant_class].item())

    return {
        'epoch': int(payload['epoch']),
        'checkpoint': str(checkpoint_path),
        'snapshot_path': str(snapshot_path),
        'val_acc': float(payload['metrics']['val_acc']),
        'train_acc': float(payload['metrics']['train_acc']),
        'mean_effective_rank': float(effective_rank.mean().item()),
        'max_effective_rank': float(effective_rank.max().item()),
        'dominant_class': dominant_class,
        'dominant_top_eigenvalue': float(decomposition.eigenvalues[dominant_class, -1].item()),
        'dominant_effective_rank': top_effective_rank,
    }


def build_eigenvector_evolution_plot(
    dynamics_df: pd.DataFrame,
    output_dir: Path,
    input_shape: tuple[int, int, int],
    dataset_name: str,
) -> list[Path]:
    epochs = dynamics_df['epoch'].tolist()
    if not epochs:
        return []

    last_snapshot = torch.load(Path(dynamics_df.iloc[-1]['snapshot_path']), map_location='cpu')
    last_eigenvalues = last_snapshot['eigenvalues']
    num_classes = last_eigenvalues.shape[0]
    class_names = DIGIT_NAMES.get(dataset_name, [f'Class {i}' for i in range(num_classes)])

    plot_paths = []
    for class_idx in range(num_classes):
        channels, height, width = input_shape
        if channels == 1:
            fig, axes = plt.subplots(1, len(epochs), figsize=(3 * len(epochs), 3), squeeze=False)
        else:
            fig, axes = plt.subplots(channels, len(epochs), figsize=(3 * len(epochs), 3 * channels), squeeze=False)

        for col_idx, (_, row) in enumerate(dynamics_df.iterrows()):
            snapshot = torch.load(Path(row['snapshot_path']), map_location='cpu')
            eigenvector = snapshot['eigenvectors_input'][class_idx, -1].reshape(channels, height, width)
            eigenvalue = float(snapshot['eigenvalues'][class_idx, -1].item())

            if channels == 1:
                ax = axes[0][col_idx]
                ax.imshow(eigenvector[0].numpy(), cmap='coolwarm')
                ax.set_title(f"Epoch {int(row['epoch'])}\nEig {eigenvalue:.3f}")
                ax.axis('off')
            else:
                for channel_idx in range(channels):
                    ax = axes[channel_idx][col_idx]
                    ax.imshow(eigenvector[channel_idx].numpy(), cmap='coolwarm')
                    if channel_idx == 0:
                        ax.set_title(f"Epoch {int(row['epoch'])}\nEig {eigenvalue:.3f}")
                    if col_idx == 0:
                        ax.set_ylabel(f'Channel {channel_idx}')
                    ax.axis('off')

        fig.suptitle(f'Top eigenvector evolution for {class_names[class_idx]} (class {class_idx})', fontsize=12)
        fig.tight_layout()
        output_path = output_dir / f'eigenvector_formation_class_{class_idx:02d}.png'
        fig.savefig(output_path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        plot_paths.append(output_path)

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
    if args.checkpoint_after is not None:
        config.setdefault('train', {})['checkpoint_after'] = args.checkpoint_after

    checkpoint_after = int(config['train'].get('checkpoint_after', 0) or 0)
    artifacts = train_with_periodic_checkpoints(config)
    run_dir = Path(artifacts['run_dir'])
    dynamics_dir = ensure_dir(args.analysis_dir / run_dir.name / 'training_dynamics')
    snapshots_dir = ensure_dir(dynamics_dir / 'snapshots')

    checkpoint_paths = [Path(path) for path in artifacts['periodic_checkpoint_paths']]
    latest_path = Path(artifacts['latest_checkpoint_path'])
    if latest_path not in checkpoint_paths:
        checkpoint_paths.append(latest_path)
    rows = [analyze_checkpoint(path, snapshots_dir) for path in checkpoint_paths]
    dynamics_df = pd.DataFrame(rows).sort_values('epoch').reset_index(drop=True)
    dynamics_df.to_csv(dynamics_dir / 'effective_rank_history.csv', index=False)

    input_shape = (
        int(config['dataset']['channels']),
        int(config['dataset']['image_size']),
        int(config['dataset']['image_size']),
    )
    build_effective_rank_plot(dynamics_df, dynamics_dir / 'effective_rank_evolution.png')
    eigenvector_plot_paths = build_eigenvector_evolution_plot(
        dynamics_df=dynamics_df,
        output_dir=dynamics_dir,
        input_shape=input_shape,
        dataset_name=config['dataset']['name'],
    )

    write_json(
        dynamics_dir / 'summary.json',
        {
            'run_dir': str(run_dir),
            'metrics_path': str(artifacts['metrics_path']),
            'best_checkpoint_path': str(artifacts['best_checkpoint_path']),
            'latest_checkpoint_path': str(artifacts['latest_checkpoint_path']),
            'tracked_checkpoints': [str(path) for path in checkpoint_paths],
            'checkpoint_after': checkpoint_after,
            'plots': {
                'effective_rank_evolution': str(dynamics_dir / 'effective_rank_evolution.png'),
                'eigenvector_formation': [str(path) for path in eigenvector_plot_paths],
            },
        },
    )

    print('Training dynamics complete.')
    print(f"run_dir: {run_dir}")
    print(f"dynamics_dir: {dynamics_dir}")
    print(f"history_csv: {dynamics_dir / 'effective_rank_history.csv'}")


if __name__ == '__main__':
    main()
