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
from torch import Tensor, nn
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
from src.utils import ensure_dir, resolve_device, set_seed, write_json


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
            'noise_sweep_analysis currently supports n_layer=1 only '
            '(decomposition limitation).'
        )
    if gate not in (None, 'sqs'):
        raise ValueError(
            'noise_sweep_analysis requires a decomposable gate: None or "sqs"; '
            f'got gate={gate!r}.'
        )


class GaussianNoise(nn.Module):
    def __init__(self, *, mode: str, value: float, clamp_unit: bool) -> None:
        super().__init__()
        if mode not in ('norm', 'std'):
            raise ValueError(f"mode must be 'norm' or 'std', got {mode!r}")
        self.mode = mode
        self.value = float(value)
        self.clamp_unit = clamp_unit

    def forward(self, x: Tensor) -> Tensor:
        if self.value == 0:
            return x

        noise = torch.randn_like(x)
        if self.mode == 'norm':
            flat = noise.flatten(start_dim=1)
            scale = self.value / flat.norm(dim=1, keepdim=True).clamp_min(1e-8)
            noise = (flat * scale).view_as(x)
        else:
            noise = self.value * noise

        x_noisy = x + noise
        if self.clamp_unit:
            x_noisy = x_noisy.clamp(0.0, 1.0)
        return x_noisy


def _safe_value(value: float) -> str:
    return str(value).replace('-', 'm').replace('.', 'p')


def _resolve_clamp(dataset_name: str, clamp: str) -> bool:
    if clamp == 'unit':
        return True
    if clamp == 'none':
        return False
    return dataset_name in {'mnist', 'fashion_mnist'}


def _input_shape(config: dict) -> tuple[int, int, int]:
    dataset_cfg = config['dataset']
    return (
        int(dataset_cfg['channels']),
        int(dataset_cfg['image_size']),
        int(dataset_cfg['image_size']),
    )


def train_one_model(
    config: dict,
    *,
    noise: nn.Module,
    device: torch.device,
    epochs: int,
    max_train_batches: int | None,
) -> tuple[nn.Module, object]:
    dataset_bundle = build_image_dataloaders(config['dataset'], config['train'])
    model = build_image_classifier(config['model'], seed=int(config['seed'])).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=float(config['train']['lr']),
        weight_decay=float(config['train']['wd']),
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    for _epoch in tqdm(range(1, epochs + 1), desc='training', leave=False):
        model.train()
        for batch_idx, (x, y) in enumerate(dataset_bundle.train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(noise(x))
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
        scheduler.step()

    return model, dataset_bundle


def plot_noise_sweep_eigenvectors(
    *,
    output_dir: Path,
    dataset_name: str,
    input_shape: tuple[int, int, int],
    noise_mode: str,
    noise_values: list[float],
    artifacts_by_value: list,
    metrics_rows: list[dict[str, float | int | str]],
) -> list[str]:
    channels, height, width = input_shape
    num_classes = artifacts_by_value[0].eigenvalues.shape[0]
    class_names = CLASS_NAMES.get(dataset_name, [f'class_{idx}' for idx in range(num_classes)])
    plot_paths: list[str] = []

    for class_idx, class_name in enumerate(class_names):
        if channels == 1:
            fig, axes = plt.subplots(1, len(noise_values), figsize=(3 * len(noise_values), 3), squeeze=False)
        else:
            fig, axes = plt.subplots(
                channels,
                len(noise_values),
                figsize=(3 * len(noise_values), 2.4 * channels),
                squeeze=False,
            )

        for col_idx, (noise_value, artifacts, metrics) in enumerate(
            zip(noise_values, artifacts_by_value, metrics_rows)
        ):
            eigenvector = artifacts.eigenvectors_input[class_idx, -1].reshape(channels, height, width)
            vmax = float(eigenvector.abs().max().clamp_min(1e-8).item())
            for channel_idx in range(channels):
                ax = axes[channel_idx][col_idx]
                ax.imshow(eigenvector[channel_idx].cpu().numpy(), cmap='RdBu_r', vmin=-vmax, vmax=vmax)
                if channel_idx == 0:
                    ax.set_title(f'{noise_mode}={noise_value}\n{100 * float(metrics["eval_acc"]):.1f}%')
                if channels > 1 and col_idx == 0:
                    ax.set_ylabel(f'channel {channel_idx}')
                ax.axis('off')

        fig.suptitle(f'Top eigenvector noise sweep: {class_name} ({dataset_name})', fontsize=11)
        fig.tight_layout()
        plot_path = output_dir / f'class_{class_idx:02d}_{class_name.lower().replace("/", "_")}.png'
        fig.savefig(plot_path, dpi=180, bbox_inches='tight')
        plt.close(fig)
        plot_paths.append(str(plot_path))

    return plot_paths


def run_noise_sweep(
    config: dict,
    *,
    config_path: Path,
    noise_mode: str,
    noise_values: list[float],
    epochs: int | None,
    eval_split: str,
    output_dir: Path,
    clamp: str,
    max_train_batches: int | None,
    max_eval_batches: int | None,
) -> dict[str, Path]:
    config = deepcopy(config)
    config.setdefault('train', {})
    config['train'].setdefault('split_seed', int(config['seed']))
    validate_decomposition_config(config)
    if max_train_batches is not None:
        config['train']['max_train_batches'] = max_train_batches
    if max_eval_batches is not None:
        config['train']['max_eval_batches'] = max_eval_batches

    dataset_name = str(config['dataset']['name'])
    device = resolve_device(config['train'].get('device', 'auto'))
    resolved_epochs = int(epochs if epochs is not None else config['train']['epochs'])
    clamp_unit = _resolve_clamp(dataset_name, clamp)
    run_dir = ensure_dir(output_dir / dataset_name / noise_mode)

    artifacts_by_value = []
    metrics_rows: list[dict[str, float | int | str]] = []
    criterion = nn.CrossEntropyLoss()

    for sweep_idx, noise_value in enumerate(noise_values):
        print(f'training noise_mode={noise_mode} noise_value={noise_value}')
        set_seed(int(config['seed']) + sweep_idx)
        noise = GaussianNoise(mode=noise_mode, value=noise_value, clamp_unit=clamp_unit).to(device)
        model, dataset_bundle = train_one_model(
            config,
            noise=noise,
            device=device,
            epochs=resolved_epochs,
            max_train_batches=max_train_batches,
        )

        eval_loader = dataset_bundle.val_loader if eval_split == 'val' else dataset_bundle.test_loader
        eval_loss, eval_acc = evaluate(
            model,
            eval_loader,
            criterion,
            device,
            max_batches=max_eval_batches,
        )

        model_cpu = model.cpu().eval()
        artifacts = decompose_bilinear_model(model_cpu)
        effective_rank = spectral_effective_rank(artifacts.eigenvalues)
        artifact_path = run_dir / f'decomposition_{noise_mode}_{_safe_value(noise_value)}.pt'
        torch.save(artifacts.to_payload(), artifact_path)

        row = {
            'noise_mode': noise_mode,
            'noise_value': float(noise_value),
            'epochs': resolved_epochs,
            'eval_split': eval_split,
            'eval_loss': float(eval_loss),
            'eval_acc': float(eval_acc),
            'mean_effective_rank': float(effective_rank.mean().item()),
            'max_effective_rank': float(effective_rank.max().item()),
            'decomposition': str(artifact_path),
        }
        metrics_rows.append(row)
        artifacts_by_value.append(artifacts)
        print(f'  {eval_split}_acc={eval_acc:.4f}')

    metrics_path = run_dir / 'noise_sweep_metrics.csv'
    pd.DataFrame(metrics_rows).to_csv(metrics_path, index=False)
    plot_paths = plot_noise_sweep_eigenvectors(
        output_dir=run_dir,
        dataset_name=dataset_name,
        input_shape=_input_shape(config),
        noise_mode=noise_mode,
        noise_values=noise_values,
        artifacts_by_value=artifacts_by_value,
        metrics_rows=metrics_rows,
    )
    summary_path = run_dir / 'summary.json'
    write_json(
        summary_path,
        {
            'config': str(config_path),
            'dataset': dataset_name,
            'noise_mode': noise_mode,
            'noise_values': noise_values,
            'epochs': resolved_epochs,
            'eval_split': eval_split,
            'device': str(device),
            'clamp_unit': clamp_unit,
            'metrics_path': str(metrics_path),
            'plots': plot_paths,
            'rows': metrics_rows,
        },
    )

    return {'run_dir': run_dir, 'metrics_path': metrics_path, 'summary_path': summary_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Run Task B noise sweep eigenspectrum analysis.')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/baselines/mnist_baseline.yaml'),
        help='Path to the baseline YAML config.',
    )
    parser.add_argument('--mode', default='norm', choices=['norm', 'std'], help='Noise parameterization.')
    parser.add_argument(
        '--values',
        nargs='+',
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
        help='Noise values to sweep.',
    )
    parser.add_argument('--epochs', type=int, default=None, help='Override training epochs per noise value.')
    parser.add_argument('--eval-split', default='val', choices=['val', 'test'], help='Evaluation split to report.')
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('results/analysis/noise_sweep'),
        help='Directory for sweep artifacts.',
    )
    parser.add_argument(
        '--clamp',
        default='auto',
        choices=['auto', 'unit', 'none'],
        help='Clamp noisy inputs to [0, 1]. Auto clamps MNIST/Fashion-MNIST only.',
    )
    parser.add_argument('--max-train-batches', type=int, default=None, help='Debug limiter for training batches.')
    parser.add_argument('--max-eval-batches', type=int, default=None, help='Debug limiter for eval batches.')
    parser.add_argument('--num-workers', type=int, default=None, help='Override DataLoader workers.')
    parser.add_argument('--device', default=None, help='Override train.device, e.g. cpu, cuda, mps, or auto.')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.num_workers is not None:
        config.setdefault('train', {})['num_workers'] = int(args.num_workers)
        config['train']['pin_memory'] = False
    if args.device is not None:
        config.setdefault('train', {})['device'] = args.device

    artifacts = run_noise_sweep(
        config,
        config_path=args.config,
        noise_mode=args.mode,
        noise_values=list(args.values),
        epochs=args.epochs,
        eval_split=args.eval_split,
        output_dir=args.output_dir,
        clamp=args.clamp,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
    print('Noise sweep complete.')
    for key, value in artifacts.items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    main()
