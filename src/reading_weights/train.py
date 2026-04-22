from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from reading_weights.data import build_image_dataloaders
from reading_weights.model import build_image_classifier
from reading_weights.utils import ensure_dir, resolve_device, set_seed, timestamp, write_json


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for batch_idx, (x, y) in enumerate(loader, start=1):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(dim=-1) == y).sum().item()
        total_examples += y.size(0)

        if max_batches is not None and batch_idx >= max_batches:
            break

    return total_loss / total_examples, total_correct / total_examples


def train_image_experiment(config: dict) -> dict[str, Path]:
    set_seed(int(config['seed']))
    config.setdefault('train', {})
    config['train'].setdefault('split_seed', int(config['seed']))

    dataset_bundle = build_image_dataloaders(config['dataset'], config['train'])
    device = resolve_device(config['train'].get('device', 'auto'))
    max_train_batches = config['train'].get('max_train_batches')
    max_eval_batches = config['train'].get('max_eval_batches')
    max_train_batches = None if max_train_batches in (None, 0) else int(max_train_batches)
    max_eval_batches = None if max_eval_batches in (None, 0) else int(max_eval_batches)
    model = build_image_classifier(config['model'], seed=int(config['seed'])).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=float(config['train']['lr']),
        weight_decay=float(config['train']['wd']),
    )
    # CosineAnnealingLR here is epoch-based by design; max_train_batches is only
    # a smoke-test/debug limiter and does not redefine the scheduler timescale.
    scheduler = CosineAnnealingLR(optimizer, T_max=int(config['train']['epochs']))

    run_name = f"{config['experiment_name']}_{timestamp()}"
    metrics_dir = ensure_dir(config.get('output_dir', 'results/metrics'))
    checkpoint_dir = ensure_dir(config.get('checkpoint_dir', 'checkpoints'))
    run_dir = ensure_dir(metrics_dir / run_name)

    history: list[dict[str, float | int]] = []
    best_val_acc = -1.0
    best_checkpoint_path = checkpoint_dir / f'{run_name}.pt'
    latest_checkpoint_path = checkpoint_dir / f'{run_name}_latest.pt'

    for epoch in tqdm(range(1, int(config['train']['epochs']) + 1), desc='training'):
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
        },
    )

    return {
        'run_dir': run_dir,
        'metrics_path': metrics_path,
        'best_checkpoint_path': best_checkpoint_path,
        'latest_checkpoint_path': latest_checkpoint_path,
    }
