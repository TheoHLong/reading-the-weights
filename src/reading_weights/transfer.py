from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from reading_weights.data import build_image_dataloaders
from reading_weights.guide import load_frozen_teacher
from reading_weights.model import build_image_classifier
from reading_weights.train import evaluate
from reading_weights.utils import ensure_dir, resolve_device, set_seed, timestamp, write_json


def kd_loss(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    alpha: float,
    temperature: float,
) -> Tensor:
    ce = F.cross_entropy(student_logits, labels)
    kl = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction='batchmean',
    )
    return alpha * ce + (1.0 - alpha) * (temperature**2) * kl


def train_kd_experiment(config: dict) -> dict[str, Path]:
    set_seed(int(config['seed']))
    config.setdefault('train', {})
    config['train'].setdefault('split_seed', int(config['seed']))

    transfer_cfg = config.get('transfer', {})
    if transfer_cfg.get('method') != 'kd':
        raise ValueError("transfer.method must be 'kd' for train_kd_experiment().")

    dataset_bundle = build_image_dataloaders(config['dataset'], config['train'])
    device = resolve_device(config['train'].get('device', 'auto'))
    max_train_batches = config['train'].get('max_train_batches')
    max_eval_batches = config['train'].get('max_eval_batches')
    max_train_batches = None if max_train_batches in (None, 0) else int(max_train_batches)
    max_eval_batches = None if max_eval_batches in (None, 0) else int(max_eval_batches)
    alpha = float(transfer_cfg.get('alpha', 0.5))
    temperature = float(transfer_cfg.get('temperature', 4.0))

    teacher = load_frozen_teacher(transfer_cfg['teacher_checkpoint'], device)
    student = build_image_classifier(config['model'], seed=int(config['seed'])).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        student.parameters(),
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

    for epoch in tqdm(range(1, int(config['train']['epochs']) + 1), desc='training'):
        student.train()
        running_loss = 0.0
        running_correct = 0
        running_examples = 0

        for batch_idx, (x, y) in enumerate(dataset_bundle.train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                teacher_logits = teacher(x)
            student_logits = student(x)
            loss = kd_loss(
                student_logits,
                teacher_logits,
                y,
                alpha=alpha,
                temperature=temperature,
            )
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * y.size(0)
            running_correct += (student_logits.argmax(dim=-1) == y).sum().item()
            running_examples += y.size(0)

            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

        scheduler.step()

        train_loss = running_loss / running_examples
        train_acc = running_correct / running_examples
        val_loss, val_acc = evaluate(
            student,
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
            'model_state_dict': student.state_dict(),
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
            'teacher_checkpoint': str(transfer_cfg['teacher_checkpoint']),
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
