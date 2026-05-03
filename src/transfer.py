from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm.auto import tqdm

from src.cka import cka_distance
from src.data import build_image_dataloaders
from src.guide import load_frozen_guide
from src.hooks import ActivationCapture
from src.model import build_image_classifier
from src.train import evaluate
from src.utils import ensure_dir, resolve_device, set_seed, timestamp, write_json


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


def resolve_layer_map(
    transfer_cfg: dict,
    teacher_layers: list[str],
    student_layers: list[str],
) -> dict[str, str]:
    explicit = transfer_cfg.get('layer_map')
    if explicit:
        for teacher_name, student_name in explicit.items():
            if teacher_name not in teacher_layers:
                raise ValueError(f'layer_map references unknown teacher layer: {teacher_name}')
            if student_name not in student_layers:
                raise ValueError(f'layer_map references unknown student layer: {student_name}')
        return dict(explicit)

    n_teacher = len(teacher_layers)
    n_student = len(student_layers)
    if n_student == 0:
        return {}
    step = (n_student - 1) / max(n_teacher - 1, 1)
    return {
        teacher_layers[i]: student_layers[min(round(i * step), n_student - 1)]
        for i in range(n_teacher)
    }


def cka_loss(
    student_acts: dict[str, Tensor],
    teacher_acts: dict[str, Tensor],
    layer_map: dict[str, str],
    reduction: str = 'mean',
) -> Tensor:
    if reduction not in ('mean', 'sum'):
        raise ValueError(f"reduction must be 'mean' or 'sum', got {reduction!r}")

    if not layer_map:
        any_student = next(iter(student_acts.values()))
        return any_student.new_zeros(())

    pair_losses = []
    for teacher_name, student_name in layer_map.items():
        teacher_tensor = teacher_acts[teacher_name].reshape(teacher_acts[teacher_name].size(0), -1)
        student_tensor = student_acts[student_name].reshape(student_acts[student_name].size(0), -1)
        pair_losses.append(cka_distance(student_tensor, teacher_tensor))

    stacked = torch.stack(pair_losses)
    return stacked.mean() if reduction == 'mean' else stacked.sum()


def build_teacher_inputs(inputs: Tensor, transfer_cfg: dict) -> Tensor:
    teacher_input = str(transfer_cfg.get('teacher_input', 'student'))
    if teacher_input == 'student':
        return inputs
    if teacher_input == 'noise':
        return torch.randn_like(inputs)
    raise ValueError(f"teacher_input must be 'student' or 'noise', got {teacher_input!r}")


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

    teacher = load_frozen_guide(config, transfer_cfg, device)
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
            'teacher_source': str(transfer_cfg.get('teacher_source', 'checkpoint')),
            'teacher_checkpoint': str(transfer_cfg.get('teacher_checkpoint', '')),
            'guide_architecture': str(config.get('guide', {}).get('architecture', '')),
            'guide_seed': transfer_cfg.get('guide_seed'),
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


def train_cka_experiment(config: dict) -> dict[str, Path]:
    set_seed(int(config['seed']))
    config.setdefault('train', {})
    config['train'].setdefault('split_seed', int(config['seed']))

    transfer_cfg = config.get('transfer', {})
    if transfer_cfg.get('method') != 'cka':
        raise ValueError("transfer.method must be 'cka' for train_cka_experiment().")

    dataset_bundle = build_image_dataloaders(config['dataset'], config['train'])
    device = resolve_device(config['train'].get('device', 'auto'))
    max_train_batches = config['train'].get('max_train_batches')
    max_eval_batches = config['train'].get('max_eval_batches')
    max_train_batches = None if max_train_batches in (None, 0) else int(max_train_batches)
    max_eval_batches = None if max_eval_batches in (None, 0) else int(max_eval_batches)
    alpha = float(transfer_cfg.get('alpha', 3.0))
    reduction = str(transfer_cfg.get('reduction', 'mean'))
    teacher_layers = list(transfer_cfg.get('teacher_layers', ['layer2', 'layer4']))
    student_layers = list(transfer_cfg.get('student_layers', ['embed', 'blocks.0']))
    early_stop_steps = transfer_cfg.get('early_stop_steps')

    teacher = load_frozen_guide(config, transfer_cfg, device)
    student = build_image_classifier(config['model'], seed=int(config['seed'])).to(device)
    layer_map = resolve_layer_map(transfer_cfg, teacher_layers, student_layers)

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
    global_step = 0

    for epoch in tqdm(range(1, int(config['train']['epochs']) + 1), desc='training'):
        student.train()
        running_total_loss = 0.0
        running_ce_loss = 0.0
        running_cka_loss = 0.0
        running_correct = 0
        running_examples = 0

        for batch_idx, (x, y) in enumerate(dataset_bundle.train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            teacher_x = build_teacher_inputs(x, transfer_cfg)

            optimizer.zero_grad(set_to_none=True)

            with torch.no_grad():
                with ActivationCapture(teacher, teacher_layers) as teacher_capture:
                    _ = teacher(teacher_x)
                teacher_acts = {name: act.detach() for name, act in teacher_capture.activations.items()}

            with ActivationCapture(student, student_layers) as student_capture:
                student_logits = student(x)
                student_acts = dict(student_capture.activations)

            ce = criterion(student_logits, y)
            use_cka = early_stop_steps is None or global_step < int(early_stop_steps)
            if use_cka:
                rep = cka_loss(student_acts, teacher_acts, layer_map, reduction=reduction)
                loss = ce + alpha * rep
            else:
                rep = ce.new_zeros(())
                loss = ce

            loss.backward()
            optimizer.step()
            global_step += 1

            running_total_loss += loss.item() * y.size(0)
            running_ce_loss += ce.item() * y.size(0)
            running_cka_loss += rep.item() * y.size(0)
            running_correct += (student_logits.argmax(dim=-1) == y).sum().item()
            running_examples += y.size(0)

            if max_train_batches is not None and batch_idx >= max_train_batches:
                break

        scheduler.step()

        train_total_loss = running_total_loss / running_examples
        train_ce_loss = running_ce_loss / running_examples
        train_cka_loss = running_cka_loss / running_examples
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
            'train_total_loss': train_total_loss,
            'train_ce_loss': train_ce_loss,
            'train_cka_loss': train_cka_loss,
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
            'teacher_source': str(transfer_cfg.get('teacher_source', 'checkpoint')),
            'teacher_checkpoint': str(transfer_cfg.get('teacher_checkpoint', '')),
            'guide_architecture': str(config.get('guide', {}).get('architecture', '')),
            'guide_seed': transfer_cfg.get('guide_seed'),
            'teacher_input': str(transfer_cfg.get('teacher_input', 'student')),
            'layer_map': layer_map,
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
