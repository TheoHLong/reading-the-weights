#!/usr/bin/env python
"""Reproduce the Figure 7 adversarial-mask experiment.

The implementation follows the official image model's decomposition contract:
single-layer bilinear checkpoints are decomposed into input-space eigenvectors,
then Figure 7 masks are built as rows of the pseudoinverse of the top positive
eigenvector frame.  The paper baseline is a random permutation of each mask.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import nn

from _bootstrap import ensure_project_on_path

ensure_project_on_path()

from src.adversarial import (
    AdversarialMasks,
    AttackEvaluation,
    apply_spatial_mask,
    compute_adversarial_masks,
    compute_low_activity_mask,
    compute_permuted_masks,
    evaluate_attacks,
)
from src.data import build_image_dataloaders
from src.model import build_image_classifier
from src.utils import ensure_dir, load_checkpoint, resolve_device


COLORS = {
    'adversarial': (31, 119, 180),
    'eigvec': (44, 160, 44),
    'permuted': (214, 39, 40),
    'clean': (110, 110, 110),
    'grid': (225, 225, 225),
    'axis': (30, 30, 30),
}


def _parse_magnitudes(spec: str) -> list[float]:
    spec = spec.strip()
    if spec.startswith('linspace:'):
        _, lo, hi, n = spec.split(':')
        return torch.linspace(float(lo), float(hi), int(n)).tolist()
    return [float(tok) for tok in spec.split(',') if tok.strip()]


def _parse_int_list(spec: str | None) -> list[int] | None:
    if spec is None or spec.strip().lower() == 'all':
        return None
    values: list[int] = []
    for chunk in spec.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            lo, hi = chunk.split('-', 1)
            values.extend(range(int(lo), int(hi) + 1))
        else:
            values.append(int(chunk))
    return values


def _build_model(payload: dict, device: torch.device) -> nn.Module:
    config = payload.get('config')
    if config is None or 'model' not in config:
        raise ValueError(
            'Checkpoint must contain a bilinear student config. Teacher '
            'checkpoints are not supported by Figure 7 decomposition.'
        )
    model = build_image_classifier(config['model'], seed=int(config['seed']))
    model.load_state_dict(payload['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    bbox = draw.textbbox((0, 0), text, font=font)
    return bbox[2] - bbox[0], bbox[3] - bbox[1]


def _to_image_array(vec: torch.Tensor, input_shape: tuple[int, int, int]) -> np.ndarray:
    channels, height, width = input_shape
    arr = vec.detach().float().cpu().view(channels, height, width).numpy()
    if channels == 1:
        return arr[0]
    return np.transpose(arr, (1, 2, 0))


def _signed_to_rgb(vec: torch.Tensor, input_shape: tuple[int, int, int]) -> Image.Image:
    arr = _to_image_array(vec, input_shape)
    max_abs = float(np.percentile(np.abs(arr), 99.5))
    if max_abs <= 1e-12:
        max_abs = float(np.abs(arr).max() or 1.0)
    scaled = np.clip(arr / max_abs, -1.0, 1.0)

    if arr.ndim == 2:
        pos = np.clip(scaled, 0.0, 1.0)
        neg = np.clip(-scaled, 0.0, 1.0)
        rgb = np.empty((*scaled.shape, 3), dtype=np.uint8)
        rgb[..., 0] = (255 * (1.0 - neg)).astype(np.uint8)
        rgb[..., 1] = (255 * (1.0 - np.maximum(pos, neg))).astype(np.uint8)
        rgb[..., 2] = (255 * (1.0 - pos)).astype(np.uint8)
    else:
        rgb = ((scaled + 1.0) * 127.5).astype(np.uint8)
    return Image.fromarray(rgb, mode='RGB')


def _input_to_rgb(vec: torch.Tensor, input_shape: tuple[int, int, int]) -> Image.Image:
    arr = _to_image_array(vec, input_shape)
    if arr.ndim == 2:
        gray = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        return Image.fromarray(gray, mode='L').convert('RGB')

    lo = float(arr.min())
    hi = float(arr.max())
    if hi - lo < 1e-12:
        norm = np.zeros_like(arr)
    else:
        norm = (arr - lo) / (hi - lo)
    return Image.fromarray((norm * 255).astype(np.uint8), mode='RGB')


def _resize_cell(img: Image.Image, size: int) -> Image.Image:
    resampling = getattr(Image, 'Resampling', Image).NEAREST
    return img.resize((size, size), resampling)


def _save_grid(
    rows: list[list[Image.Image | None]],
    row_labels: list[str],
    col_labels: list[str],
    title: str,
    path: Path,
    *,
    cell: int = 92,
) -> None:
    font = _font()
    label_w = 120
    title_h = 34
    header_h = 34
    row_gap = 10
    col_gap = 12
    width = label_w + len(col_labels) * cell + (len(col_labels) - 1) * col_gap + 18
    height = title_h + header_h + len(rows) * cell + (len(rows) - 1) * row_gap + 18
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)

    tw, _ = _text_size(draw, title, font)
    draw.text(((width - tw) // 2, 10), title, fill='black', font=font)

    y0 = title_h
    for col, label in enumerate(col_labels):
        x = label_w + col * (cell + col_gap)
        lw, _ = _text_size(draw, label, font)
        draw.text((x + (cell - lw) // 2, y0 + 8), label, fill='black', font=font)

    y = title_h + header_h
    for row_idx, row in enumerate(rows):
        label = row_labels[row_idx]
        draw.text((8, y + cell // 2 - 6), label, fill='black', font=font)
        for col, img in enumerate(row):
            x = label_w + col * (cell + col_gap)
            draw.rectangle((x - 1, y - 1, x + cell, y + cell), outline=(60, 60, 60))
            if img is None:
                draw.text((x + 22, y + cell // 2 - 6), 'no flip', fill=(120, 120, 120), font=font)
            else:
                canvas.paste(_resize_cell(img, cell), (x, y))
        y += cell + row_gap

    canvas.save(path)


def _line_points(xs: np.ndarray, ys: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    left, top, right, bottom = box
    x_min = float(xs.min())
    x_max = float(xs.max())
    if x_max - x_min < 1e-12:
        x_max = x_min + 1.0
    pts = []
    for x, y in zip(xs, ys):
        px = left + int((float(x) - x_min) / (x_max - x_min) * (right - left))
        py = bottom - int(np.clip(float(y), 0.0, 1.0) * (bottom - top))
        pts.append((px, py))
    return pts


def _draw_polyline(draw: ImageDraw.ImageDraw, pts: list[tuple[int, int]], color: tuple[int, int, int]) -> None:
    if len(pts) >= 2:
        draw.line(pts, fill=color, width=3)
    elif pts:
        x, y = pts[0]
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    xs: np.ndarray,
    series: list[tuple[str, np.ndarray, tuple[int, int, int]]],
    title: str,
    y_label: str,
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = box
    draw.rectangle(box, outline=COLORS['axis'])
    for frac in [0.25, 0.5, 0.75]:
        y = bottom - int(frac * (bottom - top))
        draw.line((left, y, right, y), fill=COLORS['grid'])
        draw.text((left - 30, y - 6), f'{frac:.2g}', fill='black', font=font)
    draw.text((left - 30, bottom - 8), '0', fill='black', font=font)
    draw.text((left - 30, top - 6), '1', fill='black', font=font)

    tw, _ = _text_size(draw, title, font)
    draw.text((left + (right - left - tw) // 2, top - 24), title, fill='black', font=font)
    draw.text((left, bottom + 10), 'epsilon', fill='black', font=font)
    draw.text((left - 46, top + 6), y_label, fill='black', font=font)

    legend_x = left + 12
    legend_y = top + 10
    for label, ys, color in series:
        pts = _line_points(xs, ys, box)
        _draw_polyline(draw, pts, color)
        draw.line((legend_x, legend_y + 6, legend_x + 22, legend_y + 6), fill=color, width=3)
        draw.text((legend_x + 28, legend_y), label, fill='black', font=font)
        legend_y += 16


def _save_success_curves(
    evals: dict[str, AttackEvaluation],
    path: Path,
    dataset_name: str,
) -> None:
    font = _font()
    width, height = 980, 400
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    title = f'Figure 7 attack curves - {dataset_name}'
    tw, _ = _text_size(draw, title, font)
    draw.text(((width - tw) // 2, 12), title, fill='black', font=font)

    eps = evals['adversarial'].magnitudes.numpy()
    clean = evals['adversarial'].clean_acc
    clean_acc = np.full_like(eps, clean, dtype=np.float64)
    clean_err = np.full_like(eps, 1.0 - clean, dtype=np.float64)

    _draw_panel(
        draw,
        (70, 70, 460, 320),
        eps,
        [
            ('original', clean_acc, COLORS['clean']),
            ('adversarial', evals['adversarial'].accuracy_mean.numpy(), COLORS['adversarial']),
            ('permuted', evals['permuted'].accuracy_mean.numpy(), COLORS['permuted']),
        ],
        'accuracy under attack',
        'accuracy',
        font,
    )
    _draw_panel(
        draw,
        (560, 70, 950, 320),
        eps,
        [
            ('original error', clean_err, COLORS['clean']),
            ('adversarial', evals['adversarial'].target_success_mean.numpy(), COLORS['adversarial']),
            ('permuted', evals['permuted'].target_success_mean.numpy(), COLORS['permuted']),
        ],
        'misclassification as target',
        'rate',
        font,
    )
    canvas.save(path)


@torch.no_grad()
def _find_successful_examples(
    model: nn.Module,
    loader,
    masks: torch.Tensor,
    target_classes: torch.Tensor,
    epsilon: float,
    device: torch.device,
    clip_range: tuple[float, float] | None,
    input_shape: tuple[int, int, int],
) -> list[tuple[torch.Tensor, torch.Tensor, int, int] | None]:
    examples: list[tuple[torch.Tensor, torch.Tensor, int, int] | None] = [None] * masks.shape[0]
    needed = set(range(masks.shape[0]))
    masks = masks.to(device)
    targets = target_classes.to(device)
    for x, y in loader:
        if not needed:
            break
        x = x.to(device)
        y = y.to(device)
        flat = x.flatten(start_dim=1)
        for idx in list(needed):
            target = int(targets[idx].item())
            adv_flat = flat + epsilon * masks[idx].view(1, -1)
            if clip_range is not None:
                adv_flat = adv_flat.clamp(*clip_range)
            pred = model(adv_flat.view(-1, *input_shape)).argmax(dim=-1)
            eligible = y != target
            hit = ((pred == target) & eligible).nonzero(as_tuple=False)
            if hit.numel() > 0:
                j = int(hit[0].item())
                examples[idx] = (
                    x[j].detach().cpu(),
                    adv_flat[j].detach().cpu(),
                    int(y[j].item()),
                    target,
                )
                needed.remove(idx)
    return examples


def _figure_indices(
    masks: AdversarialMasks,
    figure_classes: list[int] | None,
    figure_ranks: list[int] | None,
    max_rows: int,
) -> list[int]:
    targets = masks.target_classes.cpu()
    ranks = masks.eigen_ranks.cpu()
    if figure_classes is None:
        if (targets == 3).any():
            figure_classes = [3]
        else:
            figure_classes = [int(targets[0].item())]
    if figure_ranks is None:
        figure_ranks = sorted(set(int(x) for x in ranks.tolist()))

    rows = []
    for idx, (target, rank) in enumerate(zip(targets.tolist(), ranks.tolist())):
        if target in figure_classes and rank in figure_ranks:
            rows.append(idx)
    return rows[:max_rows]


def _save_fig7_panel(
    masks: AdversarialMasks,
    masks_adv: torch.Tensor,
    masks_permuted: torch.Tensor,
    examples: list[tuple[torch.Tensor, torch.Tensor, int, int] | None],
    row_indices: list[int],
    input_shape: tuple[int, int, int],
    path: Path,
    dataset_name: str,
    epsilon: float,
) -> None:
    rows: list[list[Image.Image | None]] = []
    labels: list[str] = []
    for idx in row_indices:
        example = examples[idx]
        adv_example = None if example is None else _input_to_rgb(example[1], input_shape)
        rows.append([
            _signed_to_rgb(masks.masks_eigvec[idx], input_shape),
            adv_example,
            _signed_to_rgb(masks_adv[idx], input_shape),
            _signed_to_rgb(masks_permuted[idx], input_shape),
        ])
        labels.append(f'target {int(masks.target_classes[idx])}, rank {int(masks.eigen_ranks[idx])}')

    _save_grid(
        rows,
        labels,
        ['eigenvector', f'misclassified eps={epsilon:g}', 'adversarial mask', 'permuted mask'],
        f'Figure 7 panel - {dataset_name}',
        path,
    )


def _save_mask_grid(
    masks: AdversarialMasks,
    masks_adv: torch.Tensor,
    masks_permuted: torch.Tensor,
    row_indices: list[int],
    input_shape: tuple[int, int, int],
    path: Path,
    dataset_name: str,
) -> None:
    rows = []
    labels = []
    for idx in row_indices:
        rows.append([
            _signed_to_rgb(masks.masks_eigvec[idx], input_shape),
            _signed_to_rgb(masks_adv[idx], input_shape),
            _signed_to_rgb(masks_permuted[idx], input_shape),
        ])
        labels.append(f'target {int(masks.target_classes[idx])}, rank {int(masks.eigen_ranks[idx])}')
    _save_grid(rows, labels, ['eigenvector', 'adversarial', 'permuted'], f'Masks - {dataset_name}', path)


def _evaluation_to_dict(label: str, ev: AttackEvaluation) -> dict:
    return {
        'method': label,
        'magnitudes': ev.magnitudes.tolist(),
        'target_success_by_mask': ev.target_success_by_mask.tolist(),
        'target_success_mean': ev.target_success_mean.tolist(),
        'accuracy_by_mask': ev.accuracy_by_mask.tolist(),
        'accuracy_mean': ev.accuracy_mean.tolist(),
        'clean_acc': ev.clean_acc,
        'num_examples': ev.num_examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Task F - Figure 7 adversarial masks.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('results/adversarial'))
    parser.add_argument('--magnitudes', default='linspace:0.0:5.0:11')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--target-classes', default='all', help='Comma list, range, or "all".')
    parser.add_argument('--basis-size', type=int, default=10)
    parser.add_argument('--attack-ranks', type=int, default=3)
    parser.add_argument('--example-eps', type=float, default=3.0)
    parser.add_argument('--max-batches', type=int, default=0)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--random-seed', type=int, default=42)
    parser.add_argument('--figure-classes', default=None, help='Default: class 3 if present, else first target.')
    parser.add_argument('--figure-ranks', default='1,2,3')
    parser.add_argument('--figure-max-rows', type=int, default=6)
    parser.add_argument('--clip', action='store_true', help='Clip perturbed inputs to [0, 1].')
    parser.add_argument(
        '--low-activity-mask',
        action='store_true',
        help='Restrict masks to pixels active on fewer than --activity-threshold train samples.',
    )
    parser.add_argument('--activity-threshold', type=float, default=0.01)
    parser.add_argument('--pixel-threshold', type=float, default=0.0)
    args = parser.parse_args()

    payload = load_checkpoint(args.checkpoint, map_location='cpu')
    config = payload.get('config') or {}
    config.setdefault('train', {})
    config['train'].setdefault('split_seed', int(config.get('seed', 42)))
    config['train']['num_workers'] = int(args.num_workers)
    if str(args.device) == 'cpu':
        config['train']['pin_memory'] = False

    dataset_name = config['dataset']['name']
    input_shape = (
        int(config['dataset']['channels']),
        int(config['dataset']['image_size']),
        int(config['dataset']['image_size']),
    )
    d_input = int(config['model']['d_input'])

    device = resolve_device(args.device)
    model = _build_model(payload, device)
    bundle = build_image_dataloaders(config['dataset'], config['train'])
    max_batches = None if args.max_batches in (0, None) else int(args.max_batches)
    clip_range = (0.0, 1.0) if args.clip else None
    target_classes = _parse_int_list(args.target_classes)

    masks = compute_adversarial_masks(
        model,
        target_classes=target_classes,
        basis_size=int(args.basis_size),
        attack_ranks=int(args.attack_ranks),
    )
    masks_adv = masks.masks_pinv.to(device)
    masks_eigvec = masks.masks_eigvec.to(device)

    low_activity_keep = None
    if args.low_activity_mask:
        low_activity_keep = compute_low_activity_mask(
            bundle.train_loader,
            d_input=d_input,
            device=device,
            active_threshold=float(args.activity_threshold),
            pixel_threshold=float(args.pixel_threshold),
            max_batches=max_batches,
        )
        masks_adv = apply_spatial_mask(masks_adv, low_activity_keep)
        masks_eigvec = apply_spatial_mask(masks_eigvec, low_activity_keep)

    masks_permuted = compute_permuted_masks(masks_adv, seed=int(args.random_seed))
    target_tensor = masks.target_classes.to(device)
    magnitudes = _parse_magnitudes(args.magnitudes)

    evals = {
        'adversarial': evaluate_attacks(
            model,
            bundle.test_loader,
            masks_adv,
            target_tensor,
            magnitudes,
            device,
            clip_range=clip_range,
            input_shape=input_shape,
            max_batches=max_batches,
        ),
        'eigvec': evaluate_attacks(
            model,
            bundle.test_loader,
            masks_eigvec,
            target_tensor,
            magnitudes,
            device,
            clip_range=clip_range,
            input_shape=input_shape,
            max_batches=max_batches,
        ),
        'permuted': evaluate_attacks(
            model,
            bundle.test_loader,
            masks_permuted,
            target_tensor,
            magnitudes,
            device,
            clip_range=clip_range,
            input_shape=input_shape,
            max_batches=max_batches,
        ),
    }

    out_dir = ensure_dir(args.output_dir / args.checkpoint.stem)
    torch.save(
        {
            'masks_adversarial': masks_adv.detach().cpu(),
            'masks_pinv_raw': masks.masks_pinv_raw.detach().cpu(),
            'masks_eigvec': masks_eigvec.detach().cpu(),
            'masks_permuted': masks_permuted.detach().cpu(),
            'target_classes': masks.target_classes.detach().cpu(),
            'eigen_ranks': masks.eigen_ranks.detach().cpu(),
            'eigvals_selected': masks.eigvals_selected.detach().cpu(),
            'basis_vectors': masks.basis_vectors.detach().cpu(),
            'basis_eigvals': masks.basis_eigvals.detach().cpu(),
            'low_activity_keep': None if low_activity_keep is None else low_activity_keep.detach().cpu(),
            'magnitudes': torch.tensor(magnitudes),
            'input_shape': torch.tensor(input_shape),
        },
        out_dir / 'masks.pt',
    )

    metrics = {
        'checkpoint': str(args.checkpoint),
        'dataset': dataset_name,
        'input_shape': list(input_shape),
        'val_acc': float(payload.get('metrics', {}).get('val_acc', float('nan'))),
        'basis_size': int(args.basis_size),
        'attack_ranks': int(args.attack_ranks),
        'target_classes': masks.target_classes.cpu().tolist(),
        'eigen_ranks': masks.eigen_ranks.cpu().tolist(),
        'random_seed': int(args.random_seed),
        'low_activity_mask': bool(args.low_activity_mask),
        'clip_range': None if clip_range is None else list(clip_range),
        **{method: _evaluation_to_dict(method, ev) for method, ev in evals.items()},
    }
    (out_dir / 'attack_metrics.json').write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding='utf-8')

    figure_classes = _parse_int_list(args.figure_classes)
    figure_ranks = _parse_int_list(args.figure_ranks)
    row_indices = _figure_indices(masks, figure_classes, figure_ranks, int(args.figure_max_rows))
    if row_indices:
        examples = _find_successful_examples(
            model,
            bundle.test_loader,
            masks_adv,
            target_tensor,
            epsilon=float(args.example_eps),
            device=device,
            clip_range=clip_range,
            input_shape=input_shape,
        )
        _save_fig7_panel(
            masks,
            masks_adv.detach().cpu(),
            masks_permuted.detach().cpu(),
            examples,
            row_indices,
            input_shape,
            out_dir / 'fig7_panel.png',
            dataset_name,
            float(args.example_eps),
        )
        _save_mask_grid(
            masks,
            masks_adv.detach().cpu(),
            masks_permuted.detach().cpu(),
            row_indices,
            input_shape,
            out_dir / 'masks.png',
            dataset_name,
        )
    _save_success_curves(evals, out_dir / 'success_curves.png', dataset_name)

    print(f'Figure 7 artifacts saved to {out_dir}')
    print(f'  clean_acc={evals["adversarial"].clean_acc:.4f}')
    for method, ev in evals.items():
        peak = float(ev.target_success_mean.max())
        final_acc = float(ev.accuracy_mean[-1])
        print(f'  {method:11s} peak target-rate={peak:.4f}, final accuracy={final_acc:.4f}')


if __name__ == '__main__':
    main()
