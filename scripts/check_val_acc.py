#!/usr/bin/env python
"""Print epoch / val_acc / val_loss for one or more checkpoints.

Used to verify a baseline + SQS pair converged to comparable accuracy before
trusting the eigendecomposition comparison. SQS paper Table 1 reports for n=1
shallow nets:
    MNIST  : Bilinear ~0.978 / SQS ~0.979
    FMNIST : Bilinear ~0.846 / SQS ~0.862

Example
-------
    python scripts/check_val_acc.py \\
        checkpoints/mnist_paper_bilinear_*.pt \\
        checkpoints/mnist_paper_sqs_*.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_on_path

ensure_project_on_path()

from src.utils import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Print val_acc / val_loss for checkpoints.')
    parser.add_argument(
        'checkpoints',
        type=Path,
        nargs='+',
        help='One or more checkpoint .pt files. Glob expansion happens in your shell.',
    )
    parser.add_argument(
        '--skip-latest',
        action='store_true',
        help='Skip files whose stem ends with _latest (training-time, not best).',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    rows: list[tuple[str, str, str, str, str, str]] = []
    for path in args.checkpoints:
        if args.skip_latest and path.stem.endswith('_latest'):
            continue
        if not path.is_file():
            rows.append((path.name, '!missing', '', '', '', ''))
            continue
        try:
            payload = load_checkpoint(path, map_location='cpu')
        except Exception as exc:  # noqa: BLE001
            rows.append((path.name, f'!error: {exc.__class__.__name__}', '', '', '', ''))
            continue

        cfg = payload.get('config', {})
        model_cfg = cfg.get('model', {})
        train_cfg = cfg.get('train', {})
        metrics = payload.get('metrics', {})

        gate = model_cfg.get('gate') or 'none'
        n_layer = model_cfg.get('n_layer', '?')
        epoch = payload.get('epoch', '?')
        val_acc = metrics.get('val_acc')
        val_loss = metrics.get('val_loss')
        noise = train_cfg.get('input_noise_std', 0.0)

        rows.append((
            path.name,
            f'{gate}',
            f'n={n_layer}',
            f'epoch={epoch}',
            f'val_acc={val_acc:.4f}' if isinstance(val_acc, (int, float)) else 'val_acc=?',
            f'val_loss={val_loss:.4f}' if isinstance(val_loss, (int, float)) else 'val_loss=?',
        ))
        # Stash extras for trailing print
        rows[-1] = rows[-1] + (f'wd={train_cfg.get("wd", "?")} noise={noise}',)

    name_w = max((len(r[0]) for r in rows), default=20)
    for r in rows:
        name = r[0].ljust(name_w)
        rest = '  '.join(c for c in r[1:] if c)
        print(f'{name}  {rest}')

    # Quick summary if exactly two passed (typical bilinear vs sqs)
    accs = []
    for r in rows:
        for cell in r[1:]:
            if cell.startswith('val_acc=') and not cell.endswith('?'):
                accs.append(float(cell.split('=')[1]))
                break
    if len(accs) == 2:
        diff = accs[1] - accs[0]
        print(f'\nval_acc delta (second - first) = {diff:+.4f}')
        if abs(diff) > 0.02:
            print('  WARNING: |delta| > 0.02 — eigendecomposition comparison may be unreliable.')


if __name__ == '__main__':
    main()
