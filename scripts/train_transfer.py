#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_on_path

ensure_project_on_path()

from src.config import load_config
from src.transfer import train_cka_experiment, train_kd_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a bilinear student with transfer guidance.')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/transfer/cifar10_kd.yaml'),
        help='Path to the YAML config file.',
    )
    args = parser.parse_args()

    config = load_config(args.config)
    method = config.get('transfer', {}).get('method')
    if method == 'kd':
        artifacts = train_kd_experiment(config)
    elif method == 'cka':
        artifacts = train_cka_experiment(config)
    else:
        raise ValueError(f"Unknown transfer.method: {method!r}")

    print(f'{method.upper()} training complete.')
    for key, value in artifacts.items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    main()
