#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_src_on_path

ensure_src_on_path()

from reading_weights.config import load_config
from reading_weights.transfer import train_kd_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a bilinear student with knowledge distillation.')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/cifar10_kd.yaml'),
        help='Path to the YAML config file.',
    )
    args = parser.parse_args()

    artifacts = train_kd_experiment(load_config(args.config))
    print('KD training complete.')
    for key, value in artifacts.items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    main()
