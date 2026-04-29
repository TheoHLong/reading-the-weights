#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

from _bootstrap import ensure_project_on_path

ensure_project_on_path()

from src.config import load_config
from src.train import train_image_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description='Train a Task A bilinear baseline.')
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('configs/baselines/mnist_baseline.yaml'),
        help='Path to the YAML config file.',
    )
    args = parser.parse_args()

    artifacts = train_image_experiment(load_config(args.config))
    print('Training complete.')
    for key, value in artifacts.items():
        print(f'{key}: {value}')


if __name__ == '__main__':
    main()
