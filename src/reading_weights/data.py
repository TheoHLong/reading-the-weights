from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATASETS = {
    'mnist': datasets.MNIST,
    'fashion_mnist': datasets.FashionMNIST,
}


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    input_shape: tuple[int, int, int]
    num_classes: int


def build_image_dataloaders(dataset_cfg: dict, train_cfg: dict) -> DatasetBundle:
    dataset_name = dataset_cfg['name']
    if dataset_name not in DATASETS:
        raise ValueError(f'Unsupported dataset for Task A: {dataset_name}')

    dataset_cls = DATASETS[dataset_name]
    root = dataset_cfg.get('root', 'data/raw')
    batch_size = int(train_cfg['batch_size'])
    num_workers = int(train_cfg.get('num_workers', 2))
    pin_memory = bool(train_cfg.get('pin_memory', True))
    val_fraction = float(train_cfg.get('val_fraction', 0.1))
    split_seed = int(train_cfg.get('split_seed', 42))

    transform = transforms.ToTensor()
    train_dataset = dataset_cls(root=root, train=True, download=True, transform=transform)
    test_dataset = dataset_cls(root=root, train=False, download=True, transform=transform)

    num_examples = len(train_dataset)
    num_val = int(round(num_examples * val_fraction))
    if num_val <= 0 or num_val >= num_examples:
        raise ValueError(f'val_fraction must leave non-empty train and val splits, got {val_fraction}')
    num_train = num_examples - num_val

    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset,
        [num_train, num_val],
        generator=torch.Generator().manual_seed(split_seed),
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return DatasetBundle(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_shape=(
            int(dataset_cfg['channels']),
            int(dataset_cfg['image_size']),
            int(dataset_cfg['image_size']),
        ),
        num_classes=int(dataset_cfg['num_classes']),
    )
