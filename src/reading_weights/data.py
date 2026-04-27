from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


DATASETS = {
    'cifar10': datasets.CIFAR10,
    'mnist': datasets.MNIST,
    'fashion_mnist': datasets.FashionMNIST,
}

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    input_shape: tuple[int, int, int]
    num_classes: int


def build_dataset_transforms(dataset_name: str) -> tuple[transforms.Compose, transforms.Compose]:
    if dataset_name == 'cifar10':
        eval_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
        return train_transform, eval_transform

    base_transform = transforms.Compose([transforms.ToTensor()])
    return base_transform, base_transform


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

    train_transform, eval_transform = build_dataset_transforms(dataset_name)

    if dataset_name == 'cifar10':
        train_dataset_aug = dataset_cls(
            root=root,
            train=True,
            download=True,
            transform=train_transform,
        )
        train_dataset_eval = dataset_cls(
            root=root,
            train=True,
            download=True,
            transform=eval_transform,
        )
        test_dataset = dataset_cls(
            root=root,
            train=False,
            download=True,
            transform=eval_transform,
        )
        num_examples = len(train_dataset_aug)
    else:
        train_dataset = dataset_cls(root=root, train=True, download=True, transform=train_transform)
        test_dataset = dataset_cls(root=root, train=False, download=True, transform=eval_transform)
        num_examples = len(train_dataset)

    num_val = int(round(num_examples * val_fraction))
    if num_val <= 0 or num_val >= num_examples:
        raise ValueError(f'val_fraction must leave non-empty train and val splits, got {val_fraction}')
    num_train = num_examples - num_val

    split_generator = torch.Generator().manual_seed(split_seed)
    if dataset_name == 'cifar10':
        indices = torch.randperm(num_examples, generator=split_generator).tolist()
        train_idx = indices[:num_train]
        val_idx = indices[num_train:]
        train_subset = Subset(train_dataset_aug, train_idx)
        val_subset = Subset(train_dataset_eval, val_idx)
    else:
        train_subset, val_subset = torch.utils.data.random_split(
            train_dataset,
            [num_train, num_val],
            generator=split_generator,
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
