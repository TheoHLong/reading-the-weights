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


def build_dataset_transforms(dataset_cfg: dict) -> tuple[transforms.Compose, transforms.Compose]:
    dataset_name = dataset_cfg['name']
    image_size = int(dataset_cfg['image_size'])

    normalize_mean = dataset_cfg.get('normalize_mean')
    normalize_std = dataset_cfg.get('normalize_std')
    if normalize_mean is None and normalize_std is None and dataset_name == 'cifar10':
        normalize_mean = CIFAR10_MEAN
        normalize_std = CIFAR10_STD
    if (normalize_mean is None) != (normalize_std is None):
        raise ValueError('normalize_mean and normalize_std must be provided together')

    normalize = None
    if normalize_mean is not None and normalize_std is not None:
        normalize = transforms.Normalize(normalize_mean, normalize_std)

    train_ops = []
    default_cifar_aug = dataset_name == 'cifar10'
    if bool(dataset_cfg.get('train_random_crop', default_cifar_aug)):
        padding = int(dataset_cfg.get('random_crop_padding', 4))
        train_ops.append(transforms.RandomCrop(image_size, padding=padding))
    if bool(dataset_cfg.get('train_random_horizontal_flip', default_cifar_aug)):
        train_ops.append(transforms.RandomHorizontalFlip())
    train_ops.append(transforms.ToTensor())
    if normalize is not None:
        train_ops.append(normalize)

    eval_ops = [transforms.ToTensor()]
    if normalize is not None:
        eval_ops.append(normalize)

    return transforms.Compose(train_ops), transforms.Compose(eval_ops)


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

    train_transform, eval_transform = build_dataset_transforms(dataset_cfg)
    train_dataset_aug = dataset_cls(root=root, train=True, download=True, transform=train_transform)
    train_dataset_eval = dataset_cls(root=root, train=True, download=True, transform=eval_transform)
    test_dataset = dataset_cls(root=root, train=False, download=True, transform=eval_transform)
    num_examples = len(train_dataset_aug)

    num_val = int(round(num_examples * val_fraction))
    if num_val <= 0 or num_val >= num_examples:
        raise ValueError(f'val_fraction must leave non-empty train and val splits, got {val_fraction}')
    num_train = num_examples - num_val

    split_generator = torch.Generator().manual_seed(split_seed)
    indices = torch.randperm(num_examples, generator=split_generator).tolist()
    train_idx = indices[:num_train]
    val_idx = indices[num_train:]
    train_subset = Subset(train_dataset_aug, train_idx)
    val_subset = Subset(train_dataset_eval, val_idx)

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
