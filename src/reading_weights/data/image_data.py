from __future__ import annotations

from dataclasses import dataclass

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


DATASETS = {
    'mnist': datasets.MNIST,
    'fashion_mnist': datasets.FashionMNIST,
    'cifar10': datasets.CIFAR10,
}


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    test_loader: DataLoader
    input_shape: tuple[int, int, int]
    num_classes: int


def build_image_dataloaders(dataset_cfg: dict, train_cfg: dict) -> DatasetBundle:
    dataset_name = dataset_cfg['name']
    if dataset_name not in DATASETS:
        raise ValueError(f'Unsupported dataset: {dataset_name}')

    dataset_cls = DATASETS[dataset_name]
    root = dataset_cfg.get('root', 'data/raw')
    channels = int(dataset_cfg['channels'])
    image_size = int(dataset_cfg['image_size'])
    batch_size = int(train_cfg['batch_size'])

    transform = transforms.ToTensor()
    train_dataset = dataset_cls(root=root, train=True, download=True, transform=transform)
    test_dataset = dataset_cls(root=root, train=False, download=True, transform=transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    return DatasetBundle(
        train_loader=train_loader,
        test_loader=test_loader,
        input_shape=(channels, image_size, image_size),
        num_classes=int(dataset_cfg['num_classes']),
    )
