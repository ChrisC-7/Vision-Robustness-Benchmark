"""CIFAR-10 loading and corruption transforms for the Vision Robustness Benchmark."""

from typing import Callable

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def clean_transform() -> Callable:
    return transforms.ToTensor()


def gaussian_blur_transform(kernel_size: int = 5, sigma: float = 1.0) -> Callable:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.GaussianBlur(kernel_size=kernel_size, sigma=sigma),
        ]
    )


class GaussianNoise:
    """Additive Gaussian noise, clamped to [0, 1], with dedicated RNG state.

    A dedicated generator (rather than the global RNG) keeps this transform's
    randomness independent of unrelated calls to torch's global RNG, so
    corruption results stay reproducible regardless of what else runs first.
    """

    def __init__(self, mean: float, std: float, generator: torch.Generator) -> None:
        self.mean = mean
        self.std = std
        self.generator = generator

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        noise = self.std * torch.randn(img.shape, generator=self.generator) + self.mean
        return torch.clamp(img + noise, min=0, max=1)


def gaussian_noise_transform(seed: int, mean: float = 0.0, std: float = 0.10) -> Callable:
    generator = torch.Generator().manual_seed(seed)
    return transforms.Compose(
        [
            transforms.ToTensor(),
            GaussianNoise(mean, std, generator),
        ]
    )

_TRANSFORMS = {
    "clean": clean_transform,
    "blur": gaussian_blur_transform,
    "noise": gaussian_noise_transform,
}


def get_train_val_dataloaders(
    root: str = "data", seed: int = 42, batch_size: int = 64
) -> tuple[DataLoader, DataLoader]:
    cifar_train = datasets.CIFAR10(
        root=root,
        train=True,
        transform=clean_transform(),
        download=True,
    )

    split_generator = torch.Generator().manual_seed(seed)
    shuffle_generator = torch.Generator().manual_seed(seed)

    train_data, val_data = random_split(cifar_train, [0.9, 0.1], split_generator)
    train_dataloader = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, generator=shuffle_generator
    )
    val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
    return train_dataloader, val_dataloader


def get_test_dataloader(
    transform: Callable, root: str = "data", batch_size: int = 64
) -> DataLoader:
    cifar_test = datasets.CIFAR10(
        root=root,
        train=False,
        transform=transform,
        download=True,
    )
    return DataLoader(cifar_test, batch_size=batch_size, shuffle=False)
