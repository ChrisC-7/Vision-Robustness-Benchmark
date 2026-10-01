"""CIFAR-10 loading and corruption transforms for the Vision Robustness Benchmark."""

from typing import Callable, Iterator, Sequence

import numpy as np
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


def gaussian_noise_transform(noise_seed: int, mean: float = 0.0, std: float = 0.10) -> Callable:
    generator = torch.Generator().manual_seed(noise_seed)
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


class InMemoryLoader:
    """Batches of (images, labels) gathered from tensors that already live on a device.

    Batch order comes from a real ``DataLoader`` over the sample indices, so with
    the same ``shuffle`` / ``generator`` the batches are exactly those of
    ``DataLoader(dataset, shuffle=..., generator=...)``. Only the per-sample PIL
    decoding and ``ToTensor`` are skipped, and with ``images`` on the GPU the
    batches need no host-to-device copy.
    """

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        indices: Sequence[int],
        batch_size: int,
        shuffle: bool = False,
        generator: torch.Generator | None = None,
    ) -> None:
        self.images = images
        self.labels = labels
        self.dataset = indices  # sized like DataLoader.dataset; engine uses len(loader.dataset)
        self._index_loader = DataLoader(
            indices, batch_size=batch_size, shuffle=shuffle, generator=generator
        )

    def __len__(self) -> int:
        return len(self._index_loader)

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        for idx in self._index_loader:
            idx = idx.to(self.images.device)
            yield self.images[idx], self.labels[idx]


def to_image_tensor(images: np.ndarray) -> torch.Tensor:
    """uint8 NHWC array -> float32 NCHW in [0, 1]; the same values as ``ToTensor`` per image."""
    return torch.from_numpy(images).permute(0, 3, 1, 2).float().div(255)


def get_train_val_dataloaders(
    root: str = "data",
    training_seed: int = 42,
    split_seed: int = 42,
    batch_size: int = 64,
    device: str = "cpu",
) -> tuple[InMemoryLoader, InMemoryLoader]:
    # Training uses only the clean transform (no augmentation), so the whole train
    # split is converted once and kept on ``device`` instead of decoded every epoch.
    cifar_train = datasets.CIFAR10(root=root, train=True, download=True)
    images = to_image_tensor(cifar_train.data).to(device)
    labels = torch.tensor(cifar_train.targets).to(device)

    split_generator = torch.Generator().manual_seed(split_seed)
    shuffle_generator = torch.Generator().manual_seed(training_seed)

    train_data, val_data = random_split(cifar_train, [0.9, 0.1], split_generator)
    train_dataloader = InMemoryLoader(
        images, labels, train_data.indices, batch_size, shuffle=True, generator=shuffle_generator
    )
    val_dataloader = InMemoryLoader(images, labels, val_data.indices, batch_size)
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
