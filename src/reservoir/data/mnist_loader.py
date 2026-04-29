"""/home/yoshi/PycharmProjects/Reservoir/src/reservoir/data/mnist_loader.py
MNIST dataset utilities used by reservoir.data.* modules."""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from beartype import beartype

import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from reservoir.core.types import NpF64

if TYPE_CHECKING:
    from torch import Tensor

MNIST_ROOT = Path("data/mnist")


def get_mnist_datasets() -> tuple[datasets.MNIST, datasets.MNIST]:
    """Download (if needed) and return MNIST train/test datasets."""
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: x.float())
    ])
    train_set = datasets.MNIST(root=str(MNIST_ROOT), train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=str(MNIST_ROOT), train=False, download=True, transform=transform)
    return train_set, test_set


@beartype
def image_to_sequence(image: NpF64, *, n_steps: int) -> NpF64:
    """Convert a single MNIST image into a (time, features) sequence."""
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]
    if image.shape != (28, 28):
        raise ValueError(f"Expected image shape (28,28), got {image.shape}")
    total_pixels = 28 * 28
    n_steps = int(n_steps)
    if n_steps <= 0 or total_pixels % n_steps != 0:
        raise ValueError(f"n_steps must be positive and divide {total_pixels}, got {n_steps}")
    features_per_step = total_pixels // n_steps
    flat = image.reshape(total_pixels)
    result = flat.reshape(n_steps, features_per_step)
    if result.dtype != np.float64:
        raise ValueError(f"MNIST sequence must be float64, got {result.dtype.name}")
    return result


def get_mnist_dataloaders(
    batch_size: int = 128,
    shuffle_train: bool = True,
    num_workers: int = 0,
) -> tuple[DataLoader[tuple[Tensor, int]], DataLoader[tuple[Tensor, int]]]:
    """Convenience PyTorch dataloaders for MNIST."""
    train_set, test_set = get_mnist_datasets()
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle_train, num_workers=num_workers)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
