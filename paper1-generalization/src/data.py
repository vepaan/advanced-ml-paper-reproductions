"""CIFAR-10 data pipeline for Zhang et al. (2017), Figure 1.

Independent PyTorch reproduction. Random transformations are deterministic by
sample index and seed, including when DataLoader uses multiple workers.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10


class InputCondition(str, Enum):
    ORIGINAL = "original"
    SHARED_SHUFFLE = "shared_shuffle"
    INDEPENDENT_SHUFFLE = "independent_shuffle"
    GAUSSIAN = "gaussian"


@dataclass(frozen=True)
class DataConfig:
    root: str = "data"
    train: bool = True
    condition: InputCondition = InputCondition.ORIGINAL
    corruption_probability: float = 0.0
    label_seed: int = 1234
    input_seed: int = 5678
    download: bool = False


def whiten_per_image(image: torch.Tensor) -> torch.Tensor:
    """Match TensorFlow per_image_standardization for one CHW image.

    Uses population standard deviation and adjusted_std=max(std, 1/sqrt(N)).
    """
    if image.ndim != 3 or not image.is_floating_point():
        raise ValueError("Expected one floating-point image of shape (C,H,W)")
    x = image.to(torch.float32)
    centered = x - x.mean()
    variance = (centered * centered).mean()
    adjusted_std = torch.maximum(variance.sqrt(), x.new_tensor(x.numel() ** -0.5))
    return centered / adjusted_std


def _rng(seed: int, index: int, stream: int) -> np.random.Generator:
    """Independent deterministic RNG stream; does not change global RNG."""
    return np.random.default_rng(np.random.SeedSequence([seed, index, stream]))


class Figure1CIFAR10(Dataset):
    """CIFAR-10 with fixed training-label corruption or fixed input changes.

    Access .original_targets and .targets to inspect realized label changes.
    Test data always uses original images and labels; its condition must ORIGINAL.
    """

    def __init__(self, config: DataConfig):
        if not 0 <= config.corruption_probability <= 1:
            raise ValueError("corruption_probability must be in [0, 1]")
        if not config.train and (
            config.condition != InputCondition.ORIGINAL
            or config.corruption_probability != 0
        ):
            raise ValueError("Test dataset must use original inputs and labels")

        self.config = config
        self.base = CIFAR10(root=config.root, train=config.train, download=config.download)
        self.original_targets = np.asarray(self.base.targets, dtype=np.int64)
        self.targets = self.original_targets.copy()

        if config.train and config.corruption_probability > 0:
            rng = _rng(config.label_seed, 0, 1)
            mask = rng.random(len(self.targets)) < config.corruption_probability
            replacements = rng.integers(0, 10, size=len(self.targets))
            self.targets[mask] = replacements[mask]

        self.shared_permutation: Optional[np.ndarray] = None
        if config.condition == InputCondition.SHARED_SHUFFLE:
            self.shared_permutation = _rng(config.input_seed, 0, 2).permutation(28 * 28 * 3)

        self.gaussian_mean: Optional[float] = None
        self.gaussian_std: Optional[float] = None
        if config.condition == InputCondition.GAUSSIAN:
            # Statistics computed from the ORIGINAL training split, not test data.
            raw = self.base.data[:, 2:30, 2:30, :]
            pixels = raw.astype(np.float64) / 255.0
            self.gaussian_mean = float(pixels.mean())
            self.gaussian_std = float(pixels.std(ddof=0))

    def __len__(self) -> int:
        return len(self.targets)

    @property
    def realized_label_change_rate(self) -> float:
        return float(np.mean(self.targets != self.original_targets))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        # Cropping before randomization is a documented reproduction choice.
        image = self.base.data[index, 2:30, 2:30, :].astype(np.float32) / 255.0
        condition = self.config.condition

        if condition == InputCondition.SHARED_SHUFFLE:
            assert self.shared_permutation is not None
            image = image.reshape(-1)[self.shared_permutation].reshape(28, 28, 3)
        elif condition == InputCondition.INDEPENDENT_SHUFFLE:
            perm = _rng(self.config.input_seed, index, 3).permutation(image.size)
            image = image.reshape(-1)[perm].reshape(28, 28, 3)
        elif condition == InputCondition.GAUSSIAN:
            assert self.gaussian_mean is not None and self.gaussian_std is not None
            image = _rng(self.config.input_seed, index, 4).normal(
                loc=self.gaussian_mean, scale=self.gaussian_std, size=(28, 28, 3)
            ).astype(np.float32)
        elif condition != InputCondition.ORIGINAL:
            raise ValueError(f"Unsupported input condition: {condition}")

        tensor = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        return whiten_per_image(tensor), int(self.targets[index])


def make_loader(
    dataset: Dataset,
    batch_size: int = 128,
    shuffle: bool = False,
    num_workers: int = 0,
    seed: int = 42,
    pin_memory: bool = False,
) -> DataLoader:
    """Deterministic sample order for a fixed seed and worker configuration."""
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers nonnegative")
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        pin_memory=pin_memory,
        drop_last=False,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Inspect Figure 1 CIFAR-10 transformations")
    parser.add_argument("--root", default="data")
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()

    for condition in InputCondition:
        ds = Figure1CIFAR10(DataConfig(
            root=args.root, condition=condition, download=args.download,
            corruption_probability=1.0 if condition == InputCondition.ORIGINAL else 0.0,
        ))
        a, y = ds[0]
        b, y2 = ds[0]
        assert a.shape == (3, 28, 28)
        assert torch.equal(a, b) and y == y2, "Transformation changed between accesses"
        assert torch.isfinite(a).all()
        assert abs(float(a.mean())) < 1e-5
        print(f"{condition.value:>20}: shape={tuple(a.shape)}, "
              f"label={y}, mean={a.mean():+.2e}, "
              f"std={a.std(unbiased=False):.4f}, "
              f"changed_labels={ds.realized_label_change_rate:.4f}")
