"""Offline-only dSprites loader, validating official factor order and image archive.

Compressed .npz imgs are unpacked ONCE to a disk-backed .npy cache, not 3GB RAM.
No network access, torchvision downloads, or per-image Python data transforms.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import shutil
import zipfile
import numpy as np
import torch

FILENAME = 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'
FACTOR_NAMES = ('color', 'shape', 'scale', 'orientation', 'posX', 'posY')
FACTOR_SIZES = (1, 3, 6, 40, 32, 32)
N_IMAGES = int(np.prod(FACTOR_SIZES))
PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = PROJECT / 'data' / FILENAME


def index_from_factors(factors: np.ndarray) -> np.ndarray:
    """Last factor posY is fastest varying; accepts (...,6) integer classes."""
    factors = np.asarray(factors, dtype=np.int64)
    if factors.shape[-1] != 6:
        raise ValueError('Expected [...,6] factor classes in official order')
    for d, size in enumerate(FACTOR_SIZES):
        if np.any((factors[..., d] < 0) | (factors[..., d] >= size)):
            raise ValueError(f'Out of bounds factor: {FACTOR_NAMES[d]}')
    return np.ravel_multi_index(np.moveaxis(factors, -1, 0), FACTOR_SIZES)


def factors_from_index(indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    if np.any((indices < 0) | (indices >= N_IMAGES)):
        raise ValueError('Invalid image index')
    parts = np.unravel_index(indices, FACTOR_SIZES)
    return np.stack(parts, axis=-1)


class DSprites:
    def __init__(self, archive: str | Path = DEFAULT_DATA, cache_dir: str | Path | None = None):
        self.archive = Path(archive).expanduser().resolve()
        if not self.archive.is_file():
            raise FileNotFoundError(
                f'Missing official dSprites file: {self.archive}\n'
                'Download it manually from the Google DeepMind dSprites GitHub page. '
                'No automatic download is attempted.'
            )
        if cache_dir is None:
            cache_dir = Path(os.environ.get('PAPER3_CACHE_DIR', '/content/paper3_runs/data_cache' if Path('/content').exists() else str(PROJECT / 'data' / '.cache')))
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        # NPZ metadata is a NumPy object pickle (from the OFFICIAL trusted archive).
        with np.load(self.archive, allow_pickle=True, encoding='latin1') as archive_data:
            if not {'imgs', 'metadata', 'latents_classes', 'latents_values'}.issubset(set(archive_data.files)):
                raise ValueError('Archive does not have the official dSprites keys')
            metadata = archive_data['metadata'].item()
            if tuple(metadata['latents_sizes']) != FACTOR_SIZES:
                raise ValueError(f'Unexpected factor sizes: {metadata["latents_sizes"]}')
            if tuple(metadata['latents_names']) != FACTOR_NAMES:
                raise ValueError(f'Unexpected factor names: {metadata["latents_names"]}')
            self.metadata = metadata
            check_indices = np.array([0, 1, 31, 32, 1023, 1024, 245759, N_IMAGES-1])
            if not np.array_equal(archive_data['latents_classes'][check_indices], factors_from_index(check_indices)):
                raise ValueError('Official factor index mapping does not match archive')
        self.cache_file = self.cache_dir / f'{self.archive.stem}_imgs.npy'
        # Check file header and expected shape; write via temp and atomic replace.
        if not self._valid_cache():
            tmp_file = self.cache_file.with_suffix('.npy.part')
            try:
                with zipfile.ZipFile(self.archive) as zf, zf.open('imgs.npy') as source, open(tmp_file, 'wb') as target:
                    shutil.copyfileobj(source, target, 8 * 1024 * 1024)
                arr = np.load(tmp_file, mmap_mode='r')
                if arr.shape != (N_IMAGES, 64, 64) or arr.dtype != np.uint8:
                    raise ValueError('dSprites imgs has unexpected dimensions or dtype')
                del arr
                os.replace(tmp_file, self.cache_file)
            finally:
                if tmp_file.exists():
                    tmp_file.unlink()
        self.images = np.load(self.cache_file, mmap_mode='r')
        if self.images.shape != (N_IMAGES, 64, 64):
            raise ValueError('Expected 737280 images at 64x64')
        if not set(np.unique(self.images[[0, 1, 1000, N_IMAGES-1]])).issubset({0, 1}):
            raise ValueError('Expected binary images only')

    def _valid_cache(self) -> bool:
        try:
            a = np.load(self.cache_file, mmap_mode='r')
            ok = a.shape == (N_IMAGES, 64, 64) and a.dtype == np.uint8
            del a
            return ok
        except (OSError, ValueError, FileNotFoundError):
            return False

    def batch(self, indices: np.ndarray) -> torch.Tensor:
        """Vectorized on-demand uint8 reads (no image-at-a-time Python loops)."""
        arr = self.images[np.asarray(indices, dtype=np.int64)]  # fancy index returns owned, writable batch
        return torch.from_numpy(arr).view(-1, 4096)

    def factor_image(self, shape: int, scale: int, orientation: int, pos_x: int, pos_y: int) -> np.ndarray:
        index = int(index_from_factors(np.array([0, shape, scale, orientation, pos_x, pos_y])))
        return np.asarray(self.images[index])


def batches(size: int, batch_size: int, *, seed: int, epoch: int, start_batch: int = 0):
    """Recomputable epoch shuffling: restart at exact batch index on resume."""
    rng = np.random.default_rng(np.random.SeedSequence([int(seed), int(epoch)]))
    order = rng.permutation(size)
    for i in range(start_batch * batch_size, size, batch_size):
        yield i // batch_size, order[i:i + batch_size]


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, default=DEFAULT_DATA)
    p.add_argument('--cache-dir', type=Path)
    args = p.parse_args()
    ds = DSprites(args.data, args.cache_dir)
    print('archive:', ds.archive)
    print('images:', ds.images.shape, ds.images.dtype)
    print('factor order:', FACTOR_NAMES)
    print('factor sizes:', FACTOR_SIZES)
    print('cache:', ds.cache_file)
    print('sample indices:', index_from_factors(factors_from_index(np.array([0, 1, 737279]))))
