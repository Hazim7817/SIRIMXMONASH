"""Image I/O, normalization and patch sampling for brightfield/DAPI pairs.

All images are handled as float32 arrays of shape (C, H, W). For brightfield
z-stacks, C is the number of focal planes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import Dataset


def to_chw(img: np.ndarray) -> np.ndarray:
    """Convert (H, W), (C, H, W) or channels-last RGB(A) (H, W, 3|4) to (C, H, W)."""
    img = np.asarray(img)
    if img.ndim == 2:
        return img[None]
    if img.ndim == 3:
        if img.shape[-1] in (3, 4) and img.shape[0] > 4:
            return np.moveaxis(img, -1, 0)
        return img
    raise ValueError(f"expected a 2D or 3D image, got shape {img.shape}")


def load_image(path: str | Path) -> np.ndarray:
    """Load a .tif/.tiff, .npy or Pillow-readable image as float32 (C, H, W)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        import tifffile

        img = tifffile.imread(path)
    elif suffix == ".npy":
        img = np.load(path)
    else:
        from PIL import Image

        img = np.asarray(Image.open(path))
    return to_chw(np.squeeze(img) if img.ndim > 3 else img).astype(np.float32)


def save_image(path: str | Path, img: np.ndarray) -> None:
    """Save a float32 image as .tif/.tiff or .npy."""
    path = Path(path)
    img = np.asarray(img, dtype=np.float32)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        tifffile.imwrite(path, img, photometric="minisblack")
    elif path.suffix.lower() == ".npy":
        np.save(path, img)
    else:
        raise ValueError(f"unsupported output format {path.suffix!r}; use .tif or .npy")


def normalize_percentile(
    img: np.ndarray, low: float = 1.0, high: float = 99.8, eps: float = 1e-6
) -> np.ndarray:
    """Map the [low, high] percentile range of ``img`` to [0, 1] (not clipped).

    Statistics are computed over the whole array, so the relative intensity
    of planes in a z-stack is preserved. Percentiles are robust to the bright
    outliers (debris, saturated pixels) common in both modalities.
    """
    img = np.asarray(img, dtype=np.float32)
    lo, hi = np.percentile(img, [low, high])
    return ((img - lo) / (hi - lo + eps)).astype(np.float32)


class PairedPatchDataset(Dataset):
    """Random aligned patches from brightfield/DAPI image pairs.

    Each item is a ``(source, target)`` pair of float32 tensors with shapes
    (C_in, P, P) and (C_out, P, P). Images are sampled proportionally to their
    area. Augmentation applies the same random rotation by a multiple of 90
    degrees and flip to both images (microscopy has no preferred orientation),
    plus a random gain/offset to the brightfield only, to make the model
    robust to lamp intensity and exposure differences between sessions.

    Args:
        sources: Normalized brightfield images, each (C_in, H, W) or (H, W).
        targets: Normalized DAPI images, each (C_out, H, W) or (H, W).
        patch_size: Side length of the square patches.
        samples_per_epoch: Length reported by ``__len__``.
        augment: Enable geometric and intensity augmentation.
        intensity_jitter: Max relative gain change and additive offset.
    """

    def __init__(
        self,
        sources: Sequence[np.ndarray],
        targets: Sequence[np.ndarray],
        patch_size: int = 256,
        samples_per_epoch: int = 1000,
        augment: bool = True,
        intensity_jitter: float = 0.1,
    ):
        if len(sources) != len(targets) or not sources:
            raise ValueError("sources and targets must be non-empty and of equal length")
        self.sources, self.targets = [], []
        for src, tgt in zip(sources, targets):
            src, tgt = to_chw(src).astype(np.float32), to_chw(tgt).astype(np.float32)
            if src.shape[-2:] != tgt.shape[-2:]:
                raise ValueError(f"spatial shapes differ: {src.shape} vs {tgt.shape}")
            pad_h = max(patch_size - src.shape[1], 0)
            pad_w = max(patch_size - src.shape[2], 0)
            if pad_h or pad_w:
                pad = ((0, 0), (0, pad_h), (0, pad_w))
                src, tgt = np.pad(src, pad, mode="reflect"), np.pad(tgt, pad, mode="reflect")
            self.sources.append(src)
            self.targets.append(tgt)
        areas = np.array([s.shape[1] * s.shape[2] for s in self.sources], dtype=np.float64)
        self.weights = areas / areas.sum()
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        self.intensity_jitter = intensity_jitter

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Fresh OS entropy per item so DataLoader workers never share a stream.
        rng = np.random.default_rng()
        i = rng.choice(len(self.sources), p=self.weights)
        src, tgt = self.sources[i], self.targets[i]
        p = self.patch_size
        y = rng.integers(0, src.shape[1] - p + 1)
        x = rng.integers(0, src.shape[2] - p + 1)
        src = src[:, y : y + p, x : x + p]
        tgt = tgt[:, y : y + p, x : x + p]

        if self.augment:
            k = int(rng.integers(4))
            src, tgt = np.rot90(src, k, axes=(1, 2)), np.rot90(tgt, k, axes=(1, 2))
            if rng.random() < 0.5:
                src, tgt = src[:, :, ::-1], tgt[:, :, ::-1]
            if self.intensity_jitter > 0:
                j = self.intensity_jitter
                src = src * (1 + rng.uniform(-j, j)) + rng.uniform(-j, j)

        return (
            torch.from_numpy(np.ascontiguousarray(src, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(tgt, dtype=np.float32)),
        )
