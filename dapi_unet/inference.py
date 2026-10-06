"""Tiled inference for whole fields of view."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _tile_starts(length: int, tile: int, overlap: int) -> list[int]:
    if tile >= length:
        return [0]
    stride = tile - overlap
    starts = list(range(0, length - tile, stride))
    starts.append(length - tile)
    return starts


def _ramp(n: int, overlap: int) -> np.ndarray:
    # Linear cross-fade over the overlap; never zero, so pixels covered by a
    # single tile (image borders) are still recovered exactly.
    w = np.ones(n, dtype=np.float32)
    r = min(overlap, n // 2)
    if r > 0:
        ramp = np.arange(1, r + 1, dtype=np.float32) / (r + 1)
        w[:r] = ramp
        w[-r:] = ramp[::-1]
    return w


@torch.inference_mode()
def predict_tiled(
    model: nn.Module,
    image: np.ndarray,
    tile_size: int = 512,
    overlap: int = 64,
    batch_size: int = 4,
    device: torch.device | str | None = None,
    amp: bool = False,
) -> np.ndarray:
    """Predict a (C_out, H, W) image from a normalized (C_in, H, W) or (H, W) input.

    The image is split into overlapping ``tile_size`` tiles whose predictions
    are blended with linear ramps, which removes the seams that border effects
    would otherwise leave. ``overlap`` should be large compared with the size
    of a nucleus in pixels.
    """
    if not 0 <= overlap <= tile_size // 2:
        raise ValueError(f"overlap must be in [0, tile_size // 2], got {overlap}")
    image = np.asarray(image, dtype=np.float32)
    if image.ndim == 2:
        image = image[None]
    _, height, width = image.shape
    tile_h, tile_w = min(tile_size, height), min(tile_size, width)
    coords = [
        (y, x)
        for y in _tile_starts(height, tile_h, overlap)
        for x in _tile_starts(width, tile_w, overlap)
    ]
    weight = torch.from_numpy(np.outer(_ramp(tile_h, overlap), _ramp(tile_w, overlap)))

    if device is None:
        device = next(model.parameters()).device
    device = torch.device(device)
    was_training = model.training
    model.eval()

    output = None
    weight_sum = torch.zeros(height, width)
    try:
        for start in range(0, len(coords), batch_size):
            batch_coords = coords[start : start + batch_size]
            tiles = torch.stack(
                [torch.from_numpy(image[:, y : y + tile_h, x : x + tile_w]) for y, x in batch_coords]
            ).to(device)
            with torch.autocast(device.type, enabled=amp):
                pred = model(tiles)
            pred = pred.float().cpu()
            if output is None:
                output = torch.zeros(pred.shape[1], height, width)
            for tile_pred, (y, x) in zip(pred, batch_coords):
                output[:, y : y + tile_h, x : x + tile_w] += tile_pred * weight
                weight_sum[y : y + tile_h, x : x + tile_w] += weight
    finally:
        model.train(was_training)
    return (output / weight_sum).numpy()
