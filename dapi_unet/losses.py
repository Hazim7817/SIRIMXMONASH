"""Loss and metrics for fluorescence-intensity regression.

Pixel-wise losses alone are dominated by the large dark background of a DAPI
image and tend to give blurry nuclei. Mixing L1 with SSIM (Zhao et al., 2017,
"Loss Functions for Image Restoration with Neural Networks") rewards correct
local structure - nuclear boundaries and chromatin texture - while L1 keeps
absolute intensities calibrated.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _gaussian_window(size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window = torch.outer(g, g)
    return window.expand(channels, 1, size, size).contiguous()


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """Structural similarity between (B, C, H, W) tensors.

    Computed in float32 for numerical stability under mixed precision.
    ``reduction="none"`` returns one value per sample.
    """
    pred, target = pred.float(), target.float()
    channels = pred.shape[1]
    window_size = min(window_size, *pred.shape[-2:])
    window = _gaussian_window(window_size, sigma, channels, pred.device, pred.dtype)

    def filt(x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, window, groups=channels)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    mu_x, mu_y = filt(pred), filt(target)
    var_x = filt(pred * pred) - mu_x**2
    var_y = filt(target * target) - mu_y**2
    cov_xy = filt(pred * target) - mu_x * mu_y
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2)
    )
    per_sample = ssim_map.flatten(1).mean(dim=1)
    if reduction == "none":
        return per_sample
    if reduction == "mean":
        return per_sample.mean()
    raise ValueError(f"reduction must be 'mean' or 'none', got {reduction!r}")


def pearson_r(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Per-sample Pearson correlation, the standard metric for label-free prediction."""
    p = pred.float().flatten(1)
    t = target.float().flatten(1)
    p = p - p.mean(dim=1, keepdim=True)
    t = t - t.mean(dim=1, keepdim=True)
    return (p * t).sum(dim=1) / (p.norm(dim=1) * t.norm(dim=1) + eps)


class DapiLoss(nn.Module):
    """``(1 - w) * L1 + w * (1 - SSIM)`` for percentile-normalized targets.

    Args:
        ssim_weight: Weight ``w`` of the SSIM term in [0, 1].
        data_range: Dynamic range of the target used by SSIM (1.0 for
            targets normalized to roughly [0, 1]).
    """

    def __init__(self, ssim_weight: float = 0.5, data_range: float = 1.0):
        super().__init__()
        if not 0.0 <= ssim_weight <= 1.0:
            raise ValueError(f"ssim_weight must be in [0, 1], got {ssim_weight}")
        self.ssim_weight = ssim_weight
        self.data_range = data_range

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # F.l1_loss would silently broadcast mismatched shapes.
        if pred.shape != target.shape:
            raise ValueError(f"prediction shape {tuple(pred.shape)} != target shape {tuple(target.shape)}")
        l1 = F.l1_loss(pred.float(), target.float())
        if self.ssim_weight == 0:
            return l1
        structural = 1 - ssim(pred, target, data_range=self.data_range)
        return (1 - self.ssim_weight) * l1 + self.ssim_weight * structural
