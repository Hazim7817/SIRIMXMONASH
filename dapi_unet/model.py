"""U-Net for predicting DAPI nuclear staining from brightfield images.

Design choices for brightfield -> fluorescence regression:

* Brightfield z-planes are stacked as input channels ("2.5D"). Defocused
  planes carry most of the phase-like contrast that reveals nuclei, so
  ``in_channels`` should usually equal the number of z-planes acquired.
* Residual double-conv blocks with normalization and SiLU train stably on
  the small datasets typical of microscopy.
* BatchNorm is the default: in eval mode it uses running statistics, so the
  prediction is the same whether an image is processed whole or in tiles.
  Per-sample norms ("group", "instance") rescale every tile independently and
  can amplify noise into spurious nuclei on empty background tiles; prefer
  them only when GPU memory forces very small batches.
* Bilinear upsampling + convolution instead of transposed convolutions,
  avoiding checkerboard artifacts in the predicted intensity image.
* Reflection padding in every convolution reduces border artifacts, which
  matters when large fields of view are stitched from tiles.
* Linear (identity) output head: the target is a continuous intensity, not a
  segmentation mask.
* Inputs of any size are reflection-padded to a multiple of ``2**depth`` and
  the output is cropped back, so the output always matches the input size.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

NORMS = ("batch", "group", "instance", "none")


def _num_groups(channels: int, max_groups: int) -> int:
    """Largest group count <= max_groups that divides channels."""
    for groups in range(min(max_groups, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def make_norm(kind: str, channels: int, groups: int = 8) -> nn.Module:
    if kind == "batch":
        return nn.BatchNorm2d(channels)
    if kind == "group":
        return nn.GroupNorm(_num_groups(channels, groups), channels)
    if kind == "instance":
        return nn.InstanceNorm2d(channels, affine=True)
    if kind == "none":
        return nn.Identity()
    raise ValueError(f"norm must be one of {NORMS}, got {kind!r}")


class ResidualBlock(nn.Module):
    """Two 3x3 conv -> norm -> SiLU layers with a residual shortcut."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm: str = "batch",
        padding_mode: str = "reflect",
        dropout: float = 0.0,
    ):
        super().__init__()
        bias = norm == "none"
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, padding=1, padding_mode=padding_mode, bias=bias
        )
        self.norm1 = make_norm(norm, out_channels)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, 3, padding=1, padding_mode=padding_mode, bias=bias
        )
        self.norm2 = make_norm(norm, out_channels)
        self.act = nn.SiLU(inplace=True)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.dropout(h)
        h = self.norm2(self.conv2(h))
        return self.act(h + self.shortcut(x))


class AttentionGate(nn.Module):
    """Additive attention gate (Oktay et al., 2018) applied to skip features.

    The decoder signal decides which encoder features pass through the skip
    connection, which helps suppress debris and out-of-focus clutter that is
    visible in brightfield but has no DAPI signal.
    """

    def __init__(self, skip_channels: int, gate_channels: int):
        super().__init__()
        inter = max(skip_channels // 2, 1)
        self.skip_proj = nn.Conv2d(skip_channels, inter, 1, bias=False)
        self.gate_proj = nn.Conv2d(gate_channels, inter, 1)
        self.psi = nn.Conv2d(inter, 1, 1)

    def forward(self, skip: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        attn = torch.sigmoid(self.psi(F.relu(self.skip_proj(skip) + self.gate_proj(gate))))
        return skip * attn


class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        norm: str,
        padding_mode: str,
        attention: bool,
    ):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(in_channels, out_channels, 1),
        )
        self.gate = AttentionGate(skip_channels, out_channels) if attention else None
        self.block = ResidualBlock(out_channels + skip_channels, out_channels, norm, padding_mode)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if self.gate is not None:
            skip = self.gate(skip, x)
        return self.block(torch.cat([x, skip], dim=1))


class DapiUNet(nn.Module):
    """U-Net mapping brightfield (B, in_channels, H, W) to DAPI (B, out_channels, H, W).

    Args:
        in_channels: Number of brightfield input channels (e.g. z-planes).
        out_channels: Number of predicted fluorescence channels (1 for DAPI).
        base_channels: Feature channels at full resolution; doubled per level.
        depth: Number of 2x downsampling steps. The default of 4 gives a
            receptive field of ~200 pixels, enough to see a whole nucleus
            plus its surroundings at 20-40x magnification.
        max_channels: Cap on feature channels at deep levels.
        norm: "batch" (default), "group", "instance" or "none".
        dropout: Spatial dropout in the bottleneck block.
        attention: Gate skip connections with additive attention.
        padding_mode: Padding mode of 3x3 convolutions ("reflect" or "zeros").
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 4,
        max_channels: int = 512,
        norm: str = "batch",
        dropout: float = 0.1,
        attention: bool = False,
        padding_mode: str = "reflect",
    ):
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")
        if norm not in NORMS:
            raise ValueError(f"norm must be one of {NORMS}, got {norm!r}")
        self.config = dict(
            in_channels=in_channels,
            out_channels=out_channels,
            base_channels=base_channels,
            depth=depth,
            max_channels=max_channels,
            norm=norm,
            dropout=dropout,
            attention=attention,
            padding_mode=padding_mode,
        )
        self.depth = depth
        chans = [min(base_channels * 2**i, max_channels) for i in range(depth + 1)]

        self.stem = ResidualBlock(in_channels, chans[0], norm, padding_mode)
        self.down = nn.ModuleList(
            ResidualBlock(
                chans[i],
                chans[i + 1],
                norm,
                padding_mode,
                dropout=dropout if i == depth - 1 else 0.0,
            )
            for i in range(depth)
        )
        self.pool = nn.MaxPool2d(2)
        self.up = nn.ModuleList(
            UpBlock(chans[i + 1], chans[i], chans[i], norm, padding_mode, attention)
            for i in reversed(range(depth))
        )
        self.head = nn.Conv2d(chans[0], out_channels, 1)

    def _pad_to_valid_size(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        # Each spatial dim must be divisible by 2**depth, and the bottleneck
        # must be at least 2x2 so its reflection padding is well defined.
        multiple = 2**self.depth
        h, w = x.shape[-2:]
        target_h = max(-(-h // multiple) * multiple, 2 * multiple)
        target_w = max(-(-w // multiple) * multiple, 2 * multiple)
        pad_h, pad_w = target_h - h, target_w - w
        if pad_h or pad_w:
            mode = "reflect" if pad_h < h and pad_w < w else "replicate"
            x = F.pad(x, (0, pad_w, 0, pad_h), mode=mode)
        return x, h, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, h, w = self._pad_to_valid_size(x)
        x = self.stem(x)
        skips = []
        for block in self.down:
            skips.append(x)
            x = block(self.pool(x))
        for block, skip in zip(self.up, reversed(skips)):
            x = block(x, skip)
        return self.head(x)[..., :h, :w]


def save_checkpoint(path: str | Path, model: DapiUNet, **extra) -> None:
    """Save weights together with the model config needed to rebuild it."""
    torch.save({"config": model.config, "state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> tuple[DapiUNet, dict]:
    """Rebuild a DapiUNet from ``save_checkpoint`` output. Returns (model, checkpoint)."""
    ckpt = torch.load(path, map_location=map_location, weights_only=True)
    model = DapiUNet(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    return model, ckpt
