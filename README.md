# DAPI U-Net

A PyTorch U-Net that predicts DAPI nuclear staining from brightfield images
(in-silico labeling or label-free prediction). Train it once on paired
brightfield/DAPI acquisitions. After that, nuclei can be predicted from
brightfield alone, without staining or a fluorescence channel.

## Architecture

`dapi_unet.DapiUNet` is a 4-level residual U-Net (7.6M parameters, ~200 px
receptive field with default settings). Its design choices target
brightfield → fluorescence regression:

| Choice | Why |
| --- | --- |
| Z-planes as input channels | Defocused brightfield planes carry most of the contrast that reveals nuclei. Set `in_channels` to the number of planes. |
| Residual conv blocks + SiLU | Stable optimization on small microscopy datasets. |
| BatchNorm (default) | In eval mode the output does not depend on tile size or content, so tiled and whole-image predictions agree. Per-sample norms (`group`, `instance`) can amplify noise into spurious nuclei on empty tiles. |
| Bilinear upsampling + conv | Avoids the checkerboard artifacts of transposed convolutions in intensity images. |
| Reflection padding | Fewer border artifacts when stitching tiles. |
| Linear output head | DAPI intensity is a continuous regression target, not a mask. |
| Optional attention gates (`attention=True`) | Suppress skip features from debris and out-of-focus clutter that has no DAPI signal. |
| Bottleneck spatial dropout | Regularization for small datasets. |
| Automatic padding | Any input size works. Inputs are padded to a multiple of `2**depth` and the output is cropped back. |

Supporting pieces:

- **Loss** (`DapiLoss`): `(1 - w)·L1 + w·(1 - SSIM)`. L1 alone is dominated by
  the dark background and gives blurry nuclei. The SSIM term rewards correct
  boundaries and chromatin texture.
- **Metrics**: per-image Pearson r (the standard label-free metric) and SSIM.
- **Normalization**: per-image percentile normalization (1st → 0,
  99.8th → 1) of both modalities, so the model is robust to exposure and
  lamp changes. Statistics are computed over the whole z-stack, which keeps
  the relative intensity of each plane.
- **Augmentation**: dihedral flips and rotations applied identically to
  both images, plus gain/offset jitter applied to brightfield only.
- **Tiled inference** (`predict_tiled`): overlapping tiles with
  linear-ramp blending for large fields of view. In a synthetic check
  with 64 px overlap, tiled output matched whole-image output to within ~3%
  of the dynamic range at worst. Without overlap, seams are clearly visible.

## Install

```bash
pip install -r requirements.txt
```

## Data layout

Put brightfield and DAPI images in two folders, paired by file name.
Supported formats are `.tif`, `.tiff`, `.npy` and `.png`. Brightfield can be
a single plane `(H, W)` or a z-stack `(Z, H, W)`. Images must be registered.

```
data/brightfield/well_A01.tif
data/dapi/well_A01.tif
```

## Train

```bash
python train.py --source-dir data/brightfield --target-dir data/dapi --out runs/exp1
```

Each epoch logs training loss, validation Pearson r and SSIM. The best
checkpoint by validation Pearson r goes to `runs/exp1/best.pt`. Useful flags:
`--attention`, `--norm group` (for very small batches), `--patch-size`,
`--batch-size` and `--ssim-weight`. Mixed precision is on by default on CUDA.

## Predict

```bash
python predict.py runs/exp1/best.pt data/test/*.tif --out-dir predictions
```

The checkpoint stores the model config and the normalization percentiles,
so prediction uses the same preprocessing as training.

## Python API

```python
import torch
from dapi_unet import DapiUNet, DapiLoss, load_image, normalize_percentile, predict_tiled

model = DapiUNet(in_channels=3, attention=True)       # e.g. 3 z-planes
pred = model(torch.randn(4, 3, 256, 256))             # -> (4, 1, 256, 256)
loss = DapiLoss()(pred, torch.rand(4, 1, 256, 256))

image = normalize_percentile(load_image("field.tif"))  # (3, H, W)
dapi = predict_tiled(model, image, tile_size=512, overlap=64)  # (1, H, W)
```

## Tests

```bash
python -m pytest
```

The tests cover output shapes for arbitrary sizes, every norm and attention
variant, checkpoint round-trips, loss and metric identities, exact stitching
for a pointwise model, paired augmentation consistency, and a small
end-to-end learning check on synthetic nuclei.
