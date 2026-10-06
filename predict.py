"""Predict DAPI images from brightfield images with a trained checkpoint.

Example::

    python predict.py runs/exp1/best.pt data/test/*.tif --out-dir predictions
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from dapi_unet import load_checkpoint, load_image, normalize_percentile, predict_tiled, save_image


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("checkpoint", type=Path)
    p.add_argument("inputs", type=Path, nargs="+", help="brightfield images")
    p.add_argument("--out-dir", type=Path, default=Path("predictions"))
    p.add_argument("--tile-size", type=int, default=512)
    p.add_argument("--overlap", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=4)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(args.checkpoint, map_location=device)
    model.to(device)
    norm_kwargs = ckpt.get("normalization", {})

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in args.inputs:
        image = normalize_percentile(load_image(path), **norm_kwargs)
        pred = predict_tiled(
            model,
            image,
            tile_size=args.tile_size,
            overlap=args.overlap,
            batch_size=args.batch_size,
            device=device,
            amp=device.type == "cuda",
        )
        out_path = args.out_dir / f"{path.stem}_dapi.tif"
        save_image(out_path, pred)
        print(f"{path} -> {out_path}")


if __name__ == "__main__":
    main()
