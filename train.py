"""Train DapiUNet on paired brightfield / DAPI images.

Expects two directories with matching file stems, e.g.::

    data/brightfield/well_A01.tif   (H, W) or (Z, H, W)
    data/dapi/well_A01.tif          (H, W)

Example::

    python train.py --source-dir data/brightfield --target-dir data/dapi --out runs/exp1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dapi_unet import (
    DapiLoss,
    DapiUNet,
    PairedPatchDataset,
    load_image,
    normalize_percentile,
    pearson_r,
    predict_tiled,
    save_checkpoint,
    ssim,
)

IMAGE_SUFFIXES = {".tif", ".tiff", ".npy", ".png"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source-dir", type=Path, required=True, help="brightfield images")
    p.add_argument("--target-dir", type=Path, required=True, help="DAPI images (same file stems)")
    p.add_argument("--out", type=Path, default=Path("runs/dapi_unet"))
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--depth", type=int, default=4)
    p.add_argument("--norm", choices=["batch", "group", "instance", "none"], default="batch")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--attention", action="store_true", help="attention-gated skip connections")
    p.add_argument("--ssim-weight", type=float, default=0.5)
    p.add_argument("--low-percentile", type=float, default=1.0)
    p.add_argument("--high-percentile", type=float, default=99.8)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-amp", action="store_true", help="disable mixed precision on CUDA")
    return p.parse_args()


def find_pairs(source_dir: Path, target_dir: Path) -> list[tuple[Path, Path]]:
    targets = {p.stem: p for p in target_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES}
    pairs = [
        (src, targets[src.stem])
        for src in sorted(source_dir.iterdir())
        if src.suffix.lower() in IMAGE_SUFFIXES and src.stem in targets
    ]
    if not pairs:
        raise SystemExit(f"no matching image pairs in {source_dir} and {target_dir}")
    return pairs


@torch.no_grad()
def evaluate(model, images, device, amp) -> dict[str, float]:
    rs, ssims = [], []
    for src, tgt in images:
        pred = predict_tiled(model, src, tile_size=512, overlap=64, device=device, amp=amp)
        pred_t, tgt_t = torch.from_numpy(pred)[None], torch.from_numpy(tgt)[None]
        rs.append(pearson_r(pred_t, tgt_t).item())
        ssims.append(ssim(pred_t, tgt_t).item())
    return {"pearson_r": float(np.mean(rs)), "ssim": float(np.mean(ssims))}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    norm_kwargs = {"low": args.low_percentile, "high": args.high_percentile}

    def load(path: Path) -> np.ndarray:
        return normalize_percentile(load_image(path), **norm_kwargs)

    pairs = find_pairs(args.source_dir, args.target_dir)
    random.Random(args.seed).shuffle(pairs)
    n_val = max(1, round(len(pairs) * args.val_fraction)) if len(pairs) > 1 else 0
    val_images = [(load(s), load(t)) for s, t in pairs[:n_val]]
    train_images = [(load(s), load(t)) for s, t in pairs[n_val:]]
    print(f"{len(train_images)} training / {len(val_images)} validation images on {device}")

    sources, targets = zip(*train_images)
    dataset = PairedPatchDataset(
        sources,
        targets,
        patch_size=args.patch_size,
        samples_per_epoch=args.steps_per_epoch * args.batch_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.workers > 0,
    )

    model = DapiUNet(
        in_channels=sources[0].shape[0],
        out_channels=targets[0].shape[0],
        base_channels=args.base_channels,
        depth=args.depth,
        norm=args.norm,
        dropout=args.dropout,
        attention=args.attention,
    ).to(device)
    print(f"DapiUNet: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    loss_fn = DapiLoss(ssim_weight=args.ssim_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(loader), pct_start=0.05
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    args.out.mkdir(parents=True, exist_ok=True)
    best_r = -float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for src, tgt in loader:
            src = src.to(device, non_blocking=True)
            tgt = tgt.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=amp):
                pred = model(src)
            loss = loss_fn(pred, tgt)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total += loss.item()

        log = {"epoch": epoch, "train_loss": total / len(loader)}
        if val_images:
            log.update({f"val_{k}": v for k, v in evaluate(model, val_images, device, amp).items()})
        print(json.dumps(log))

        save_checkpoint(args.out / "last.pt", model, normalization=norm_kwargs, epoch=epoch)
        score = log.get("val_pearson_r", -log["train_loss"])
        if score > best_r:
            best_r = score
            save_checkpoint(args.out / "best.pt", model, normalization=norm_kwargs, epoch=epoch)


if __name__ == "__main__":
    main()
