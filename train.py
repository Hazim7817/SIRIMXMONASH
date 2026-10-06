"""Train DapiUNet on paired brightfield / DAPI images.

Expects two directories with matching file stems, e.g.::

    data/brightfield/well_A01.tif   (H, W) or (Z, H, W)
    data/dapi/well_A01.tif          (H, W)

Example::

    python train.py --source-dir data/brightfield --target-dir data/dapi --out runs/exp1

Checks that the run is working:

* Before training, every image pair is validated (matching sizes, consistent
  channel counts, no NaN/Inf, not nearly blank or saturated), so bad data
  fails immediately rather than mid-run.
* Every epoch prints the training loss and, when validation images exist,
  the validation loss, Pearson r and SSIM. With no validation images
  (one pair, or ``--val-fraction 0``) validation is skipped.
* The same numbers are written to ``<out>/history.csv``.
* Training stops with an error as soon as a loss becomes NaN or Inf.
* The run ends with a summary that warns if the training loss never
  decreased, the model never beat the best constant prediction, or its
  predictions are barely correlated with DAPI (e.g. mismatched pairs).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
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
# Below this best validation Pearson r the model has not learned to predict
# nuclei; a constant output scores 0 and mismatched pairs score about 0.
MIN_PEARSON_R = 0.3


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    p.add_argument(
        "--val-fraction", type=float, default=0.15, help="fraction of pairs held out; 0 disables validation"
    )
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
    return p.parse_args(argv)


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


def split_pairs(pairs: list, val_fraction: float, seed: int) -> tuple[list, list]:
    """Shuffle and split into (train, val).

    ``val_fraction == 0`` (or a single pair) gives no validation set. Otherwise
    at least one pair is held out and at least one is kept for training.
    """
    if not 0 <= val_fraction < 1:
        raise SystemExit(f"--val-fraction must be in [0, 1), got {val_fraction}")
    pairs = list(pairs)
    random.Random(seed).shuffle(pairs)
    n_val = round(len(pairs) * val_fraction)
    if val_fraction > 0 and len(pairs) > 1:
        n_val = max(n_val, 1)
    n_val = min(n_val, len(pairs) - 1)
    return pairs[n_val:], pairs[:n_val]


def validate_images(
    images: list[tuple[np.ndarray, np.ndarray]], names: list[str], low: float = 1.0, high: float = 99.8
) -> list[str]:
    """Check normalized (brightfield, DAPI) pairs before training starts.

    Raises ValueError for problems that would crash or corrupt training and
    returns warnings for images that are usable but suspicious. ``low`` and
    ``high`` must be the percentiles the images were normalized with.
    """
    warnings = []
    for (src, tgt), name in zip(images, names):
        if src.shape[1:] != tgt.shape[1:]:
            raise ValueError(
                f"{name}: brightfield size {src.shape[1:]} differs from DAPI size {tgt.shape[1:]}"
            )
        for kind, img in (("brightfield", src), ("DAPI", tgt)):
            if not np.isfinite(img).all():
                raise ValueError(f"{name}: {kind} image contains NaN or Inf values")
            if img.min() == img.max():
                warnings.append(f"{name}: {kind} image is constant (blank or saturated)")
                continue
            # A normalized image spans ~1 between these percentiles. ~0 means
            # they were equal and normalization divided by eps, inflating the
            # few remaining pixels (e.g. a hot pixel on a blank field) to ~1e9.
            lo, hi = np.percentile(img, [low, high])
            if hi - lo < 0.5:
                raise ValueError(
                    f"{name}: {kind} image is nearly blank or saturated, so normalization inflates "
                    f"its values to {np.abs(img).max():.3g}; remove it or widen "
                    "--low-percentile/--high-percentile"
                )
    for kind, index in (("brightfield", 0), ("DAPI", 1)):
        channels = {pair[index].shape[0] for pair in images}
        if len(channels) > 1:
            raise ValueError(f"{kind} images have different channel counts: {sorted(channels)}")
    return warnings


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, scaler, device, amp, epoch) -> float:
    """Run one epoch and return the mean training loss; stops on NaN/Inf loss."""
    model.train()
    total = 0.0
    for step, (src, tgt) in enumerate(loader, start=1):
        src = src.to(device, non_blocking=True)
        tgt = tgt.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=amp):
            pred = model(src)
        loss = loss_fn(pred, tgt)
        value = loss.item()
        if not math.isfinite(value):
            raise FloatingPointError(
                f"epoch {epoch}, step {step}: training loss is {value}; "
                "try a lower --lr, --no-amp, or check the input images"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += value
    return total / len(loader)


@torch.no_grad()
def evaluate(model, images, loss_fn, device, amp) -> dict[str, float]:
    """Validation loss, Pearson r and SSIM averaged over whole validation images."""
    losses, rs, ssims = [], [], []
    for src, tgt in images:
        pred = predict_tiled(model, src, device=device, amp=amp)
        pred_t, tgt_t = torch.from_numpy(pred)[None], torch.from_numpy(tgt)[None]
        losses.append(loss_fn(pred_t, tgt_t).item())
        rs.append(pearson_r(pred_t, tgt_t).item())
        ssims.append(ssim(pred_t, tgt_t).item())
    metrics = {
        "val_loss": float(np.mean(losses)),
        "val_pearson_r": float(np.mean(rs)),
        "val_ssim": float(np.mean(ssims)),
    }
    bad = [k for k, v in metrics.items() if not math.isfinite(v)]
    if bad:
        raise FloatingPointError(f"validation produced non-finite {', '.join(bad)}")
    return metrics


def constant_baseline_loss(train_images, val_images, loss_fn) -> float:
    """Lowest validation loss achievable by predicting one constant everywhere.

    DAPI images are mostly background, so the mean intensity is a weak
    constant; the median (optimal for L1) and other quantiles of the training
    DAPI intensities are tried too. A model that learned to predict nuclei
    from brightfield ends up well below this; one that only outputs the
    background level does not.
    """
    values = np.concatenate([tgt[:, ::4, ::4].ravel() for _, tgt in train_images])
    candidates = [float(values.mean()), *np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9]).tolist()]
    targets = [torch.from_numpy(tgt)[None] for _, tgt in val_images]
    return min(
        float(np.mean([loss_fn(torch.full_like(t, value), t).item() for t in targets]))
        for value in candidates
    )


def append_history(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def format_epoch(row: dict, epochs: int) -> str:
    metrics = [f"{k} {v:.4f}" for k, v in row.items() if k not in ("epoch", "lr", "seconds")]
    return "  ".join(
        [f"epoch {row['epoch']}/{epochs}", *metrics, f"lr {row['lr']:.2e}", f"{row['seconds']:.1f}s"]
    )


def summarize(history: list[dict], baseline: float | None, best_epoch: int) -> list[str]:
    """End-of-run report, with warnings when the run does not look healthy."""
    first, last = history[0], history[-1]
    lines = [
        f"train_loss {first['train_loss']:.4f} (epoch 1) -> {last['train_loss']:.4f} (epoch {last['epoch']})"
    ]
    if len(history) > 1 and last["train_loss"] >= first["train_loss"]:
        lines.append("WARNING: training loss did not decrease; try another --lr")
    if "val_loss" not in first:
        lines.append("validation was skipped (no validation images)")
        lines.append(f"best.pt is from epoch {best_epoch} (lowest train_loss)")
        return lines

    best = min(history, key=lambda r: r["val_loss"])
    lines.append(
        f"best val_loss {best['val_loss']:.4f} at epoch {best['epoch']} "
        f"(best constant prediction {baseline:.4f})"
    )
    if best["val_loss"] >= baseline:
        lines.append("WARNING: the model never beat a constant prediction on the validation images")
    best_r = max(r["val_pearson_r"] for r in history)
    if best_r < MIN_PEARSON_R:
        lines.append(
            f"WARNING: predictions are barely correlated with DAPI (best val_pearson_r {best_r:.3f}); "
            "check that brightfield and DAPI files are correctly paired and registered"
        )
    saved = next(r for r in history if r["epoch"] == best_epoch)
    lines.append(
        f"best.pt is from epoch {best_epoch} (highest val_pearson_r {saved['val_pearson_r']:.4f}, "
        f"val_loss {saved['val_loss']:.4f})"
    )
    return lines


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda" and not args.no_amp
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    norm_kwargs = {"low": args.low_percentile, "high": args.high_percentile}

    def load(pairs: list[tuple[Path, Path]]) -> list[tuple[np.ndarray, np.ndarray]]:
        return [
            tuple(normalize_percentile(load_image(path), **norm_kwargs) for path in pair) for pair in pairs
        ]

    pairs = find_pairs(args.source_dir, args.target_dir)
    train_pairs, val_pairs = split_pairs(pairs, args.val_fraction, args.seed)
    train_images, val_images = load(train_pairs), load(val_pairs)
    names = [s.name for s, _ in train_pairs + val_pairs]
    try:
        warnings = validate_images(train_images + val_images, names, **norm_kwargs)
    except ValueError as e:
        raise SystemExit(f"data check failed: {e}") from e
    for warning in warnings:
        print(f"WARNING: {warning}")
    print(f"{len(train_images)} training / {len(val_images)} validation images on {device}")

    loss_fn = DapiLoss(ssim_weight=args.ssim_weight)
    baseline = None
    if val_images:
        baseline = constant_baseline_loss(train_images, val_images, loss_fn)
        print(f"validation: best constant prediction scores val_loss {baseline:.4f}; the model must beat it")
    elif args.val_fraction == 0:
        print("validation: skipped (disabled by --val-fraction 0)")
    else:
        print("validation: skipped (only one image pair; add more pairs to enable it)")

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

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * len(loader), pct_start=0.05
    )
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    args.out.mkdir(parents=True, exist_ok=True)
    history_path = args.out / "history.csv"
    history_path.unlink(missing_ok=True)
    history = []
    best_score, best_epoch = -float("inf"), None
    print(f"training {args.epochs} epochs x {len(loader)} steps; history -> {history_path}")
    for epoch in range(1, args.epochs + 1):
        start = time.perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        train_loss = train_one_epoch(model, loader, loss_fn, optimizer, scheduler, scaler, device, amp, epoch)
        row = {"epoch": epoch, "train_loss": train_loss}
        if val_images:
            row.update(evaluate(model, val_images, loss_fn, device, amp))
        row.update(lr=lr, seconds=time.perf_counter() - start)
        history.append(row)
        append_history(history_path, row)
        print(format_epoch(row, args.epochs))

        save_checkpoint(args.out / "last.pt", model, normalization=norm_kwargs, epoch=epoch)
        score = row.get("val_pearson_r", -row["train_loss"])
        if score > best_score:
            best_score, best_epoch = score, epoch
            save_checkpoint(args.out / "best.pt", model, normalization=norm_kwargs, epoch=epoch)

    print("summary:")
    for line in summarize(history, baseline, best_epoch):
        print(f"  {line}")


if __name__ == "__main__":
    main()
