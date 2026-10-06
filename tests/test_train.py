import csv
import math

import numpy as np
import pytest
import tifffile
import torch

import train
from dapi_unet import DapiLoss, DapiUNet


def write_pairs(root, n, size=(48, 40), planes=2):
    rng = np.random.default_rng(0)
    (root / "bf").mkdir()
    (root / "dapi").mkdir()
    for i in range(n):
        dapi = (rng.random(size) > 0.9).astype(np.float32)
        bf = np.stack([dapi * 0.5 + rng.normal(0, 0.1, size) for _ in range(planes)]).astype(np.float32)
        tifffile.imwrite(root / "bf" / f"img{i}.tif", bf, photometric="minisblack")
        tifffile.imwrite(root / "dapi" / f"img{i}.tif", dapi)
    return root / "bf", root / "dapi"


def run_train(tmp_path, n_pairs, *extra):
    bf, dapi = write_pairs(tmp_path, n_pairs)
    out = tmp_path / "run"
    train.main([
        "--source-dir", str(bf), "--target-dir", str(dapi), "--out", str(out),
        "--epochs", "2", "--steps-per-epoch", "2", "--batch-size", "2", "--patch-size", "32",
        "--base-channels", "4", "--depth", "2", "--workers", "0", *extra,
    ])  # fmt: skip
    with (out / "history.csv").open() as f:
        return out, list(csv.DictReader(f))


def test_train_logs_train_and_val_loss_every_epoch(tmp_path, capsys):
    out, rows = run_train(tmp_path, n_pairs=4)
    assert [int(r["epoch"]) for r in rows] == [1, 2]
    for row in rows:
        for key in ("train_loss", "val_loss", "val_pearson_r", "val_ssim"):
            assert math.isfinite(float(row[key]))
    printed = capsys.readouterr().out
    assert "epoch 2/2" in printed and "val_loss" in printed and "baseline" in printed
    assert (out / "best.pt").exists() and (out / "last.pt").exists()


@pytest.mark.parametrize(("n_pairs", "extra"), [(1, ()), (3, ("--val-fraction", "0"))])
def test_train_skips_validation_without_validation_images(tmp_path, capsys, n_pairs, extra):
    _, rows = run_train(tmp_path, n_pairs, *extra)
    assert len(rows) == 2
    assert "val_loss" not in rows[0]
    assert math.isfinite(float(rows[0]["train_loss"]))
    assert "validation: skipped" in capsys.readouterr().out


def test_train_rejects_mismatched_validation_pair_before_training(tmp_path):
    bf, dapi = write_pairs(tmp_path, 2)
    tifffile.imwrite(dapi / "img0.tif", np.zeros((10, 10), np.float32))
    tifffile.imwrite(dapi / "img1.tif", np.zeros((10, 10), np.float32))
    with pytest.raises(SystemExit, match="data check failed"):
        train.main(["--source-dir", str(bf), "--target-dir", str(dapi), "--out", str(tmp_path / "run")])
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize(
    ("n", "fraction", "expected_val"),
    [(10, 0.15, 2), (10, 0.0, 0), (1, 0.5, 0), (2, 0.15, 1), (3, 0.9, 2)],
)
def test_split_pairs(n, fraction, expected_val):
    train_pairs, val_pairs = train.split_pairs(list(range(n)), fraction, seed=0)
    assert len(val_pairs) == expected_val
    assert len(train_pairs) >= 1
    assert sorted(train_pairs + val_pairs) == list(range(n))


@pytest.mark.parametrize("fraction", [-0.1, 1.0])
def test_split_pairs_rejects_invalid_fraction(fraction):
    with pytest.raises(SystemExit):
        train.split_pairs([1, 2, 3], fraction, seed=0)


def test_validate_images():
    good = (np.random.rand(2, 8, 8), np.random.rand(1, 8, 8))
    assert train.validate_images([good], ["a"]) == []
    with pytest.raises(ValueError, match="size"):
        train.validate_images([(np.random.rand(2, 8, 8), np.random.rand(1, 8, 9))], ["a"])
    with pytest.raises(ValueError, match="NaN"):
        train.validate_images([(np.full((2, 8, 8), np.nan), np.random.rand(1, 8, 8))], ["a"])
    with pytest.raises(ValueError, match="channel counts"):
        train.validate_images([good, (np.random.rand(3, 8, 8), np.random.rand(1, 8, 8))], ["a", "b"])
    warnings = train.validate_images([(np.random.rand(2, 8, 8), np.zeros((1, 8, 8)))], ["blank"])
    assert len(warnings) == 1 and "blank" in warnings[0]


def test_train_one_epoch_stops_on_nan_loss():
    model = DapiUNet(base_channels=4, depth=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    scaler = torch.amp.GradScaler("cpu", enabled=False)
    loader = [(torch.rand(2, 1, 32, 32), torch.rand(2, 1, 32, 32))]

    def nan_loss(pred, target):
        return (pred * float("nan")).mean()

    with pytest.raises(FloatingPointError, match="epoch 3, step 1"):
        train.train_one_epoch(model, loader, nan_loss, optimizer, scheduler, scaler, torch.device("cpu"), False, 3)


def test_summarize_warns_when_not_learning():
    flat = [
        {"epoch": 1, "train_loss": 0.5, "val_loss": 0.6},
        {"epoch": 2, "train_loss": 0.5, "val_loss": 0.6},
    ]
    report = "\n".join(train.summarize(flat, baseline=0.55))
    assert "training loss did not decrease" in report
    assert "never beat the constant-prediction baseline" in report

    learning = [
        {"epoch": 1, "train_loss": 0.5, "val_loss": 0.4},
        {"epoch": 2, "train_loss": 0.2, "val_loss": 0.2},
    ]
    assert "WARNING" not in "\n".join(train.summarize(learning, baseline=0.55))
    assert "skipped" in "\n".join(train.summarize([{"epoch": 1, "train_loss": 0.5}], baseline=None))


def test_constant_baseline_loss_is_finite_and_beaten_by_perfect_prediction():
    tgt = (np.random.rand(1, 32, 32) > 0.8).astype(np.float32)
    loss_fn = DapiLoss()
    baseline = train.constant_baseline_loss([(None, tgt)], [(None, tgt)], loss_fn)
    perfect = loss_fn(torch.from_numpy(tgt)[None], torch.from_numpy(tgt)[None]).item()
    assert math.isfinite(baseline) and perfect < baseline


def test_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        DapiLoss()(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 9))
