import csv
import math
import re

import numpy as np
import pytest
import tifffile
import torch

import train
from dapi_unet import DapiLoss, DapiUNet, normalize_percentile

TINY = ["--steps-per-epoch", "2", "--batch-size", "2", "--patch-size", "32", "--base-channels", "4",
        "--depth", "2", "--workers", "0"]  # fmt: skip


def write_pairs(root, n, size=(48, 40), planes=2, mismatched=False):
    """Brightfield shows the DAPI nuclei plus noise; ``mismatched`` pairs each
    brightfield with an unrelated DAPI image, so nothing can be learned."""
    rng = np.random.default_rng(0)
    (root / "bf").mkdir()
    (root / "dapi").mkdir()
    for i in range(n):
        dapi = (rng.random(size) > 0.9).astype(np.float32)
        bf = np.stack([dapi * 0.5 + rng.normal(0, 0.1, size) for _ in range(planes)]).astype(np.float32)
        if mismatched:
            dapi = (rng.random(size) > 0.9).astype(np.float32)
        tifffile.imwrite(root / "bf" / f"img{i}.tif", bf, photometric="minisblack")
        tifffile.imwrite(root / "dapi" / f"img{i}.tif", dapi)
    return root / "bf", root / "dapi"


def run_train(tmp_path, n_pairs, *extra, epochs=2, mismatched=False):
    bf, dapi = write_pairs(tmp_path, n_pairs, mismatched=mismatched)
    out = tmp_path / "run"
    train.main(["--source-dir", str(bf), "--target-dir", str(dapi), "--out", str(out),
                "--epochs", str(epochs), *TINY, *extra])  # fmt: skip
    with (out / "history.csv").open() as f:
        return out, list(csv.DictReader(f))


def test_train_logs_train_and_val_loss_every_epoch(tmp_path, capsys):
    out, rows = run_train(tmp_path, n_pairs=4)
    assert [int(r["epoch"]) for r in rows] == [1, 2]
    for row in rows:
        for key in ("train_loss", "val_loss", "val_pearson_r", "val_ssim"):
            assert math.isfinite(float(row[key]))
    printed = capsys.readouterr().out
    for epoch in (1, 2):
        assert re.search(rf"^epoch {epoch}/2  train_loss \d\S*  val_loss \d", printed, re.M)
    assert "summary:" in printed and "best.pt is from epoch" in printed
    assert (out / "best.pt").exists() and (out / "last.pt").exists()


def test_train_warns_when_pairs_are_mismatched(tmp_path, capsys):
    run_train(tmp_path, n_pairs=4, epochs=3, mismatched=True)
    assert "barely correlated with DAPI" in capsys.readouterr().out


@pytest.mark.parametrize(("n_pairs", "extra"), [(1, ()), (3, ("--val-fraction", "0"))])
def test_train_skips_validation_without_validation_images(tmp_path, capsys, n_pairs, extra):
    _, rows = run_train(tmp_path, n_pairs, *extra)
    assert len(rows) == 2
    assert list(rows[0]) == ["epoch", "train_loss", "lr", "seconds"]
    assert math.isfinite(float(rows[0]["train_loss"]))
    printed = capsys.readouterr().out
    assert "validation: skipped" in printed and "validation was skipped" in printed


def test_train_rejects_bad_validation_pair_before_training(tmp_path):
    # Only the validation pair is broken, so this fails if just the training
    # images were checked (the run would crash after the first epoch instead).
    bf, dapi = write_pairs(tmp_path, 2)
    _, val_pairs = train.split_pairs(train.find_pairs(bf, dapi), 0.15, seed=0)
    tifffile.imwrite(val_pairs[0][1], np.zeros((10, 10), np.float32))
    with pytest.raises(SystemExit, match=f"data check failed: {val_pairs[0][0].name}"):
        train.main(["--source-dir", str(bf), "--target-dir", str(dapi), "--out", str(tmp_path / "run"),
                    "--epochs", "1", *TINY])  # fmt: skip
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


def test_validate_images_rejects_nearly_blank_image():
    # One hot pixel on a blank field: percentile normalization divides by eps.
    raw = np.zeros((1, 64, 64), np.float32)
    raw[0, 5, 5] = 4095
    with pytest.raises(ValueError, match="hot.tif: DAPI image is nearly blank"):
        train.validate_images([(np.random.rand(1, 64, 64), normalize_percentile(raw))], ["hot.tif"])
    healthy = normalize_percentile(np.random.rand(1, 64, 64) ** 4 * 4095)
    assert train.validate_images([(np.random.rand(1, 64, 64), healthy)], ["ok.tif"]) == []


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


def report(history, baseline=0.55, best_epoch=2):
    return "\n".join(train.summarize(history, baseline, best_epoch))


def test_summarize_warns_when_not_learning():
    flat = [
        {"epoch": 1, "train_loss": 0.5, "val_loss": 0.6, "val_pearson_r": 0.01},
        {"epoch": 2, "train_loss": 0.5, "val_loss": 0.6, "val_pearson_r": 0.02},
    ]
    text = report(flat)
    assert "training loss did not decrease" in text
    assert "never beat a constant prediction" in text
    assert "barely correlated" in text

    learning = [
        {"epoch": 1, "train_loss": 0.5, "val_loss": 0.4, "val_pearson_r": 0.6},
        {"epoch": 2, "train_loss": 0.2, "val_loss": 0.2, "val_pearson_r": 0.9},
    ]
    text = report(learning)
    assert "WARNING" not in text
    assert "best.pt is from epoch 2 (highest val_pearson_r 0.9000, val_loss 0.2000)" in text
    no_val = report([{"epoch": 1, "train_loss": 0.5}], baseline=None, best_epoch=1)
    assert "validation was skipped" in no_val and "WARNING" not in no_val


def test_summarize_flags_low_correlation_even_when_loss_beats_baseline():
    # Inverted or uncorrelated predictions can still have a lowish loss.
    history = [{"epoch": 1, "train_loss": 0.5, "val_loss": 0.3, "val_pearson_r": -0.9}]
    text = report(history, best_epoch=1)
    assert "never beat" not in text and "barely correlated" in text


def test_constant_baseline_is_the_best_constant():
    # Sparse nuclei: predicting the background (median) beats predicting the
    # mean, and the baseline must not be easier to beat than either.
    tgt = (np.random.rand(1, 64, 64) > 0.9).astype(np.float32)
    t = torch.from_numpy(tgt)[None]
    loss_fn = DapiLoss()
    baseline = train.constant_baseline_loss([(None, tgt)], [(None, tgt)], loss_fn)
    mean_loss = loss_fn(torch.full_like(t, float(tgt.mean())), t).item()
    background_loss = loss_fn(torch.zeros_like(t), t).item()
    assert baseline <= background_loss < mean_loss
    assert loss_fn(t, t).item() < baseline


def test_evaluate_stops_on_non_finite_predictions():
    class NanModel(torch.nn.Conv2d):
        def forward(self, x):
            return super().forward(x) * float("nan")

    images = [(np.random.rand(1, 32, 32).astype(np.float32), np.random.rand(1, 32, 32).astype(np.float32))]
    with pytest.raises(FloatingPointError, match="val_loss"):
        train.evaluate(NanModel(1, 1, 1), images, DapiLoss(), torch.device("cpu"), amp=False)


def test_loss_rejects_shape_mismatch():
    with pytest.raises(ValueError, match="shape"):
        DapiLoss()(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 9))
