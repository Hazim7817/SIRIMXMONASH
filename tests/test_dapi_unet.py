import numpy as np
import pytest
import torch
import torch.nn.functional as F
from torch import nn

from dapi_unet import (
    DapiLoss,
    DapiUNet,
    PairedPatchDataset,
    load_checkpoint,
    load_image,
    normalize_percentile,
    pearson_r,
    predict_tiled,
    save_checkpoint,
    save_image,
    ssim,
)


def small_model(**kwargs) -> DapiUNet:
    return DapiUNet(**{"base_channels": 8, "depth": 3, **kwargs})


@pytest.mark.parametrize("size", [(64, 64), (100, 77), (17, 130), (5, 9)])
def test_output_matches_arbitrary_input_size(size):
    model = small_model().eval()
    x = torch.randn(2, 1, *size)
    assert model(x).shape == (2, 1, *size)


@pytest.mark.parametrize("norm", ["batch", "group", "instance", "none"])
@pytest.mark.parametrize("attention", [False, True])
def test_variants_and_zstack_input(norm, attention):
    model = small_model(in_channels=5, norm=norm, attention=attention)
    x = torch.randn(2, 5, 48, 48)
    y = model(x)
    assert y.shape == (2, 1, 48, 48)
    y.mean().backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_default_model_shape():
    model = DapiUNet().eval()
    with torch.no_grad():
        assert model(torch.randn(1, 1, 256, 256)).shape == (1, 1, 256, 256)


def test_invalid_config():
    with pytest.raises(ValueError):
        DapiUNet(norm="layer")
    with pytest.raises(ValueError):
        DapiUNet(depth=0)


def test_checkpoint_roundtrip(tmp_path):
    model = small_model(in_channels=3, attention=True).eval()
    save_checkpoint(tmp_path / "m.pt", model, normalization={"low": 1.0, "high": 99.8})
    loaded, ckpt = load_checkpoint(tmp_path / "m.pt")
    loaded.eval()
    x = torch.randn(1, 3, 40, 40)
    torch.testing.assert_close(model(x), loaded(x))
    assert ckpt["normalization"] == {"low": 1.0, "high": 99.8}


def test_ssim_and_pearson_identities():
    x = torch.rand(3, 1, 32, 32)
    torch.testing.assert_close(ssim(x, x), torch.tensor(1.0))
    torch.testing.assert_close(pearson_r(x, 2 * x + 1), torch.ones(3))
    torch.testing.assert_close(pearson_r(x, -x), -torch.ones(3))
    assert ssim(x, torch.rand_like(x)) < 0.5
    assert DapiLoss()(x, x).item() == pytest.approx(0.0, abs=1e-6)


def test_tiled_prediction_matches_pointwise_model():
    # A pointwise model has no context, so stitching must reproduce it exactly.
    model = nn.Conv2d(2, 1, 1)
    image = np.random.rand(2, 300, 270).astype(np.float32)
    expected = model(torch.from_numpy(image)[None])[0].detach().numpy()
    tiled = predict_tiled(model, image, tile_size=128, overlap=32, batch_size=3)
    np.testing.assert_allclose(tiled, expected, rtol=1e-5, atol=1e-5)


def test_tiled_prediction_unet_restores_mode():
    model = small_model().train()
    image = np.random.rand(150, 90).astype(np.float32)
    pred = predict_tiled(model, image, tile_size=64, overlap=16)
    assert pred.shape == (1, 150, 90)
    assert np.isfinite(pred).all()
    assert model.training


def test_normalize_percentile():
    img = np.linspace(0, 1000, 10001, dtype=np.float32)
    out = normalize_percentile(img, low=1, high=99)
    assert out.dtype == np.float32
    assert np.percentile(out, 1) == pytest.approx(0, abs=1e-4)
    assert np.percentile(out, 99) == pytest.approx(1, abs=1e-4)


@pytest.mark.parametrize("suffix", [".tif", ".npy"])
def test_image_io_roundtrip(tmp_path, suffix):
    img = np.random.rand(3, 20, 30).astype(np.float32)
    save_image(tmp_path / f"x{suffix}", img)
    np.testing.assert_array_equal(load_image(tmp_path / f"x{suffix}"), img)


def test_dataset_applies_same_transform_to_both_images():
    source = np.random.rand(2, 80, 60).astype(np.float32)
    target = source[:1] * 3 + source[1:]
    ds = PairedPatchDataset([source], [target], patch_size=48, samples_per_epoch=20, intensity_jitter=0)
    assert len(ds) == 20
    for i in range(len(ds)):
        src, tgt = ds[i]
        assert src.shape == (2, 48, 48) and tgt.shape == (1, 48, 48)
        assert src.dtype == tgt.dtype == torch.float32
        torch.testing.assert_close(tgt, src[:1] * 3 + src[1:])


def test_dataset_pads_small_images():
    ds = PairedPatchDataset([np.random.rand(30, 40)], [np.random.rand(30, 40)], patch_size=64)
    src, tgt = ds[0]
    assert src.shape == tgt.shape == (1, 64, 64)


def test_model_learns_synthetic_nuclei():
    # Brightfield-like input: nuclei appear as faint rings (edges) on a noisy
    # background; the target is the filled, blurred nucleus.
    torch.manual_seed(0)
    yy, xx = torch.meshgrid(torch.arange(64.0), torch.arange(64.0), indexing="ij")
    masks = []
    for _ in range(8):
        img = torch.zeros(64, 64)
        for cy, cx in torch.randint(8, 56, (4, 2)).tolist():
            img = torch.maximum(img, (((yy - cy) ** 2 + (xx - cx) ** 2) < 36).float())
        masks.append(img)
    target = F.avg_pool2d(torch.stack(masks)[:, None], 3, 1, 1)
    edges = (F.max_pool2d(target, 3, 1, 1) - target).abs()
    source = edges + 0.05 * torch.randn_like(edges)

    model = small_model(depth=2, dropout=0.0)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    loss_fn = DapiLoss()
    losses = []
    for _ in range(60):
        opt.zero_grad()
        loss = loss_fn(model(source), target)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0]
