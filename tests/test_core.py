"""Core contract: utils, scenes and the Pipeline skeleton."""
import numpy as np
import pytest

import isp
from isp import scenes, utils
from isp.pipeline import Pipeline, Stage


def test_srgb_round_trip():
    x = np.linspace(0, 1, 101, dtype=np.float32)
    assert np.allclose(utils.linear_to_srgb(utils.srgb_to_linear(x)), x, atol=1e-5)
    assert utils.srgb_to_linear(0.5) == pytest.approx(0.2140, abs=1e-3)


def test_ycbcr_round_trip_and_neutral():
    rgb = utils.rng(0).random((8, 8, 3)).astype(np.float32)
    assert np.allclose(utils.ycbcr_to_rgb(utils.rgb_to_ycbcr(rgb)), rgb, atol=1e-5)
    ycc = utils.rgb_to_ycbcr(np.full((2, 2, 3), 0.4, np.float32))
    assert np.allclose(ycc[..., 0], 0.4, atol=1e-6) and np.allclose(ycc[..., 1:], 0.5, atol=1e-6)


def test_metrics():
    a = np.zeros((4, 4), np.float32)
    assert utils.psnr(a, a) == float("inf")
    assert utils.psnr(a, a + 0.1) == pytest.approx(20.0, abs=1e-4)
    assert utils.snr_db(200, 1) - utils.snr_db(100, 1) == pytest.approx(6.02, abs=0.01)
    assert utils.gaussian_kernel(1.5).sum() == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("name", scenes.NAMES)
def test_scenes(name):
    img = scenes.load(name, size=100)                     # rounded down to 96
    assert img.shape == (96, 96, 3) and img.dtype == np.float32
    assert img.min() >= 0.0 and img.max() <= 1.0


def test_patch_boxes_neutral_row():
    img = scenes.patches(192)
    boxes = scenes.patch_boxes(192)
    assert len(boxes) == 24
    for b in boxes[18:]:
        p = utils.crop(img, b)
        assert p.std() < 0.01 and np.ptp(p.reshape(-1, 3).mean(0)) < 0.01   # uniform and grey
    assert utils.crop(img, boxes[14]).reshape(-1, 3).mean(0).argmax() == 0  # red patch


def _gain(img, meta, g=2.0):
    meta["gain"] = meta.get("gain", 1.0) * g
    return img * g, meta


def _offset(img, meta, o=1.0):
    return img + o, meta


def test_pipeline_run_toggle_move():
    p = Pipeline([Stage("gain", _gain, {"g": 2.0}, group="adc"), Stage("offset", _offset, {"o": 1.0})])
    x = np.ones((2, 2), np.float32)
    meta = {"domain": "scene"}
    res = p.run(x, meta, metrics=lambda im, m, n: {"mean": float(im.mean())})
    assert res.names == ["gain", "offset"] and len(res) == 2
    assert res["gain"].metrics["mean"] == 2.0 and res.final.metrics["mean"] == 3.0
    assert res["gain"].meta["gain"] == 2.0 and meta == {"domain": "scene"}   # input meta untouched
    assert res.table()[0]["group"] == "adc"

    q = p.copy().set("gain", g=3.0).move("offset", before="gain")           # (1 + 1) * 3
    assert q.run(x).final.image.mean() == 6.0
    assert p.run(x).final.image.mean() == 3.0                                # original unchanged

    off = p.copy().enable("gain", False).run(x)
    assert off["gain"].enabled is False and off.final.image.mean() == 2.0


def test_pipeline_rejects_bad_input():
    with pytest.raises(ValueError):
        Pipeline([Stage("a", _gain), Stage("a", _gain)])
    with pytest.raises(ValueError):
        Stage("a", _gain, group="nope")


def test_lazy_modules():
    assert isp.scenes is scenes
    with pytest.raises(AttributeError):
        isp.not_a_module
