"""Tests for isp.whitebalance (lecture 4, first half)."""
import numpy as np
import pytest

from isp import scenes
from isp import whitebalance as W


# ---------------------------------------------------------------- CMF / chromaticity
def test_cmf_equal_energy_white_is_one_third():
    c = W.cie_cmf(np.arange(360.0, 831.0)).sum(axis=0)
    x, y = (c / c.sum())[:2]
    assert abs(x - 1 / 3) < 0.01 and abs(y - 1 / 3) < 0.01


def test_spectral_locus_peak_and_bounds():
    loc = W.spectral_locus_xy()
    i = np.argmax(loc[:, 1])
    assert 510 <= 420 + i <= 535
    assert abs(loc[i, 1] - 0.83) < 0.02
    assert loc.min() > -0.01 and loc.max() < 0.85
    assert abs(loc[0, 0] - 0.17) < 0.03 and abs(loc[-1, 0] - 0.72) < 0.02   # violet end, red end


def test_rgb_chromaticity():
    c = W.rgb_chromaticity(np.array([[[0.2, 0.2, 0.2], [1.0, 0.0, 0.0], [0, 0, 0]]], np.float32))
    assert np.allclose(c[0, 0], [1 / 3, 1 / 3], atol=1e-6)
    assert np.allclose(c[0, 1], [1, 0], atol=1e-6)
    assert np.allclose(c[0, 2], [1 / 3, 1 / 3], atol=1e-6)


def test_srgb_primaries_xy():
    p = W.srgb_gamut_xy()
    assert np.allclose(p, [[0.64, 0.33], [0.30, 0.60], [0.15, 0.06]], atol=0.005)


# ---------------------------------------------------------------- colour temperature
def test_d65_xy():
    x, y = W.cct_to_xy(6504)
    assert abs(x - 0.3127) < 0.01 and abs(y - 0.3290) < 0.01


def test_cct_xy_known_points():
    x, y = W.cct_to_xy(3200)                               # Planckian locus: 0.4232, 0.3990
    assert abs(x - 0.4232) < 0.01 and abs(y - 0.3990) < 0.01


def test_cct_round_trip():
    for T in np.linspace(2500, 10000, 16):
        T2 = W.xy_to_cct(*W.cct_to_xy(T))
        assert abs(T2 - T) / T < 0.02
    assert abs(W.mccamy_cct(*W.cct_to_xy(5000)) - 5000) / 5000 < 0.02


def test_illuminant_rgb_colours():
    a = W.illuminant_rgb(3200)
    assert a[1] == pytest.approx(1.0) and a[0] > a[1] > a[2]
    b = W.illuminant_rgb(9000)
    assert b[2] > b[1] > b[0]
    assert np.allclose(W.illuminant_rgb(6504), 1.0, atol=0.08)


def test_presets():
    assert W.PRESETS["incandescent"] == 3200 and W.PRESETS["fluorescent"] == 4000
    assert W.PRESETS["d65"] in (6500, 6504) and W.PRESETS["shade"] == 7500


# ---------------------------------------------------------------- relight and gains
def test_apply_illuminant_roundtrip():
    img = scenes.load("astronaut", 64)
    for T in (3200, 4000, 7500):
        lit = W.apply_illuminant(img, T)
        assert lit.dtype == np.float32 and lit.min() >= 0 and lit.max() <= 1 + 1e-6
        back = W.apply_gains(lit, W.gains_for_illuminant(T))
        e = W._illum_relative(T)
        assert np.allclose(back, img / e.max(), atol=1e-5)        # documented global scale
    assert np.allclose(W.apply_illuminant(img, 6504), img, atol=1e-6)


def test_apply_illuminant_rgb_triple():
    img = np.full((4, 4, 3), 0.5, np.float32)
    out = W.apply_illuminant(img, (1.0, 0.5, 0.25))
    assert np.allclose(out[0, 0], [0.5, 0.25, 0.125])
    assert np.allclose(W.gains_for_illuminant((1.0, 0.5, 0.25)), [0.5, 1, 2])


def test_apply_gains_clip():
    img = np.full((2, 2, 3), 0.8, np.float32)
    assert W.apply_gains(img, [2, 1, 1]).max() == pytest.approx(1.6)
    assert W.apply_gains(img, [2, 1, 1], clip=True).max() == 1.0


# ---------------------------------------------------------------- estimation
def test_from_patch_slide9_formula():
    img = np.zeros((8, 8, 3), np.float32)
    img[2:6, 2:6] = [0.6, 0.3, 0.15]                       # patch mean: R+G+B = 1.05
    s = W.from_patch(img, (2, 6, 2, 6), normalize_green=False)
    assert np.allclose(s, [1.05 / 1.8, 1.05 / 0.9, 1.05 / 0.45], atol=1e-6)
    g = W.from_patch(img, (2, 6, 2, 6))
    assert g[1] == pytest.approx(1.0) and np.allclose(g, [0.5, 1, 2], atol=1e-6)
    assert np.allclose(W.apply_gains(img, s)[3, 3], 0.35, atol=1e-6)


def test_from_patch_on_chart_neutral():
    chart = scenes.patches(256)
    boxes = scenes.patch_boxes(256)
    for T in (3200, 7500):
        lit = W.apply_illuminant(chart, T)
        gains = W.from_patch(lit, boxes[20])               # a grey patch
        err = W.angular_error(W.gains_to_illuminant(gains), W._illum_relative(T))
        assert err < 0.5
        r0, r1, c0, c1 = boxes[20]
        m = W.apply_gains(lit, gains)[r0:r1, c0:c1].reshape(-1, 3).mean(0)
        assert np.ptp(m) < 1e-4


def test_gray_world_exact_on_grey_mean():
    rs = np.random.default_rng(0)
    img = rs.random((64, 64, 3)).astype(np.float32)
    img *= 0.4 / img.reshape(-1, 3).mean(0)                # per-channel mean exactly 0.4
    lit = W.apply_illuminant(img, 3200)
    gains = W.gray_world(lit)
    assert W.angular_error(W.gains_to_illuminant(gains), W._illum_relative(3200)) < 1.0
    assert gains[1] == pytest.approx(1.0)


def test_gray_world_fails_on_sunset():
    sun = scenes.sunset(128)
    chart = scenes.patches(128)
    ref = np.ones(3)                                       # scenes are lit by neutral light
    e_sun = W.angular_error(W.gains_to_illuminant(W.gray_world(sun)), ref)
    e_chart = W.angular_error(W.gains_to_illuminant(W.gray_world(chart)), ref)
    assert e_sun > 15 and e_sun > 2 * e_chart


def test_white_patch_works_on_chart_and_fails_on_saturated_colour():
    chart = W.apply_illuminant(scenes.patches(256), 3200)
    truth = W._illum_relative(3200)
    assert W.angular_error(W.gains_to_illuminant(W.white_patch(chart)), truth) < 1.0
    # brightest region = saturated orange area: R and G clipped at 1, B low
    img = np.full((64, 64, 3), 0.3, np.float32)
    img[:12, :12] = [1.0, 1.0, 0.2]
    lit = W.apply_illuminant(img, 5000)
    err = W.angular_error(W.gains_to_illuminant(W.white_patch(lit)), W._illum_relative(5000))
    assert err > 10


# ---------------------------------------------------------------- mosaic
@pytest.mark.parametrize("pattern", ["RGGB", "BGGR", "GRBG", "GBRG"])
def test_apply_gains_mosaic_matches_rgb(pattern):
    rs = np.random.default_rng(1)
    img = rs.random((16, 20, 3)).astype(np.float32)
    gains = np.array([1.9, 1.0, 0.6], np.float32)
    idx = np.array([["RGB".index(c) for c in pattern[:2]], ["RGB".index(c) for c in pattern[2:]]])
    ch = np.tile(idx, (8, 10))                             # channel index of every pixel
    rr, cc = np.mgrid[0:16, 0:20]

    def mosaic(a):
        return a[rr, cc, ch]

    got = W.apply_gains_mosaic(mosaic(img), gains, pattern)
    assert np.allclose(got, mosaic(W.apply_gains(img, gains)), atol=1e-6)
    assert got.dtype == np.float32


def test_apply_gains_mosaic_bad_pattern():
    with pytest.raises(ValueError):
        W.apply_gains_mosaic(np.zeros((4, 4), np.float32), [1, 1, 1], "RGXB")


# ---------------------------------------------------------------- metric
def test_angular_error():
    assert W.angular_error([1, 1, 1], [2, 2, 2]) == pytest.approx(0.0, abs=1e-5)
    assert W.angular_error([1, 0, 0], [0, 1, 0]) == pytest.approx(90.0)
    assert W.angular_error([1, 1, 1], [1, 0, 0]) == pytest.approx(54.7356, abs=1e-3)


@pytest.mark.parametrize("name", sorted(W.PRESETS))
def test_every_preset_relights(name):
    """Every named light source is inside the sRGB gamut, so it can relight a scene."""
    rgb = W.illuminant_rgb(W.PRESETS[name])
    assert np.all(rgb > 0)
    img = W.apply_illuminant(scenes.patches(64), W.PRESETS[name])
    assert np.isfinite(img).all() and img.min() >= 0 and img.max() <= 1
