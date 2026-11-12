"""Tests for isp.cfa and isp.demosaic (lecture 3)."""
import numpy as np
import pytest
import cv2

from isp import cfa, demosaic as D, scenes, utils

# OpenCV names Bayer codes by the 2x2 tile read from the *second* row/column; found
# empirically below (test_bilinear_matches_opencv): our 'RGGB' is cv2.COLOR_BayerBG2RGB.
CV_CODE = {"RGGB": "BG", "BGGR": "RG", "GRBG": "GB", "GBRG": "GR"}

PHOTO = scenes.photo("astronaut", 128)


# ---------------------------------------------------------------- cfa
@pytest.mark.parametrize("pattern", cfa.PATTERNS)
def test_masks_partition(pattern):
    mk = cfa.masks((16, 20), pattern)
    assert mk.shape == (16, 20, 3) and mk.dtype == bool
    assert np.all(mk.sum(axis=2) == 1)
    frac = mk.mean(axis=(0, 1))
    assert np.allclose(frac, [0.25, 0.5, 0.25]) if pattern in ("RGGB", "BGGR", "GRBG", "GBRG") else True


@pytest.mark.parametrize("pattern", cfa.PATTERNS)
def test_mosaic_picks_right_samples(pattern):
    img = PHOTO[:16, :16]
    m = cfa.mosaic(img, pattern)
    assert m.shape == (16, 16) and m.dtype == np.float32
    ch = {"R": 0, "G": 1, "B": 2}
    for k, letter in enumerate(pattern):
        i, j = divmod(k, 2)
        assert np.array_equal(m[i::2, j::2], img[i::2, j::2, ch[letter]])


def test_rggb_layout():
    img = np.zeros((4, 4, 3), np.float32)
    img[..., 0], img[..., 1], img[..., 2] = 0.1, 0.5, 0.9
    m = cfa.mosaic(img, "RGGB")
    assert np.allclose(m[:2, :2], [[0.1, 0.5], [0.5, 0.9]])


@pytest.mark.parametrize("pattern", cfa.PATTERNS)
def test_colorize(pattern):
    m = cfa.mosaic(PHOTO[:16, :16], pattern)
    c = cfa.colorize(m, pattern)
    mk = cfa.masks(m.shape, pattern)
    assert np.array_equal(c.sum(axis=2), m)               # one non-zero channel per site
    assert np.all(c[~mk] == 0)
    assert np.array_equal(c[mk], m.ravel())                # sample sits in its filter channel


def test_display_tiles():
    assert cfa.SONY_RGBE_TILE.shape == (4, 4) and cfa.KODAK_RGBW_TILE.shape == (4, 4)
    assert (cfa.KODAK_RGBW_TILE == "W").mean() == 0.5
    assert cfa.tile_colors(cfa.SONY_RGBE_TILE, 2).shape == (8, 8, 3)


# ---------------------------------------------------------------- demosaic basics
@pytest.mark.parametrize("method", D.METHODS)
@pytest.mark.parametrize("pattern", cfa.PATTERNS)
def test_constant_colour_exact(method, pattern):
    col = np.array([0.2, 0.55, 0.8], np.float32)
    img = np.broadcast_to(col, (32, 32, 3)).copy()
    out = D.demosaic(cfa.mosaic(img, pattern), pattern, method)
    assert out.shape == img.shape and out.dtype == np.float32
    assert np.abs(out[4:-4, 4:-4] - img[4:-4, 4:-4]).max() < 1e-5


@pytest.mark.parametrize("method", [m for m in D.METHODS if m != "nearest"])
@pytest.mark.parametrize("pattern", ["RGGB", "GBRG"])
def test_keeps_measured_samples(method, pattern):
    m = cfa.mosaic(PHOTO[:64, :64], pattern)
    out = D.demosaic(m, pattern, method)
    mk = cfa.masks(m.shape, pattern)
    assert np.abs(out[mk] - np.repeat(m[:, :, None], 3, 2)[mk]).max() < 1e-5


def test_nearest_replicates_tile():
    m = cfa.mosaic(PHOTO[:16, :16], "RGGB")
    out = D.demosaic(m, "RGGB", "nearest")
    assert np.array_equal(out[0, 0], [m[0, 0], m[0, 1], m[1, 1]])
    assert np.array_equal(out[1, 1], out[0, 0])


def test_bad_args():
    with pytest.raises(ValueError):
        D.demosaic(np.zeros((8, 8), np.float32), "RGGB", "nope")
    with pytest.raises(ValueError):
        D.demosaic(np.zeros((8, 8), np.float32), "XXXX")


@pytest.mark.parametrize("pattern", cfa.PATTERNS)
def test_bilinear_matches_opencv(pattern):
    """Our bilinear == cv2 Bayer bilinear (16-bit) away from the border; 'RGGB' <-> COLOR_BayerBG2RGB."""
    img = PHOTO
    m = cfa.mosaic(img, pattern)
    m16 = np.round(m * 65535).astype(np.uint16)
    ours = D.demosaic(m16.astype(np.float32) / 65535, pattern, "bilinear")
    errs = {}
    for name in ("BG", "RG", "GB", "GR"):
        ref = cv2.cvtColor(m16, getattr(cv2, f"COLOR_Bayer{name}2RGB")).astype(np.float32) / 65535
        errs[name] = np.abs(ours - ref)[4:-4, 4:-4].max()
    best = min(errs, key=errs.get)
    assert best == CV_CODE[pattern]
    assert errs[best] < 1.0 / 255


# ---------------------------------------------------------------- quality
def test_median_beats_bilinear_on_zone_plate():
    z = scenes.zone_plate(128)
    m = cfa.mosaic(z)
    fc = {k: D.false_color(D.demosaic(m, "RGGB", k), z) for k in ("bilinear", "median")}
    assert fc["median"] < fc["bilinear"]


def test_malvar_beats_bilinear_psnr():
    ref = scenes.photo("astronaut", 128)
    m = cfa.mosaic(ref)
    p = {k: utils.psnr(D.demosaic(m, "RGGB", k), ref) for k in ("bilinear", "malvar")}
    assert p["malvar"] > p["bilinear"] + 1.0


def test_median_size_kwarg_and_false_color_zero():
    m = cfa.mosaic(PHOTO[:64, :64])
    a = D.demosaic(m, "RGGB", "median", median_size=3)
    b = D.demosaic(m, "RGGB", "median", median_size=7)
    assert not np.allclose(a, b)
    assert D.false_color(PHOTO, PHOTO) == 0.0


def test_false_color_ignores_luminance_blur():
    ref = scenes.photo("astronaut", 64)
    only_y = D.blur_channel(ref, "Y", 2.0)
    only_c = D.blur_channel(ref, "CbCr", 2.0)
    assert D.false_color(only_y, ref) < D.false_color(only_c, ref)


def test_blur_channel_touches_only_that_channel():
    img = 0.3 + 0.4 * utils.linear_to_srgb(scenes.photo("astronaut", 64))   # keep away from the gamut edge
    y0 = utils.rgb_to_ycbcr(img)
    for ch, k in (("Y", 0), ("Cb", 1), ("Cr", 2)):
        out = utils.rgb_to_ycbcr(D.blur_channel(img, ch, 2.0))
        others = [i for i in range(3) if i != k]
        assert np.abs(out[..., others] - y0[..., others]).max() < 2e-3
        assert np.abs(out[..., k] - y0[..., k]).max() > 1e-2


# ---------------------------------------------------------------- colour sensing technologies
def test_three_sensor_exact():
    s = scenes.photo("coffee", 64)
    assert np.array_equal(D.three_sensor(s), s)


def test_field_sequential():
    s = scenes.photo("astronaut", 64)
    assert np.allclose(D.field_sequential(s), s)
    mov = D.field_sequential(s, motion=(0.0, 3.0))
    assert D.false_color(mov, s) > 0.02
    ref_mov = D.field_sequential(s, motion=(0.0, 0.0))
    assert D.false_color(ref_mov, s) < 1e-6


def test_stacked_exact_and_noisy():
    s = scenes.photo("astronaut", 64)
    assert np.abs(D.stacked_sensor(s) - s).max() < 1e-5
    sig = 0.01
    e3 = np.sqrt(np.mean((D.three_sensor(s, noise=sig, seed=1) - s) ** 2))
    es = np.sqrt(np.mean((D.stacked_sensor(s, noise=sig, seed=1) - s) ** 2))
    assert es > 1.5 * e3
    assert abs(e3 - sig) < 1e-3


def test_cfa_sensor_and_sense():
    s = scenes.photo("astronaut", 64)
    out = D.sense(s, "cfa", method="malvar")
    assert out.shape == s.shape
    assert utils.psnr(out, s) < 60            # not exact: 2/3 of the data was interpolated
    assert np.array_equal(D.sense(s, "three_sensor"), s)
