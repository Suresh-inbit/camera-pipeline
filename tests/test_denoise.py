"""Tests for isp.denoise (lecture 5, denoising).  Small images, seed 0."""
import numpy as np
import cv2
import pytest

from isp import denoise as D, scenes
from isp.utils import psnr, rng, rgb_to_ycbcr


def add_noise(img, a=0.01, b=0.01, seed=0):
    """Signal-dependent Poisson-Gaussian noise on linear data: var = a * signal + b^2."""
    r = rng(seed)
    s = np.clip(img, 0, None)
    return (img + np.sqrt(a * s + b * b) * r.standard_normal(img.shape)).astype(np.float32)


def noisy_step(h=48, w=96, lo=0.3, hi=0.7, sigma=0.03, seed=0):
    clean = np.full((h, w), lo, np.float32)
    clean[:, w // 2:] = hi
    return clean, (clean + sigma * rng(seed).standard_normal(clean.shape)).astype(np.float32)


@pytest.fixture(scope="module")
def photo():
    return scenes.photo("astronaut", 96)


# ---------------------------------------------------------------- bilateral limits
def test_bilateral_to_gaussian_for_large_sigma_r(photo):
    x = add_noise(photo)
    r = 6
    a = D.bilateral(x, 2.0, 1e6, radius=r)
    b = D.gaussian(x, 2.0, radius=r)
    assert np.abs(a - b).max() < 1e-3
    g = x[..., 1]
    assert np.abs(D.bilateral(g, 1.5, 1e6, radius=5) - D.gaussian(g, 1.5, radius=5)).max() < 1e-3


def test_bilateral_to_identity_for_small_sigma_r(photo):
    x = add_noise(photo)
    out = D.bilateral(x, 2.0, 1e-7)
    assert np.abs(out - x).max() < 1e-5


def test_bilateral_constant_shape_dtype_range(photo):
    c = np.full((20, 24, 3), 0.37, np.float32)
    assert np.allclose(D.bilateral(c, 2.0, 0.1), c, atol=1e-6)
    x = add_noise(photo)
    for im in (x, x[..., 0]):
        out = D.bilateral(im, 2.0, 0.1)
        assert out.shape == im.shape and out.dtype == np.float32
        assert out.min() >= im.min() - 1e-6 and out.max() <= im.max() + 1e-6


def test_guide_changes_weights():
    clean, x = noisy_step()
    # a clean guide gives a perfectly sharp edge even for a noisy image
    out = D.bilateral(x, 2.0, 0.1, guide=clean)
    assert abs(out[:, 47].mean() - 0.3) < 0.01 and abs(out[:, 48].mean() - 0.7) < 0.01


def test_weight_window_at_edge():
    clean, _ = noisy_step()
    w = D.bilateral_weights(clean, (20, 47), 3.0, 0.1)      # last pixel of the dark side
    g = D.bilateral_weights(clean, (20, 47), 3.0, np.inf)
    r = w.shape[0] // 2
    assert w[r, r] == pytest.approx(1.0)
    assert w[:, r + 1:].max() < 1e-3 and g[:, r + 1:].max() > 0.1   # bright side cut off


# ---------------------------------------------------------------- vs OpenCV
def test_matches_opencv_gray(photo):
    g = add_noise(photo[..., 1])
    mine = D.bilateral(g, 2.0, 0.1, radius=5, window="circle")
    ref = cv2.bilateralFilter(g, 11, 0.1, 2.0)
    assert np.abs(mine - ref).max() < 1e-4


def test_matches_opencv_color(photo):
    """OpenCV uses a circular window and, for 3 channels, the *L1* colour distance
    (sum |dc|, squared) instead of the L2 distance on the slide.  With
    window='circle', color_dist='l1' we reproduce it; the default L2 differs slightly."""
    x = add_noise(photo)
    ref = cv2.bilateralFilter(x, 11, 0.1, 2.0)
    mine = D.bilateral(x, 2.0, 0.1, radius=5, window="circle", color_dist="l1")
    assert np.abs(mine - ref).max() < 1e-4
    l2 = D.bilateral(x, 2.0, 0.1, radius=5, window="circle")
    inner = (slice(8, -8), slice(8, -8))
    assert 1e-4 < np.abs(l2 - ref)[inner].max() < 0.1


# ---------------------------------------------------------------- edge vs noise
def test_edge_preserved_noise_removed():
    clean, x = noisy_step()
    sd = 2.0
    gau = D.gaussian(x, sd)
    bil = D.bilateral(x, sd, 0.1)
    flat = (slice(8, -8), slice(6, 30))                    # away from borders and the edge
    s_in, s_g, s_b = (np.std(a[flat]) for a in (x, gau, bil))
    assert s_g < 0.4 * s_in and s_b < 0.4 * s_in           # both remove noise
    assert 0.6 < s_b / s_g < 1.6                           # about equally well
    # max slope of the row-averaged profile
    slope = lambda a: np.abs(np.diff(a.mean(axis=0))).max()
    assert slope(bil) > 2.5 * slope(gau)
    assert slope(bil) > 0.3                                # clean step: 0.4 per pixel


# ---------------------------------------------------------------- best-of-grid PSNR
def test_bilateral_beats_gaussian_best_parameters(photo):
    x = add_noise(photo, a=0.01, b=0.01)
    best_g = max(psnr(photo, D.gaussian(x, s)) for s in (0.7, 1.0, 1.4, 2.0, 3.0))
    best_b = max(psnr(photo, D.bilateral(x, sd, sr))
                 for sd in (1.5, 2.5) for sr in (0.05, 0.1, 0.2, 0.4))
    assert best_b > best_g + 0.2
    assert best_b > psnr(photo, x) + 3


# ---------------------------------------------------------------- luma / chroma
def test_luma_chroma_removes_chroma_noise(photo):
    x = add_noise(photo, a=0.01, b=0.01)
    out = D.luma_chroma(x)
    yc, yn, yo = (rgb_to_ycbcr(a) for a in (photo, x, out))
    chroma = lambda a: np.sqrt(np.mean((a[..., 1:] - yc[..., 1:]) ** 2))
    assert chroma(yo) < 0.55 * chroma(yn)
    grad = lambda a: np.mean(np.hypot(*np.gradient(a[..., 0])))
    assert grad(yo) > 0.8 * grad(yc)                        # luma detail kept ...
    blur = rgb_to_ycbcr(D.gaussian(x, 3.0))
    assert grad(yo) > 1.4 * grad(blur)                      # ... unlike a strong blur


# ---------------------------------------------------------------- temporal
def test_average_frames_one_over_sqrt_m():
    clean = np.full((64, 64), 0.5, np.float32)
    r = rng(0)
    frames = [clean + 0.05 * r.standard_normal(clean.shape).astype(np.float32) for _ in range(16)]
    s1 = np.std(frames[0] - clean)
    for m in (4, 16):
        sm = np.std(D.average_frames(frames[:m]) - clean)
        assert sm == pytest.approx(s1 / np.sqrt(m), rel=0.1)


# ---------------------------------------------------------------- misc
def test_median_nlm_dispatch_estimate(photo):
    x = add_noise(photo, a=0.01, b=0.01)
    assert abs(D.estimate_noise(add_noise(np.full((64, 64, 3), 0.5, np.float32), a=0, b=0.03)) - 0.03) < 0.003
    for m, kw in [("gaussian", dict(sigma=1.0)), ("bilateral", dict(sigma_d=2, sigma_r=0.1)),
                  ("median", dict(size=3)), ("luma_chroma", {}), ("nlm", dict(h=0.03))]:
        out = D.denoise(x, m, **kw)
        assert out.shape == x.shape and out.dtype == np.float32
        if m in ("bilateral", "luma_chroma", "nlm"):
            assert psnr(photo, out) > psnr(photo, x) + 1
    with pytest.raises(ValueError):
        D.denoise(x, "nope")
