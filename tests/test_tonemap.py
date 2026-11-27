import cv2
import numpy as np
import pytest

from isp import scenes, tonemap as T, utils


def lowc(size=96):
    """Continuous-valued, mildly coloured, low-contrast display image (values ~0.35..0.65)."""
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:size, 0:size] / size
    base = 0.5 + 0.12 * np.sin(5 * x + 2 * y) * np.cos(3 * y) + 0.03 * rng.standard_normal((size, size))
    tint = np.array([1.06, 1.0, 0.94])
    return np.clip(0.35 + 0.3 * (base[..., None] - 0.3) * tint, 0, 1).astype(np.float32)


def test_gamma_properties():
    x = np.random.default_rng(0).random((16, 16, 3)).astype(np.float32)
    assert np.allclose(T.gamma(x, 1.0), x, atol=1e-7)
    assert np.allclose(T.gamma(T.gamma(x, 0.5), 2.0), x, atol=1e-5)
    mid = np.full((2, 2, 3), 0.5, np.float32)
    assert T.gamma(mid, 0.5).mean() > 0.5 > T.gamma(mid, 2.0).mean()
    e = np.array([0.0, 1.0], np.float32)
    assert np.allclose(T.gamma(e, 0.3), e) and np.allclose(T.gamma(e, 3.0), e)


def test_hist_equalize_uniform_cdf():
    out = T.hist_equalize(lowc(), mode="luminance")
    y = np.sort(utils.luminance(out).ravel())
    cdf = np.arange(1, y.size + 1) / y.size
    assert np.max(np.abs(cdf - y)) < 0.05  # CDF of output ~ diagonal
    g = lowc().mean(axis=-1)
    yg = np.sort(T.hist_equalize(g).ravel())
    assert np.max(np.abs(np.arange(1, yg.size + 1) / yg.size - yg)) < 0.02


def test_hist_equalize_monotone_grey_hue():
    x = lowc()
    out = T.hist_equalize(x)
    # order is exact for grey; for colour only the brightest pixels are rescaled by the gamut guard
    g2 = x.mean(-1)
    assert np.all(np.diff(np.sort(T.hist_equalize(g2).ravel())) >= 0)
    yi, yo = utils.luminance(x).ravel(), utils.luminance(out).ravel()
    o = np.argsort(yi, kind="stable")
    assert np.all(np.diff(yo[o][: int(0.9 * o.size)]) >= -1e-4)  # order preserved (up to float noise)
    grey = np.repeat(x[..., :1], 3, axis=2)
    for mode in ("luminance", "channel"):
        g = T.hist_equalize(grey, mode=mode)
        assert np.allclose(g[..., 0], g[..., 1], atol=1e-4) and np.allclose(g[..., 1], g[..., 2], atol=1e-4)

    def chrom(a):
        return a[..., :2] / np.maximum(a.sum(-1, keepdims=True), 1e-6)

    ok = utils.luminance(x) > 0.1
    d_lum = np.abs(chrom(out) - chrom(x))[ok].max()
    d_ch = np.abs(chrom(T.hist_equalize(x, mode="channel")) - chrom(x))[ok].max()
    assert d_lum < 1e-3 < d_ch


def test_levels():
    x = np.array([0.2, 0.7, 0.45], np.float32)
    o = T.levels(x, 0.2, 0.7)
    assert o[0] == 0 and o[1] == 1 and 0 < o[2] < 1
    with pytest.raises(ValueError):
        T.levels(x, 0.5, 0.5)
    a = utils.luminance(T.auto_levels(lowc()))
    assert a.max() > 0.99 and a.min() < 0.01


def _halves(y):
    w = y.shape[1]
    return y[:, : w // 4].mean(), y[:, 3 * w // 4:].mean()


def _detail(y, s=1.5):
    L = np.log2(y + 1e-4).astype(np.float32)
    return float((L - cv2.GaussianBlur(L, (0, 0), s)).std())


def test_dodge_burn_high_contrast():
    img = scenes.high_contrast(128)
    out = T.dodge_burn(img, strength=0.6)
    assert out.dtype == np.float32 and out.min() >= 0 and out.max() <= 1
    yi, yo = utils.luminance(img), utils.luminance(out)
    li, ri = _halves(yi)
    lo, ro = _halves(yo)
    assert (ro / lo) < 0.5 * (ri / li)
    assert 0.6 < _detail(yo) / _detail(yi) < 1.6
    assert np.array_equal(T.dodge_burn(img, strength=0.0), np.clip(img, 0, 1))


def _overshoot(y):
    row = y[y.shape[0] // 2]
    n = row.size
    lo_far, hi_far = row[: n // 8].mean(), row[-n // 8:].mean()
    return float(max(row.max() - hi_far, lo_far - row.min(), 0) / (hi_far - lo_far))


def test_bilateral_less_halo():
    img = scenes.step_edge(128, lo=0.05, hi=0.8, angle_deg=0.0)
    kw = dict(sigma=8, strength=0.7)
    og = _overshoot(utils.luminance(T.dodge_burn(img, base="gaussian", **kw)))
    ob = _overshoot(utils.luminance(T.dodge_burn(img, base="bilateral", **kw)))
    assert ob < 0.5 * og


def test_dodge_burn_mask():
    img = np.full((8, 8, 3), 0.2, np.float32)
    m = np.zeros((8, 8), np.float32)
    m[:, :4] = 1
    out = T.dodge_burn_mask(img, m, stops=1.0)
    assert np.allclose(out[:, :4], 0.4) and np.allclose(out[:, 4:], 0.2)
    assert np.allclose(T.dodge_burn_mask(img, m, stops=-1.0)[:, :4], 0.1)


@pytest.mark.parametrize("method", T.METHODS)
def test_tonemap_dispatcher(method):
    img = scenes.high_contrast(64)
    out = T.tonemap(img, method=method)
    assert out.dtype == np.float32 and out.shape == img.shape
    assert out.min() >= 0 and out.max() <= 1


def test_tonemap_srgb_and_unknown():
    img = scenes.photo("coffee", 64)
    assert np.array_equal(T.tonemap(img, "srgb"), utils.linear_to_srgb(img))
    with pytest.raises(ValueError):
        T.tonemap(img, "nope")


def test_histogram():
    img = scenes.gray_ramp(64)
    assert T.histogram(img, 16).sum() == 64 * 64
    assert T.histogram(img, 16, per_channel=True).shape == (3, 16)


def test_dodge_burn_keeps_black_object_dark():
    # regression: a deep-black object in the bright half used to be lifted ~12 stops and
    # blow out to white with speckles; now it is lifted by a bounded amount.
    img = scenes.high_contrast(128).copy()
    yy, xx = np.mgrid[:128, :128]
    disc = (yy - 90) ** 2 + (xx - 100) ** 2 < 18 ** 2
    img[disc] = 0.002
    img[(yy - 90) ** 2 + (xx - 100) ** 2 < 4 ** 2] = 0.0          # true black pixels inside
    left = utils.luminance(img)[:, :32].mean()
    for base in ("gaussian", "bilateral"):
        out = T.dodge_burn(img, sigma=8, strength=0.7, base=base)
        yo = utils.luminance(out)
        assert yo[disc].mean() < 0.03, (base, yo[disc].mean())
        assert yo[disc].max() < 0.04, (base, yo[disc].max())
        assert out[(yy - 90) ** 2 + (xx - 100) ** 2 < 4 ** 2].max() == 0.0   # zero stays zero
        assert yo[:, :32].mean() > 5 * left                          # shadows are still lifted
