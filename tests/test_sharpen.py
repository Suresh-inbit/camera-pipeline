"""isp.sharpen: slide kernels, unsharp masking, edge/noise behaviour (L5 s12-17)."""
import numpy as np
import pytest
from scipy.ndimage import convolve

from isp import scenes, utils
from isp import sharpen as S


@pytest.fixture(scope="module")
def photo():
    return utils.linear_to_srgb(scenes.photo("astronaut", 96))


def interior(a, m=2):
    return a[m:-m, m:-m]


def test_slide_kernels():
    assert np.array_equal(S.sharpening_kernel(1), [[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    assert np.array_equal(S.sharpening_kernel(2), [[0, -2, 0], [-2, 9, -2], [0, -2, 0]])
    assert np.array_equal(S.LAPLACIAN, [[0, -1, 0], [-1, 4, -1], [0, -1, 0]])


@pytest.mark.parametrize("lam", [1, 2])
def test_laplacian_unsharp_equals_kernel_convolution(photo, lam):
    out = S.unsharp_mask(photo, lam, kind="laplacian", clip=False)
    k = S.sharpening_kernel(lam)
    ref = np.stack([convolve(photo[..., c], k, mode="reflect") for c in range(3)], axis=2)
    assert np.allclose(interior(out), interior(ref), atol=1e-5)


def test_lambda_zero_flat_and_mean(photo):
    assert np.array_equal(S.unsharp_mask(photo, 0.0), photo)
    flat = np.full((16, 16, 3), 0.4, np.float32)
    for kind in ("laplacian", "laplacian8", "gaussian"):
        assert np.allclose(S.unsharp_mask(flat, 2.0, kind), 0.4, atol=1e-6)
        assert S.high_pass_kernel(kind).sum() == pytest.approx(0.0, abs=1e-6)
        assert S.sharpening_kernel(2.0, kind).sum() == pytest.approx(1.0, abs=1e-6)
    out = S.unsharp_mask(photo, 2.0, clip=False)
    assert interior(out, 8).mean() == pytest.approx(interior(photo, 8).mean(), abs=2e-3)


def test_edge_overshoot_grows_with_lambda():
    img = scenes.step_edge(128)
    ov, rise = [], []
    for lam in (0.0, 0.5, 1.0, 2.0, 3.0):
        d, p = S.edge_profile(S.unsharp_mask(img, lam, clip=False))
        ov.append(S.overshoot(p))
        rise.append(S.rise_distance(d, p))
    assert ov[0] == 0.0
    assert all(b > a for a, b in zip(ov, ov[1:]))
    assert all(b <= a + 0.05 for a, b in zip(rise, rise[1:]))
    assert rise[-1] < rise[0]


def test_overshoot_function():
    p = np.r_[np.zeros(20), 1.2, np.ones(19)]
    assert S.overshoot(p) == pytest.approx(0.2)
    assert S.overshoot(np.r_[np.zeros(20), np.ones(20)]) == 0.0
    q = np.r_[np.zeros(19), -0.3, np.ones(20)]
    assert S.overshoot(q) == pytest.approx(0.3)


@pytest.mark.parametrize("lam", [1.0, 2.0])
def test_noise_gain_matches_measurement(lam):
    noise = (0.5 + 0.02 * utils.rng(0).standard_normal((256, 256))).astype(np.float32)
    out = S.unsharp_mask(noise, lam, clip=False)
    meas = interior(out, 3).std() / interior(noise, 3).std()
    gain = S.noise_gain(S.sharpening_kernel(lam))
    assert gain == pytest.approx(np.sqrt((1 + 4 * lam) ** 2 + 4 * lam ** 2))
    assert meas == pytest.approx(gain, rel=0.04)
    assert S.noise_gain(S.sharpening_kernel(1)) == pytest.approx(5.385, abs=1e-3)


def test_threshold_limits_noise_but_keeps_edges():
    edge = scenes.step_edge(128)
    noisy = edge + 0.01 * utils.rng(1).standard_normal(edge.shape).astype(np.float32)
    flat = (slice(8, 40), slice(8, 40))              # far from the edge
    plain = S.unsharp_mask(noisy, 1.0, clip=False)[flat].std()
    thr = S.unsharp_mask(noisy, 1.0, threshold=0.15, clip=False)[flat].std()
    assert thr < 0.5 * plain
    assert thr < 1.5 * noisy[flat].std()
    sharp = S.unsharp_mask(edge, 1.0, threshold=0.15, clip=False)
    assert S.overshoot(S.edge_profile(sharp)[1]) > 0.05


def test_frequency_response():
    for lam in (0.5, 1.0, 2.0):
        h = S.unsharp_response(lam, "laplacian", n=64)
        assert h[0, 0] == pytest.approx(1.0)
        assert h[32, 32] == pytest.approx(1 + 8 * lam)
        assert h[32, 0] == pytest.approx(1 + 4 * lam) and h[0, 32] == pytest.approx(1 + 4 * lam)
    hg = S.unsharp_response(1.0, "gaussian", sigma=1.5, n=64)
    assert hg[0, 0] == pytest.approx(1.0, abs=1e-6)
    assert 1.5 < hg[32, 32] <= 2.0 + 1e-6            # 1 + lam (1 - G(pi)) <= 2
    assert np.allclose(S.frequency_response(S.LAPLACIAN)[32, 32], 8.0)


def test_luminance_only_keeps_chroma(photo):
    out = S.unsharp_mask(photo, 2.0, luminance_only=True, clip=False)
    a, b = utils.rgb_to_ycbcr(photo), utils.rgb_to_ycbcr(out)
    assert np.allclose(a[..., 1:], b[..., 1:], atol=1e-5)
    assert np.abs(a[..., 0] - b[..., 0]).max() > 0.01
    ref = S.unsharp_mask(a[..., 0], 2.0, clip=False)
    assert np.allclose(b[..., 0], ref, atol=1e-5)


def test_dtype_shape_range(photo):
    for kw in ({}, {"luminance_only": True}, {"kind": "gaussian", "sigma": 2.0}):
        out = S.unsharp_mask(photo, 3.0, **kw)
        assert out.dtype == np.float32 and out.shape == photo.shape
        assert out.min() >= 0.0 and out.max() <= 1.0
    g = S.sharpen(photo[..., 0], "unsharp", 1.0, sigma=1.0)
    assert g.shape == photo.shape[:2] and g.dtype == np.float32
    assert np.array_equal(S.sharpen(photo, "laplacian", 1.0), S.unsharp_mask(photo, 1.0))
    with pytest.raises(ValueError):
        S.high_pass(photo, "nope")


def test_mean_gradient_increases(photo):
    assert S.mean_gradient(S.unsharp_mask(photo, 1.0)) > S.mean_gradient(photo)
