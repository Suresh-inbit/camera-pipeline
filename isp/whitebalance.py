"""White balance (lecture 4, first half): chromaticity, colour temperature, gains.

Model (von Kries / diagonal): the sensor sees ``scene * illuminant`` per channel, so
dividing each channel by the illuminant colour (a per-channel gain) restores the scene.
This is only valid on *linear* data; applying gains after gamma encoding changes the
hue of every non-grey pixel.  Everything here works on linear ``float32`` data.
"""
from __future__ import annotations

import numpy as np

from .utils import RGB2XYZ, XYZ2RGB

_EPS = 1e-12

# Correlated colour temperatures from L4 s7-8 (kelvin).
PRESETS = {
    "candle": 2000,        # slide range 1000-2000 K; below ~1950 K blue leaves the sRGB gamut
    "incandescent": 3200,
    "sunrise": 3500,
    "fluorescent": 4000,
    "equal_energy": 5000,        # slide 7 lists "equal energy white (D50, E)" at 5000 K
    "flash": 5500,
    "daylight": 6000,            # midday sun, photo flash (slide 7: 6000 K)
    "d65": 6504,                 # slide 7 says 6500 K "overcast, television"; D65 is 6504 K
    "overcast": 6500,
    "shade": 7500,               # northern blue sky
    "heavy_overcast": 9500,
}

_D65_T = 6504.0
_C2 = 1.4388e-2                  # second radiation constant (m K)


# ---------------------------------------------------------------------------
# CIE 1931 colour matching functions: Wyman, Sloan, Shirley (2013) multi-lobe fit
# ---------------------------------------------------------------------------
def _g(lam, mu, s1, s2):
    s = np.where(lam < mu, s1, s2)
    return np.exp(-0.5 * ((lam - mu) / s) ** 2)


def cie_cmf(wavelengths) -> np.ndarray:
    """CIE 1931 2-degree colour matching functions, shape (N, 3) for x-bar, y-bar, z-bar.

    Multi-lobe piecewise Gaussian fit of Wyman, Sloan and Shirley (2013); each lobe is
    ``a exp(-(lam - mu)^2 / (2 s^2))`` with ``s = s1`` below ``mu`` and ``s2`` above.
    Reference: Wyman et al., JCGT 2(2), 2013.
    """
    lam = np.atleast_1d(np.asarray(wavelengths, dtype=np.float64))
    x = (1.056 * _g(lam, 599.8, 37.9, 31.0) + 0.362 * _g(lam, 442.0, 16.0, 26.7)
         - 0.065 * _g(lam, 501.1, 20.4, 26.2))
    y = 0.821 * _g(lam, 568.8, 46.9, 40.5) + 0.286 * _g(lam, 530.9, 16.3, 31.1)
    z = 1.217 * _g(lam, 437.0, 11.8, 36.0) + 0.681 * _g(lam, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], axis=-1)


def spectral_locus_xy(lo: float = 420.0, hi: float = 650.0, step: float = 1.0) -> np.ndarray:
    """(N, 2) xy chromaticity of monochromatic light from ``lo`` to ``hi`` nm.

    x = X / (X+Y+Z), y = Y / (X+Y+Z) with XYZ taken from :func:`cie_cmf`.  The default range
    is 420-650 nm: the Gaussian fit is accurate there, but its tails below ~420 nm and above
    ~660 nm have a few-percent relative error that bends the xy curve (at 700 nm it gives
    x = 0.57 instead of 0.735), so the ends are cut where the true locus is almost straight.
    Reference: L4 s6 (xy chromaticity diagram), Wyman et al. 2013.
    """
    cmf = cie_cmf(np.arange(lo, hi + 0.5 * step, step))
    return (cmf[:, :2] / cmf.sum(axis=1, keepdims=True)).astype(np.float64)


# ---------------------------------------------------------------------------
# chromaticity
# ---------------------------------------------------------------------------
def rgb_chromaticity(img) -> np.ndarray:
    """rgb chromaticity ``(..., 2)``: r = R/(R+G+B), g = G/(R+G+B).  Black maps to (1/3, 1/3).

    Neutral colours of any brightness land at (1/3, 1/3).
    Reference: L4 s5-6.
    """
    a = np.asarray(img, dtype=np.float64)
    s = a.sum(axis=-1, keepdims=True)
    ok = s > _EPS
    c = np.where(ok, a / np.where(ok, s, 1.0), 1.0 / 3.0)
    return c[..., :2].astype(np.float32)


def rgb_to_xy(rgb) -> np.ndarray:
    """CIE xy chromaticity ``(..., 2)`` of linear sRGB colours (via ``utils.RGB2XYZ``)."""
    xyz = np.asarray(rgb, dtype=np.float64) @ RGB2XYZ.T
    s = xyz.sum(axis=-1, keepdims=True)
    return (xyz[..., :2] / np.maximum(s, _EPS)).astype(np.float64)


def srgb_gamut_xy() -> np.ndarray:
    """xy of the sRGB primaries R, G, B (3, 2): the triangle of reproducible colours."""
    return rgb_to_xy(np.eye(3))


# ---------------------------------------------------------------------------
# colour temperature
# ---------------------------------------------------------------------------
_LAM = np.arange(360.0, 831.0, 1.0)
_CMF = cie_cmf(_LAM)


def _planck_xyz(T) -> np.ndarray:
    T = np.atleast_1d(np.asarray(T, dtype=np.float64))
    lam_m = _LAM * 1e-9
    m = lam_m[None, :] ** -5 / np.expm1(_C2 / (lam_m[None, :] * T[:, None]))
    return m @ _CMF                                     # (N, 3), sum over wavelength


def planckian_locus_xy(Ts) -> np.ndarray:
    """(N, 2) xy of black bodies at temperatures ``Ts`` (K).

    Planck's law  M(lam, T) ~ lam^-5 / (exp(c2 / (lam T)) - 1)  integrated against the
    Wyman et al. colour matching functions, then normalised to xy.
    Reference: L4 s7-8; Wyman et al. 2013.
    """
    xyz = _planck_xyz(Ts)
    return xyz[:, :2] / xyz.sum(axis=1, keepdims=True)


def cct_to_xy(T) -> tuple:
    """Chromaticity (x, y) of a black body at ``T`` kelvin (valid about 1667-25000 K).

    Computed by integrating Planck's law against the CMF fit (not a polynomial); for
    arrays use :func:`planckian_locus_xy`.  Reference: L4 s7.
    """
    x, y = planckian_locus_xy([float(T)])[0]
    return float(x), float(y)


_T_GRID = np.geomspace(1000.0, 40000.0, 4000)
_UV_GRID = None


def _xy_to_uv(xy):
    x, y = xy[:, 0], xy[:, 1]
    den = -2 * x + 12 * y + 3
    return np.stack([4 * x / den, 6 * y / den], axis=1)


def xy_to_cct(x, y=None) -> float:
    """Correlated colour temperature (K) of chromaticity (x, y): the nearest Planckian point.

    Numerical inverse of :func:`cct_to_xy` (nearest point on a fine locus table, measured in
    the CIE 1960 uv plane where distances are closer to perceptual).  ``xy_to_cct((x, y))``
    also works.  Reference: L4 s7 ("nearest point on the curve").
    """
    global _UV_GRID
    if y is None:
        x, y = x
    if _UV_GRID is None:
        _UV_GRID = _xy_to_uv(planckian_locus_xy(_T_GRID))
    d = np.sum((_UV_GRID - _xy_to_uv(np.array([[x, y]], dtype=np.float64))) ** 2, axis=1)
    return float(_T_GRID[int(np.argmin(d))])


def mccamy_cct(x, y) -> float:
    """McCamy's (1992) cubic approximation of CCT from xy; good to a few K near 2000-12500 K."""
    n = (x - 0.3320) / (0.1858 - y)
    return float(449.0 * n ** 3 + 3525.0 * n ** 2 + 6823.3 * n + 5520.33)


def illuminant_rgb(T) -> np.ndarray:
    """Linear sRGB colour of a black body at ``T`` kelvin, normalised to G = 1, shape (3,).

    XYZ of the black body (overall scale is irrelevant) -> ``utils.XYZ2RGB``.
    Out-of-gamut values are not clipped; blue can approach 0 at low T.
    Reference: L4 s7-8.
    """
    xyz = _planck_xyz([float(T)])[0]
    rgb = XYZ2RGB @ xyz
    return (rgb / rgb[1]).astype(np.float32)


# ---------------------------------------------------------------------------
# relighting and ideal gains
# ---------------------------------------------------------------------------
def _illum_relative(T_or_rgb) -> np.ndarray:
    """Illuminant colour relative to D65 (float64, (3,), G = 1).  RGB triples are used as is."""
    a = np.asarray(T_or_rgb, dtype=np.float64)
    if a.ndim == 0:
        e = illuminant_rgb(float(a)).astype(np.float64) / illuminant_rgb(_D65_T).astype(np.float64)
    else:
        e = a.reshape(3)
    if np.any(e <= 0):
        raise ValueError("illuminant colour must be positive")
    return e / e[1]


def apply_illuminant(img, T_or_rgb) -> np.ndarray:
    """Relight a scene (assumed lit by D65-neutral light): ``out = img * e / max(e)``.

    ``e`` is the black-body colour at ``T`` divided by the black body at 6504 K (so 6504 K
    leaves the image unchanged), or the RGB triple given.  Dividing by ``max(e)`` keeps the
    result in [0, 1] (the strongest channel keeps its value, the others shrink); G therefore
    changes by the factor ``G_e / max(e)`` (1 for a greenish cast, ~0.6 for a very red one).
    Von Kries / diagonal model.  Reference: L4 s9.
    """
    e = _illum_relative(T_or_rgb)
    return (np.asarray(img, dtype=np.float32) * (e / e.max()).astype(np.float32)).astype(np.float32)


def gains_for_illuminant(T_or_rgb) -> np.ndarray:
    """Ideal white-balance gains for a known illuminant, ``1/e`` normalised to G = 1, (3,).

    ``apply_gains(apply_illuminant(x, T), gains_for_illuminant(T))`` equals ``x / max(e)``,
    i.e. the scene up to the global scale ``1 / max(e)`` (G of the result is G of ``x``
    times that scale).  Reference: L4 s9.
    """
    e = _illum_relative(T_or_rgb)
    return (1.0 / e).astype(np.float32)


def gains_to_illuminant(gains) -> np.ndarray:
    """Illuminant colour implied by gains: ``1/gains`` normalised to G = 1.  Use it to
    compare an estimate with the truth through :func:`angular_error`."""
    e = 1.0 / np.asarray(gains, dtype=np.float64)
    return (e / e[1]).astype(np.float32)


# ---------------------------------------------------------------------------
# estimating gains
# ---------------------------------------------------------------------------
def _norm_g(g, normalize_green=True):
    g = np.asarray(g, dtype=np.float64)
    if normalize_green:
        g = g / g[1]
    return g.astype(np.float32)


def from_patch(img, box, normalize_green=True) -> np.ndarray:
    """Gains from a patch the user knows is neutral (L4 s9):  S_R = (R+G+B) / (3 R), etc.

    ``box = (row0, row1, col0, col1)``; (R, G, B) is the mean over the box.  The slide
    formula keeps the patch mean brightness; with ``normalize_green=True`` (default) the
    gains are divided by S_G so that G is untouched (S_G = 1).
    Reference: L4 s9.
    """
    r0, r1, c0, c1 = box
    m = np.asarray(img, dtype=np.float64)[r0:r1, c0:c1].reshape(-1, 3).mean(axis=0)
    s = m.sum() / (3.0 * np.maximum(m, _EPS))
    return _norm_g(s, normalize_green)


def gray_world(img, normalize_green=True) -> np.ndarray:
    """Gray-world gains: force the mean colour of the image to grey.

    ``S_c = mean(R,G,B) / mean_c``, then G = 1.  Fails when the scene is not grey on
    average (a sunset, a lawn).  Reference: L4 s10.
    """
    m = np.asarray(img, dtype=np.float64).reshape(-1, 3).mean(axis=0)
    return _norm_g(m.mean() / np.maximum(m, _EPS), normalize_green)


def white_patch(img, percentile=99, normalize_green=True) -> np.ndarray:
    """White-patch (brightest-pixel) gains: the brightest pixels are assumed white.

    Takes the pixels whose R+G+B is above the given percentile, averages them per channel
    and uses ``S_c = 1 / mean_c``, then G = 1.  Fails if those pixels are saturated (the
    channel clipped at 1 hides the true ratio) or coloured (gold highlight, a bright
    flower).  Reference: L4 s11.
    """
    a = np.asarray(img, dtype=np.float64).reshape(-1, 3)
    s = a.sum(axis=1)
    sel = a[s >= np.percentile(s, percentile)]
    return _norm_g(1.0 / np.maximum(sel.mean(axis=0), _EPS), normalize_green)


# ---------------------------------------------------------------------------
# applying gains
# ---------------------------------------------------------------------------
def apply_gains(img, gains, clip=False) -> np.ndarray:
    """Multiply the three channels by ``gains``.  Not clipped unless ``clip=True`` (clip to [0, 1])."""
    out = np.asarray(img, dtype=np.float32) * np.asarray(gains, dtype=np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32) if clip else out.astype(np.float32)


def apply_gains_mosaic(mosaic, gains, pattern="RGGB", clip=False) -> np.ndarray:
    """White balance a RAW mosaic: each pixel is scaled by the gain of its CFA site.

    ``pattern`` reads row-major over the 2x2 tile (``'RGGB'`` = [[R, G], [G, B]]).  Doing
    it before demosaicing is equivalent to doing it on the RGB image afterwards, but lets
    the demosaicer see balanced channels.  Reference: L4 s11.
    """
    pattern = pattern.upper()
    if len(pattern) != 4 or any(ch not in "RGB" for ch in pattern):
        raise ValueError(f"bad CFA pattern {pattern!r}")
    g = np.asarray(gains, dtype=np.float32)
    tile = np.array([g["RGB".index(ch)] for ch in pattern], dtype=np.float32).reshape(2, 2)
    m = np.asarray(mosaic, dtype=np.float32)
    h, w = m.shape
    out = m * np.tile(tile, (h // 2 + 1, w // 2 + 1))[:h, :w]
    return np.clip(out, 0.0, 1.0).astype(np.float32) if clip else out.astype(np.float32)


# ---------------------------------------------------------------------------
# metric
# ---------------------------------------------------------------------------
def angular_error(a, b) -> float:
    """Angle in degrees between two RGB vectors: ``acos(a.b / (|a||b|))``.

    The standard colour-constancy error; independent of brightness.  Compare estimated and
    true *illuminants* (``1/gains``), not gains.  0 = perfect, ~3 deg is just noticeable.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    c = a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), _EPS)
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))
