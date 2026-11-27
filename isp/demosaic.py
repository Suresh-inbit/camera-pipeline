"""Demosaicking and colour-sensing technologies (lecture 3, Sensing Color).

Demosaicking turns the single-plane Bayer mosaic into three full colour planes.
Two thirds of the output numbers are interpolated ("2/3 of your data is made up", L3 s28).

Methods (``METHODS``):

* ``nearest``  - replicate the sample of the same 2x2 tile.
* ``bilinear`` - average of the nearest same-colour neighbours (L3 s17).
* ``bicubic``  - Keys cubic kernel, 16 neighbours (L3 s17).
* ``median``   - Freeman's median-filtered chroma (L3 s26-27).
* ``malvar``   - Malvar-He-Cutler 2004 gradient-corrected 5x5 filters (extension).

All linear methods are implemented as *normalised convolution*: for each colour,
``out = (K * (mosaic . mask)) / (K * mask)``.  The denominator renormalises the weights
of the samples that really exist, which handles the image borders and makes the measured
sample at its own site pass through unchanged.

Everything works on linear float32 data.  No plotting here.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import median_filter, shift as nd_shift

from . import cfa
from .utils import rgb_to_ycbcr, ycbcr_to_rgb, gaussian_kernel, rng

METHODS = ("nearest", "bilinear", "bicubic", "median", "malvar")


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------
def _keys(t):
    """Keys cubic-convolution kernel (a = -0.5), elementwise."""
    t = np.abs(np.asarray(t, dtype=np.float64))
    return np.where(t <= 1, 1.5 * t ** 3 - 2.5 * t ** 2 + 1,
                    np.where(t < 2, -0.5 * t ** 3 + 2.5 * t ** 2 - 4 * t + 2, 0.0))


_BILIN_RB = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32) / 4.0
_BILIN_G = np.array([[0, 1, 0], [1, 4, 1], [0, 1, 0]], dtype=np.float32) / 4.0

# Bicubic: R and B live on a lattice of spacing 2, so the kernel is the separable Keys
# kernel stretched by 2 (taps -3..3).  G lives on a 45-degree lattice: use Keys along the
# diagonal coordinates u = (dx+dy)/2, v = (dx-dy)/2 (taps -3..3 as well).
_h = _keys(np.arange(-3, 4) / 2.0)
_BICUBIC_RB = np.outer(_h, _h).astype(np.float32)
_dy, _dx = np.mgrid[-3:4, -3:4]
_BICUBIC_G = (_keys((_dx + _dy) / 2.0) * _keys((_dx - _dy) / 2.0)).astype(np.float32)


def _filt(a, k, border):
    return cv2.filter2D(np.ascontiguousarray(a, dtype=np.float32), -1, np.ascontiguousarray(k[::-1, ::-1]),
                        borderType=border)


def _normalized_conv(m, pattern, k_rb, k_g):
    """Per-colour normalised convolution of the mosaic with the given kernels."""
    mk = cfa.masks(m.shape, pattern)
    out = np.empty(m.shape + (3,), np.float32)
    for c in range(3):
        k = k_g if c == 1 else k_rb
        w = mk[..., c].astype(np.float32)
        num = _filt(m * w, k, cv2.BORDER_CONSTANT)
        den = _filt(w, k, cv2.BORDER_CONSTANT)
        out[..., c] = num / den
    return out


# --------------------------------------------------------------------------
# the methods
# --------------------------------------------------------------------------
def _nearest(m, pattern):
    """Each pixel takes the sample of its own 2x2 tile (first G of the tile for green)."""
    idx = cfa.channel_index((2, 2), pattern)
    h, w = m.shape
    out = np.empty((h, w, 3), np.float32)
    for c in range(3):
        a, b = np.argwhere(idx == c)[0]
        r = np.minimum(2 * (np.arange(h) // 2) + a, h - 1)
        q = np.minimum(2 * (np.arange(w) // 2) + b, w - 1)
        out[..., c] = m[r][:, q]
    return out


def bilinear(m, pattern="RGGB"):
    """Bilinear demosaic: average of the nearest same-colour neighbours (L3 s17)."""
    return _normalized_conv(m, pattern, _BILIN_RB, _BILIN_G)


def bicubic(m, pattern="RGGB"):
    """Bicubic demosaic: Keys cubic kernel over 4x4 same-colour neighbours (L3 s17)."""
    return _normalized_conv(m, pattern, _BICUBIC_RB, _BICUBIC_G)


def median(m, pattern="RGGB", median_size=3, iterations=1, base="bilinear"):
    """Freeman's median-filtered chroma demosaic (L3 s26-27).

    1. naive interpolation (``base``, default bilinear);
    2. convert to YCbCr;
    3. median-filter Cb and Cr with a ``median_size`` window (``iterations`` times):
       colour fringes are chroma outliers that last only a pixel or two;
    4. rebuild R, G, B from the sensor value and the filtered chroma: at a site that
       measured channel c with value s, ``Y = s - o_c`` with ``o = rgb(Y=0, Cb', Cr')``
       and ``rgb = Y + o``.  The measured sample is kept exactly.
    """
    est = demosaic(m, pattern, base)
    ycc = rgb_to_ycbcr(est)
    cbcr = ycc[..., 1:].copy()
    for _ in range(int(iterations)):
        for c in range(2):
            cbcr[..., c] = median_filter(cbcr[..., c], size=median_size, mode="reflect")
    o = ycbcr_to_rgb(np.concatenate([np.zeros_like(ycc[..., :1]), cbcr], axis=2))  # offsets for Y = 0
    idx = cfa.channel_index(m.shape, pattern)
    o_c = np.take_along_axis(o, idx[..., None], axis=2)[..., 0]
    y = m - o_c
    return (y[..., None] + o).astype(np.float32)


# Malvar-He-Cutler 2004 filters (x 1/8), centre tap = the measured site.
_MHC_G_AT_RB = np.array([[0, 0, -1, 0, 0], [0, 0, 2, 0, 0], [-1, 2, 4, 2, -1],
                         [0, 0, 2, 0, 0], [0, 0, -1, 0, 0]], np.float32) / 8
_MHC_RB_AT_G_H = np.array([[0, 0, .5, 0, 0], [0, -1, 0, -1, 0], [-1, 4, 5, 4, -1],
                           [0, -1, 0, -1, 0], [0, 0, .5, 0, 0]], np.float32) / 8   # same-colour neighbours left/right
_MHC_RB_AT_G_V = _MHC_RB_AT_G_H.T.copy()                                            # ... above/below
_MHC_RB_AT_BR = np.array([[0, 0, -1.5, 0, 0], [0, 2, 0, 2, 0], [-1.5, 0, 6, 0, -1.5],
                          [0, 2, 0, 2, 0], [0, 0, -1.5, 0, 0]], np.float32) / 8


def malvar(m, pattern="RGGB", clip=True):
    """Malvar-He-Cutler (2004) high-quality linear interpolation (extension, not in the lecture).

    Bilinear estimate plus a gradient correction from the other colour's Laplacian:
    e.g. ``G(R site) = G_bilinear + alpha * (R - R_avg)`` with alpha = 1/2, giving fixed
    5x5 filters.  Borders use reflection (keeps the Bayer parity).  Output clipped to [0, 1].
    """
    m = np.ascontiguousarray(m, dtype=np.float32)
    f = lambda k: _filt(m, k, cv2.BORDER_REFLECT_101)
    g_rb, rb_h, rb_v, rb_d = f(_MHC_G_AT_RB), f(_MHC_RB_AT_G_H), f(_MHC_RB_AT_G_V), f(_MHC_RB_AT_BR)
    idx = cfa.channel_index(m.shape, pattern)
    ri, rj = np.argwhere(cfa.channel_index((2, 2), pattern) == 0)[0]
    yy = np.arange(m.shape[0])[:, None]
    r_row = (yy % 2 == ri)                      # rows that contain R samples (G sites there have R left/right)
    out = np.empty(m.shape + (3,), np.float32)
    g_site = idx == 1
    out[..., 1] = np.where(g_site, m, g_rb)
    for c in (0, 2):
        own = idx == c
        other = idx == (2 - c)
        # G site: the colour-c neighbours are horizontal if this row holds colour c
        row_has_c = r_row if c == 0 else ~r_row
        at_g = np.where(row_has_c, rb_h, rb_v)
        out[..., c] = np.where(own, m, np.where(g_site, at_g, rb_d))
    return np.clip(out, 0, 1) if clip else out


def demosaic(mosaic, pattern: str = "RGGB", method: str = "bilinear", **kw) -> np.ndarray:
    """Demosaic an ``(H, W)`` Bayer mosaic into an ``(H, W, 3)`` float32 image.

    ``method`` is one of ``METHODS``; extra keyword arguments go to the method
    (e.g. ``median_size=5`` for ``'median'``).  ``'median'`` is Freeman's algorithm
    (bilinear, then median filter the chroma, L3 s26-27).

    Reference: L3 s17 (linear, bicubic), s26-27 (median); Malvar et al. 2004.
    """
    m = np.asarray(mosaic, dtype=np.float32)
    cfa._check(pattern)
    if m.ndim != 2:
        raise ValueError("demosaic expects an (H, W) mosaic")
    if method == "nearest":
        out = _nearest(m, pattern)
    elif method == "bilinear":
        out = bilinear(m, pattern)
    elif method == "bicubic":
        out = bicubic(m, pattern)
    elif method == "median":
        out = median(m, pattern, **kw)
    elif method == "malvar":
        out = malvar(m, pattern, **kw)
    else:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# metrics and luma / chroma tools
# --------------------------------------------------------------------------
def false_color(img, reference) -> float:
    """Chroma error: RMS of the Cb and Cr differences to the reference (JPEG YCbCr, range 0..1).

    ``sqrt(mean((Cb - Cb_ref)^2 + (Cr - Cr_ref)^2))``.  Luminance errors (blur) do not
    count, colour fringes and moire do (L3 s18).  Lower is better; 0 for identical images.
    """
    a = rgb_to_ycbcr(np.clip(img, 0, 1))[..., 1:].astype(np.float64)
    b = rgb_to_ycbcr(np.clip(reference, 0, 1))[..., 1:].astype(np.float64)
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=-1))))


def blur_channel(img, channel: str = "Y", sigma: float = 3.0) -> np.ndarray:
    """Gaussian-blur only one of Y, Cb, Cr of an image (L3 s20-25 demonstration).

    ``channel`` in ``'Y'``, ``'Cb'``, ``'Cr'`` (or ``'CbCr'`` for both chroma planes).
    Apply it to a display-referred (sRGB) image to mimic the slides.  Output clipped to [0, 1].
    """
    ycc = rgb_to_ycbcr(np.clip(img, 0, 1))
    k = gaussian_kernel(sigma)
    sel = {"Y": [0], "Cb": [1], "Cr": [2], "CbCr": [1, 2]}[channel]
    for c in sel:
        ycc[..., c] = cv2.filter2D(np.ascontiguousarray(ycc[..., c]), -1, k, borderType=cv2.BORDER_REFLECT_101)
    return np.clip(ycbcr_to_rgb(ycc), 0, 1).astype(np.float32)


# --------------------------------------------------------------------------
# colour sensing technologies (L3 s4-12, summary s29)
# --------------------------------------------------------------------------
TECHNOLOGIES = ("field_sequential", "three_sensor", "stacked", "cfa")

# Vertically stacked sensor (L3 s11): layer outputs = STACK_MIX @ (R, G, B).  Rows are the
# top (blue-ish), middle and bottom (red-ish) layers, in the order out = (L_R, L_G, L_B)
# used below: broad overlapping responses, so the mixing matrix has to be inverted.
STACK_MIX = np.array([[0.60, 0.30, 0.10],
                      [0.30, 0.45, 0.25],
                      [0.15, 0.35, 0.50]], dtype=np.float64)


def _noise(shape, sigma, seed):
    return rng(seed).normal(0.0, sigma, shape).astype(np.float32) if sigma > 0 else 0.0


def field_sequential(scene, motion=(0.0, 0.0), frames=None, noise=0.0, seed=0) -> np.ndarray:
    """Field-sequential capture: R, G, B are three exposures at times 0, 1, 2 (L3 s6-7).

    The scene moves by ``motion = (dy, dx)`` pixels per exposure (or pass your own
    ``frames=(f_R, f_G, f_B)``).  ``out[..., c] = frame_c[..., c]``.  Static: exact.
    Moving: the planes are misregistered -> colour break-up at edges.
    """
    s = np.asarray(scene, dtype=np.float32)
    if frames is None:
        frames = [s if k == 0 or not any(motion) else
                  nd_shift(s, (k * motion[0], k * motion[1], 0), order=1, mode="nearest") for k in range(3)]
    out = np.stack([np.asarray(frames[c], dtype=np.float32)[..., c] for c in range(3)], axis=-1)
    return (out + _noise(out.shape, noise, seed)).astype(np.float32)


def three_sensor(scene, noise=0.0, seed=0) -> np.ndarray:
    """Ideal 3-sensor camera (prism + 3 sensors, L3 s9): every pixel gets R, G and B.  Exact when noise = 0."""
    s = np.asarray(scene, dtype=np.float32)
    return (s + _noise(s.shape, noise, seed)).astype(np.float32)


def stacked_sensor(scene, mix=STACK_MIX, noise=0.0, seed=0) -> np.ndarray:
    """Vertically stacked (Foveon-like) sensor (L3 s11).

    Layers record ``y = mix @ rgb + n`` (broad, overlapping responses, noise ``n`` with
    std ``noise`` per layer); the camera recovers ``rgb = mix^-1 y``.  Noise is amplified
    by the rows of ``inv(mix)``: ``std_out = noise * ||row of inv(mix)||``.
    """
    s = np.asarray(scene, dtype=np.float64)
    y = s @ np.asarray(mix, dtype=np.float64).T
    y = y + _noise(y.shape, noise, seed)
    return (y @ np.linalg.inv(mix).T).astype(np.float32)


def cfa_sensor(scene, pattern="RGGB", method="bilinear", noise=0.0, seed=0, **kw) -> np.ndarray:
    """CFA camera: mosaic (each site sees 1/3 of the light, 2/3 of the samples discarded),
    noise, then demosaic (L3 s13-28).  Interpolation artefacts (moire, fringes) appear at fine detail."""
    m = cfa.mosaic(scene, pattern)
    m = m + _noise(m.shape, noise, seed)
    return demosaic(m, pattern, method, **kw)


def sense(scene, technology="cfa", **kw) -> np.ndarray:
    """Dispatch to one of ``TECHNOLOGIES`` (keyword arguments are passed on)."""
    fn = {"field_sequential": field_sequential, "three_sensor": three_sensor,
          "stacked": stacked_sensor, "cfa": cfa_sensor}[technology]
    return fn(scene, **kw)
