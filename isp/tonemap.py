"""Tone mapping / contrast correction (lecture 4, second half).

Convention: the building blocks (``gamma``, ``levels``, ``hist_equalize``, ``clahe``,
``dodge_burn``, ``dodge_burn_mask``) are plain image -> image functions that keep the
domain of their input (linear in -> linear out for ``dodge_burn*``; display in ->
display out for the histogram methods, see their docstrings).  :func:`tonemap` is the
pipeline entry point: LINEAR [0, 1] in, DISPLAY-referred (sRGB-encoded) [0, 1] out.

Local operator implemented: the base/detail scheme of Durand and Dorsey (2002), whose
"compress the smooth layer, keep the detail layer" is the automatic version of dodging
and burning; the global anchor idea follows Reinhard et al. (2002).
"""
from __future__ import annotations

import cv2
import numpy as np

from .utils import linear_to_srgb, luminance

_EPS = 1e-6
_LOG_EPS = 1e-4  # floor for log luminance: 13 stops below white, keeps pure black from dominating
_MAX_LIFT = 6.0  # stops: soft limit of the base-layer lift in dodge_burn
METHODS = ("srgb", "linear", "gamma", "hist", "clahe", "local")


def _f32(img) -> np.ndarray:
    return np.asarray(img, dtype=np.float32)


# --------------------------------------------------------------------------
# global point operations
# --------------------------------------------------------------------------
def gamma(img, g: float) -> np.ndarray:
    """Gamma transform  out = in ** g  for 0 <= in <= 1.

    g < 1 lifts the midtones (brightens), g > 1 darkens them; 0 and 1 are fixed points.
    Input is clipped to [0, 1].

    Reference: L4 s14.
    """
    return (np.clip(_f32(img), 0.0, 1.0) ** np.float32(g)).astype(np.float32)


def levels(img, black: float = 0.0, white: float = 1.0, g: float = 1.0) -> np.ndarray:
    """Black point / white point / midtone edit (the 'levels' dialog).

    out = clip((in - black) / (white - black), 0, 1) ** g

    ``black`` maps to 0, ``white`` to 1; ``g`` is a gamma on the stretched range.

    Reference: L4 s13 ("fiddle with the histogram").
    """
    if white <= black:
        raise ValueError("white must be larger than black")
    x = np.clip((_f32(img) - black) / (white - black), 0.0, 1.0)
    return (x ** np.float32(g)).astype(np.float32)


def auto_levels(img, clip_percent: float = 0.5, g: float = 1.0) -> np.ndarray:
    """Contrast stretch: black/white points at the ``clip_percent`` / ``100 - clip_percent``
    percentiles of the luminance, then :func:`levels`."""
    y = luminance(_f32(img))
    black, white = np.percentile(y, [clip_percent, 100.0 - clip_percent])
    if white - black < _EPS:
        return _f32(img).copy()
    return levels(img, float(black), float(white), g)


def histogram(img, bins: int = 256, per_channel: bool = False) -> np.ndarray:
    """Histogram counts over [0, 1].

    RGB input gives the luminance histogram, or a ``(3, bins)`` array when
    ``per_channel`` is True.  Use ``np.linspace(0, 1, bins + 1)`` for the bin edges.
    """
    a = _f32(img)
    if a.ndim == 3 and per_channel:
        return np.stack([np.histogram(a[..., c], bins=bins, range=(0, 1))[0] for c in range(3)])
    return np.histogram(np.clip(luminance(a), 0, 1), bins=bins, range=(0, 1))[0]


def _apply_luminance_map(img: np.ndarray, y_new: np.ndarray) -> np.ndarray:
    """Give ``img`` the luminance ``y_new`` keeping its colour ratios (hue).

    If a channel would exceed 1 the pixel is scaled down as a whole (hue preserved,
    luminance slightly reduced) instead of clipping channels independently.
    """
    if img.ndim == 2:
        return np.clip(y_new, 0, 1).astype(np.float32)
    y = luminance(img)
    ratio = (y_new / np.maximum(y, _EPS))[..., None]
    out = np.where((y > _EPS)[..., None], img * ratio, y_new[..., None])
    out = out / np.maximum(out.max(axis=-1, keepdims=True), 1.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def hist_equalize(img, bins: int = 256, mode: str = "luminance") -> np.ndarray:
    """Global histogram equalisation: histogram -> CDF -> lookup.

    out = CDF(in), with the CDF linearly interpolated between bin edges.  The result
    has (approximately) a flat histogram.  mode='luminance' equalises the luminance
    and rescales RGB by Y_out / Y_in so the colour ratios (hue) stay;
    mode='channel' equalises R, G, B separately, which shifts hue and saturation.

    Operates on the values it is given; for a display result pass sRGB-encoded data
    (:func:`tonemap` does that).  Output is monotone in luminance.

    Reference: L4 s13 (histograms, "fiddle with the histogram"); Gonzalez and Woods, Digital Image Processing.
    """
    a = np.clip(_f32(img), 0.0, 1.0)
    edges = np.linspace(0.0, 1.0, bins + 1)

    def eq(x):
        counts, _ = np.histogram(x, bins=bins, range=(0, 1))
        cdf = np.concatenate([[0.0], np.cumsum(counts)]) / max(x.size, 1)
        return np.interp(x, edges, cdf).astype(np.float32)

    if a.ndim == 2:
        return eq(a)
    if mode == "channel":
        return np.stack([eq(a[..., c]) for c in range(3)], axis=-1)
    if mode != "luminance":
        raise ValueError("mode must be 'luminance' or 'channel'")
    y = np.clip(luminance(a), 0.0, 1.0)
    return _apply_luminance_map(a, eq(y))


def clahe(img, clip: float = 2.0, tiles: int = 8) -> np.ndarray:
    """Contrast-limited adaptive histogram equalisation on the luminance (extension).

    Equalises each of ``tiles x tiles`` regions with the histogram clipped at ``clip``
    times the mean bin height and blends neighbouring tiles bilinearly (cv2).  Colour
    ratios are kept.  Like :func:`hist_equalize` it works on the values given.

    Reference: Zuiderveld, "Contrast Limited Adaptive Histogram Equalization", Graphics Gems IV, 1994.
    """
    a = np.clip(_f32(img), 0.0, 1.0)
    y = luminance(a)
    op = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tiles), int(tiles)))
    y16 = np.round(y * 65535.0).astype(np.uint16)
    y_new = op.apply(y16).astype(np.float32) / 65535.0
    return _apply_luminance_map(a, y_new)


# --------------------------------------------------------------------------
# local operators: automatic dodging and burning
# --------------------------------------------------------------------------
def _bilateral_log(L: np.ndarray, sigma_s: float, sigma_r: float) -> np.ndarray:
    """Bilateral filter of a log image; computed on a downsampled copy for speed."""
    f = max(1, int(sigma_s // 6))
    h, w = L.shape
    small = cv2.resize(L, (w // f, h // f), interpolation=cv2.INTER_AREA) if f > 1 else L
    s = sigma_s / f
    d = int(2 * np.ceil(2.0 * s) + 1)
    out = cv2.bilateralFilter(np.ascontiguousarray(small), d, float(sigma_r), float(s),
                              borderType=cv2.BORDER_REFLECT)
    if f > 1:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def decompose(img, sigma=None, base: str = "gaussian", sigma_r: float = 0.5):
    """Split log2 luminance into a smooth base layer and a detail layer.

    L = log2(Y + eps); base = smooth(L); detail = L - base.
    ``base='gaussian'`` blurs L with a Gaussian of std ``sigma`` pixels (default 5 % of
    the shorter side); ``base='bilateral'`` uses an edge-preserving bilateral filter
    (spatial std ``sigma``, range std ``sigma_r`` in log2 units = stops), which does not
    average across strong edges and so avoids halos.

    Returns (base, detail), both float32 ``(H, W)`` in stops.

    Reference: Durand and Dorsey, SIGGRAPH 2002.
    """
    a = np.clip(_f32(img), 0.0, None)
    L = np.log2(luminance(a) + _LOG_EPS).astype(np.float32)
    if sigma is None:
        sigma = 0.05 * min(L.shape)
    if base == "gaussian":
        b = cv2.GaussianBlur(L, (0, 0), float(sigma), borderType=cv2.BORDER_REFLECT)
    elif base == "bilateral":
        b = _bilateral_log(L, float(sigma), sigma_r)
    else:
        raise ValueError("base must be 'gaussian' or 'bilateral'")
    return b.astype(np.float32), (L - b).astype(np.float32)


def dodge_burn(img, sigma=None, strength: float = 0.6, base: str = "gaussian",
               sigma_r: float = 0.5, saturation: float = 1.0, return_layers: bool = False):
    """Automatic dodging and burning (local tone mapping), linear in -> linear out.

    With L = log2 Y split into base B and detail D (:func:`decompose`), the anchor is the
    brightest base value B_max (Durand-Dorsey: the max base maps to 0 stops, the rest is
    compressed towards it):

        L_out = D + (1 - strength) * (B - B_max) + B_max  =  L + strength * (B_max - B)

    The base layer's contrast is compressed towards its brightest value (dark regions
    are lifted = dodged relative to bright ones = burned) while the detail layer is
    kept untouched.  Two guards keep deep blacks black:

    * the lift ``x = B_max - B`` (stops) is soft-limited, ``R tanh(x / R)`` with R = 6
      stops (about ``x`` for small ``x``), so the darkest regions are not lifted without
      bound (a black object is lifted by at most ~``strength * 6`` stops, not by the full
      13-stop range of the scene);
    * the gain is applied multiplicatively, ``Y_out = Y * 2 ** lift``, so a true zero
      stays zero (no floor offset is added back).

    Colour is restored as  rgb_out = (rgb / Y)^saturation * Y_out.
    ``strength=0`` returns the input; ``strength=1`` flattens the base layer up to the
    soft limit.  The result is scaled down per pixel where a channel would exceed 1 (hue
    kept).
    Apply :func:`isp.utils.linear_to_srgb` afterwards for display.

    This is the Durand-Dorsey base/detail operator; Reinhard et al. (2002) is the classic
    automatic dodging-and-burning operator (global key + local adaptation).

    Reference: L4 s15-18; Durand and Dorsey 2002; Reinhard et al. 2002.
    """
    a = np.clip(_f32(img), 0.0, None)
    if strength == 0:
        out = np.clip(a, 0.0, 1.0)
        return (out, decompose(a, sigma, base, sigma_r)) if return_layers else out
    B, D = decompose(a, sigma, base, sigma_r)
    y = luminance(a)
    x = B.max() - B                                    # how far below the brightest base, stops
    lift = strength * _MAX_LIFT * np.tanh(x / _MAX_LIFT)
    y_out = (y * np.exp2(lift)).astype(np.float32)     # multiplicative: true black stays black
    if a.ndim == 2:
        out = np.clip(y_out, 0, 1)
    else:
        ratio = np.where(y[..., None] > _EPS, a / np.maximum(y, _EPS)[..., None], 1.0)
        out = (ratio ** saturation) * y_out[..., None]
        out = out / np.maximum(out.max(axis=-1, keepdims=True), 1.0)
        out = np.clip(out, 0.0, 1.0)
    out = out.astype(np.float32)
    return (out, (B, D)) if return_layers else out


def dodge_burn_mask(img, mask, stops: float = 1.0, feather: float = 0.0) -> np.ndarray:
    """Manual dodge / burn: change exposure by ``stops`` where ``mask`` is 1 (linear data).

    out = clip(in * 2 ** (stops * mask), 0, 1);  stops > 0 dodges (lightens), < 0 burns
    (darkens), like the darkroom card of L4 s16.  ``mask`` is (H, W) in [0, 1];
    ``feather`` blurs it with a Gaussian (std in pixels) for soft transitions.

    Reference: L4 s16.
    """
    a = _f32(img)
    m = np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)
    if feather > 0:
        m = cv2.GaussianBlur(m, (0, 0), float(feather), borderType=cv2.BORDER_REFLECT)
    if a.ndim == 3:
        m = m[..., None]
    return np.clip(a * np.exp2(np.float32(stops) * m), 0.0, 1.0).astype(np.float32)


# --------------------------------------------------------------------------
# dispatcher
# --------------------------------------------------------------------------
def tonemap(img, method: str = "srgb", **kw) -> np.ndarray:
    """Tone-map LINEAR [0, 1] input to a DISPLAY-referred (sRGB-encoded) float32 image.

    Where the display encoding happens:
      'srgb'   linear_to_srgb(img): the camera default curve (kw: none)
      'linear' clip only, NO encoding: what a linear image looks like on a screen (dark)
      'gamma'  img ** g with g = 1/2.2 by default (kw ``g``) *instead of* the sRGB curve
      'hist'   linear_to_srgb first, then hist_equalize on the encoded values
               (kw: bins, mode); equalised so the *displayed* histogram is flat
      'clahe'  linear_to_srgb first, then clahe (kw: clip, tiles)
      'local'  dodge_burn on the linear data (kw: sigma, strength, base, ...), then
               linear_to_srgb
    """
    a = np.clip(_f32(img), 0.0, 1.0)
    if method == "srgb":
        out = linear_to_srgb(a)
    elif method == "linear":
        out = a
    elif method == "gamma":
        out = gamma(a, kw.pop("g", 1.0 / 2.2))
    elif method == "hist":
        out = hist_equalize(linear_to_srgb(a), **kw)
    elif method == "clahe":
        out = clahe(linear_to_srgb(a), **kw)
    elif method == "local":
        out = linear_to_srgb(dodge_burn(a, **kw))
    else:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    return np.clip(out, 0.0, 1.0).astype(np.float32)
