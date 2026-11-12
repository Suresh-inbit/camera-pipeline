"""Denoising (lecture 5, s5-10): Gaussian baseline, bilateral filter, median, luma/chroma
bilateral, non-local means (extension) and temporal averaging (L2 s43).

Everything works on linear ``float32`` images, grey ``(H, W)`` or colour ``(H, W, 3)``.
No plotting here; see ``scripts/make_figures_denoise.py``.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .utils import gaussian_kernel, rgb_to_ycbcr, ycbcr_to_rgb

METHODS = ("gaussian", "bilateral", "median", "luma_chroma", "nlm")


def _f32(img) -> np.ndarray:
    return np.asarray(img, dtype=np.float32)


def _radius(sigma_d: float, radius):
    return int(radius) if radius is not None else max(1, int(np.ceil(2.0 * sigma_d)))


# --------------------------------------------------------------------------
# Gaussian baseline (L5 s5)
# --------------------------------------------------------------------------
def gaussian(img, sigma: float, radius: int | None = None) -> np.ndarray:
    """Gaussian blur, the baseline that "works, but also destroys edges" (L5 s5).

    out = img * g_sigma, with g normalised over a (2r+1)^2 window, r = ceil(3 sigma) by
    default (the same kernel as ``utils.gaussian_kernel``).  Borders are reflected
    (without repeating the edge sample), exactly like :func:`bilateral`.

    Parameters: ``img`` grey or RGB float image; ``sigma`` in pixels; ``radius`` optional
    window half-width.  Returns a float32 image of the same shape.
    Reference: L5 s5.
    """
    img = _f32(img)
    # the 2-D kernel is outer(g, g) / S, so sqrt of its diagonal is g / sqrt(S): a 1-D kernel
    k = np.sqrt(np.diag(gaussian_kernel(sigma, radius).astype(np.float64)))
    k /= k.sum()
    out = img
    for ax in (0, 1):
        out = ndi.correlate1d(out, k, axis=ax, mode="mirror")
    return out.astype(np.float32)


# --------------------------------------------------------------------------
# Bilateral filter (L5 s6-9)
# --------------------------------------------------------------------------
def _offsets(r: int, window: str):
    """Spatial offsets (dy, dx) of the window, optionally restricted to a disc."""
    if window not in ("square", "circle"):
        raise ValueError("window must be 'square' or 'circle'")
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if window == "circle" and dy * dy + dx * dx > r * r:
                continue
            yield dy, dx


def _color_d2(diff: np.ndarray, color_dist: str) -> np.ndarray:
    """Squared colour distance from a (H, W, C) difference array."""
    if color_dist == "l2":
        return np.sum(diff * diff, axis=-1)
    if color_dist == "l1":
        s = np.sum(np.abs(diff), axis=-1)
        return s * s
    raise ValueError("color_dist must be 'l2' or 'l1'")


def bilateral(img, sigma_d: float, sigma_r: float, radius: int | None = None, guide=None,
              window: str = "square", color_dist: str = "l2") -> np.ndarray:
    """Bilateral filter, own vectorised implementation of L5 s8-9.

        w(i,j,k,l) = exp( -((i-k)^2 + (j-l)^2) / (2 sigma_d^2)  -  ||I(i,j) - I(k,l)||^2 / (2 sigma_r^2) )
        I_D(i,j)   = sum_{k,l} I(k,l) w(i,j,k,l)  /  sum_{k,l} w(i,j,k,l)

    The sums run over a (2r+1)^2 window.  We loop over the window *offsets* (shifted copies
    of a reflect-padded array), never over pixels.  Cost: O(H W r^2 C).

    Parameters
    ----------
    img : (H, W) or (H, W, 3) float image (any range; ``sigma_r`` is in the same units).
    sigma_d : spatial (domain) sigma in pixels.
    sigma_r : range sigma, in intensity units.  Large -> Gaussian blur, small -> identity.
    radius : window half-width; default ``ceil(2 sigma_d)``.
    guide : optional image (H, W[, C]) that defines the range distance (joint / cross
        bilateral); the weights come from ``guide``, the averaged values from ``img``.
    window : ``'square'`` (default) or ``'circle'`` (disc of the same radius, as OpenCV).
    color_dist : ``'l2'`` (the formula on the slide, default) or ``'l1'`` (sum of absolute
        channel differences, squared; what OpenCV does for 3 channels).

    Returns float32, same shape as ``img``.  The output is a convex combination of input
    pixels, so it never leaves the input's value range.
    Reference: L5 s8-9; Tomasi & Manduchi, ICCV 1998.
    """
    img = _f32(img)
    g = img if guide is None else _f32(guide)
    if g.shape[:2] != img.shape[:2]:
        raise ValueError("guide must have the same height and width as img")
    r = _radius(sigma_d, radius)
    H, W = img.shape[:2]
    gc = g if g.ndim == 3 else g[..., None]
    ic = img if img.ndim == 3 else img[..., None]
    pad = ((r, r), (r, r), (0, 0))
    gp = np.pad(gc, pad, mode="reflect")
    ip = np.pad(ic, pad, mode="reflect")
    inv_r = 0.5 / float(sigma_r) ** 2 if sigma_r > 0 else np.inf
    inv_d = 0.5 / float(sigma_d) ** 2
    num = np.zeros(ic.shape, np.float32)
    den = np.zeros((H, W), np.float32)
    with np.errstate(over="ignore", invalid="ignore"):
        for dy, dx in _offsets(r, window):
            ws = np.float32(np.exp(-(dy * dy + dx * dx) * inv_d))
            sl = (slice(r + dy, r + dy + H), slice(r + dx, r + dx + W))
            d2 = _color_d2(gp[sl] - gc, color_dist)
            e = d2 * inv_r
            w = np.exp(-e, dtype=np.float32) * ws if np.isfinite(inv_r) else \
                (ws * (d2 == 0)).astype(np.float32)
            num += ip[sl] * w[..., None]
            den += w
    out = num / den[..., None]
    return (out if img.ndim == 3 else out[..., 0]).astype(np.float32)


def bilateral_weights(img, center, sigma_d: float, sigma_r: float, radius: int | None = None,
                      window: str = "square", color_dist: str = "l2") -> np.ndarray:
    """Weight window ``w(i,j,.,.)`` of the bilateral filter at one pixel (L5 s6-7 picture).

    ``center = (row, col)``.  Returns ``(2r+1, 2r+1)`` float32 (not normalised; the centre
    weight is 1).  With ``sigma_r = inf`` this is the Gaussian kernel (L5 s6: "blurring"),
    with finite ``sigma_r`` the part of the window on the other side of an edge fades out
    (L5 s7).  Reference: L5 s7-9.
    """
    img = _f32(img)
    r = _radius(sigma_d, radius)
    c = img if img.ndim == 3 else img[..., None]
    p = np.pad(c, ((r, r), (r, r), (0, 0)), mode="reflect")
    y, x = int(center[0]), int(center[1])
    win = p[y:y + 2 * r + 1, x:x + 2 * r + 1]
    ax = np.arange(-r, r + 1, dtype=np.float64)
    ds = (ax[:, None] ** 2 + ax[None, :] ** 2) / (2.0 * sigma_d ** 2)
    d2 = _color_d2(win - c[y, x], color_dist)
    w = np.exp(-ds - (d2 / (2.0 * sigma_r ** 2) if np.isfinite(sigma_r) else 0.0))
    if window == "circle":
        w = w * (ax[:, None] ** 2 + ax[None, :] ** 2 <= r * r)
    return w.astype(np.float32)


# --------------------------------------------------------------------------
# Other denoisers
# --------------------------------------------------------------------------
def median(img, size: int = 3) -> np.ndarray:
    """Median filter, per channel, ``size x size`` window (good against salt-and-pepper
    / hot pixels, edge preserving but removes thin detail).  Reference: L3 s27 (same idea)."""
    img = _f32(img)
    if img.ndim == 2:
        return ndi.median_filter(img, size=size, mode="mirror").astype(np.float32)
    return np.stack([ndi.median_filter(img[..., c], size=size, mode="mirror")
                     for c in range(img.shape[2])], axis=-1).astype(np.float32)


def estimate_noise(img) -> float:
    """Noise standard deviation from a robust MAD of a high-pass response.

    Convolve with K = [[1,-2,1],[-2,4,-2],[1,-2,1]] (kills smooth image content; for white
    noise the response has std 6 sigma) and take sigma = 1.4826 MAD / 6.  Colour images:
    the mean over channels.  Biased up on very textured images.
    """
    img = _f32(img)
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    chans = [img] if img.ndim == 2 else [img[..., c] for c in range(img.shape[2])]
    out = []
    for ch in chans:
        resp = ndi.correlate(ch, k, mode="mirror")[1:-1, 1:-1]
        out.append(1.4826 * np.median(np.abs(resp - np.median(resp))) / 6.0)
    return float(np.mean(out))


def nlm(img, h: float, patch_size: int = 5, patch_distance: int = 6, sigma: float | None = None) -> np.ndarray:
    """Non-local means (Buades et al. 2005) -- EXTENSION, not on the slides.

    Wraps ``skimage.restoration.denoise_nl_means``: each pixel becomes a weighted mean of
    pixels in a search window, weighted by the similarity of the surrounding *patches*
    (not just single values as in the bilateral filter).  ``h`` is the filtering strength
    in intensity units (about 0.6-1 times the noise sigma works well); ``sigma`` is the
    noise std subtracted from patch distances (default: :func:`estimate_noise`).
    """
    from skimage.restoration import denoise_nl_means
    img = _f32(img)
    if sigma is None:
        sigma = estimate_noise(img)
    return denoise_nl_means(img, patch_size=patch_size, patch_distance=patch_distance, h=h,
                            sigma=sigma, fast_mode=True,
                            channel_axis=-1 if img.ndim == 3 else None).astype(np.float32)


def average_frames(frames) -> np.ndarray:
    """Temporal denoising: mean of M aligned frames (L2 s43).

    Independent noise: std falls as 1/sqrt(M) (the SNR gain of longer exposures without
    saturation).  ``frames`` is a sequence of arrays or an array with frames on axis 0.
    """
    return np.mean(np.asarray(frames, dtype=np.float32), axis=0, dtype=np.float32)


# --------------------------------------------------------------------------
# Luma / chroma (L5 s10)
# --------------------------------------------------------------------------
LUMA_KW = dict(sigma_d=1.5, sigma_r=0.05)
CHROMA_KW = dict(sigma_d=4.0, sigma_r=0.10)


def luma_chroma(img, luma_kw: dict | None = None, chroma_kw: dict | None = None,
                guide_chroma: bool = False) -> np.ndarray:
    """Bilateral filter in YCbCr: gentle on luma, strong on chroma (L5 s10).

    "can be applied more (or less) strongly to chrominance than luminance": the eye
    resolves luminance detail much better than colour detail, so Cb and Cr can be smoothed
    with a big window without visible loss.  Steps: RGB -> YCbCr (``utils.rgb_to_ycbcr``);
    Y <- bilateral(Y, **luma_kw); (Cb, Cr) <- bilateral((Cb, Cr), **chroma_kw); back to RGB.

    ``luma_kw`` / ``chroma_kw`` hold ``sigma_d``, ``sigma_r`` (and optionally ``radius``).
    With ``guide_chroma=True`` the chroma range distance is measured on the denoised luma
    (a joint bilateral filter; keeps colour edges aligned with luma edges, but on noisy
    data the luma is a poorer edge detector than it looks, so it is off by default).
    Output is not clipped.
    Reference: L5 s10.
    """
    img = _f32(img)
    if img.ndim != 3:
        raise ValueError("luma_chroma needs an RGB image")
    lk = dict(LUMA_KW, **(luma_kw or {}))
    ck = dict(CHROMA_KW, **(chroma_kw or {}))
    ycc = rgb_to_ycbcr(img)
    y = bilateral(ycc[..., 0], **lk)
    cc = bilateral(ycc[..., 1:], guide=y if guide_chroma else None, **ck)
    return ycbcr_to_rgb(np.concatenate([y[..., None], cc], axis=-1)).astype(np.float32)


def denoise(img, method: str = "bilateral", **kw) -> np.ndarray:
    """Dispatcher: ``method`` in ``METHODS``; keyword arguments go to that function.

    Defaults (when no kwargs are given): gaussian sigma=1.5; bilateral sigma_d=2,
    sigma_r=0.1; median size=3; luma_chroma as in :func:`luma_chroma`; nlm h=0.05.
    """
    defaults = {"gaussian": dict(sigma=1.5), "bilateral": dict(sigma_d=2.0, sigma_r=0.1),
                "median": dict(size=3), "luma_chroma": {}, "nlm": dict(h=0.05)}
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; choose from {METHODS}")
    fn = {"gaussian": gaussian, "bilateral": bilateral, "median": median,
          "luma_chroma": luma_chroma, "nlm": nlm}[method]
    return fn(img, **(kw or defaults[method]))
