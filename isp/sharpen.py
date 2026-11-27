"""Sharpening (lecture 5, slides 12-17): high-pass filter + unsharp masking.

Sharpening adds a scaled high-pass version of the image back to itself,

    S = x + lam * F(x)                      (L5 s15)

where F is a high-pass filter.  With the 4-neighbour Laplacian F = [[0,-1,0],[-1,4,-1],[0,-1,0]]
this is the single 3x3 kernel [[0,-1,0],[-1,1+4lam,-1],[0,-1,0]]: [[0,-1,0],[-1,5,-1],[0,-1,0]]
for lam=1 (L5 s13, s16) and [[0,-2,0],[-2,9,-2],[0,-2,0]] for lam=2 (L5 s14, s17).

Sharpening works on display-referred (gamma-encoded) values and comes *after* denoising:
the kernel amplifies noise by ``noise_gain`` (5.39 for lam=1).  Everything here is a pure
function on float32 images, ``(H, W)`` or ``(H, W, 3)``, nominal range [0, 1].
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage

from . import utils

__all__ = ["LAPLACIAN", "LAPLACIAN8", "sharpening_kernel", "high_pass_kernel", "high_pass",
           "unsharp_mask", "sharpen", "edge_profile", "overshoot", "rise_distance",
           "mean_gradient", "frequency_response", "unsharp_response", "noise_gain"]

#: 4-neighbour Laplacian used as the high-pass filter on L5 s16 (taps sum to 0).
LAPLACIAN = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32)
#: 8-neighbour variant (taps sum to 0).
LAPLACIAN8 = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)

_KINDS = ("laplacian", "laplacian8", "gaussian")


def _identity(n: int = 3) -> np.ndarray:
    k = np.zeros((n, n), np.float32)
    k[n // 2, n // 2] = 1.0
    return k


def high_pass_kernel(kind: str = "laplacian", sigma: float = 1.0) -> np.ndarray:
    """Kernel of the high-pass filter F.

    'laplacian' / 'laplacian8': the 3x3 kernels above.  'gaussian': delta - G_sigma, the
    classical unsharp-mask high-pass (x minus a blurred copy).  All taps sum to 0.
    Reference: L5 s15-16."""
    if kind == "laplacian":
        return LAPLACIAN.copy()
    if kind == "laplacian8":
        return LAPLACIAN8.copy()
    if kind == "gaussian":
        g = utils.gaussian_kernel(sigma)
        return (_identity(g.shape[0]) - g).astype(np.float32)
    raise ValueError(f"kind must be one of {_KINDS}, got {kind!r}")


def sharpening_kernel(lam: float = 1.0, kind: str = "laplacian", sigma: float = 1.0) -> np.ndarray:
    """Single kernel equal to ``identity + lam * high_pass_kernel``.

    ``sharpening_kernel(1)`` = [[0,-1,0],[-1,5,-1],[0,-1,0]] and ``sharpening_kernel(2)`` =
    [[0,-2,0],[-2,9,-2],[0,-2,0]], the Photoshop 'Custom' kernels (scale 1, offset 0) of
    L5 s13-14.  Taps sum to 1, so mean brightness is preserved."""
    hp = high_pass_kernel(kind, sigma)
    return (_identity(hp.shape[0]) + float(lam) * hp).astype(np.float32)


def _f32(img) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(img, dtype=np.float32))


def high_pass(img, kind: str = "laplacian", sigma: float = 1.0) -> np.ndarray:
    """High-pass image F(x).

    'laplacian': 4*x - (4 neighbours); 'laplacian8': 8*x - (8 neighbours);
    'gaussian': x - GaussianBlur(x, sigma).  Borders are reflected.  Not clipped (signed).
    Reference: L5 s15-16."""
    img = _f32(img)
    if kind == "gaussian":
        return img - cv2.GaussianBlur(img, (0, 0), sigma, borderType=cv2.BORDER_REFLECT_101)
    k = high_pass_kernel(kind)
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT_101)


def unsharp_mask(img, lam: float = 1.0, kind: str = "laplacian", sigma: float = 1.0,
                 threshold: float = 0.0, clip: bool = True, luminance_only: bool = False) -> np.ndarray:
    """Unsharp masking  S = x + lam * F(x)  (L5 s15).

    Parameters
    ----------
    lam : strength (>= 0); 0 returns the input.
    kind, sigma : high-pass filter, see :func:`high_pass`.
    threshold : where |F(x)| < threshold the high-pass term is set to 0, so weak detail
        (mostly noise) is not amplified while strong edges are still sharpened.
    clip : clip the result to [0, 1].
    luminance_only : for RGB input, sharpen only Y of YCbCr (JPEG BT.601) and keep Cb, Cr;
        this avoids colour fringes.  Cb and Cr are exactly preserved before clipping.

    Returns float32 of the input shape.  Reference: L5 s15."""
    img = _f32(img)
    if luminance_only and img.ndim == 3 and img.shape[2] == 3:
        ycc = utils.rgb_to_ycbcr(img)
        ycc[..., 0] = unsharp_mask(ycc[..., 0], lam, kind, sigma, threshold, clip=False)
        out = utils.ycbcr_to_rgb(ycc)
    else:
        f = high_pass(img, kind, sigma)
        if threshold > 0:
            f = np.where(np.abs(f) >= threshold, f, 0.0).astype(np.float32)
        out = img + np.float32(lam) * f
    if clip:
        out = np.clip(out, 0.0, 1.0)
    return out.astype(np.float32)


def sharpen(img, method: str = "laplacian", lam: float = 1.0, **kw) -> np.ndarray:
    """Dispatcher: 'laplacian' (slide kernels) or 'unsharp' (Gaussian unsharp mask,
    ``sigma`` keyword).  Other keywords go to :func:`unsharp_mask`."""
    if method == "laplacian":
        return unsharp_mask(img, lam, kind=kw.pop("kind", "laplacian"), **kw)
    if method == "unsharp":
        return unsharp_mask(img, lam, kind="gaussian", **kw)
    raise ValueError("method must be 'laplacian' or 'unsharp'")


# --------------------------------------------------------------------------
# analysis helpers
# --------------------------------------------------------------------------
def edge_profile(img, angle_deg: float = 5.0, half: float = 12.0, step: float = 0.25):
    """1-D profile across the edge of ``scenes.step_edge(size, angle_deg=...)``.

    Samples the (grey) image on the line through the image centre along the edge normal,
    with bilinear interpolation.  Returns ``(d, profile)``; ``d`` in pixels, 0 = edge."""
    a = _f32(img)
    if a.ndim == 3:
        a = a.mean(axis=2)
    h, w = a.shape
    t = np.deg2rad(angle_deg)
    d = np.arange(-half, half + 1e-9, step)
    rows = h / 2.0 + d * np.sin(t)
    cols = w / 2.0 + d * np.cos(t)
    prof = ndimage.map_coordinates(a, [rows, cols], order=1, mode="nearest")
    return d.astype(np.float32), prof.astype(np.float32)


def _plateaus(p: np.ndarray):
    n = max(2, len(p) // 8)
    return float(np.median(p[:n])), float(np.median(p[-n:]))


def overshoot(profile) -> float:
    """Halo size of a 1-D edge profile, as a fraction of the step.

    lo / hi are the plateau levels (medians of the first / last eighth of the profile);
    the result is max(max(p) - hi, lo - min(p)) / |hi - lo| for a rising edge (mirrored
    for a falling one).  0 for a monotone edge, growing with lam after sharpening."""
    p = np.asarray(profile, dtype=np.float64)
    lo, hi = _plateaus(p)
    if hi < lo:
        p, lo, hi = -p, -lo, -hi
    step = hi - lo
    if step <= 0:
        return 0.0
    return float(max(p.max() - hi, lo - p.min(), 0.0) / step)


def rise_distance(d, profile, lo_frac: float = 0.1, hi_frac: float = 0.9) -> float:
    """10-90 % rise distance (pixels) of an edge profile: distance between the first
    upward crossings of lo + 0.1 step and lo + 0.9 step (linear interpolation).  Smaller =
    steeper edge."""
    d = np.asarray(d, dtype=np.float64)
    p = np.asarray(profile, dtype=np.float64)
    lo, hi = _plateaus(p)
    if hi < lo:
        p, lo, hi = -p, -lo, -hi
    step = hi - lo

    def cross(level):
        idx = np.nonzero((p[:-1] < level) & (p[1:] >= level))[0]
        if len(idx) == 0:
            return np.nan
        i = idx[0]
        return d[i] + (level - p[i]) / (p[i + 1] - p[i]) * (d[i + 1] - d[i])

    return float(cross(lo + hi_frac * step) - cross(lo + lo_frac * step))


def mean_gradient(img) -> float:
    """Mean gradient magnitude (central differences, grey): a simple sharpness / acutance
    proxy that grows when edges become steeper (and when noise is amplified)."""
    a = _f32(img)
    if a.ndim == 3:
        a = a.mean(axis=2)
    gy, gx = np.gradient(a)
    return float(np.mean(np.hypot(gx, gy)))


def frequency_response(kernel, n: int = 64) -> np.ndarray:
    """Frequency response H = FFT2 of a (symmetric) kernel zero-padded to ``n x n``.

    Real-valued, in FFT order: ``H[0, 0]`` is DC, ``H[n//2, 0]`` / ``H[0, n//2]`` is
    (pi, 0) / (0, pi) and ``H[n//2, n//2]`` is (pi, pi), in radians per pixel."""
    k = np.asarray(kernel, dtype=np.float64)
    pad = np.zeros((n, n))
    for (i, j), v in np.ndenumerate(k):     # centre the kernel at the origin (wrap-around)
        pad[(i - k.shape[0] // 2) % n, (j - k.shape[1] // 2) % n] = v
    return np.real(np.fft.fft2(pad))


def unsharp_response(lam: float = 1.0, kind: str = "laplacian", sigma: float = 1.0, n: int = 64) -> np.ndarray:
    """Response of the unsharp operator, 1 + lam * H_highpass(f), same layout as
    :func:`frequency_response`.  1 at DC; 1 + 8 lam at (pi, pi) and 1 + 4 lam at (pi, 0)
    for the 4-neighbour Laplacian."""
    return frequency_response(sharpening_kernel(lam, kind, sigma), n)


def noise_gain(kernel) -> float:
    """Factor by which a linear kernel multiplies the std of white noise:
    sqrt(sum of squared taps).  sqrt(5^2 + 4*1^2) = 5.39 for lam=1, sqrt(81+16) = 9.85 for lam=2."""
    return float(np.sqrt(np.sum(np.asarray(kernel, dtype=np.float64) ** 2)))
