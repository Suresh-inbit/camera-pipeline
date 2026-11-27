"""Shared helpers: transfer functions, colour transforms, metrics, small array tools."""
from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------
# dtype helpers
# --------------------------------------------------------------------------
def to_float(img) -> np.ndarray:
    """Any image -> float32.  Integer types are scaled by their maximum to [0, 1]."""
    a = np.asarray(img)
    if np.issubdtype(a.dtype, np.integer):
        return (a.astype(np.float32) / np.iinfo(a.dtype).max).astype(np.float32)
    return a.astype(np.float32)


def to_uint8(img) -> np.ndarray:
    """Float [0, 1] image -> uint8 with rounding and clipping."""
    return np.round(np.clip(np.asarray(img, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)


def rng(seed=0) -> np.random.Generator:
    """The project's random generator.  Always pass ``seed`` for reproducibility."""
    return seed if isinstance(seed, np.random.Generator) else np.random.default_rng(seed)


# --------------------------------------------------------------------------
# sRGB transfer function (IEC 61966-2-1)
# --------------------------------------------------------------------------
def srgb_to_linear(x) -> np.ndarray:
    """sRGB-encoded [0, 1] -> linear [0, 1]:  x/12.92 or ((x+0.055)/1.055)^2.4."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def linear_to_srgb(x) -> np.ndarray:
    """Linear [0, 1] -> sRGB-encoded [0, 1]:  12.92 x or 1.055 x^(1/2.4) - 0.055."""
    x = np.clip(np.asarray(x, dtype=np.float32), 0.0, 1.0)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)


# --------------------------------------------------------------------------
# colour transforms
# --------------------------------------------------------------------------
# JPEG / JFIF full-range YCbCr (ITU-R BT.601 luma weights); Cb, Cr centred on 0.5.
_RGB2YCC = np.array([[0.299, 0.587, 0.114],
                     [-0.168736, -0.331264, 0.5],
                     [0.5, -0.418688, -0.081312]], dtype=np.float64)
_YCC2RGB = np.linalg.inv(_RGB2YCC)

# linear sRGB (D65) <-> CIE XYZ
RGB2XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                    [0.2126729, 0.7151522, 0.0721750],
                    [0.0193339, 0.1191920, 0.9503041]], dtype=np.float64)
XYZ2RGB = np.linalg.inv(RGB2XYZ)


def rgb_to_ycbcr(rgb) -> np.ndarray:
    """(H, W, 3) RGB in [0, 1] -> YCbCr with Y in [0, 1] and Cb, Cr in [0, 1] (0.5 = neutral)."""
    ycc = np.asarray(rgb, dtype=np.float64) @ _RGB2YCC.T
    ycc[..., 1:] += 0.5
    return ycc.astype(np.float32)


def ycbcr_to_rgb(ycc) -> np.ndarray:
    """Inverse of :func:`rgb_to_ycbcr` (not clipped)."""
    y = np.asarray(ycc, dtype=np.float64).copy()
    y[..., 1:] -= 0.5
    return (y @ _YCC2RGB.T).astype(np.float32)


def luminance(rgb) -> np.ndarray:
    """Relative luminance Y of linear sRGB (Rec. 709 weights).  2-D input is returned as is."""
    a = np.asarray(rgb, dtype=np.float32)
    if a.ndim == 2:
        return a
    return (a[..., 0] * 0.2126729 + a[..., 1] * 0.7151522 + a[..., 2] * 0.0721750).astype(np.float32)


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def mse(a, b) -> float:
    """Mean squared error."""
    return float(np.mean((np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)) ** 2))


def psnr(a, b, peak: float = 1.0) -> float:
    """PSNR = 10 log10(peak^2 / MSE) in dB; ``inf`` for identical images."""
    m = mse(a, b)
    return float("inf") if m == 0 else float(10.0 * np.log10(peak ** 2 / m))


def ssim(a, b) -> float:
    """Structural similarity (scikit-image) for [0, 1] images, grey or RGB."""
    from skimage.metrics import structural_similarity
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    return float(structural_similarity(a, b, data_range=1.0, channel_axis=-1 if a.ndim == 3 else None))


def snr(mean: float, std: float) -> float:
    """SNR = mean / std (lecture 2 slide 22)."""
    return float(mean / std) if std > 0 else float("inf")


def snr_db(mean: float, std: float) -> float:
    """SNR in dB = 20 log10(mean / std)."""
    return float(20.0 * np.log10(snr(mean, std)))


# --------------------------------------------------------------------------
# small array tools
# --------------------------------------------------------------------------
def crop(img, box) -> np.ndarray:
    """Crop with ``box = (row0, row1, col0, col1)`` (end-exclusive)."""
    r0, r1, c0, c1 = box
    return np.asarray(img)[r0:r1, c0:c1]


def center_box(shape, size) -> tuple:
    """A centred ``size x size`` box for an image of the given shape."""
    h, w = shape[:2]
    r0, c0 = (h - size) // 2, (w - size) // 2
    return (r0, r0 + size, c0, c0 + size)


def gaussian_kernel(sigma: float, radius: int | None = None) -> np.ndarray:
    """Normalised 2-D Gaussian kernel of half-width ``radius`` (default ceil(3 sigma))."""
    if radius is None:
        radius = max(1, int(np.ceil(3.0 * sigma)))
    ax = np.arange(-radius, radius + 1, dtype=np.float64)
    g = np.exp(-(ax ** 2) / (2.0 * sigma ** 2))
    k = np.outer(g, g)
    return (k / k.sum()).astype(np.float32)
