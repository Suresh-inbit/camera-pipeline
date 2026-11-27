"""Test scenes: the ground truth every experiment starts from.

All scenes are *linear* RGB, ``float32``, shape ``(size, size, 3)``, values in [0, 1].
``size`` is rounded down to a multiple of 16 (even for the Bayer pattern, whole 8x8 /
16x16 blocks for JPEG).  Photographs come from ``skimage.data`` (no downloads) and are
linearised with the inverse sRGB curve; the charts are synthetic.
"""
from __future__ import annotations

import numpy as np

from .utils import srgb_to_linear

PHOTOS = ("astronaut", "coffee", "chelsea", "rocket")
CHARTS = ("patches", "zone_plate", "step_edge", "gray_ramp", "high_contrast", "sunset")
NAMES = PHOTOS + CHARTS

# sRGB values of the 24 patches of a Macbeth-style colour chart (row-major, 4 x 6).
# The last row is the neutral series, white to black.
_PATCH_SRGB = np.array([
    [115, 82, 68], [194, 150, 130], [98, 122, 157], [87, 108, 67], [133, 128, 177], [103, 189, 170],
    [214, 126, 44], [80, 91, 166], [193, 90, 99], [94, 60, 108], [157, 188, 64], [224, 163, 46],
    [56, 61, 150], [70, 148, 73], [175, 54, 60], [231, 199, 31], [187, 86, 149], [8, 133, 161],
    [243, 243, 242], [200, 200, 200], [160, 160, 160], [122, 122, 121], [85, 85, 85], [52, 52, 52],
], dtype=np.float32) / 255.0


def _size(size: int) -> int:
    return max(32, (int(size) // 16) * 16)


def patch_boxes(size: int = 256, rows: int = 4, cols: int = 6, margin: float = 0.2) -> list:
    """Inner boxes ``(r0, r1, c0, c1)`` of the 24 chart patches, row-major.

    Boxes are shrunk by ``margin`` on each side so they avoid patch borders.
    Patches 18..23 are neutral (white ... black)."""
    size = _size(size)
    ph, pw = size / rows, size / cols
    out = []
    for r in range(rows):
        for c in range(cols):
            out.append((int((r + margin) * ph), int((r + 1 - margin) * ph),
                        int((c + margin) * pw), int((c + 1 - margin) * pw)))
    return out


def patches(size: int = 256) -> np.ndarray:
    """24-patch colour chart (4 rows x 6 columns), no borders."""
    size = _size(size)
    lin = srgb_to_linear(_PATCH_SRGB).reshape(4, 6, 3)
    rr = np.minimum((np.arange(size) * 4) // size, 3)
    cc = np.minimum((np.arange(size) * 6) // size, 5)
    return lin[rr[:, None], cc[None, :]].astype(np.float32)


def zone_plate(size: int = 256) -> np.ndarray:
    """Grey circular zone plate 0.5 + 0.5 cos(pi r^2 / size): the spatial frequency grows
    linearly with radius and reaches Nyquist at the image edge (moire / aliasing test)."""
    size = _size(size)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64) - size / 2.0
    z = 0.5 + 0.5 * np.cos(np.pi * (x ** 2 + y ** 2) / size)
    return np.repeat(z[..., None], 3, axis=2).astype(np.float32)


def step_edge(size: int = 256, lo: float = 0.2, hi: float = 0.8, angle_deg: float = 5.0) -> np.ndarray:
    """Slightly slanted grey step edge (anti-aliased): sharpening overshoot / edge test."""
    size = _size(size)
    y, x = np.mgrid[0:size, 0:size].astype(np.float64) - size / 2.0
    t = np.deg2rad(angle_deg)
    d = x * np.cos(t) + y * np.sin(t)                    # signed distance to the edge
    z = lo + (hi - lo) * np.clip(d + 0.5, 0.0, 1.0)
    return np.repeat(z[..., None], 3, axis=2).astype(np.float32)


def gray_ramp(size: int = 256) -> np.ndarray:
    """Horizontal linear grey ramp from 0 to 1."""
    size = _size(size)
    z = np.tile(np.linspace(0.0, 1.0, size), (size, 1))
    return np.repeat(z[..., None], 3, axis=2).astype(np.float32)


def sunset(size: int = 256) -> np.ndarray:
    """Sunset-like scene (orange horizon, blue sky, black silhouette): its average is far
    from grey, so gray-world white balance fails on it (lecture 4 slide 10)."""
    size = _size(size)
    v = np.linspace(0.0, 1.0, size)[:, None]             # 0 = top, 1 = bottom
    top, hor = np.array([0.02, 0.05, 0.25]), np.array([0.95, 0.35, 0.05])
    w = np.clip(v / 0.75, 0, 1) ** 2.0
    sky = (1 - w)[..., None] * top + w[..., None] * hor  # (size, 1, 3)
    img = np.repeat(sky, size, axis=1)
    x = np.linspace(0.0, 1.0, size)[None, :]
    hills = 0.75 + 0.06 * np.sin(6.0 * x) + 0.03 * np.sin(17.0 * x + 1.0)
    img[np.broadcast_to(v, (size, size)) > hills] = 0.005
    return img.astype(np.float32)


def photo(name: str = "astronaut", size: int = 256) -> np.ndarray:
    """A ``skimage.data`` photograph, centre-cropped to a square, resized and linearised."""
    import skimage.data as skdata
    from skimage.transform import resize
    size = _size(size)
    img = getattr(skdata, name)()
    h, w = img.shape[:2]
    s = min(h, w)
    img = img[(h - s) // 2:(h - s) // 2 + s, (w - s) // 2:(w - s) // 2 + s]
    img = resize(img.astype(np.float32) / 255.0, (size, size), anti_aliasing=True, order=1)
    return srgb_to_linear(img)


def high_contrast(size: int = 256, name: str = "astronaut") -> np.ndarray:
    """High dynamic range scene: a photograph whose left side is ~50x darker than its
    right side (dim interior next to a bright window).  For tone-mapping experiments."""
    img = photo(name, size)
    x = np.linspace(0.0, 1.0, img.shape[1])
    s = np.clip((x - 0.35) / 0.3, 0.0, 1.0)
    s = s * s * (3 - 2 * s)                              # smoothstep
    gain = 0.02 + 0.98 * s
    return (img * gain[None, :, None]).astype(np.float32)


def load(name: str = "astronaut", size: int = 256) -> np.ndarray:
    """Load a scene by name (see ``NAMES``) as linear float32 RGB ``(size, size, 3)``."""
    if name in PHOTOS:
        return photo(name, size)
    if name in CHARTS:
        return globals()[name](size)
    raise ValueError(f"unknown scene {name!r}; choose from {NAMES}")
