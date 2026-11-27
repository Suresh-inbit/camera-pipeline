"""Colour filter arrays (lecture 3, Sensing Color).

A sensor site records one number, so a colour filter array (CFA) lets each site see
only one of R, G, B.  A RAW *mosaic* is therefore a single ``(H, W)`` plane that
interleaves the three colour planes (L3 s13-16).  This module builds those mosaics
from an RGB image and shows them again in colour.

Bayer pattern strings read row-major over the 2x2 tile: ``'RGGB'`` is
``[[R, G], [G, B]]`` with R at ``(0, 0)``.
"""
from __future__ import annotations

import numpy as np

PATTERNS = ("RGGB", "BGGR", "GRBG", "GBRG")

_CH = {"R": 0, "G": 1, "B": 2}


def _check(pattern: str) -> str:
    if pattern not in PATTERNS:
        raise ValueError(f"unknown Bayer pattern {pattern!r}; choose from {PATTERNS}")
    return pattern


def channel_index(shape, pattern: str = "RGGB") -> np.ndarray:
    """``(H, W)`` int array: 0/1/2 = which colour (R/G/B) each site samples."""
    _check(pattern)
    h, w = shape[:2]
    tile = np.array([_CH[c] for c in pattern]).reshape(2, 2)
    return np.tile(tile, ((h + 1) // 2, (w + 1) // 2))[:h, :w]


def masks(shape, pattern: str = "RGGB") -> np.ndarray:
    """Boolean ``(H, W, 3)`` array: where each colour is sampled.

    Every pixel belongs to exactly one colour; for Bayer, G is 50 % and R, B 25 % each.

    Parameters: ``shape`` (H, W[, 3]) of the image, ``pattern`` one of ``PATTERNS``.
    Reference: L3 s13 (Bayer pattern), s17 (neighbour layout per colour).
    """
    idx = channel_index(shape, pattern)
    return idx[..., None] == np.arange(3)


def mosaic(img, pattern: str = "RGGB") -> np.ndarray:
    """Sample an RGB image through the CFA: ``out[y, x] = img[y, x, channel(y, x)]``.

    Returns a float32 ``(H, W)`` mosaic.  Two thirds of the numbers in ``img`` are discarded.

    Reference: L3 s13, s16 ("before demosaicking").
    """
    img = np.asarray(img, dtype=np.float32)
    idx = channel_index(img.shape, pattern)
    return np.take_along_axis(img, idx[..., None], axis=2)[..., 0].astype(np.float32)


def colorize(mosaic_img, pattern: str = "RGGB") -> np.ndarray:
    """Show a mosaic in colour: each sample is placed in its own filter channel, others 0.

    This is the "what the sensor really recorded" view; it is not demosaicking.

    Reference: L3 s13.
    """
    m = np.asarray(mosaic_img, dtype=np.float32)
    return (masks(m.shape, pattern) * m[..., None]).astype(np.float32)


# --------------------------------------------------------------------------
# Other CFAs of L3 s13 (display only: demosaic.py handles Bayer only)
# --------------------------------------------------------------------------
# Sony RGB+E: better colour (an emerald fourth filter).
SONY_RGBE_TILE = np.array([list("RERE"), list("GBGB"), list("RERE"), list("GBGB")])
# Kodak RGB+W (panchromatic "white" sites): about 2x more light, less noise.
KODAK_RGBW_TILE = np.array([list("WBWG"), list("BWGW"), list("WGWR"), list("GWRW")])
BAYER_TILE = np.array([list("RG"), list("GB")])

_FILTER_RGB = {"R": (0.85, 0.12, 0.10), "G": (0.25, 0.70, 0.20), "B": (0.20, 0.30, 0.80),
               "E": (0.25, 0.75, 0.75), "W": (1.0, 1.0, 1.0)}


def tile_colors(tile, reps: int = 1) -> np.ndarray:
    """Letter tile (array of 'R','G','B','E','W') -> ``(h, w, 3)`` swatch image, repeated ``reps`` times."""
    t = np.asarray(tile)
    img = np.array([[_FILTER_RGB[c] for c in row] for row in t], dtype=np.float32)
    return np.tile(img, (reps, reps, 1))
