"""Reading and writing images: ordinary pictures as linear scenes, RAW files as mosaics.

* ``load_image``  PNG / JPEG / ... -> linear float32 RGB (a ground-truth scene for ``simulate``)
* ``load_raw``    camera RAW (needs the optional ``rawpy`` package) -> ``(mosaic_dn, meta)`` that
                  enters the pipeline at ``start='black_level'`` (see ``pipeline.process_raw``)
* ``save_png``    float [0, 1] or uint8 image -> PNG file
"""
from __future__ import annotations

import io as _io
import os

import numpy as np

from . import cfa as _cfa
from . import utils


def _open(path_or_bytes):
    from PIL import Image, ImageOps
    src = _io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    im = Image.open(src)
    return ImageOps.exif_transpose(im)


def load_image(path_or_bytes, size=None, square=True) -> np.ndarray:
    """Load a picture as a *linear* float32 RGB scene ``(H, W, 3)`` in [0, 1].

    The 8- or 16-bit sRGB values are linearised with the inverse sRGB curve.  Without
    ``size`` the picture is cropped (not resized) so both sides are multiples of 16, which
    the Bayer pattern and the JPEG blocks need.  With ``size`` it is centre-cropped to a square
    (unless ``square=False``, then the aspect ratio is kept and ``size`` is the longer side)
    and resized to a multiple of 16 (rounded down, at least 32).
    """
    im = _open(path_or_bytes)
    if im.mode in ("I;16", "I;16L", "I;16B", "I"):
        a = np.asarray(im, dtype=np.float32) / 65535.0
        a = np.repeat(a[..., None], 3, axis=2)
    else:
        a = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    h, w = a.shape[:2]
    if size is None:
        h16, w16 = max(32, h // 16 * 16), max(32, w // 16 * 16)
        r0, c0 = (h - h16) // 2, (w - w16) // 2
        a = a[r0:r0 + h16, c0:c0 + w16]
    else:
        size = max(32, int(size) // 16 * 16)
        if square:
            s = min(h, w)
            r0, c0 = (h - s) // 2, (w - s) // 2
            a = a[r0:r0 + s, c0:c0 + s]
            nh = nw = size
        else:
            k = size / max(h, w)
            nh, nw = max(32, int(h * k) // 16 * 16), max(32, int(w * k) // 16 * 16)
        import cv2
        interp = cv2.INTER_AREA if nh * nw < a.shape[0] * a.shape[1] else cv2.INTER_LINEAR
        a = cv2.resize(a, (nw, nh), interpolation=interp)
    return utils.srgb_to_linear(a)


def save_png(img, path) -> str:
    """Save a float [0, 1] (clipped) or uint8 image, grey or RGB, as PNG.  Returns ``path``.
    Pass display-referred data (e.g. ``result.final.image``); linear data would look dark."""
    from PIL import Image
    a = np.asarray(img)
    if a.dtype != np.uint8:
        a = utils.to_uint8(a)
    Image.fromarray(a).save(os.fspath(path), format="PNG")
    return path


# --------------------------------------------------------------------------
# RAW files
# --------------------------------------------------------------------------
def _rawpy():
    try:
        import rawpy
    except ImportError as e:                                     # pragma: no cover - message tested
        raise ImportError(
            "load_raw needs the optional package 'rawpy' (LibRaw bindings): pip install rawpy. "
            "Without it, use load_image for ordinary pictures or isp.sensor to simulate RAW data."
        ) from e
    return rawpy


def _pattern_from(raw):
    """Bayer pattern string of the visible area from rawpy's documented attributes.

    ``raw_colors_visible`` (colour index per pixel of ``raw_image_visible``) is preferred because
    it already accounts for the crop margins; otherwise ``raw_pattern`` (2x2 colour indices)
    is used.  Indices point into ``color_desc`` (e.g. b'RGBG'); the second green maps to 'G'.
    """
    desc = raw.color_desc.decode() if isinstance(raw.color_desc, (bytes, bytearray)) else str(raw.color_desc)
    cv = getattr(raw, "raw_colors_visible", None)
    tile = np.asarray(cv)[:2, :2] if cv is not None else np.asarray(raw.raw_pattern)
    if tile.shape != (2, 2):
        raise ValueError("only 2x2 Bayer CFAs are supported (got a "
                         f"{tile.shape[0]}x{tile.shape[1]} pattern, e.g. X-Trans)")
    pattern = "".join(desc[int(i)] for i in tile.ravel())
    if pattern not in _cfa.PATTERNS:
        raise ValueError(f"unsupported CFA pattern {pattern!r} (colours {desc!r})")
    return pattern, desc, tile


def load_raw(path_or_bytes):
    """Read a camera RAW file (needs ``rawpy``) -> ``(mosaic_dn, meta)``.

    ``mosaic_dn`` is the visible area ``raw_image_visible`` as float32 DN, cropped to even
    size.  ``meta`` follows docs/conventions.md: ``domain='dn'``, ``cfa`` (e.g. 'RGGB'),
    ``black_level`` (mean of ``black_level_per_channel``), ``white_level``, ``bits``, and
    ``camera_wb`` = the camera's white-balance multipliers (R, G, B) normalised to G = 1
    (``camera_whitebalance``), so ``white_balance(method='camera')`` can use them.  If the
    per-channel black levels differ, each site is shifted so that one common ``black_level``
    applies.  Feed the result to ``pipeline.process_raw`` or ``Pipeline.run(..., start='black_level')``.
    """
    rawpy = _rawpy()
    src = _io.BytesIO(path_or_bytes) if isinstance(path_or_bytes, (bytes, bytearray)) else path_or_bytes
    with rawpy.imread(src) as raw:
        mosaic = np.array(raw.raw_image_visible, dtype=np.float32)
        pattern, desc, tile = _pattern_from(raw)
        blacks = np.asarray(raw.black_level_per_channel, dtype=np.float64)
        white = float(raw.white_level)
        cam_wb = np.asarray(raw.camera_whitebalance, dtype=np.float64)
    h, w = mosaic.shape[0] // 2 * 2, mosaic.shape[1] // 2 * 2
    mosaic = mosaic[:h, :w]
    black = float(blacks.mean())
    if np.ptp(blacks) > 0:                                       # per-site black -> one common value
        site = np.tile(blacks[np.asarray(tile, dtype=int)], (h // 2, w // 2))
        mosaic = mosaic - site.astype(np.float32) + np.float32(black)
    names = {c: desc.index(c) for c in "RGB"}                    # first index of each colour
    g = cam_wb[names["G"]] if cam_wb[names["G"]] > 0 else 1.0
    camera_wb = [float(cam_wb[names[c]] / g) for c in "RGB"]
    meta = dict(domain="dn", cfa=pattern, black_level=black, white_level=white,
                bits=int(np.ceil(np.log2(white + 1.0))), camera_wb=camera_wb)
    return mosaic, meta
