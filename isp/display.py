"""Turn any pipeline output into something you can look at (used by the web app and figures).

``render`` accepts the output of every stage (electron / DN / linear mosaics, linear RGB,
display RGB, decoded JPEG) and returns an ``(H, W, 3)`` uint8 image.  No matplotlib here:
only NumPy, OpenCV and Pillow, so the module is safe to import in a web app.

Viewing rules (``render``):

=========  ===========================================================  =================
domain     normalisation                                                curve
=========  ===========================================================  =================
scene      values in [0, 1] (clipped)                                   sRGB
electrons  divided by ``meta['full_well']`` (stretched if dim, see ``stretch``)  sRGB
dn         (DN - black) / (white - black)                               sRGB
linear     as is                                                        sRGB
display    as is                                                        none
bytes      as is                                                        none
=========  ===========================================================  =================

A mosaic is shown through ``cfa.colorize`` (every sample in its own filter colour;
``mosaic_mode='gray'`` shows the plain grey plane instead).  Because only one of three
channels is lit at each pixel, a colourised mosaic looks dark and speckled: that *is*
what the sensor recorded.
"""
from __future__ import annotations

import base64
import io

import cv2
import numpy as np

from . import cfa as _cfa
from . import utils


def _meta_of(obj, meta):
    """(image, meta) from a StageRecord-like object or from an array plus optional meta."""
    if hasattr(obj, "image") and hasattr(obj, "meta"):
        return np.asarray(obj.image), dict(obj.meta)
    img = np.asarray(obj)
    if meta is None:
        if img.dtype == np.uint8:
            meta = {"domain": "display"}
        else:
            meta = {"domain": "display" if img.ndim == 3 else "linear"}
    return img, dict(meta)


def _normalise(img, meta, stretch=True):
    """-> (float array in [0, 1] or beyond, encode_with_srgb flag)."""
    d = meta.get("domain", "display")
    a = img.astype(np.float32)
    if img.dtype == np.uint8:
        return a / 255.0, False
    if d == "electrons":
        fw = float(meta.get("full_well") or max(float(a.max()), 1.0))
        top = float(np.percentile(a, 99.9)) / fw
        if stretch and 0.0 < top < 0.5:                  # the ADC gain will bring this up: show it so
            fw *= top / 0.9
        return a / fw, True
    if d == "dn":
        bl, wl = float(meta.get("black_level", 0.0)), float(meta.get("white_level", max(float(a.max()), 1.0)))
        return (a - bl) / max(wl - bl, 1e-9), True
    if d in ("scene", "linear"):
        return a, True
    return a, False                                  # display, bytes


def _resize(img, size, nearest):
    if size is None:
        return img
    h, w = img.shape[:2]
    if isinstance(size, (tuple, list)):
        nh, nw = int(size[0]), int(size[1])
    else:                                           # longest side
        s = float(size) / max(h, w)
        nh, nw = max(1, round(h * s)), max(1, round(w * s))
    if (nh, nw) == (h, w):
        return img
    interp = cv2.INTER_NEAREST if nearest else (cv2.INTER_AREA if nh * nw < h * w else cv2.INTER_LINEAR)
    return cv2.resize(img, (nw, nh), interpolation=interp)


def render(record_or_image, meta=None, size=None, mosaic_mode="color", stretch=True) -> np.ndarray:
    """Any stage output -> uint8 ``(H, W, 3)`` for display.

    ``record_or_image`` is a ``StageRecord`` (image and meta are taken from it) or an array
    plus ``meta``.  ``size`` is the longest side in pixels (or ``(h, w)``); mosaics are resized
    with nearest-neighbour so the Bayer pattern stays visible, everything else with area /
    bilinear interpolation.  ``mosaic_mode``: ``'color'`` (CFA-coloured) or ``'gray'``.
    ``stretch``: electrons images that use less than half of the full well (typical at high ISO,
    where the ADC gain does the brightening) are scaled so that their 99.9th percentile shows at
    0.9; ``stretch=False`` always divides by the full well.
    """
    img, meta = _meta_of(record_or_image, meta)
    a, encode = _normalise(img, meta, stretch)
    mosaic = a.ndim == 2
    if mosaic:
        a = np.clip(a, 0.0, 1.0)
        if mosaic_mode == "gray":
            a = np.repeat(a[..., None], 3, axis=2)
        elif mosaic_mode == "color":
            a = _cfa.colorize(a, meta.get("cfa", "RGGB"))
        else:
            raise ValueError("mosaic_mode must be 'color' or 'gray'")
    elif a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"cannot render an array of shape {img.shape}")
    a = np.clip(a, 0.0, 1.0)
    if encode:
        a = utils.linear_to_srgb(a)
    a = _resize(a.astype(np.float32), size, nearest=mosaic and mosaic_mode == "color")
    return utils.to_uint8(a)


def crop_zoom(img, box, scale=4) -> np.ndarray:
    """Crop ``box = (row0, row1, col0, col1)`` and enlarge it ``scale`` times with
    nearest-neighbour (pixels stay visible).  Works on uint8 or float, grey or RGB."""
    c = utils.crop(np.asarray(img), box)
    scale = max(1, int(scale))
    return np.ascontiguousarray(np.repeat(np.repeat(c, scale, axis=0), scale, axis=1))


def _unit(a):
    a = np.asarray(a)
    return a.astype(np.float32) / 255.0 if a.dtype == np.uint8 else a.astype(np.float32)


def difference(a, b, gain=4.0) -> np.ndarray:
    """Heat map (uint8 RGB, inferno colours) of ``gain * |a - b|`` averaged over channels.

    ``a`` and ``b`` have the same shape and are [0, 1] floats or uint8; black = identical."""
    a, b = _unit(a), _unit(b)
    if a.shape != b.shape:
        raise ValueError("difference needs two images of the same shape")
    d = np.abs(a - b)
    if d.ndim == 3:
        d = d.mean(axis=2)
    d8 = utils.to_uint8(d * float(gain))
    return cv2.cvtColor(cv2.applyColorMap(d8, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def histograms(img, bins=64):
    """Per-channel histogram of an image in [0, 1] (or uint8).

    Returns ``(edges, counts)``: ``edges`` has ``bins + 1`` entries, ``counts`` is
    ``(C, bins)`` with C = 3 for RGB (R, G, B order) and 1 for a grey image."""
    a = _unit(img)
    chans = [a] if a.ndim == 2 else [a[..., c] for c in range(a.shape[2])]
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    counts = np.stack([np.histogram(np.clip(c, 0, 1), bins=edges)[0] for c in chans])
    return edges, counts


def to_data_uri(img, fmt="png") -> str:
    """uint8 image -> ``data:image/<fmt>;base64,...`` string (for ``<img src=...>``)."""
    from PIL import Image
    fmt = fmt.lower()
    pil_fmt = {"png": "PNG", "jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP"}.get(fmt)
    if pil_fmt is None:
        raise ValueError("fmt must be png, jpeg or webp")
    buf = io.BytesIO()
    Image.fromarray(np.asarray(img, dtype=np.uint8)).save(buf, format=pil_fmt)
    mime = "jpeg" if fmt == "jpg" else fmt
    return f"data:image/{mime};base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def stage_thumbnails(result, size=96) -> dict:
    """``{stage_id: uint8 thumbnail}`` for every record of a ``PipelineResult``
    (longest side ``size``; mosaics keep their Bayer colours)."""
    return {r.name: render(r, size=size) for r in result}
