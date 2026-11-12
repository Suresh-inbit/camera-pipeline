"""The camera pipeline as an inspectable list of stages.

A *stage* is any function ``fn(image, meta, **params) -> (image, meta)``.
``Pipeline.run`` applies the enabled stages in order and keeps a ``StageRecord`` for
every stage, so each intermediate image can be displayed, measured and compared.
Stages can be switched off, re-parameterised and re-ordered ("the processing order
may vary", lecture 2 slide 3).

The default camera (scene -> sensor -> ADC -> processing -> storage, the boxes of L2 s3)
is built by ``default_pipeline``; ``simulate`` runs it on a test scene with per-stage
metrics against the ground truth.  ``STAGE_INFO`` / ``GROUP_INFO`` describe the stages for
diagrams and READMEs.  See docs/00_camera_pipeline.md.
"""
from __future__ import annotations

import copy
import inspect
import time
from dataclasses import asdict, dataclass, field, replace
from typing import Callable

import numpy as np

from . import cfa as _cfa
from . import demosaic as _demosaic
from . import denoise as _denoise
from . import jpeg as _jpeg
from . import scenes as _scenes
from . import sensor as _sensor
from . import sharpen as _sharpen
from . import tonemap as _tonemap
from . import utils
from . import whitebalance as _wb

GROUPS = ("scene", "sensor", "adc", "processing", "storage")   # the boxes of slide 3


@dataclass
class Stage:
    """One step of the pipeline.

    name    unique id, e.g. ``'demosaic'``
    fn      ``fn(image, meta, **params) -> (image, meta)``; must not modify its inputs
    params  keyword arguments for ``fn``
    enabled disabled stages pass their input through unchanged
    group   one of ``GROUPS`` (which box of the slide-3 diagram the stage belongs to)
    label   human-readable name for diagrams (defaults to ``name``)
    """
    name: str
    fn: Callable
    params: dict = field(default_factory=dict)
    enabled: bool = True
    group: str = "processing"
    label: str | None = None

    def __post_init__(self):
        if self.group not in GROUPS:
            raise ValueError(f"group must be one of {GROUPS}, got {self.group!r}")
        if self.label is None:
            self.label = self.name


@dataclass
class StageRecord:
    """What a stage produced: its output, the meta data and how long it took."""
    name: str
    label: str
    group: str
    image: np.ndarray
    meta: dict
    params: dict
    enabled: bool
    seconds: float
    metrics: dict = field(default_factory=dict)
    reused: bool = False          # True when copied from a ``previous`` result (not recomputed)

    @property
    def summary(self) -> dict:
        a = self.image
        return dict(name=self.name, group=self.group, enabled=self.enabled,
                    domain=self.meta.get("domain"), shape=tuple(a.shape), dtype=str(a.dtype),
                    min=float(np.min(a)), max=float(np.max(a)), seconds=self.seconds, **self.metrics)


class PipelineResult:
    """Ordered stage records; index by position or by stage name."""

    def __init__(self, records: list, input_image=None, input_meta=None, start=None,
                 sigs=None, metrics_key=None, metrics=None):
        self.records = list(records)
        # bookkeeping for ``Pipeline.run(previous=...)``
        self.input_image = input_image
        self.input_meta = input_meta
        self.start = start
        self.sigs = sigs if sigs is not None else []
        self.metrics_key = metrics_key
        self.metrics = metrics          # the StageMetrics object used (or None)

    def __getitem__(self, key) -> StageRecord:
        if isinstance(key, str):
            for r in self.records:
                if r.name == key:
                    return r
            raise KeyError(key)
        return self.records[key]

    def __iter__(self):
        return iter(self.records)

    def __len__(self):
        return len(self.records)

    @property
    def names(self) -> list:
        return [r.name for r in self.records]

    @property
    def final(self) -> StageRecord:
        return self.records[-1]

    def table(self) -> list:
        """One summary dict per stage (handy for ``pandas.DataFrame``)."""
        return [r.summary for r in self.records]


class Pipeline:
    """An ordered, editable list of stages."""

    def __init__(self, stages: list):
        self.stages = list(stages)
        names = [s.name for s in self.stages]
        if len(set(names)) != len(names):
            raise ValueError("stage names must be unique")

    # ---- editing ---------------------------------------------------------
    @property
    def names(self) -> list:
        return [s.name for s in self.stages]

    def index(self, name: str) -> int:
        return self.names.index(name)

    def __getitem__(self, name: str) -> Stage:
        return self.stages[self.index(name)]

    def copy(self) -> "Pipeline":
        """Independent copy (stage params are deep-copied, functions are shared)."""
        return Pipeline([Stage(s.name, s.fn, copy.deepcopy(s.params), s.enabled, s.group, s.label)
                         for s in self.stages])

    def set(self, name: str, **params) -> "Pipeline":
        """Update parameters of a stage.  Returns ``self`` for chaining."""
        self[name].params.update(params)
        return self

    def enable(self, name: str, enabled: bool = True) -> "Pipeline":
        self[name].enabled = bool(enabled)
        return self

    def move(self, name: str, before: str | None = None, after: str | None = None) -> "Pipeline":
        """Move a stage directly before or after another one."""
        if (before is None) == (after is None):
            raise ValueError("give exactly one of before= / after=")
        stage = self.stages.pop(self.index(name))
        i = self.index(before) if before is not None else self.index(after) + 1
        self.stages.insert(i, stage)
        return self

    # ---- running ---------------------------------------------------------
    def run(self, image, meta: dict | None = None, metrics: Callable | None = None,
            start: str | None = None, previous: "PipelineResult | None" = None) -> PipelineResult:
        """Run the stages on ``image``.

        meta      initial meta dict (default ``{'domain': 'scene'}``)
        metrics   optional ``metrics(image, meta, stage_name) -> dict`` evaluated on every
                  stage output (e.g. :class:`StageMetrics`, PSNR against the ground truth)
        start     begin at this stage id and skip the earlier ones (a real RAW file enters at
                  ``'black_level'``); the result then holds records for the stages run only
        previous  a earlier :class:`PipelineResult` of this pipeline.  The longest prefix of
                  stages whose id, function, enabled flag, parameters and input image are
                  unchanged is copied (``record.reused = True``), everything after the first
                  change is recomputed.  Result is identical to a full run, only faster.
        """
        img = np.asarray(image)
        meta0 = dict(meta) if meta is not None else {"domain": "scene"}
        i0 = 0 if start is None else self.index(start)
        run_stages = self.stages[i0:]
        mkey = getattr(metrics, "key", None) if metrics is not None else None
        if metrics is not None and mkey is None:
            mkey = id(metrics)
        sigs = [(s.name, s.fn, s.enabled, copy.deepcopy(s.params)) for s in run_stages]

        records, k = [], 0
        if (previous is not None and previous.start == start and previous.metrics_key == mkey
                and previous.input_image is not None
                and _same(previous.input_image, img) and _same(previous.input_meta, meta0)):
            for a, b in zip(previous.sigs, sigs):
                if not (a[0] == b[0] and a[1] is b[1] and a[2] == b[2] and _same(a[3], b[3])):
                    break
                k += 1
            records = [replace(r, reused=True) for r in previous.records[:k]]
        meta = dict(meta0)
        if k:
            img, meta = records[-1].image, dict(records[-1].meta)

        for s in run_stages[k:]:
            t0 = time.perf_counter()
            if s.enabled:
                img, meta = s.fn(img, dict(meta), **s.params)
                meta = dict(meta)
            dt = time.perf_counter() - t0
            m = dict(metrics(img, meta, s.name)) if metrics is not None else {}
            records.append(StageRecord(s.name, s.label, s.group, img, dict(meta),
                                       copy.deepcopy(s.params), s.enabled, dt, m))
        return PipelineResult(records, input_image=np.asarray(image), input_meta=meta0, start=start,
                              sigs=sigs, metrics_key=mkey, metrics=metrics)


def _same(a, b) -> bool:
    """Deep equality that understands NumPy arrays (for parameter / input comparison)."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return bool(a == b)


# ==========================================================================
# The default camera: stage functions
# ==========================================================================
# Every stage function is ``fn(image, meta, **params) -> (image, meta)``; it never modifies
# its inputs and keeps ``meta`` consistent with docs/conventions.md:
#   meta['domain']       scene | electrons | dn | linear | display | bytes
#   meta['cfa']          Bayer pattern, present only while the data is a mosaic
#   meta['scene_scale']  linear units per unit of scene value (exposure, ISO, digital gain),
#                        so the ground truth can be rescaled to what the camera recorded
# Order rules (a stage that cannot work raises ValueError naming itself):
#   illuminant, exposure   need scene-domain light         (before the sensor)
#   adc                    needs electrons                 (after exposure)
#   black_level            needs DN                        (after adc)
#   white_balance, tonemap need linear data (or scene)     (after black_level; WB works on a
#                          mosaic or on RGB, tonemap needs RGB)
#   cfa                    needs an RGB image, no-op on a mosaic that already has meta['cfa']
#   demosaic               no-op on an RGB image
#   hot_pixels, denoise    any mosaic / RGB (denoise filters each CFA site plane of a mosaic)
#   sharpen                any RGB;  jpeg: any image (a mosaic is stored as a grey image)
def _f32(a):
    return np.ascontiguousarray(np.asarray(a, dtype=np.float32))


def _need(stage, meta, *domains):
    d = meta.get("domain")
    if d not in domains:
        raise ValueError(f"stage '{stage}' needs data in domain {' or '.join(domains)}, got "
                         f"{d!r}; check the stage order (adc after exposure, black_level after "
                         f"adc, tonemap after black_level)")


def _accepted(fn, kw):
    """Keep only the keyword arguments ``fn`` takes (so stale params of another method are ignored)."""
    ps = inspect.signature(fn).parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in ps.values()):
        return dict(kw)
    return {k: v for k, v in kw.items() if k in ps}


def bin_bayer(mosaic, pattern="RGGB"):
    """Half-resolution RGB from a mosaic: each 2x2 tile becomes one pixel (G = mean of the two
    greens).  Cheap and demosaic-free; used by white balance to estimate the illuminant."""
    m = _f32(mosaic)
    h, w = m.shape[0] // 2 * 2, m.shape[1] // 2 * 2
    out = np.zeros((h // 2, w // 2, 3), np.float32)
    cnt = np.zeros(3)
    for k, ch in enumerate(_cfa._check(pattern)):
        c = "RGB".index(ch)
        i, j = divmod(k, 2)
        out[..., c] += m[i:h:2, j:w:2]
        cnt[c] += 1
    return out / cnt.astype(np.float32)


def scene_stage(image, meta):
    """Ground truth: linear RGB scene, passed through unchanged (identity)."""
    a = _f32(image)
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError("stage 'scene' needs an (H, W, 3) linear RGB image")
    m = dict(meta)
    m.update(domain="scene", scene_scale=float(m.get("scene_scale", 1.0)))
    return a, m


def illuminant_stage(image, meta, cct=3200.0):
    """Light the scene with a black-body source of ``cct`` kelvin (colour cast, L4 s9).

    out = scene * e / e_G  (green kept, red up, blue down for warm light), so the cast is a
    pure diagonal transform that ideal white balance inverts exactly.  ``cct=None`` keeps
    the neutral (D65-like) light.  Red may exceed 1 for a very warm source.
    """
    _need("illuminant", meta, "scene")
    m = dict(meta)
    a = _f32(image)
    if cct is None:
        return a, m
    cast = 1.0 / _wb.gains_for_illuminant(cct)
    if a.ndim == 3:
        out = _wb.apply_gains(a, cast)
    else:
        out = _wb.apply_gains_mosaic(a, cast, m["cfa"])
    m["illuminant"] = float(cct) if np.ndim(cct) == 0 else [float(v) for v in cct]
    return out, m


def cfa_stage(image, meta, pattern="RGGB"):
    """Sample the scene through a colour filter array: RGB -> one number per pixel (L3 s13)."""
    a = _f32(image)
    m = dict(meta)
    if a.ndim == 2:
        if "cfa" not in m:
            raise ValueError("stage 'cfa' got a 2-D image that is not a mosaic")
        return a, m                                       # already sampled: nothing to do
    if a.shape[0] % 2 or a.shape[1] % 2:
        raise ValueError("stage 'cfa' needs even image dimensions")
    m["cfa"] = pattern
    return _cfa.mosaic(a, pattern), m


def _sensor_from_meta(meta, who):
    if "sensor_params" not in meta:
        raise ValueError(f"stage '{who}' needs the sensor description set by the 'exposure' stage")
    return _sensor.SensorParams(**meta["sensor_params"])


def exposure_stage(image, meta, sensor="dslr", exposure=0.0125, flux_at_white=_sensor.DEFAULT_FLUX,
                   seed=0, sensor_kw=None, oracle=False):
    """Scene light -> electrons: photons x QE x t, shot noise, dark current, hot pixels,
    PRNU, full-well clipping (L2 s34, ``sensor.expose``).

    sensor     preset name in ``sensor.PRESETS`` (or a ``SensorParams``)
    exposure   seconds;  flux_at_white  photons / pixel / s delivered by scene value 1.0
    sensor_kw  overrides of ``SensorParams`` fields, e.g. ``{'hot_pixel_rate': 8000.0}``
    oracle     (internal, used by the metrics) noise-free expected value, no defects
    """
    _need("exposure", meta, "scene")
    p = sensor if isinstance(sensor, _sensor.SensorParams) else _sensor.PRESETS[sensor]
    if sensor_kw:
        p = replace(p, **sensor_kw)
    a = _f32(image)
    if oracle:
        q = _sensor.clean_params(p)
        e = np.minimum(np.maximum(a, 0) * flux_at_white * q.qe * exposure
                       + q.dark_current * exposure, q.full_well).astype(np.float32)
    else:
        e = _sensor.expose(a, exposure, p, flux_at_white=flux_at_white, seed=seed)
    m = dict(meta)
    m.update(domain="electrons", sensor=p.name, sensor_params=asdict(p), exposure=float(exposure),
             flux_at_white=float(flux_at_white), full_well=float(p.full_well),
             scene_scale=float(m.get("scene_scale", 1.0)) * flux_at_white * p.qe * exposure / p.full_well)
    return e, m


def adc_stage(image, meta, iso=3200, bits=None, seed=1, oracle=False):
    """Electrons -> RAW digital numbers: read noise, ISO gain, black level, rounding (L2 s34, s39).

    ``bits`` overrides the preset's ADC depth; the black level is rescaled with it and the
    gain still maps full well (ISO 100) to the white level.  ``oracle`` is internal.
    """
    _need("adc", meta, "electrons")
    p = _sensor_from_meta(meta, "adc")
    if bits is not None and int(bits) != p.bits:
        p = replace(p, black_level=float(round(p.black_level * 2.0 ** (int(bits) - p.bits))), bits=int(bits))
    e = _f32(image)
    if oracle:
        dn = np.clip(p.gain(iso) * e.astype(np.float64) + p.black_level, 0, p.white_level).astype(np.float32)
    else:
        dn = _sensor.adc(e, p, iso=iso, seed=seed)
    m = dict(meta)
    m.update(domain="dn", bits=p.bits, black_level=p.black_level, white_level=p.white_level,
             iso=float(iso), sensor_params=asdict(p),
             scene_scale=float(m["scene_scale"]) * p.gain(iso) * p.full_well / (p.white_level - p.black_level))
    return dn, m


def black_level_stage(image, meta, dark_frame=False, digital_gain=1.0, oracle=False):
    """Subtract the black level and normalise to linear [0, 1] (``sensor.normalize``).

    dark_frame    also subtract a master dark frame (mean of 16 lens-cap exposures of the same
                  shutter / ISO), which removes hot pixels and dark-current pattern (L2 s34)
    digital_gain  exposure compensation applied after normalisation (result clipped to [0, 1])
    """
    _need("black_level", meta, "dn")
    dn = _f32(image)
    m = dict(meta)
    if dark_frame and not oracle:
        p = _sensor_from_meta(meta, "black_level")
        dark = _sensor.master_dark(dn.shape, m["exposure"], m["iso"], p, n=16)
        dn = _sensor.subtract_dark(dn, dark, m["black_level"])
    out = np.clip(_sensor.normalize(dn, m, clip=False) * digital_gain, 0.0, 1.0).astype(np.float32)
    m.update(domain="linear", scene_scale=float(m.get("scene_scale", 1.0)) * float(digital_gain))
    return out, m


def hot_pixels_stage(image, meta, threshold=None):
    """Replace isolated hot pixels by the median of their same-colour neighbours (``sensor.fix_hot_pixels``).

    ``threshold`` is in image units; ``None`` = 8 x the robust spread of the residual."""
    return _sensor.fix_hot_pixels(_f32(image), meta.get("cfa", "RGGB"), threshold), dict(meta)


WB_METHODS = ("none", "true", "preset", "camera", "gray_world", "white_patch", "patch")


def _gains_from(method, rgb, meta, cct, box, percentile, half_res):
    if method == "none":
        return np.ones(3, np.float32)
    if method == "true":
        return (_wb.gains_for_illuminant(meta["illuminant"]) if "illuminant" in meta
                else np.ones(3, np.float32))            # neutral light: nothing to correct
    if method == "preset":
        if cct is None:
            raise ValueError("stage 'white_balance': method 'preset' needs cct=<kelvin>")
        return _wb.gains_for_illuminant(cct)
    if method == "camera":
        if "camera_wb" not in meta:
            raise ValueError("stage 'white_balance': method 'camera' needs meta['camera_wb'] (load_raw sets it)")
        g = np.asarray(meta["camera_wb"], np.float32)
        return g / g[1]
    if method == "gray_world":
        return _wb.gray_world(rgb)
    if method == "white_patch":
        return _wb.white_patch(rgb, percentile)
    if method == "patch":
        if box is None:
            raise ValueError("stage 'white_balance': method 'patch' needs box=(r0, r1, c0, c1) of a neutral patch")
        r0, r1, c0, c1 = box
        if half_res:
            box = (r0 // 2, max(r1 // 2, r0 // 2 + 1), c0 // 2, max(c1 // 2, c0 // 2 + 1))
        return _wb.from_patch(rgb, box)
    raise ValueError(f"unknown white-balance method {method!r}; choose from {WB_METHODS}")


def white_balance_stage(image, meta, method="gray_world", cct=None, box=None, percentile=99):
    """Estimate gains and apply them (L4 s9).  On a mosaic the gains are estimated from a
    half-resolution RGB (each 2x2 tile binned) and applied per CFA site, before demosaicing;
    on an RGB image they are estimated and applied directly.

    method  'none' | 'true' (oracle: the known illuminant) | 'preset' (+ ``cct`` kelvin) |
            'camera' (``meta['camera_wb']`` of a RAW file) | 'gray_world' | 'white_patch' |
            'patch' (+ ``box`` of a neutral patch, full-resolution pixel coordinates)
    Gains are normalised to G = 1 and the result is not clipped.  Stores ``meta['wb_gains']`` and,
    if the illuminant is known, ``meta['wb_angular_error']`` (degrees between the estimated and
    true illuminant colour).
    """
    _need("white_balance", meta, "linear", "scene")
    a = _f32(image)
    m = dict(meta)
    mosaic = a.ndim == 2
    rgb = bin_bayer(a, m["cfa"]) if mosaic else a
    g = np.asarray(_gains_from(method, rgb, m, cct, box, percentile, mosaic), np.float32)
    out = _wb.apply_gains_mosaic(a, g, m["cfa"]) if mosaic else _wb.apply_gains(a, g)
    m["wb_gains"] = [float(v) for v in g]
    m["wb_method"] = method
    m["scene_scale"] = float(m.get("scene_scale", 1.0)) * float(g[1])
    if "illuminant" in m:
        truth = _wb.gains_to_illuminant(_wb.gains_for_illuminant(m["illuminant"]))
        m["wb_angular_error"] = _wb.angular_error(_wb.gains_to_illuminant(g), truth)
    return out, m


def demosaic_stage(image, meta, method="malvar", **kw):
    """Mosaic -> RGB (``demosaic.demosaic``).  An image that is already RGB passes through."""
    a = _f32(image)
    m = dict(meta)
    if a.ndim == 3:
        return a, m
    out = _demosaic.demosaic(a, m.pop("cfa", "RGGB"), method, **kw)
    return out.astype(np.float32), m


def _denoise_planes(a, method, kw):
    """Denoise a mosaic one CFA site plane at a time (the 4 half-resolution sub-images)."""
    out = np.empty_like(a)
    for i in range(2):
        for j in range(2):
            out[i::2, j::2] = _denoise.denoise(np.ascontiguousarray(a[i::2, j::2]), method, **kw)
    return out


def denoise_stage(image, meta, method="bilateral", **kw):
    """Noise reduction (``denoise.denoise``, L5 s10).  On a mosaic each CFA site plane is filtered
    separately (so colours are never mixed); on RGB the method sees the whole image.
    Keyword arguments that the chosen method does not take are ignored."""
    a = _f32(image)
    fn = getattr(_denoise, method, None)
    if method not in _denoise.METHODS or fn is None:
        raise ValueError(f"unknown denoise method {method!r}; choose from {_denoise.METHODS}")
    kw = _accepted(fn, kw)
    if a.ndim == 2:
        if method == "luma_chroma":
            raise ValueError("stage 'denoise': 'luma_chroma' needs an RGB image (put denoise after demosaic)")
        return _denoise_planes(a, method, kw), dict(meta)
    return _denoise.denoise(a, method, **kw), dict(meta)


def tonemap_stage(image, meta, method="srgb", **kw):
    """Linear -> display-referred (``tonemap.tonemap``, L4 s14).  'srgb' is the camera default curve."""
    _need("tonemap", meta, "linear", "scene")
    a = _f32(image)
    if a.ndim != 3:
        raise ValueError("stage 'tonemap' needs an RGB image (put it after demosaic)")
    m = dict(meta)
    m.update(domain="display", tonemap=method)
    return _tonemap.tonemap(a, method, **kw), m


def sharpen_stage(image, meta, method="unsharp", lam=0.5, **kw):
    """Sharpening S = x + lam F(x) (``sharpen.sharpen``, L5 s15).  Best applied to display data."""
    a = _f32(image)
    if a.ndim != 3:
        raise ValueError("stage 'sharpen' needs an RGB image (put it after demosaic)")
    if method == "laplacian":
        kw = {k: v for k, v in kw.items() if k != "sigma"}
    return _sharpen.sharpen(a, method, lam, **kw), dict(meta)


def _unit_view(image, meta):
    """Image as [0, 1] floats for storage (non-display data is stored as it is, clipped)."""
    a = _f32(image)
    d = meta.get("domain")
    if d == "electrons":
        a = a / float(meta.get("full_well", max(float(a.max()), 1.0)))
    elif d == "dn":
        a = (a - meta.get("black_level", 0.0)) / (meta.get("white_level", 255.0) - meta.get("black_level", 0.0))
    return np.clip(a, 0.0, 1.0)


def jpeg_stage(image, meta, quality=85, subsampling="4:2:0"):
    """Store as JPEG (own codec, ``jpeg.roundtrip``): output is the decoded file, size in
    ``meta['nbytes']`` / ``['bpp']``.  Non-RGB / non-display input is stored as it is (a mosaic
    becomes a grey image): that is rarely useful, it lets 'processing off' experiments run."""
    a = _unit_view(image, meta)
    m = dict(meta)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
        m.pop("cfa", None)
    out, info = _jpeg.roundtrip(a, int(quality), subsampling)
    m.update(domain="bytes", nbytes=int(info["nbytes"]), bpp=float(info["bpp"]),
             quality=int(quality), subsampling=subsampling, bits=8)
    return out.astype(np.float32), m


# ==========================================================================
# Stage catalogue (drives the diagram, README and the app)
# ==========================================================================
STAGE_ORDER = ("scene", "illuminant", "cfa", "exposure", "adc", "black_level", "hot_pixels",
               "white_balance", "demosaic", "denoise", "tonemap", "sharpen", "jpeg")

STAGE_FUNCS = {"scene": scene_stage, "illuminant": illuminant_stage, "cfa": cfa_stage,
               "exposure": exposure_stage, "adc": adc_stage, "black_level": black_level_stage,
               "hot_pixels": hot_pixels_stage, "white_balance": white_balance_stage,
               "demosaic": demosaic_stage, "denoise": denoise_stage, "tonemap": tonemap_stage,
               "sharpen": sharpen_stage, "jpeg": jpeg_stage}

# Default parameters.  Chosen so that the RAW is clearly noisy under tungsten light (docs/00).
DEFAULT_PARAMS = {
    "scene": {},
    "illuminant": {"cct": 3200.0},
    "cfa": {"pattern": "RGGB"},
    "exposure": {"sensor": "dslr", "exposure": 0.0125, "flux_at_white": _sensor.DEFAULT_FLUX, "seed": 0,
                 "sensor_kw": {"hot_pixel_fraction": 5e-4, "hot_pixel_rate": 40000.0}},
    "adc": {"iso": 3200, "bits": None, "seed": 1},
    "black_level": {"dark_frame": False, "digital_gain": 1.0},
    "hot_pixels": {"threshold": 0.3},
    "white_balance": {"method": "white_patch"},
    "demosaic": {"method": "median"},
    "denoise": {"method": "bilateral", "sigma_d": 1.5, "sigma_r": 0.05},
    "tonemap": {"method": "srgb"},
    "sharpen": {"method": "unsharp", "lam": 0.5, "sigma": 1.0},
    "jpeg": {"quality": 85, "subsampling": "4:2:0"},
}

GROUP_INFO = {
    "scene": {"label": "Scene", "colour": "#999999"},
    "sensor": {"label": "Sensor", "colour": "#E69F00"},
    "adc": {"label": "ADC", "colour": "#56B4E9"},
    "processing": {"label": "Processing", "colour": "#009E73"},
    "storage": {"label": "Storage", "colour": "#CC79A7"},
}

_G = {"scene": "scene", "illuminant": "scene", "cfa": "sensor", "exposure": "sensor", "adc": "adc",
      "black_level": "processing", "hot_pixels": "processing", "white_balance": "processing",
      "demosaic": "processing", "denoise": "processing", "tonemap": "processing",
      "sharpen": "processing", "jpeg": "storage"}

STAGE_INFO = {
    "scene": dict(
        label="Scene", group="scene", short="Ground-truth linear RGB (identity); the reference for metrics",
        description="The light the camera would ideally record, as linear RGB in [0, 1] (1.0 = scene "
                    "white). Nothing happens here; the stage only records the ground truth that every "
                    "later stage is measured against.",
        slides="L2 s3", doc="docs/00_camera_pipeline.md", params_help={}),
    "illuminant": dict(
        label="Light source", group="scene", short="Colour cast of the light source (black-body, kelvin)",
        description="Relights the scene with a black-body source: warm light boosts red and starves blue "
                    "(green is kept). It is a per-channel multiplication, so ideal white balance can undo "
                    "it exactly; whether the camera knows the light is what the white-balance stage is about.",
        slides="L4 s9", doc="docs/03_white_balance.md",
        params_help={"cct": "colour temperature in kelvin (3200 tungsten, 6500 daylight, None = neutral)"}),
    "cfa": dict(
        label="Colour filter array", group="sensor", short="Each pixel keeps one of R, G, B (Bayer mosaic)",
        description="Every photosite sits behind a red, green or blue filter, so two thirds of the colour "
                    "values are never measured. The image becomes a single (H, W) plane in which the three "
                    "colours are interleaved.",
        slides="L3 s13", doc="docs/02_color_sensing_demosaicing.md",
        params_help={"pattern": "Bayer layout of the 2x2 tile: RGGB, BGGR, GRBG or GBRG"}),
    "exposure": dict(
        label="Exposure", group="sensor", short="Photons x QE x t, shot noise, dark current, defects -> electrons",
        description="Light becomes charge: mean electrons = photons x quantum efficiency x exposure time, "
                    "plus dark current and hot pixels, with Poisson shot noise and a per-pixel gain error "
                    "(PRNU). Charge saturates at the full-well capacity.",
        slides="L2 s34", doc="docs/01_sensor_and_noise.md",
        params_help={"sensor": "preset: dslr, phone, retiga_4000r, aptina_mt9p031",
                     "exposure": "shutter time in seconds",
                     "flux_at_white": "photons per pixel per second delivered by scene white",
                     "seed": "random seed of the shot noise",
                     "sensor_kw": "overrides of sensor fields, e.g. {'read_noise': 10}"}),
    "adc": dict(
        label="ADC", group="adc", short="Read noise, ISO gain, black level, quantisation -> RAW numbers",
        description="The analog readout adds read noise, the ISO gain amplifies the signal before "
                    "quantisation, a black-level offset is added and the value is rounded to an integer "
                    "of the chosen bit depth. Raising ISO brightens the picture but not the signal-to-noise ratio.",
        slides="L2 s34, s39", doc="docs/01_sensor_and_noise.md",
        params_help={"iso": "gain relative to ISO 100", "bits": "ADC bit depth (None = sensor preset)",
                     "seed": "random seed of the read noise"}),
    "black_level": dict(
        label="Black level", group="processing", short="Subtract black level, normalise RAW to linear [0, 1]",
        description="Turns RAW numbers into linear light: (DN - black) / (white - black). Optionally subtracts "
                    "a master dark frame, which removes hot pixels and fixed dark-current patterns, and "
                    "applies a digital exposure compensation.",
        slides="L2 s34", doc="docs/01_sensor_and_noise.md",
        params_help={"dark_frame": "also subtract a master dark frame (16 averaged lens-cap shots)",
                     "digital_gain": "exposure compensation after normalisation (clipped to 1)"}),
    "hot_pixels": dict(
        label="Hot pixels", group="processing", short="Replace outlier pixels by same-colour neighbours",
        description="A pixel that is far brighter than the median of its eight same-colour neighbours is "
                    "replaced by that median. On a mosaic the neighbours sit two pixels away, so the "
                    "filter never mixes colours.",
        slides="L2 s34", doc="docs/01_sensor_and_noise.md",
        params_help={"threshold": "excess over the median that counts as hot (None = automatic)"}),
    "white_balance": dict(
        label="White balance", group="processing", short="Per-channel gains that make the light source neutral",
        description="Estimates the colour of the light and multiplies R, G and B so that neutral objects "
                    "become neutral. Here it runs on the RAW mosaic, before demosaicing, with gains "
                    "estimated from a half-resolution RGB made by binning each 2x2 tile.",
        slides="L4 s9", doc="docs/03_white_balance.md",
        params_help={"method": "none, true (oracle), preset, camera, gray_world, white_patch, patch",
                     "cct": "kelvin, for method 'preset'", "box": "(r0, r1, c0, c1) neutral patch, for 'patch'",
                     "percentile": "brightness percentile for white_patch"}),
    "demosaic": dict(
        label="Demosaic", group="processing", short="Interpolate the two missing colours at every pixel",
        description="Rebuilds a full RGB image from the mosaic. Two thirds of the output numbers are "
                    "interpolated, so edges can show colour fringes and zipper patterns; the methods "
                    "differ in how well they avoid them.",
        slides="L3 s13", doc="docs/02_color_sensing_demosaicing.md",
        params_help={"method": "nearest, bilinear, bicubic, median (Freeman) or malvar"}),
    "denoise": dict(
        label="Denoise", group="processing", short="Smooth noise but keep edges (bilateral filter by default)",
        description="Averages neighbouring pixels that look alike, so flat areas lose their noise while "
                    "edges survive. It works on linear RGB; if it runs on a mosaic it filters each colour "
                    "site plane separately.",
        slides="L5 s10", doc="docs/05_denoising.md",
        params_help={"method": "gaussian, bilateral, median, luma_chroma or nlm",
                     "sigma_d": "bilateral: spatial sigma in pixels",
                     "sigma_r": "bilateral: intensity sigma (linear units)"}),
    "tonemap": dict(
        label="Tone map", group="processing", short="Linear light -> display values (sRGB curve by default)",
        description="Maps linear scene values to display-referred values: the sRGB curve, a gamma, "
                    "histogram equalisation or a local operator. After this point numbers are "
                    "gamma-encoded, not proportional to light.",
        slides="L4 s14", doc="docs/04_tone_mapping.md",
        params_help={"method": "srgb, linear, gamma, hist, clahe or local"}),
    "sharpen": dict(
        label="Sharpen", group="processing", short="Unsharp masking: add back a high-pass copy",
        description="S = x + lambda F(x), where F is a high-pass version of the image. It restores the "
                    "crispness that demosaicing and denoising took away, but also amplifies the remaining "
                    "noise and can create halos at strong edges.",
        slides="L5 s15", doc="docs/06_sharpening.md",
        params_help={"method": "unsharp (Gaussian) or laplacian", "lam": "strength lambda",
                     "sigma": "unsharp: Gaussian sigma in pixels"}),
    "jpeg": dict(
        label="JPEG", group="storage", short="8x8 DCT, quantisation and Huffman sizing (own codec)",
        description="Lossy storage: YCbCr conversion, optional chroma subsampling, 8x8 DCT, quantisation "
                    "with the standard tables scaled by the quality, then entropy coding. The output is the "
                    "decoded file and its size in bytes; noisy input costs many more bytes.",
        slides="L2 s15", doc="docs/07_compression.md",
        params_help={"quality": "1-100 (higher = bigger file, fewer artefacts)",
                     "subsampling": "chroma subsampling: 4:4:4, 4:2:2 or 4:2:0"}),
}
for _k, _v in STAGE_INFO.items():
    assert _v["group"] == _G[_k] and len(_v["short"]) <= 70, _k


def default_pipeline(**overrides) -> Pipeline:
    """The default camera: 13 stages in the order of ``STAGE_ORDER`` (boxes of L2 s3).

    Overrides are ``stage_id={'param': value, ...}`` (merged into that stage's defaults; if
    ``method`` changes the other default parameters of that stage are dropped) and
    ``enabled={'denoise': False}``.  Example::

        default_pipeline(adc={'iso': 6400}, white_balance={'method': 'true'},
                         enabled={'jpeg': False})
    """
    pl = Pipeline([Stage(n, STAGE_FUNCS[n], copy.deepcopy(DEFAULT_PARAMS[n]), True, _G[n],
                         STAGE_INFO[n]["label"]) for n in STAGE_ORDER])
    return apply_overrides(pl, overrides)


def apply_overrides(pipeline: Pipeline, overrides: dict) -> Pipeline:
    """Apply ``stage_id={...}`` / ``enabled={...}`` overrides to ``pipeline`` in place."""
    for key, val in overrides.items():
        if key == "enabled":
            for name, flag in val.items():
                pipeline.enable(name, flag)
            continue
        if key not in pipeline.names:
            raise ValueError(f"unknown stage {key!r}; stages are {pipeline.names}")
        st = pipeline[key]
        val = dict(val)
        if "method" in val and val["method"] != st.params.get("method"):
            st.params = {}
        st.params.update(val)
    return pipeline


# ==========================================================================
# Metrics against the ground truth
# ==========================================================================
_ORACLE_SKIP = ("hot_pixels", "denoise")          # nothing to remove from the noise-free oracle
_ORACLE_AWARE = ("exposure", "adc", "black_level")           # stages that take oracle=True


def _unit(image, meta, egain=1.0):
    """Stage output in linear units where the ground truth lives: electrons x egain / full well
    (``egain`` = ISO/100 of the later ADC, so that the range is comparable with the DN stages),
    (DN - black) / (white - black), everything else as it is (not clipped)."""
    a = np.asarray(image, np.float64)
    d = meta.get("domain")
    if d == "electrons":
        return a / meta.get("full_well", 1.0) * egain
    if d == "dn":
        return (a - meta["black_level"]) / (meta["white_level"] - meta["black_level"])
    return a


def _mean_rgb(image, meta, egain=1.0):
    a = _unit(image, meta, egain)
    if a.ndim == 2:
        a = bin_bayer(a, meta.get("cfa", "RGGB"))
    if meta.get("domain") in ("display", "bytes"):
        a = utils.srgb_to_linear(np.clip(a, 0, 1))
    return a.reshape(-1, 3).mean(0)


def _noise_sigma(image, meta, egain=1.0):
    a = _unit(image, meta, egain).astype(np.float32)
    if a.ndim == 2:
        return float(np.mean([_denoise.estimate_noise(np.ascontiguousarray(a[i::2, j::2]))
                              for i in range(2) for j in range(2)]))
    return float(_denoise.estimate_noise(a))


class StageMetrics:
    """Callable ``metrics(image, meta, stage_name) -> dict`` for ``Pipeline.run``.

    Reference (target) of a stage output, by domain; ``truth`` is the ground-truth scene:

    * mosaic stages (electrons / dn / linear with ``meta['cfa']``): the truth sampled through
      the same CFA, ``clip(cfa.mosaic(truth) * scene_scale, 0, 1)``;
    * linear RGB stages: ``clip(truth * scene_scale, 0, 1)``;
    * display / bytes stages: the same scaled truth through the same tone-mapping stage;
    * the ``scene`` / ``illuminant`` / ``cfa`` stages: the unlit truth itself (the cast is the error).

    ``scene_scale`` (in ``meta``) is the brightness change caused by exposure, ISO and digital
    gain, so brightness choices are not counted as errors; only colour, noise and detail are.

    Metrics returned: ``psnr`` (dB vs the reference), ``ssim``, ``sigma`` (estimated noise std in
    the stage's unit space, ``denoise.estimate_noise``), ``cast_deg`` (angle between the mean
    colour and the truth's mean colour), ``wb_angular_error`` (white-balance stage, if the
    illuminant is known) and, for stages that still hold a mosaic past the sensor,
    ``psnr_noise``: PSNR against the *noise-free expected value of the same stage*, from an
    oracle run of the same pipeline with a noise-free, defect-free sensor (isolates noise from
    colour cast).
    """

    def __init__(self, scene, pipeline: Pipeline | None = None):
        self.scene = _f32(scene)
        tm = {"method": "srgb"}
        order, stages = [], []
        if pipeline is not None:
            order = [s.name for s in pipeline.stages]
            stages = pipeline.stages
            if "tonemap" in order and pipeline["tonemap"].enabled:
                tm = dict(pipeline["tonemap"].params)
        self.tonemap_params = tm
        self.egain = 1.0
        if "adc" in order and pipeline["adc"].enabled:
            self.egain = float(pipeline["adc"].params.get("iso", 100.0)) / 100.0
        self.oracle = self._run_oracle(stages)
        cut = order.index("demosaic") if "demosaic" in order else len(order)
        sig = [(s.name, s.enabled, repr(sorted(s.params.items(), key=str)))
               for s in stages[:cut] if s.name not in _ORACLE_SKIP]
        self.key = (hash(self.scene.tobytes()), self.scene.shape, tuple(order), tuple(sig),
                    repr(sorted(tm.items(), key=str)))

    def _run_oracle(self, stages):
        """Noise-free, defect-free run of the mosaic part of the pipeline: name -> (image, meta)."""
        out, img, meta = {}, self.scene, {"domain": "scene"}
        for s in stages:
            if s.enabled and s.name not in _ORACLE_SKIP:
                kw = dict(s.params)
                if s.name in _ORACLE_AWARE:
                    kw["oracle"] = True
                try:
                    img, meta = s.fn(img, dict(meta), **kw)
                except (ValueError, TypeError):
                    break
            if img.ndim == 3 and meta.get("domain") != "scene":
                break                                   # past the mosaic: references are the truth
            out[s.name] = (img, dict(meta))
        return out

    def reference(self, image, meta):
        """Ground-truth target of an output (see the class docstring)."""
        d, scale = meta.get("domain"), float(meta.get("scene_scale", 1.0))
        truth = self.scene
        mosaic = np.ndim(image) == 2
        if d == "scene":
            return _cfa.mosaic(truth, meta["cfa"]).astype(np.float64) if mosaic else truth
        if mosaic:
            truth = _cfa.mosaic(truth, meta.get("cfa", "RGGB"))
        t = np.clip(truth * scale * (self.egain if d == "electrons" else 1.0), 0, 1)
        if d in ("display", "bytes"):
            return _tonemap.tonemap(t, **self.tonemap_params) if t.ndim == 3 else utils.linear_to_srgb(t)
        return t

    def __call__(self, image, meta, name):
        a = np.asarray(image)
        ref = self.reference(a, meta)
        if ref.shape != a.shape:
            return {}
        out = {}
        u = _unit(a, meta, self.egain)
        out["psnr"] = utils.psnr(ref, u)
        if name != "scene":
            out["ssim"] = utils.ssim(np.clip(ref, 0, 1), np.clip(u, 0, 1))
        out["sigma"] = _noise_sigma(a, meta, self.egain)
        try:
            out["cast_deg"] = _wb.angular_error(_mean_rgb(a, meta, self.egain), self.scene.reshape(-1, 3).mean(0))
        except Exception:
            pass
        if "wb_angular_error" in meta:
            out["wb_angular_error"] = meta["wb_angular_error"]
        if a.ndim == 2 and meta.get("domain") != "scene" and name in self.oracle:
            o_img, o_meta = self.oracle[name]
            if o_img.shape == a.shape:
                out["psnr_noise"] = utils.psnr(_unit(o_img, o_meta, self.egain), u)
        return out


def stage_metrics(scene, pipeline: Pipeline | None = None) -> StageMetrics:
    """Build the per-stage metrics callable for ``Pipeline.run(..., metrics=...)``."""
    return StageMetrics(scene, pipeline)


# ==========================================================================
# Convenience runners
# ==========================================================================
def simulate(scene="astronaut", size=256, pipeline: Pipeline | None = None, previous=None,
             with_metrics=True, **overrides) -> PipelineResult:
    """Load a scene (name from ``isp.scenes.NAMES`` or a linear RGB array), run the camera and
    return a :class:`PipelineResult` with per-stage metrics against the ground truth.

    ``pipeline`` defaults to ``default_pipeline()``; ``overrides`` (``stage_id={...}``,
    ``enabled={...}``) are applied to a copy of it.  Pass an earlier result as ``previous`` to
    recompute only what changed.
    """
    img = _scenes.load(scene, size) if isinstance(scene, str) else _f32(scene)
    pl = default_pipeline() if pipeline is None else pipeline.copy()
    apply_overrides(pl, overrides)
    met = StageMetrics(img, pl) if with_metrics else None
    return pl.run(img, {"domain": "scene"}, metrics=met, previous=previous)


def process_raw(mosaic, meta, pipeline: Pipeline | None = None, **overrides) -> PipelineResult:
    """Run the processing half of the camera on a real RAW mosaic (see ``isp.io.load_raw``):
    the pipeline starts at ``'black_level'`` and there is no ground truth, so no metrics."""
    pl = default_pipeline() if pipeline is None else pipeline.copy()
    apply_overrides(pl, overrides)
    return pl.run(np.asarray(mosaic, np.float32), dict(meta), start="black_level")
