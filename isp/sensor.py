"""Image sensor and noise model (lecture 2, "Sensors and Noise").

Block diagram of L2 s34::

    radiance -> (+ dark current) x exposure -> photon noise -> saturation (full well)
             -> + read noise -> gain (ISO) -> ADC quantisation -> digital number

``expose`` covers everything up to the full well and returns electrons; ``adc`` adds
read noise, gain, black level and quantisation and returns digital numbers (DN).
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from scipy.ndimage import median_filter

from .utils import rng


# --------------------------------------------------------------------------
# parameters
# --------------------------------------------------------------------------
@dataclass
class SensorParams:
    """Physical description of a sensor.

    qe                  quantum efficiency (electrons per photon)
    full_well           saturation charge (electrons)
    dark_current        mean dark current D (e-/pixel/s)
    read_noise          read noise N_r (e- rms)
    bits                ADC bit depth
    black_level         DN that a zero-electron pixel maps to
    base_gain           DN per electron at ISO 100; ``None`` maps full well to white level
    prnu                gain fixed pattern: std of the multiplicative pixel gain (fraction)
    dsnu                dark-current fixed pattern: std (e-/s)
    hot_pixel_fraction  fraction of pixels that are hot
    hot_pixel_rate      extra dark current of a hot pixel (e-/s)
    name                label
    sensor_seed         seed of the fixed-pattern maps (the "silicon"); not the per-shot seed
    """
    qe: float = 0.5
    full_well: float = 20000.0
    dark_current: float = 1.0
    read_noise: float = 5.0
    bits: int = 12
    black_level: float = 64.0
    base_gain: float | None = None
    prnu: float = 0.0
    dsnu: float = 0.0
    hot_pixel_fraction: float = 0.0
    hot_pixel_rate: float = 0.0
    name: str = "custom"
    sensor_seed: int = 1234

    @property
    def white_level(self) -> float:
        return float(2 ** self.bits - 1)

    def gain(self, iso: float = 100.0) -> float:
        """DN per electron at the given ISO: base_gain * iso / 100."""
        g0 = self.base_gain if self.base_gain is not None else \
            (self.white_level - self.black_level) / self.full_well
        return float(g0 * iso / 100.0)


PRESETS = {
    # numbers of L2 s31 / s33
    "retiga_4000r": SensorParams(qe=0.55, full_well=40000, dark_current=1.64, read_noise=12.0,
                                 bits=12, black_level=64, prnu=0.005, dsnu=0.2,
                                 hot_pixel_fraction=1e-4, hot_pixel_rate=30.0, name="Retiga 4000R"),
    "aptina_mt9p031": SensorParams(qe=0.69, full_well=8500, dark_current=25.0, read_noise=2.6,
                                   bits=12, black_level=64, prnu=0.01, dsnu=4.0,
                                   hot_pixel_fraction=5e-4, hot_pixel_rate=300.0, name="Aptina MT9P031"),
    "dslr": SensorParams(qe=0.40, full_well=30000, dark_current=0.5, read_noise=5.0,
                         bits=14, black_level=512, prnu=0.005, dsnu=0.1,
                         hot_pixel_fraction=2e-4, hot_pixel_rate=20.0, name="DSLR"),
    "phone": SensorParams(qe=0.60, full_well=5000, dark_current=5.0, read_noise=2.5,
                          bits=10, black_level=64, prnu=0.01, dsnu=1.0,
                          hot_pixel_fraction=5e-4, hot_pixel_rate=100.0, name="Phone"),
}

DEFAULT_FLUX = 1.0e5      # photons / pixel / s that scene white (value 1.0) delivers


# --------------------------------------------------------------------------
# fixed pattern
# --------------------------------------------------------------------------
def fixed_pattern(shape, params: SensorParams) -> dict:
    """The sensor's fixed-pattern maps for an array ``shape``.

    Returns dict(prnu, dark, hot): ``prnu`` is the per-pixel gain error (fractional),
    ``dark`` the per-pixel dark current in e-/s (mean + DSNU + hot pixels, >= 0) and
    ``hot`` the boolean hot-pixel map.  Derived only from ``params.sensor_seed`` and
    ``shape``: identical on every call, so a dark frame measured once describes every
    later exposure (L2 s43).
    """
    shape = tuple(int(s) for s in shape)
    r = np.random.default_rng(np.random.SeedSequence([int(params.sensor_seed), *shape]))
    prnu = r.standard_normal(shape) * params.prnu          # always draw: maps independent
    dsnu = r.standard_normal(shape) * params.dsnu
    hot = r.random(shape) < params.hot_pixel_fraction
    dark = np.maximum(params.dark_current + dsnu + hot * params.hot_pixel_rate, 0.0)
    return dict(prnu=prnu, dark=dark, hot=hot)


# --------------------------------------------------------------------------
# forward model
# --------------------------------------------------------------------------
def expose(image, exposure, params: SensorParams, flux_at_white=DEFAULT_FLUX, seed=0):
    """Scene -> electrons: photon + dark charge, Poisson noise, full-well clipping.

    mean e- = P QE t (1 + PRNU) + (D + DSNU + hot) t,   P = image * flux_at_white
    e- ~ Poisson(mean), then clipped at ``full_well``.  Works on (H, W) or (H, W, 3).

    Parameters: image in [0, 1] (linear), exposure in s, flux_at_white in photons/pixel/s,
    seed for the per-shot noise (fixed pattern does not depend on it).
    Returns float32 electrons.   Reference: L2 s34, s45.
    """
    img = np.clip(np.asarray(image, dtype=np.float64), 0.0, None)
    fp = fixed_pattern(img.shape, params)
    photons = img * flux_at_white
    mean = photons * params.qe * exposure * (1.0 + fp["prnu"]) + fp["dark"] * exposure
    e = rng(seed).poisson(np.maximum(mean, 0.0)).astype(np.float64)
    return np.minimum(e, params.full_well).astype(np.float32)


def adc(electrons, params: SensorParams, iso=100, seed=0):
    """Electrons -> digital numbers: + read noise, gain, black level, rounding, clipping.

    DN = clip(round(gain (e + N(0, N_r^2)) + black), 0, 2^bits - 1),  gain = base_gain iso/100.
    Returns float32 holding integers.   Reference: L2 s34, s40.
    """
    e = np.asarray(electrons, dtype=np.float64)
    e = e + rng(seed).standard_normal(e.shape) * params.read_noise
    dn = np.round(params.gain(iso) * e + params.black_level)
    return np.clip(dn, 0, params.white_level).astype(np.float32)


def capture(image, exposure, iso, params: SensorParams, flux_at_white=DEFAULT_FLUX, seed=0,
            meta=None):
    """Full sensor: ``expose`` then ``adc``.  Returns ``(dn, meta)``.

    meta carries domain='dn', bits, black_level, white_level, iso, exposure; keys of an
    input ``meta`` (e.g. ``cfa``) are kept.
    """
    r = rng(seed)
    dn = adc(expose(image, exposure, params, flux_at_white, seed=r), params, iso, seed=r)
    m = dict(meta) if meta else {}
    m.update(domain="dn", bits=params.bits, black_level=params.black_level,
             white_level=params.white_level, iso=iso, exposure=exposure)
    return dn, m


# --------------------------------------------------------------------------
# calibration and clean-up
# --------------------------------------------------------------------------
def dark_frame(shape, exposure, iso, params: SensorParams, seed=1):
    """Lens-cap exposure: dark current, hot pixels, FPN and read noise only (L2 s42-43)."""
    return adc(expose(np.zeros(shape), exposure, params, seed=seed), params, iso, seed=seed + 7919)


def master_dark(shape, exposure, iso, params: SensorParams, n=16, seed=1):
    """Average of ``n`` dark frames: keeps the fixed pattern, shrinks the random noise."""
    return average_frames([dark_frame(shape, exposure, iso, params, seed + k) for k in range(n)])


def subtract_dark(dn, dark, black_level=0.0):
    """dn - dark (+ black_level to keep the DN convention).  Not clipped."""
    return (np.asarray(dn, np.float32) - np.asarray(dark, np.float32) + np.float32(black_level)
            ).astype(np.float32)


def normalize(dn, meta, clip=True):
    """DN -> linear [0, 1]:  (dn - black) / (white - black).  ``clip=False`` keeps noise."""
    x = (np.asarray(dn, np.float64) - meta["black_level"]) / (meta["white_level"] - meta["black_level"])
    return (np.clip(x, 0, 1) if clip else x).astype(np.float32)


def fix_hot_pixels(mosaic, pattern="RGGB", threshold=None):
    """Replace pixels far above the median of their same-colour neighbours.

    On a mosaic the 8 same-colour neighbours of a pixel sit at offsets 0, +-2 (any CFA
    pattern); on an RGB image (H, W, 3) the same stride-2 footprint is applied per channel
    too (cheap, slightly wider than a 3x3).  A pixel is replaced if ``x - median >
    threshold``; default threshold = 8 x the robust (MAD) spread of ``x - median``.
    One-sided: hot pixels are bright.
    """
    a = np.asarray(mosaic, np.float32)
    if a.ndim == 3:
        return np.stack([fix_hot_pixels(a[..., c], pattern, threshold) for c in range(a.shape[2])], -1)
    fp = np.zeros((5, 5), bool)
    fp[::2, ::2] = True
    fp[2, 2] = False
    med = median_filter(a, footprint=fp, mode="reflect")
    d = a - med
    if threshold is None:
        mad = np.median(np.abs(d - np.median(d)))
        threshold = 8.0 * 1.4826 * max(float(mad), 1e-6)
    return np.where(d > threshold, med, a).astype(np.float32)


def average_frames(frames):
    """Mean of a stack of frames: random noise drops by sqrt(M) (L2 s43)."""
    return np.mean(np.asarray(frames, np.float64), axis=0).astype(np.float32)


# --------------------------------------------------------------------------
# theory
# --------------------------------------------------------------------------
def snr_theory(P, qe, t, D, Nr):
    """SNR = P QE t / sqrt(P QE t + D t + Nr^2).   Reference: L2 s31, s45."""
    s = np.asarray(P, np.float64) * qe * t
    return s / np.sqrt(s + D * t + Nr ** 2)


def dynamic_range(full_well, D, t, Nr):
    """DR = (full_well - D t) / sqrt(D t + Nr^2).   Reference: L2 s33."""
    return (full_well - D * t) / np.sqrt(D * t + Nr ** 2)


def snr_theory_params(value, exposure, params: SensorParams, flux_at_white=DEFAULT_FLUX):
    """``snr_theory`` for a pixel of scene value ``value`` using the mean dark current."""
    return snr_theory(value * flux_at_white, params.qe, exposure, params.dark_current,
                      params.read_noise)


# --------------------------------------------------------------------------
# measurements on simulated captures
# --------------------------------------------------------------------------
def measure_snr(params: SensorParams, value=0.5, exposure=0.01, iso=100, n=32, shape=(32, 32),
                flux_at_white=DEFAULT_FLUX, seed=0):
    """Empirical SNR from ``n`` captures of a flat field.

    Uses the temporal std of each pixel (so fixed pattern does not count as noise), in
    electrons.  ``mean_e`` is the signal charge (mean dark charge removed, as in the
    numerator of L2 s45).  Returns dict(snr, mean_e, std_e).  Compare with ``snr_theory_params``.
    """
    img = np.full(shape, value, np.float64)
    stack = np.stack([capture(img, exposure, iso, params, flux_at_white, seed=seed + k)[0]
                      for k in range(n)]).astype(np.float64)
    g = params.gain(iso)
    dark_e = float(np.mean(fixed_pattern(shape, params)["dark"])) * exposure
    mean_e = float(np.mean(stack - params.black_level) / g) - dark_e     # signal charge only
    std_e = float(np.sqrt(np.mean(np.var(stack, axis=0, ddof=1))) / g)
    return dict(snr=mean_e / std_e, mean_e=mean_e, std_e=std_e)


def photon_transfer(params: SensorParams, exposure=0.01, iso=100, values=None, shape=(64, 64),
                    flux_at_white=DEFAULT_FLUX, seed=0):
    """Photon-transfer curve: noise variance vs mean signal (DN), one point per flat field.

    Variance comes from the difference of two captures / 2, so fixed pattern cancels.
    Returns dict(mean, var, values); in the shot-noise regime var = gain * mean (slope =
    DN per electron), at low signal var -> (gain N_r)^2.
    """
    values = np.linspace(0.0, 1.0, 17) if values is None else np.asarray(values)
    mean, var = [], []
    for k, v in enumerate(values):
        img = np.full(shape, v, np.float64)
        a = capture(img, exposure, iso, params, flux_at_white, seed=seed + 2 * k)[0].astype(np.float64)
        b = capture(img, exposure, iso, params, flux_at_white, seed=seed + 2 * k + 1)[0].astype(np.float64)
        mean.append(0.5 * (a.mean() + b.mean()) - params.black_level)
        var.append(np.var(a - b, ddof=1) / 2.0)
    return dict(mean=np.array(mean), var=np.array(var), values=values)


def clean_params(params: SensorParams, **kw) -> SensorParams:
    """Copy of ``params`` with all fixed-pattern sources off (override anything via kw)."""
    return replace(params, **{**dict(prnu=0.0, dsnu=0.0, hot_pixel_fraction=0.0), **kw})
