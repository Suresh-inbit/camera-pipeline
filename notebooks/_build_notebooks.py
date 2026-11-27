"""Generate the four guided notebooks of the camera-pipeline toolkit in this folder.

Run from the project root:  python notebooks/_build_notebooks.py
Each notebook is a list of (kind, source) cells; kind is "md" or "code".
Execute them afterwards with nbconvert:

  jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=300 notebooks/0*.ipynb

Slide references: "L2 s34" = lecture 2, PDF page 34 of the decks.
"""
from __future__ import annotations
import pathlib
import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

HERE = pathlib.Path(__file__).resolve().parent

HEADER = r"""%matplotlib inline
# JPEG keeps the executed notebooks small (plots are photographs).
%config InlineBackend.figure_format = 'jpeg'
import sys, pathlib, time
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 80
# Make `import isp` work whether Jupyter was started in the project root or in notebooks/.
_cwd = pathlib.Path.cwd().resolve()
ROOT = _cwd.parent if _cwd.name == 'notebooks' else _cwd
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import isp
from isp import utils, scenes
srgb = utils.linear_to_srgb      # linear light -> display values for imshow


def show_row(images, titles, figsize=(12, 4), cmap=None, vmin=None, vmax=None):
    # one row of image panels, axes off
    fig, axs = plt.subplots(1, len(images), figsize=figsize)
    for ax, im, t in zip(np.atleast_1d(axs), images, titles):
        ax.imshow(np.clip(im, 0, 1) if (vmin is None and im.ndim == 3) else im,
                  cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
        ax.set_title(t, fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
"""


def md(text: str):
    return ("md", text.strip("\n"))


def code(text: str):
    return ("code", text.strip("\n"))


def build(filename: str, cells):
    nb = new_notebook()
    nb.cells = []
    for kind, src in cells:
        nb.cells.append(new_markdown_cell(src) if kind == "md" else new_code_cell(src))
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.metadata["language_info"] = {"name": "python"}
    nbformat.validate(nb)
    nbformat.write(nb, str(HERE / filename))
    print("wrote", HERE / filename, len(nb.cells), "cells")


# =============================================================================
# 01  Sensor and noise
# =============================================================================
NB01 = [
    md(r"""
# 01. Sensor and noise

A sensor counts photons, and counting is noisy. This notebook follows the signal along the block diagram of L2 s34: photons become electrons, noise gets added at several places, the ADC turns charge into digital numbers. You will measure shot noise, check the SNR formula of L2 s31-33 against the simulator in `isp.sensor`, compare exposure time against ISO, and average frames to beat noise (L2 s40-43). The theory is in `docs/01_sensor_and_noise.md`.
"""),
    code(HEADER + r"""
from dataclasses import replace
from isp import sensor, cfa

P = sensor.PRESETS
print({k: (v.name, v.bits, v.full_well, v.read_noise) for k, v in P.items()})
"""),
    md(r"""
## From photons to digital numbers (L2 s34)

`expose` is the left half of the diagram (photons, QE, dark current, Poisson noise, full-well clipping). `adc` is the right half (read noise, gain, black level, rounding). Below, the astronaut goes through both, one step at a time, as a single-plane RAW mosaic.
"""),
    code(r"""
scene = scenes.load('astronaut', 128)
mos = cfa.mosaic(scene)                          # what the CFA lets through (H, W)
par = P['dslr']
e = sensor.expose(mos, exposure=0.02, params=par, seed=0)             # electrons
dn = sensor.adc(e, par, iso=400, seed=1)                              # digital numbers
lin = sensor.normalize(dn, dict(black_level=par.black_level, white_level=par.white_level))

fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for ax, a, t in zip(axs, [mos, e, dn], ['scene mosaic (linear light)', 'electrons', 'DN (14 bit RAW)']):
    ax.imshow(a, cmap='gray'); ax.set_title(f'{t}\nrange {a.min():.0f}..{a.max():.0f}' if a.max() > 2 else f'{t}\nrange {a.min():.2f}..{a.max():.2f}', fontsize=10)
    ax.axis('off')
plt.tight_layout(); plt.show()
print('full well', par.full_well, 'e-  | white level', par.white_level, 'DN | black level', par.black_level, 'DN')
print('scene white gives', 0.4 * 1e5 * 0.02, 'e- (flux 1e5 photons/px/s x QE 0.4 x 0.02 s)')
"""),
    md(r"""
The three panels look alike but the numbers underneath change units: unit-less light, then electrons, then integer DN with an offset (the black level). Nothing reaches the full well here, so the picture is dark; ISO is the knob that makes it bright again without adding any light.
"""),
    md(r"""
## Shot noise: $\sigma = \sqrt{\lambda}$

Photons arrive randomly, so the electron count is Poisson distributed with variance equal to its mean. Here a perfect sensor (no read noise, no dark current) looks at flat fields of increasing brightness.
"""),
    code(r"""
ideal = sensor.clean_params(P['dslr'], dark_current=0.0, read_noise=0.0)

def shot_noise_demo(values=(0.002, 0.01, 0.05, 0.25, 1.0), exposure=0.025, plot=True):
    means, stds = [], []
    for k, v in enumerate(values):
        e = sensor.expose(np.full((128, 128), v), exposure, ideal, seed=k)
        means.append(e.mean()); stds.append(e.std())
    means, stds = np.array(means), np.array(stds)
    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        for k in (0, 2, 4):
            e = sensor.expose(np.full((128, 128), values[k]), exposure, ideal, seed=k)
            axs[0].hist(e.ravel(), bins=60, alpha=0.6, label=f'mean {e.mean():.0f} e-')
        axs[0].set_xscale('log'); axs[0].set_xlabel('electrons'); axs[0].legend(); axs[0].set_title('count distributions')
        axs[1].loglog(means, stds, 'o', label='measured std')
        axs[1].loglog(means, np.sqrt(means), '-', label=r'$\sqrt{mean}$')
        axs[1].set_xlabel('mean (e-)'); axs[1].set_ylabel('std (e-)'); axs[1].legend(); axs[1].set_title('noise grows as sqrt(signal)')
        axs[2].semilogx(means, means / stds, 'o', label='measured SNR')
        axs[2].semilogx(means, np.sqrt(means), '-', label=r'$\sqrt{mean}$')
        axs[2].set_xlabel('mean (e-)'); axs[2].set_ylabel('SNR'); axs[2].legend(); axs[2].set_title('SNR = mean / std')
        plt.tight_layout(); plt.show()
    return means, stds

means, stds = shot_noise_demo()
print('mean (e-):', np.round(means, 1))
print('std  (e-):', np.round(stds, 2), ' sqrt(mean):', np.round(np.sqrt(means), 2))
"""),
    md(r"""
The measured standard deviation follows $\sqrt{\text{mean}}$ over the whole range. Brighter pixels have more absolute noise but a better SNR, which is why shadows are always the noisy part of a photo.

## The SNR formula and the slide examples (L2 s31-33)

With shot noise, dark current $D$ and read noise $N_r$ adding in variance, $\mathrm{SNR}=\dfrac{P Q_e t}{\sqrt{P Q_e t + D t + N_r^2}}$. The dynamic range is the same idea for the largest signal against the noise in the dark. The two sensors on the slides are the `retiga_4000r` and `aptina_mt9p031` presets.
"""),
    code(r"""
R, A = P['retiga_4000r'], P['aptina_mt9p031']
snr_retiga = sensor.snr_theory(1000, R.qe, 1.0, R.dark_current, R.read_noise)
snr_aptina = sensor.snr_theory(1000 / 11, A.qe, 1.0, A.dark_current, A.read_noise)   # pixels 1/11 the area
print(f'SNR  Retiga 4000R: {snr_retiga:.2f}:1   (slide: 20.8:1)')
print(f'SNR  Aptina MT9P031: {snr_aptina:.2f}:1   (slide: 6.5:1)')
for name, p in (('Retiga', R), ('Aptina', A)):
    for t in (1.0, 1 / 60):
        dr = sensor.dynamic_range(p.full_well, p.dark_current, t, p.read_noise)
        print(f'DR   {name:7s} t={t:6.3f} s: {dr:6.0f}:1 = {np.log2(dr):.1f} bits')

def snr_curves(params_list, t=1.0, plot=True):
    # SNR against photon flux for several sensors
    P_ = np.logspace(0, 5, 200)
    curves = {p.name: sensor.snr_theory(P_, p.qe, t, p.dark_current, p.read_noise) for p in params_list}
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        for name, c in curves.items():
            axs[0].loglog(P_, c, label=name)
        axs[0].loglog(P_, np.sqrt(P_ * R.qe * t), 'k--', lw=1, label='shot-noise limit (Retiga)')
        axs[0].set_xlabel('photons / pixel / s'); axs[0].set_ylabel('SNR'); axs[0].legend(); axs[0].grid(alpha=.3)
        axs[0].set_title(f'SNR for t = {t:g} s')
        ts = np.logspace(-3, 1, 100)
        for p in params_list:
            axs[1].loglog(ts, sensor.snr_theory(1000, p.qe, ts, p.dark_current, p.read_noise), label=p.name)
        axs[1].set_xlabel('exposure time t (s)'); axs[1].set_ylabel('SNR at 1000 photons/px/s'); axs[1].legend(); axs[1].grid(alpha=.3)
        axs[1].set_title('slope 1 = read-noise limited, slope 1/2 = shot-noise limited')
        plt.tight_layout(); plt.show()
    return curves

_ = snr_curves([R, A, P['dslr'], P['phone']])
"""),
    code(r"""
def check_snr(names=('retiga_4000r', 'aptina_mt9p031', 'phone'), values=(0.01, 0.05, 0.2, 0.6, 1.0), exposure=0.02, plot=True):
    # simulated flat-field SNR (temporal std over 24 captures) against the formula
    rows = {}
    for nm in names:
        p = P[nm]
        th = [float(sensor.snr_theory_params(v, exposure, p)) for v in values]
        me = [sensor.measure_snr(p, value=v, exposure=exposure, iso=100, n=24, shape=(24, 24))['snr'] for v in values]
        rows[nm] = (th, me)
    if plot:
        plt.figure(figsize=(6, 4))
        for nm, (th, me) in rows.items():
            l, = plt.loglog(np.array(values) * 1e5 * P[nm].qe * exposure, th, '-', label=f'{nm} theory')
            plt.loglog(np.array(values) * 1e5 * P[nm].qe * exposure, me, 'o', color=l.get_color())
        plt.xlabel('signal (e-)'); plt.ylabel('SNR'); plt.legend(fontsize=8); plt.grid(alpha=.3)
        plt.title('lines: formula, dots: simulated'); plt.tight_layout(); plt.show()
    return rows

rows = check_snr()
for nm, (th, me) in rows.items():
    print(nm, 'max relative deviation: %.1f %%' % (100 * np.max(np.abs(np.array(me) / np.array(th) - 1))))
"""),
    md(r"""
Our numbers match the slides. In the left plot the curves bend from a slope of 1 (read noise dominates, every doubling of light doubles the SNR) to a slope of 1/2 (shot noise dominates); the Aptina's high dark current hurts it at long exposures, which is the point of slide 33. In the second figure the simulated dots (`measure_snr`, temporal std of a flat field) sit on the formula, up to a few percent of scatter from using only 24 frames.
"""),
    md(r"""
## A gallery of noise types (L2 s18-30)

Each panel switches on one noise source of the model, with the others off. Noise panels are stretched to their own mean $\pm 4\sigma$; the number in the title is the standard deviation in DN. For the two fixed-pattern sources the mean of 16 shots is shown, so the random part averages out and only what the sensor repeats every time is left.
"""),
    code(r"""
def noise_gallery(plot=True):
    base = replace(P['dslr'], bits=12, black_level=64, full_well=10000, qe=0.5, dark_current=0.0, read_noise=0.0,
                   prnu=0.0, dsnu=0.0, hot_pixel_fraction=0.0, hot_pixel_rate=0.0)
    fixed = replace(base, dark_current=50.0, dsnu=40.0, hot_pixel_fraction=2e-3, hot_pixel_rate=2000.0, prnu=0.03)
    quiet = replace(base, full_well=1e7, bits=3, black_level=0, qe=1.0)          # so much light that shot noise vanishes
    ramp = np.tile(np.linspace(0.1, 1.0, 96), (96, 1))
    flat = lambda v: np.full((96, 96), v)

    def avg(img, p, t, n=16):                    # n shots averaged: temporal noise shrinks, fixed pattern stays
        return np.mean([sensor.capture(img, t, 100, p, seed=10 + k)[0] for k in range(n)], axis=0)

    cases = [
        ('photon shot noise (one shot)', sensor.capture(flat(0.3), 0.02, 100, base, seed=3)[0]),
        ('+ read noise 30 e- (one shot)', sensor.capture(flat(0.3), 0.02, 100, replace(base, read_noise=30.0), seed=3)[0]),
        ('dark current + shot noise (one shot, 0.5 s)', sensor.capture(flat(0.0), 0.5, 100, replace(base, dark_current=300.0), seed=3)[0]),
        ('PRNU: gain pattern (mean of 16 shots)', avg(flat(0.8), replace(base, prnu=0.03), 0.02)),
        ('DSNU + hot pixels (mean of 16 dark shots)', avg(flat(0.0), fixed, 0.5)),
        ('quantisation: 3-bit ADC on a ramp', sensor.capture(ramp, 0.1, 100, quiet, flux_at_white=1e8, seed=3)[0]),
    ]
    out = []
    fig, axs = plt.subplots(2, 3, figsize=(14, 8))
    for ax, (name, dn) in zip(axs.ravel(), cases):
        m, s = dn.mean(), dn.std()
        if 'ramp' in name:
            ax.imshow(dn, cmap='gray', vmin=0, vmax=7, interpolation='nearest')
        else:
            ax.imshow(dn, cmap='gray', vmin=m - 4 * s, vmax=m + 4 * s, interpolation='nearest')
        ax.set_title(f'{name}\nstd {s:.2f} DN', fontsize=10); ax.axis('off')
        out.append((name, s))
    plt.tight_layout(); plt.show()
    return out

_ = noise_gallery()
"""),
    md(r"""
Shot and read noise are temporal: a new shot gives a new pattern. PRNU, DSNU and hot pixels are fixed in the sensor, so they come back identically every frame, which is why they can be calibrated away (see the dark-frame section below). Quantisation noise only matters when the ADC steps are comparable to the other noise.
"""),
    md(r"""
## Exposure time against ISO at equal brightness (L2 s36-39)

Brightness is $\propto$ light $\times$ ISO, so a long exposure at low ISO and a short exposure at high ISO give the same picture brightness. They do not give the same noise: the short one collected fewer electrons.
"""),
    code(r"""
def iso_vs_exposure(isos=(100, 200, 400, 800, 1600, 3200), product=40.0, plot=True):
    # t * ISO = product (so ISO 100 -> 0.4 s)
    p = P['dslr']
    sc = scenes.load('astronaut', 128)
    imgs, snrs, mean_e = [], [], []
    for iso in isos:
        t = product / iso
        dn, meta = sensor.capture(sc, t, iso, p, seed=5)
        imgs.append(srgb(sensor.normalize(dn, meta) * 2.0))        # x2 so the picture is not too dark
        r = sensor.measure_snr(p, value=0.5, exposure=t, iso=iso, n=16, shape=(24, 24))
        snrs.append(r['snr']); mean_e.append(r['mean_e'])
    if plot:
        pick = [0, 3, 5]
        show_row([imgs[k] for k in pick], [f'ISO {isos[k]}, t = {product / isos[k]:.4f} s' for k in pick])
        plt.figure(figsize=(6, 4))
        plt.semilogx(isos, 20 * np.log10(snrs), 'o-')
        plt.xlabel('ISO (t x ISO constant)'); plt.ylabel('SNR of a mid-grey pixel (dB)'); plt.grid(alpha=.3)
        plt.tight_layout(); plt.show()
    return dict(isos=isos, snr=snrs, electrons=mean_e)

r = iso_vs_exposure()
print('electrons at mid-grey:', np.round(r['electrons']).astype(int))
"""),
    md(r"""
The three pictures have the same brightness, but the high-ISO one is visibly grainier, and the curve drops with every doubling of ISO because the number of electrons halves. This is the rule of L2 s39: use the longest exposure that motion and saturation allow, and raise ISO last.
"""),
    md(r"""
## Averaging frames (L2 s40-43)

If the noise of each frame is independent, the mean of $M$ frames has $\sigma/\sqrt{M}$. A tripod and many short exposures are the camera version of this.
"""),
    code(r"""
def averaging_demo(Ms=(1, 2, 4, 8, 16, 32), plot=True):
    p = sensor.clean_params(P['phone'])
    flat = np.full((64, 64), 0.15)
    frames_flat = [sensor.normalize(sensor.capture(flat, 0.01, 800, p, seed=k)[0],
                                    dict(black_level=p.black_level, white_level=p.white_level), clip=False) for k in range(max(Ms))]
    stds = [sensor.average_frames(frames_flat[:M]).std() for M in Ms]
    sc = scenes.load('astronaut', 128)
    frames = [sensor.normalize(sensor.capture(sc, 0.01, 800, p, seed=k)[0],
                               dict(black_level=p.black_level, white_level=p.white_level)) for k in range(16)]
    if plot:
        pick = (1, 4, 16)
        show_row([srgb(sensor.average_frames(frames[:M]) * 3) for M in pick], [f'average of {M} frame(s)' for M in pick])
        plt.figure(figsize=(6, 4))
        plt.loglog(Ms, stds, 'o', label='measured std')
        plt.loglog(Ms, stds[0] / np.sqrt(Ms), '-', label=r'$\sigma_1 / \sqrt{M}$')
        plt.xlabel('frames M'); plt.ylabel('noise std (linear units)'); plt.legend(); plt.grid(alpha=.3)
        plt.tight_layout(); plt.show()
    return dict(Ms=Ms, stds=stds)

res = averaging_demo()
print('std(1) / std(M):', np.round(res['stds'][0] / np.array(res['stds']), 2), ' sqrt(M):', np.round(np.sqrt(res['Ms']), 2))
"""),
    md(r"""
The ratio of the noise of one frame to the noise of the average tracks $\sqrt{M}$, so 16 frames buy a factor 4. The cost is time, and the scene has to hold still, which is the trade-off that makes burst photography on phones work for static scenes only.
"""),
    md(r"""
## Dark frames and hot pixels (L2 s24-30)

Dark current, DSNU and hot pixels are the same every shot, so a picture taken with the lens cap on measures them. Averaging a few dark frames (a *master dark*) removes most of the random noise from that measurement; subtracting it removes the fixed pattern from a real shot.
"""),
    code(r"""
def dark_frame_demo(plot=True):
    from scipy.ndimage import median_filter
    p = replace(P['phone'], bits=12, black_level=64, dark_current=20.0, dsnu=15.0, hot_pixel_fraction=1.5e-3, hot_pixel_rate=800.0)
    sc = scenes.load('astronaut', 128) * 0.08
    t, iso = 0.5, 100
    dn, meta = sensor.capture(sc, t, iso, p, seed=2)
    dark = sensor.master_dark(sc.shape, t, iso, p, n=16)
    fixed = sensor.subtract_dark(dn, dark, p.black_level)
    raw, sub = sensor.normalize(dn, meta), sensor.normalize(fixed, meta)
    med = sensor.fix_hot_pixels(raw, threshold=0.03)
    hot = sensor.fixed_pattern(sc.shape, p)['hot']          # where the defects really are

    def excess(x):                  # how far the true hot pixels stick out above their 3x3 neighbourhood
        return float((x - median_filter(x, size=(3, 3, 1)))[hot].mean())
    if plot:
        show_row([srgb(raw * 4), sensor.normalize(dark, meta) * 4, srgb(sub * 4), srgb(med * 4)],
                 ['raw capture', 'master dark (x4)', 'dark subtracted', 'median hot-pixel fix'], figsize=(14, 4))
    return dict(raw=excess(raw), subtracted=excess(sub), median=excess(med), n_hot=int(hot.sum()))

res = dark_frame_demo()
print('true hot pixels:', res.pop('n_hot'), '| mean excess over the 3x3 median at those pixels (linear units):',
      {k: round(v, 3) for k, v in res.items()})
"""),
    md(r"""
Dark-frame subtraction removes the hot pixels and the dark-current glow together, and the median fix removes the bright outliers only. The subtraction is the cleaner one because it knows where the defects are; the median has to guess from the neighbours, so it can hurt real detail (doc 08 measures this).
"""),
    md(r"""
## The photon transfer curve

Plot noise variance against mean signal. In the shot-noise regime $\sigma^2_{DN} = g \cdot \mu_{DN}$, so the slope is the gain $g$ in DN per electron; the intercept at zero signal gives the read noise. This is how real sensors are characterised without knowing anything inside them.
"""),
    code(r"""
def ptc_demo(plot=True):
    p = sensor.clean_params(replace(P['dslr'], bits=12, black_level=64, full_well=20000, qe=0.5, dark_current=0.0, read_noise=8.0))
    iso = 100
    ptc = sensor.photon_transfer(p, exposure=0.02, iso=iso, values=np.linspace(0.0, 1.0, 21), shape=(64, 64))
    mean, var = ptc['mean'], ptc['var']
    ok = (mean > 0.05 * mean.max()) & (mean < 0.8 * mean.max())      # away from the floor and from the clipping
    slope, icpt = np.polyfit(mean[ok], var[ok], 1)
    read_e = np.sqrt(max(icpt - 1 / 12, 0)) / slope                    # remove quantisation (1/12 DN^2)
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        axs[0].plot(mean, var, 'o'); axs[0].plot(mean, slope * mean + icpt, '-')
        axs[0].set_xlabel('mean signal (DN above black)'); axs[0].set_ylabel('variance (DN$^2$)'); axs[0].set_title('photon transfer curve')
        axs[1].loglog(mean[1:], var[1:], 'o'); axs[1].loglog(mean[1:], slope * mean[1:] + icpt, '-')
        axs[1].set_xlabel('mean signal (DN)'); axs[1].set_title('log-log: floor at low signal, slope 1 above')
        plt.tight_layout(); plt.show()
    return dict(gain=slope, read_noise_e=read_e, true_gain=p.gain(iso), true_read=p.read_noise)

r = ptc_demo()
print(f"gain from the slope: {r['gain']:.4f} DN/e-   (sensor: {r['true_gain']:.4f})")
print(f"read noise from the intercept: {r['read_noise_e']:.1f} e-   (sensor: {r['true_read']:.1f})")
"""),
    md(r"""
The straight line recovers both the gain and the read noise of the simulated sensor, using nothing but flat-field statistics. The log-log view shows the two regimes: a flat floor where read noise dominates and a slope-1 rise where shot noise takes over.
"""),
    md(r"""
## Interactive capture

Change the exposure time, the ISO and the sensor and watch brightness, grain and clipping. The title reports the electrons that scene white collects and the formula's SNR for a mid-grey pixel.
"""),
    code(r"""
import ipywidgets as W
from IPython.display import display

_scene_small = scenes.load('astronaut', 128)

def capture_view(sensor_name='dslr', log2_exposure=-6.0, iso=800, flux=1e5):
    p = P[sensor_name]
    t = 2.0 ** log2_exposure
    dn, meta = sensor.capture(_scene_small, t, iso, p, flux_at_white=flux, seed=0)
    lin = sensor.normalize(dn, meta)
    e_white = flux * p.qe * t
    snr = sensor.snr_theory(0.18 * flux, p.qe, t, p.dark_current, p.read_noise)
    clipped = 100 * np.mean(lin >= 0.999)
    plt.figure(figsize=(5, 5)); plt.imshow(srgb(lin)); plt.axis('off')
    plt.title(f't = {t:.4f} s, ISO {iso}: white = {e_white:.0f} e- (well {p.full_well:.0f}), grey SNR {snr:.1f}\n{clipped:.1f} % clipped', fontsize=9)
    plt.tight_layout(); plt.show()

W.interact(capture_view, sensor_name=list(P), log2_exposure=W.FloatSlider(min=-10, max=0, step=0.5, value=-6, description='log2 t'),
           iso=W.SelectionSlider(options=[100, 200, 400, 800, 1600, 3200, 6400, 12800], value=800), flux=W.fixed(1e5));
"""),
    md(r"""
## Try it

1. Pick the `phone` preset in the widget above and move the exposure until the image is well exposed at ISO 100. How long is that, compared with the DSLR? The full-well capacity explains most of it.
2. Add a fourth sensor of your own to `snr_curves` (`replace(P['dslr'], read_noise=1.0)`) and find at which photon flux the read noise stops mattering.
3. In `averaging_demo`, replace `clean_params(P['phone'])` by `P['phone']` (with its fixed pattern) and check whether the $\sqrt{M}$ law still holds for the flat field. What does the average converge to?
"""),
    code(r"""
# Exercise 2: a quieter sensor. Edit read_noise and compare the curves.
mine = replace(P['dslr'], read_noise=1.0, name='quiet DSLR')
_ = snr_curves([P['dslr'], mine], t=0.01)

# Exercise 3: how does the std of the averaged flat field behave when PRNU is left on?
p = replace(P['phone'], prnu=0.02)
flat = np.full((64, 64), 0.3)
meta = dict(black_level=p.black_level, white_level=p.white_level)
fr = [sensor.normalize(sensor.capture(flat, 0.01, 400, p, seed=k)[0], meta, clip=False) for k in range(32)]
for M in (1, 4, 16, 32):
    print(M, 'frames: spatial std of the average = %.4f' % sensor.average_frames(fr[:M]).std())
"""),
]



# =============================================================================
# 02  Colour, demosaicing and white balance
# =============================================================================
NB02 = [
    md(r"""
# 02. Colour, demosaicing and white balance

A colour sensor does not record colour. Every pixel sits behind one filter and stores one number, so two thirds of the colour data is missing before the first line of processing. This notebook builds the mosaic (L3 s13), compares the ways of filling the gaps (L3 s17, s26-27), shows why colour can be treated more roughly than brightness (L3 s20-25), and then moves to the light source: colour temperature, the xy diagram and three ways of estimating white balance (L4 s5-11). Theory: `docs/02_color_sensing_demosaicing.md`, `docs/03_white_balance.md`.
"""),
    code(HEADER + r"""
from isp import cfa, demosaic, whitebalance as wb
import ipywidgets as W

scene = scenes.load('astronaut', 256)            # linear light, the ground truth
print(scene.shape, scene.dtype, 'patterns:', cfa.PATTERNS, 'methods:', demosaic.METHODS)
"""),
    md(r"""
## The colour filter array

Four arrangements of the 2x2 Bayer tile are in use; the string is read row by row, so `RGGB` puts red at the top left. Green appears twice because the eye is most sensitive to it and it carries most of the luminance.
"""),
    code(r"""
def show_patterns(patterns=cfa.PATTERNS, n=8, plot=True):
    tiles = [cfa.colorize(cfa.mosaic(np.ones((n, n, 3), np.float32), p), p) for p in patterns]
    if plot:
        show_row(tiles, list(patterns), figsize=(12, 3.4))
    return tiles

_ = show_patterns()

m = cfa.mosaic(scene)
print('RGB image:', scene.size, 'numbers | mosaic:', m.size, 'numbers | kept: %.0f %%' % (100 * m.size / scene.size))
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(srgb(scene)); axs[0].set_title('scene (3 numbers per pixel)')
axs[1].imshow(m, cmap='gray'); axs[1].set_title('mosaic (1 number per pixel)')
axs[2].imshow(cfa.colorize(m)[60:92, 100:132], interpolation='nearest'); axs[2].set_title('32 x 32 crop, in filter colours')
for ax in axs: ax.axis('off')
plt.tight_layout(); plt.show()
"""),
    md(r"""
The mosaic looks like a grey photograph with a fine checker texture, which is the colour information encoded as brightness differences between neighbours. Demosaicing has to turn that back into three full planes.

## Demosaicing methods on a zone plate

A zone plate has spatial frequency that grows with the radius, so it shows where each interpolation starts to fail. The scene is grey, so every coloured fringe is an error (false colour). Methods: nearest, bilinear, bicubic, Freeman's median (L3 s26-27) and Malvar's gradient-corrected filter.
"""),
    code(r"""
def compare_demosaic(img, methods=demosaic.METHODS, crop=None, plot=True, display=lambda x: x, **kw):
    # mosaic -> each method; returns {method: (psnr dB, false-colour error)}; grid of crops when plot=True
    m = cfa.mosaic(img)
    outs = {k: demosaic.demosaic(m, 'RGGB', k, **kw) for k in methods}
    res = {k: (utils.psnr(img, o), demosaic.false_color(o, img)) for k, o in outs.items()}
    if plot:
        sl = (slice(None), slice(None)) if crop is None else (slice(crop[0], crop[1]), slice(crop[2], crop[3]))
        fig, axs = plt.subplots(2, 3, figsize=(14, 8))
        panels = [('ground truth', img)] + [(k, outs[k]) for k in methods]
        for ax, (name, im) in zip(axs.ravel(), panels):
            ax.imshow(np.clip(display(im[sl]), 0, 1), interpolation='nearest'); ax.axis('off')
            ax.set_title(name if name == 'ground truth' else f'{name}: {res[name][0]:.1f} dB, false colour {res[name][1]:.3f}', fontsize=10)
        plt.tight_layout(); plt.show()
    return res

zp = scenes.zone_plate(256)
res_zp = compare_demosaic(zp, crop=(0, 128, 0, 128))
"""),
    md(r"""
The crops show the top-left quarter, where the pattern is at its highest frequency. Nearest neighbour and bilinear show strong coloured moire; the methods that look across channels (median, Malvar) keep the colour error lower. A grey scene is a harsh test because any frequency above the sampling limit of the red and blue planes (half that of the full grid) aliases.

## The same on a photograph

On natural images the differences are smaller and live on edges. The metrics are over the whole image; the crop is a region of the astronaut shown through the sRGB curve.
"""),
    code(r"""
res_ph = compare_demosaic(scene, crop=(110, 174, 0, 64), display=srgb)
print('{:10s} {:>8s} {:>13s}'.format('method', 'PSNR dB', 'false colour'))
for k, (p, f) in res_ph.items():
    print('{:10s} {:8.2f} {:13.4f}'.format(k, p, f))
"""),
    md(r"""
Higher PSNR and lower false colour go together here. Bilinear is the cheap baseline; once the method looks across channels, the colour fringes along edges mostly disappear.

## Freeman's median method and noise (L3 s26-27)

Freeman first interpolates bilinearly, then median-filters the colour *differences* (R-G and B-G). A median throws away the outliers that sit on edges, so zipper colours go, and it also suppresses colour noise. The window size is the knob; we test it on the clean mosaic and on the same mosaic with noise added.
"""),
    code(r"""
def median_sweep(img, sizes=(1, 3, 5, 7), noise=0.0, plot=True):
    m = cfa.mosaic(img)
    if noise:
        m = (m + utils.rng(0).normal(0, noise, m.shape)).astype(np.float32)
    outs, rows = {}, []
    for k in sizes:
        o = demosaic.demosaic(m, 'RGGB', 'bilinear') if k == 1 else demosaic.demosaic(m, 'RGGB', 'median', median_size=k)
        outs[k] = o
        rows.append((k, utils.psnr(img, o), demosaic.false_color(o, img)))
    if plot:
        show_row([srgb(np.clip(outs[k], 0, 1))[110:174, 0:64] for k in sizes],
                 [f'median_size={k}' if k > 1 else 'bilinear (no median)' for k in sizes], figsize=(14, 3.6))
    return rows

for noise in (0.0, 0.03):
    print(f'--- mosaic noise std {noise}')
    for k, p, f in median_sweep(scene, noise=noise, plot=(noise > 0)):
        print(f'median_size {k}: PSNR {p:5.2f} dB, false colour {f:.4f}')
"""),
    md(r"""
Any median beats plain bilinear, and a small window is the sweet spot: a very large one starts to wipe out real colour edges. With noise the median also acts as a colour-noise filter, which is why the default camera in notebook 04 uses it.

## Two thirds of your data is made up

Only one of three channel values at each pixel was measured. The maps below show where the bilinear reconstruction differs from the truth, per channel: black where the sensor measured the value, brighter where it was interpolated.
"""),
    code(r"""
def error_maps(img, method='bilinear', gain=6.0, plot=True):
    m = cfa.mosaic(img)
    out = demosaic.demosaic(m, 'RGGB', method)
    mk = cfa.masks(m.shape, 'RGGB')
    err = np.abs(out - img)
    share = [float(err[..., c][~mk[..., c]].sum() / err[..., c].sum()) for c in range(3)]
    if plot:
        show_row([np.clip(err[..., c] * gain, 0, 1) for c in range(3)],
                 [f'{n} error (x{gain:g})' for n in 'RGB'], cmap='gray', figsize=(12, 4), vmin=0, vmax=1)
    return dict(measured_fraction=[float(mk[..., c].mean()) for c in range(3)], error_share_at_interpolated=share)

r = error_maps(scene)
print('fraction of pixels where each channel was measured (R, G, B):', np.round(r['measured_fraction'], 2))
print('share of the absolute error that sits on interpolated pixels:', np.round(r['error_share_at_interpolated'], 3))
"""),
    md(r"""
The measured pixels are exact and every bit of error sits on invented values: red and blue are measured on one pixel in four, green on one in two. Fine detail in the red and blue planes is therefore the hardest to recover, and the grid structure of the maps is the Bayer pattern itself.

## Luminance against chrominance (L3 s18-25)

Why is it acceptable to under-sample colour? Blur only the luma channel of a picture and you destroy it; blur only the chroma channels and it still looks sharp. JPEG (notebook 04) and luma/chroma denoising (notebook 03) rely on exactly this.
"""),
    code(r"""
def luma_chroma_blur(img_display, sigma=4.0, plot=True):
    y = demosaic.blur_channel(img_display, 'Y', sigma)
    c = demosaic.blur_channel(img_display, 'CbCr', sigma)
    m = {'blur Y': utils.psnr(img_display, y), 'blur CbCr': utils.psnr(img_display, c)}
    if plot:
        show_row([img_display, y, c], ['original', f'luminance blurred (sigma {sigma:g})', f'chrominance blurred (sigma {sigma:g})'])
    return m

disp = srgb(scene)
print({k: round(v, 1) for k, v in luma_chroma_blur(disp).items()}, '(PSNR dB vs original)')
"""),
    md(r"""
The chroma-blurred picture is hard to tell from the original while the luma-blurred one has lost its detail, although both used the same blur. The PSNR numbers agree with what you see.

## Colour temperature and the xy diagram (L4 s5-8)

Light sources are described by the temperature of a black body with the same colour: low temperatures are orange, high ones blue. On the CIE xy diagram the black-body colours form the Planckian locus; white balance has to undo where along that curve the light sits.
"""),
    code(r"""
def xy_diagram(presets=('incandescent', 'sunrise', 'fluorescent', 'daylight', 'shade'), plot=True):
    locus = wb.spectral_locus_xy()
    T = np.linspace(1700, 12000, 120)
    pl = wb.planckian_locus_xy(T)
    gamut = wb.srgb_gamut_xy()
    pts = {k: wb.cct_to_xy(wb.PRESETS[k]) for k in presets}
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw=dict(width_ratios=[1, 1.2]))
        ax = axs[0]
        ax.plot(*np.vstack([locus, locus[:1]]).T, 'k-', lw=1)
        ax.plot(*np.vstack([gamut, gamut[:1]]).T, 'b--', lw=1, label='sRGB gamut')
        ax.plot(*pl.T, 'r-', label='Planckian locus')
        for k, (x, y) in pts.items():
            ax.plot(x, y, 'o'); ax.annotate(f'{k} {wb.PRESETS[k]} K', (x, y), textcoords='offset points', xytext=(6, 4), fontsize=8)
        ax.set_xlabel('x'); ax.set_ylabel('y'); ax.legend(fontsize=8); ax.set_aspect('equal'); ax.set_title('CIE xy chromaticity')
        sw = [srgb(np.clip(wb.apply_illuminant(np.full((40, 60, 3), 0.5, np.float32), wb.PRESETS[k]), 0, 1)) for k in presets]
        axs[1].imshow(np.concatenate(sw, axis=1)); axs[1].axis('off')
        axs[1].set_title('a grey card lit by: ' + ', '.join(presets), fontsize=9)
        plt.tight_layout(); plt.show()
    return pts

_ = xy_diagram()
print('illuminant colours, G = 1 (R, G, B):')
for k in ('incandescent', 'daylight', 'shade'):
    print(f'  {k:13s} {wb.PRESETS[k]} K ->', np.round(wb.illuminant_rgb(wb.PRESETS[k]), 2))
"""),
    md(r"""
Tungsten sits at the orange end, shade at the blue end, and daylight lands near the neutral point (D65 is the white of sRGB). A grey card keeps its reflectance and takes on the colour of the light, which is why the sensor sees a cast instead of grey.

## Estimating the light: patch, gray world, white patch (L4 s9-11)

All three methods find gains $S_R, S_G, S_B$ and multiply the linear channels. They differ in how they guess the light: from a patch you say is neutral (`S_R = (R+G+B)/(3R)` on the slide), from the assumption that the scene averages to grey, or from the assumption that the brightest pixels are white. The error is the angle between the estimated and the true illuminant colour, which ignores brightness.
"""),
    code(r"""
def compare_wb(img, T=3200, patch_box=None, plot=True):
    lit = wb.apply_illuminant(img, T)
    true_light = wb.gains_to_illuminant(wb.gains_for_illuminant(T))
    gains = {'none': np.ones(3, np.float32), 'gray world': wb.gray_world(lit), 'white patch': wb.white_patch(lit)}
    if patch_box is not None:
        gains['manual patch'] = wb.from_patch(lit, patch_box)
    gains['true illuminant'] = wb.gains_for_illuminant(T)
    errs = {k: wb.angular_error(wb.gains_to_illuminant(g), true_light) for k, g in gains.items()}
    if plot:
        n = len(gains)
        fig, axs = plt.subplots(1, n + 1, figsize=(3.2 * (n + 1), 3.6))
        axs[0].imshow(srgb(img)); axs[0].set_title('original scene'); axs[0].axis('off')
        for ax, (k, g) in zip(axs[1:], gains.items()):
            ax.imshow(srgb(np.clip(wb.apply_gains(lit, g) / (1.0 if k != 'none' else wb.illuminant_rgb(T).max()), 0, 1)))
            ax.set_title(f'{k}\nerror {errs[k]:.1f} deg', fontsize=9); ax.axis('off')
        plt.tight_layout(); plt.show()
    return errs

chart = scenes.patches(256)
boxes = scenes.patch_boxes(256)
errs_chart = compare_wb(chart, T=3200, patch_box=boxes[20])          # patch 20 is a neutral grey
errs_astro = compare_wb(scene, T=3200)
print('chart :', {k: round(v, 1) for k, v in errs_chart.items()})
print('astro :', {k: round(v, 1) for k, v in errs_astro.items()})
"""),
    md(r"""
A neutral patch you point at (and the white patch, when the scene really contains white) gets the light almost exactly. Gray world is off because neither scene averages to grey; the astronaut has a lot of orange suit. Which automatic method is better depends on the scene; neither is right in general.

## When the assumptions fail: the sunset (L4 s10)

The sunset scene is orange on purpose. Its light is neutral (nothing to correct), but gray world sees a very orange average and tries to cancel it.
"""),
    code(r"""
def sunset_failure(plot=True):
    sun = scenes.sunset(256)
    truth = np.ones(3, np.float32)                       # neutral light: the right gains are all 1
    out = {}
    for name, g in (('gray world', wb.gray_world(sun)), ('white patch', wb.white_patch(sun))):
        out[name] = (wb.angular_error(wb.gains_to_illuminant(g), truth), wb.apply_gains(sun, g))
    if plot:
        show_row([srgb(sun)] + [srgb(np.clip(v[1], 0, 1)) for v in out.values()],
                 ['sunset as shot (light is neutral)'] + [f'{k}: error {v[0]:.1f} deg' for k, v in out.items()])
    return {k: v[0] for k, v in out.items()}

errs_sun = sunset_failure()
print({k: round(v, 1) for k, v in errs_sun.items()})
"""),
    md(r"""
Both automatic methods are fooled. Gray world reads the orange as a cast and drags the horizon towards olive and the sky towards teal; white patch takes the brightest pixels (the orange horizon) as white and turns the sky a strong blue, which is even further off. Real cameras combine several estimators with priors on plausible illuminants.

## Interactive: illuminant and method

Pick a scene, the light and the estimator. The right panel is the corrected linear image shown through the sRGB curve.
"""),
    code(r"""
_scenes_i = {n: scenes.load(n, 160) for n in ('astronaut', 'coffee', 'chelsea', 'sunset', 'patches')}

def wb_view(scene_name='astronaut', cct=3200, method='gray world'):
    img = _scenes_i[scene_name]
    lit = wb.apply_illuminant(img, cct)
    g = {'none': np.ones(3), 'gray world': wb.gray_world(lit), 'white patch': wb.white_patch(lit), 'true': wb.gains_for_illuminant(cct)}[method]
    err = wb.angular_error(wb.gains_to_illuminant(g), wb.gains_to_illuminant(wb.gains_for_illuminant(cct)))
    show_row([srgb(np.clip(lit, 0, 1)), srgb(np.clip(wb.apply_gains(lit, g), 0, 1))],
             [f'lit by {cct} K', f'{method}: gains {np.round(g, 2)}, error {err:.1f} deg'], figsize=(9, 4.2))

W.interact(wb_view, scene_name=list(_scenes_i), cct=W.IntSlider(min=2000, max=10000, step=250, value=3200),
           method=['none', 'gray world', 'white patch', 'true']);
"""),
    md(r"""
## Try it

1. In `compare_demosaic`, run the `coffee` and `chelsea` scenes and see whether the ranking of the five methods changes. Fur is a good stress test.
2. Add noise to the mosaic (as in `median_sweep`) and ask which method degrades most.
3. Run `compare_wb` for 6500 K and 9500 K. Why is the error of the true-illuminant row always about zero, and what does the error of `none` tell you about the strength of the cast?
"""),
    code(r"""
# Exercises 1 and 3: edit the scene name / temperature and re-run
coffee = scenes.load('coffee', 256)
print({k: tuple(round(x, 3) for x in v) for k, v in compare_demosaic(coffee, plot=False).items()})
print({k: round(v, 1) for k, v in compare_wb(scene, T=9500, plot=False).items()})
"""),
]



# =============================================================================
# 03  Tone mapping, denoising and sharpening
# =============================================================================
NB03 = [
    md(r"""
# 03. Tone mapping, denoising and sharpening

After white balance and demosaicing the image is still linear, noisy and soft. Three jobs remain: squeeze the scene's brightness range into something a display can show (L4 s13-18), remove noise without erasing edges (L5 s5-10), and put crispness back (L5 s12-16). Each step has a cheap global version and a better edge-aware one, and each can be overdone. Theory: `docs/04_tone_mapping.md`, `docs/05_denoising.md`, `docs/06_sharpening.md`.
"""),
    code(HEADER + r"""
from isp import tonemap, denoise, sharpen
import ipywidgets as W

hc = scenes.high_contrast(256)                       # linear; the left side is ~50x darker than the right
clean = srgb(scenes.load('astronaut', 256))          # display-referred clean photo for the denoise / sharpen parts
noisy = np.clip(clean + utils.rng(0).normal(0, 0.06, clean.shape), 0, 1).astype(np.float32)
print('high-contrast scene: mean linear value left third %.3f, right third %.3f' % (hc[:, :85].mean(), hc[:, 171:].mean()))
print('noisy photo PSNR vs clean: %.1f dB' % utils.psnr(noisy, clean))
"""),
    md(r"""
## Gamma and the sRGB curve (L4 s14)

Linear values put almost everything of a dim scene in the first few display levels. A power curve $\text{out}=\text{in}^{g}$ with $g<1$ lifts the shadows; sRGB is close to $g=1/2.2$ with a straight start near black. It is one fixed curve for the whole picture.
"""),
    code(r"""
def gamma_demo(img, gammas=(1.0, 0.7, 0.45), plot=True):
    x = np.linspace(0, 1, 256)
    outs = [tonemap.tonemap(img, 'linear')] + [tonemap.tonemap(img, 'gamma', g=g) for g in gammas[1:]] + [tonemap.tonemap(img, 'srgb')]
    if plot:
        fig, axs = plt.subplots(1, 5, figsize=(15, 3.4), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1]))
        for g in gammas:
            axs[0].plot(x, x ** g, label=f'g = {g}')
        axs[0].plot(x, utils.linear_to_srgb(x), 'k--', label='sRGB'); axs[0].legend(fontsize=7); axs[0].set_title('curves')
        for ax, o, t in zip(axs[1:], outs, ['linear (no curve)'] + [f'gamma {g}' for g in gammas[1:]] + ['sRGB']):
            ax.imshow(o); ax.axis('off'); ax.set_title(t, fontsize=10)
        plt.tight_layout(); plt.show()
    return outs

_ = gamma_demo(hc)
"""),
    md(r"""
Without a curve the dark half is nearly black. Gamma and sRGB open it up, and they do it everywhere, including the bright window, which begins to wash out. A curve that depends only on the input value cannot treat the two regions differently.

## Histogram equalisation (L4 s15)

Equalisation picks the curve from the picture itself: the CDF of the luminance histogram, so every output level is used about equally. CLAHE does it per tile with a clip limit so noise in flat areas is not blown up.
"""),
    code(r"""
def hist_demo(img, plot=True):
    outs = {'sRGB': tonemap.tonemap(img, 'srgb'), 'global histogram equalisation': tonemap.tonemap(img, 'hist'),
            'CLAHE (8 x 8 tiles)': tonemap.tonemap(img, 'clahe')}
    if plot:
        fig, axs = plt.subplots(2, 3, figsize=(14, 8), gridspec_kw=dict(height_ratios=[3, 1.3]))
        for j, (name, o) in enumerate(outs.items()):
            axs[0, j].imshow(o); axs[0, j].axis('off'); axs[0, j].set_title(name, fontsize=10)
            axs[1, j].hist(utils.luminance(o).ravel(), bins=64, range=(0, 1), color='gray'); axs[1, j].set_xlabel('display luminance')
        plt.tight_layout(); plt.show()
    return outs

_ = hist_demo(hc)
"""),
    md(r"""
The sRGB histogram piles up at the dark end; equalisation spreads it flat and pulls detail out of the dark half, at the price of an unnatural, contrasty look. Both are still global (CLAHE less so). The step that follows is the one that treats regions differently.

## Global against local: dodging and burning (L4 s16-18)

Local tone mapping splits log-luminance into a smooth *base* (the large brightness differences between regions) and *detail*. It compresses the base towards its brightest value and keeps the detail, which is what a darkroom printer does with dodge and burn. `strength` is how much of the base contrast is removed.
"""),
    code(r"""
def tone_compare(img, strengths=(0.4, 0.75), plot=True):
    outs = {'sRGB (global)': tonemap.tonemap(img, 'srgb'), 'histogram eq.': tonemap.tonemap(img, 'hist')}
    for s in strengths:
        outs[f'dodge and burn, strength {s}'] = tonemap.tonemap(img, 'local', strength=s)
    stats = {k: (utils.luminance(o)[:, :85].mean(), utils.luminance(o)[:, 171:].mean()) for k, o in outs.items()}
    if plot:
        show_row(list(outs.values()), list(outs), figsize=(15, 3.8))
        base, detail = tonemap.decompose(img)
        show_row([base, detail], ['base layer (log2 luminance, smooth)', 'detail layer'], figsize=(8, 3.4), cmap='gray', vmin=None, vmax=None)
    return stats

stats = tone_compare(hc)
print('{:32s} {:>10s} {:>10s} {:>8s}'.format('', 'dark left', 'bright right', 'ratio'))
for k, (a, b) in stats.items():
    print('{:32s} {:10.3f} {:10.3f} {:8.1f}'.format(k, a, b, b / a))
"""),
    md(r"""
The ratio of the two sides in display luminance shrinks as the local operator gets stronger, while the detail inside each region survives (look at the face and suit on the left). The cost is an artificial, flat look at high strength; the base layer's blur radius sets the halos around bright objects.

## Gaussian against bilateral denoising (L5 s5-9)

A Gaussian averages neighbours by distance only, so it blurs across edges. The bilateral filter multiplies that weight by a second one that depends on how different the intensity is:
$$w(i,j,k,l)=\exp\!\Big(-\tfrac{(i-k)^2+(j-l)^2}{2\sigma_d^2}-\tfrac{\|I(i,j)-I(k,l)\|^2}{2\sigma_r^2}\Big),\qquad I_D(i,j)=\frac{\sum I(k,l)\,w}{\sum w}.$$
The photo below has Gaussian noise added in the display domain; the numbers are PSNR and SSIM against the clean picture.
"""),
    code(r"""
def denoise_compare(noisy, clean, crop=(30, 130, 60, 160), plot=True):
    outs = {'noisy': noisy,
            'Gaussian sigma 1.5': denoise.gaussian(noisy, 1.5),
            'median 3x3': denoise.median(noisy, 3),
            'bilateral (2, 0.1)': denoise.bilateral(noisy, sigma_d=2.0, sigma_r=0.1),
            'luma / chroma': denoise.luma_chroma(noisy)}
    outs = {k: np.clip(v, 0, 1) for k, v in outs.items()}
    m = {k: (utils.psnr(clean, v), utils.ssim(clean, v)) for k, v in outs.items()}
    if plot:
        sl = (slice(crop[0], crop[1]), slice(crop[2], crop[3]))
        fig, axs = plt.subplots(2, 3, figsize=(14, 8))
        for ax, (k, v) in zip(axs.ravel(), [('clean', clean)] + list(outs.items())):
            ax.imshow(v[sl], interpolation='nearest'); ax.axis('off')
            ax.set_title(k if k == 'clean' else f'{k}: {m[k][0]:.1f} dB, SSIM {m[k][1]:.3f}', fontsize=10)
        plt.tight_layout(); plt.show()
    return m

m_dn = denoise_compare(noisy, clean)
"""),
    md(r"""
The Gaussian removes noise but smears the eyes and the collar ring; the bilateral keeps those edges and still smooths the flat skin, which shows in PSNR and SSIM. The median is better than the Gaussian but still rounds off fine features.

## What the bilateral filter does at an edge (L5 s6-9)

The weight window around one pixel is the thing to look at. Next to a step edge, the Gaussian window straddles it and averages the two sides; the bilateral one fades out the pixels across the edge, because their value differs by more than $\sigma_r$.
"""),
    code(r"""
def weight_windows(sigma_d=3.0, sigma_r=0.1, offset=3, noise=0.02, plot=True):
    img = scenes.step_edge(64)[..., 0]
    img = (img + utils.rng(1).normal(0, noise, img.shape)).astype(np.float32)
    c = (32, 32 + offset)                                   # a pixel `offset` px to the bright side of the edge
    gauss = denoise.bilateral_weights(img, c, sigma_d, np.inf)
    bil = denoise.bilateral_weights(img, c, sigma_d, sigma_r)
    r = gauss.shape[0] // 2
    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(img[c[0] - r:c[0] + r + 1, c[1] - r:c[1] + r + 1], cmap='gray', vmin=0, vmax=1)
        axs[0].plot(r, r, 'r+', ms=14); axs[0].set_title('image around the pixel (red +)')
        axs[1].imshow(gauss, vmin=0, vmax=1); axs[1].set_title('Gaussian weights (sigma_r = inf)')
        axs[2].imshow(bil, vmin=0, vmax=1); axs[2].set_title(f'bilateral weights (sigma_r = {sigma_r})')
        for ax in axs: ax.axis('off')
        plt.tight_layout(); plt.show()
    dark = img[c[0] - r:c[0] + r + 1, c[1] - r:c[1] + r + 1] < 0.5          # window pixels on the dark side of the edge
    return float(gauss[dark].sum() / gauss.sum()), float(bil[dark].sum() / bil.sum())

wg, wbil = weight_windows()
print(f'share of the weight that falls on the dark side of the edge: Gaussian {100 * wg:.0f} %, bilateral {100 * wbil:.1f} %')
"""),
    md(r"""
The bilateral window keeps only the bright side, so averaging with it cannot pull the dark side's values in. The tuning rule follows from this: $\sigma_r$ should be a bit above the noise level (so noise differences count as "similar") and below the contrast of the edges you want to keep.

## Luma and chroma denoising (L5 s10)

Colour noise can be smoothed harder than brightness noise (notebook 02, luma against chroma). `luma_chroma` filters Y gently and Cb, Cr with a big window.
"""),
    code(r"""
def chroma_demo(noisy, clean, crop=(30, 130, 60, 160), plot=True):
    rgb_f = np.clip(denoise.bilateral(noisy, sigma_d=1.5, sigma_r=0.1), 0, 1)
    lc = np.clip(denoise.luma_chroma(noisy), 0, 1)
    from isp import demosaic
    m = {k: (utils.psnr(clean, v), demosaic.false_color(v, clean)) for k, v in {'noisy': noisy, 'bilateral on RGB': rgb_f, 'luma / chroma': lc}.items()}
    if plot:
        sl = (slice(crop[0], crop[1]), slice(crop[2], crop[3]))
        show_row([clean[sl], noisy[sl], rgb_f[sl], lc[sl]], ['clean', 'noisy'] + [f'{k}\n{m[k][0]:.1f} dB, colour error {m[k][1]:.3f}' for k in list(m)[1:]], figsize=(14, 3.8))
    return m

_ = chroma_demo(noisy, clean)
"""),
    md(r"""
The luma/chroma version has a clearly lower colour error (the number under each panel) at a similar PSNR, because chroma is smoothed over a large window while luma is filtered gently. What is left is mostly luminance grain, which the eye finds less objectionable than colour speckle.

## Choosing $(\sigma_d, \sigma_r)$

A small sweep on a 128 x 128 version of the noisy photo: PSNR for each pair, and three pictures from the grid. Too small a $\sigma_r$ leaves the noise, too large turns the filter into a Gaussian.
"""),
    code(r"""
def sweep_bilateral(noisy, clean, sigma_ds=(1.0, 2.0, 3.0, 4.0), sigma_rs=(0.03, 0.06, 0.1, 0.2, 0.5), plot=True):
    grid = np.array([[utils.psnr(clean, np.clip(denoise.bilateral(noisy, d, r), 0, 1)) for r in sigma_rs] for d in sigma_ds])
    i, j = np.unravel_index(grid.argmax(), grid.shape)
    if plot:
        fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
        axs[0].imshow(grid, cmap='viridis', aspect='auto')
        axs[0].set_xticks(range(len(sigma_rs)), sigma_rs); axs[0].set_yticks(range(len(sigma_ds)), sigma_ds)
        axs[0].set_xlabel('sigma_r'); axs[0].set_ylabel('sigma_d'); axs[0].set_title('PSNR (dB)')
        for a in range(grid.shape[0]):
            for b in range(grid.shape[1]):
                axs[0].text(b, a, f'{grid[a, b]:.1f}', ha='center', va='center', color='w' if (a, b) != (i, j) else 'yellow', fontsize=8)
        picks = [(sigma_ds[i], sigma_rs[0]), (sigma_ds[i], sigma_rs[j]), (sigma_ds[i], sigma_rs[-1])]
        for ax, (d, r), t in zip(axs[1:], picks, ['sigma_r too small', 'best of the grid', 'sigma_r too large']):
            ax.imshow(np.clip(denoise.bilateral(noisy, d, r), 0, 1)); ax.axis('off')
            ax.set_title(f'{t}: ({d}, {r})', fontsize=10)
        plt.tight_layout(); plt.show()
    return grid, (sigma_ds[i], sigma_rs[j])

small_clean = clean[::2, ::2]
small_noisy = np.clip(small_clean + utils.rng(2).normal(0, 0.06, small_clean.shape), 0, 1).astype(np.float32)
grid, best = sweep_bilateral(small_noisy, small_clean)
print('best (sigma_d, sigma_r):', best, '  noise std was 0.06')
"""),
    md(r"""
The best $\sigma_r$ sits somewhat above the noise standard deviation, as the rule says, and larger $\sigma_d$ helps while noise dominates. At the large end of $\sigma_r$ the filter degrades towards a Gaussian blur; at the small end it does almost nothing.

## Laplacian sharpening (L5 s12-14)

Sharpening adds a high-pass copy of the image back to itself, $S = x + \lambda F(x)$. With the 4-neighbour Laplacian as $F$, the whole operation is a single 3x3 kernel whose taps sum to 1 (brightness is preserved). $\lambda=1$ and $\lambda=2$ give the two Photoshop kernels shown on the slides.
"""),
    code(r"""
def laplacian_demo(img, lams=(1.0, 2.0), blur=1.2, crop=(30, 130, 60, 160), plot=True):
    soft = np.clip(denoise.gaussian(img, blur), 0, 1)
    outs = {'softened input': soft}
    for lam in lams:
        outs[f'lambda = {lam:g}'] = sharpen.sharpen(soft, 'laplacian', lam=lam)
    grads = {k: sharpen.mean_gradient(v) for k, v in outs.items()}
    if plot:
        sl = (slice(crop[0], crop[1]), slice(crop[2], crop[3]))
        show_row([v[sl] for v in outs.values()], [f'{k}\nmean gradient {grads[k]:.3f}' for k in outs], figsize=(12, 4.4))
    return grads

for lam in (1, 2):
    print(f'sharpening_kernel(lam={lam}):\n', sharpen.sharpening_kernel(lam))
_ = laplacian_demo(clean)
"""),
    md(r"""
The kernel centre is $1+4\lambda$ and the four neighbours are $-\lambda$. The mean gradient (a crude sharpness measure) goes up with $\lambda$; the picture gains crispness first and then starts to look harsh, with bright and dark lines along edges.

## Unsharp masking (L5 s15) and why it works

The classic version uses $F = x - G_\sigma * x$: subtract a blurred copy. $\sigma$ chooses which detail size is boosted, whereas the Laplacian always acts at pixel scale. In frequency terms the sharpening filter has gain 1 at zero frequency and more than 1 at high frequencies, which is also where noise lives.
"""),
    code(r"""
def response_demo(plot=True):
    n = 64
    cases = [('laplacian lam 1', dict(lam=1.0, kind='laplacian')), ('laplacian lam 2', dict(lam=2.0, kind='laplacian')),
             ('gaussian sigma 1 lam 1', dict(lam=1.0, kind='gaussian', sigma=1.0)), ('gaussian sigma 2 lam 1', dict(lam=1.0, kind='gaussian', sigma=2.0))]
    resp = {k: sharpen.unsharp_response(n=n, **kw)[0, :n // 2 + 1] for k, kw in cases}
    if plot:
        f = np.linspace(0, 0.5, n // 2 + 1)
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        for k, r in resp.items():
            axs[0].plot(f, r, label=k)
        axs[0].set_xlabel('frequency (cycles / pixel)'); axs[0].set_ylabel('gain'); axs[0].legend(fontsize=7); axs[0].set_title('frequency response'); axs[0].grid(alpha=.3)
        soft = np.clip(denoise.gaussian(clean, 1.2), 0, 1)
        sl = (slice(30, 130), slice(60, 160))
        axs[1].imshow(sharpen.sharpen(soft, 'unsharp', lam=1.0, sigma=1.0)[sl]); axs[1].set_title('unsharp, sigma 1, lam 1'); axs[1].axis('off')
        axs[2].imshow(sharpen.sharpen(soft, 'unsharp', lam=1.0, sigma=3.0)[sl]); axs[2].set_title('unsharp, sigma 3, lam 1'); axs[2].axis('off')
        plt.tight_layout(); plt.show()
    return {k: float(r[-1]) for k, r in resp.items()}

print({k: round(v, 2) for k, v in response_demo().items()}, '<- gain at the Nyquist frequency')
"""),
    md(r"""
All the filters leave flat areas (frequency 0) untouched and boost high frequencies; the Laplacian boosts the most at Nyquist, the Gaussian version with a larger $\sigma$ lifts a wider band, so it looks like a local contrast boost rather than pixel-level crispness.

## Overshoot and noise amplification

Every sharpener has two side effects. Around a step edge it overshoots both plateaus (the halo), and since it amplifies high frequencies it amplifies noise: for white noise the standard deviation is multiplied by $\sqrt{\sum \text{taps}^2}$ of the kernel.
"""),
    code(r"""
def side_effects(lams=(0.0, 0.5, 1.0, 2.0), noise=0.02, plot=True):
    edge = scenes.step_edge(128)
    flat = np.full((128, 128, 3), 0.5, np.float32)
    flat_n = (flat + utils.rng(3).normal(0, noise, flat.shape)).astype(np.float32)
    rows = []
    fig, ax = (plt.subplots(1, 2, figsize=(12, 4)) if plot else (None, [None, None]))
    for lam in lams:
        d, prof = sharpen.edge_profile(sharpen.sharpen(edge, 'laplacian', lam=lam, clip=False))
        os_ = sharpen.overshoot(prof)
        out = sharpen.sharpen(flat_n, 'laplacian', lam=lam, clip=False)
        gain = out.std() / flat_n.std()
        rows.append((lam, os_, gain, sharpen.noise_gain(sharpen.sharpening_kernel(lam))))
        if plot:
            ax[0].plot(d, prof, label=f'lam {lam:g} (overshoot {100 * os_:.0f} %)')
    if plot:
        ax[0].set_xlabel('distance from the edge (px)'); ax[0].set_ylabel('value'); ax[0].legend(fontsize=8); ax[0].set_title('edge profile'); ax[0].grid(alpha=.3)
        ax[1].imshow(np.clip(out[:64, :64] * 1.0, 0, 1)); ax[1].set_title(f'flat grey + noise after lam = {lams[-1]:g}'); ax[1].axis('off')
        plt.tight_layout(); plt.show()
    return rows

rows = side_effects()
print('{:>6s} {:>12s} {:>16s} {:>14s}'.format('lam', 'overshoot', 'noise std gain', 'sqrt(sum k^2)'))
for lam, o, g, th in rows:
    print('{:6.1f} {:11.1f}% {:16.2f} {:14.2f}'.format(lam, 100 * o, g, th))
"""),
    md(r"""
The overshoot grows with $\lambda$ and the measured noise gain matches the kernel formula. A sharpener that is strong enough to look good on a clean picture makes the noise several times larger, which leads to the order of operations below.

## Denoise first, then sharpen

If you sharpen a noisy image you amplify the noise and the denoiser then has to deal with stronger, more structured noise; if you denoise first, the sharpener works on a cleaner signal. The test is the same pair of operations in both orders.
"""),
    code(r"""
def order_demo(noisy, clean, lam=0.4, crop=(30, 130, 60, 160), plot=True):
    dn = lambda x: np.clip(denoise.bilateral(x, 2.0, 0.1), 0, 1)
    sh = lambda x: sharpen.sharpen(x, 'unsharp', lam=lam, sigma=1.5)
    outs = {'denoise, then sharpen': sh(dn(noisy)), 'sharpen, then denoise': dn(sh(noisy))}
    m = {k: (utils.psnr(clean, v), utils.ssim(clean, v)) for k, v in outs.items()}
    if plot:
        sl = (slice(crop[0], crop[1]), slice(crop[2], crop[3]))
        show_row([noisy[sl]] + [v[sl] for v in outs.values()], ['noisy'] + [f'{k}\n{m[k][0]:.1f} dB, SSIM {m[k][1]:.3f}' for k in outs], figsize=(12, 4.4))
    return m

_ = order_demo(noisy, clean)
"""),
    md(r"""
Denoising first gives the cleaner result in both metrics. The reverse order sharpens noise into small blobs that a bilateral filter reads as real texture and therefore keeps. The default camera of notebook 04 follows this order, and it also sharpens after tone mapping, on display data.

## Interactive: denoise and sharpen

Change the noise level and the filter settings and watch the PSNR. The crop is the same region as above.
"""),
    code(r"""
_clean_i = clean[::1, ::1]

def ds_view(noise=0.06, sigma_d=2.0, sigma_r=0.1, lam=0.5, usigma=1.5):
    nz = np.clip(_clean_i + utils.rng(5).normal(0, noise, _clean_i.shape), 0, 1).astype(np.float32)
    den = np.clip(denoise.bilateral(nz, sigma_d, sigma_r), 0, 1)
    out = sharpen.sharpen(den, 'unsharp', lam=lam, sigma=usigma) if lam > 0 else den
    sl = (slice(30, 130), slice(60, 160))
    show_row([nz[sl], den[sl], out[sl]],
             [f'noisy {utils.psnr(_clean_i, nz):.1f} dB', f'bilateral({sigma_d}, {sigma_r}): {utils.psnr(_clean_i, den):.1f} dB', f'+ unsharp({lam}, {usigma}): {utils.psnr(_clean_i, out):.1f} dB'],
             figsize=(12, 4.4))

W.interact(ds_view, noise=W.FloatSlider(min=0.0, max=0.15, step=0.01, value=0.06), sigma_d=W.FloatSlider(min=0.5, max=4, step=0.5, value=2.0),
           sigma_r=W.FloatSlider(min=0.02, max=0.5, step=0.02, value=0.1), lam=W.FloatSlider(min=0, max=3, step=0.25, value=0.5),
           usigma=W.FloatSlider(min=0.5, max=4, step=0.5, value=1.5));
"""),
    md(r"""
## Try it

1. In `tone_compare`, use `scenes.high_contrast(256, 'coffee')` and a stronger `strength`. Where do halos first appear? Try `base='bilateral'` in `tonemap.dodge_burn` to see if they go away.
2. In `sweep_bilateral`, raise the noise to 0.12 and see how the best $\sigma_r$ moves.
3. In `side_effects`, add `threshold=0.02` to the `sharpen.sharpen` call that processes `flat_n`. What does a coring threshold do to noise and to edges?
"""),
    code(r"""
# Exercise 1: local tone mapping of another scene, with the bilateral base layer (edge-preserving, fewer halos)
hc2 = scenes.high_contrast(256, 'coffee')
show_row([tonemap.tonemap(hc2, 'srgb'), tonemap.tonemap(hc2, 'local', strength=0.8),
          tonemap.tonemap(hc2, 'local', strength=0.8, base='bilateral')],
         ['sRGB', 'local, gaussian base', 'local, bilateral base'], figsize=(12, 4))
"""),
]



# =============================================================================
# 04  JPEG and the full pipeline
# =============================================================================
NB04 = [
    md(r"""
# 04. JPEG and the full camera

The last box of the camera pixel pipeline (L2 s3) stores the picture as a JPEG. We open that box first: colour transform, chroma subsampling, 8x8 DCT, quantisation and entropy coding (L5 s19-22), using `isp.jpeg`, which is a from-scratch codec checked against Pillow. Then we put all thirteen stages together with `isp.pipeline`, look at what each one does to the picture, measure it against the ground truth, change parameters, switch stages off and swap their order. Theory: `docs/07_compression.md`, `docs/00_camera_pipeline.md`, `docs/08_pipeline_experiments.md`.
"""),
    code(HEADER + r"""
from isp import jpeg, display, pipeline as P
import ipywidgets as W

clean = srgb(scenes.load('astronaut', 256))          # display-referred photo to feed the codec
print('stages of the default camera:', P.default_pipeline().names)
"""),
    md(r"""
## Colour transform and chroma subsampling (L5 s20)

JPEG first converts RGB to YCbCr: one brightness plane and two colour-difference planes. The eye resolves colour detail poorly (notebook 02), so the colour planes are stored at lower resolution. `4:2:0` keeps one chroma sample per 2x2 pixels.
"""),
    code(r"""
def subsampling_demo(img, quality=90, plot=True):
    ycc = utils.rgb_to_ycbcr(img)
    rows = {}
    outs = {}
    for mode in ('4:4:4', '4:2:2', '4:2:0'):
        out, info = jpeg.roundtrip(img, quality, mode)
        outs[mode] = out
        rows[mode] = (info['nbytes'], info['psnr'])
    if plot:
        fig, axs = plt.subplots(1, 4, figsize=(15, 3.6))
        for ax, k, t in zip(axs[:3], range(3), ['Y (luma)', 'Cb', 'Cr']):
            ax.imshow(ycc[..., k], cmap='gray', vmin=0, vmax=1); ax.set_title(t); ax.axis('off')
        sl = (slice(0, 48), slice(0, 48))                       # flag stripes: saturated colour edges
        axs[3].imshow(np.concatenate([img[sl], outs['4:2:0'][sl]], axis=1), interpolation='nearest')
        axs[3].set_title('flag detail: original | 4:2:0, q=%d' % quality); axs[3].axis('off')
        plt.tight_layout(); plt.show()
    return rows

rows = subsampling_demo(clean)
for k, (nb, ps) in rows.items():
    print(f'{k}: {nb:6d} bytes, PSNR {ps:.1f} dB')
"""),
    md(r"""
Y carries the picture, Cb and Cr look like soft sketches. Going from 4:4:4 to 4:2:0 saves a good share of the file for a small PSNR change; the cost shows up only on thin saturated colour edges such as the red and blue flag stripes (right panel).

## One 8x8 block through DCT and quantisation (L5 s20)

The DCT re-expresses a block as a sum of 64 cosine patterns; quantisation divides each coefficient by a table entry and rounds, which is the only lossy step. High-frequency entries have big divisors, so most of them become zero.
"""),
    code(r"""
def block_demo(img, quality=50, plot=True):
    gray = utils.luminance(img)
    # pick the 8x8 block with the most structure (largest variance)
    v = jpeg.split_blocks(gray[:gray.shape[0] // 8 * 8, :gray.shape[1] // 8 * 8]).reshape(-1, 64).var(axis=1)
    k = int(np.argmax(v)); nbx = gray.shape[1] // 8
    blk = (k // nbx, k % nbx)
    r = jpeg.block_before_after(gray, blk, quality)
    if plot:
        fig, axs = plt.subplots(2, 3, figsize=(14, 8))
        axs[0, 0].imshow(r['pixels'], cmap='gray', vmin=0, vmax=255); axs[0, 0].set_title(f'block {blk}: pixels')
        axs[0, 1].imshow(np.log1p(np.abs(r['coefs'])), cmap='magma'); axs[0, 1].set_title('log |DCT coefficients| (DC top left)')
        axs[0, 2].imshow(np.log1p(r['q']), cmap='viridis'); axs[0, 2].set_title(f'log quantisation table (quality {quality})')
        axs[1, 0].imshow(np.abs(r['quantised']) > 0, cmap='gray'); axs[1, 0].set_title(f"non-zero after quantisation: {int((r['quantised'] != 0).sum())} of 64")
        axs[1, 1].imshow(r['recon'], cmap='gray', vmin=0, vmax=255); axs[1, 1].set_title('reconstruction')
        im = axs[1, 2].imshow(r['error'], cmap='coolwarm', vmin=-40, vmax=40); axs[1, 2].set_title(f"error, rms {np.sqrt(np.mean(r['error'] ** 2)):.1f} grey levels")
        for ax in axs.ravel(): ax.axis('off')
        plt.tight_layout(); plt.show()
        B = jpeg.dct_basis()
        fig, axs = plt.subplots(8, 8, figsize=(4.5, 4.5))
        for u in range(8):
            for w in range(8):
                axs[u, w].imshow(B[u, w], cmap='gray'); axs[u, w].axis('off')
        plt.suptitle('the 64 DCT basis functions', fontsize=9); plt.tight_layout(); plt.show()
    return r

r = block_demo(clean)
print('quantised block (integers):\n', r['quantised'])
"""),
    md(r"""
The energy of a real image block concentrates in the low frequencies (top left), so after quantisation only a handful of integers survive, and the reconstruction is close to the block with an error spread over its pixels. A lower quality number means a bigger table, more zeros and a larger error.

## Entropy coding

The integers are read in zig-zag order (low to high frequency), so the zeros pile up at the end. Runs of zeros and the size of the next value form `(run, size)` symbols, which Huffman coding stores with short codes for common symbols. The DC coefficients are coded as differences between neighbouring blocks.
"""),
    code(r"""
def entropy_demo(img, quality=75, plot=True):
    out, info = jpeg.roundtrip(img, quality)
    return info

blk = r['quantised']
zz = jpeg.zigzag(blk)
print('zig-zag order of the block above:', zz.tolist())
syms = jpeg.rle_encode_ac(zz[1:])
print('(run, size, amplitude bits) symbols, first 8 of %d:' % len(syms), syms[:8], '... the last symbol is end-of-block (0, 0, 0)' if syms[-1] == (0, 0, 0) else '')

info = entropy_demo(clean)
bits = info['bits']
fig, ax = plt.subplots(figsize=(6, 3.2))
bottom = np.zeros(3)
for part, col in (('DC', 'tab:blue'), ('AC', 'tab:orange')):
    vals = np.array([bits[c][part] for c in ('Y', 'Cb', 'Cr')]) / 8 / 1024
    ax.bar(['Y', 'Cb', 'Cr'], vals, bottom=bottom, label=part, color=col); bottom += vals
ax.set_ylabel('KiB'); ax.legend(); ax.set_title('where the bits go (q=75, 4:2:0)'); plt.tight_layout(); plt.show()
print(f"{info['nbytes']} bytes = {info['bpp']:.2f} bits per pixel, {info['ratio']:.1f}:1 against 24 bpp, PSNR {info['psnr']:.1f} dB")
"""),
    md(r"""
Almost all of the bits are AC coefficients of Y: that is where the picture detail (and, in a noisy picture, the noise) lives. The two chroma planes together take a small share, which is the payoff of subsampling and heavy chroma quantisation.

## Quality sweep and rate-distortion (L5 s19-20)

Our codec estimates the file size from optimal Huffman tables. Real libjpeg (through Pillow) is the reference: if both land on the same rate-distortion curve the implementation is trustworthy.
"""),
    code(r"""
def quality_sweep(img, qualities=(5, 10, 20, 35, 50, 65, 80, 90, 95), plot=True):
    own, ref = [], []
    for q in qualities:
        out, info = jpeg.roundtrip(img, q)
        own.append((info['bpp'], info['psnr']))
        o2, nb = jpeg.pillow_roundtrip(img, q)
        ref.append((8.0 * nb / (img.shape[0] * img.shape[1]), utils.psnr(img, o2)))
    if plot:
        fig, axs = plt.subplots(1, 4, figsize=(15, 3.8))
        axs[0].plot(*zip(*own), 'o-', label='isp.jpeg'); axs[0].plot(*zip(*ref), 's--', label='Pillow (libjpeg)')
        axs[0].set_xlabel('bits per pixel'); axs[0].set_ylabel('PSNR (dB)'); axs[0].legend(); axs[0].grid(alpha=.3); axs[0].set_title('rate-distortion')
        for ax, q in zip(axs[1:], (10, 50, 90)):
            out, info = jpeg.roundtrip(img, q)
            ax.imshow(out[30:130, 60:160], interpolation='nearest'); ax.axis('off')
            ax.set_title(f"q={q}: {info['bpp']:.2f} bpp, {info['psnr']:.1f} dB, blockiness {jpeg.blockiness(out):.2f}", fontsize=9)
        plt.tight_layout(); plt.show()
    return own, ref

own, ref = quality_sweep(clean)
print('max |bpp difference| between our codec and libjpeg: %.1f %%' % (100 * max(abs(a[0] / b[0] - 1) for a, b in zip(own, ref))))
"""),
    md(r"""
The two curves nearly coincide, so the codec does what libjpeg does. Along the curve, going from quality 90 to 50 buys a lot of bits for a moderate PSNR drop, and below about 20 the 8x8 grid becomes visible as blocks (the blockiness score rises above 1).

## Generation loss (L5 s19)

Every time a JPEG is opened, edited and saved again, it is encoded once more. If the picture lands on exactly the same 8x8 grid, the second encode is almost a projection and little more is lost. If the content shifts by a few pixels (a crop, a resize), the blocks cut through different content and every generation adds new error.
"""),
    code(r"""
def generation_demo(img, n=10, quality=75, shifts=(0, 3), plot=True):
    res = {s: jpeg.generation_loss(img, n=n, quality=quality, shift=s) for s in shifts}
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        for s, r in res.items():
            axs[0].plot([g['gen'] for g in r], [g['psnr'] for g in r], 'o-', label=f'shift {s} px')
            axs[1].plot([g['gen'] for g in r], [g['nbytes'] for g in r], 'o-', label=f'shift {s} px')
        axs[0].set_xlabel('generation'); axs[0].set_ylabel('PSNR vs the (shifted) original (dB)'); axs[0].legend(); axs[0].grid(alpha=.3)
        axs[1].set_xlabel('generation'); axs[1].set_ylabel('bytes'); axs[1].legend(); axs[1].grid(alpha=.3)
        plt.tight_layout(); plt.show()
    return res

res_g = generation_demo(clean[:128, :128])
for s, r in res_g.items():
    print(f'shift {s}: PSNR generation 1 -> 10: {r[0]["psnr"]:.2f} -> {r[-1]["psnr"]:.2f} dB')
"""),
    md(r"""
Re-saving on the same grid costs almost nothing after the first generation; the shifted copy keeps losing quality. The practical lesson is to keep the original and re-export from it rather than editing a JPEG over and over.

## The whole camera: `simulate()` (L2 s3, s34)

`P.simulate` loads a ground-truth scene and runs 13 stages: scene, light, CFA, exposure, ADC, black level, hot pixels, white balance, demosaic, denoise, tone map, sharpen, JPEG. Each record holds the stage output, its meta data and metrics against the truth. `display.render` turns any of them into something you can look at, including the Bayer mosaics and the DN values.
"""),
    code(r"""
res = P.simulate('astronaut', 256)
print('{:14s} {:10s} {:9s} {:>12s} {:>8s} {:>6s} {:>8s}'.format('stage', 'group', 'domain', 'shape', 'PSNR', 'SSIM', 'ms'))
for rec in res:
    m = rec.metrics
    print('{:14s} {:10s} {:9s} {:>12s} {:8.2f} {:6.3f} {:8.1f}'.format(rec.name, rec.group, rec.meta['domain'], str(rec.image.shape), m['psnr'], m.get('ssim', float('nan')), 1000 * rec.seconds))

def filmstrip(result, truth, plot=True):
    imgs = [(f'{i + 1}. {r.label}', display.render(r)) for i, r in enumerate(result)]
    final = display.render(result.final)
    imgs.append(('ground truth (sRGB)', utils.to_uint8(srgb(truth))))
    imgs.append(('final, 64 x 64 crop zoomed', display.crop_zoom(final, (30, 94, 100, 164), 4)))
    if plot:
        fig, axs = plt.subplots(3, 5, figsize=(15, 9.6))
        for ax, (t, im) in zip(axs.ravel(), imgs):
            ax.imshow(im, interpolation='nearest'); ax.set_title(t, fontsize=9); ax.axis('off')
        plt.tight_layout(); plt.show()
    return imgs

_ = filmstrip(res, scenes.load('astronaut', 256))
"""),
    md(r"""
Read the strip left to right and top to bottom. The picture starts with a strong orange cast (the 3200 K lamp), becomes a noisy, coloured mosaic at the sensor stages, is cleaned up and colour-corrected by the processing stages, and ends as a JPEG that differs from the truth mainly at fine detail and edges. White balance is the biggest visible step; the mosaic stages look dark because the sensor is run at high ISO and scene white sits at a fraction of the range.
"""),
    code(r"""
def quality_trace(result, plot=True):
    names = [r.label for r in result]
    keys = [('psnr', 'PSNR vs ground truth (dB)'), ('sigma', 'estimated noise std'), ('cast_deg', 'colour cast (degrees)')]
    vals = {k: np.array([r.metrics.get(k, np.nan) for r in result], float) for k, _ in keys}
    vals['psnr'][~np.isfinite(vals['psnr'])] = np.nan
    if plot:
        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        cols = {g: P.GROUP_INFO[g]['colour'] for g in P.GROUP_INFO}
        for ax, (k, t) in zip(axs, keys):
            ax.bar(range(len(names)), vals[k], color=[cols[r.group] for r in result])
            ax.set_xticks(range(len(names)), names, rotation=70, ha='right', fontsize=8); ax.set_title(t); ax.grid(axis='y', alpha=.3)
        plt.tight_layout(); plt.show()
    return vals

tr = quality_trace(res)
i_wb = res.names.index('white_balance')
print('PSNR before / after white balance: %.1f -> %.1f dB; colour cast %.1f -> %.1f degrees' % (
    tr['psnr'][i_wb - 1], tr['psnr'][i_wb], tr['cast_deg'][i_wb - 1], tr['cast_deg'][i_wb]))
print('noise estimate before / after denoise: %.4f -> %.4f' % (tr['sigma'][res.names.index('demosaic')], tr['sigma'][res.names.index('denoise')]))
"""),
    md(r"""
The cast bars show white balance removing the light colour in one step and the PSNR jumping with it; noise appears at the sensor stages and drops at the denoiser; the PSNR later falls where the tone curve amplifies the remaining shadow noise. Stages that do not remove a defect (sharpen, JPEG) trade a small amount of PSNR for appearance or file size.

## Edit a stage and re-run only what changed

Every stage has a dictionary of parameters. If you pass the previous result, `Pipeline.run` copies the prefix of stages whose parameters and input did not change, so moving the sharpen strength recomputes only the last two stages.
"""),
    code(r"""
def incremental_demo(base, lam=1.0, repeat=3):
    t_full, t_inc = [], []
    for _ in range(repeat):
        t0 = time.perf_counter(); full = P.simulate('astronaut', 256, sharpen={'lam': lam}); t_full.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); inc = P.simulate('astronaut', 256, sharpen={'lam': lam}, previous=base); t_inc.append(time.perf_counter() - t0)
    return full, inc, min(t_full), min(t_inc)

full, inc, tf, ti = incremental_demo(res)
print('stages reused from the previous run:', [r.name for r in inc if r.reused])
print('recomputed:', [r.name for r in inc if not r.reused])
print('identical final image to a full run:', np.array_equal(full.final.image, inc.final.image))
print(f'full run {1000 * tf:.0f} ms, incremental {1000 * ti:.0f} ms; final PSNR {res.final.metrics["psnr"]:.2f} -> {inc.final.metrics["psnr"]:.2f} dB with lam = 1')
"""),
    md(r"""
The incremental result is identical to a full run, only faster. This is what the interactive widgets below rely on to stay responsive.

## Switching stages off

Each processing stage is there for a reason. Switching one off at a time shows how much of the final quality it buys (`black_level` cannot be skipped because the later stages expect linear data).
"""),
    code(r"""
def ablation(stages=('hot_pixels', 'white_balance', 'denoise', 'sharpen', 'jpeg'), plot=True):
    base = P.simulate('astronaut', 256)
    rows = {'default camera': base}
    for st in stages:
        rows[f'{st} off'] = P.simulate('astronaut', 256, enabled={st: False})
    rows['demosaic: nearest'] = P.simulate('astronaut', 256, demosaic={'method': 'nearest'})
    table = {k: (r.final.metrics['psnr'], r.final.metrics['ssim'], r.final.meta.get('nbytes', np.nan)) for k, r in rows.items()}
    if plot:
        fig, axs = plt.subplots(1, 2, figsize=(14, 4))
        names = list(table)
        axs[0].barh(names, [v[0] for v in table.values()], color=['k'] + ['tab:blue'] * (len(names) - 1)); axs[0].set_xlabel('final PSNR (dB)'); axs[0].invert_yaxis()
        axs[1].barh(names, [v[2] for v in table.values()], color=['k'] + ['tab:orange'] * (len(names) - 1)); axs[1].set_xlabel('JPEG bytes'); axs[1].invert_yaxis()
        plt.tight_layout(); plt.show()
        pick = ['default camera', 'white_balance off', 'denoise off', 'demosaic: nearest']
        show_row([display.crop_zoom(display.render(rows[k].final), (30, 94, 100, 164), 3) for k in pick], pick, figsize=(14, 4))
    return table

table = ablation()
print('{:22s} {:>8s} {:>7s} {:>8s}'.format('variant', 'PSNR', 'SSIM', 'bytes'))
for k, (p, s, b) in table.items():
    print('{:22s} {:8.2f} {:7.3f} {:8.0f}'.format(k, p, s, b))
"""),
    md(r"""
White balance is the largest single contribution, because without it the whole picture keeps the lamp's cast. Denoising is worth a few dB and makes the JPEG smaller, since the encoder no longer spends bits on noise. Sharpening is the odd one out: it can lower PSNR and raise the file size, and it is there because the picture looks better.

## Reordering stages

The slide says the processing order may vary (L2 s3). Some orders are forbidden by physics, others legal but worse. We test the legal ones with the same scene and seed, and show what the pipeline says to an illegal one.
"""),
    code(r"""
def order_experiment(plot=True):
    default = P.default_pipeline(white_balance={'method': 'true'})              # oracle gains so the order, not the estimate, is what changes
    variants = {'default order': default.copy(),
                'denoise before demosaic': default.copy().move('denoise', before='demosaic'),
                'white balance after demosaic': default.copy().move('white_balance', after='demosaic'),
                'sharpen before denoise (linear data)': default.copy().move('sharpen', before='denoise')}
    out = {k: P.simulate('astronaut', 256, pipeline=pl) for k, pl in variants.items()}
    ref = display.render(out['default order'].final).astype(float)
    table = {k: (r.final.metrics['psnr'], r.final.metrics['ssim'], r.final.meta['nbytes'], float(np.abs(display.render(r.final).astype(float) - ref).mean()))
             for k, r in out.items()}
    if plot:
        show_row([display.crop_zoom(display.render(r.final), (30, 94, 100, 164), 3) for r in out.values()],
                 [f'{k}\n{table[k][0]:.1f} dB' for k in out], figsize=(15, 4.4))
    return table

table = order_experiment()
print('{:38s} {:>8s} {:>7s} {:>8s} {:>10s}'.format('order', 'PSNR', 'SSIM', 'bytes', 'mean diff'))
for k, (p, s, b, d) in table.items():
    print('{:38s} {:8.2f} {:7.3f} {:8d} {:10.2f}'.format(k, p, s, b, d))

bad = P.default_pipeline().move('adc', before='exposure')                         # electrons do not exist yet
try:
    P.simulate('astronaut', 128, pipeline=bad)
except ValueError as e:
    print('\nadc before exposure ->', e)
"""),
    md(r"""
Denoising before demosaicing is worse: each colour plane is a quarter or half of the pixels and the filter cannot use the other channels. Doing the white balance after the demosaic changes the result because the median demosaicer works on colour differences, so it prefers to see balanced channels. Sharpening noisy linear data before the denoiser loses against the default order. The sensor half of the chain (exposure, ADC, black level) has only one valid order, and the stage tells you so by name.

## Interactive: the whole camera

Pick the ISO, the white-balance method, the denoiser's range sigma (0 switches it off) and the JPEG quality. The simulation uses `previous=`, so changing only the JPEG quality or the sharpen strength is quick. The title reports the final PSNR, the file size and how many stages were recomputed.
"""),
    code(r"""
_cache = {}

def camera_view(iso=3200, wb_method='white_patch', denoise_sigma_r=0.05, jpeg_quality=85):
    ov = dict(adc={'iso': iso}, white_balance={'method': wb_method}, jpeg={'quality': jpeg_quality})
    if denoise_sigma_r <= 0:
        ov['enabled'] = {'denoise': False}
    else:
        ov['denoise'] = {'sigma_d': 1.5, 'sigma_r': denoise_sigma_r}
    r = P.simulate('astronaut', 192, previous=_cache.get('prev'), **ov)
    _cache['prev'] = r
    n_new = sum(not s.reused for s in r)
    img = display.render(r.final)
    fig, axs = plt.subplots(1, 2, figsize=(10, 4.4), gridspec_kw=dict(width_ratios=[1, 1]))
    axs[0].imshow(img); axs[0].axis('off')
    axs[0].set_title(f"{r.final.metrics['psnr']:.1f} dB, SSIM {r.final.metrics['ssim']:.2f}, {r.final.meta['nbytes']} B, {n_new} of {len(r)} stages recomputed", fontsize=9)
    axs[1].imshow(display.crop_zoom(img, (20, 84, 70, 134), 4), interpolation='nearest'); axs[1].axis('off'); axs[1].set_title('crop', fontsize=9)
    plt.tight_layout(); plt.show()

W.interact(camera_view, iso=W.SelectionSlider(options=[100, 200, 400, 800, 1600, 3200, 6400, 12800], value=3200),
           wb_method=['none', 'gray_world', 'white_patch', 'true'], denoise_sigma_r=W.FloatSlider(min=0, max=0.2, step=0.01, value=0.05),
           jpeg_quality=W.IntSlider(min=5, max=100, step=5, value=85));
"""),
    md(r"""
## Try it

1. Use `P.simulate('astronaut', 256, adc={'iso': 200}, exposure={'exposure': 0.2})` to get a low-noise capture at about the same brightness as the default. How many dB does it gain over the default, and which stages hold the gain?
2. Run the camera on `scenes.sunset(256)` with the three white-balance methods and compare the final PSNR (use `method='true'` as the reference).
3. Move `sharpen` before `tonemap` on a copy of the pipeline (`move('sharpen', before='tonemap')`). The data is now linear; what happens to the halos and to the PSNR?
"""),
    code(r"""
# Exercises 1-3: edit and re-run
a = P.simulate('astronaut', 256)
b = P.simulate('astronaut', 256, adc={'iso': 200}, exposure={'exposure': 0.2})
print('default %.2f dB | low ISO, long exposure %.2f dB' % (a.final.metrics['psnr'], b.final.metrics['psnr']))
for m in ('gray_world', 'white_patch', 'true'):
    r = P.simulate('sunset', 256, white_balance={'method': m})
    print('sunset, %-11s %.2f dB, wb error %s deg' % (m, r.final.metrics['psnr'], r['white_balance'].metrics.get('wb_angular_error', r['white_balance'].meta.get('wb_angular_error'))))
pl = P.default_pipeline().move('sharpen', before='tonemap')
print('sharpen before tonemap: %.2f dB' % P.simulate('astronaut', 256, pipeline=pl).final.metrics['psnr'])
"""),
]


NOTEBOOKS = {
    "01_sensor_and_noise.ipynb": "NB01",
    "02_color_demosaic_white_balance.ipynb": "NB02",
    "03_tone_denoise_sharpen.ipynb": "NB03",
    "04_jpeg_and_full_pipeline.ipynb": "NB04",
}

if __name__ == "__main__":
    for fname, var in NOTEBOOKS.items():
        if var in globals():
            build(fname, globals()[var])
