"""Figures and numbers for docs/01_sensor_and_noise.md  ->  docs/figures/sensor_*.png"""
import os
import sys
from dataclasses import replace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from isp import scenes, sensor as S                      # noqa: E402
from isp.utils import luminance, psnr, rng                # noqa: E402

OUT = os.path.join(ROOT, "docs", "figures")
os.makedirs(OUT, exist_ok=True)
DPI = 90


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name), dpi=DPI)
    plt.close(fig)


def photo(size=128):
    return np.clip(luminance(scenes.load("astronaut", size)), 0, 1).astype(np.float64)


def show(ax, img, title):
    ax.imshow(np.clip(img, 0, 1), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# ---------------------------------------------------------------------------
# 1. noise-type gallery (each source alone, deliberately exaggerated)
# ---------------------------------------------------------------------------
def fig_gallery():
    n = 128
    t, flux, qe = 0.01, 2e4, 0.6
    r = rng(0)
    p = replace(S.PRESETS["phone"], dark_current=1500.0, dsnu=800.0, prnu=0.10,
                hot_pixel_fraction=0.03, hot_pixel_rate=4000.0, read_noise=8.0, qe=qe)
    fp = S.fixed_pattern((n, n), p)
    full = flux * qe * t                      # electrons that scene white produces (= display white)
    scenes_ = {"flat grey (0.3)": np.full((n, n), 0.3), "photo": photo(n)}
    names = ["photon (shot)", "dark current", "hot pixels", "fixed pattern", "read"]
    fig, axs = plt.subplots(2, 6, figsize=(14, 5))
    stats = {}
    for row, (sname, img) in enumerate(scenes_.items()):
        mean_sig = img * full
        layers = {
            "photon (shot)": r.poisson(mean_sig) - mean_sig,
            "dark current": r.poisson(np.full(img.shape, p.dark_current * t)) - p.dark_current * t,
            "hot pixels": fp["hot"] * p.hot_pixel_rate * t,
            "fixed pattern": fp["prnu"] * mean_sig + (fp["dark"] - p.dark_current - fp["hot"] * p.hot_pixel_rate) * t,
            "read": r.standard_normal(img.shape) * p.read_noise,
        }
        show(axs[row, 0], img, f"clean: {sname}")
        for k, nm in enumerate(names):
            show(axs[row, k + 1], (mean_sig + layers[nm]) / full, nm)
            if row == 0:
                stats[nm] = float(np.std(layers[nm]))
    save(fig, "sensor_noise_gallery.png")
    print("[gallery] std in electrons on flat grey 0.3 (white = %.0f e-):" % full)
    for k, v in stats.items():
        print(f"    {k:15s} {v:7.2f} e-")
    print(f"    theory: shot sqrt({0.3*full:.0f}) = {np.sqrt(0.3*full):.2f}, dark sqrt({p.dark_current*t:.0f}) = "
          f"{np.sqrt(p.dark_current*t):.2f}, read {p.read_noise}")


# ---------------------------------------------------------------------------
# 2. SNR vs exposure: theory vs Monte Carlo
# ---------------------------------------------------------------------------
def fig_snr():
    cfg = [("retiga_4000r", 1000.0, "C0"), ("aptina_mt9p031", 1000.0 / 11, "C1")]
    ts = np.logspace(-2, 1.3, 120)
    tm = np.logspace(-2, 1.3, 9)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    print("[snr] Monte Carlo (48 frames of 32x32, FPN off) vs theory")
    for name, flux, c in cfg:
        p = S.clean_params(S.PRESETS[name], bits=16)
        ax.loglog(ts, S.snr_theory(flux, p.qe, ts, p.dark_current, p.read_noise), c, label=f"{p.name} theory")
        mc, th = [], []
        for t in tm:
            m = S.measure_snr(p, 1.0, t, n=48, shape=(32, 32), flux_at_white=flux, seed=10)
            mc.append(m["snr"]); th.append(float(S.snr_theory(flux, p.qe, t, p.dark_current, p.read_noise)))
        ax.loglog(tm, mc, "o", color=c, mfc="none", label=f"{p.name} Monte Carlo")
        err = np.abs(np.array(mc) / np.array(th) - 1)
        print(f"    {p.name}: max |MC/theory - 1| = {100*err.max():.1f} %  (t = {tm[0]:.2f}..{tm[-1]:.1f} s)")
        print(f"      theory at t=1 s: {S.snr_theory(flux, p.qe, 1.0, p.dark_current, p.read_noise):.2f}:1")
    ax.set_xlabel("exposure time t (s)"); ax.set_ylabel("SNR (mean / std)")
    ax.set_title("SNR vs exposure (Retiga: 1000 ph/px/s, Aptina: 1000/11)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    save(fig, "sensor_snr_vs_exposure.png")
    for name, flux in (("retiga_4000r", 1000.0), ("aptina_mt9p031", 1000.0 / 11)):
        p = S.PRESETS[name]
        print(f"    L2 s32 {p.name}: SNR(1 s) = {S.snr_theory(flux, p.qe, 1.0, p.dark_current, p.read_noise):.1f}")
        for t in (1, 1 / 60):
            print(f"    L2 s33 {p.name} DR(t={t:.4f}) = {S.dynamic_range(p.full_well, p.dark_current, t, p.read_noise):.0f}:1")


# ---------------------------------------------------------------------------
# 3. exposure vs ISO at equal output brightness
# ---------------------------------------------------------------------------
def fig_iso():
    n = 128
    p = S.clean_params(replace(S.PRESETS["dslr"], read_noise=15.0))
    img = 0.08 * photo(n)
    flux = 1e5
    cases = [("long exposure, ISO 100\n(t = 1/2 s)", 0.5, 100, 1.0),
             ("short exposure, ISO 800\n(t = 1/16 s)", 1 / 16, 800, 1.0),
             ("short exposure, ISO 100,\nbrightened x8 in software", 1 / 16, 100, 8.0)]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.4))
    print("[iso] equal-brightness captures of a dark scene (dslr, read noise 15 e-); SNR of flat patch 0.08")
    for ax, (title, t, iso, boost) in zip(axs, cases):
        dn, meta = S.capture(img, t, iso, p, flux_at_white=flux, seed=3)
        est = (dn - p.black_level) / p.gain(iso) / (flux * p.qe * t)     # scene value estimate
        show(ax, est * 12, title)
        m = S.measure_snr(p, 0.08, t, iso, n=24, shape=(32, 32), flux_at_white=flux, seed=20)
        print(f"    t={t:.4f} s ISO {iso}: e-/pixel={m['mean_e']:.0f}  SNR={m['snr']:.2f}")
    save(fig, "sensor_exposure_vs_iso.png")


# ---------------------------------------------------------------------------
# 4. dark-frame subtraction
# ---------------------------------------------------------------------------
def fig_dark():
    n = 128
    p = S.PRESETS["aptina_mt9p031"]
    img = 0.5 * photo(n)
    flux, t = 400.0, 4.0
    raw, meta = S.capture(img, t, 100, p, flux_at_white=flux, seed=5)
    ref = p.gain(100) * img * flux * p.qe * t + p.black_level       # noise-free expected DN
    one = S.dark_frame((n, n), t, 100, p, seed=100)
    master = S.master_dark((n, n), t, 100, p, n=16, seed=100)
    s1 = S.subtract_dark(raw, one, p.black_level)
    s16 = S.subtract_dark(raw, master, p.black_level)
    rng_dn = p.white_level - p.black_level
    f = lambda a: S.normalize(a, meta, clip=True) * 8.0
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.8))
    show(axs[0], f(raw), "raw capture (t = 4 s)")
    show(axs[1], f(master) * 4, "master dark (16 frames, extra x4)")
    show(axs[2], f(s1), "minus 1 dark frame")
    show(axs[3], f(s16), "minus 16-frame master dark")
    save(fig, "sensor_dark_subtraction.png")
    e = lambda a: float(np.sqrt(np.mean((a - ref) ** 2)))
    print("[dark] RMS DN error vs the noise-free expected image: "
          f"raw {e(raw):.2f}, 1 dark {e(s1):.2f}, 16-frame master {e(s16):.2f}  (irreducible shot noise of signal+dark ~ {p.gain()*np.sqrt(np.mean(img)*flux*p.qe*t+p.dark_current*t):.1f} DN)")
    hot = S.fixed_pattern((n, n), p)["hot"]
    print(f"    hot pixels: {int(hot.sum())}; raw mean excess {np.mean(raw[hot]-np.median(raw)):.0f} DN -> "
          f"{np.mean(s16[hot]-np.median(s16)):.0f} DN after subtraction")


# ---------------------------------------------------------------------------
# 5. frame averaging
# ---------------------------------------------------------------------------
def fig_average():
    n = 128
    p = S.clean_params(S.PRESETS["aptina_mt9p031"], bits=14)
    img = 0.3 * photo(n)
    flux, t = 3000.0, 0.02
    frames = [S.capture(img, t, 100, p, flux_at_white=flux, seed=k)[0] for k in range(256)]
    meta = dict(black_level=p.black_level, white_level=p.white_level)
    ref = S.average_frames(frames)                     # 256-frame average ~ ground truth
    lin = lambda a: S.normalize(a, meta, clip=False)
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
    print("[average] noise std (DN) vs the 256-frame mean, and sqrt(M) prediction")
    s_one = None
    for ax, M in zip(axs, (1, 4, 16)):
        # independent groups of M frames, so the residual is not correlated with ref
        avg = S.average_frames(frames[:M])
        resid = avg - S.average_frames(frames[128:])
        s = float(np.std(resid))
        s_one = s_one or s
        show(ax, lin(avg) * 3, f"average of {M} frame(s)")
        print(f"    M={M:2d}: std {s:.2f} DN  ratio to M=1: {s_one/s:.2f}  (sqrt(M) = {np.sqrt(M):.2f})  "
              f"PSNR {psnr(lin(avg), lin(ref)):.1f} dB")
    save(fig, "sensor_averaging.png")


# ---------------------------------------------------------------------------
# 6. photon transfer curve
# ---------------------------------------------------------------------------
def fig_ptc():
    p = S.clean_params(S.PRESETS["dslr"], bits=14, dark_current=0.2)
    vals = np.concatenate([[0.0], np.logspace(-3, 1.2, 34)])
    t, flux = 0.2, 1e5
    ptc = S.photon_transfer(p, t, 100, vals, (64, 64), flux, seed=0)
    mean, var = ptc["mean"], ptc["var"]
    g = p.gain(100)
    mid = (mean > 50 * g * 20) & (mean < 0.6 * g * p.full_well)
    slope = np.polyfit(mean[mid], var[mid], 1)[0]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    k = mean > 0
    ax.loglog(mean[k], var[k], "o", ms=4, label="simulated")
    xx = np.logspace(np.log10(mean[k].min()), np.log10(mean.max()), 100)
    ax.loglog(xx, g * xx + (g * p.read_noise) ** 2, "k--", lw=1, label="g x + (g N_r)^2")
    ax.loglog(xx, g * xx, ":", color="gray", label="shot only: slope 1")
    ax.set_xlabel("mean signal (DN above black)"); ax.set_ylabel("temporal variance (DN$^2$)")
    ax.set_title("Photon-transfer curve (DSLR preset, ISO 100)")
    ax.grid(True, which="both", alpha=0.3); ax.legend(fontsize=8)
    save(fig, "sensor_photon_transfer.png")
    floor = float(np.sqrt(var[0]) / g)
    print(f"[ptc] fitted slope {slope:.3f} DN/e-  vs true gain {g:.3f};  read noise from darkest point "
          f"{floor:.2f} e- (true {p.read_noise});  variance peaks at {var.max():.0f} DN^2 "
          f"then collapses at full well")


# ---------------------------------------------------------------------------
# 7. quantisation at low signal
# ---------------------------------------------------------------------------
def fig_quant():
    n = 256
    p = S.clean_params(S.PRESETS["dslr"], bits=8, black_level=0, dark_current=0.0, read_noise=0.4,
                       full_well=30000, qe=0.4)
    ramp = np.tile(np.linspace(0.0, 0.03, n), (n // 2, 1))
    flux, t = 1e5, 0.5
    cases = [("ISO 100", 100, 1600 / 100), ("ISO 1600", 1600, 1.0)]
    fig, axs = plt.subplots(2, 2, figsize=(11, 5.5))
    print("[quant] 8-bit ADC, dark ramp (0..0.03 of white), same exposure")
    for j, (title, iso, boost) in enumerate(cases):
        dn, meta = S.capture(ramp, t, iso, p, flux_at_white=flux, seed=2)
        lin = S.normalize(dn, meta, clip=True) * boost * 2.5
        show(axs[0, j], lin, f"{title}  (shown at equal brightness)")
        axs[1, j].plot(dn[n // 4], lw=0.8)
        axs[1, j].set_xlabel("column"); axs[1, j].set_ylabel("DN"); axs[1, j].set_title(f"{title}: one row", fontsize=9)
        print(f"    {title}: {len(np.unique(dn))} distinct output levels, max DN {int(dn.max())}")
    save(fig, "sensor_quantisation.png")


if __name__ == "__main__":
    fig_gallery()
    fig_snr()
    fig_iso()
    fig_dark()
    fig_average()
    fig_ptc()
    fig_quant()
