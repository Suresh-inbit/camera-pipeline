"""Figures for docs/05_denoising.md (lecture 5, denoising).  Headless; prints every number
quoted in the doc.  Run from the project root:  python scripts/make_figures_denoise.py"""
import sys, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isp import denoise as D, scenes
from isp.utils import psnr, ssim, rng, linear_to_srgb, rgb_to_ycbcr, ycbcr_to_rgb

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 90
B = 0.01                                    # read-noise floor (std at zero signal)
LEVELS = {"low": 0.003, "mid": 0.01, "high": 0.03}   # shot-noise gain a: var = a s + b^2


def add_noise(img, a=0.01, b=B, seed=0):
    """Poisson-Gaussian noise on linear data: var = a * signal + b^2."""
    s = np.clip(img, 0, None)
    return (img + np.sqrt(a * s + b * b) * rng(seed).standard_normal(img.shape)).astype(np.float32)


def srgb(x):
    return linear_to_srgb(np.clip(x, 0, 1))


def show(ax, img, title, gray=False):
    """Colour images are sRGB-encoded for display; 2-D arrays are shown as grey in [0, 1]."""
    if img.ndim == 3:
        ax.imshow(srgb(img), interpolation="nearest")
    else:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=8)
    ax.axis("off")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print("wrote", name)


def best_of(fn, grid, clean, x):
    """Oracle best parameters of a method on a grid (by PSNR); returns (psnr, params, out)."""
    best = (-1, None, None)
    for kw in grid:
        o = fn(x, **kw)
        p = psnr(clean, o)
        if p > best[0]:
            best = (p, kw, o)
    return best


G_GAUSS = [dict(sigma=s) for s in (0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5)]
G_BIL = [dict(sigma_d=sd, sigma_r=sr) for sd in (1.5, 2.0, 3.0) for sr in (0.03, 0.06, 0.1, 0.2, 0.4)]
G_NLM = [dict(h=h) for h in (0.02, 0.04, 0.06, 0.1)]

t0 = time.time()
photo256 = scenes.photo("astronaut", 256)
photo128 = scenes.photo("astronaut", 128)

# --------------------------------------------------------------------------
# 1. L5 s6-7 re-made: piecewise-constant patch, Gaussian kernel vs bilateral weights
# --------------------------------------------------------------------------
n = 64
patch = np.full((n, n), 0.3, np.float32)
yy, xx = np.mgrid[0:n, 0:n]
patch[(xx > 28 + 0.5 * yy) & (yy < 50)] = 0.8        # slanted bright wedge, as on the slide
npatch = (patch + 0.05 * rng(1).standard_normal(patch.shape)).astype(np.float32)
sd, sr = 5.0, 0.12
pos = (30, 40)                                      # pixel just inside the dark side next to the edge
radius = 12
wg = D.bilateral_weights(npatch, pos, sd, np.inf, radius)
wb = D.bilateral_weights(npatch, pos, sd, sr, radius)
og = D.gaussian(npatch, sd, radius)
ob = D.bilateral(npatch, sd, sr, radius)
fig, axs = plt.subplots(2, 4, figsize=(13, 6.6))
show(axs[0, 0], npatch, "noisy piecewise-constant patch")
axs[0, 0].plot(pos[1], pos[0], "r+", ms=10)
axs[0, 0].add_patch(plt.Rectangle((pos[1] - radius - .5, pos[0] - radius - .5), 2 * radius + 1, 2 * radius + 1, fill=False, ec="r"))
show(axs[0, 1], wg / wg.max(), r"Gaussian kernel at the red +  ($\sigma_d$=%g)" % sd)
show(axs[0, 2], wb / wb.max(), r"bilateral weights ($\sigma_r$=%g)" % sr)
show(axs[0, 3], patch, "clean")
show(axs[1, 0], npatch, "noisy")
show(axs[1, 1], og, "Gaussian output")
show(axs[1, 2], ob, "bilateral output")
axs[1, 3].plot(npatch[pos[0]], c="0.7", label="noisy")
axs[1, 3].plot(og[pos[0]], label="Gaussian")
axs[1, 3].plot(ob[pos[0]], label="bilateral")
axs[1, 3].set_title("row through the red +", fontsize=8)
axs[1, 3].legend(fontsize=7)
save(fig, "denoise_kernels.png")
flat = (slice(5, 25), slice(3, 20))
print(f"[kernels] flat-region std: noisy {npatch[flat].std():.4f}, gaussian {og[flat].std():.4f}, bilateral {ob[flat].std():.4f}")
bright = patch[pos[0] - radius:pos[0] + radius + 1, pos[1] - radius:pos[1] + radius + 1] > 0.55
print(f"[kernels] fraction of the weight on the bright side of the edge: gaussian {(wg * bright).sum() / wg.sum():.3f}, bilateral {(wb * bright).sum() / wb.sum():.4f}")
print(f"[kernels] PSNR vs clean: noisy {psnr(patch, npatch):.2f}, gaussian {psnr(patch, og):.2f}, bilateral {psnr(patch, ob):.2f}")

# --------------------------------------------------------------------------
# 2. L5 s10 layout on a photo
# --------------------------------------------------------------------------
x = add_noise(photo256, LEVELS["mid"])
print(f"[compare] noise level mid: a={LEVELS['mid']}, b={B}; estimate_noise = {D.estimate_noise(x):.4f}")
pg, kg, og2 = best_of(D.gaussian, G_GAUSS, photo256, x)
pb, kb, ob2 = best_of(D.bilateral, G_BIL, photo256, x)
t = time.time(); olc = D.luma_chroma(x); t_lc = time.time() - t
pn, kn, on = best_of(D.nlm, G_NLM, photo256, x)
results = [("noisy", x, ""), ("Gaussian", og2, str(kg)), ("bilateral", ob2, str(kb)),
           ("luma/chroma", olc, "luma 1.5/0.05, chroma 4/0.1"), ("NLM (ext.)", on, str(kn))]
box = (64, 192, 64, 192)
fig, axs = plt.subplots(2, 5, figsize=(16, 6.8))
for j, (name, o, kw) in enumerate(results):
    p, s = psnr(photo256, o), ssim(srgb(photo256), srgb(o))
    show(axs[0, j], o, f"{name}\nPSNR {p:.2f} dB, SSIM {s:.3f}")
    r0, r1, c0, c1 = box
    show(axs[1, j], o[r0:r1, c0:c1], "crop")
    print(f"[compare] {name:12s} PSNR {p:6.2f}  SSIM {s:.3f}  {kw}")
save(fig, "denoise_compare.png")
t = time.time(); D.bilateral(x, 2.5, 0.1, radius=7); t_b = time.time() - t
print(f"[compare] bilateral 256x256x3 radius 7: {t_b:.2f} s; luma_chroma: {t_lc:.2f} s")

# --------------------------------------------------------------------------
# 3. PSNR heat-map over (sigma_d, sigma_r) and best sigma_r vs noise level
# --------------------------------------------------------------------------
sds = np.array([1.0, 1.5, 2.0, 3.0, 4.0])
srs = np.array([0.02, 0.03, 0.05, 0.08, 0.12, 0.2, 0.3, 0.5])
x128 = add_noise(photo128, LEVELS["mid"])
H = np.array([[psnr(photo128, D.bilateral(x128, a, b)) for b in srs] for a in sds])
i, j = np.unravel_index(H.argmax(), H.shape)
gbest = max(psnr(photo128, D.gaussian(x128, s)) for s in (0.4, 0.5, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0))
print(f"[heatmap] mid noise: best bilateral PSNR {H.max():.2f} at sigma_d={sds[i]}, sigma_r={srs[j]}; best Gaussian {gbest:.2f}; noisy {psnr(photo128, x128):.2f}")
print(f"[heatmap] sigma_d=2: sigma_r=0.5 (Gaussian limit) {H[2, -1]:.2f} dB, sigma_r=0.02 (nearly identity) {H[2, 0]:.2f} dB")
sig_levels = [0.005, 0.01, 0.02, 0.04, 0.06]       # constant noise std, for a clean scaling law
fine = np.array([0.01, 0.015, 0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24, 0.32])
best_sr = []
for s_ in sig_levels:
    xn = add_noise(photo128, a=0.0, b=s_)
    pp = [psnr(photo128, D.bilateral(xn, 2.0, v)) for v in fine]
    best_sr.append(float(fine[int(np.argmax(pp))]))
print("[heatmap] noise std -> best sigma_r (sigma_d=2):", dict(zip(sig_levels, best_sr)))
fig, axs = plt.subplots(1, 2, figsize=(12, 4.4))
im = axs[0].imshow(H, origin="lower", cmap="viridis", aspect="auto")
axs[0].set_xticks(range(len(srs)), [f"{v:g}" for v in srs]); axs[0].set_yticks(range(len(sds)), [f"{v:g}" for v in sds])
axs[0].set_xlabel(r"$\sigma_r$"); axs[0].set_ylabel(r"$\sigma_d$"); axs[0].set_title("PSNR (dB), mid noise, 128x128 (star = best)")
axs[0].plot(j, i, "r*", ms=14)
for a_ in range(H.shape[0]):
    for b_ in range(H.shape[1]):
        axs[0].text(b_, a_, f"{H[a_, b_]:.1f}", ha="center", va="center", fontsize=6, color="w")
fig.colorbar(im, ax=axs[0])
axs[1].loglog(sig_levels, best_sr, "o-", label=r"best $\sigma_r$")
axs[1].loglog(sig_levels, [2 * s_ for s_ in sig_levels], "k--", lw=1, label=r"$2\,\sigma_{noise}$")
axs[1].set_xlabel("noise std"); axs[1].set_ylabel(r"best $\sigma_r$"); axs[1].legend(); axs[1].grid(alpha=.3)
axs[1].set_title(r"optimal $\sigma_r$ grows with the noise ($\sigma_d$=2)")
save(fig, "denoise_heatmap.png")

# --------------------------------------------------------------------------
# 4. edge profile across a noisy step
# --------------------------------------------------------------------------
h, w = 64, 128
step = np.full((h, w), 0.3, np.float32); step[:, w // 2:] = 0.7
xs = (step + 0.04 * rng(2).standard_normal(step.shape)).astype(np.float32)
outs = {"noisy": xs, "Gaussian": D.gaussian(xs, 2.5), "bilateral": D.bilateral(xs, 2.5, 0.15),
        "median 7": D.median(xs, 7), "NLM": D.nlm(xs, 0.04, sigma=0.04)}
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
cols = np.arange(w // 2 - 12, w // 2 + 12)
for k_, o in outs.items():
    prof = o.mean(axis=0)
    axs[0].plot(cols, prof[cols], label=k_)
    axs[1].plot(range(w), o[h // 2], label=k_, lw=1)
    slope = np.abs(np.diff(prof)).max()
    if k_ == "noisy":
        width = np.nan
    else:
        lo, hi = 0.34, 0.66
        width = np.interp(hi, prof[cols], cols) - np.interp(lo, prof[cols], cols)
    print(f"[edge] {k_:10s} max slope of row-mean profile {slope:.3f} (ideal 0.400)  10-90% width {width:.2f} px  flat-region std {o[8:-8, 8:40].std():.4f}")
axs[0].set_title("row-averaged profile across the step"); axs[0].legend(fontsize=8)
axs[1].set_title("a single row"); axs[1].legend(fontsize=7)
for a_ in axs:
    a_.grid(alpha=.3)
save(fig, "denoise_edge_profile.png")

# --------------------------------------------------------------------------
# 5. chroma blotches before / after luma-chroma
# --------------------------------------------------------------------------
xm = add_noise(photo256, LEVELS["high"])
lc = D.luma_chroma(xm)
bil = D.bilateral(xm, 1.5, 0.05)           # the gentle (luma) strength applied to all channels


def chroma_only(a):
    y = rgb_to_ycbcr(a); y[..., 0] = 0.5
    return np.clip(ycbcr_to_rgb(y), 0, 1)


ycc_clean = rgb_to_ycbcr(photo256)
cr = lambda a: float(np.sqrt(np.mean((rgb_to_ycbcr(a)[..., 1:] - ycc_clean[..., 1:]) ** 2)))
yr = lambda a: float(np.sqrt(np.mean((rgb_to_ycbcr(a)[..., 0] - ycc_clean[..., 0]) ** 2)))
grad = lambda a: float(np.mean(np.hypot(*np.gradient(rgb_to_ycbcr(a)[..., 0]))))
fig, axs = plt.subplots(2, 4, figsize=(15, 7.4))
b2 = (96, 160, 96, 160)
for j, (nm, a) in enumerate([("clean", photo256), ("noisy (high)", xm), ("gentle bilateral on RGB", bil), ("luma / chroma", lc)]):
    axs[0, j].imshow(chroma_only(a)[b2[0]:b2[1], b2[2]:b2[3]], interpolation="nearest")
    axs[0, j].set_title(f"{nm}: chroma only (Y = 0.5)", fontsize=8); axs[0, j].axis("off")
    show(axs[1, j], a[b2[0]:b2[1], b2[2]:b2[3]], f"{nm}: crop")
    print(f"[chroma] {nm:24s} chroma RMSE {cr(a):.4f}  luma RMSE {yr(a):.4f}  luma gradient {grad(a):.4f}  PSNR {psnr(photo256, a):.2f}")
save(fig, "denoise_chroma.png")

# --------------------------------------------------------------------------
# 6. frame averaging 1 / 4 / 16
# --------------------------------------------------------------------------
frames = [add_noise(photo256, LEVELS["high"], seed=100 + k) for k in range(16)]
fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
s1 = None
for ax, m in zip(axs, (1, 4, 16)):
    o = D.average_frames(frames[:m])
    e = float(np.std(o - photo256))
    s1 = s1 or e
    show(ax, o[64:192, 64:192], f"{m} frame(s): noise std {e:.4f}, PSNR {psnr(photo256, o):.2f} dB")
    print(f"[frames] M={m:2d} noise std {e:.4f} (ratio to M=1: {e / s1:.3f}, 1/sqrt(M) = {1 / np.sqrt(m):.3f})  PSNR {psnr(photo256, o):.2f}")
save(fig, "denoise_frames.png")

# --------------------------------------------------------------------------
# 7. results table: method x noise level (128x128; oracle-best parameters for Gaussian / bilateral / NLM)
# --------------------------------------------------------------------------
rows = ["noisy", "Gaussian", "bilateral", "median 3", "luma/chroma", "NLM (ext.)"]
tab = np.zeros((len(rows), len(LEVELS)))
par = {}
for c, (ln, a) in enumerate(LEVELS.items()):
    xn = add_noise(photo128, a)
    tab[0, c] = psnr(photo128, xn)
    pg_, kg_, _ = best_of(D.gaussian, G_GAUSS, photo128, xn)
    pb_, kb_, _ = best_of(D.bilateral, G_BIL, photo128, xn)
    pn_, kn_, _ = best_of(D.nlm, G_NLM, photo128, xn)
    tab[1, c], tab[2, c], tab[5, c] = pg_, pb_, pn_
    tab[3, c] = psnr(photo128, D.median(xn, 3))
    tab[4, c] = psnr(photo128, D.luma_chroma(xn))
    par[ln] = (kg_, kb_, kn_)
print("[table] PSNR (dB), astronaut 128x128, noise a = " + ", ".join(f"{k}:{v}" for k, v in LEVELS.items()) + f", b = {B}")
print("| method | " + " | ".join(LEVELS) + " |\n|---|" + "---|" * len(LEVELS))
for r_, row in zip(rows, tab):
    print(f"| {r_} | " + " | ".join(f"{v:.2f}" for v in row) + " |")
print("[table] oracle parameters (gaussian, bilateral, nlm):", par)
fig, ax = plt.subplots(figsize=(7, 3))
ax.axis("off")
tb = ax.table(cellText=[[f"{v:.2f}" for v in row] for row in tab], rowLabels=rows,
              colLabels=[f"{k} (a={v})" for k, v in LEVELS.items()], loc="center")
tb.auto_set_font_size(False); tb.set_fontsize(9); tb.scale(1, 1.5)
for c in range(len(LEVELS)):
    tb[int(np.argmax(tab[1:, c])) + 2, c].set_facecolor("#cfe8cf")
ax.set_title("PSNR (dB) by method and noise level (best denoiser shaded)", fontsize=9)
save(fig, "denoise_table.png")
print(f"done in {time.time() - t0:.1f} s")
