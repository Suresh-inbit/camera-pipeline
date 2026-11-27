"""Figures for docs/04_tone_mapping.md (writes docs/figures/tonemap_*.png). Headless."""
import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isp import scenes, tonemap as T
from isp.utils import linear_to_srgb, srgb_to_linear, luminance

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SIZE = 256


def show(ax, img, title):
    ax.imshow(np.clip(img, 0, 1), vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def hist_ax(ax, img, title, color="0.3"):
    h = T.histogram(img, 64)
    ax.bar(np.linspace(0, 1, 64, endpoint=False), h, width=1 / 64, color=color, align="edge")
    ax.set_title(title, fontsize=8)
    ax.set_yticks([])
    ax.set_xlim(0, 1)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=90)
    plt.close(fig)


# ---- 1. gamma curves and a photo at gamma 0.5 / 1 / 2 (L4 s14) -----------------------
photo_lin = scenes.photo("astronaut", SIZE)
photo = linear_to_srgb(photo_lin)           # display-referred starting point (like the slide)
fig = plt.figure(figsize=(13, 3.6))
ax = fig.add_subplot(1, 4, 1)
x = np.linspace(0, 1, 256)
for g in (1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8):
    ax.plot(x, x ** g, label=f"{g:g}")
ax.plot(x, linear_to_srgb(x), "k--", lw=1, label="sRGB")
ax.set_title("out = in^gamma", fontsize=9)
ax.legend(fontsize=6, ncol=2)
ax.set_aspect("equal")
for i, g in enumerate((0.5, 1.0, 2.0)):
    show(fig.add_subplot(1, 4, i + 2), T.gamma(photo, g), f"gamma = {g:g}")
save(fig, "tonemap_gamma.png")
m = lambda a: float(luminance(a).mean())
print(f"gamma: mean luminance of display image g=0.5/1/2: {m(T.gamma(photo, .5)):.3f} / {m(photo):.3f} / {m(T.gamma(photo, 2)):.3f}")

# ---- 2. low vs high contrast with histograms (L4 s13) --------------------------------
low = (0.38 + 0.24 * photo).astype(np.float32)
high = T.auto_levels(low, clip_percent=0.5)
fig, axs = plt.subplots(2, 2, figsize=(8, 6), gridspec_kw=dict(height_ratios=[3, 1.3]))
show(axs[0, 0], low, "low contrast")
show(axs[0, 1], high, "after levels (black/white point)")
hist_ax(axs[1, 0], low, "histogram")
hist_ax(axs[1, 1], high, "histogram")
save(fig, "tonemap_levels.png")
yl, yh = luminance(low), luminance(high)
print(f"levels: luminance range low [{yl.min():.2f}, {yl.max():.2f}] std {yl.std():.3f} -> stretched std {yh.std():.3f}")

# ---- 3. histogram equalisation before / after, CDFs ---------------------------------
eq = T.hist_equalize(low)
fig, axs = plt.subplots(2, 3, figsize=(12, 6), gridspec_kw=dict(height_ratios=[3, 1.4]))
show(axs[0, 0], low, "input (display values)")
show(axs[0, 1], eq, "hist_equalize (luminance)")
hist_ax(axs[1, 0], low, "histogram, input")
hist_ax(axs[1, 1], eq, "histogram, equalised")
for im, lab in ((low, "input"), (eq, "equalised")):
    y = np.sort(luminance(im).ravel())
    axs[0, 2].plot(y[::50], np.arange(y.size)[::50] / y.size, label=lab)
axs[0, 2].plot([0, 1], [0, 1], "k:", lw=1)
axs[0, 2].set_title("CDF of luminance", fontsize=9)
axs[0, 2].legend(fontsize=8)
axs[1, 2].axis("off")
save(fig, "tonemap_histeq.png")
ye = np.sort(luminance(eq).ravel())
dev = np.max(np.abs(np.arange(1, ye.size + 1) / ye.size - ye))
print(f"hist_eq: luminance std {luminance(low).std():.3f} -> {luminance(eq).std():.3f}, max CDF deviation from diagonal {dev:.3f}")

# ---- 4. luminance vs per-channel equalisation: colour shift --------------------------
tinted = np.clip(low * np.array([1.15, 1.0, 0.8], np.float32), 0, 1)
e_lum, e_ch = T.hist_equalize(tinted, mode="luminance"), T.hist_equalize(tinted, mode="channel")


def sat(a):
    return float(np.mean((a.max(-1) - a.min(-1)) / np.maximum(a.max(-1), 1e-3)))


def hue_err(a, ref):
    ca = a[..., :2] / np.maximum(a.sum(-1, keepdims=True), 1e-3)
    cr = ref[..., :2] / np.maximum(ref.sum(-1, keepdims=True), 1e-3)
    return float(np.abs(ca - cr).mean())


fig, axs = plt.subplots(1, 3, figsize=(12, 4.2))
show(axs[0], tinted, "input (warm cast)")
show(axs[1], e_lum, f"luminance mode  (chroma err {hue_err(e_lum, tinted):.3f})")
show(axs[2], e_ch, f"per-channel mode  (chroma err {hue_err(e_ch, tinted):.3f})")
save(fig, "tonemap_histeq_color.png")
print(f"hist_eq colour: mean r,g chromaticity error vs input: luminance {hue_err(e_lum, tinted):.4f}, "
      f"per-channel {hue_err(e_ch, tinted):.4f}; mean R/B ratio input {np.mean(tinted[...,0]/np.maximum(tinted[...,2],1e-3)):.2f}, "
      f"luminance {np.mean(e_lum[...,0]/np.maximum(e_lum[...,2],1e-3)):.2f}, per-channel {np.mean(e_ch[...,0]/np.maximum(e_ch[...,2],1e-3)):.2f}")

# ---- 5. global vs local on scenes.high_contrast -------------------------------------
hc = scenes.high_contrast(SIZE)
kw = dict(sigma=16, strength=0.7)
results = {
    "linear (clip)": T.tonemap(hc, "linear"),
    "sRGB": T.tonemap(hc, "srgb"),
    "gamma 1/2.2": T.tonemap(hc, "gamma"),
    "hist-eq": T.tonemap(hc, "hist"),
    "CLAHE": T.tonemap(hc, "clahe", clip=3.0),
    "dodge&burn, Gaussian base": T.tonemap(hc, "local", base="gaussian", **kw),
    "dodge&burn, bilateral base": T.tonemap(hc, "local", base="bilateral", **kw),
}


def detail(y, s=1.5):
    L = np.log2(y + 1e-3).astype(np.float32)
    return float((L - cv2.GaussianBlur(L, (0, 0), s)).std())


rows = []
w = SIZE
for name, im in results.items():
    # numbers on linear light after display decoding (gamma/linear: take the values as shown)
    # displayed values for the means; linear light (display decoded) for detail contrast
    yd = luminance(im)
    dec = {"linear (clip)": lambda a: a, "gamma 1/2.2": lambda a: a ** 2.2}.get(name, srgb_to_linear)
    y = luminance(dec(im))
    L, R = yd[:, : w // 4], yd[:, 3 * w // 4:]
    rows.append((name, L.mean(), R.mean(), R.mean() / max(L.mean(), 1e-6), detail(y[:, : w // 4]), detail(y[:, 3 * w // 4:])))
fig, axs = plt.subplots(2, 4, figsize=(14, 7.4))
show(axs[0, 0], hc ** 1.0, "scene (linear, as is)")
for ax, (name, im) in zip(axs.ravel()[1:], results.items()):
    show(ax, im, name)
save(fig, "tonemap_global_vs_local.png")
print("global vs local on high_contrast (means on the displayed values, detail = std of high-passed log2 luminance of the decoded linear light; L = darkest quarter, R = brightest quarter):")
print(f"{'method':28s} {'mean L':>7s} {'mean R':>7s} {'R/L':>6s} {'detail L':>9s} {'detail R':>9s}")
for r in rows:
    print(f"{r[0]:28s} {r[1]:7.3f} {r[2]:7.3f} {r[3]:6.2f} {r[4]:9.3f} {r[5]:9.3f}")
yh = luminance(hc)
print(f"input linear: mean L {yh[:, :w//4].mean():.4f}, mean R {yh[:, 3*w//4:].mean():.4f}, ratio {yh[:, 3*w//4:].mean()/yh[:, :w//4].mean():.1f}, "
      f"detail L {detail(yh[:, :w//4]):.3f}, detail R {detail(yh[:, 3*w//4:]):.3f}")

# zoom on the halo region: the transition between dark and bright
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
crop = (slice(0, 128), slice(w // 4, 3 * w // 4))
for ax, k in zip(axs, ("sRGB", "dodge&burn, Gaussian base", "dodge&burn, bilateral base")):
    show(ax, results[k][crop], k)
save(fig, "tonemap_halo_zoom.png")

# ---- 6. halo measure on a step edge --------------------------------------------------
step = scenes.step_edge(128, lo=0.05, hi=0.8, angle_deg=0.0)


def overshoot(y):
    row = y[y.shape[0] // 2]
    n = row.size
    lo_far, hi_far = row[: n // 8].mean(), row[-n // 8:].mean()
    return float(max(row.max() - hi_far, lo_far - row.min(), 0) / (hi_far - lo_far))


fig, ax = plt.subplots(figsize=(6, 3.4))
ax.plot(luminance(step)[64] / luminance(step)[64].max(), "k", label="input (scaled)")
for b in ("gaussian", "bilateral"):
    o = luminance(T.dodge_burn(step, sigma=8, strength=0.7, base=b))
    ax.plot(o[64], label=f"{b} base")
    print(f"step edge overshoot ({b} base, sigma=8, strength=0.7): {overshoot(o):.3f}")
ax.legend(fontsize=8)
ax.set_title("row through a step edge after dodge_burn (linear)", fontsize=9)
save(fig, "tonemap_halo_profile.png")

# ---- 7. base / detail decomposition --------------------------------------------------
B, D = T.decompose(hc, sigma=16, base="bilateral")
Bg, Dg = T.decompose(hc, sigma=16, base="gaussian")
fig, axs = plt.subplots(2, 3, figsize=(12, 8))
Y = np.log2(luminance(hc) + 1e-4)
for ax, a, t, lim in ((axs[0, 0], Y, "log2 luminance  L", None), (axs[0, 1], B, "base (bilateral)", None),
                      (axs[0, 2], D, "detail  L - base", (-1.5, 1.5)),
                      (axs[1, 0], Bg, "base (Gaussian)", None), (axs[1, 1], Dg, "detail (Gaussian)", (-1.5, 1.5))):
    im = ax.imshow(a, cmap="gray" if lim else "viridis", vmin=lim[0] if lim else Y.min(), vmax=lim[1] if lim else Y.max())
    ax.set_title(t, fontsize=9); ax.axis("off")
show(axs[1, 2], T.tonemap(hc, "local", sigma=16, strength=0.7, base="bilateral"), "recombined: compressed base + detail")
save(fig, "tonemap_base_detail.png")
print(f"base/detail: range of L {np.ptp(Y):.1f} stops, base (bilateral) {np.ptp(B):.1f}, base (Gaussian) {np.ptp(Bg):.1f}; "
      f"robust (1-99 pct) range of L {np.ptp(np.percentile(Y,[1,99])):.1f}, base bil. {np.ptp(np.percentile(B,[1,99])):.1f}, base gau. {np.ptp(np.percentile(Bg,[1,99])):.1f}, "
      f"detail MAD-std {1.4826*np.median(np.abs(D-np.median(D))):.3f} (bilateral) / {1.4826*np.median(np.abs(Dg-np.median(Dg))):.3f} (Gaussian)")

# ---- 8. manual mask dodge / burn -----------------------------------------------------
dark = photo_lin * 0.5
yy, xx = np.mgrid[0:SIZE, 0:SIZE]
face = ((xx - 0.44 * SIZE) ** 2 / (0.14 * SIZE) ** 2 + (yy - 0.25 * SIZE) ** 2 / (0.17 * SIZE) ** 2 < 1).astype(np.float32)
sky = (yy > 0.8 * SIZE).astype(np.float32)
d1 = T.dodge_burn_mask(dark, face, stops=+1.0, feather=8)
d2 = T.dodge_burn_mask(d1, sky, stops=-1.0, feather=8)
fig, axs = plt.subplots(1, 4, figsize=(14, 3.8))
show(axs[0], linear_to_srgb(dark), "start (sRGB view)")
show(axs[1], np.stack([face, sky, 0 * face], -1), "masks: dodge (red), burn (green)")
show(axs[2], linear_to_srgb(d1), "dodge face +1 stop")
show(axs[3], linear_to_srgb(d2), "then burn bottom -1 stop")
save(fig, "tonemap_mask.png")
fm = face > 0.5
print(f"mask dodge: mean linear luminance inside mask {luminance(dark)[fm].mean():.3f} -> {luminance(d1)[fm].mean():.3f} (x{luminance(d1)[fm].mean()/luminance(dark)[fm].mean():.2f}); "
      f"burn: {luminance(d1)[sky > 0.5].mean():.3f} -> {luminance(d2)[sky > 0.5].mean():.3f}")
