"""Figures for docs/06_sharpening.md (L5 s12-17).  Headless; prints the numbers quoted in the doc."""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isp import scenes, utils
from isp import sharpen as S

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)


def show(ax, img, title, fs=9):
    ax.imshow(np.clip(img, 0, 1), cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=fs)
    ax.axis("off")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=90)
    plt.close(fig)


def kernel_text(k):
    return "\n".join("[" + " ".join(f"{int(v):2d}" for v in row) + "]" for row in k)


def noisy_of(img, sigma, seed=0):
    return (img + sigma * utils.rng(seed).standard_normal(img.shape)).astype(np.float32)


def bilateral(img, sc=0.08, ss=2.0):
    """Small local denoiser (the real one lives in isp.denoise)."""
    return cv2.bilateralFilter(np.ascontiguousarray(img, dtype=np.float32), 7, sc, ss)


# display-referred photo: sharpening happens after tone mapping (L5 s12)
photo = utils.linear_to_srgb(scenes.photo("astronaut", 320))
BOX = (20, 120, 120, 220)          # face crop (r0, r1, c0, c1)
crop = lambda a: utils.crop(a, BOX)

# ---------------------------------------------------------------- 1. photo, lam = 1, 2
k1, k2 = S.sharpening_kernel(1), S.sharpening_kernel(2)
s1, s2 = S.unsharp_mask(photo, 1), S.unsharp_mask(photo, 2)
fig, axs = plt.subplots(2, 3, figsize=(14, 8))
for j, (im, t, k) in enumerate([(photo, "original (L5 s12)", None), (s1, "lam = 1 (L5 s13, s16)", k1),
                                (s2, "lam = 2 (L5 s14, s17)", k2)]):
    show(axs[0, j], im, t, 11)
    show(axs[1, j], crop(im), "crop (nearest-neighbour zoom)", 10)
    if k is not None:
        axs[0, j].text(0.02, 0.02, "kernel\n" + kernel_text(k), transform=axs[0, j].transAxes, fontsize=10,
                       family="monospace", va="bottom", bbox=dict(fc="white", ec="k", alpha=0.85))
save(fig, "sharpen_photo.png")
print("photo mean gradient: orig %.4f  lam1 %.4f  lam2 %.4f" % tuple(S.mean_gradient(a) for a in (photo, s1, s2)))
print("photo mean brightness: orig %.4f  lam1 %.4f  lam2 %.4f" % tuple(a.mean() for a in (photo, s1, s2)))
print("clipped pixels (any channel at 0 or 1) lam1 %.2f%%  lam2 %.2f%%" % tuple(
    100 * np.mean(np.any((a <= 0) | (a >= 1), axis=2)) for a in (s1, s2)) + " | original %.2f%%" %
    (100 * np.mean(np.any((photo <= 0) | (photo >= 1), axis=2))))

# ---------------------------------------------------------------- 2. unsharp-masking diagram (L5 s15), real 1-D data
n, w = 128, 8
x = np.zeros((w, n), np.float32)
x[:, 40:88] = 0.6
x = cv2.GaussianBlur(x, (0, 0), 1.5) + 0.2                       # a soft pulse on a 0.2 pedestal
lam, sig = 1.5, 2.0
F = S.high_pass(x, "gaussian", sig)
sharp = S.unsharp_mask(x, lam, "gaussian", sig, clip=False)
row = lambda a: a[w // 2]
assert np.allclose(row(sharp), row(x) + lam * row(F), atol=1e-6)
fig, axs = plt.subplots(1, 4, figsize=(14, 3.2), sharex=True)
for ax, (y, t) in zip(axs, [(row(x), "original x"), (row(F), "high-pass F(x)"),
                            (lam * row(F), f"lam F(x), lam = {lam}"), (row(sharp), "sharpened S = x + lam F(x)")]):
    ax.plot(y, "k", lw=1.8)
    ax.axhline(0, color="0.7", lw=0.8)
    ax.set_title(t, fontsize=10)
    ax.set_xlabel("pixel")
axs[3].plot(row(x), color="tab:blue", lw=1, ls="--", label="original")
axs[3].legend(fontsize=8, loc="upper left")
axs[0].set_ylim(-0.3, 1.0)
for a in axs[1:]:
    a.set_ylim(-0.5, 0.5) if a is not axs[3] else a.set_ylim(-0.3, 1.0)
save(fig, "sharpen_unsharp_diagram.png")
pulse = row(x)
print("diagram: plateau %.3f, sharpened max %.3f (overshoot %.3f of the 0.6 step), min %.3f"
      % (pulse[64], row(sharp).max(), S.overshoot(row(sharp)[:64]), row(sharp).min()))

# ---------------------------------------------------------------- 3. edge profiles
edge = scenes.step_edge(128)
lams = [0, 0.5, 1, 2, 3]
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
print("edge profiles (clip off): kind, lam, overshoot%, rise10-90, mean_gradient")
for ax, (kind, sg, ttl) in zip(axs, [("laplacian", 1.0, "Laplacian kernel (slides)"),
                                     ("gaussian", 2.0, "Gaussian unsharp, sigma = 2")]):
    for lam_ in lams:
        o = S.unsharp_mask(edge, lam_, kind, sg, clip=False)
        d, p = S.edge_profile(o)
        ov, rd = S.overshoot(p), S.rise_distance(d, p)
        ax.plot(d, p, label=f"lam={lam_}: overshoot {100 * ov:.0f}%, rise {rd:.2f}px")
        print(f"  {kind:9s} lam={lam_:<4} overshoot {100 * ov:5.1f}%  rise {rd:.2f}px  grad {S.mean_gradient(o):.5f}")
    ax.set_xlim(-8, 8)
    ax.set_xlabel("distance across the edge (px)")
    ax.set_title(ttl)
    ax.legend(fontsize=8)
axs[0].set_ylabel("intensity")
save(fig, "sharpen_edge_profiles.png")

# ---------------------------------------------------------------- 4. frequency responses
fx = np.linspace(0, np.pi, 33)
fig, axs = plt.subplots(1, 3, figsize=(14, 4))
for lam_ in (0.5, 1, 2):
    h = S.unsharp_response(lam_, "laplacian", n=64)
    axs[0].plot(fx, h[0, :33], label=f"lam={lam_}  (pi,0): {h[0, 32]:.1f}")
    axs[1].plot(fx, np.diag(h)[:33], label=f"lam={lam_}  (pi,pi): {h[32, 32]:.1f}")
axs[0].set_title("Laplacian unsharp, cut along $f_x$")
axs[1].set_title("Laplacian unsharp, diagonal cut")
for sg, ls in ((1.0, "-"), (2.0, "--")):
    for lam_ in (1, 2):
        h = S.unsharp_response(lam_, "gaussian", sg, n=64)
        axs[2].plot(fx, h[0, :33], ls, label=f"sigma={sg}, lam={lam_}")
h = S.unsharp_response(1, "laplacian", n=64)
axs[2].plot(fx, h[0, :33], "k:", label="Laplacian lam=1")
axs[2].set_title("Gaussian unsharp (cut along $f_x$)")
for a in axs:
    a.axhline(1, color="0.7", lw=0.8)
    a.set_xlabel("$\\omega$ (rad/pixel)")
    a.set_ylabel("gain")
    a.legend(fontsize=8)
save(fig, "sharpen_frequency_response.png")
for lam_ in (1, 2):
    h = S.unsharp_response(lam_, "laplacian", n=64)
    print(f"freq response lam={lam_}: DC {h[0, 0]:.2f}  (pi,0) {h[0, 32]:.2f}  (pi,pi) {h[32, 32]:.2f}")
for sg in (1.0, 2.0):
    h = S.unsharp_response(1, "gaussian", sg, n=64)
    print(f"gaussian sigma={sg} lam=1: peak gain {h[0, :33].max():.3f} at w={fx[np.argmax(h[0, :33])]:.2f}, gain at pi {h[0, 32]:.3f}")

# ---------------------------------------------------------------- 5. noise amplification and thresholding
sn = 0.02
noisy = noisy_of(photo, sn)
plain = S.unsharp_mask(noisy, 1.0)
thr = S.unsharp_mask(noisy, 1.0, threshold=0.3)
flat = (slice(10, 50), slice(40, 80))                           # smooth mid-grey wall patch (clean std 0.02)
print("noise-region patch std: noisy %.4f | sharpened %.4f | thresholded %.4f | original clean %.4f"
      % (noisy[flat].std(), plain[flat].std(), thr[flat].std(), photo[flat].std()))
print("PSNR vs clean: noisy %.2f | sharpened %.2f | thresholded %.2f dB" %
      (utils.psnr(photo, noisy), utils.psnr(photo, plain), utils.psnr(photo, thr)))
print("noise gain lam=1 %.3f  lam=2 %.3f" % (S.noise_gain(k1), S.noise_gain(k2)))
wn = (0.5 + 0.02 * utils.rng(0).standard_normal((256, 256))).astype(np.float32)
for lam_ in (1, 2):
    meas = S.unsharp_mask(wn, lam_, clip=False)[4:-4, 4:-4].std() / wn[4:-4, 4:-4].std()
    print(f"white-noise std ratio lam={lam_}: measured {meas:.3f}  analytic {S.noise_gain(S.sharpening_kernel(lam_)):.3f}")
fig, axs = plt.subplots(2, 3, figsize=(14, 8))
nb = (60, 140, 120, 200)
for j, (im, t) in enumerate([(noisy, f"noisy input (std {sn})"), (plain, "sharpened, lam=1"),
                             (thr, "sharpened, lam=1, threshold 0.3")]):
    show(axs[0, j], im, t, 11)
    show(axs[1, j], utils.crop(im, nb), "crop", 10)
save(fig, "sharpen_noise.png")

# ---------------------------------------------------------------- 5b. order: denoise -> sharpen vs sharpen -> denoise
sharpen_f = lambda a: S.unsharp_mask(a, 1.0, "gaussian", 1.5)
a_ds = sharpen_f(bilateral(noisy))                              # denoise, then sharpen (camera order)
a_sd = bilateral(sharpen_f(noisy))                              # sharpen, then denoise
ref = sharpen_f(photo)
res = {"denoise only": bilateral(noisy), "denoise -> sharpen": a_ds, "sharpen -> denoise": a_sd}
print("order experiment, PSNR vs clean / vs sharpened clean / mean gradient (clean %.4f, sharpened clean %.4f):"
      % (S.mean_gradient(photo), S.mean_gradient(ref)))
for k_, v in res.items():
    print(f"  {k_:20s} {utils.psnr(photo, v):6.2f} dB  {utils.psnr(ref, v):6.2f} dB  grad {S.mean_gradient(v):.4f}")
fig, axs = plt.subplots(2, 3, figsize=(14, 8))
for j, (im, t) in enumerate([(noisy, "noisy"), (a_ds, "denoise -> sharpen"), (a_sd, "sharpen -> denoise")]):
    show(axs[0, j], im, t + (f"  PSNR {utils.psnr(photo, im):.1f} dB" if j else ""), 11)
    show(axs[1, j], utils.crop(im, nb), "crop", 10)
save(fig, "sharpen_order.png")

# ---------------------------------------------------------------- 6. over-sharpening halos
settings = [("original", None), ("Laplacian lam=1", dict(lam=1, kind="laplacian")),
            ("Laplacian lam=3", dict(lam=3, kind="laplacian")),
            ("Gaussian sigma=4, lam=2", dict(lam=2, kind="gaussian", sigma=4.0))]
fig, axs = plt.subplots(2, 4, figsize=(15, 8))
for j, (t, kw) in enumerate(settings):
    im = photo if kw is None else S.unsharp_mask(photo, **kw)
    show(axs[0, j], utils.crop(im, (20, 120, 100, 200)), t, 11)
    o = edge if kw is None else S.unsharp_mask(edge, clip=False, **kw)
    d, p = S.edge_profile(o, half=16)
    axs[1, j].plot(d, p, "k")
    axs[1, j].axhline(0.2, color="0.7", lw=0.8)
    axs[1, j].axhline(0.8, color="0.7", lw=0.8)
    axs[1, j].set_title(f"step edge: overshoot {100 * S.overshoot(p):.0f}%", fontsize=10)
    axs[1, j].set_xlabel("px")
    axs[1, j].set_ylim(-0.1, 1.1)
    print(f"halo {t:25s}: overshoot {100 * S.overshoot(p):5.1f}%  halo width (px beyond |dev|>0.02 of step): "
          f"{(np.abs(p - np.clip(p, 0.2, 0.8)) > 0.012).sum() * 0.25:.2f}")
save(fig, "sharpen_halos.png")

# ---------------------------------------------------------------- 7. luminance-only vs RGB on a colour edge
cedge = np.zeros((96, 96, 3), np.float32)
cedge[:, :48] = (0.85, 0.2, 0.2)
cedge[:, 48:] = (0.15, 0.5, 0.9)
cedge = cv2.GaussianBlur(cedge, (0, 0), 1.0)
rgb_s = S.unsharp_mask(cedge, 2.0, "gaussian", 1.5)
lum_s = S.unsharp_mask(cedge, 2.0, "gaussian", 1.5, luminance_only=True)
ycc0 = utils.rgb_to_ycbcr(cedge)
for nm, im in (("RGB", rgb_s), ("luminance only", lum_s)):
    y = utils.rgb_to_ycbcr(im)
    dc = np.abs(y[..., 1:] - ycc0[..., 1:]).max()
    print(f"colour edge, {nm:15s}: max |dCb,dCr| {dc:.3f}, min channel {im.min():.3f}, max channel {im.max():.3f}")
# unclipped chroma change
for nm, kw in (("RGB", {}), ("luminance only", {"luminance_only": True})):
    y = utils.rgb_to_ycbcr(S.unsharp_mask(cedge, 2.0, "gaussian", 1.5, clip=False, **kw))
    print(f"  unclipped {nm:15s}: max |dCb,dCr| {np.abs(y[..., 1:] - ycc0[..., 1:]).max():.4f}")
ph = photo[:160, 100:260]
fig, axs = plt.subplots(2, 3, figsize=(13, 8))
show(axs[0, 0], cedge[30:66, 30:66], "original colour edge")
show(axs[0, 1], rgb_s[30:66, 30:66], "RGB sharpening (per channel)")
show(axs[0, 2], lum_s[30:66, 30:66], "luminance-only (Y of YCbCr)")
dd = lambda a: np.clip(0.5 + 2.5 * (utils.rgb_to_ycbcr(a)[..., 1] - ycc0[..., 1]), 0, 1)
show(axs[1, 0], dd(cedge)[30:66, 30:66], "Cb change x2.5 (original: none)")
show(axs[1, 1], dd(rgb_s)[30:66, 30:66], "Cb change, RGB")
show(axs[1, 2], dd(lum_s)[30:66, 30:66], "Cb change, luminance-only")
for a in axs[1]:
    a.images[0].set_cmap("gray")
save(fig, "sharpen_luma_vs_rgb.png")

# ---------------------------------------------------------------- 8. numbers table
rows = []
for lab, kw in [("none", dict(lam=0)), ("Laplacian lam=0.5", dict(lam=0.5)), ("Laplacian lam=1", dict(lam=1)),
                ("Laplacian lam=2", dict(lam=2)), ("Laplacian lam=3", dict(lam=3)),
                ("Gaussian s=1 lam=1", dict(lam=1, kind="gaussian", sigma=1.0)),
                ("Gaussian s=2 lam=2", dict(lam=2, kind="gaussian", sigma=2.0))]:
    kern = S.sharpening_kernel(kw["lam"], kw.get("kind", "laplacian"), kw.get("sigma", 1.0))
    d, p = S.edge_profile(S.unsharp_mask(edge, clip=False, **kw))
    h = S.unsharp_response(kw["lam"], kw.get("kind", "laplacian"), kw.get("sigma", 1.0))
    rows.append([lab, f"{S.noise_gain(kern):.2f}", f"{100 * S.overshoot(p):.1f}", f"{S.rise_distance(d, p):.2f}",
                 f"{S.mean_gradient(S.unsharp_mask(photo, **kw)):.4f}", f"{h[0, 32]:.2f}"])
hdr = ["operator", "noise gain", "overshoot %", "10-90% rise (px)", "photo mean grad", "gain at (pi, 0)"]
print("\t".join(hdr))
for r in rows:
    print("\t".join(r))
fig, ax = plt.subplots(figsize=(10, 3))
ax.axis("off")
tb = ax.table(cellText=rows, colLabels=hdr, loc="center", cellLoc="center")
tb.auto_set_font_size(False)
tb.set_fontsize(9)
tb.scale(1, 1.5)
save(fig, "sharpen_table.png")
print("figures written to", OUT)
