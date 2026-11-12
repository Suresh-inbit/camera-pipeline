"""Figures for docs/07_compression.md (JPEG).  Headless; prints the numbers quoted in the doc."""
import pathlib
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from isp import jpeg, scenes, utils

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 90
srgb = utils.linear_to_srgb


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)


def show(ax, img, title, **kw):
    ax.imshow(np.clip(img, 0, 1), cmap=None if np.ndim(img) == 3 else "gray", vmin=0, vmax=1,
              interpolation="nearest", **kw)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


photo = srgb(scenes.photo("astronaut", 256))
gray = utils.luminance(photo)

# ------------------------------------------------------------------ 1. DCT basis (L5 s20)
B = jpeg.dct_basis()
fig, axs = plt.subplots(8, 8, figsize=(6, 6))
for u in range(8):
    for v in range(8):
        axs[u, v].imshow(B[u, v], cmap="gray", interpolation="nearest")
        axs[u, v].axis("off")
fig.suptitle("The 64 DCT basis functions (row = vertical, column = horizontal frequency)", fontsize=10)
save(fig, "jpeg_dct_basis.png")
print(f"[basis] max |B B^T - I| = {np.abs(B.reshape(64, 64) @ B.reshape(64, 64).T - np.eye(64)).max():.1e}")

# ------------------------------------------------------------------ 2. one block step by step
# pick a block on the astronaut's face edge: highest AC energy among a few candidates
c = jpeg.block_dct(gray)
energy = (c ** 2).sum(axis=(2, 3)) - c[..., 0, 0] ** 2
cand = [(i, j) for i in range(8, 24) for j in range(8, 24)]
bi, bj = sorted(cand, key=lambda ij: abs(energy[ij] - np.median(energy[8:24, 8:24])))[0]
r = jpeg.block_before_after(gray, (bi, bj), quality=50)
fig, axs = plt.subplots(2, 3, figsize=(12, 7.5))


def numbers(ax, arr, title, fmt="{:.0f}", cmap="gray", vmin=None, vmax=None, sym=False):
    a = np.asarray(arr)
    lim = np.abs(a).max() or 1
    ax.imshow(a, cmap=cmap if not sym else "RdBu_r", vmin=-lim if sym else vmin, vmax=lim if sym else vmax)
    for (i, j), v in np.ndenumerate(a):
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=6.5,
                color="k" if sym else ("w" if v < (a.max() + a.min()) / 2 else "k"))
    ax.set_title(title, fontsize=9)
    ax.axis("off")


numbers(axs[0, 0], r["pixels"], f"pixels (block {bi},{bj})")
numbers(axs[0, 1], r["coefs"], "DCT coefficients (after -128)", sym=True)
numbers(axs[0, 2], r["q"], "quantisation table, quality 50", cmap="viridis")
numbers(axs[1, 0], r["quantised"], f"quantised: {(r['quantised'] != 0).sum()} of 64 non-zero", sym=True)
numbers(axs[1, 1], r["recon"], "reconstruction")
numbers(axs[1, 2], r["error"], f"error (rms {np.sqrt((r['error'] ** 2).mean()):.1f} levels)", fmt="{:.1f}", sym=True)
save(fig, "jpeg_block.png")
print(f"[block] block ({bi},{bj}): non-zero after quantisation {(r['quantised'] != 0).sum()}/64, "
      f"rms error {np.sqrt((r['error'] ** 2).mean()):.2f} levels (0..255)")

# ------------------------------------------------------------------ 3. quality sweep
box = (24, 120, 88, 184)
qs = [5, 20, 50, 90]
fig, axs = plt.subplots(1, 5, figsize=(15, 3.6))
show(axs[0], utils.crop(photo, box), "original (24 bpp)")
print("[sweep] quality  bpp   PSNR   ratio  blockiness  (astronaut 256, 4:2:0)")
for ax, q in zip(axs[1:], qs):
    out, info = jpeg.roundtrip(photo, q, "4:2:0")
    show(ax, utils.crop(out, box), f"q={q}: {info['bpp']:.2f} bpp, {info['psnr']:.1f} dB")
    print(f"[sweep] {q:5d}  {info['bpp']:.2f}  {info['psnr']:.2f}  {info['ratio']:5.1f}  {jpeg.blockiness(out):.3f}")
print(f"[sweep] blockiness of the original: {jpeg.blockiness(photo):.3f}")
save(fig, "jpeg_quality_sweep.png")

# ------------------------------------------------------------------ 4. rate-distortion, ours vs Pillow
Q = [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
fig, ax = plt.subplots(figsize=(7, 4.6))
gaps = []
for sub, col in (("4:4:4", "tab:blue"), ("4:2:0", "tab:orange")):
    ours = [jpeg.roundtrip(photo, q, sub)[1] for q in Q]
    ref = []
    for q in Q:
        o, n = jpeg.pillow_roundtrip(photo, q, sub, optimize=True)
        ref.append((8.0 * n / photo.shape[0] / photo.shape[1], utils.psnr(photo, o)))
    ax.plot([i["bpp"] for i in ours], [i["psnr"] for i in ours], "-o", color=col, ms=4, label=f"ours {sub}")
    ax.plot([a for a, _ in ref], [b for _, b in ref], "--x", color=col, ms=5, label=f"Pillow {sub}")
    dsz = [abs(i["bpp"] - a) / a * 100 for i, (a, _) in zip(ours, ref)]
    dps = [abs(i["psnr"] - b) for i, (_, b) in zip(ours, ref)]
    gaps.append((sub, np.mean(dsz), np.max(dsz), np.mean([i["bpp"] / a - 1 for i, (a, _) in zip(ours, ref)]) * 100, max(dps)))
    print(f"[rd] {sub}: size gap vs Pillow mean {np.mean(dsz):.1f}% max {np.max(dsz):.1f}%, "
          f"signed mean {gaps[-1][3]:+.1f}%, max PSNR gap {max(dps):.2f} dB")
    if sub == "4:2:0":
        for q, i, (a, b) in zip(Q, ours, ref):
            print(f"[rd]   q={q:3d} ours {i['bpp']:.3f} bpp {i['psnr']:.2f} dB | Pillow {a:.3f} bpp {b:.2f} dB")
ax.set_xlabel("bits per pixel")
ax.set_ylabel("PSNR (dB)")
ax.set_title("Rate-distortion, astronaut 256x256 (q = 5 ... 95)")
ax.grid(alpha=0.3)
ax.legend()
save(fig, "jpeg_rate_distortion.png")
i444 = jpeg.roundtrip(photo, 75, "4:4:4")[1]
i420 = jpeg.roundtrip(photo, 75, "4:2:0")[1]
print(f"[rd] q=75: 4:4:4 {i444['bpp']:.2f} bpp {i444['psnr']:.2f} dB; 4:2:0 {i420['bpp']:.2f} bpp {i420['psnr']:.2f} dB "
      f"(saves {100 * (1 - i420['bpp'] / i444['bpp']):.0f}% of the bits)")

# ------------------------------------------------------------------ 5. Y / Cb / Cr and chroma subsampling
chart = srgb(scenes.patches(256))
ycc = utils.rgb_to_ycbcr(chart)
box = (136, 184, 60, 108)                    # boundary between a green and a red patch
fig, axs = plt.subplots(2, 4, figsize=(14, 7))
show(axs[0, 0], chart, "patch chart")
for k, nm in enumerate(("Y", "Cb", "Cr")):
    show(axs[0, k + 1], ycc[..., k], f"{nm} plane")
for ax, sub in zip(axs[1, :3], ("4:4:4", "4:2:2", "4:2:0")):
    out, info = jpeg.roundtrip(chart, 90, sub)
    show(ax, utils.crop(out, box), f"{sub}, q=90: {info['bpp']:.2f} bpp, {info['psnr']:.1f} dB")
    err = np.abs(utils.rgb_to_ycbcr(out)[..., 1:] - ycc[..., 1:]).max()
    print(f"[chroma] {sub} q=90: {info['bpp']:.3f} bpp, PSNR {info['psnr']:.2f} dB, max chroma error {err:.3f}")
show(axs[1, 3], utils.crop(chart, box), "original crop")
save(fig, "jpeg_chroma.png")

# ------------------------------------------------------------------ 6. bit budget
Qb = [10, 20, 30, 50, 70, 90]
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
parts = [("Y", "AC"), ("Y", "DC"), ("Cb", "AC"), ("Cb", "DC"), ("Cr", "AC"), ("Cr", "DC")]
cols = ["#1f77b4", "#aec7e8", "#2ca02c", "#98df8a", "#d62728", "#ff9896"]
npix = photo.shape[0] * photo.shape[1]
print("[budget] quality  Y%  chroma%  AC%  DC%  (4:2:0)")
for ax, sub in zip(axs, ("4:4:4", "4:2:0")):
    bottom = np.zeros(len(Qb))
    for (comp, kind), col in zip(parts, cols):
        vals = np.array([jpeg.encode(photo, q, sub).bits[comp][kind] / npix for q in Qb])
        ax.bar([str(q) for q in Qb], vals, bottom=bottom, color=col, label=f"{comp} {kind}")
        bottom += vals
    ax.set_xlabel("quality")
    ax.set_ylabel("bits per pixel (payload only)")
    ax.set_title(f"Where the bits go, {sub}")
    ax.legend(fontsize=7, ncol=2)
for q in Qb:
    b = jpeg.encode(photo, q, "4:2:0").bits
    tot = sum(v for c in b.values() for v in c.values())
    y = sum(b["Y"].values())
    ac = sum(c["AC"] for c in b.values())
    print(f"[budget] {q:7d}  {100 * y / tot:.0f}  {100 * (tot - y) / tot:.0f}  {100 * ac / tot:.0f}  {100 * (tot - ac) / tot:.0f}")
save(fig, "jpeg_bit_budget.png")

# ------------------------------------------------------------------ 7. generation loss
n = 20
fig, ax = plt.subplots(figsize=(7, 4.4))
res = {}
for shift, col in ((0, "tab:green"), (3, "tab:red"), (4, "tab:orange")):
    r = jpeg.generation_loss(photo, n=n, quality=50, shift=shift)
    res[shift] = r
    ax.plot([d["gen"] for d in r], [d["psnr"] for d in r], "-o", ms=3, color=col,
            label="same block grid (shift 0)" if shift == 0 else f"shifted {shift} px per save")
ax.set_xlabel("generation (open -> shift -> save)")
ax.set_ylabel("PSNR vs original (dB)")
ax.set_title("Generation loss, quality 50, 4:2:0")
ax.grid(alpha=0.3)
ax.legend()
save(fig, "jpeg_generation_loss.png")
for s, r in res.items():
    print(f"[gen] shift {s}: gen1 {r[0]['psnr']:.2f} dB, gen2 {r[1]['psnr']:.2f}, gen5 {r[4]['psnr']:.2f}, "
          f"gen{n} {r[-1]['psnr']:.2f}; total loss after gen 1: {r[0]['psnr'] - r[-1]['psnr']:.2f} dB")

# ------------------------------------------------------------------ 8. clean vs noisy (L2 s15)
lin = scenes.load("sunset", 256)
rng = utils.rng(0)
noisy_lin = np.clip(lin + rng.normal(size=lin.shape) * (0.01 + 0.03 * np.sqrt(lin)), 0, 1).astype(np.float32)
clean, noisy = srgb(lin), srgb(noisy_lin)
q = 30
oc, ic = jpeg.roundtrip(clean, q)
on, inn = jpeg.roundtrip(noisy, q)
box = (150, 214, 40, 104)                              # dark part of the scene
gain = lambda x: np.clip((x - x.min()) / (np.percentile(x, 99) - x.min() + 1e-9), 0, 1)
fig, axs = plt.subplots(2, 4, figsize=(14, 7.2))
show(axs[0, 0], utils.crop(clean, box), "clean input")
show(axs[0, 1], utils.crop(oc, box), f"clean -> JPEG q={q}\n{ic['bpp']:.2f} bpp")
show(axs[0, 2], utils.crop(noisy, box), "noisy input")
show(axs[0, 3], utils.crop(on, box), f"noisy -> JPEG q={q}\n{inn['bpp']:.2f} bpp")
for ax, a, t in ((axs[1, 0], oc, "clean JPEG, contrast x4"), (axs[1, 2], on, "noisy JPEG, contrast x4")):
    show(ax, np.clip(0.5 + 4 * (utils.crop(a, box) - utils.crop(a, box).mean()), 0, 1), t)
for ax, a, t in ((axs[1, 1], oc - clean, "clean JPEG error x8"), (axs[1, 3], on - noisy, "noisy JPEG error x8")):
    show(ax, np.clip(0.5 + 8 * utils.crop(a, box), 0, 1), t)
save(fig, "jpeg_noise.png")
print(f"[noise] q={q}: clean {ic['bpp']:.3f} bpp ({ic['nbytes']} B), noisy {inn['bpp']:.3f} bpp ({inn['nbytes']} B), "
      f"x{inn['bpp'] / ic['bpp']:.2f}; blockiness clean {jpeg.blockiness(oc):.3f}, noisy {jpeg.blockiness(on):.3f}")
for qq in (30, 75, 95):
    a = jpeg.encode(clean, qq).nbytes
    b = jpeg.encode(noisy, qq).nbytes
    print(f"[noise] q={qq}: clean {a} B, noisy {b} B (x{b / a:.2f})")
flat = np.full_like(clean, 0.5)
print(f"[noise] flat grey image at q=75: {jpeg.encode(flat, 75).nbytes} B ({jpeg.encode(flat, 75).header_bytes} B of it header)")

# ------------------------------------------------------------------ 9. photo vs graphics
gfx = np.full((256, 256, 3), 1.0, np.float32)
cv2.rectangle(gfx, (20, 150), (110, 230), (0.9, 0.1, 0.1), -1)
cv2.circle(gfx, (180, 190), 40, (0.1, 0.2, 0.9), -1)
cv2.line(gfx, (10, 140), (245, 100), (0, 0, 0), 2)
cv2.putText(gfx, "JPEG 5176", (14, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
cv2.putText(gfx, "ringing near edges", (14, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
q = 20
og, ig = jpeg.roundtrip(gfx, q, "4:2:0")
op, ip = jpeg.roundtrip(photo, q, "4:2:0")
og4, ig4 = jpeg.roundtrip(gfx, q, "4:4:4")
box = (20, 84, 10, 106)
fig, axs = plt.subplots(2, 4, figsize=(14, 7.2))
show(axs[0, 0], photo, "photo")
show(axs[0, 1], op, f"photo q={q}: {ip['bpp']:.2f} bpp, {ip['psnr']:.1f} dB")
show(axs[0, 2], np.clip(0.5 + 4 * (op - photo), 0, 1), "photo error x4")
show(axs[0, 3], np.clip(0.5 + 4 * (og - gfx), 0, 1), "graphics error x4")
show(axs[1, 0], gfx, "graphics (text, flat fills)")
show(axs[1, 1], og, f"graphics q={q}: {ig['bpp']:.2f} bpp, {ig['psnr']:.1f} dB")
show(axs[1, 2], utils.crop(gfx, box), "original text crop")
show(axs[1, 3], utils.crop(og, box), "decoded text crop (ringing)")
save(fig, "jpeg_graphics.png")
print(f"[gfx] q={q}: photo {ip['bpp']:.2f} bpp {ip['psnr']:.2f} dB; graphics 4:2:0 {ig['bpp']:.2f} bpp {ig['psnr']:.2f} dB, "
      f"4:4:4 {ig4['bpp']:.2f} bpp {ig4['psnr']:.2f} dB")
import io
from PIL import Image
buf = io.BytesIO()
Image.fromarray(utils.to_uint8(gfx)).save(buf, format="PNG", optimize=True)
bufp = io.BytesIO()
Image.fromarray(utils.to_uint8(photo)).save(bufp, format="PNG", optimize=True)
print(f"[gfx] lossless PNG: graphics {buf.tell()} B ({8 * buf.tell() / 256 ** 2:.2f} bpp, "
      f"vs JPEG q=95 {jpeg.encode(gfx, 95).nbytes} B), photo {bufp.tell()} B ({8 * bufp.tell() / 256 ** 2:.2f} bpp, "
      f"vs JPEG q=95 {jpeg.encode(photo, 95).nbytes} B)")

# ------------------------------------------------------------------ 10. RAW vs JPEG size, EXIF
h, w = photo.shape[:2]
jb = {q: jpeg.encode(photo, q).nbytes for q in (50, 75, 95)}
raw12 = jpeg.raw_vs_jpeg_size(photo.shape, 12, jb[95])
fig, ax = plt.subplots(figsize=(7, 4))
labels = ["RAW 14-bit", "RAW 12-bit", "RGB 8-bit\nuncompressed", "JPEG q=95", "JPEG q=75", "JPEG q=50"]
sizes = [h * w * 14 / 8, h * w * 12 / 8, h * w * 3, jb[95], jb[75], jb[50]]
ax.bar(labels, [s / 1024 for s in sizes], color=["#555", "#777", "#999", "#1f77b4", "#4f97c9", "#8fbbe0"])
for i, s in enumerate(sizes):
    ax.text(i, s / 1024, f"{s / 1024:.0f}", ha="center", va="bottom", fontsize=8)
ax.set_ylabel("size (KiB), 256x256 image")
ax.set_title("RAW (one sample per pixel) vs JPEG")
ax.tick_params(axis="x", labelsize=8)
save(fig, "jpeg_raw_vs_jpeg.png")
for lab, s in zip(labels, sizes):
    print(f"[raw] {lab.replace(chr(10), ' '):22s} {s / 1024:7.1f} KiB")
print(f"[raw] 12-bit RAW / JPEG q=95: {raw12['ratio']:.1f}x; JPEG q=95 is {raw12['jpeg_bpp']:.2f} bpp")

with tempfile.TemporaryDirectory() as d:
    p = pathlib.Path(d) / "t.jpg"
    nb = jpeg.write_jpeg_with_exif(photo, p, 90, exif={"Make": "EE5176", "Model": "Toy ISP", "Software": "isp.jpeg",
                                                       "ExposureTime": 0.01, "FNumber": 2.8, "ISOSpeedRatings": 200})
    ex = jpeg.read_exif(p)
    print(f"[exif] file {nb} B; tags read back: " +
          ", ".join(f"{k}={ex[k]}" for k in ("Make", "Model", "Software", "ISOSpeedRatings") if k in ex))
