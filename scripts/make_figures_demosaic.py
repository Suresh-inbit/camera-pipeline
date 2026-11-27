"""Figures and numbers for docs/02_color_sensing_demosaicing.md (lecture 3).  Headless."""
import sys, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isp import cfa, demosaic as D, scenes, utils

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
disp = lambda x: utils.linear_to_srgb(np.clip(x, 0, 1))      # linear -> display for plotting
LABEL = {"nearest": "nearest", "bilinear": "bilinear", "bicubic": "bicubic",
         "median": "median (Freeman)", "malvar": "Malvar-He-Cutler"}


def show(ax, img, title, fs=9):
    img = np.clip(img, 0, 1)
    ax.imshow(img, cmap="gray" if img.ndim == 2 else None, vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(title, fontsize=fs)
    ax.axis("off")


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=90)
    plt.close(fig)


t0 = time.time()
photo = scenes.photo("astronaut", 256)

# ---------------------------------------------------------------- 1. CFA patterns + mosaic view
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for ax, (name, tile, reps) in zip(axs, [("Bayer (RGGB)", cfa.BAYER_TILE, 3),
                                        ("Sony RGB+E: better colour", cfa.SONY_RGBE_TILE, 2),
                                        ("Kodak RGB+W: less noise", cfa.KODAK_RGBW_TILE, 2)]):
    ax.imshow(cfa.tile_colors(tile, reps), interpolation="nearest")
    h, w = tile.shape
    for i in range(h * reps):
        for j in range(w * reps):
            ax.text(j, i, tile[i % h, j % w], ha="center", va="center", fontsize=11, fontweight="bold",
                    color="k" if tile[i % h, j % w] in "WE" else "w")
    ax.set_title(name, fontsize=10)
    ax.axis("off")
save(fig, "demosaic_cfa_patterns.png")
bayer_green = cfa.masks((256, 256))[..., 1].mean()
print(f"Bayer sample fractions R,G,B = {cfa.masks((256, 256)).mean(axis=(0, 1)).round(3).tolist()}")

box = (132, 164, 100, 132)                   # 32x32 crop with fine detail (helmet ring)
m = cfa.mosaic(photo)
fig, axs = plt.subplots(1, 4, figsize=(14, 3.8))
show(axs[0], disp(photo), "scene (what we want)")
axs[0].add_patch(plt.Rectangle((box[2] - .5, box[0] - .5), box[3] - box[2], box[1] - box[0], fill=False, ec="r", lw=1.5))
show(axs[1], utils.crop(disp(m), box), "grey mosaic crop (RAW as stored)")
show(axs[2], utils.crop(cfa.colorize(disp(m)), box), "same samples in filter colours")
show(axs[3], utils.crop(disp(D.demosaic(m, "RGGB", "bilinear")), box), "after bilinear demosaic")
save(fig, "demosaic_mosaic_view.png")

# ---------------------------------------------------------------- 2. zone plate
z = scenes.zone_plate(256)
zm = cfa.mosaic(z)
fig, axs = plt.subplots(2, 3, figsize=(14, 9))
show(axs[0, 0], z, "grey zone plate (ground truth)", 10)
fc_zone = {}
for ax, k in zip(axs.ravel()[1:], D.METHODS):
    out = D.demosaic(zm, "RGGB", k)
    fc_zone[k] = D.false_color(out, z)
    show(ax, out, f"{LABEL[k]}: false colour {fc_zone[k]:.3f}", 10)
save(fig, "demosaic_zoneplate.png")
print("zone plate false_color:", {k: round(v, 4) for k, v in fc_zone.items()})

# ---------------------------------------------------------------- 3. photo crops per method
# the 48x48 window (stride 16) where bilinear makes the largest chroma error
bil = D.demosaic(m, "RGGB", "bilinear")
best, box2 = -1, None
for r in range(0, 256 - 48, 16):
    for c in range(0, 256 - 48, 16):
        b = (r, r + 48, c, c + 48)
        e = D.false_color(utils.crop(bil, b), utils.crop(photo, b))
        if e > best:
            best, box2 = e, b
print("detail window (r0, r1, c0, c1):", box2, "bilinear false_color there", round(best, 4))
fig, axs = plt.subplots(2, 3, figsize=(14, 9))
show(axs[0, 0], utils.crop(disp(photo), box2), "ground truth", 10)
crop_stats = {}
for ax, k in zip(axs.ravel()[1:], D.METHODS):
    out = D.demosaic(m, "RGGB", k)
    cr, rf = utils.crop(out, box2), utils.crop(photo, box2)
    crop_stats[k] = (utils.psnr(cr, rf), D.false_color(cr, rf))
    show(ax, disp(cr), f"{LABEL[k]}: {crop_stats[k][0]:.1f} dB, fc {crop_stats[k][1]:.3f}", 10)
save(fig, "demosaic_crops.png")
print("detail crop (PSNR dB, false colour):", {k: (round(a, 2), round(b, 4)) for k, (a, b) in crop_stats.items()})

# L3 s28 layout: linear interpolation vs median-filtered interpolation, a scene with thin dark lines
scn = scenes.photo("coffee", 256)
sm = cfa.mosaic(scn)
lin, med = D.demosaic(sm, "RGGB", "bilinear"), D.demosaic(sm, "RGGB", "median")
# fine-detail window for coffee
bb, bbox = -1, None
for r in range(0, 256 - 64, 16):
    for c in range(0, 256 - 64, 16):
        b = (r, r + 64, c, c + 64)
        e = D.false_color(utils.crop(lin, b), utils.crop(scn, b))
        if e > bb:
            bb, bbox = e, b
fig, axs = plt.subplots(2, 2, figsize=(10, 10))
show(axs[0, 0], disp(lin), f"linear interpolation (fc {D.false_color(lin, scn):.3f})", 10)
show(axs[0, 1], disp(med), f"median-filtered interpolation (fc {D.false_color(med, scn):.3f})", 10)
show(axs[1, 0], utils.crop(disp(lin), bbox), "linear, zoom", 10)
show(axs[1, 1], utils.crop(disp(med), bbox), "median-filtered, zoom", 10)
for ax in axs[0]:
    ax.add_patch(plt.Rectangle((bbox[2] - .5, bbox[0] - .5), 64, 64, fill=False, ec="r", lw=1.5))
save(fig, "demosaic_linear_vs_median.png")
print("coffee: whole image false_color linear/median:", round(D.false_color(lin, scn), 4), round(D.false_color(med, scn), 4),
      "window", bbox, "zoom fc", round(D.false_color(utils.crop(lin, bbox), utils.crop(scn, bbox)), 4),
      round(D.false_color(utils.crop(med, bbox), utils.crop(scn, bbox)), 4))

# ---------------------------------------------------------------- 4. table across the photos
rows = {k: [] for k in D.METHODS}
for name in scenes.PHOTOS:
    ref = scenes.photo(name, 256)
    mm = cfa.mosaic(ref)
    for k in D.METHODS:
        out = D.demosaic(mm, "RGGB", k)
        rows[k].append((utils.psnr(out, ref), D.false_color(out, ref)))
hdr = "| method | " + " | ".join(f"{n} PSNR / fc" for n in scenes.PHOTOS) + " | mean PSNR | mean fc |"
lines = [hdr, "|" + "---|" * (len(scenes.PHOTOS) + 3)]
for k in D.METHODS:
    cells = " | ".join(f"{p:.2f} / {f:.4f}" for p, f in rows[k])
    lines.append(f"| {k} | {cells} | {np.mean([p for p, _ in rows[k]]):.2f} | {np.mean([f for _, f in rows[k]]):.4f} |")
print("\nMarkdown table (PSNR in dB of linear data, fc = false_color):\n" + "\n".join(lines) + "\n")
fig, axs = plt.subplots(1, 2, figsize=(12, 4))
w = 0.16
for i, k in enumerate(D.METHODS):
    axs[0].bar(np.arange(4) + (i - 2) * w, [p for p, _ in rows[k]], w, label=LABEL[k])
    axs[1].bar(np.arange(4) + (i - 2) * w, [f for _, f in rows[k]], w, label=LABEL[k])
for ax, t in zip(axs, ["PSNR vs ground truth (dB, higher better)", "false colour (RMS Cb, Cr error, lower better)"]):
    ax.set_xticks(range(4)); ax.set_xticklabels(scenes.PHOTOS); ax.set_title(t, fontsize=10)
axs[0].set_ylim(20, None)
axs[0].legend(fontsize=8)
save(fig, "demosaic_table.png")

# ---------------------------------------------------------------- 5. luma vs chroma blur
pic = utils.linear_to_srgb(scenes.photo("coffee", 256))        # display-referred, as on the slides
sig = 4.0
fig, axs = plt.subplots(1, 4, figsize=(14, 3.8))
show(axs[0], pic, "original")
lc = {}
for ax, ch in zip(axs[1:], ["Cb", "Cr", "Y"]):
    out = D.blur_channel(pic, ch, sig)
    lc[ch] = (utils.psnr(out, pic), utils.ssim(out, pic))
    show(ax, out, f"{ch} blurred (sigma {sig:g}): SSIM {lc[ch][1]:.3f}")
save(fig, "demosaic_luma_chroma.png")
print("luma/chroma blur (PSNR dB, SSIM):", {k: (round(a, 2), round(b, 4)) for k, (a, b) in lc.items()})
both = D.blur_channel(pic, "CbCr", sig)
print("CbCr both blurred: PSNR %.2f SSIM %.4f" % (utils.psnr(both, pic), utils.ssim(both, pic)))

# ---------------------------------------------------------------- 6. colour sensing technologies
scene = scenes.photo("astronaut", 256)
sigma = 0.02
mv = (0.0, 4.0)
tech = [
    ("ground truth", scene),
    ("field sequential\n(scene moves 4 px/exposure)", D.field_sequential(scene, motion=mv)),
    (f"3 sensor\n(noise {sigma})", D.three_sensor(scene, noise=sigma, seed=0)),
    (f"stacked (Foveon-like)\n(noise {sigma})", D.stacked_sensor(scene, noise=sigma, seed=0)),
    (f"CFA + bilinear\n(noise {sigma})", D.cfa_sensor(scene, "RGGB", "bilinear", noise=sigma, seed=0)),
]
zbox = (110, 174, 90, 154)
fig, axs = plt.subplots(2, 5, figsize=(16, 7))
stats = {}
for j, (name, out) in enumerate(tech):
    show(axs[0, j], disp(out), name)
    show(axs[1, j], utils.crop(disp(out), zbox), "zoom")
    stats[name.split("\n")[0]] = (utils.psnr(out, scene), D.false_color(out, scene),
                                  float(np.sqrt(np.mean((out - scene) ** 2))))
save(fig, "demosaic_sensing.png")
print("sensing tech (PSNR dB, false colour, RMS error):")
for k, v in stats.items():
    print(f"  {k:28s} {v[0]:6.2f} dB  fc {v[1]:.4f}  rms {v[2]:.4f}")
inv = np.linalg.inv(D.STACK_MIX)
print("stacked: cond(mix) = %.1f, noise gain per channel = %s" % (np.linalg.cond(D.STACK_MIX),
                                                                    np.sqrt((inv ** 2).sum(axis=1)).round(2).tolist()))
print("stacked noise-free RMS error: %.2e; static field-sequential RMS error: %.2e" % (
    np.sqrt(np.mean((D.stacked_sensor(scene) - scene) ** 2)),
    np.sqrt(np.mean((D.field_sequential(scene) - scene) ** 2))))
print(f"done in {time.time() - t0:.1f}s")
