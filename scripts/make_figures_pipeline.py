"""Figures for docs/00_camera_pipeline.md and docs/08_pipeline_experiments.md.  Headless; prints
every number quoted in the docs.  Run from the project root:  python scripts/make_figures_pipeline.py

Writes docs/figures/pipeline_*.png:
  pipeline_overview.png          block diagram, thumbnails, zoomed crops and PSNR trace (README hero)
  pipeline_quality_trace.png     PSNR, SSIM, noise sigma and colour error per stage
  pipeline_ablation_stages.png   every processing stage switched off in turn
  pipeline_ablation_order.png    denoise / demosaic, sharpen / denoise, white balance orders
  pipeline_ablation_jpeg.png     clean vs noisy RAW into JPEG (L2 s15)
  pipeline_ablation_iso.png      ISO vs exposure time at equal brightness (L2 s39)
  pipeline_ablation_bits.png     ADC bit depth in a dark scene
"""
import sys, pathlib, time
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from isp import pipeline as P, display, scenes, utils

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
DPI = 90
SIZE = 256
SCENE = scenes.load("astronaut", SIZE)
CROP = (172, 204, 66, 98)             # fixed 32 x 32 crop of the overview (mission patch on the orange suit)
ABL = (150, 214, 56, 120)             # 64 x 64 crop of the ablation figures
IDS = list(P.STAGE_ORDER)
GC = {g: v["colour"] for g, v in P.GROUP_INFO.items()}
SHORT = {"scene": "Scene", "illuminant": "Light", "cfa": "CFA", "exposure": "Exposure", "adc": "ADC",
         "black_level": "Black level", "hot_pixels": "Hot pixels", "white_balance": "White bal.",
         "demosaic": "Demosaic", "denoise": "Denoise", "tonemap": "Tone map", "sharpen": "Sharpen",
         "jpeg": "JPEG"}
# a quiet sensor: huge photon count, no read noise or defects (only quantisation remains)
QUIET = {"sensor_kw": {"full_well": 1e7, "read_noise": 0.0, "dark_current": 0.0, "prnu": 0.0,
                       "dsnu": 0.0, "hot_pixel_fraction": 0.0}, "flux_at_white": 1e9}


def save(fig, name):
    fig.savefig(OUT / name, dpi=DPI)
    plt.close(fig)
    print("wrote", name)


def sim(scene=SCENE, **ov):
    return P.simulate(scene, **ov)


def crop_img(img_u8, box=CROP, scale=1):
    return display.crop_zoom(img_u8, box, scale)


def final_view(res):
    return display.render(res.final)


def fm(res, key):
    return res.final.metrics.get(key, float("nan"))


def domain_label(rec):
    m, d = rec.meta, rec.meta["domain"]
    mos = "mosaic" if rec.image.ndim == 2 else "RGB"
    if d == "scene":
        return f"light {mos}\nfloat"
    if d == "electrons":
        return f"electrons\n<= {int(m['full_well'] / 1000)}k e-"
    if d == "dn":
        return f"RAW {mos}\n{m['bits']} bit"
    if d == "linear":
        return f"linear {mos}\nfloat [0,1]"
    if d == "display":
        return "sRGB RGB\nfloat [0,1]"
    return f"JPEG file\n8 bit, {m['nbytes'] / 1000:.1f} kB"


# ==========================================================================
# 1. overview
# ==========================================================================
def fig_overview(res):
    n = len(IDS)
    fig = plt.figure(figsize=(16, 9.4))
    gs = fig.add_gridspec(5, n, height_ratios=[0.28, 1.0, 1.25, 1.25, 2.1], hspace=0.12, wspace=0.12,
                          left=0.035, right=0.99, top=0.975, bottom=0.07)
    # group bands
    spans, start = [], 0
    for i in range(1, n + 1):
        if i == n or P._G[IDS[i]] != P._G[IDS[start]]:
            spans.append((P._G[IDS[start]], start, i - 1))
            start = i
    for g, a, b in spans:
        ax = fig.add_subplot(gs[0, a:b + 1])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.0, 0.1), 1.0, 0.8, boxstyle="round,pad=0,rounding_size=0.15",
                                    fc=GC[g], ec="none", alpha=0.9))
        ax.text(0.5, 0.5, P.GROUP_INFO[g]["label"] + {"scene": "", "sensor": " (L2 s3)", "adc": "",
                                                       "processing": " (L2 s3: the order may vary)",
                                                       "storage": ""}[g],
                ha="center", va="center", fontsize=9, fontweight="bold", color="black")
    for k, sid in enumerate(IDS):
        rec = res[sid]
        ax = fig.add_subplot(gs[1, k])
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
        ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0,rounding_size=0.08",
                                    fc=GC[P._G[sid]], ec="black", lw=0.6, alpha=0.9))
        ax.text(0.5, 0.74, SHORT[sid], ha="center", va="center", fontsize=8.2, fontweight="bold")
        ax.text(0.5, 0.33, domain_label(rec), ha="center", va="center", fontsize=7, linespacing=1.15)
        if k < n - 1:
            ax.annotate("", xy=(1.2, 0.5), xytext=(0.96, 0.5), annotation_clip=False,
                        arrowprops=dict(arrowstyle="-|>", color="black", lw=0.9))
        ax = fig.add_subplot(gs[2, k])
        ax.imshow(display.render(rec, size=110), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(GC[P._G[sid]]); sp.set_linewidth(2)
        # fixed zoomed crop
        ax = fig.add_subplot(gs[3, k])
        full = display.render(rec)
        ax.imshow(crop_img(full), interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_edgecolor(GC[P._G[sid]]); sp.set_linewidth(2)
    fig.text(0.008, 0.69, "stage output", rotation=90, va="center", fontsize=8)
    fig.text(0.008, 0.49, "32x32 crop, zoomed", rotation=90, va="center", fontsize=8)
    # trace
    ax = fig.add_subplot(gs[4, :])
    xs = np.arange(n)
    p = np.array([res[s].metrics.get("psnr", np.nan) for s in IDS])
    p[~np.isfinite(p)] = np.nan
    ax.plot(xs, p, "o-", color="#222222", lw=1.6, ms=5, label="PSNR vs ground truth (stage's own domain)")
    pn = np.array([res[s].metrics.get("psnr_noise", np.nan) for s in IDS])
    ax.plot(xs, pn, "s--", color="#0072B2", lw=1.2, ms=4, label="PSNR vs noise-free sensor (mosaic stages)")
    for g, a, b in spans:
        ax.axvspan(a - 0.5, b + 0.5, color=GC[g], alpha=0.15, lw=0)
    for x, v in zip(xs, p):
        if np.isfinite(v):
            ax.annotate(f"{v:.1f}", (x, v), textcoords="offset points", xytext=(0, 6), ha="center",
                        fontsize=7.5)
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_xticks(xs); ax.set_xticklabels([SHORT[s] for s in IDS], fontsize=7.5)
    ax.set_ylabel("PSNR (dB)", fontsize=9)
    ax.set_ylim(8, 52)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=8, frameon=True)
    ax.set_title("quality trace: colour cast hurts first, white balance fixes it, noise and "
                 "demosaicing take a little, denoising gives some back", fontsize=8.5)
    save(fig, "pipeline_overview.png")


def fig_trace(res):
    xs = np.arange(len(IDS))
    fig, axs = plt.subplots(2, 2, figsize=(12, 6.4), sharex=True)
    def line(ax, key, title, fmt="o-", color="#222222", ylab=""):
        v = np.array([res[s].metrics.get(key, np.nan) for s in IDS], float)
        v[~np.isfinite(v)] = np.nan
        ax.plot(xs, v, fmt, color=color, lw=1.4, ms=4)
        ax.set_title(title, fontsize=9); ax.set_ylabel(ylab, fontsize=8); ax.grid(alpha=0.3)
        return v
    line(axs[0, 0], "psnr", "PSNR vs ground truth", ylab="dB")
    pn = np.array([res[s].metrics.get("psnr_noise", np.nan) for s in IDS], float)
    axs[0, 0].plot(xs, pn, "s--", color="#0072B2", ms=4, lw=1.1, label="vs noise-free sensor")
    axs[0, 0].legend(fontsize=7)
    line(axs[0, 1], "ssim", "SSIM vs ground truth")
    line(axs[1, 0], "sigma", "estimated noise sigma (MAD high-pass)", color="#D55E00", ylab="linear / display units")
    axs[1, 0].set_yscale("log")
    line(axs[1, 1], "cast_deg", "colour cast: mean colour vs truth", color="#CC79A7", ylab="degrees")
    for ax in axs[1]:
        ax.set_xticks(xs); ax.set_xticklabels([SHORT[s] for s in IDS], rotation=45, ha="right", fontsize=7.5)
    for ax in axs.ravel():
        for k, s in enumerate(IDS):
            ax.axvspan(k - 0.5, k + 0.5, color=GC[P._G[s]], alpha=0.10, lw=0)
    fig.tight_layout()
    save(fig, "pipeline_quality_trace.png")


# ==========================================================================
# 2. ablations
# ==========================================================================
def run_variant(**ov):
    return sim(**ov)


def row(name, res):
    f = res.final
    nb = f.meta.get("nbytes")
    return dict(name=name, psnr=f.metrics["psnr"], ssim=f.metrics["ssim"], nbytes=nb, res=res)


def ablation_stages(base):
    variants = [
        ("baseline", {}),
        ("hot pixels off", dict(enabled={"hot_pixels": False})),
        ("hot pixels off + dark frame", dict(enabled={"hot_pixels": False}, black_level={"dark_frame": True})),
        ("white balance off", dict(white_balance={"method": "none"})),
        ("demosaic: nearest", dict(demosaic={"method": "nearest"})),
        ("denoise off", dict(enabled={"denoise": False})),
        ("tone map: linear (no curve)", dict(tonemap={"method": "linear"})),
        ("sharpen off", dict(enabled={"sharpen": False})),
        ("JPEG off", dict(enabled={"jpeg": False})),
    ]
    rows = [row(n, base if n == "baseline" else run_variant(**ov)) for n, ov in variants]
    print("\n== ablation: processing stages switched off in turn (final PSNR / SSIM / bytes)")
    for r in rows:
        print(f"  {r['name']:30s} PSNR {r['psnr']:6.2f}  SSIM {r['ssim']:.3f}  bytes {r['nbytes']}")
    try:
        sim(enabled={"black_level": False})
    except ValueError as e:
        print("  black_level off ->", str(e)[:90])
    # also: JPEG of everything but denoise at stage level
    fig = plt.figure(figsize=(14, 8.6))
    gs = fig.add_gridspec(3, 5, height_ratios=[1, 1, 1.05], hspace=0.35, wspace=0.08)
    for k, r in enumerate(rows):
        ax = fig.add_subplot(gs[k // 5, k % 5])
        ax.imshow(crop_img(final_view(r["res"]), ABL, 1), interpolation="nearest")
        ax.set_title(f"{r['name']}\n{r['psnr']:.1f} dB, SSIM {r['ssim']:.2f}", fontsize=8)
        ax.axis("off")
    ax = fig.add_subplot(gs[1, 4]); ax.axis("off")
    ax.text(0, 0.5, "64x64 crop of the final\nimage. PSNR and SSIM are\nagainst the ground truth\nthrough the same tone curve\n"
            "(the linear tone map is\nscored against a linear\nreference).", fontsize=8, va="center")
    ax = fig.add_subplot(gs[2, :3])
    names = [r["name"] for r in rows]
    cols = ["#222222"] + ["#009E73"] * (len(rows) - 1)
    cols[names.index("tone map: linear (no curve)")] = "#BBBBBB"      # scored against a different reference
    ax.barh(names[::-1], [r["psnr"] for r in rows][::-1], color=cols[::-1])
    for i, r in enumerate(rows[::-1]):
        ax.text(r["psnr"] + 0.2, i, f"{r['psnr']:.1f}", va="center", fontsize=8)
    ax.set_xlim(12, 37); ax.set_xlabel("final PSNR (dB); grey bar: different reference, not comparable", fontsize=9); ax.tick_params(labelsize=8)
    ax.axvline(rows[0]["psnr"], color="#222222", lw=0.8, ls=":")
    ax = fig.add_subplot(gs[2, 3:])
    nb = [r["nbytes"] or np.nan for r in rows]
    ax.barh(names[::-1], nb[::-1], color="#CC79A7")
    ax.set_yticklabels([]); ax.set_xlabel("JPEG size (bytes; none when JPEG is off)", fontsize=9)
    ax.tick_params(labelsize=8)
    save(fig, "pipeline_ablation_stages.png")
    return rows


def hot_pixel_sweep():
    from isp import sensor
    sp = sensor.replace(sensor.PRESETS["dslr"], **P.DEFAULT_PARAMS["exposure"]["sensor_kw"])
    hot = sensor.fixed_pattern((SIZE, SIZE), sp)["hot"]
    print(f"\n== hot-pixel detector threshold sweep ({int(hot.sum())} true hot pixels of {SIZE * SIZE})")
    for name in ("astronaut", "chelsea"):
        sc = scenes.load(name, SIZE)
        for thr in (None, 0.2, 0.3, 0.4):
            r = sim(sc, hot_pixels={"threshold": thr}, enabled={"denoise": False, "sharpen": False, "jpeg": False})
            ch = r["black_level"].image != r["hot_pixels"].image
            print(f"  {name:9s} threshold {str(thr):5s} replaced {int(ch.sum()):4d}  true defects caught {int((ch & hot).sum()):2d}"
                  f"  PSNR vs noise-free {r['hot_pixels'].metrics['psnr_noise']:.2f} dB "
                  f"(before the stage {r['black_level'].metrics['psnr_noise']:.2f})")


def ablation_order():
    out = {}
    base = sim()
    # 1) denoise before vs after demosaic
    early = P.default_pipeline().move("denoise", before="demosaic")
    r_early = sim(pipeline=early)
    # 2) sharpen before vs after denoise
    sh_first = P.default_pipeline().move("sharpen", before="denoise")
    r_sh = sim(pipeline=sh_first)
    # 3) white balance on the mosaic vs after demosaic (oracle gains: same gains in both)
    wb_m = sim(white_balance={"method": "true"})
    wb_late = sim(pipeline=P.default_pipeline(white_balance={"method": "true"}).move("white_balance", after="demosaic"))
    bil = dict(demosaic={"method": "bilinear"}, white_balance={"method": "true"})
    wb_m_bil = sim(**bil)
    wb_late_bil = sim(pipeline=P.default_pipeline(**bil).move("white_balance", after="demosaic"))
    wbg_m = base
    wbg_late = sim(pipeline=P.default_pipeline().move("white_balance", after="demosaic"))
    pairs = [("denoise: after demosaic (default)", base, "denoise: before demosaic (CFA planes)", r_early),
             ("sharpen: after denoise, display (default)", base, "sharpen: before denoise, linear", r_sh),
             ("white balance on mosaic ('true' gains)", wb_m, "white balance after demosaic ('true')", wb_late),
             ("WB on mosaic, bilinear demosaic", wb_m_bil, "WB after demosaic, bilinear", wb_late_bil)]
    print("\n== order experiments (final PSNR / SSIM / bytes)")
    for a_n, a, b_n, b in pairs:
        for n, r in ((a_n, a), (b_n, b)):
            print(f"  {n:46s} PSNR {fm(r, 'psnr'):6.2f}  SSIM {fm(r, 'ssim'):.3f}  bytes {r.final.meta['nbytes']}")
        va, vb = final_view(a), final_view(b)
        print(f"    mean abs difference of the two final images: {np.abs(va.astype(float) - vb).mean():.2f} /255")
    print("  white-patch gains on the mosaic:", np.round(wbg_m['white_balance'].meta['wb_gains'], 3),
          " after demosaic:", np.round(wbg_late['white_balance'].meta['wb_gains'], 3),
          " PSNR", round(fm(wbg_m, 'psnr'), 2), round(fm(wbg_late, 'psnr'), 2))
    print("  default (white patch) gains mosaic:", np.round(base['white_balance'].meta['wb_gains'], 3))
    fig, axs = plt.subplots(4, 3, figsize=(11, 13.8))
    for i, (a_n, a, b_n, b) in enumerate(pairs):
        va, vb = final_view(a), final_view(b)
        box = ABL
        axs[i, 0].imshow(crop_img(va, box), interpolation="nearest")
        axs[i, 0].set_title(f"{a_n}\n{fm(a, 'psnr'):.2f} dB, SSIM {fm(a, 'ssim'):.3f}", fontsize=8)
        axs[i, 1].imshow(crop_img(vb, box), interpolation="nearest")
        axs[i, 1].set_title(f"{b_n}\n{fm(b, 'psnr'):.2f} dB, SSIM {fm(b, 'ssim'):.3f}", fontsize=8)
        axs[i, 2].imshow(crop_img(display.difference(va, vb, 6.0), box), interpolation="nearest")
        axs[i, 2].set_title("difference x6", fontsize=8)
        for ax in axs[i]:
            ax.axis("off")
    fig.tight_layout()
    save(fig, "pipeline_ablation_order.png")


def ablation_jpeg():
    qs = (30, 50, 70, 85, 95)
    noisy = {q: sim(jpeg={"quality": q}, enabled={"denoise": False}) for q in qs}
    noisy_dn = {q: sim(jpeg={"quality": q}) for q in qs}
    clean = {q: sim(exposure=QUIET, adc={"iso": 100}, white_balance={"method": "true"}, jpeg={"quality": q},
                    enabled={"denoise": False}) for q in qs}
    print("\n== clean vs noisy RAW into JPEG (bytes at equal quality)")
    for q in qs:
        print(f"  q={q}: clean {clean[q].final.meta['nbytes']:6d} B   noisy {noisy[q].final.meta['nbytes']:6d} B   "
              f"noisy+denoise {noisy_dn[q].final.meta['nbytes']:6d} B   ratio noisy/clean "
              f"{noisy[q].final.meta['nbytes'] / clean[q].final.meta['nbytes']:.2f}   "
              f"PSNR clean {fm(clean[q], 'psnr'):.1f} noisy {fm(noisy[q], 'psnr'):.1f} denoised {fm(noisy_dn[q], 'psnr'):.1f}")
    fig, axs = plt.subplots(1, 4, figsize=(15, 3.9), gridspec_kw=dict(width_ratios=[1, 1, 1, 1.5]))
    box = ABL
    for ax, (n, r) in zip(axs[:3], (("clean RAW (quiet sensor)", clean[85]), ("noisy RAW", noisy[85]),
                                    ("noisy RAW + denoise", noisy_dn[85]))):
        ax.imshow(crop_img(final_view(r), box), interpolation="nearest")
        ax.set_title(f"{n}, JPEG q=85\n{r.final.meta['nbytes']} bytes, {r.final.meta['bpp']:.2f} bpp", fontsize=9)
        ax.axis("off")
    ax = axs[3]
    ax.plot(qs, [clean[q].final.meta["nbytes"] / 1000 for q in qs], "o-", color="#009E73", label="clean RAW")
    ax.plot(qs, [noisy[q].final.meta["nbytes"] / 1000 for q in qs], "o-", color="#D55E00", label="noisy RAW")
    ax.plot(qs, [noisy_dn[q].final.meta["nbytes"] / 1000 for q in qs], "o--", color="#0072B2", label="noisy RAW + denoise")
    ax.set_xlabel("JPEG quality"); ax.set_ylabel("file size (kB)"); ax.grid(alpha=0.3); ax.legend(fontsize=8)
    ax.set_title("noise is expensive to store (L2 s15)", fontsize=9)
    fig.tight_layout()
    save(fig, "pipeline_ablation_jpeg.png")


def ablation_iso():
    combos = [(200, 0.2), (400, 0.1), (800, 0.05), (1600, 0.025), (3200, 0.0125), (6400, 0.00625)]
    nodef = {"sensor_kw": {"hot_pixel_fraction": 0.0}}          # isolate the noise physics from defects
    rs = []
    print("\n== ISO vs exposure time at equal brightness (no hot pixels)")
    for iso, t in combos:
        r = sim(adc={"iso": iso}, exposure={"exposure": t, **nodef}, enabled={"hot_pixels": False})
        m = r["demosaic"].image.mean()
        rs.append((iso, t, r))
        print(f"  ISO {iso:5d}  t = 1/{round(1 / t):4d} s  mean brightness {m:.3f}  "
              f"{1e5 * 0.4 * t:.0f} e- at white  final PSNR {fm(r, 'psnr'):.2f}  SSIM {fm(r, 'ssim'):.3f}  demosaic PSNR "
              f"{r['demosaic'].metrics['psnr']:.2f}  clipped {np.mean(r['adc'].image >= r['adc'].meta['white_level']) * 100:.2f}%")
    # electrons at scene white (G): flux * qe * t
    fig, axs = plt.subplots(1, 7, figsize=(16, 3.2), gridspec_kw=dict(width_ratios=[1] * 6 + [2.2]))
    box = ABL
    for ax, (iso, t, r) in zip(axs[:6], rs):
        ax.imshow(crop_img(final_view(r), box), interpolation="nearest")
        ax.set_title(f"ISO {iso}, 1/{round(1 / t)} s\n{fm(r, 'psnr'):.1f} dB", fontsize=9)
        ax.axis("off")
    ax = axs[6]
    ax.semilogx([t for _, t, _ in rs], [fm(r, "psnr") for _, _, r in rs], "o-", color="#222222", label="final")
    ax.semilogx([t for _, t, _ in rs], [r["demosaic"].metrics["psnr"] for _, _, r in rs], "s--", color="#0072B2",
                label="linear RGB (before tone map)")
    ax.set_xlabel("exposure time (s), ISO x t = const"); ax.set_ylabel("PSNR (dB)"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8); ax.set_title("same brightness, less noise with longer exposure", fontsize=9)
    fig.tight_layout()
    save(fig, "pipeline_ablation_iso.png")


def ablation_bits():
    dark = (SCENE * 0.04).astype(np.float32)
    print("\n== ADC bit depth, dark scene (scene x 0.04, ISO 100, 0.5 s, digital gain x16 to brighten)")
    rs = []
    for b in (8, 10, 12, 14):
        r = sim(dark, adc={"iso": 100, "bits": b}, exposure={"exposure": 0.5, "sensor_kw": {"hot_pixel_fraction": 0.0}},
                black_level={"digital_gain": 16.0}, enabled={"hot_pixels": False, "denoise": False})
        rs.append((b, r))
        dn = r["adc"].image
        levels = len(np.unique(dn))
        print(f"  {b:2d} bit: distinct DN values {levels:5d}  signal range {np.ptp(dn):7.0f} DN  PSNR linear RGB "
              f"{r['demosaic'].metrics['psnr']:.2f}  final {fm(r, 'psnr'):.2f}  SSIM {fm(r, 'ssim'):.3f}  "
              f"bytes {r.final.meta['nbytes']}  sigma(demosaic) {r['demosaic'].metrics['sigma']:.4f}")
    fig, axs = plt.subplots(1, 5, figsize=(15, 3.4), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1.4]))
    box = ABL
    for ax, (b, r) in zip(axs[:4], rs):
        ax.imshow(crop_img(final_view(r), box), interpolation="nearest")
        ax.set_title(f"{b}-bit ADC\n{fm(r, 'psnr'):.1f} dB, {len(np.unique(r['adc'].image))} DN values", fontsize=9)
        ax.axis("off")
    ax = axs[4]
    ax.plot([b for b, _ in rs], [fm(r, "psnr") for _, r in rs], "o-", color="#222222", label="final")
    ax.plot([b for b, _ in rs], [r["demosaic"].metrics["psnr"] for _, r in rs], "s--", color="#0072B2",
            label="linear RGB")
    ax.set_xticks([8, 10, 12, 14]); ax.set_xlabel("ADC bits"); ax.set_ylabel("PSNR (dB)"); ax.grid(alpha=0.3)
    ax.legend(fontsize=8); ax.set_title("bit depth matters in the shadows", fontsize=9)
    fig.tight_layout()
    save(fig, "pipeline_ablation_bits.png")


# ==========================================================================
def main():
    t0 = time.time()
    res = sim()
    print("== default run: astronaut", SIZE, "x", SIZE)
    print("   default params:")
    for s in IDS:
        print(f"     {s:14s}", P.DEFAULT_PARAMS[s])
    print(f"\n   {'stage':14s}{'domain':10s}{'psnr':>8s}{'ssim':>7s}{'sigma':>8s}{'cast':>7s}{'psnr_noise':>11s}{'ms':>8s}")
    for r in res:
        m = r.metrics
        print(f"   {r.name:14s}{r.meta['domain']:10s}{m.get('psnr', float('nan')):8.2f}{m.get('ssim', float('nan')):7.3f}"
              f"{m.get('sigma', float('nan')):8.4f}{m.get('cast_deg', float('nan')):7.2f}"
              f"{m.get('psnr_noise', float('nan')):11.2f}{r.seconds * 1000:8.1f}")
    print("   white balance gains", np.round(res["white_balance"].meta["wb_gains"], 3),
          "angular error", round(res["white_balance"].meta["wb_angular_error"], 2), "deg")
    print("   JPEG:", res.final.meta["nbytes"], "bytes,", round(res.final.meta["bpp"], 2), "bpp;",
          "raw DN range", res["adc"].image.min(), res["adc"].image.max())
    print("   total simulate time", round(sum(r.seconds for r in res), 2), "s")
    # demosaic choice and white balance method on the default scene
    print("\n== default-choice checks (linear RGB at the demosaic stage, no denoise)")
    for dm in ("bilinear", "malvar", "median"):
        r = sim(demosaic={"method": dm}, enabled={"denoise": False, "sharpen": False, "jpeg": False})
        print(f"  demosaic {dm:9s} PSNR {r['demosaic'].metrics['psnr']:.2f}  SSIM {r['demosaic'].metrics['ssim']:.3f}")
    for wb in ("none", "gray_world", "white_patch", "true"):
        r = sim(white_balance={"method": wb}, enabled={"denoise": False, "sharpen": False, "jpeg": False})
        print(f"  white balance {wb:11s} angular error {r['white_balance'].meta.get('wb_angular_error', float('nan')):6.2f} deg"
              f"  PSNR at demosaic {r['demosaic'].metrics['psnr']:.2f}")
    # incremental re-run timing
    t = time.perf_counter(); r2 = P.simulate(SCENE, previous=res, sharpen={"lam": 1.0}); dt_inc = time.perf_counter() - t
    t = time.perf_counter(); P.simulate(SCENE, sharpen={"lam": 1.0}); dt_full = time.perf_counter() - t
    print(f"\n== incremental re-run (sharpen slider): {dt_inc * 1000:.0f} ms vs full {dt_full * 1000:.0f} ms; "
          f"reused {sum(r.reused for r in r2)} of {len(r2)} stages")
    fig_overview(res)
    fig_trace(res)
    ablation_stages(res)
    hot_pixel_sweep()
    ablation_order()
    ablation_jpeg()
    ablation_iso()
    ablation_bits()
    print(f"\ndone in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
