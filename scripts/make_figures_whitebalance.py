"""Figures for doc 03 (white balance).  Headless; prints the numbers quoted in the doc."""
import sys
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from isp import scenes, utils
from isp import whitebalance as W

OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SIZE = 256
srgb = lambda x: utils.linear_to_srgb(np.clip(x, 0, 1))
ILL = (3200, 4000, 7500)
SCENES = ("chart", "astronaut", "coffee", "sunset")
CARD = (SIZE - 48, SIZE - 16, 16, 48)          # a neutral grey card pasted into photos


def scene_with_neutral(name):
    """Scene plus the box of a patch known to be neutral (the card, or chart patch 20)."""
    if name == "chart":
        return scenes.patches(SIZE), scenes.patch_boxes(SIZE)[20]
    img = scenes.load(name, SIZE).copy()
    r0, r1, c0, c1 = CARD
    img[r0:r1, c0:c1] = 0.4
    return img, CARD


def estimates(lit, box):
    return {"manual patch": W.from_patch(lit, box), "gray world": W.gray_world(lit),
            "white patch": W.white_patch(lit, 99)}


def err(gains, T):
    return W.angular_error(W.gains_to_illuminant(gains), W._illum_relative(T))


def match_exposure(out, ref):
    """Display only: scale so the mean luminance equals the reference's."""
    return out * utils.luminance(ref).mean() / max(utils.luminance(out).mean(), 1e-9)


# ---------------------------------------------------------------- 1. one scene, four lights
print("illuminant colours (linear sRGB, G=1):")
for T in (3200, 4000, 6504, 7500):
    print(f"  {T} K: {np.round(W.illuminant_rgb(T), 3)}  relative-to-D65 {np.round(W._illum_relative(T), 3)}")
temps = (3200, 4000, 6504, 7500)
fig, axs = plt.subplots(2, 4, figsize=(14, 7.4))
for row, name in enumerate(("astronaut", "patches")):
    base = scenes.load(name, SIZE)
    for ax, T in zip(axs[row], temps):
        ax.imshow(srgb(W.apply_illuminant(base, T)))
        ax.set_title(f"{name} under {T} K", fontsize=10)
        ax.axis("off")
fig.tight_layout()
fig.savefig(OUT / "whitebalance_illuminants.png", dpi=90)
plt.close(fig)

# ---------------------------------------------------------------- 2. CIE xy diagram
loc = W.spectral_locus_xy()
tri = W.srgb_gamut_xy()
fig, ax = plt.subplots(figsize=(7.2, 7))
ax.plot(loc[:, 0], loc[:, 1], "k-", lw=1.5, label="spectral locus (Wyman fit)")
ax.plot([loc[0, 0], loc[-1, 0]], [loc[0, 1], loc[-1, 1]], "k--", lw=1, label="line of purples")
for nm in (450, 480, 500, 520, 550, 580, 600, 640):
    p = W.spectral_locus_xy(nm, nm)[0]
    ax.annotate(str(nm), p, textcoords="offset points", xytext=(-14 if nm < 520 else 5, 3), fontsize=7)
t = np.append(tri[:, 0], tri[0, 0]), np.append(tri[:, 1], tri[0, 1])
ax.plot(*t, color="tab:green", lw=1.3, label="sRGB triangle")
Ts = np.geomspace(1700, 20000, 200)
pl = W.planckian_locus_xy(Ts)
ax.plot(pl[:, 0], pl[:, 1], color="tab:orange", lw=2, label="Planckian locus")
for T in (2000, 3000, 4000, 5000, 6500, 10000):
    p = np.array(W.cct_to_xy(T))
    ax.plot(*p, "o", color="tab:orange", ms=4)
    ax.annotate(f"{T} K", p, textcoords="offset points", xytext=(6, 6), fontsize=8, color="tab:orange")
ax.plot(0.3127, 0.3290, "k*", ms=11, label="D65 white point")

# true vs estimated illuminants of the experiment (astronaut + card under 3200 K)
img, box = scene_with_neutral("astronaut")
lit = W.apply_illuminant(img, 3200)
truth = W._illum_relative(3200)
ax.plot(*W.rgb_to_xy(truth), "o", mfc="none", mec="k", ms=11, mew=2, label="true illuminant (3200 K)")
mk = {"manual patch": ("s", "tab:blue"), "gray world": ("x", "tab:red"), "white patch": ("+", "tab:purple")}
xy_est = {}
for k, g in estimates(lit, box).items():
    xy_est[k] = W.rgb_to_xy(W.gains_to_illuminant(g))
    ax.plot(*xy_est[k], mk[k][0], color=mk[k][1], ms=9, mew=2, label=f"{k} estimate")
ax.set_xlim(-0.02, 0.8)
ax.set_ylim(-0.02, 0.9)
ax.set_aspect("equal")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.legend(fontsize=7, loc="upper right")
ax.set_title("CIE 1931 xy chromaticity diagram")
fig.tight_layout()
fig.savefig(OUT / "whitebalance_xy_diagram.png", dpi=90)
plt.close(fig)
print("\nxy diagram: D65 black body xy", np.round(W.cct_to_xy(6504), 4), " 3200 K", np.round(W.cct_to_xy(3200), 4))
print("sRGB primaries xy:", np.round(tri, 3).tolist())
print("true 3200 K illuminant xy", np.round(W.rgb_to_xy(truth), 4),
      {k: np.round(v, 4).tolist() for k, v in xy_est.items()})
print("CCT of true illuminant xy:", round(W.xy_to_cct(*W.rgb_to_xy(truth))),
      " gray-world estimate CCT:", round(W.xy_to_cct(*xy_est["gray world"])))
c = W.cie_cmf(np.arange(360.0, 831.0)).sum(0)
print("equal-energy white from CMF fit:", np.round(c[:2] / c.sum(), 4), " y peak", round(float(loc[:, 1].max()), 4))
Tt = np.linspace(2500, 10000, 16)
rt = [abs(W.xy_to_cct(*W.cct_to_xy(T)) - T) / T for T in Tt]
print("cct round-trip max rel error 2500-10000 K:", f"{max(rt):.4f}",
      " McCamy at 5000 K:", round(W.mccamy_cct(*W.cct_to_xy(5000))))

# ---------------------------------------------------------------- 3. rgb chromaticity before/after
rs = np.random.default_rng(0)
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
gt = img
manual = W.apply_gains(lit, W.from_patch(lit, box))
for ax, (a, title) in zip(axs, ((gt, "ground-truth scene"), (lit, "under 3200 K"), (manual, "after manual-patch WB"))):
    c = W.rgb_chromaticity(a).reshape(-1, 2)
    c = c[rs.choice(len(c), 6000, replace=False)]
    ax.scatter(c[:, 0], c[:, 1], s=1, c="k", alpha=0.25)
    ax.plot(1 / 3, 1 / 3, "r+", ms=14, mew=2)
    m = W.rgb_chromaticity(a).reshape(-1, 2).mean(0)
    ax.plot(*m, "bo", ms=6)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_xlabel("r = R/(R+G+B)")
    ax.set_ylabel("g = G/(R+G+B)")
    ax.set_title(f"{title}\nmean (r,g) = ({m[0]:.3f}, {m[1]:.3f})", fontsize=9)
    print(f"rgb chromaticity {title}: mean r,g = {m.round(3)}")
fig.tight_layout()
fig.savefig(OUT / "whitebalance_rgb_chromaticity.png", dpi=90)
plt.close(fig)

# ---------------------------------------------------------------- 4. three estimators, one cast image
print("\nmethod comparison, astronaut + grey card under 3200 K:")
gt = img
ests = estimates(lit, box)
fig, axs = plt.subplots(1, 5, figsize=(17, 3.8))
panels = [(lit, "input, 3200 K"), ]
for k, g in ests.items():
    e = err(g, 3200)
    print(f"  {k:13s} gains {np.round(g, 3)}  angular error {e:.2f} deg")
    panels.append((match_exposure(W.apply_gains(lit, g), gt), f"{k}\nerror {e:.2f} deg"))
panels.append((gt, "ground truth"))
for ax, (a, t) in zip(axs, panels):
    ax.imshow(srgb(a))
    ax.set_title(t, fontsize=9)
    ax.axis("off")
r0, r1, c0, c1 = box
axs[0].add_patch(plt.Rectangle((c0, r0), c1 - c0, r1 - r0, fill=False, ec="yellow", lw=1.5))
fig.tight_layout()
fig.savefig(OUT / "whitebalance_methods.png", dpi=90)
plt.close(fig)
print("  no-correction angular error:", round(W.angular_error(np.ones(3), truth), 2), "deg")

# ---------------------------------------------------------------- 5. gray-world failure on the sunset
sun = scenes.sunset(SIZE)
m = sun.reshape(-1, 3).mean(0)
gw = W.gray_world(sun)
fixed = W.apply_gains(sun, gw)
chart = scenes.patches(SIZE)
print("\nsunset mean RGB (R normalised to 1):", np.round(m / m[0], 3),
      " gray-world gains", np.round(gw, 3))
e_sun = W.angular_error(W.gains_to_illuminant(gw), np.ones(3))
e_chart = W.angular_error(W.gains_to_illuminant(W.gray_world(chart)), np.ones(3))
print(f"gray-world error with true illuminant neutral: sunset {e_sun:.1f} deg, chart {e_chart:.1f} deg")
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
axs[0].imshow(srgb(sun)); axs[0].set_title("sunset (true light = neutral)")
axs[1].imshow(srgb(match_exposure(fixed, sun))); axs[1].set_title(f"after gray world\nerror {e_sun:.1f} deg")
axs[2].imshow(srgb(W.apply_gains(chart, W.gray_world(chart)))); axs[2].set_title(f"chart after gray world\nerror {e_chart:.1f} deg")
for a in axs:
    a.axis("off")
fig.tight_layout()
fig.savefig(OUT / "whitebalance_grayworld_failure.png", dpi=90)
plt.close(fig)

# ---------------------------------------------------------------- 6. error table
methods = ("manual patch", "gray world", "white patch")
table = {}
rows = []
for name in SCENES:
    base, box = scene_with_neutral(name)
    for T in ILL:
        lit_ = W.apply_illuminant(base, T)
        ests_ = estimates(lit_, box)
        row = [err(ests_[m_], T) for m_ in methods]
        table[(name, T)] = row
        rows.append((name, T, row))
print("\nangular errors (deg), scene / illuminant: manual, gray world, white patch")
for name, T, row in rows:
    print(f"  {name:10s} {T:5d} K  " + "  ".join(f"{v:6.2f}" for v in row))
for j, m_ in enumerate(methods):
    print(f"  mean {m_:13s}: {np.mean([r[2][j] for r in rows]):.2f} deg")
arr = np.array([r[2] for r in rows])
fig, ax = plt.subplots(figsize=(7.5, 5.2))
im = ax.imshow(arr, cmap="viridis_r", vmin=0, vmax=max(20, arr.max()), aspect="auto")
ax.set_xticks(range(3), methods)
ax.set_yticks(range(len(rows)), [f"{n} @ {T} K" for n, T, _ in rows], fontsize=8)
for i in range(arr.shape[0]):
    for j in range(3):
        ax.text(j, i, f"{arr[i, j]:.1f}", ha="center", va="center", fontsize=8,
                color="w" if arr[i, j] > 10 else "k")
fig.colorbar(im, label="angular error (deg)")
ax.set_title("Angular error of the estimated illuminant")
fig.tight_layout()
fig.savefig(OUT / "whitebalance_error_table.png", dpi=90)
plt.close(fig)

# ---------------------------------------------------------------- 7. extra numbers for the doc
print("\nWB on the linear mosaic vs after gamma (hue shift of a mid-grey card under 3200 K):")
card = np.full((1, 1, 3), 0.18, np.float32)
lit_card = W.apply_illuminant(card, 3200)
g = W.gains_for_illuminant(3200)
lin = W.apply_gains(lit_card, g)[0, 0]
enc = utils.linear_to_srgb(lit_card)
wrong = (enc * g)[0, 0]
print("  gains on linear data -> neutral:", np.round(lin / lin[1], 3), " gains after sRGB encoding:", np.round(wrong / wrong[1], 3))
