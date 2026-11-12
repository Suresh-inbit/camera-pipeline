"""One compact, cached experiment per stage (the "Experiment" expander of each stage page).

Every ``exp_<stage>(ctx)`` draws into the current Streamlit container.  Heavy work sits in
``st.cache_data`` functions keyed on ``prefix_key`` (scene + stage parameters up to this stage), so
moving a downstream slider or the zoom crop does not recompute them.  Figures are matplotlib (Agg)
and come back as PNG bytes.  ``ctx`` is a ``SimpleNamespace`` made by the app: ``res`` (the
PipelineResult), ``pl`` (the Pipeline), ``img`` (ground-truth scene or None for a RAW input),
``skey`` (scene key), ``size``, ``rec`` (this stage's record), ``prev`` (its input record),
``box`` / ``zoom`` (the shared crop), ``key`` (prefix key of this stage).
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import streamlit as st
from scipy import ndimage

from _common import draw_box, fig_png, plt, show_png
from isp import cfa as _cfa
from isp import demosaic as _demosaic
from isp import denoise as _denoise
from isp import display, jpeg, scenes, sensor, sharpen, tonemap, utils
from isp import pipeline as P
from isp import whitebalance as wb


# ==========================================================================
# helpers
# ==========================================================================
def _linear_view(a):
    """Linear float image/mosaic -> uint8 for display (sRGB curve)."""
    return display.render(np.asarray(a, np.float32), {"domain": "linear"})


def _rgb_of(rec_or_img, meta=None):
    """RGB float view of a record: mosaics are 2x2-binned (half resolution)."""
    img = np.asarray(getattr(rec_or_img, "image", rec_or_img))
    meta = getattr(rec_or_img, "meta", meta) or {}
    return P.bin_bayer(img, meta.get("cfa", "RGGB")) if img.ndim == 2 else img


def _crop_zoomed(u8, box, zoom):
    return display.crop_zoom(u8, box, zoom)


def _grid_images(items, ncols):
    cols = st.columns(ncols)
    for i, (cap, arr) in enumerate(items):
        with cols[i % ncols]:
            st.image(arr, caption=cap, width="stretch")


def _need_note(msg):
    st.info(msg)


# ==========================================================================
# scene
# ==========================================================================
@st.cache_data(show_spinner=False)
def _scene_grid():
    return {n: display.render(scenes.load(n, 96), {"domain": "scene"}) for n in scenes.NAMES}


def _pick_scene(name):
    st.session_state["scene_name"] = name
    st.session_state["use_up"] = False


def exp_scene(ctx):
    grid = _scene_grid()
    names = list(grid)
    for r in range(0, len(names), 5):
        cols = st.columns(5)
        for c, n in zip(cols, names[r:r + 5]):
            with c:
                st.image(grid[n], width="stretch")
                st.button(("* " if n == st.session_state.get("scene_name") else "") + n, key=f"pick|{n}",
                          on_click=_pick_scene, args=(n,), width="stretch")
    st.caption("Photographs come from scikit-image (linearised); the charts are synthetic: a colour chart, "
               "a zone plate (aliasing), a slanted edge (sharpening), a grey ramp (tone curves).")


# ==========================================================================
# illuminant: CIE xy diagram
# ==========================================================================
@st.cache_data(show_spinner=False)
def _fig_xy(cct, gains, method):
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    loc = wb.spectral_locus_xy()
    ax.plot(np.r_[loc[:, 0], loc[0, 0]], np.r_[loc[:, 1], loc[0, 1]], "k-", lw=1.2, label="spectral locus (420-650 nm)")
    Ts = np.geomspace(1500, 12000, 80)
    pk = wb.planckian_locus_xy(Ts)
    ax.plot(pk[:, 0], pk[:, 1], color="#E69F00", lw=2, label="Planckian locus")
    for T in (2000, 3200, 4000, 6500, 10000):
        x, y = wb.cct_to_xy(T)
        ax.plot(x, y, ".", color="#E69F00")
        ax.annotate(f"{T} K", (x, y), textcoords="offset points", xytext=(6, 5), fontsize=8)
    g = wb.srgb_gamut_xy()
    ax.fill(g[:, 0], g[:, 1], facecolor="none", edgecolor="#56B4E9", lw=1.2, ls="--", label="sRGB gamut")
    if cct:
        x, y = wb.cct_to_xy(cct)
        ax.plot(x, y, "*", ms=15, color="#D55E00", label=f"light source: {cct:.0f} K")
    if gains is not None:
        e = wb.gains_to_illuminant(np.array(gains)) * wb.illuminant_rgb(6504.0)
        ex, ey = wb.rgb_to_xy(e)
        ax.plot(ex, ey, "x", ms=11, mew=2.5, color="#009E73",
                label=f"WB estimate ({method}): {wb.xy_to_cct(ex, ey):.0f} K")
    ax.set_xlim(0.1, 0.7); ax.set_ylim(0.1, 0.9)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_aspect("equal"); ax.grid(alpha=.3)
    ax.legend(fontsize=7, loc="upper right")
    return fig_png(fig)


def exp_illuminant(ctx):
    cct = ctx.pl["illuminant"].params.get("cct")
    gains, method = None, None
    if "white_balance" in ctx.res.names and ctx.pl["white_balance"].enabled:
        m = ctx.res["white_balance"].meta
        if "wb_gains" in m and m.get("wb_method") != "none":
            gains, method = tuple(float(v) for v in m["wb_gains"]), m["wb_method"]
    c1, c2 = st.columns([3, 2])
    with c1:
        show_png(_fig_xy(None if cct is None else float(cct), gains, method))
    with c2:
        st.markdown("The red star is the light of this stage, the green cross is what the white-balance "
                    "stage *thinks* the light is (estimated from the picture). The distance between them is "
                    "the angular error of the white-balance stage; the colour cast it leaves is the part "
                    "of the quality drop that is not noise.")
        if cct:
            r = wb.illuminant_rgb(float(cct))
            st.caption(f"Colour of {cct:.0f} K relative to green: R {r[0]:.2f}, G 1, B {r[2]:.2f}")


# ==========================================================================
# cfa
# ==========================================================================
@st.cache_data(show_spinner=False)
def _fig_tiles():
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.2))
    for ax, (name, tile) in zip(axes, [("Bayer RGGB", _cfa.BAYER_TILE), ("Sony RGBE", _cfa.SONY_RGBE_TILE),
                                       ("Kodak RGBW", _cfa.KODAK_RGBW_TILE)]):
        reps = 4 // tile.shape[0]
        ax.imshow(_cfa.tile_colors(tile, reps), interpolation="nearest")
        big = np.tile(tile, (reps, reps))
        for (i, j), ch in np.ndenumerate(big):
            ax.text(j, i, ch, ha="center", va="center", fontsize=9, color="k")
        ax.set_title(name, fontsize=10); ax.axis("off")
    fig.tight_layout()
    return fig_png(fig)


def exp_cfa(ctx):
    show_png(_fig_tiles())
    rec = ctx.rec
    if rec.image.ndim == 2:
        r0, r1, c0, c1 = ctx.box
        r1, c1 = min(r1, r0 + 16), min(c1, c0 + 16)
        patch = rec.image[r0:r1, c0:c1]
        col = display.render(patch, {"domain": "scene", "cfa": _shift_pattern(rec.meta.get("cfa", "RGGB"), r0, c0)})
        gray = display.render(patch, {"domain": "scene", "cfa": "RGGB"}, mosaic_mode="gray")
        _grid_images([(f"coloured mosaic ({r1 - r0}x{c1 - c0} px at the crop)", display.crop_zoom(col, (0, r1 - r0, 0, c1 - c0), 12)),
                      ("the same samples as grey numbers", display.crop_zoom(gray, (0, r1 - r0, 0, c1 - c0), 12))], 2)
        st.caption("Only one of three channels exists at each pixel: the other two will be interpolated by the demosaic stage.")
    else:
        _need_note("The CFA stage is disabled, so the picture is still full RGB.")


def _shift_pattern(p, r0, c0):
    """Pattern string seen from an offset (r0, c0) (rows/cols mod 2)."""
    t = [[p[0], p[1]], [p[2], p[3]]]
    a = [[t[(i + r0) % 2][(j + c0) % 2] for j in range(2)] for i in range(2)]
    return "".join(a[0] + a[1])


# ==========================================================================
# exposure: SNR theory
# ==========================================================================
def _sensor_params(pl):
    p = pl["exposure"].params
    sp = p.get("sensor", "dslr")
    sp = sp if isinstance(sp, sensor.SensorParams) else sensor.PRESETS[sp]
    return replace(sp, **(p.get("sensor_kw") or {}))


@st.cache_data(show_spinner=False)
def _fig_snr(sp, t_now, flux):
    ts = np.geomspace(1e-4, 1.0, 120)
    vals = [0.02, 0.1, 0.5, 1.0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw=dict(width_ratios=[1.3, 1]))
    for v, c in zip(vals, ["#999999", "#56B4E9", "#E69F00", "#009E73"]):
        s = 20 * np.log10(sensor.snr_theory_params(v, ts, sp, flux))
        a1.plot(ts, s, color=c, label=f"scene value {v}")
        a1.plot([t_now], [20 * np.log10(sensor.snr_theory_params(v, t_now, sp, flux))], "o", color=c, ms=8)
    a1.axvline(t_now, color="k", ls=":", lw=1)
    a1.set_xscale("log"); a1.set_xlabel("exposure time (s)"); a1.set_ylabel("SNR (dB)")
    a1.set_title(f"{sp.name}: SNR = S / sqrt(S + D t + Nr^2)", fontsize=9); a1.grid(alpha=.3); a1.legend(fontsize=7)
    # noise-type breakdown at the current exposure: share of the variance
    S = np.array(vals) * flux * sp.qe * t_now
    shot, dark, read = S, np.full_like(S, sp.dark_current * t_now), np.full_like(S, sp.read_noise ** 2)
    tot = shot + dark + read
    x = np.arange(len(vals))
    a2.bar(x, shot / tot, color="#E69F00", label="shot (photon)")
    a2.bar(x, dark / tot, bottom=shot / tot, color="#CC79A7", label="dark current")
    a2.bar(x, read / tot, bottom=(shot + dark) / tot, color="#56B4E9", label="read noise")
    a2.set_xticks(x, [str(v) for v in vals]); a2.set_xlabel("scene value"); a2.set_ylabel("share of noise variance")
    a2.set_title(f"noise at t = {t_now * 1000:.1f} ms", fontsize=9); a2.legend(fontsize=7)
    fig.tight_layout()
    return fig_png(fig)


def exp_exposure(ctx):
    p = ctx.pl["exposure"].params
    sp = _sensor_params(ctx.pl)
    t, flux = float(p.get("exposure", 0.0125)), float(p.get("flux_at_white", sensor.DEFAULT_FLUX))
    show_png(_fig_snr(sp, t, flux))
    snr = 20 * np.log10(sensor.snr_theory_params(0.5, t, sp, flux))
    st.caption(f"At scene value 0.5 the theory gives {snr:.1f} dB with {0.5 * flux * sp.qe * t:.0f} signal electrons "
               f"(full well {sp.full_well:.0f}). Shot noise dominates bright pixels, read noise the shadows.")


# ==========================================================================
# adc
# ==========================================================================
@st.cache_data(show_spinner=False)
def _fig_adc(counts, edges, bl, wl, bits):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4))
    a1.stairs(counts, edges, fill=True, color="#56B4E9")
    a1.axvline(bl, color="k", ls=":", label="black level"); a1.axvline(wl, color="r", ls=":", label="white level")
    a1.set_yscale("log"); a1.set_xlabel("digital number (DN)"); a1.set_ylabel("pixels"); a1.legend(fontsize=7)
    a1.set_title("histogram of the RAW numbers", fontsize=9)
    x = np.linspace(0, 1, 400)
    for b, c in ((3, "#D55E00"), (bits, "#009E73")):
        L = 2 ** b - 1
        a2.plot(x, np.round(x * L) / L, color=c, drawstyle="steps-post", label=f"{b} bit ({2 ** b} levels)")
    a2.plot(x, x, "k:", lw=1)
    a2.set_xlim(0, 0.5); a2.set_ylim(0, 0.5); a2.set_xlabel("analog value"); a2.set_ylabel("quantised value")
    a2.set_title("quantisation (zoom on the shadows)", fontsize=9); a2.legend(fontsize=7)
    fig.tight_layout()
    return fig_png(fig)


def exp_adc(ctx):
    rec = ctx.rec
    m = rec.meta
    if m.get("domain") != "dn":
        return _need_note("The ADC stage is disabled or not at the sensor output; no RAW numbers to show.")
    sp = sensor.SensorParams(**m["sensor_params"])
    counts, edges = np.histogram(rec.image, bins=96, range=(0, sp.white_level))
    show_png(_fig_adc(counts, edges, float(sp.black_level), float(sp.white_level), int(sp.bits)))
    t, iso = float(m["exposure"]), float(m["iso"])
    dr = sensor.dynamic_range(sp.full_well, sp.dark_current, t, sp.read_noise)
    clipped = float(np.mean(rec.image >= sp.white_level - 0.5)) * 100
    c = st.columns(4)
    c[0].metric("levels", f"{2 ** sp.bits}")
    c[1].metric("electrons per DN", f"{1.0 / sp.gain(iso):.2f}")
    c[2].metric("dynamic range", f"{20 * np.log10(dr):.0f} dB", f"{np.log2(dr):.1f} stops", delta_color="off")
    c[3].metric("clipped pixels", f"{clipped:.2f} %")
    st.caption("Dynamic range = (full well - D t) / sqrt(D t + Nr^2) of the sensor preset (L2 s33). Higher ISO means fewer "
               "electrons per DN: the picture gets brighter, the SNR does not improve.")


# ==========================================================================
# black level: dark frame
# ==========================================================================
@st.cache_data(show_spinner=False)
def _dark_compare(_dn, _meta, key, _met, _bl_meta):
    sp = sensor.SensorParams(**_meta["sensor_params"])
    dark = sensor.master_dark(_dn.shape, _meta["exposure"], _meta["iso"], sp, n=16)
    plain = sensor.normalize(_dn, _meta)
    sub = sensor.normalize(sensor.subtract_dark(_dn, dark, _meta["black_level"]), _meta)
    out = dict(plain=_linear_view(plain), sub=_linear_view(sub), diff=display.difference(plain, sub, gain=4.0))
    if _met is not None:
        ref = _met.reference(plain, _bl_meta)
        out["psnr_plain"], out["psnr_sub"] = utils.psnr(ref, plain), utils.psnr(ref, sub)
    return out


def exp_black_level(ctx):
    m = ctx.prev.meta
    if m.get("domain") != "dn":
        return _need_note("This stage does not receive RAW numbers (ADC disabled or moved).")
    r = _dark_compare(np.asarray(ctx.prev.image), m, ctx.key, ctx.res.metrics, ctx.rec.meta)
    items = [("without dark frame", r["plain"]), ("minus master dark (16 frames)", r["sub"]),
             ("difference x4", r["diff"])]
    _grid_images([(c, _crop_zoomed(a, ctx.box, ctx.zoom)) for c, a in items], 3)
    if "psnr_plain" in r:
        st.caption(f"PSNR against the truth: {r['psnr_plain']:.2f} dB without, {r['psnr_sub']:.2f} dB with the dark frame. "
                   "The dark frame removes hot pixels and the dark-current pattern (fixed), not the random noise.")


# ==========================================================================
# hot pixels
# ==========================================================================
def exp_hot_pixels(ctx):
    a, b = np.asarray(ctx.prev.image), np.asarray(ctx.rec.image)
    if not ctx.pl["hot_pixels"].enabled:
        return _need_note("The stage is disabled: nothing is replaced.")
    changed = np.abs(a - b).max(axis=-1) > 0 if a.ndim == 3 else (a != b)
    sp = ctx.prev.meta.get("sensor_params")
    base = display.render(b, ctx.rec.meta, mosaic_mode="gray") if b.ndim == 2 else display.render(ctx.rec)
    over = (base.astype(np.float32) * 0.6).astype(np.uint8)
    n = int(changed.sum())
    if sp is not None and changed.shape == tuple(ctx.prev.image.shape[:2]):
        true = sensor.fixed_pattern(changed.shape, sensor.SensorParams(**sp))["hot"]
        tp, fp, fn = int((changed & true).sum()), int((changed & ~true).sum()), int((~changed & true).sum())
        d = lambda m_: ndimage.binary_dilation(m_, iterations=1)  # noqa: E731
        over[d(true & ~changed)] = (0, 120, 255)        # missed hot pixels
        over[d(changed & ~true)] = (255, 220, 0)        # false positives
        over[d(changed & true)] = (255, 40, 40)         # replaced and really hot
        c = st.columns(4)
        c[0].metric("replaced", n); c[1].metric("really hot (true positive)", tp)
        c[2].metric("false positives", fp); c[3].metric("missed", fn)
        cap = "red = hot pixel replaced, yellow = good pixel replaced (false positive), blue = hot pixel missed."
    else:
        over[ndimage.binary_dilation(changed)] = (255, 40, 40)
        st.metric("replaced pixels", n)
        cap = "red = replaced pixels."
    c1, c2 = st.columns(2)
    c1.image(over, caption=cap, width="stretch")
    c2.image(_crop_zoomed(over, ctx.box, ctx.zoom), caption="zoomed crop", width="stretch")
    st.caption("The threshold trades the two kinds of error: a low one removes real detail (yellow), a high one lets hot pixels through (blue).")


# ==========================================================================
# white balance
# ==========================================================================
@st.cache_data(show_spinner=False)
def _wb_compare(_img, meta_illum, key, percentile):
    rgb = _img
    rows = []
    true = wb.gains_for_illuminant(meta_illum) if meta_illum is not None else None
    cands = {"none": np.ones(3, np.float32), "gray world": wb.gray_world(rgb),
             "white patch": wb.white_patch(rgb, percentile)}
    if true is not None:
        cands["true gains"] = true
    scale = 0.5 / max(float(np.percentile((rgb * cands["none"]).mean(-1), 99)), 1e-6)
    imgs = {}
    for name, g in cands.items():
        err = (wb.angular_error(wb.gains_to_illuminant(g), wb.gains_to_illuminant(true)) if true is not None else np.nan)
        rows.append(dict(method=name, R=float(g[0]), G=float(g[1]), B=float(g[2]), angular_error_deg=float(err)))
        imgs[name] = _linear_view(np.clip(rgb * g * scale, 0, 1))
    return rows, imgs


def exp_white_balance(ctx):
    prev = ctx.prev
    rgb = _rgb_of(prev)
    perc = float(ctx.pl["white_balance"].params.get("percentile", 99))
    rows, imgs = _wb_compare(np.ascontiguousarray(rgb), prev.meta.get("illuminant"), ctx.key, perc)
    _grid_images([(k, v) for k, v in imgs.items()], len(imgs))
    import pandas as pd
    st.dataframe(pd.DataFrame(rows).set_index("method").round(3), width="stretch")
    st.caption("Gains are estimated on the 2x2-binned mosaic (or the RGB image). Gray world fails when the scene is not grey on "
               "average; white patch needs a bright neutral object; the true gains use the known light and give the best possible result.")


# ==========================================================================
# demosaic
# ==========================================================================
@st.cache_data(show_spinner="Comparing demosaicing methods ...")
def _demosaic_compare(_mosaic, pattern, key, _met, _meta):
    out, rows = {}, []
    ref = _met.reference(np.zeros(_mosaic.shape + (3,)), _meta) if _met is not None else None
    for m in _demosaic.METHODS:
        img = _demosaic.demosaic(_mosaic, pattern, m)
        out[m] = _linear_view(img)
        row = dict(method=m)
        if ref is not None:
            row["PSNR dB"] = utils.psnr(ref, np.clip(img, 0, 1))
            row["false colour"] = _demosaic.false_color(np.clip(img, 0, 1), ref)
        rows.append(row)
    return rows, out


def exp_demosaic(ctx):
    prev = ctx.prev
    if prev.image.ndim != 2:
        return _need_note("The demosaic stage receives RGB data here (it passes it through), so there is no mosaic to interpolate.")
    rows, imgs = _demosaic_compare(np.asarray(prev.image), prev.meta.get("cfa", "RGGB"), ctx.key, ctx.res.metrics, ctx.rec.meta)
    _grid_images([(k, _crop_zoomed(v, ctx.box, ctx.zoom)) for k, v in imgs.items()], 5)
    import pandas as pd
    df = pd.DataFrame(rows).set_index("method")
    st.dataframe(df.round(3), width="stretch")
    st.caption("PSNR is measured on the whole image against the ground truth, 'false colour' is the RMS chroma error (colour fringes and "
               "zipper patterns). Crops are the shared zoom window; use a zone plate or the colour chart to see the differences clearly.")


# ==========================================================================
# denoise
# ==========================================================================
@st.cache_data(show_spinner="Denoising with three methods ...")
def _denoise_compare(_rgb, key, _ref, sigma_d, sigma_r):
    runs = {"input": _rgb, "gaussian": _denoise.gaussian(_rgb, 1.2),
            "bilateral": _denoise.bilateral(_rgb, sigma_d, sigma_r), "luma_chroma": _denoise.luma_chroma(_rgb)}
    rows, imgs = [], {}
    for k, v in runs.items():
        row = dict(method=k, sigma=float(_denoise.estimate_noise(np.ascontiguousarray(v))))
        if _ref is not None:
            row["PSNR dB"] = utils.psnr(_ref, np.clip(v, 0, 1))
        rows.append(row)
        imgs[k] = _linear_view(v)
    return rows, imgs


def exp_denoise(ctx):
    prev = ctx.prev
    p = ctx.pl["denoise"].params
    rgb = _rgb_of(prev)
    ref = None
    if prev.image.ndim == 3 and ctx.res.metrics is not None:
        ref = np.asarray(ctx.res.metrics.reference(prev.image, ctx.rec.meta))
    else:
        st.caption("The input is a mosaic here: showing the 2x2-binned half-resolution RGB, no ground-truth PSNR.")
    sd = float(p.get("sigma_d", 1.5)) if p.get("method") == "bilateral" else 1.5
    sr = float(p.get("sigma_r", 0.05)) if p.get("method") == "bilateral" else 0.05
    rows, imgs = _denoise_compare(np.ascontiguousarray(rgb), ctx.key, ref, sd, sr)
    box = tuple(v // 2 for v in ctx.box) if prev.image.ndim == 2 else ctx.box
    _grid_images([(k, _crop_zoomed(v, box, ctx.zoom)) for k, v in imgs.items()], 4)
    import pandas as pd
    st.dataframe(pd.DataFrame(rows).set_index("method").round(4), width="stretch")
    st.markdown("**Bilateral weight window at one pixel** (L5 s7): neighbours across an edge get small weights.")
    H, W = rgb.shape[:2]
    c1, c2 = st.columns(2)
    row = c1.slider("pixel row", 4, H - 5, H // 2, key=f"exp|dn|row|{H}")
    col = c2.slider("pixel column", 4, W - 5, W // 2, key=f"exp|dn|col|{W}")
    w = _denoise.bilateral_weights(rgb, (row, col), sd, sr)
    r = w.shape[0] // 2
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(6, 2.8))
    patch = np.clip(rgb[max(row - r, 0):row + r + 1, max(col - r, 0):col + r + 1], 0, 1)
    a1.imshow(utils.linear_to_srgb(patch), interpolation="nearest"); a1.set_title("input patch", fontsize=9); a1.axis("off")
    im = a2.imshow(w, vmin=0, vmax=1, cmap="viridis", interpolation="nearest")
    a2.set_title(f"weights (sigma_d={sd:g}, sigma_r={sr:g})", fontsize=9); a2.axis("off")
    fig.colorbar(im, ax=a2, fraction=0.046); fig.tight_layout()
    show_png(fig_png(fig))


# ==========================================================================
# tone mapping
# ==========================================================================
@st.cache_data(show_spinner="Tone mapping with six operators ...")
def _fig_tone(_rgb, key):
    lum = np.clip(utils.luminance(np.clip(_rgb, 0, 1)), 0, 1)
    fig, axes = plt.subplots(2, 4, figsize=(11, 5))
    ax = axes[0, 0]
    edges = np.linspace(0, 1, 33)
    mid = 0.5 * (edges[1:] + edges[:-1])
    idx = np.clip(np.digitize(lum, edges) - 1, 0, 31)
    outs = {}
    for m in tonemap.METHODS:
        out = tonemap.tonemap(np.clip(_rgb, 0, 1), m)
        outs[m] = out
        ol = utils.luminance(out)
        curve = np.array([ol[idx == i].mean() if np.any(idx == i) else np.nan for i in range(32)])
        ax.plot(mid, curve, label=m)
    ax.plot([0, 1], [0, 1], "k:", lw=.8)
    ax.set_title("tone curves (mean output luminance\nper input bin)", fontsize=8); ax.legend(fontsize=6)
    ax.set_xlabel("linear input"); ax.set_ylabel("output")
    axes[0, 1].hist(lum.ravel(), bins=48, color="#999999"); axes[0, 1].set_title("input histogram (linear)", fontsize=8)
    for a, m in zip(list(axes.ravel())[2:], tonemap.METHODS):
        a.hist(utils.luminance(outs[m]).ravel(), bins=48, color="#009E73"); a.set_title(f"{m}: output", fontsize=8)
    fig.tight_layout()
    return fig_png(fig)


def exp_tonemap(ctx):
    rgb = _rgb_of(ctx.prev)
    show_png(_fig_tone(np.ascontiguousarray(rgb, dtype=np.float32), ctx.key))
    st.caption("Linear data has most pixels in the shadows; the sRGB curve and gamma spread them out, histogram equalisation makes the "
               "output histogram as flat as possible, CLAHE does it locally, 'local' compresses a smooth base layer (dodge and burn).")


# ==========================================================================
# sharpen: edge profile overshoot
# ==========================================================================
@st.cache_data(show_spinner=False)
def _fig_sharpen(method, sigma):
    edge = scenes.step_edge(128).mean(axis=2)
    lams = [0.0, 0.5, 1.0, 2.0, 3.0]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw=dict(width_ratios=[2, 1]))
    over = []
    for lam in lams:
        s = sharpen.sharpen(edge, method, lam, **({"sigma": sigma} if method == "unsharp" else {}))
        d, prof = sharpen.edge_profile(s)
        a1.plot(d, prof, label=f"lam {lam:g}")
        over.append(sharpen.overshoot(prof) * 100)
    a1.set_xlim(-8, 8); a1.set_xlabel("distance from the edge (px)"); a1.set_ylabel("value")
    a1.set_title(f"edge profile, {method}", fontsize=9); a1.legend(fontsize=7); a1.grid(alpha=.3)
    a2.bar([str(v) for v in lams], over, color="#CC79A7"); a2.set_xlabel("lambda"); a2.set_ylabel("overshoot (% of the step)")
    a2.set_title("halo size", fontsize=9)
    fig.tight_layout()
    return fig_png(fig)


def exp_sharpen(ctx):
    p = ctx.pl["sharpen"].params
    show_png(_fig_sharpen(p.get("method", "unsharp"), float(p.get("sigma", 1.0))))
    st.caption("A slanted step edge is sharpened with growing lambda: the edge gets steeper but over- and undershoots (halos). "
               "On real pictures the same kernel also amplifies the remaining noise.")


# ==========================================================================
# jpeg
# ==========================================================================
def _unit_rgb(rec):
    a = np.asarray(rec.image, np.float32)
    d = rec.meta.get("domain")
    if d == "electrons":
        a = a / float(rec.meta.get("full_well", max(float(a.max()), 1.0)))
    elif d == "dn":
        a = (a - rec.meta["black_level"]) / (rec.meta["white_level"] - rec.meta["black_level"])
    a = np.clip(a, 0, 1)
    return np.repeat(a[..., None], 3, axis=2) if a.ndim == 2 else a


@st.cache_data(show_spinner="Sweeping JPEG quality ...")
def _jpeg_sweep(_img, key, sub, quality):
    qs = [5, 10, 20, 35, 50, 65, 80, 90, 95, 99]
    rows = []
    for q in sorted(set(qs + [int(quality)])):
        _, info = jpeg.roundtrip(_img, q, sub)
        rows.append((q, info["bpp"], info["psnr"]))
    gl0 = jpeg.generation_loss(_img, 8, int(quality), 0, sub)
    gl1 = jpeg.generation_loss(_img, 8, int(quality), 3, sub)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    q_, b_, p_ = zip(*rows)
    axes[0].plot(q_, b_, "o-"); axes[0].plot([quality], [b_[q_.index(int(quality))]], "r*", ms=14)
    axes[0].set_xlabel("quality"); axes[0].set_ylabel("bits per pixel"); axes[0].set_title("rate vs quality", fontsize=9)
    axes[1].plot(b_, p_, "o-"); axes[1].plot([b_[q_.index(int(quality))]], [p_[q_.index(int(quality))]], "r*", ms=14)
    axes[1].set_xlabel("bits per pixel"); axes[1].set_ylabel("PSNR (dB)"); axes[1].set_title("rate - distortion", fontsize=9)
    axes[2].plot([g["gen"] for g in gl0], [g["psnr"] for g in gl0], "o-", label="same block grid")
    axes[2].plot([g["gen"] for g in gl1], [g["psnr"] for g in gl1], "s-", label="grid shifted 3 px")
    axes[2].set_xlabel("generation"); axes[2].set_ylabel("PSNR vs original (dB)"); axes[2].legend(fontsize=7)
    axes[2].set_title(f"generation loss at q={int(quality)}", fontsize=9)
    for a in axes:
        a.grid(alpha=.3)
    fig.tight_layout()
    return fig_png(fig)


@st.cache_data(show_spinner=False)
def _fig_block(_gray, key, i, j, quality):
    r = jpeg.block_before_after(_gray, (i, j), int(quality))
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    panels = [("pixels (0-255)", r["pixels"], "gray", "%d"), ("DCT coefficients", r["coefs"], "coolwarm", "%d"),
              (f"quantised (q={int(quality)})", r["quantised"], "coolwarm", "%d"), ("reconstruction", r["recon"], "gray", "%d")]
    for a, (t, arr, cm, f) in zip(axes, panels):
        lim = max(float(np.abs(arr).max()), 1.0)
        a.imshow(arr, cmap=cm, vmin=0 if cm == "gray" else -lim, vmax=255 if cm == "gray" else lim)
        for (y, x), v in np.ndenumerate(arr):
            a.text(x, y, f % round(float(v)), ha="center", va="center", fontsize=6, color="#d62728" if cm != "gray" else "#ff7f0e")
        a.set_title(t, fontsize=9); a.axis("off")
    fig.tight_layout()
    return fig_png(fig), float(np.abs(r["error"]).max()), int((r["quantised"] != 0).sum())


def exp_jpeg(ctx):
    img = np.ascontiguousarray(_unit_rgb(ctx.prev))
    p = ctx.pl["jpeg"].params
    q, sub = int(p.get("quality", 85)), p.get("subsampling", "4:2:0")
    gray = np.ascontiguousarray(utils.luminance(img))
    H, W = gray.shape
    c1, c2 = st.columns(2)
    bi = c1.slider("block row (8x8 blocks)", 0, H // 8 - 1, H // 16, key=f"exp|jpg|i|{H}")
    bj = c2.slider("block column", 0, W // 8 - 1, W // 16, key=f"exp|jpg|j|{W}")
    png, err, nz = _fig_block(gray, ctx.key, bi, bj, q)
    show_png(png)
    st.caption(f"One 8x8 block: only {nz} of 64 coefficients survive quantisation at q={q}; the largest pixel error is {err:.1f} levels (of 255).")
    show_png(_jpeg_sweep(img, ctx.key, sub, q))


EXPERIMENTS = {
    "scene": exp_scene, "illuminant": exp_illuminant, "cfa": exp_cfa, "exposure": exp_exposure, "adc": exp_adc,
    "black_level": exp_black_level, "hot_pixels": exp_hot_pixels, "white_balance": exp_white_balance,
    "demosaic": exp_demosaic, "denoise": exp_denoise, "tonemap": exp_tonemap, "sharpen": exp_sharpen, "jpeg": exp_jpeg,
}
