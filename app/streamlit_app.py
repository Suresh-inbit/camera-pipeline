"""Interactive camera-pipeline explorer (EE 5176 Computational Photography).

Run:  streamlit run app/streamlit_app.py

The pipeline diagram at the top of every page is the navigation: click a stage to open its page.
Fallbacks: Prev / Next buttons, the sidebar stage selector (same session-state key ``stage``) and
deep links ``?stage=<id>``.
"""
from __future__ import annotations

import hashlib
import pathlib
import sys
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

import _common as C  # noqa: E402
from _common import OVERVIEW, ROOT, plt  # noqa: E402
import stage_params  # noqa: E402
from experiments import EXPERIMENTS  # noqa: E402
from nav import new_click, pipeline_nav  # noqa: E402
from isp import display, scenes  # noqa: E402
from isp import pipeline as P  # noqa: E402

st.set_page_config(page_title="Camera pipeline explorer", layout="wide")

VIEW_KEYS = ("v_zoom", "v_r", "v_c", "v_gain")
C.init_state()
ss = st.session_state

# --------------------------------------------------------------------------
# deep link (read once) and view state that must survive pages where its widgets are not drawn
# --------------------------------------------------------------------------
if "stage" not in ss:
    q = st.query_params.get("stage")
    ss["stage"] = q if q in C.valid_pages(ss["pipeline"]) else OVERVIEW
if ss["stage"] not in C.valid_pages(ss["pipeline"]):
    ss["stage"] = OVERVIEW
for _k in VIEW_KEYS:
    if _k in ss:
        ss[_k] = ss[_k]

# --------------------------------------------------------------------------
# sidebar: scene, size, seed, uploads
# --------------------------------------------------------------------------
with st.sidebar:
    st.title("Camera pipeline")
    st.caption("Simulate a digital camera, stage by stage (EE 5176).")
    scene_name = st.selectbox("Scene", scenes.NAMES, key="scene_name")
    size = st.selectbox("Image size (px)", C.SIZES, index=1, key="size")
    seed = st.number_input("Random seed", 0, 10_000, 0, 1, key="seed",
                           help="Seed of the shot noise (exposure) and read noise (ADC, seed + 1).")
    up = st.file_uploader("Upload your own image", type=["png", "jpg", "jpeg"], key="up_img",
                          help="Centre-cropped to a square of the chosen size, linearised, and fed in at the 'scene' stage.")
    use_up = up is not None and st.checkbox("Use the uploaded image", value=True, key="use_up")
    raw_up = None
    if C.rawpy_available():
        raw_up = st.file_uploader("Upload a RAW file", key="up_raw",
                                  help="Enters the pipeline at 'black_level' (no ground truth, so no PSNR).")
    else:
        st.caption("RAW upload: `pip install rawpy` enables it.")

# --------------------------------------------------------------------------
# input image and the (incremental) pipeline run
# --------------------------------------------------------------------------
pl = ss["pipeline"]
pl["exposure"].params["seed"] = int(seed)
pl["adc"].params["seed"] = int(seed) + 1

raw_mode = False
meta0, start, scene_img = None, None, None
if raw_up is not None:
    data = raw_up.getvalue()
    mosaic0, meta0 = C.load_raw_cached(data, hashlib.sha1(data).hexdigest())
    img, start, raw_mode = mosaic0, "black_level", True
    skey = ("raw", hashlib.sha1(data).hexdigest())
else:
    if up is not None and use_up:
        data = up.getvalue()
        dig = hashlib.sha1(data).hexdigest()
        scene_img = C.load_uploaded_image(data, dig, int(size))
        skey = ("upload", dig, int(size))
    else:
        scene_img = C.load_scene(scene_name, int(size))
        skey = (scene_name, int(size))
    img = scene_img

res = C.update_result(img, meta0, start, with_metrics=not raw_mode, scene_key=skey)
pl = ss["pipeline"]            # may have been restored after a rejected order
H = int(res.records[0].image.shape[0])

# --------------------------------------------------------------------------
# the diagram (navigation)
# --------------------------------------------------------------------------
thumbs = C.thumbnails(res, ss["thumb_cache"])
payload = C.build_nav_payload(res, pl, ss["stage"], thumbs)
click = pipeline_nav(payload)
if isinstance(click, dict) and click.get("nonce") and click["nonce"] != ss["nav_seen"]:
    target = new_click(click, ss["nav_seen"])
    ss["nav_seen"] = click["nonce"]
    if target in C.valid_pages(pl) and target != ss["stage"]:
        ss["stage"] = target
        st.query_params["stage"] = target
        st.rerun()

info = ss["run_info"]
st.caption(f"Last run: recomputed {len(info['recomputed'])} of {info['n_total']} stages in {info['seconds'] * 1000:.0f} ms"
           + (f" (reused: {', '.join(info['reused'])})" if info["reused"] else "") + ".  Click a box to open that stage.")
if ss["run_error"]:
    st.warning(f"That change was rejected and the previous pipeline restored: {ss['run_error']}")

# sidebar stage selector (fallback navigation), drawn after the diagram so a click can set the key first
_pages = C.valid_pages(pl)
st.sidebar.selectbox("Go to", _pages, key="stage",
                     format_func=lambda s: "Overview" if s == OVERVIEW else f"{_pages.index(s)}. {P.STAGE_INFO[s]['label']}")
if st.query_params.get("stage") != ss["stage"]:
    st.query_params["stage"] = ss["stage"]


def _go(stage: str):
    ss["stage"] = stage
    st.query_params["stage"] = stage


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def fmt(v, spec="{:.2f}", unit=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "n/a"
    if isinstance(v, float) and np.isinf(v):
        return "inf"
    return spec.format(v) + unit


@st.cache_data(show_spinner=False)
def fig_trace(labels, psnr, ssim, colours):
    fig, ax = plt.subplots(figsize=(11, 3.6))
    x = np.arange(len(labels))
    ax.bar(x, np.nan_to_num(np.minimum(psnr, 80)), color=colours, alpha=.9)
    ax.set_xticks(x, labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("PSNR vs ground truth (dB)")
    ax2 = ax.twinx()
    ax2.plot(x, ssim, "ko-", ms=4, lw=1.2)
    ax2.set_ylabel("SSIM"); ax2.set_ylim(0, 1.02)
    ax.grid(axis="y", alpha=.3)
    fig.tight_layout()
    return C.fig_png(fig)


def stage_chip(stage: str):
    info = P.STAGE_INFO[stage]
    g = P.GROUP_INFO[info["group"]]
    st.markdown(f"<span style='background:{g['colour']};color:#fff;padding:2px 12px;border-radius:12px;"
                f"font-size:0.85em;font-weight:600'>{g['label']}</span> &nbsp; <span style='opacity:.7'>"
                f"slides: {info['slides']}</span>", unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Overview
# --------------------------------------------------------------------------
def page_overview():
    st.header("Overview")
    st.markdown("This app simulates a digital camera as a chain of 13 stages, from the scene through the sensor and ADC to "
                "processing and JPEG storage. Click any box above to open a stage, change its parameters and see what it does to "
                "the picture and to the quality (PSNR / SSIM against the ground truth).")
    st.caption("The processing order may vary between cameras (L2 s3): switch stages off or try the order presets below.")
    final = res.final
    c1, c2 = st.columns(2)
    c1.image(display.render(final), caption=f"final image ({final.label}), {final.meta.get('nbytes', 0) / 1024:.0f} kB"
             if final.meta.get("nbytes") else f"final image ({final.label})", width="stretch")
    if scene_img is not None:
        c2.image(C.scene_image(scene_img), caption="ground-truth scene (linear, shown with the sRGB curve)", width="stretch")
    else:
        c2.info("RAW input: there is no ground truth, so no quality metrics.")

    recs = [r for r in res]
    if scene_img is not None:
        st.subheader("Quality trace")
        ps = tuple(float(r.metrics.get("psnr", np.nan)) for r in recs)
        ss_ = tuple(float(r.metrics.get("ssim", np.nan)) for r in recs)
        cols = tuple(P.GROUP_INFO[r.group]["colour"] for r in recs)
        st.image(fig_trace(tuple(r.label for r in recs), ps, ss_, cols), width="stretch")
        st.caption("Bars: PSNR after each stage (colour = box of the camera diagram); line: SSIM. Before white balance the PSNR mostly "
                   "measures the colour cast, not noise; see the 'PSNR vs noise-free' of the sensor stages on each page.")
    st.subheader("Stages")
    rows = []
    for r in recs:
        m = r.metrics
        rows.append({"id": r.name, "label": r.label, "enabled": pl[r.name].enabled, "domain": r.meta.get("domain"),
                     "shape": "x".join(str(s) for s in r.image.shape), "time ms": round(r.seconds * 1000, 1),
                     "PSNR dB": m.get("psnr"), "SSIM": m.get("ssim"), "reused": r.reused})
    st.dataframe(pd.DataFrame(rows).set_index("id"), width="stretch")

    st.subheader("Pipeline controls")
    st.markdown("**Enable / disable stages** (a disabled stage passes its input through; `black_level` cannot be skipped).")
    cols = st.columns(5)
    for i, s in enumerate(pl.stages):
        with cols[i % 5]:
            st.checkbox(P.STAGE_INFO[s.name]["label"], value=s.enabled, key=f"en|{s.name}",
                        on_change=C.set_enabled, args=(s.name,))
    st.markdown("**Order presets** (docs/08): " + " , ".join(f"`{n}`" for n in pl.names))
    bcols = st.columns(len(C.ORDER_PRESETS) + 1)
    cur = C.order_name(pl)
    for col, (key, (label, _)) in zip(bcols, C.ORDER_PRESETS.items()):
        col.button(label, key=f"order|{key}", on_click=C.set_order, args=(key,), type="primary" if cur == key else "secondary",
                   width="stretch")
    bcols[-1].button("Reset to defaults", key="reset", on_click=C.reset_pipeline, width="stretch")
    st.button("Open the first stage", key="first", on_click=_go, args=(pl.names[0],))


# --------------------------------------------------------------------------
# Stage page
# --------------------------------------------------------------------------
def page_stage(stage: str):
    info = P.STAGE_INFO[stage]
    pages = C.valid_pages(pl)
    i = pages.index(stage)

    top = st.columns([1, 1, 6])
    top[0].button("< Prev", key="prev", on_click=_go, args=(pages[max(i - 1, 0)],), disabled=i == 0, width="stretch")
    top[1].button("Next >", key="next", on_click=_go, args=(pages[min(i + 1, len(pages) - 1)],), disabled=i == len(pages) - 1, width="stretch")
    top[2].markdown(f"Stage {i} of {len(pages) - 1}")
    st.header(info["label"])
    stage_chip(stage)
    st.markdown(info["description"])

    if stage not in res.names:
        st.info("This stage is before the RAW entry point (`black_level`), so it does not exist for an uploaded RAW file.")
        return
    rec = res[stage]
    prev = C.input_record(res, stage)

    # shared zoom / crop (before and after use the same window)
    left, right = st.columns([1, 3])
    with left:
        st.subheader("Parameters")
        st.checkbox("stage enabled", value=pl[stage].enabled, key=f"en|{stage}", on_change=C.set_enabled, args=(stage,))
        shown = stage_params.render_controls(stage, H, has_camera_wb=bool(res.input_meta and "camera_wb" in (res.input_meta or {})))
        if not shown and stage != "scene":
            st.caption("No parameters.")
        if stage == "exposure":
            st.caption("The random seed is in the sidebar.")
        st.subheader("View")
        ss.setdefault("v_zoom", 4)
        ss.setdefault("v_gain", 4)
        zoom = st.select_slider("zoom", [2, 4, 8, 16], key="v_zoom")
        side = max(8, H // zoom)
        ss["v_r"] = int(min(ss.get("v_r", (H - side) // 2), H - side))
        ss["v_c"] = int(min(ss.get("v_c", (H - side) // 2), H - side))
        r0 = st.slider("crop row", 0, H - side, step=2, key="v_r")
        c0 = st.slider("crop column", 0, H - side, step=2, key="v_c")
        gain = st.slider("difference gain", 1, 16, key="v_gain")
    box = (r0, r0 + side, c0, c0 + side)

    before_u8, after_u8 = display.render(prev), display.render(rec)
    with right:
        a, b = st.columns(2)
        a.image(C.draw_box(before_u8, box), caption=f"before: output of '{prev.name}' ({C.domain_text(prev)})",
                width="stretch")
        b.image(C.draw_box(after_u8, box), caption=f"after: {info['label']} output ({C.domain_text(rec)})", width="stretch")
        if not pl[stage].enabled:
            st.warning("This stage is disabled: it passes its input through, so before and after are identical.")

    st.subheader("Zoomed crop")
    z1, z2, z3 = st.columns(3)
    z1.image(display.crop_zoom(before_u8, box, zoom), caption="before (zoom)", width="stretch")
    z2.image(display.crop_zoom(after_u8, box, zoom), caption="after (zoom)", width="stretch")
    z3.image(display.crop_zoom(display.difference(before_u8, after_u8, gain=float(gain)), box, zoom),
             caption=f"difference x{gain} (black = identical)", width="stretch")

    st.subheader("Histograms")
    h1, h2 = st.columns(2)
    for col, r_, cap in ((h1, prev, "before"), (h2, rec, "after")):
        arr = display.render(r_, mosaic_mode="gray")
        arr = arr[..., 0] if np.asarray(r_.image).ndim == 2 else arr
        edges, counts = display.histograms(arr, bins=64)
        names = ["R", "G", "B"] if counts.shape[0] == 3 else ["grey level"]
        df = pd.DataFrame(np.log1p(counts.T), columns=names, index=np.round(edges[:-1], 3))
        col.caption(f"{cap}: log(1 + pixel count) per value")
        col.line_chart(df, color=["#d62728", "#2ca02c", "#1f77b4"][:len(names)], height=200)

    st.subheader("Metrics")
    m = rec.metrics
    pm = prev.metrics if hasattr(prev, "metrics") else {}
    tiles = []
    if m:
        d = (m["psnr"] - pm["psnr"]) if ("psnr" in m and "psnr" in pm and np.isfinite(m["psnr"]) and np.isfinite(pm["psnr"])) else None
        tiles.append(("PSNR vs reference", fmt(m.get("psnr"), "{:.2f}", " dB"), None if d is None else f"{d:+.2f} dB vs input"))
        tiles.append(("SSIM", fmt(m.get("ssim"), "{:.3f}"), None))
        tiles.append(("noise sigma (estimated)", fmt(m.get("sigma"), "{:.4f}"), None))
        tiles.append(("colour cast", fmt(m.get("cast_deg"), "{:.1f}", " deg"), None))
        if "wb_angular_error" in m:
            tiles.append(("WB angular error", fmt(m["wb_angular_error"], "{:.1f}", " deg"), None))
        if "psnr_noise" in m:
            tiles.append(("PSNR vs noise-free", fmt(m["psnr_noise"], "{:.2f}", " dB"), None))
    if "nbytes" in rec.meta:
        tiles.append(("JPEG size", f"{rec.meta['nbytes'] / 1024:.1f} kB", f"{rec.meta['bpp']:.2f} bits/pixel"))
    tiles.append(("time", f"{rec.seconds * 1000:.1f} ms", "reused (not recomputed)" if rec.reused else "recomputed"))
    for row in range(0, len(tiles), 4):
        cols = st.columns(4)
        for col, (lab, val, delta) in zip(cols, tiles[row:row + 4]):
            col.metric(lab, val, delta, delta_color="off")
    if not m:
        st.caption("No ground-truth metrics for this input.")

    with st.expander("Experiment: " + info["label"], expanded=False):
        if st.toggle("Run the experiment", value=True, key="exp_on"):
            ctx = SimpleNamespace(res=res, pl=pl, img=scene_img, skey=skey, size=H, rec=rec, prev=prev, box=box, zoom=zoom,
                                  key=C.prefix_key(skey, pl, stage))
            C.safe(stage, EXPERIMENTS[stage], ctx)

    with st.expander("Documentation: " + info["doc"], expanded=False):
        C.render_note(pathlib.Path(info["doc"]).name)


# --------------------------------------------------------------------------
if ss["stage"] == OVERVIEW:
    page_overview()
else:
    page_stage(ss["stage"])
