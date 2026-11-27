"""Shared helpers for the camera-pipeline explorer: state, running, thumbnails, nav payload, docs."""
from __future__ import annotations

import copy
import hashlib
import io
import pathlib
import re
import sys
import time

import matplotlib
import numpy as np
import streamlit as st

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from isp import display, io as isp_io, pipeline as P, scenes, utils  # noqa: E402

DOCS = ROOT / "docs"
OVERVIEW = "overview"
SIZES = (128, 256, 384)

# Orders of docs/08 (section 5.2).  Each is applied to the canonical order, so presets do not stack.
ORDER_PRESETS = {
    "default": ("Default order", None),
    "denoise_first": ("Denoise before demosaic", ("denoise", "before", "demosaic")),
    "sharpen_first": ("Sharpen before denoise", ("sharpen", "before", "denoise")),
    "wb_late": ("White balance after demosaic", ("white_balance", "after", "demosaic")),
}


# --------------------------------------------------------------------------
# scenes (cached)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=24)
def load_scene(name: str, size: int) -> np.ndarray:
    return scenes.load(name, int(size))


@st.cache_data(show_spinner=False, max_entries=4)
def load_uploaded_image(data: bytes, digest: str, size: int) -> np.ndarray:
    """Decode an uploaded picture (cached on its SHA-1) as a linear scene, centre-cropped to size."""
    return isp_io.load_image(data, size=int(size))


@st.cache_data(show_spinner="Reading RAW file ...", max_entries=2)
def load_raw_cached(data: bytes, digest: str):
    """(mosaic DN, meta) of an uploaded RAW file; needs the optional rawpy package."""
    return isp_io.load_raw(data)


def rawpy_available() -> bool:
    import importlib.util
    return importlib.util.find_spec("rawpy") is not None


# --------------------------------------------------------------------------
# pipeline state
# --------------------------------------------------------------------------
def canonical_order(pl: P.Pipeline) -> P.Pipeline:
    pl.stages.sort(key=lambda s: P.STAGE_ORDER.index(s.name))
    return pl


def apply_order(pl: P.Pipeline, preset: str) -> P.Pipeline:
    """Re-order ``pl`` in place to one of ``ORDER_PRESETS`` (params and enabled flags are kept)."""
    canonical_order(pl)
    spec = ORDER_PRESETS[preset][1]
    if spec is not None:
        name, where, other = spec
        pl.move(name, **{where: other})
    return pl


def order_name(pl: P.Pipeline) -> str:
    for key in ORDER_PRESETS:
        if apply_order(pl.copy(), key).names == pl.names:
            return key
    return "custom"


def clear_widget_keys(*prefixes: str):
    """Drop the widget state of parameter widgets (they are re-created from the pipeline)."""
    for k in [k for k in st.session_state if isinstance(k, str) and k.startswith(prefixes)]:
        del st.session_state[k]


def init_state():
    ss = st.session_state
    if "pipeline" not in ss:
        ss["pipeline"] = P.default_pipeline()
        ss["pipeline_good"] = ss["pipeline"].copy()
        ss["result"] = None
        ss["scene_key"] = None
        ss["run_info"] = {}
        ss["run_error"] = None
        ss["nav_seen"] = None
        ss["thumb_cache"] = {}
        ss["run_count"] = 0


def reset_pipeline():
    """Callback: back to the default camera (params, enabled flags, order)."""
    st.session_state["pipeline"] = P.default_pipeline()
    st.session_state["pipeline_good"] = st.session_state["pipeline"].copy()
    st.session_state["run_error"] = None
    clear_widget_keys("w|", "en|")


def set_order(preset: str):
    """Callback: apply an order preset."""
    apply_order(st.session_state["pipeline"], preset)


def set_enabled(stage: str):
    """Callback of the enable checkboxes on the Overview."""
    st.session_state["pipeline"].enable(stage, bool(st.session_state[f"en|{stage}"]))


def prefix_key(scene_key, pl: P.Pipeline, upto: str, include=True) -> str:
    """Stable string describing the scene and the stages up to (and including) ``upto``; keys
    st.cache_data experiments so they recompute only when something upstream changed."""
    parts = [repr(scene_key)]
    for s in pl.stages:
        parts.append(repr((s.name, s.enabled, sorted(s.params.items(), key=str))))
        if s.name == upto:
            break
    return hashlib.sha1("|".join(parts).encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------
def run_pipeline(img, pl: P.Pipeline, previous=None, meta=None, start=None, with_metrics=True):
    """Run ``pl`` on ``img`` (incrementally when ``previous`` is given).  Returns (result, seconds)."""
    t0 = time.perf_counter()
    met = P.StageMetrics(img, pl) if (with_metrics and start is None) else None
    res = pl.run(img, meta if meta is not None else {"domain": "scene"}, metrics=met,
                 start=start, previous=previous)
    return res, time.perf_counter() - t0


def update_result(img, meta=None, start=None, with_metrics=True, scene_key=None):
    """Run the session pipeline on ``img``.  On failure (e.g. an order that the domain rules
    reject) the last working pipeline is restored and a message is kept in ``run_error``."""
    ss = st.session_state
    ss["run_error"] = None
    if ss.get("scene_key") != scene_key:
        ss["result"] = None                        # new scene / size: drop the result cache
        ss["thumb_cache"] = {}
        ss["scene_key"] = scene_key
    try:
        res, secs = run_pipeline(img, ss["pipeline"], ss["result"], meta, start, with_metrics)
    except Exception as e:  # noqa: BLE001
        ss["run_error"] = f"{type(e).__name__}: {e}"
        ss["pipeline"] = ss["pipeline_good"].copy()
        clear_widget_keys("w|", "en|")
        res, secs = run_pipeline(img, ss["pipeline"], ss["result"], meta, start, with_metrics)
    else:
        ss["pipeline_good"] = ss["pipeline"].copy()
    ss["result"] = res
    ss["run_count"] = ss.get("run_count", 0) + 1
    ss["run_info"] = dict(seconds=secs, reused=[r.name for r in res if r.reused],
                          recomputed=[r.name for r in res if not r.reused],
                          n_reused=sum(r.reused for r in res), n_total=len(res))
    return res


# --------------------------------------------------------------------------
# thumbnails and the navigation payload
# --------------------------------------------------------------------------
def thumbnails(result, cache: dict | None = None, size: int = 96) -> dict:
    """``{stage: data URI}``; only records whose output array is new are rendered (the cache is
    keyed on the identity of ``record.image``, which reused records share with the previous run)."""
    cache = {} if cache is None else cache
    todo = [r for r in result if r.name not in cache or cache[r.name][0] is not r.image]
    if todo:
        thumbs = display.stage_thumbnails(todo, size=size)
        for r in todo:
            fmt = "png" if r.image.ndim == 2 else "jpg"          # keep the Bayer pixels crisp
            cache[r.name] = (r.image, display.to_data_uri(thumbs[r.name], fmt))
    for k in [k for k in cache if k not in result.names]:
        del cache[k]
    return {k: v[1] for k, v in cache.items()}


def domain_text(rec) -> str:
    """'dn - mosaic - 14 bit' style one-liner for a stage box."""
    m = rec.meta
    d = m.get("domain", "?")
    shape = "mosaic" if rec.image.ndim == 2 else "RGB"
    if d == "dn":
        depth = f"{int(m.get('bits', 0))} bit"
    elif d == "bytes":
        depth = f"8 bit, {m.get('nbytes', 0) / 1024:.0f} kB"
    elif d == "electrons":
        depth = "e-"
    else:
        depth = "float"
    return f"{d} / {shape} / {depth}"


def build_nav_payload(result, pipeline: P.Pipeline, current: str, thumbs: dict) -> dict:
    """Arguments of the ``pipeline_nav`` component: one node per stage of ``pipeline`` (in its
    current order), band colours and the current page.  Pure Python, no Streamlit."""
    nodes = []
    for i, s in enumerate(pipeline.stages):
        info = P.STAGE_INFO[s.name]
        have = s.name in result.names
        rec = result[s.name] if have else None
        nodes.append(dict(
            id=s.name, label=info["label"], group=s.group, index=i, enabled=bool(s.enabled),
            available=have, thumb=thumbs.get(s.name), short=info["short"],
            domain=domain_text(rec) if rec is not None else "not in RAW input"))
    groups = {k: dict(v) for k, v in P.GROUP_INFO.items()}
    return dict(nodes=nodes, groups=groups, current=current)


def valid_pages(pipeline: P.Pipeline) -> list:
    return [OVERVIEW] + pipeline.names


# --------------------------------------------------------------------------
# display helpers
# --------------------------------------------------------------------------
def fig_png(fig, dpi: int = 90) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def show_png(data: bytes, caption=None):
    st.image(data, caption=caption, width="stretch")


def scene_image(img):
    """Ground-truth scene as uint8 for display."""
    return display.render(img, {"domain": "scene"})


def draw_box(img_u8, box, colour=(255, 40, 40)):
    """Rectangle outline on a uint8 RGB copy (box = r0, r1, c0, c1)."""
    a = np.array(img_u8)
    r0, r1, c0, c1 = (int(v) for v in box)
    t = max(1, min(a.shape[:2]) // 128)
    for sl in (a[r0:r0 + t, c0:c1], a[max(r1 - t, 0):r1, c0:c1], a[r0:r1, c0:c0 + t], a[r0:r1, max(c1 - t, 0):c1]):
        sl[...] = colour
    return a


# --------------------------------------------------------------------------
# documentation rendering (docs/*.md with figures)
# --------------------------------------------------------------------------
_IMG_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")


def _resolve(href: str):
    for p in (DOCS / href, ROOT / href, DOCS / "figures" / pathlib.Path(href).name):
        if p.exists() and p.is_file():
            return p
    return None


def render_note(filename: str):
    """Render docs/<filename> with st.markdown; figures referenced as ![..](figures/x.png) via st.image."""
    path = DOCS / pathlib.Path(filename).name
    if not path.exists():
        st.warning(f"{path} not found")
        return
    text = path.read_text(encoding="utf-8")
    pos = 0
    for m in _IMG_RE.finditer(text):
        chunk = text[pos:m.start()]
        if chunk.strip():
            st.markdown(chunk)
        p = _resolve(m.group(2))
        if p is not None:
            st.image(str(p), caption=m.group(1), width="stretch")
        pos = m.end()
    if text[pos:].strip():
        st.markdown(text[pos:])


def safe(name: str, fn, *args):
    """Run a page section; one failing experiment must not take the page down."""
    try:
        fn(*args)
    except Exception as e:  # noqa: BLE001
        st.error(f"'{name}' failed: {type(e).__name__}: {e}")


def input_record(res, name: str):
    """The record (or a stand-in with ``image`` / ``meta``) that feeds stage ``name``: the previous
    stage's output, or the pipeline input for the first stage."""
    from types import SimpleNamespace
    i = res.names.index(name)
    if i == 0:
        return SimpleNamespace(image=res.input_image, meta=res.input_meta, name="input")
    return res.records[i - 1]
