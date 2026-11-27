"""Parameter widgets of the stage pages.

Each stage has a list of ``Ctl`` records (what to show, its range, how to map between the widget
value and the stage parameter).  The widgets use ``on_change`` callbacks that write into the
session pipeline *before* the script reruns, so the pipeline run and the diagram at the top of the
page already see the new value.  ``tests/test_app.py`` checks every ``Ctl.name`` against the
signature of the stage function.
"""
from __future__ import annotations

import copy
import inspect
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import streamlit as st

from isp import cfa as _cfa
from isp import demosaic as _demosaic
from isp import denoise as _denoise
from isp import pipeline as P
from isp import sensor as _sensor
from isp import tonemap as _tonemap

ISOS = [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]
BITS = ["preset", 8, 10, 12, 14, 16]
_IDENT = lambda v: v  # noqa: E731


@dataclass
class Ctl:
    """One parameter widget.

    path       parameter name, or ('sensor_kw', 'hot_pixel_fraction') for a key of a dict parameter
    kind       slider | select | select_slider | checkbox
    to_widget  parameter value -> widget value;  from_widget  widget value -> parameter value
    show       predicate on the stage's params dict (e.g. only for one method)
    """
    path: Any
    label: str
    kind: str
    options: Any = None                 # select options / (lo, hi, step) for sliders
    default: Any = None                 # used when the parameter is missing
    to_widget: Callable = _IDENT
    from_widget: Callable = _IDENT
    show: Callable = lambda p: True
    help: str | None = None
    fmt: str | None = None
    on_change_method: bool = False      # changing this widget switches the method (resets other params)

    @property
    def name(self):
        return self.path if isinstance(self.path, str) else self.path[0]


def _get(params, path, default):
    if isinstance(path, str):
        return params.get(path, default)
    d = params.get(path[0]) or {}
    return d.get(path[1], default)


def _set(params, path, value):
    if isinstance(path, str):
        params[path] = value
    else:
        params.setdefault(path[0], {})
        params[path[0]] = {**(params[path[0]] or {}), path[1]: value}


# per-method default parameters (used when the method select changes)
METHOD_DEFAULTS = {
    "white_balance": {**{m: {"method": m} for m in P.WB_METHODS}, "preset": {"method": "preset", "cct": 3200.0},
                      "white_patch": {"method": "white_patch", "percentile": 99}},
    "demosaic": {m: {"method": m} for m in _demosaic.METHODS},
    "denoise": {"gaussian": {"method": "gaussian", "sigma": 1.5},
                "bilateral": {"method": "bilateral", "sigma_d": 1.5, "sigma_r": 0.05},
                "median": {"method": "median", "size": 3},
                "luma_chroma": {"method": "luma_chroma", "luma_kw": {"sigma_d": 1.5, "sigma_r": 0.05},
                                "chroma_kw": {"sigma_d": 4.0, "sigma_r": 0.10}},
                "nlm": {"method": "nlm", "h": 0.05}},
    "tonemap": {"srgb": {"method": "srgb"}, "linear": {"method": "linear"}, "gamma": {"method": "gamma", "g": 1 / 2.2},
                "hist": {"method": "hist"}, "clahe": {"method": "clahe", "clip": 2.0, "tiles": 8},
                "local": {"method": "local", "strength": 0.6}},
    "sharpen": {"unsharp": {"method": "unsharp", "lam": 0.5, "sigma": 1.0},
                "laplacian": {"method": "laplacian", "lam": 0.5}},
}


def _is(method):
    return lambda p: p.get("method") == method


def _in(*methods):
    return lambda p: p.get("method") in methods


def build_controls(stage: str, size: int = 256, has_camera_wb: bool = False) -> list:
    """The widgets of one stage page.  ``size`` bounds the neutral-patch box of white balance."""
    dp = P.DEFAULT_PARAMS[stage]
    H = P.STAGE_INFO[stage]["params_help"]
    h = lambda k: H.get(k)  # noqa: E731
    if stage == "scene":
        return []
    if stage == "illuminant":
        return [Ctl("cct", "colour temperature (K); 6500 = daylight, 3200 = tungsten", "slider", (2000, 10000, 100),
                    to_widget=lambda v: int(v if v else 6500), from_widget=lambda w: float(w), help=h("cct"))]
    if stage == "cfa":
        return [Ctl("pattern", "Bayer pattern", "select", list(_cfa.PATTERNS), help=h("pattern"))]
    if stage == "exposure":
        return [
            Ctl("sensor", "sensor preset", "select", list(_sensor.PRESETS), help=h("sensor")),
            Ctl("exposure", "exposure time, log2(seconds) (stops)", "slider", (-12.0, 1.0, 0.1),
                to_widget=lambda v: round(math.log2(v), 1), from_widget=lambda w: float(2.0 ** w),
                help=h("exposure"), fmt="%.1f"),
            Ctl("flux_at_white", "scene brightness, log10(photons / pixel / s at white)", "slider", (3.0, 7.0, 0.1),
                to_widget=lambda v: round(math.log10(v), 1), from_widget=lambda w: float(10.0 ** w),
                help=h("flux_at_white"), fmt="%.1f"),
            Ctl(("sensor_kw", "hot_pixel_fraction"), "hot-pixel fraction", "slider", (0.0, 0.005, 0.0001),
                default=5e-4, help="fraction of defective pixels (sensor_kw override)", fmt="%.4f"),
        ]
    if stage == "adc":
        return [
            Ctl("iso", "ISO", "select_slider", ISOS, to_widget=lambda v: min(ISOS, key=lambda o: abs(o - v)),
                help=h("iso")),
            Ctl("bits", "ADC bits", "select_slider", BITS, to_widget=lambda v: "preset" if v is None else int(v),
                from_widget=lambda w: None if w == "preset" else int(w), help=h("bits")),
        ]
    if stage == "black_level":
        return [Ctl("dark_frame", "subtract a master dark frame", "checkbox", help=h("dark_frame")),
                Ctl("digital_gain", "digital gain", "slider", (0.25, 8.0, 0.05), help=h("digital_gain"))]
    if stage == "hot_pixels":
        return [Ctl("threshold", "threshold (linear units)", "slider", (0.02, 1.0, 0.01),
                    to_widget=lambda v: 0.3 if v is None else float(v), help=h("threshold"))]
    if stage == "white_balance":
        methods = [m for m in P.WB_METHODS if m != "camera" or has_camera_wb]
        return [
            Ctl("method", "method", "select", methods, help=h("method"), on_change_method=True),
            Ctl("cct", "illuminant CCT (K)", "slider", (2000, 10000, 100), default=3200, show=_is("preset"),
                to_widget=lambda v: int(v if v else 3200), from_widget=float, help=h("cct")),
            Ctl("percentile", "white-patch percentile", "slider", (90.0, 100.0, 0.5), default=99,
                show=_is("white_patch"), to_widget=float, help=h("percentile")),
        ] + [Ctl(("box", i), lab, "slider", (0, size, 2), default=d, show=_is("patch"),
                 help="neutral patch (full-resolution pixels); drag it over a grey object")
             for i, (lab, d) in enumerate([("patch row start", size // 2 - 16), ("patch row end", size // 2 + 16),
                                           ("patch col start", size // 2 - 16), ("patch col end", size // 2 + 16)])]
    if stage == "demosaic":
        return [Ctl("method", "method", "select", list(_demosaic.METHODS), help=h("method"), on_change_method=True)]
    if stage == "denoise":
        return [
            Ctl("method", "method", "select", list(_denoise.METHODS), help=h("method"), on_change_method=True),
            Ctl("sigma", "Gaussian sigma (px)", "slider", (0.3, 5.0, 0.1), default=1.5, show=_is("gaussian")),
            Ctl("sigma_d", "spatial sigma_d (px)", "slider", (0.5, 5.0, 0.1), default=1.5, show=_is("bilateral"),
                help=h("sigma_d")),
            Ctl("sigma_r", "range sigma_r (linear units)", "slider", (0.005, 0.5, 0.005), default=0.05,
                show=_is("bilateral"), help=h("sigma_r"), fmt="%.3f"),
            Ctl("size", "median window", "select_slider", [3, 5, 7], default=3, show=_is("median")),
            Ctl(("luma_kw", "sigma_r"), "luma sigma_r", "slider", (0.005, 0.5, 0.005), default=0.05,
                show=_is("luma_chroma"), fmt="%.3f"),
            Ctl(("chroma_kw", "sigma_d"), "chroma sigma_d (px)", "slider", (0.5, 8.0, 0.5), default=4.0,
                show=_is("luma_chroma")),
            Ctl("h", "NLM filtering strength h", "slider", (0.01, 0.3, 0.01), default=0.05, show=_is("nlm")),
        ]
    if stage == "tonemap":
        return [
            Ctl("method", "method", "select", list(_tonemap.METHODS), help=h("method"), on_change_method=True),
            Ctl("g", "gamma (display exponent 1/g)", "slider", (1.0, 3.0, 0.05), default=1 / 2.2, show=_is("gamma"),
                to_widget=lambda v: round(1.0 / v, 2), from_widget=lambda w: 1.0 / w),
            Ctl("clip", "CLAHE clip limit", "slider", (1.0, 8.0, 0.5), default=2.0, show=_is("clahe")),
            Ctl("tiles", "CLAHE tiles", "select_slider", [2, 4, 8, 16], default=8, show=_is("clahe")),
            Ctl("strength", "local (dodge and burn) strength", "slider", (0.0, 1.0, 0.05), default=0.6, show=_is("local")),
        ]
    if stage == "sharpen":
        return [
            Ctl("method", "method", "select", ["unsharp", "laplacian"], help=h("method"), on_change_method=True),
            Ctl("lam", "strength lambda", "slider", (0.0, 3.0, 0.05), default=0.5, help=h("lam")),
            Ctl("sigma", "Gaussian sigma (px)", "slider", (0.5, 3.0, 0.1), default=1.0, show=_is("unsharp"),
                help=h("sigma")),
        ]
    if stage == "jpeg":
        return [Ctl("quality", "quality", "slider", (1, 100, 1), help=h("quality")),
                Ctl("subsampling", "chroma subsampling", "select", ["4:4:4", "4:2:2", "4:2:0"], help=h("subsampling"))]
    raise KeyError(stage)


# --------------------------------------------------------------------------
# widget plumbing
# --------------------------------------------------------------------------
def widget_key(stage: str, ctl: Ctl) -> str:
    p = ctl.path if isinstance(ctl.path, str) else "_".join(str(x) for x in ctl.path)
    return f"w|{stage}|{p}"


def _set_param(stage: str, ctl: Ctl):
    """on_change callback: widget value -> pipeline parameter (runs before the script reruns)."""
    ss = st.session_state
    st_ = ss["pipeline"][stage]
    w = ss[widget_key(stage, ctl)]
    if ctl.on_change_method:
        # a new method starts from that method's defaults; drop the other widgets' state
        st_.params = copy.deepcopy(METHOD_DEFAULTS[stage][w])
        for k in [k for k in ss if isinstance(k, str) and k.startswith(f"w|{stage}|") and k != widget_key(stage, ctl)]:
            del ss[k]
        if stage == "white_balance" and w == "patch":
            sz = ss.get("size", 256)
            st_.params["box"] = (sz // 2 - 16, sz // 2 + 16, sz // 2 - 16, sz // 2 + 16)
        return
    v = ctl.from_widget(w)
    if isinstance(ctl.path, tuple) and ctl.path[0] == "box":
        box = list(st_.params.get("box") or (0, 0, 0, 0))
        box[ctl.path[1]] = int(v)
        st_.params["box"] = tuple(box)
    else:
        _set(st_.params, ctl.path, v)


def render_controls(stage: str, size: int = 256, has_camera_wb: bool = False) -> int:
    """Draw the parameter widgets of ``stage``.  Returns the number of widgets shown."""
    pl = st.session_state["pipeline"]
    params = pl[stage].params
    n = 0
    for ctl in build_controls(stage, size, has_camera_wb):
        if not ctl.show(params):
            continue
        key = widget_key(stage, ctl)
        if isinstance(ctl.path, tuple) and ctl.path[0] == "box":
            box = params.get("box") or (0, 0, 0, 0)
            cur = box[ctl.path[1]] if params.get("box") else ctl.default
        else:
            cur = _get(params, ctl.path, ctl.default if ctl.default is not None else dp_default(stage, ctl))
        val = ctl.to_widget(cur)
        kw = dict(key=key, help=ctl.help, on_change=_set_param, args=(stage, ctl))
        if ctl.kind == "slider":
            lo, hi, step = ctl.options
            if isinstance(step, float) or isinstance(lo, float):
                lo, hi, step, val = float(lo), float(hi), float(step), float(min(max(val, lo), hi))
            else:
                val = int(min(max(val, lo), hi))
            st.slider(ctl.label, lo, hi, val, step, format=ctl.fmt, **kw)
        elif ctl.kind == "select":
            opts = list(ctl.options)
            st.selectbox(ctl.label, opts, index=opts.index(val) if val in opts else 0, **kw)
        elif ctl.kind == "select_slider":
            opts = list(ctl.options)
            st.select_slider(ctl.label, opts, value=val if val in opts else opts[0], **kw)
        elif ctl.kind == "checkbox":
            st.checkbox(ctl.label, value=bool(val), **kw)
        n += 1
    return n


def dp_default(stage: str, ctl: Ctl):
    return _get(P.DEFAULT_PARAMS[stage], ctl.path, None)


def check_against_signatures() -> list:
    """Problems (control names that the stage function does not take); used by the tests."""
    bad = []
    for stage in P.STAGE_ORDER:
        fn = P.STAGE_FUNCS[stage]
        sig = inspect.signature(fn).parameters
        var_kw = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.values())
        for ctl in build_controls(stage):
            if ctl.name not in sig and not var_kw:
                bad.append((stage, ctl.name))
    return bad
