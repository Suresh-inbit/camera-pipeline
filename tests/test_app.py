"""Smoke and behaviour tests of the Streamlit explorer (streamlit.testing.v1.AppTest).

The clickable diagram is a custom component, which AppTest cannot click; its Python side is tested
through a stub that returns a click value, its JS side by checking the protocol strings.
"""
import glob
import io
import os
import pathlib
import sys

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

ROOT = pathlib.Path(__file__).resolve().parents[1]
APP = str(ROOT / "app" / "streamlit_app.py")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "app"))

from isp import pipeline as P  # noqa: E402

STAGES = list(P.STAGE_ORDER)


def _app(size=128, stage=None):
    at = AppTest.from_file(APP, default_timeout=120)
    at.session_state["size"] = size
    if stage:
        at.session_state["stage"] = stage
    return at


def _no_problem(at, what=""):
    assert not at.exception, (what, [e.value for e in at.exception])
    assert not at.error, (what, [e.value for e in at.error])


def _rec_names(at):
    return at.session_state["result"].names


# --------------------------------------------------------------------------
def test_default_load_shows_overview():
    at = _app().run()
    _no_problem(at)
    assert at.session_state["stage"] == "overview"
    assert any(h.value == "Overview" for h in at.header)
    assert _rec_names(at) == STAGES
    assert at.session_state["run_error"] is None


@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_page_renders_with_session_state(stage):
    at = _app(stage=stage).run()
    _no_problem(at, stage)
    assert at.header[0].value == P.STAGE_INFO[stage]["label"]
    assert any(e.label.startswith("Experiment") for e in at.expander)


@pytest.mark.parametrize("stage", STAGES)
def test_every_stage_page_renders_with_query_param(stage):
    at = _app()
    at.query_params["stage"] = stage
    at.run()
    _no_problem(at, stage)
    assert at.session_state["stage"] == stage


def test_unknown_query_param_falls_back_to_overview():
    at = _app()
    at.query_params["stage"] = "nope"
    at.run()
    _no_problem(at)
    assert at.session_state["stage"] == "overview"


def test_prev_next_walk_the_pipeline_in_order():
    at = _app(stage="scene").run()
    seen = [at.session_state["stage"]]
    for _ in range(len(STAGES) - 1):
        at = at.button(key="next").click().run()
        _no_problem(at)
        seen.append(at.session_state["stage"])
    assert seen == STAGES
    assert at.button(key="next").disabled
    for want in reversed(STAGES[:-1]):
        at = at.button(key="prev").click().run()
        _no_problem(at)
        assert at.session_state["stage"] == want
    assert at.query_params["stage"] == "scene" or at.query_params["stage"] == ["scene"]


def test_sidebar_stage_selector_navigates():
    at = _app().run()
    at = at.sidebar.selectbox(key="stage").select("demosaic").run()
    _no_problem(at)
    assert at.header[0].value == "Demosaic"


# --------------------------------------------------------------------------
# parameters and incremental re-runs
# --------------------------------------------------------------------------
def _final_psnr(at):
    return at.session_state["result"].final.metrics["psnr"]


def test_early_parameter_changes_the_final_metric():
    at = _app(stage="adc").run()
    p0 = _final_psnr(at)
    at = at.select_slider(key="w|adc|iso").set_value(800).run()
    _no_problem(at)
    assert at.session_state["pipeline"]["adc"].params["iso"] == 800
    assert _final_psnr(at) != p0
    assert "adc" in at.session_state["run_info"]["recomputed"]
    # StageMetrics' cache key contains every parameter before demosaic (its oracle run), so an early
    # change recomputes everything; this is a property of isp.pipeline, not of the app
    assert at.session_state["run_info"]["reused"] == []
    p1 = _final_psnr(at)
    at = _app(stage="illuminant").run()
    q0 = _final_psnr(at)
    at = at.slider(key="w|illuminant|cct").set_value(6500).run()
    _no_problem(at)
    assert at.session_state["pipeline"]["illuminant"].params["cct"] == 6500.0
    assert _final_psnr(at) != q0 and p1 != q0


def test_late_parameter_reuses_upstream_records():
    at = _app(stage="sharpen").run()
    at = at.slider(key="w|sharpen|lam").set_value(1.5).run()
    _no_problem(at)
    info = at.session_state["run_info"]
    assert info["reused"] == STAGES[:11]               # scene ... tonemap copied, not recomputed
    assert info["recomputed"] == ["sharpen", "jpeg"]
    res = at.session_state["result"]
    assert [r.reused for r in res] == [True] * 11 + [False, False]
    # the Overview table shows the flags of its own (second) run: nothing changed, so all reused
    at.session_state["stage"] = "overview"
    at = at.run()
    df = [d for d in at.dataframe if "reused" in d.value.columns][0].value
    assert df["reused"].tolist() == [True] * 13


def test_mid_pipeline_parameter_reuses_the_sensor_stages():
    at = _app(stage="demosaic").run()
    at = at.selectbox(key="w|demosaic|method").select("bilinear").run()
    _no_problem(at)
    assert at.session_state["run_info"]["reused"] == STAGES[:8]


def test_method_selectors_run():
    cases = [("white_balance", "method", ["none", "true", "gray_world", "preset", "patch", "white_patch"]),
             ("demosaic", "method", ["nearest", "bilinear", "bicubic", "malvar", "median"]),
             ("denoise", "method", ["gaussian", "median", "luma_chroma", "nlm", "bilateral"]),
             ("tonemap", "method", ["linear", "gamma", "hist", "clahe", "local", "srgb"]),
             ("sharpen", "method", ["laplacian", "unsharp"]),
             ("jpeg", "subsampling", ["4:4:4", "4:2:2", "4:2:0"]),
             ("cfa", "pattern", ["BGGR", "GRBG", "GBRG", "RGGB"]),
             ("exposure", "sensor", ["phone", "retiga_4000r", "aptina_mt9p031", "dslr"])]
    for stage, name, options in cases:
        at = _app(stage=stage).run()
        for opt in options:
            at = at.selectbox(key=f"w|{stage}|{name}").select(opt).run()
            _no_problem(at, (stage, opt))
            assert at.session_state["run_error"] is None, (stage, opt, at.session_state["run_error"])
            assert at.session_state["pipeline"][stage].params[name] == opt


def test_numeric_widgets_run():
    at = _app(stage="exposure").run()
    at = at.slider(key="w|exposure|exposure").set_value(-4.0).run()
    assert abs(at.session_state["pipeline"]["exposure"].params["exposure"] - 2 ** -4) < 1e-12
    at = at.slider(key="w|exposure|flux_at_white").set_value(6.0).run()
    at = at.slider(key="w|exposure|sensor_kw_hot_pixel_fraction").set_value(0.002).run()
    assert at.session_state["pipeline"]["exposure"].params["sensor_kw"]["hot_pixel_rate"] == 40000.0   # other keys kept
    _no_problem(at)
    at = _app(stage="adc").run()
    at = at.select_slider(key="w|adc|bits").set_value(10).run()
    _no_problem(at)
    assert at.session_state["result"]["adc"].meta["bits"] == 10
    at = _app(stage="jpeg").run()
    at = at.slider(key="w|jpeg|quality").set_value(20).run()
    _no_problem(at)
    assert at.session_state["result"].final.meta["quality"] == 20
    at = _app(stage="black_level").run()
    at = at.checkbox(key="w|black_level|dark_frame").set_value(True).run()
    _no_problem(at)
    assert at.session_state["pipeline"]["black_level"].params["dark_frame"] is True


# --------------------------------------------------------------------------
# overview controls
# --------------------------------------------------------------------------
def test_disable_stage_on_overview():
    at = _app().run()
    at = at.checkbox(key="en|hot_pixels").set_value(False).run()
    _no_problem(at)
    assert at.session_state["run_error"] is None
    assert at.session_state["pipeline"]["hot_pixels"].enabled is False
    assert at.session_state["result"]["hot_pixels"].enabled is False
    df = [d for d in at.dataframe if "enabled" in d.value.columns][0].value
    assert bool(df.loc["hot_pixels", "enabled"]) is False and bool(df.loc["denoise", "enabled"]) is True
    assert len(at.session_state["result"]) == 13


def test_impossible_change_is_rolled_back():
    at = _app().run()
    at = at.checkbox(key="en|black_level").set_value(False).run()
    assert not at.exception
    assert at.session_state["run_error"] is None or "stage" in at.session_state["run_error"]
    assert at.session_state["pipeline"]["black_level"].enabled is True       # restored
    assert len(at.session_state["result"]) == 13
    assert any("rejected" in w.value for w in at.warning)


@pytest.mark.parametrize("preset,order_check", [
    ("denoise_first", lambda n: n.index("denoise") < n.index("demosaic")),
    ("sharpen_first", lambda n: n.index("sharpen") < n.index("denoise")),
    ("wb_late", lambda n: n.index("white_balance") > n.index("demosaic")),
    ("default", lambda n: n == STAGES)])
def test_order_presets_run(preset, order_check):
    at = _app().run()
    at = at.button(key=f"order|{preset}").click().run()
    _no_problem(at, preset)
    assert at.session_state["run_error"] is None, at.session_state["run_error"]
    names = at.session_state["pipeline"].names
    assert order_check(names) and sorted(names) == sorted(STAGES)
    assert at.session_state["result"].names == names
    # the diagram follows the order, the stage pages still work
    at.session_state["stage"] = "white_balance"
    at = at.run()
    _no_problem(at, preset)


def test_reset_to_defaults():
    at = _app().run()
    at = at.button(key="order|wb_late").click().run()
    at = at.checkbox(key="en|jpeg").set_value(False).run()
    at = at.button(key="reset").click().run()
    _no_problem(at)
    assert at.session_state["pipeline"].names == STAGES
    assert all(s.enabled for s in at.session_state["pipeline"].stages)


# --------------------------------------------------------------------------
# scene and uploads
# --------------------------------------------------------------------------
def test_scene_selector_changes_the_input():
    at = _app().run()
    before = at.session_state["result"].input_image.copy()
    key0 = at.session_state["scene_key"]
    at = at.sidebar.selectbox(key="scene_name").select("zone_plate").run()
    _no_problem(at)
    assert at.session_state["scene_key"] != key0
    assert not np.array_equal(before, at.session_state["result"].input_image)
    at = at.sidebar.selectbox(key="size").select(256).run()
    _no_problem(at)
    assert at.session_state["result"].input_image.shape[0] == 256


def _shim(tmp_path, body):
    """A copy of the app with ``body`` run first (to stub the uploader or the diagram component)."""
    src = pathlib.Path(APP).read_text().replace("from __future__ import annotations\n", "", 1)
    head = ("from __future__ import annotations\nimport sys, pathlib as _p, streamlit as _st\n"
            f"__file__ = {APP!r}\nsys.path.insert(0, {str(ROOT / 'app')!r})\n" + body)
    f = tmp_path / "shim_app.py"
    f.write_text(head + src)
    return AppTest.from_file(str(f), default_timeout=120)


def test_image_upload_path(tmp_path):
    from PIL import Image
    buf = io.BytesIO()
    yy, xx = np.mgrid[0:90, 0:120]
    Image.fromarray(np.stack([xx * 2, yy * 2, (xx + yy)], -1).astype(np.uint8)).save(buf, format="PNG")
    png = tmp_path / "tiny.png"
    png.write_bytes(buf.getvalue())
    at = _shim(tmp_path, f'''
class _Up:
    name = "tiny.png"
    def getvalue(self): return _p.Path({str(png)!r}).read_bytes()
_orig = _st.file_uploader
_st.file_uploader = lambda *a, **k: _Up() if k.get("key") == "up_img" else None
''')
    at.session_state["size"] = 128
    at.run()
    _no_problem(at)
    assert at.session_state["scene_key"][0] == "upload"
    assert at.session_state["result"].input_image.shape == (128, 128, 3)
    assert at.session_state["result"].names == STAGES
    # the checkbox returns to the built-in scene
    at = at.sidebar.checkbox(key="use_up").set_value(False).run()
    _no_problem(at)
    assert at.session_state["scene_key"][0] == "astronaut"


def test_raw_entry_runs_from_black_level():
    """What the RAW upload does (rawpy itself is optional and absent here): enter at black_level."""
    import _common as C
    full = P.simulate("astronaut", size=64)
    mosaic, meta = full["adc"].image, dict(full["adc"].meta)
    pl = P.default_pipeline()
    res, secs = C.run_pipeline(mosaic, pl, meta=meta, start="black_level", with_metrics=False)
    assert res.names == STAGES[STAGES.index("black_level"):]
    assert res.final.image.shape == (64, 64, 3)
    payload = C.build_nav_payload(res, pl, "overview", C.thumbnails(res))
    assert len(payload["nodes"]) == 13
    assert [n["available"] for n in payload["nodes"]] == [False] * 5 + [True] * 8


def test_rawpy_hint_when_missing():
    import _common as C
    at = _app().run()
    texts = " ".join(c.value for c in at.sidebar.caption)
    assert C.rawpy_available() or "pip install rawpy" in texts


# --------------------------------------------------------------------------
# the diagram component
# --------------------------------------------------------------------------
def test_click_nonce_logic_through_the_app(tmp_path):
    at = _shim(tmp_path, '''
import nav as _nav
_nav.pipeline_nav = lambda payload, key="pipeline_nav": _st.session_state.get("_fake_click")
''')
    at.session_state["size"] = 128
    at.run()
    _no_problem(at)
    assert at.session_state["stage"] == "overview"
    at.session_state["_fake_click"] = {"stage": "adc", "nonce": "100-1"}
    at.run()
    _no_problem(at)
    assert at.session_state["stage"] == "adc"
    assert at.header[0].value == "ADC"
    at = at.button(key="next").click().run()                    # same (old) click value: ignored
    assert at.session_state["stage"] == "black_level"
    at.session_state["_fake_click"] = {"stage": "adc", "nonce": "100-2"}      # same stage again, new nonce
    at.run()
    assert at.session_state["stage"] == "adc"
    at.session_state["_fake_click"] = {"stage": "bogus", "nonce": "100-3"}
    at.run()
    _no_problem(at)
    assert at.session_state["stage"] == "adc"


def test_new_click_helper():
    from nav import new_click
    assert new_click(None, None) is None
    assert new_click({"stage": "adc", "nonce": "1"}, None) == "adc"
    assert new_click({"stage": "adc", "nonce": "1"}, "1") is None
    assert new_click({"stage": "adc", "nonce": "2"}, "1") == "adc"
    assert new_click({"stage": 3, "nonce": "2"}, "1") is None


def test_component_file_and_protocol_strings():
    html = (ROOT / "app" / "pipeline_nav" / "index.html").read_text()
    for token in ("streamlit:componentReady", "streamlit:setComponentValue", "streamlit:setFrameHeight",
                  "streamlit:render", "isStreamlitMessage", "apiVersion: 1", "dataType", "window.parent.postMessage"):
        assert token in html, token
    import streamlit
    js = glob.glob(os.path.join(os.path.dirname(streamlit.__file__), "static", "static", "js", "index.*.js"))
    bundle = "".join(pathlib.Path(p).read_text(errors="ignore") for p in js)
    for token in ("streamlit:componentReady", "streamlit:setComponentValue", "streamlit:setFrameHeight",
                  "streamlit:render", "isStreamlitMessage", "CUSTOM_COMPONENT_API_VERSION=1"):
        assert token in bundle, token                               # the installed host speaks the same protocol


def test_nav_payload_has_a_node_with_thumbnail_per_stage():
    import _common as C
    res = P.simulate("astronaut", size=64)
    pl = P.default_pipeline(enabled={"denoise": False})
    cache = {}
    thumbs = C.thumbnails(res, cache)
    payload = C.build_nav_payload(res, pl, "adc", thumbs)
    assert payload["current"] == "adc"
    assert [n["id"] for n in payload["nodes"]] == STAGES
    for n in payload["nodes"]:
        assert n["thumb"].startswith("data:image/"), n["id"]
        assert n["group"] in P.GROUP_INFO and n["short"] and n["domain"]
    assert [n["enabled"] for n in payload["nodes"]].count(False) == 1
    assert set(payload["groups"]) == set(P.GROUP_INFO)
    # only changed records are re-rendered: same result again -> cache objects are reused
    first = {k: v[1] for k, v in cache.items()}
    C.thumbnails(res, cache)
    assert all(cache[k][1] is first[k] for k in first)


def test_nav_component_directory():
    import nav
    assert nav.COMPONENT_DIR.joinpath("index.html").exists()


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def test_controls_match_stage_function_signatures():
    import stage_params
    assert stage_params.check_against_signatures() == []


def test_order_helpers():
    import _common as C
    pl = P.default_pipeline()
    assert C.order_name(pl) == "default"
    for key in C.ORDER_PRESETS:
        assert C.order_name(C.apply_order(P.default_pipeline(), key)) == key
    res, _ = C.run_pipeline(*_tiny())
    assert len(res) == 13


def _tiny():
    from isp import scenes
    return scenes.load("astronaut", 64), P.default_pipeline()
