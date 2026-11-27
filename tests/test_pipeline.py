"""Tests for the default camera (isp.pipeline), isp.display and isp.io.  Small images, seeded."""
import base64
import io
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from isp import display, io as isp_io, pipeline as P, scenes, utils
from isp.pipeline import Pipeline, Stage

ROOT = Path(__file__).resolve().parents[1]
IDS = list(P.STAGE_ORDER)
SIZE = 64
EXPECTED_DOMAINS = ["scene", "scene", "scene", "electrons", "dn", "linear", "linear", "linear",
                    "linear", "linear", "display", "display", "bytes"]


@pytest.fixture(scope="module")
def default_result():
    return P.simulate("astronaut", SIZE)


# ------------------------------------------------------------------ the default camera
def test_default_run_end_to_end(default_result):
    r = default_result
    assert r.names == IDS and len(r) == 13
    assert [x.meta["domain"] for x in r] == EXPECTED_DOMAINS
    mosaic_stages = ["cfa", "exposure", "adc", "black_level", "hot_pixels", "white_balance"]
    for rec in r:
        want = (SIZE, SIZE) if rec.name in mosaic_stages else (SIZE, SIZE, 3)
        assert rec.image.shape == want, rec.name
        assert ("cfa" in rec.meta) == (rec.name in mosaic_stages), rec.name       # cfa only on a mosaic
    f = r.final
    assert f.image.dtype == np.float32 and 0.0 <= f.image.min() and f.image.max() <= 1.0
    assert f.meta["nbytes"] > 0 and f.meta["bpp"] > 0 and f.meta["quality"] == 85
    assert np.array_equal(utils.to_uint8(f.image).shape, (SIZE, SIZE, 3))
    assert r["adc"].meta["bits"] == 14 and r["adc"].image.max() <= 2 ** 14 - 1
    assert r["white_balance"].meta["wb_angular_error"] < 6.0           # white patch finds the lamp (0.8 deg at 256 px)


def test_stage_groups_follow_slide_3():
    pl = P.default_pipeline()
    assert [s.group for s in pl.stages] == ["scene", "scene", "sensor", "sensor", "adc"] + \
        ["processing"] * 7 + ["storage"]


def test_deterministic_for_equal_seeds(default_result):
    again = P.simulate("astronaut", SIZE)
    assert np.array_equal(again.final.image, default_result.final.image)
    other = P.simulate("astronaut", SIZE, exposure={"seed": 5})
    assert not np.array_equal(other["exposure"].image, default_result["exposure"].image)


def test_overrides_and_enabled():
    r = P.simulate("astronaut", SIZE, adc={"iso": 800, "bits": 10}, jpeg={"quality": 30},
                   enabled={"denoise": False}, demosaic={"method": "bilinear"})
    assert r["adc"].meta["iso"] == 800 and r["adc"].meta["bits"] == 10
    assert r["adc"].image.max() <= 2 ** 10 - 1
    assert r.final.meta["quality"] == 30
    assert r["denoise"].enabled is False and np.array_equal(r["denoise"].image, r["demosaic"].image)
    assert P.default_pipeline(adc={"iso": 6400})["adc"].params["iso"] == 6400
    # changing the method drops the other defaults of that stage
    assert P.default_pipeline(denoise={"method": "gaussian", "sigma": 1.0})["denoise"].params == \
        {"method": "gaussian", "sigma": 1.0}
    with pytest.raises(ValueError):
        P.default_pipeline(nonsense={"a": 1})


def test_disabling_every_processing_stage_still_runs():
    off = {s: False for s in IDS if P._G[s] == "processing"}
    r = P.simulate("astronaut", SIZE, enabled=off)
    assert r["adc"].meta["domain"] == "dn" and r.final.meta["domain"] == "bytes"
    assert r.final.image.shape == (SIZE, SIZE, 3) and r.final.meta["nbytes"] > 0
    # disabling everything except the scene still runs too
    r2 = P.simulate("astronaut", SIZE, enabled={s: False for s in IDS if s != "scene"})
    assert r2.final.image.shape == (SIZE, SIZE, 3)


def test_impossible_orders_raise_clearly():
    with pytest.raises(ValueError, match="adc"):
        P.default_pipeline().move("adc", before="exposure").run(scenes.load("astronaut", 32))
    with pytest.raises(ValueError, match="tonemap"):
        P.default_pipeline().move("tonemap", before="exposure").run(scenes.load("astronaut", 32))


def test_reordered_stages_work():
    """WB after demosaic uses RGB gains; denoise before demosaic filters CFA planes."""
    scene = scenes.load("astronaut", SIZE)
    base = P.simulate(scene)
    wb_late = P.default_pipeline(white_balance={"method": "true"}).move("white_balance", after="demosaic")
    base = P.simulate(scene, white_balance={"method": "true"})
    r = wb_late.run(scene, metrics=P.StageMetrics(scene, wb_late))
    assert r["white_balance"].image.ndim == 3 and "cfa" not in r["white_balance"].meta
    g1, g2 = np.array(base["white_balance"].meta["wb_gains"]), np.array(r["white_balance"].meta["wb_gains"])
    assert np.allclose(g1, g2)                                        # same light, same gains
    assert abs(r.final.metrics["psnr"] - base.final.metrics["psnr"]) < 3.0
    early = P.default_pipeline().move("denoise", before="demosaic")
    r2 = early.run(scene)
    assert r2["denoise"].image.shape == (SIZE, SIZE) and r2.final.image.shape == (SIZE, SIZE, 3)
    # mosaic denoise never mixes colours: a pure-red mosaic stays pure red
    from isp import cfa
    red = cfa.mosaic(np.dstack([np.full((16, 16), 0.5), np.zeros((16, 16)), np.zeros((16, 16))]).astype(np.float32))
    out, _ = P.denoise_stage(red, {"domain": "linear", "cfa": "RGGB"}, method="gaussian", sigma=1.0)
    assert np.array_equal(out == 0, red == 0)


def _wb_order_psnr(scene, cct, method, wb_after):
    p = P.default_pipeline(illuminant={"cct": cct}, white_balance={"method": "true"},
                           demosaic={"method": method}, exposure={"oracle": True}, adc={"oracle": True})
    for s in ("hot_pixels", "denoise", "sharpen", "jpeg"):
        p.enable(s, False)
    if wb_after:
        p.move("white_balance", after="demosaic")
    return P.simulate(scene, pipeline=p).final.metrics["psnr"]


@pytest.mark.parametrize("method", ["bilinear", "malvar", "median"])
def test_white_balance_before_demosaic_matters_only_under_a_cast(method):
    """Noise-free sensor, oracle gains: moving WB after demosaicing is harmless without a
    colour cast and for per-channel (bilinear) interpolation, but costs the colour-difference
    demosaicers (Malvar, Freeman median) under tungsten light (docs/08, section 5.2)."""
    scene = scenes.load("astronaut", 128)
    for cct in (6504, 3200):
        loss = _wb_order_psnr(scene, cct, method, False) - _wb_order_psnr(scene, cct, method, True)
        if method == "bilinear" or cct == 6504:
            assert abs(loss) < 0.01
        else:
            assert loss > 2.0


def test_default_order_balances_before_demosaicing():
    names = P.default_pipeline().names
    assert names.index("white_balance") < names.index("demosaic")


def test_dark_frame_and_hot_pixels_remove_defects():
    scene = scenes.load("chelsea", SIZE)           # smooth image: defects stand out
    kw = dict(exposure={"sensor_kw": {"hot_pixel_fraction": 5e-3, "hot_pixel_rate": 40000.0}},
              enabled={"denoise": False, "hot_pixels": False})
    plain = P.simulate(scene, **kw)
    dark = P.simulate(scene, black_level={"dark_frame": True}, **kw)
    assert dark["black_level"].metrics["psnr_noise"] > plain["black_level"].metrics["psnr_noise"] + 2


# ------------------------------------------------------------------ noise-free and denoising
def test_noise_free_sensor_with_true_wb_is_nearly_perfect():
    """Huge photon count, no read noise, 14-bit: only demosaicing and the tone curve remain.
    PSNR at the tone-mapped stage must exceed 28 dB (measured 29.5 dB at 64 px: demosaicing
    error plus the tone curve's amplification of dark errors)."""
    quiet = {"sensor_kw": {"full_well": 1e7, "read_noise": 0.0, "dark_current": 0.0, "prnu": 0.0,
                           "dsnu": 0.0, "hot_pixel_fraction": 0.0}, "flux_at_white": 1e9}
    r = P.simulate("astronaut", SIZE, exposure=quiet, adc={"iso": 100}, white_balance={"method": "true"},
                   enabled={"denoise": False, "sharpen": False, "jpeg": False, "hot_pixels": False})
    assert r["tonemap"].metrics["psnr"] > 28.0
    assert r["white_balance"].metrics["wb_angular_error"] < 1e-3
    assert r["white_balance"].metrics["psnr_noise"] > 60.0     # nothing left but quantisation
    assert r["adc"].metrics["psnr_noise"] > 60.0


def test_denoise_improves_the_noisy_default():
    kw = dict(enabled={"jpeg": False, "sharpen": False})
    on = P.simulate("astronaut", 96, **kw)
    off = P.simulate("astronaut", 96, enabled={"denoise": False, "jpeg": False, "sharpen": False})
    assert on["tonemap"].metrics["psnr"] > off["tonemap"].metrics["psnr"] + 0.5
    assert on["tonemap"].metrics["ssim"] > off["tonemap"].metrics["ssim"]


def test_quality_trace_tells_the_story(default_result):
    m = {r.name: r.metrics for r in default_result}
    assert m["scene"]["psnr"] == float("inf")
    assert m["illuminant"]["psnr"] < 15 and m["illuminant"]["cast_deg"] > 15       # the cast hurts
    assert m["white_balance"]["psnr"] > m["hot_pixels"]["psnr"] + 8                 # WB fixes colour
    assert m["white_balance"]["cast_deg"] < 8                 # (1.3 deg at 256 px)
    assert m["demosaic"]["psnr"] < m["white_balance"]["psnr"] + 1.0                 # demosaic costs a little
    assert m["denoise"]["psnr"] > m["demosaic"]["psnr"]                             # denoise gains
    assert m["denoise"]["sigma"] < 0.8 * m["demosaic"]["sigma"]
    assert m["exposure"]["psnr_noise"] < 45 and "psnr_noise" in m["hot_pixels"]      # noise shows up
    assert "psnr_noise" not in m["demosaic"] and "psnr_noise" not in m["scene"]
    assert all(r.seconds >= 0 for r in default_result)


def test_metric_references_are_brightness_independent():
    """A different ISO / exposure product changes brightness, not the PSNR story."""
    kw = dict(enabled={"jpeg": False})
    a = P.simulate("astronaut", SIZE, adc={"iso": 3200}, exposure={"exposure": 0.0125}, **kw)
    b = P.simulate("astronaut", SIZE, adc={"iso": 1600}, exposure={"exposure": 0.025}, **kw)
    assert abs(a["demosaic"].image.mean() - b["demosaic"].image.mean()) < 0.01       # same brightness
    assert abs(a["demosaic"].metrics["psnr"] - b["demosaic"].metrics["psnr"]) < 1.5


# ------------------------------------------------------------------ incremental re-runs
def _counting(pl):
    calls = []
    stages = []
    for s in pl.stages:
        def fn(img, meta, _f=s.fn, _n=s.name, **kw):
            calls.append(_n)
            return _f(img, meta, **kw)
        stages.append(Stage(s.name, fn, dict(s.params), s.enabled, s.group, s.label))
    return Pipeline(stages), calls


def test_incremental_rerun_reuses_prefix_and_matches_full_run():
    scene = scenes.load("astronaut", SIZE)
    pl, calls = _counting(P.default_pipeline())

    def run(previous=None, s=scene):
        met = P.StageMetrics(s, pl)
        calls.clear()                                                     # (the oracle ran stages too)
        return pl.run(s, metrics=met, previous=previous)
    first = run()
    assert calls == IDS and not any(r.reused for r in first)
    pl.set("sharpen", lam=1.5)
    second = run(first)
    assert calls == ["sharpen", "jpeg"]                                   # only the tail ran
    assert [r.reused for r in second] == [True] * 11 + [False] * 2
    full = run()
    for a, b in zip(second, full):
        assert np.array_equal(a.image, b.image) and a.meta == b.meta and a.metrics == b.metrics
    # unchanged -> everything reused; changed seed in an early stage -> everything re-run
    third = run(second)
    assert calls == [] and all(r.reused for r in third)
    pl.set("exposure", seed=3)
    fourth = run(third)
    assert calls == IDS and not any(r.reused for r in fourth)
    # a different input image invalidates the reuse
    run(fourth, scene * 0.5)
    assert calls == IDS


def test_simulate_previous():
    first = P.simulate("astronaut", SIZE)
    second = P.simulate("astronaut", SIZE, previous=first, jpeg={"quality": 50})
    assert [r.reused for r in second] == [True] * 12 + [False]
    assert second.final.meta["nbytes"] < first.final.meta["nbytes"]


def test_start_at_black_level_with_synthetic_raw():
    from isp import cfa, sensor
    scene = scenes.load("astronaut", SIZE)
    p = sensor.PRESETS["dslr"]
    dn, meta = sensor.capture(cfa.mosaic(scene * 0.4), 0.01, 200, p, flux_at_white=2e5)
    meta["cfa"] = "RGGB"
    res = P.default_pipeline().run(dn, meta, start="black_level")
    assert res.names == IDS[IDS.index("black_level"):]
    assert res["black_level"].meta["domain"] == "linear" and res.final.meta["domain"] == "bytes"
    assert res.final.image.shape == (SIZE, SIZE, 3)
    res2 = P.process_raw(dn, meta, white_balance={"method": "none"})
    assert res2["white_balance"].meta["wb_gains"] == [1.0, 1.0, 1.0]


def test_core_run_signature_is_backward_compatible():
    pl = Pipeline([Stage("a", lambda im, m: (im + 1, m))])
    res = pl.run(np.zeros((2, 2), np.float32))
    assert res.final.image.mean() == 1.0 and res.final.reused is False


# ------------------------------------------------------------------ catalogue
def test_stage_info_catalogue():
    assert list(P.STAGE_INFO) == IDS
    keys = {"label", "group", "short", "description", "slides", "doc", "params_help"}
    for sid, info in P.STAGE_INFO.items():
        assert keys <= set(info), sid
        assert len(info["short"]) <= 70 and info["group"] in P.GROUP_INFO
        assert (ROOT / info["doc"]).exists(), (sid, info["doc"])
        assert info["slides"].startswith("L")
        for pname in P.DEFAULT_PARAMS[sid]:
            assert pname in info["params_help"], (sid, pname)
    for g in P.GROUP_INFO.values():
        assert g["colour"].startswith("#") and len(g["colour"]) == 7 and g["label"]
    assert [s.label for s in P.default_pipeline().stages] == [P.STAGE_INFO[i]["label"] for i in IDS]


# ------------------------------------------------------------------ display
def test_render_every_stage(default_result):
    for rec in default_result:
        img = display.render(rec)
        assert img.dtype == np.uint8 and img.shape == (SIZE, SIZE, 3), rec.name
        small = display.render(rec, size=32)
        assert small.shape == (32, 32, 3) and small.dtype == np.uint8
    gray = display.render(default_result["adc"], mosaic_mode="gray")
    assert np.array_equal(gray[..., 0], gray[..., 1])
    colour = display.render(default_result["adc"])
    assert (colour[0::2, 0::2, 1:] == 0).all() and (colour[0::2, 1::2, 0] == 0).all()   # RGGB: R site shows only red


def test_render_mosaic_resize_is_nearest(default_result):
    big = display.render(default_result["adc"], size=SIZE * 2)
    assert set(map(tuple, big[:2, :2].reshape(-1, 3))) == {tuple(big[0, 0])}      # 2x2 block = one sample
    assert display.render(default_result["denoise"], size=(30, 40)).shape == (30, 40, 3)
    with pytest.raises(ValueError):
        display.render(np.zeros((4, 4, 2), np.float32), {"domain": "display"})


def test_display_helpers(default_result):
    img = display.render(default_result.final)
    z = display.crop_zoom(img, (0, 8, 4, 16), 3)
    assert z.shape == (24, 36, 3) and np.array_equal(z[:3, :3], np.broadcast_to(img[0, 4], (3, 3, 3)))
    d = display.difference(img, img)
    assert d.shape == img.shape and d.dtype == np.uint8 and (d == d[0, 0]).all()
    assert display.difference(img, 255 - img).mean() > d.mean()
    edges, counts = display.histograms(img, bins=16)
    assert edges.shape == (17,) and counts.shape == (3, 16) and counts.sum(axis=1).tolist() == [SIZE * SIZE] * 3
    th = display.stage_thumbnails(default_result, size=24)
    assert list(th) == IDS and all(t.dtype == np.uint8 and max(t.shape[:2]) == 24 for t in th.values())


def test_data_uri_round_trip():
    img = (np.random.default_rng(0).random((10, 12, 3)) * 255).astype(np.uint8)
    uri = display.to_data_uri(img)
    assert uri.startswith("data:image/png;base64,")
    back = np.asarray(Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGB"))
    assert np.array_equal(back, img)
    assert display.to_data_uri(img, "jpeg").startswith("data:image/jpeg;base64,")
    with pytest.raises(ValueError):
        display.to_data_uri(img, "bmp")


# ------------------------------------------------------------------ io
def test_load_image_and_save_png(tmp_path):
    disp = (np.linspace(0, 1, 100 * 150 * 3).reshape(100, 150, 3)).astype(np.float32)
    path = tmp_path / "x.png"
    assert isp_io.save_png(disp, path) == path
    lin = isp_io.load_image(path)
    assert lin.shape == (96, 144, 3) and lin.dtype == np.float32           # cropped to multiples of 16
    assert np.allclose(utils.linear_to_srgb(lin), disp[2:98, 3:147], atol=0.01)   # inverse sRGB
    sq = isp_io.load_image(path.read_bytes(), size=70)
    assert sq.shape == (64, 64, 3)
    wide = isp_io.load_image(str(path), size=96, square=False)
    assert wide.shape == (64, 96, 3)


class _FakeRaw:
    """Mimics the parts of rawpy.RawPy that load_raw documents."""
    color_desc = b"RGBG"
    raw_pattern = np.array([[0, 1], [3, 2]])             # R G / G B   (3 = second green)
    black_level_per_channel = [60, 64, 68, 64]
    white_level = 16383
    camera_whitebalance = [2.0, 1.0, 1.5, 1.0]
    raw_image_visible = (np.arange(9 * 8).reshape(9, 8) % 1000).astype(np.uint16)   # odd height

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_load_raw_without_rawpy_gives_helpful_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "rawpy", None)       # makes ``import rawpy`` fail
    with pytest.raises(ImportError, match="pip install rawpy"):
        isp_io.load_raw("photo.dng")


def test_load_raw_with_fake_rawpy(monkeypatch):
    fake = types.SimpleNamespace(imread=lambda src: _FakeRaw())
    monkeypatch.setitem(sys.modules, "rawpy", fake)
    mosaic, meta = isp_io.load_raw(b"not really a raw file")
    assert mosaic.shape == (8, 8) and mosaic.dtype == np.float32          # cropped to even size
    assert meta["domain"] == "dn" and meta["cfa"] == "RGGB"
    assert meta["black_level"] == 64.0 and meta["white_level"] == 16383.0 and meta["bits"] == 14
    assert meta["camera_wb"] == [2.0, 1.0, 1.5]
    # per-channel black levels were unified: the R site (0, 0) lost 60 and gained 64
    assert mosaic[0, 0] == int(_FakeRaw.raw_image_visible[0, 0]) - 60 + 64
    assert mosaic[0, 1] == int(_FakeRaw.raw_image_visible[0, 1])     # G site
    # it enters the pipeline at black_level and the camera white balance is available
    res = P.process_raw(mosaic, meta, white_balance={"method": "camera"}, enabled={"hot_pixels": False})
    assert res["white_balance"].meta["wb_gains"] == [2.0, 1.0, 1.5]
    assert res.final.image.shape == (8, 8, 3)
