from dataclasses import replace

import numpy as np

from isp import sensor as S


def flat(v, n=64):
    return np.full((n, n), v, np.float32)


def test_snr_theory_slide32():
    r = S.PRESETS["retiga_4000r"]
    a = S.PRESETS["aptina_mt9p031"]
    assert abs(S.snr_theory(1000, r.qe, 1.0, r.dark_current, r.read_noise) - 20.8) < 0.1
    assert abs(S.snr_theory(1000 / 11, a.qe, 1.0, a.dark_current, a.read_noise) - 6.5) < 0.1


def test_dynamic_range_slide33():
    r = S.PRESETS["retiga_4000r"]
    a = S.PRESETS["aptina_mt9p031"]
    for p, t, ref in [(r, 1, 3313), (r, 1 / 60, 3333), (a, 1, 1500), (a, 1 / 60, 3200)]:
        dr = S.dynamic_range(p.full_well, p.dark_current, t, p.read_noise)
        assert abs(dr / ref - 1) < 0.02, (p.name, t, dr)


def test_monte_carlo_matches_theory():
    for name in ("retiga_4000r", "aptina_mt9p031"):
        p = S.clean_params(S.PRESETS[name], bits=16)
        m = S.measure_snr(p, value=0.05, exposure=0.02, n=48, shape=(48, 48))
        th = S.snr_theory_params(0.05, 0.02, p)
        assert abs(m["snr"] / th - 1) < 0.03, (name, m["snr"], th)


def test_shot_noise_limit_sqrt_scaling():
    p = S.clean_params(S.PRESETS["dslr"], dark_current=0.0, read_noise=0.0, bits=16)
    m1 = S.measure_snr(p, 0.5, 0.02, n=32, shape=(48, 48))
    m2 = S.measure_snr(p, 0.5, 0.04, n=32, shape=(48, 48))
    assert abs(m1["snr"] / np.sqrt(m1["mean_e"]) - 1) < 0.03
    assert abs(m2["snr"] / m1["snr"] / np.sqrt(2) - 1) < 0.04


def test_averaging_sqrt_M():
    p = S.clean_params(S.PRESETS["aptina_mt9p031"], bits=16, dark_current=0.0)
    img = flat(0.01, 96)
    frames = [S.capture(img, 0.01, 100, p, seed=k)[0] for k in range(16)]
    s1 = np.std(frames[0])
    s16 = np.std(S.average_frames(frames))
    assert abs(s1 / s16 / 4 - 1) < 0.08


def test_saturation_and_adc_range():
    p = S.clean_params(S.PRESETS["phone"])
    e = S.expose(flat(1.0), 1.0, p)
    assert np.all(e == p.full_well)
    dn = S.adc(S.expose(flat(0.3), 0.01, p), p, iso=3200)
    assert np.all(dn == np.round(dn)) and dn.min() >= 0 and dn.max() <= p.white_level
    assert S.adc(e, p).max() <= p.white_level


def test_determinism_and_seed_dependence():
    p = S.PRESETS["dslr"]
    a, _ = S.capture(flat(0.2), 0.01, 400, p, seed=3)
    b, _ = S.capture(flat(0.2), 0.01, 400, p, seed=3)
    c, _ = S.capture(flat(0.2), 0.01, 400, p, seed=4)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_fixed_pattern_identical_across_shots():
    p = replace(S.PRESETS["aptina_mt9p031"], hot_pixel_fraction=2e-3)
    f1, f2 = S.fixed_pattern((64, 64), p), S.fixed_pattern((64, 64), p)
    assert np.array_equal(f1["dark"], f2["dark"]) and np.array_equal(f1["prnu"], f2["prnu"])
    d1 = S.dark_frame((64, 64), 1.0, 100, p, seed=1)
    d2 = S.dark_frame((64, 64), 1.0, 100, p, seed=2)
    hot = f1["hot"]
    assert hot.any()
    assert d1[hot].mean() > d1[~hot].mean() + 10 and d2[hot].mean() > d2[~hot].mean() + 10
    assert np.corrcoef(d1.ravel(), d2.ravel())[0, 1] > 0.1


def test_meta_and_normalize():
    p = S.PRESETS["dslr"]
    dn, meta = S.capture(flat(0.5), 0.01, 200, p, meta=dict(cfa="RGGB"))
    assert meta["domain"] == "dn" and meta["cfa"] == "RGGB" and meta["iso"] == 200
    assert meta["bits"] == 14 and meta["white_level"] == 2 ** 14 - 1
    x = S.normalize(dn, meta)
    assert x.min() >= 0 and x.max() <= 1


def test_dark_frame_subtraction_helps():
    p = S.PRESETS["aptina_mt9p031"]
    shape = (96, 96)
    img = np.full(shape, 0.02, np.float32)
    t = 2.0
    dn, _ = S.capture(img, t, 100, p, flux_at_white=2000, seed=5)
    dark_free = replace(p, dark_current=0.0, dsnu=0.0, hot_pixel_fraction=0.0)
    ref = S.capture(img, t, 100, dark_free, flux_at_white=2000, seed=5)[0]
    dark = S.master_dark(shape, t, 100, p, n=16)
    sub = S.subtract_dark(dn, dark, p.black_level)
    err0 = np.mean((dn - ref) ** 2)
    err1 = np.mean((sub - ref) ** 2)
    assert err1 < 0.5 * err0


def test_fix_hot_pixels():
    p = replace(S.PRESETS["aptina_mt9p031"], hot_pixel_fraction=2e-3)
    dn, _ = S.capture(flat(0.1), 1.0, 100, p, flux_at_white=500, seed=2)
    hot = S.fixed_pattern(dn.shape, p)["hot"]
    assert hot.sum() > 3
    fixed = S.fix_hot_pixels(dn)
    med = np.median(dn)
    assert np.mean(dn[hot] > med + 100) > 0.9
    assert np.mean(fixed[hot] > med + 100) < 0.1
    assert np.mean(np.abs(fixed[~hot] - dn[~hot])) < 1.0


def test_iso_vs_exposure_s40():
    p = S.clean_params(S.PRESETS["retiga_4000r"], bits=16, dark_current=0.0)
    kw = dict(value=0.5, n=32, shape=(32, 32))
    base = S.measure_snr(p, exposure=0.01, iso=100, **kw)["snr"]
    iso8 = S.measure_snr(p, exposure=0.01, iso=800, **kw)["snr"]
    long = S.measure_snr(p, exposure=0.08, iso=100, **kw)["snr"]
    assert abs(iso8 / base - 1) < 0.04
    assert long > 1.8 * base


def test_photon_transfer_slope_is_gain():
    p = S.clean_params(S.PRESETS["dslr"], dark_current=0.0)
    ptc = S.photon_transfer(p, exposure=0.01, values=np.linspace(0.05, 0.9, 8), shape=(96, 96))
    slope = np.polyfit(ptc["mean"], ptc["var"], 1)[0]
    assert abs(slope / p.gain(100) - 1) < 0.08
