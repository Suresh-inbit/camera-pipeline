"""JPEG module: DCT, quantisation, entropy coding bookkeeping, codec behaviour, Pillow reference."""
import numpy as np
import pytest

from isp import jpeg, scenes, utils


@pytest.fixture(scope="module")
def photo():
    return utils.linear_to_srgb(scenes.photo("astronaut", 128))


# ---------------------------------------------------------------- DCT
def test_dct_basis_orthonormal():
    B = jpeg.dct_basis()
    assert B.shape == (8, 8, 8, 8)
    M = B.reshape(64, 64).astype(np.float64)
    assert np.allclose(M @ M.T, np.eye(64), atol=1e-6)


def test_dct_round_trip_exact():
    x = utils.rng(0).random((5, 8, 8)) * 255 - 128
    assert np.allclose(jpeg.idct2_blocks(jpeg.dct2_blocks(x)), x, atol=1e-5)


def test_dct_matches_basis_projection():
    x = utils.rng(1).random((8, 8)) * 100
    c = jpeg.dct2_blocks(x)
    B = jpeg.dct_basis().astype(np.float64)
    assert np.allclose(c, np.einsum("uvxy,xy->uv", B, x), atol=1e-4)


def test_constant_block_only_dc():
    c = jpeg.dct2_blocks(np.full((8, 8), 37.0))
    assert c[0, 0] == pytest.approx(37.0 * 8)
    c[0, 0] = 0
    assert np.abs(c).max() < 1e-9


# ---------------------------------------------------------------- quantisation tables
def test_quant_tables():
    qy, qc = jpeg.quant_tables(50)
    assert np.array_equal(qy, jpeg.BASE_QY) and np.array_equal(qc, jpeg.BASE_QC)
    qy1, qc1 = jpeg.quant_tables(100)
    assert (qy1 == 1).all() and (qc1 == 1).all()
    prev = jpeg.quant_tables(95)[0].astype(float).mean()
    for q in (80, 60, 40, 20, 5):
        cur = jpeg.quant_tables(q)[0].astype(float).mean()
        assert cur > prev
        prev = cur
    assert jpeg.quant_tables(1)[0].max() == 255


# ---------------------------------------------------------------- entropy coding bookkeeping
def test_zigzag_is_standard_permutation():
    assert sorted(jpeg.ZIGZAG) == list(range(64))
    first = [(int(i) // 8, int(i) % 8) for i in jpeg.ZIGZAG[:6]]
    assert first == [(0, 0), (0, 1), (1, 0), (2, 0), (1, 1), (0, 2)]
    assert int(jpeg.ZIGZAG[-1]) == 63
    b = np.arange(64).reshape(8, 8)
    assert np.array_equal(jpeg.inverse_zigzag(jpeg.zigzag(b)), b)


def test_rle_round_trip_and_special_symbols():
    rng = utils.rng(0)
    for _ in range(50):
        ac = np.zeros(63, dtype=np.int64)
        k = rng.integers(0, 12)
        ac[rng.integers(0, 63, k)] = rng.integers(-300, 300, k)
        assert np.array_equal(jpeg.rle_decode_ac(jpeg.rle_encode_ac(ac)), ac)
    ac = np.zeros(63, dtype=np.int64)
    ac[40] = 5                                    # 40 zeros: two ZRL + run 8
    sym = jpeg.rle_encode_ac(ac)
    assert sym[:2] == [(15, 0, 0)] * 2 and sym[2][0] == 8 and sym[-1] == (0, 0, 0)
    ac[62] = -1                                   # last coefficient non-zero: no EOB
    assert jpeg.rle_encode_ac(ac)[-1][1] == 1
    assert jpeg.rle_encode_ac(np.zeros(63)) == [(0, 0, 0)]


def test_amplitude_bits_round_trip():
    for v in range(-1000, 1001):
        s = int(jpeg.bit_length(v))
        assert jpeg.amplitude_value(jpeg.amplitude_bits(v, s), s) == v


def test_vectorised_stats_match_rle():
    rng = utils.rng(2)
    ac = np.where(rng.random((200, 63)) < 0.08, rng.integers(-40, 40, (200, 63)), 0)
    hist, amp = jpeg._ac_stats(ac)
    ref = np.zeros(256, dtype=np.int64)
    ref_amp = 0
    for row in ac:
        for run, size, _ in jpeg.rle_encode_ac(row):
            ref[(run << 4) | size] += 1
            ref_amp += size
    assert np.array_equal(hist, ref) and amp == ref_amp


def test_huffman_prefix_free_decodable_and_limited():
    rng = utils.rng(3)
    freqs = {s: int(c) for s, c in enumerate(rng.integers(1, 5000, 60))}
    freqs[7] = 10 ** 9                            # very skewed: forces the depth limit logic
    for s in range(60, 120):
        freqs[s] = 1
    lengths = jpeg.huffman_lengths(freqs)
    assert set(lengths) == set(freqs) and max(lengths.values()) <= 16
    assert sum(2.0 ** -n for n in lengths.values()) < 1.0        # all-ones code stays free
    codes = jpeg.huffman_codes(lengths)
    words = [format(c, f"0{n}b") for c, n in codes.values()]
    assert not any(a != b and b.startswith(a) for a in words for b in words)
    stream = list(rng.choice(list(freqs), 500))
    assert jpeg.huffman_decode(jpeg.huffman_encode(stream, codes), codes) == stream


def test_huffman_frequent_symbols_are_short():
    lengths = jpeg.huffman_lengths({0: 1000, 1: 100, 2: 10, 3: 1})
    assert lengths[0] <= lengths[1] <= lengths[2] <= lengths[3]
    assert jpeg.huffman_lengths({5: 10}) == {5: 1}


# ---------------------------------------------------------------- codec
def test_quality_100_is_nearly_lossless(photo):
    out, info = jpeg.roundtrip(photo, 100, "4:4:4")
    assert info["psnr"] > 45


def test_monotone_in_quality(photo):
    psnrs, sizes = [], []
    for q in (10, 30, 50, 75, 90):
        _, info = jpeg.roundtrip(photo, q, "4:2:0")
        psnrs.append(info["psnr"])
        sizes.append(info["nbytes"])
    assert psnrs == sorted(psnrs) and sizes == sorted(sizes)
    assert len(set(sizes)) == 5


def test_420_smaller_than_444(photo):
    for q in (30, 75):
        assert jpeg.roundtrip(photo, q, "4:2:0")[1]["nbytes"] < jpeg.roundtrip(photo, q, "4:4:4")[1]["nbytes"]
        assert jpeg.roundtrip(photo, q, "4:2:2")[1]["nbytes"] < jpeg.roundtrip(photo, q, "4:4:4")[1]["nbytes"]


@pytest.mark.parametrize("shape", [(37, 53), (50, 66), (17, 31), (64, 64)])
@pytest.mark.parametrize("sub", ["4:4:4", "4:2:2", "4:2:0"])
def test_shape_preserved(shape, sub):
    img = utils.rng(0).random((*shape, 3)).astype(np.float32)
    out, info = jpeg.roundtrip(img, 80, sub)
    assert out.shape == img.shape and out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_size_breakdown_adds_up(photo):
    d = jpeg.encode(photo, 60, "4:2:0")
    assert set(d.bits) == {"Y", "Cb", "Cr"} and set(d.bits["Y"]) == {"DC", "AC"}
    assert d.payload_bits == sum(v for c in d.bits.values() for v in c.values())
    assert d.nbytes == int(np.ceil(d.payload_bits / 8)) + d.header_bytes
    assert d.bits["Y"]["AC"] > d.bits["Y"]["DC"]


@pytest.mark.parametrize("sub", ["4:4:4", "4:2:0"])
@pytest.mark.parametrize("q", [30, 75])
def test_against_pillow(photo, sub, q):
    ours, info = jpeg.roundtrip(photo, q, sub)
    ref, nbytes = jpeg.pillow_roundtrip(photo, q, sub, optimize=True)
    assert abs(info["psnr"] - utils.psnr(photo, ref)) < 1.0
    assert abs(info["nbytes"] - nbytes) / nbytes < 0.15


# ---------------------------------------------------------------- generation loss
def test_generation_loss_shift_vs_no_shift(photo):
    a = jpeg.generation_loss(photo, n=8, quality=50, shift=0)
    b = jpeg.generation_loss(photo, n=8, quality=50, shift=4)
    pa, pb = [r["psnr"] for r in a], [r["psnr"] for r in b]
    assert pa[1] - pa[-1] < 0.5                     # same grid: converges quickly
    assert pb[-1] < pb[0] - 0.8                     # moved grid: keeps losing
    assert pb[-1] < pa[-1] - 0.5


# ---------------------------------------------------------------- content dependence
def test_noise_costs_bits_flat_is_free():
    clean = utils.linear_to_srgb(scenes.load("sunset", 128))    # smooth, so noise dominates the bits
    noisy = np.clip(clean + utils.rng(0).normal(0, 0.03, clean.shape), 0, 1).astype(np.float32)
    flat = np.full_like(clean, 0.5)
    b_clean = jpeg.encode(clean, 75).payload_bits
    assert jpeg.encode(noisy, 75).payload_bits > 1.8 * b_clean
    assert jpeg.encode(flat, 75).payload_bits < 0.15 * b_clean


def test_blockiness_rises_as_quality_falls(photo):
    vals = [jpeg.blockiness(jpeg.roundtrip(photo, q, "4:2:0")[0]) for q in (90, 50, 20, 5)]
    assert vals == sorted(vals) and vals[-1] > vals[0] + 0.15


# ---------------------------------------------------------------- helpers
def test_block_helpers(photo):
    g = utils.luminance(photo)
    c = jpeg.block_dct(g)
    assert c.shape == (16, 16, 8, 8)
    b = jpeg.block_before_after(g, (3, 4), quality=50)
    assert np.allclose(b["coefs"], c[3, 4], atol=1e-6)
    assert (b["quantised"] == 0).sum() > 20 and np.abs(b["error"]).max() < 40


def test_exif_round_trip(tmp_path, photo):
    p = tmp_path / "x.jpg"
    n = jpeg.write_jpeg_with_exif(photo, p, 90, exif={"Make": "TestCam", "Model": "EE5176",
                                                      "Software": "isp", "ISOSpeedRatings": 400})
    assert n == p.stat().st_size
    ex = jpeg.read_exif(p)
    assert ex["Make"] == "TestCam" and ex["Model"] == "EE5176" and ex["ISOSpeedRatings"] == 400


def test_raw_vs_jpeg_size():
    r = jpeg.raw_vs_jpeg_size((100, 200, 3), 12, 10_000)
    assert r["raw_bytes"] == 100 * 200 * 12 / 8 and r["ratio"] == pytest.approx(3.0)
