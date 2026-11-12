"""Baseline JPEG codec (lecture 5, compression and storage).

An educational re-implementation of the codec diagram on L5 s20:

    RGB -> YCbCr -> chroma subsampling -> 8x8 blocks -> DCT -> quantisation -> entropy coding
    entropy decoding -> dequantisation -> inverse DCT -> reassemble blocks -> upsample -> RGB

What is real: the colour transform (JPEG full range), box-average chroma subsampling, whole
MCU padding by edge replication, the orthonormal 8x8 DCT-II, the Annex K quantisation tables
with the IJG quality rule, zig-zag scan, DC differential coding, AC run-length (run, size)
symbols with EOB and ZRL, and Huffman coding of those symbols.

What is simplified: no ``.jpg`` bitstream is written.  The Huffman tables are *optimal tables
built from the image's own symbol statistics* (like libjpeg ``optimize_coding=True``, code
lengths limited to 16 bits, canonical codes), not the fixed Annex K tables.  The size is
``Huffman code bits + amplitude bits`` rounded up to bytes plus an estimate of the header
(markers, 2 quantisation tables, 4 Huffman tables).  Byte stuffing (0xFF 0x00) is ignored.
The decoder upsamples chroma bilinearly and returns 8-bit quantised values.

Images are display-referred (sRGB-encoded) float32 RGB in [0, 1] (see docs/conventions.md).
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.fft import dctn, idctn

from . import utils

# ---------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------
#: Annex K, table K.1: luminance quantisation table (quality 50).
BASE_QY = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]], dtype=np.int32)
#: Annex K, table K.2: chrominance quantisation table (quality 50).
BASE_QC = np.array([
    [17, 18, 24, 47, 99, 99, 99, 99],
    [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99],
    [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99]], dtype=np.int32)

#: chroma subsampling: (row factor, column factor) of the chroma planes
SUBSAMPLING = {"4:4:4": (1, 1), "4:2:2": (1, 2), "4:2:0": (2, 2)}
#: Pillow's integer code for the same modes
_PIL_SUBSAMPLING = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}

# ---------------------------------------------------------------------------
# DCT, quantisation tables
# ---------------------------------------------------------------------------
def _dct_matrix() -> np.ndarray:
    k = np.arange(8)[:, None]
    n = np.arange(8)[None, :]
    c = np.cos((2 * n + 1) * k * np.pi / 16.0) * np.sqrt(2.0 / 8.0)
    c[0] /= np.sqrt(2.0)
    return c


def dct_basis() -> np.ndarray:
    """The 64 2-D DCT basis functions as an array ``B[u, v, x, y]`` of shape (8, 8, 8, 8).

    ``B[u, v] = c_u (x) c_v`` with ``c_k[n] = a_k cos((2n+1) k pi / 16)``, ``a_0 = sqrt(1/8)``,
    ``a_k = sqrt(2/8)``.  Orthonormal: ``sum_xy B[u,v] B[u',v'] = delta``.
    Row index ``u`` is the vertical frequency, ``v`` the horizontal one.

    Reference: L5 s20 (2-D DCT basis functions).
    """
    c = _dct_matrix()
    return (c[:, None, :, None] * c[None, :, None, :]).astype(np.float32)


def dct2_blocks(blocks) -> np.ndarray:
    """Orthonormal 2-D DCT-II over the last two axes (any number of leading block axes)."""
    return dctn(np.asarray(blocks, dtype=np.float64), type=2, axes=(-2, -1), norm="ortho")


def idct2_blocks(coefs) -> np.ndarray:
    """Inverse of :func:`dct2_blocks`."""
    return idctn(np.asarray(coefs, dtype=np.float64), type=2, axes=(-2, -1), norm="ortho")


def quant_tables(quality: int = 75) -> tuple[np.ndarray, np.ndarray]:
    """Luminance and chrominance quantisation tables for an IJG quality in 1..100.

    ``scale = 5000 / q`` for ``q < 50`` else ``200 - 2 q``;
    ``Q = clip(floor((base * scale + 50) / 100), 1, 255)``.
    Quality 50 gives the Annex K tables, quality 100 gives all ones (almost lossless).

    Returns ``(QY, QC)``, ``(8, 8)`` int32 arrays.  Reference: L5 s20 (quantisation).
    """
    q = int(np.clip(quality, 1, 100))
    scale = 5000 // q if q < 50 else 200 - 2 * q
    f = lambda base: np.clip((base.astype(np.int64) * scale + 50) // 100, 1, 255).astype(np.int32)
    return f(BASE_QY), f(BASE_QC)


# ---------------------------------------------------------------------------
# blocks
# ---------------------------------------------------------------------------
def split_blocks(plane) -> np.ndarray:
    """(H, W) plane, H and W multiples of 8 -> (H/8, W/8, 8, 8) blocks."""
    h, w = plane.shape
    return plane.reshape(h // 8, 8, w // 8, 8).transpose(0, 2, 1, 3)


def merge_blocks(blocks) -> np.ndarray:
    """Inverse of :func:`split_blocks`."""
    nby, nbx = blocks.shape[:2]
    return np.asarray(blocks).transpose(0, 2, 1, 3).reshape(nby * 8, nbx * 8)


def block_dct(img_gray) -> np.ndarray:
    """DCT of every 8x8 block of a grey image in [0, 1] (edge-padded to whole blocks).

    Works on the 0..255 scale minus 128, as the codec does.  Returns the coefficients,
    shape (H/8, W/8, 8, 8); ``out[i, j, u, v]`` is the (u, v) coefficient of block (i, j).
    """
    g = np.asarray(img_gray, dtype=np.float64) * 255.0 - 128.0
    h, w = g.shape
    g = np.pad(g, ((0, -h % 8), (0, -w % 8)), mode="edge")
    return dct2_blocks(split_blocks(g))


def block_before_after(img_gray, block=(0, 0), quality: int = 50, chroma: bool = False) -> dict:
    """Follow one 8x8 block through DCT -> quantisation -> reconstruction.

    Returns a dict with ``pixels`` (0..255), ``coefs`` (DCT), ``q`` (table), ``quantised``
    (integers), ``recon`` (0..255 after dequantisation + IDCT) and ``error``.
    """
    i, j = block
    g = np.asarray(img_gray, dtype=np.float64) * 255.0
    pix = g[8 * i:8 * i + 8, 8 * j:8 * j + 8]
    coefs = dct2_blocks(pix - 128.0)
    q = quant_tables(quality)[1 if chroma else 0]
    qz = np.round(coefs / q).astype(np.int32)
    rec = idct2_blocks(qz * q) + 128.0
    return dict(pixels=pix, coefs=coefs, q=q, quantised=qz, recon=rec, error=rec - pix)


def _zigzag_order() -> np.ndarray:
    idx = sorted(((r, c) for r in range(8) for c in range(8)),
                 key=lambda rc: (rc[0] + rc[1], rc[1] if (rc[0] + rc[1]) % 2 == 0 else rc[0]))
    return np.array([r * 8 + c for r, c in idx], dtype=np.int64)


#: flat raster index (row*8+col) of the k-th zig-zag position; first entries (0,0),(0,1),(1,0),(2,0),(1,1),(0,2)
ZIGZAG = _zigzag_order()


def zigzag(block) -> np.ndarray:
    """(..., 8, 8) -> (..., 64) in zig-zag order (low to high frequency)."""
    b = np.asarray(block)
    return b.reshape(*b.shape[:-2], 64)[..., ZIGZAG]


def inverse_zigzag(vec) -> np.ndarray:
    """(..., 64) zig-zag vector -> (..., 8, 8) block."""
    v = np.asarray(vec)
    out = np.empty_like(v)
    out[..., ZIGZAG] = v
    return out.reshape(*v.shape[:-1], 8, 8)


# ---------------------------------------------------------------------------
# entropy coding: symbols
# ---------------------------------------------------------------------------
def bit_length(a) -> np.ndarray:
    """Number of bits of ``|a|`` (the JPEG 'size' / category); 0 for 0."""
    return np.frexp(np.abs(np.asarray(a)).astype(np.float64))[1].astype(np.int64)


def amplitude_bits(v: int, size: int) -> int:
    """JPEG amplitude code of ``v`` in ``size`` bits (negative values are one's complemented)."""
    return int(v) if v > 0 else int(v) + (1 << size) - 1


def amplitude_value(bits: int, size: int) -> int:
    """Inverse of :func:`amplitude_bits`."""
    if size == 0:
        return 0
    return bits if bits >= (1 << (size - 1)) else bits - (1 << size) + 1


EOB = (0, 0)   # (run, size) of end-of-block
ZRL = (15, 0)  # sixteen zeros


def rle_encode_ac(ac) -> list:
    """Run-length code the 63 zig-zag AC coefficients of one block.

    Returns a list of ``(run, size, amplitude_bits)``: a coefficient preceded by ``run`` zeros
    (``run <= 15``, each further 16 zeros emit a ZRL ``(15, 0, 0)``), ``size`` = bit length of
    its magnitude; ``(0, 0, 0)`` = EOB when the rest of the block is zero.
    """
    ac = np.asarray(ac, dtype=np.int64)
    out, last = [], -1
    for p in np.flatnonzero(ac):
        run = int(p - last - 1)
        out += [(15, 0, 0)] * (run // 16)
        v = int(ac[p])
        s = int(bit_length(v))
        out.append((run % 16, s, amplitude_bits(v, s)))
        last = int(p)
    if last < len(ac) - 1:
        out.append((0, 0, 0))
    return out


def rle_decode_ac(symbols, n: int = 63) -> np.ndarray:
    """Inverse of :func:`rle_encode_ac`."""
    ac = np.zeros(n, dtype=np.int64)
    pos = 0
    for run, size, bits in symbols:
        if (run, size) == EOB:
            break
        if (run, size) == ZRL:
            pos += 16
            continue
        pos += run
        ac[pos] = amplitude_value(bits, size)
        pos += 1
    return ac


def _ac_stats(ac) -> tuple[np.ndarray, int]:
    """Vectorised symbol histogram (256 bins, symbol = run<<4 | size) and amplitude bits of
    ``(N, 63)`` AC arrays; identical in content to counting :func:`rle_encode_ac` outputs."""
    ac = np.asarray(ac)
    nb, npos = np.nonzero(ac)
    hist = np.zeros(256, dtype=np.int64)
    amp = 0
    if len(nb):
        vals = ac[nb, npos]
        prev = np.concatenate([[-1], npos[:-1]])
        first = np.concatenate([[True], nb[1:] != nb[:-1]])
        prev[first] = -1
        run = npos - prev - 1
        size = bit_length(vals)
        hist += np.bincount(((run % 16) << 4) | size, minlength=256)
        hist[0xF0] += int((run // 16).sum())
        amp = int(size.sum())
        last_pos = np.full(ac.shape[0], -1)
        last_pos[nb] = npos                      # row-major order: last write is the last nz
    else:
        last_pos = np.full(ac.shape[0], -1)
    hist[0] += int((last_pos < ac.shape[1] - 1).sum())   # EOB
    return hist, amp


def _dc_stats(dc) -> tuple[np.ndarray, int]:
    """DC categories (histogram over 0..11) and amplitude bits of differentially coded DCs."""
    diff = np.diff(np.asarray(dc, dtype=np.int64), prepend=0)
    cat = bit_length(diff)
    return np.bincount(cat, minlength=256).astype(np.int64), int(cat.sum())


# ---------------------------------------------------------------------------
# entropy coding: Huffman
# ---------------------------------------------------------------------------
def huffman_lengths(freqs, max_len: int = 16) -> dict:
    """Optimal prefix-code lengths (<= ``max_len``) for ``{symbol: count}``.

    Builds the Huffman tree, then limits depth with the libjpeg/Annex K.2 adjustment.
    As in libjpeg a dummy symbol of count 1 reserves the all-ones code (JPEG forbids it).
    """
    syms = [s for s, c in freqs.items() if c > 0]
    if not syms:
        return {}
    DUMMY = -1
    items = [(freqs[s], i, [s]) for i, s in enumerate(syms)] + [(1, len(syms), [DUMMY])]
    heapq.heapify(items)
    depth = {s: 0 for s in syms + [DUMMY]}
    uid = len(items)
    while len(items) > 1:
        c1, _, g1 = heapq.heappop(items)
        c2, _, g2 = heapq.heappop(items)
        for s in g1 + g2:
            depth[s] += 1
        uid += 1
        heapq.heappush(items, (c1 + c2, uid, g1 + g2))
    order = sorted(depth, key=lambda s: (depth[s], s == DUMMY, s))   # dummy last in its level
    nmax = max(depth.values())
    bits = [0] * (max(nmax, max_len) + 2)
    for s in order:
        bits[depth[s]] += 1
    for i in range(nmax, max_len, -1):                               # Annex K.2 depth limit
        while bits[i] > 0:
            j = i - 2
            while bits[j] == 0:
                j -= 1
            bits[i] -= 2
            bits[i - 1] += 1
            bits[j + 1] += 2
            bits[j] -= 1
    i = max(k for k, b in enumerate(bits) if b)
    bits[i] -= 1                                                     # drop the dummy codeword
    real = [s for s in order if s != DUMMY]
    lengths, k = {}, 0
    for length in range(1, len(bits)):
        for _ in range(bits[length]):
            lengths[real[k]] = length
            k += 1
    return lengths


def huffman_codes(lengths) -> dict:
    """Canonical Huffman codes ``{symbol: (code, length)}`` from code lengths."""
    codes, code, prev = {}, 0, 0
    for s in sorted(lengths, key=lambda s: (lengths[s], s)):
        code <<= lengths[s] - prev
        prev = lengths[s]
        codes[s] = (code, prev)
        code += 1
    return codes


def huffman_encode(symbols, codes) -> str:
    """Concatenate the codewords of ``symbols`` into a string of '0'/'1'."""
    return "".join(format(codes[s][0], f"0{codes[s][1]}b") for s in symbols)


def huffman_decode(bitstring: str, codes) -> list:
    """Decode a '0'/'1' string produced by :func:`huffman_encode`."""
    table = {format(c, f"0{n}b"): s for s, (c, n) in codes.items()}
    out, cur = [], ""
    for b in bitstring:
        cur += b
        if cur in table:
            out.append(table[cur])
            cur = ""
    if cur:
        raise ValueError("trailing bits do not form a codeword")
    return out


def _table_cost(hist, lengths) -> int:
    """Code bits of a symbol histogram under a length table."""
    return int(sum(int(c) * lengths[s] for s, c in enumerate(hist) if c))


# ---------------------------------------------------------------------------
# colour planes, subsampling
# ---------------------------------------------------------------------------
def _pad_edge(plane, mh, mw):
    h, w = plane.shape
    return np.pad(plane, ((0, -h % mh), (0, -w % mw)), mode="edge")


def subsample(plane, fy: int, fx: int) -> np.ndarray:
    """Box-average down by ``(fy, fx)``; the plane size must be a multiple of the factors."""
    if fy == 1 and fx == 1:
        return plane
    h, w = plane.shape
    return plane.reshape(h // fy, fy, w // fx, fx).mean(axis=(1, 3))


def upsample(plane, fy: int, fx: int) -> np.ndarray:
    """Bilinear upsampling by ``(fy, fx)`` (pixel centres aligned, as the box average implies)."""
    if fy == 1 and fx == 1:
        return plane
    h, w = plane.shape
    return cv2.resize(np.ascontiguousarray(plane, dtype=np.float32), (w * fx, h * fy),
                      interpolation=cv2.INTER_LINEAR).astype(np.float64)


def _scan_order(blocks, vy: int, vx: int) -> np.ndarray:
    """(nby, nbx, 8, 8) -> (N, 8, 8) in MCU order (``vy x vx`` blocks per MCU, row-major)."""
    nby, nbx = blocks.shape[:2]
    if vy == 1 and vx == 1:
        return blocks.reshape(-1, 8, 8)
    b = blocks.reshape(nby // vy, vy, nbx // vx, vx, 8, 8).transpose(0, 2, 1, 3, 4, 5)
    return b.reshape(-1, 8, 8)


# ---------------------------------------------------------------------------
# the codec
# ---------------------------------------------------------------------------
@dataclass
class JPEGData:
    """Everything the decoder needs plus the size accounting (no real bitstream)."""
    shape: tuple                 # (H, W) of the original image
    subsampling: str
    quality: int
    qy: np.ndarray
    qc: np.ndarray
    coeffs: dict                 # 'Y', 'Cb', 'Cr' -> (nby, nbx, 8, 8) int32 quantised coefficients
    tables: dict                 # 'Y_DC', 'Y_AC', 'C_DC', 'C_AC' -> {symbol: code length}
    bits: dict = field(default_factory=dict)   # breakdown: bits['Y']['DC'] ... (code + amplitude)
    header_bytes: int = 0

    @property
    def payload_bits(self) -> int:
        return int(sum(v for comp in self.bits.values() for v in comp.values()))

    @property
    def nbytes(self) -> int:
        """Estimated file size: entropy-coded payload (rounded up) + header."""
        return int(np.ceil(self.payload_bits / 8.0)) + self.header_bytes

    @property
    def bpp(self) -> float:
        return 8.0 * self.nbytes / (self.shape[0] * self.shape[1])


def _header_bytes(tables) -> int:
    """SOI + JFIF APP0 + 2 DQT + SOF0 + 4 DHT + SOS + EOI."""
    dht = sum(5 + 16 + len(t) for t in tables.values())
    return 2 + 18 + 2 * (4 + 64) + 19 + dht + 14 + 2


def encode(img, quality: int = 75, subsampling: str = "4:2:0") -> JPEGData:
    """Encode a display-referred RGB image (float32 [0, 1]) with the baseline JPEG pipeline.

    Steps: YCbCr -> chroma box-average -> MCU edge padding -> 8x8 blocks -> level shift ->
    DCT -> divide by the quality-scaled Annex K table and round -> zig-zag, DC differences,
    AC (run, size) symbols -> optimal Huffman tables -> bit counts.

    Reference: L5 s20 (codec), L5 s19.
    """
    if subsampling not in SUBSAMPLING:
        raise ValueError(f"subsampling must be one of {tuple(SUBSAMPLING)}")
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape[:2]
    vy, vx = SUBSAMPLING[subsampling]
    ycc = utils.rgb_to_ycbcr(img).astype(np.float64) * 255.0 - 128.0
    qy, qc = quant_tables(quality)
    coeffs = {}
    for k, name in enumerate(("Y", "Cb", "Cr")):
        plane = _pad_edge(ycc[..., k], 8 * vy, 8 * vx)             # whole MCUs
        if k:
            plane = subsample(plane, vy, vx)
        q = qy if k == 0 else qc
        coeffs[name] = np.round(dct2_blocks(split_blocks(plane)) / q).astype(np.int32)

    # ---- entropy coding statistics (per component), blocks in MCU scan order
    hist, amp = {}, {}
    for name in coeffs:
        scan = _scan_order(coeffs[name], vy, vx) if name == "Y" else coeffs[name].reshape(-1, 8, 8)
        zz = zigzag(scan)
        hist[name] = {"DC": _dc_stats(zz[:, 0]), "AC": _ac_stats(zz[:, 1:])}
    tables = {}
    for tname, comps in (("Y", ["Y"]), ("C", ["Cb", "Cr"])):
        for kind in ("DC", "AC"):
            tot = sum(hist[c][kind][0] for c in comps)
            tables[f"{tname}_{kind}"] = huffman_lengths({s: int(n) for s, n in enumerate(tot) if n})
    bits = {}
    for name in coeffs:
        t = "Y" if name == "Y" else "C"
        bits[name] = {kind: _table_cost(hist[name][kind][0], tables[f"{t}_{kind}"]) + hist[name][kind][1]
                      for kind in ("DC", "AC")}
    return JPEGData((h, w), subsampling, int(quality), qy, qc, coeffs, tables, bits,
                    _header_bytes(tables))


def decode(data: JPEGData) -> np.ndarray:
    """Decode :class:`JPEGData` to a float32 RGB image in [0, 1] (8-bit quantised).

    Dequantise -> inverse DCT -> +128 -> reassemble blocks -> bilinear chroma upsampling ->
    YCbCr to RGB -> clip.  Reference: L5 s20 (decoder path).
    """
    h, w = data.shape
    vy, vx = SUBSAMPLING[data.subsampling]
    planes = []
    for k, name in enumerate(("Y", "Cb", "Cr")):
        q = data.qy if k == 0 else data.qc
        plane = merge_blocks(idct2_blocks(data.coeffs[name] * q))
        if k:
            plane = upsample(plane, vy, vx)
        planes.append(plane[:h, :w])
    ycc = (np.stack(planes, axis=-1) + 128.0) / 255.0
    return utils.to_uint8(utils.ycbcr_to_rgb(ycc.astype(np.float32))).astype(np.float32) / 255.0


def roundtrip(img, quality: int = 75, subsampling: str = "4:2:0") -> tuple[np.ndarray, dict]:
    """Encode + decode.  Returns ``(decoded, info)``; ``info`` has ``nbytes``, ``bpp``,
    ``psnr`` (dB, vs the input), ``ratio`` (24 bpp / bpp), ``bits`` (breakdown) and ``data``."""
    data = encode(img, quality, subsampling)
    out = decode(data)
    info = dict(nbytes=data.nbytes, bpp=data.bpp, psnr=utils.psnr(img, out),
                ratio=24.0 / data.bpp, quality=int(quality), subsampling=subsampling,
                bits=data.bits, data=data)
    return out, info


def pillow_roundtrip(img, quality: int = 75, subsampling: str = "4:2:0",
                     optimize: bool = True) -> tuple[np.ndarray, int]:
    """Reference: compress with real libjpeg through Pillow, return ``(decoded, nbytes)``.

    ``optimize=True`` builds per-image Huffman tables, like :func:`encode` does.
    """
    import io
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(utils.to_uint8(img)).save(buf, format="JPEG", quality=int(quality),
                                              subsampling=_PIL_SUBSAMPLING[subsampling],
                                              optimize=optimize)
    nbytes = buf.tell()
    buf.seek(0)
    out = np.asarray(Image.open(buf).convert("RGB"), dtype=np.float32) / 255.0
    return out, nbytes


def generation_loss(img, n: int = 10, quality: int = 75, shift: int = 0,
                    subsampling: str = "4:2:0") -> list[dict]:
    """Repeat decode -> (shift by ``shift`` pixels) -> encode ``n`` times (L5 s19).

    The image is rolled by ``shift`` pixels in x and y between generations, so the 8x8 block
    grid lands somewhere else on the content.  PSNR is measured against the original rolled
    by the same total amount, so only compression error counts.
    Returns one dict per generation: ``gen``, ``psnr``, ``nbytes``, ``bpp``.
    """
    img = np.asarray(img, dtype=np.float32)
    cur, ref, out = img, img, []
    for g in range(1, n + 1):
        data = encode(cur, quality, subsampling)
        dec = decode(data)
        out.append(dict(gen=g, psnr=utils.psnr(dec, ref), nbytes=data.nbytes, bpp=data.bpp))
        if shift:
            dec = np.roll(dec, (shift, shift), axis=(0, 1))
            ref = np.roll(ref, (shift, shift), axis=(0, 1))
        cur = dec
    return out


def blockiness(img) -> float:
    """Blocking-artefact score: mean |jump| across 8-pixel block boundaries divided by the
    mean |jump| between pixels inside blocks (luma, both directions).  About 1 for an image
    without a visible grid; grows when quantisation makes neighbouring blocks disagree."""
    a = np.asarray(img, dtype=np.float64)
    g = a if a.ndim == 2 else utils.luminance(a).astype(np.float64)
    ratios = []
    for arr in (g, g.T):
        d = np.abs(np.diff(arr, axis=1))             # jump between column j and j+1
        on = (np.arange(d.shape[1]) % 8) == 7
        inside = d[:, ~on].mean()
        ratios.append(d[:, on].mean() / max(inside, 1e-9))
    return float(np.mean(ratios))


# ---------------------------------------------------------------------------
# storage: EXIF and RAW vs JPEG (L5 s21, s22)
# ---------------------------------------------------------------------------
_EXIF_IFD = 0x8769
_EXIF_NAMES = {"ExposureTime": 0x829A, "FNumber": 0x829D, "ISOSpeedRatings": 0x8827,
               "FocalLength": 0x920A}


def read_exif(path) -> dict:
    """Read the EXIF tags of a JPEG with Pillow: ``{tag name: value}`` (both IFDs)."""
    from PIL import Image, ExifTags
    with Image.open(path) as im:
        ex = im.getexif()
        tags = dict(ex)
        tags.update(ex.get_ifd(_EXIF_IFD))
    return {ExifTags.TAGS.get(k, k): v for k, v in tags.items()}


def write_jpeg_with_exif(img, path, quality: int = 90, subsampling: str = "4:2:0",
                         exif: dict | None = None) -> int:
    """Write a real JPEG file with a few EXIF tags (e.g. ``Make``, ``Model``, ``Software``,
    ``ExposureTime``, ``FNumber``, ``ISOSpeedRatings``).  Returns the file size in bytes."""
    import os
    from PIL import Image, ExifTags
    ids = {v: k for k, v in ExifTags.TAGS.items()}
    ex = Image.Exif()
    sub = ex.get_ifd(_EXIF_IFD)
    for name, val in (exif or {}).items():
        tag = _EXIF_NAMES.get(name) or ids[name]
        if name in _EXIF_NAMES:
            sub[tag] = val
        else:
            ex[tag] = val
    if sub:
        ex[_EXIF_IFD] = sub
    Image.fromarray(utils.to_uint8(img)).save(path, format="JPEG", quality=int(quality),
                                              subsampling=_PIL_SUBSAMPLING[subsampling], exif=ex)
    return os.path.getsize(path)


def raw_vs_jpeg_size(shape, bits: int, jpeg_nbytes: int) -> dict:
    """Uncompressed RAW size (one ``bits``-bit sample per pixel, not demosaiced) vs a JPEG.

    ``shape`` is ``(H, W[, 3])``; only H and W count.  Returns raw/jpeg bytes, the ratio
    and the bits per pixel of each.  Reference: L5 s22 (RAW files).
    """
    h, w = shape[:2]
    raw = h * w * bits / 8.0
    return dict(raw_bytes=raw, jpeg_bytes=int(jpeg_nbytes), ratio=raw / jpeg_nbytes,
                raw_bpp=float(bits), jpeg_bpp=8.0 * jpeg_nbytes / (h * w))
