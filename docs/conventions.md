# Conventions (read this first)

This file defines the data conventions, the public API and the documentation template
used by every module in `isp/`. It is written for a research student who wants to
extend the code, and it is the contract the implementation followed.

Slide references use the form *L4 s10* = lecture 4, PDF page 10 of the decks
(L2 Sensors and Noise, L3 Sensing Color, L4 White Balancing and Tone Mapping,
L5 Denoising, Sharpening, Compression).

## 1. Data convention

* Images are NumPy `float32`. RGB images are `(H, W, 3)`; a RAW mosaic is `(H, W)`.
  `H` and `W` are even (the scenes in `isp.scenes` are multiples of 16).
* **Linear until tone mapping.** Scene values are proportional to light, in `[0, 1]`
  (1.0 = scene white). White balance, demosaicing and denoising operate on linear data.
  The tone-mapping stage produces display-referred, sRGB-encoded values in `[0, 1]`;
  sharpening and JPEG operate on those.
* Every image travels with a `meta` dict. Stages copy it, never mutate their input.

| `meta['domain']` | Meaning | Array |
|---|---|---|
| `scene` | linear scene radiance, ground truth | `(H, W, 3)` in [0, 1] |
| `electrons` | charge collected per pixel | `(H, W)` mosaic, float, 0..full well |
| `dn` | digital numbers out of the ADC (RAW) | `(H, W)` mosaic, float holding integers 0..2^bits-1 |
| `linear` | black-subtracted, normalised linear data | `(H, W)` mosaic or `(H, W, 3)` in [0, 1] |
| `display` | sRGB-encoded, ready to show | `(H, W, 3)` in [0, 1] |
| `bytes` | decoded JPEG (the stored file) | `(H, W, 3)` in [0, 1], plus `meta['nbytes']` |

Other meta keys: `cfa` (`'RGGB'` etc., present while the image is a mosaic), `bits`,
`black_level`, `white_level` (DN), `iso`, `exposure` (s), `illuminant` (CCT in K or RGB
triple), `wb_gains`, `nbytes`, `bpp`.

* Bayer pattern strings read row-major over the 2x2 tile: `'RGGB'` means
  `[[R, G], [G, B]]` with R at `(0, 0)`. Default everywhere: `'RGGB'`.
* Colour: linear sRGB primaries, D65 white. YCbCr is JPEG full-range BT.601
  (`isp.utils.rgb_to_ycbcr`), Cb and Cr centred on 0.5.
* Randomness: every stochastic function takes `seed=0` and uses `isp.utils.rng(seed)`.

## 2. Code conventions

* NumPy + SciPy (+ OpenCV / scikit-image where handy). No new dependencies.
* Pure functions, `float32` in and out, **no plotting inside `isp/`** (plotting lives in
  `scripts/make_figures_*.py`, the notebooks and the app).
* Every public function has a docstring: one-line summary, the maths in plain text,
  parameters, returns, and a `Reference:` line (slide and/or paper).
* Use `isp.utils` for sRGB curves, YCbCr, PSNR / SSIM, Gaussian kernels; do not
  re-implement them.
* Tests: `tests/test_<module>.py`, run with `python -m pytest tests -q` from the project
  root. Use small images (`size=64..128`) so each file runs in well under a minute.
  Deterministic (`seed=0`).
* Figures: `scripts/make_figures_<module>.py` writes PNGs named `<module>_<what>.png` to
  `docs/figures/`, runs headless (`matplotlib.use('Agg')`), `dpi=90`, and prints the
  numbers quoted in the docs.
* Docs cite slide numbers but never embed slide images (third-party copyright).

## 3. Documentation template (`docs/NN_<topic>.md`)

1. **In one paragraph**: what the method does and why it matters.
2. **Theory**: the model and the derivation, with equations; connect to the slides.
3. **Algorithm**: numbered steps as implemented.
4. **Implementation notes**: which functions, design choices, complexity, pitfalls.
5. **Experiments**: figures (from `docs/figures/`), what to look for, measured numbers.
6. **Try it**: a short runnable snippet.
7. **Limitations and extensions**: what breaks, what a student could try next.
8. **References**: full citations.

Write like a good tutorial: short paragraphs, equations only where they carry
information, every figure explained, every number measured by the figure script.

## 4. Public API (names and meanings other modules rely on)

Modules may add more; the names below must exist with these signatures.
`img` = `(H, W, 3)` float32, `mosaic` = `(H, W)` float32.

```
# isp.sensor  (L2)
SensorParams(qe, full_well, dark_current, read_noise, bits, black_level, base_gain,
             prnu, dsnu, hot_pixel_fraction, hot_pixel_rate, name)        dataclass
    # electrons, e-/s, e- rms, ADC bits, DN, DN per e- at ISO 100 (default: full well
    # maps to white level), gain FPN fraction, dark-current FPN (e-/s std), fraction of
    # hot pixels, hot-pixel dark current e-/s
PRESETS: dict  with at least 'retiga_4000r', 'aptina_mt9p031', 'dslr', 'phone'
expose(image, exposure, params, flux_at_white=..., seed=0) -> electrons   # photons*QE*t
        + dark current + hot pixels, Poisson noise, PRNU, clipped at full well;
        works on a mosaic (H, W) or an RGB image
adc(electrons, params, iso=100, seed=0) -> dn      # + read noise, gain, black level, quantise, clip
capture(image, exposure, iso, params, flux_at_white=..., seed=0) -> (dn, meta)
dark_frame(shape, exposure, iso, params, seed=1) -> dn
subtract_dark(dn, dark) -> dn
normalize(dn, meta) -> linear in [0, 1]            # (dn - black) / (white - black)
fix_hot_pixels(mosaic, pattern='RGGB', threshold=...) -> mosaic   # median of same-colour neighbours
average_frames(frames) -> mean frame
snr_theory(P, qe, t, D, Nr) -> float               # P qe t / sqrt(P qe t + D t + Nr^2)   (L2 s31)
dynamic_range(full_well, D, t, Nr) -> float        # (full_well - D t) / sqrt(D t + Nr^2) (L2 s33)

# isp.cfa  (L3)
PATTERNS = ('RGGB', 'BGGR', 'GRBG', 'GBRG')
masks(shape, pattern='RGGB') -> (H, W, 3) bool     # where each colour is sampled
mosaic(img, pattern='RGGB') -> mosaic
colorize(mosaic, pattern='RGGB') -> img            # each sample shown in its filter colour

# isp.demosaic  (L3)
METHODS = ('nearest', 'bilinear', 'bicubic', 'median', 'malvar')
demosaic(mosaic, pattern='RGGB', method='bilinear', **kw) -> img
    # 'median' = Freeman: bilinear, then median-filter the chroma (L3 s27)
false_color(img, reference) -> float               # chroma error energy (zipper / moire measure)

# isp.whitebalance  (L4)
PRESETS: dict name -> CCT in kelvin  (incandescent 3200, fluorescent 4000, ... L4 s7-8)
cct_to_xy(T) -> (x, y);  xy_to_cct(x, y) -> T
illuminant_rgb(T) -> (3,)                          # linear sRGB colour of a black body, G = 1
apply_illuminant(img, T_or_rgb) -> img             # scene as lit by that source (colour cast)
gray_world(img) -> gains (3,);  white_patch(img, percentile=99) -> gains
from_patch(img, box) -> gains                      # S_R = (R+G+B)/(3R) etc.  (L4 s9)
gains_for_illuminant(T_or_rgb) -> gains            # the ideal correction, G = 1
apply_gains(img, gains) -> img;  apply_gains_mosaic(mosaic, gains, pattern='RGGB') -> mosaic
angular_error(a, b) -> degrees
rgb_chromaticity(img) -> (..., 2);  spectral_locus_xy() -> (N, 2);  planckian_locus_xy(Ts) -> (N, 2)

# isp.tonemap  (L4)
gamma(img, g) -> img                               # out = in ** g   (L4 s14)
hist_equalize(img, bins=256) -> img                # global, on luminance, colour preserved
clahe(img, clip=2.0, tiles=8) -> img               # extension
dodge_burn(img, sigma=..., strength=...) -> img    # local operator (L4 s16-18)
tonemap(img, method='srgb'|'gamma'|'hist'|'clahe'|'local', **kw) -> display img

# isp.denoise  (L5)
gaussian(img, sigma) -> img
bilateral(img, sigma_d, sigma_r, radius=None) -> img        # own implementation of L5 s9
median(img, size=3) -> img
luma_chroma(img, luma_kw, chroma_kw) -> img                 # bilateral in YCbCr, separate strengths
nlm(img, h) -> img                                          # extension
denoise(img, method='bilateral', **kw) -> img

# isp.sharpen  (L5)
LAPLACIAN = [[0,-1,0],[-1,4,-1],[0,-1,0]]
high_pass(img, kind='laplacian'|'gaussian', sigma=1.0) -> img
unsharp_mask(img, lam=1.0, kind='laplacian', sigma=1.0) -> img   # S = x + lam F(x)  (L5 s15)
sharpen(img, method='laplacian'|'unsharp', lam=1.0, **kw) -> img
overshoot(profile) -> float                                  # halo size on an edge profile

# isp.jpeg  (L5)
quant_tables(quality) -> (QY, QC)                            # standard tables, IJG quality scaling
dct_basis() -> (8, 8, 8, 8)
encode(img, quality=75, subsampling='4:2:0') -> JPEGData     # coefficients + size estimate
decode(data) -> img
roundtrip(img, quality=75, subsampling='4:2:0') -> (img, info)   # info: nbytes, bpp, psnr
pillow_roundtrip(img, quality=75, subsampling='4:2:0') -> (img, nbytes)   # real libjpeg reference
generation_loss(img, n=10, quality=75, shift=0) -> list of dict
```

## 5. The pipeline

`isp.pipeline.Stage(name, fn, params, enabled, group, label)` wraps any
`fn(image, meta, **params) -> (image, meta)`. `Pipeline(stages).run(image, meta)` returns
a `PipelineResult` with one `StageRecord` per stage (output image, meta, parameters,
time, metrics). `group` is one of `scene | sensor | adc | processing | storage`, the
boxes of the camera pixel pipeline diagram (L2 s3).
