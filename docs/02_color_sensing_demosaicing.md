# 02 - Colour sensing and demosaicking

## 1. In one paragraph
Silicon counts photons but cannot tell their colour, so every pixel must be filtered before the light reaches it (L3 s3). Most cameras put one of R, G, B in front of each photosite (a colour filter array, CFA, L3 s13) and later *demosaic*: reconstruct the two missing colours at every pixel from the neighbours. Two thirds of the numbers in the final RGB image are therefore interpolated, and interpolation of fine detail produces colour fringes and moire. This chapter implements the CFA (`isp.cfa`), five demosaicking methods, a chroma-error metric (`isp.demosaic`) and small simulators of the four colour-sensing technologies of L3 s4-12.

## 2. Theory
**Four ways to get colour (L3 s4-12, summary s29).**

| technology | what is recorded | pro | con |
|---|---|---|---|
| field sequential (Prokudin-Gorsky, s6-7) | three exposures at t, t+1, t+2 through R, G, B filters | full resolution, no light loss | trades off time: motion misregisters the planes |
| 3 sensor (s9) | prism splits the beam onto three sensors | full resolution, no light loss | expensive, complicated optics |
| vertically stacked (Foveon, s11) | three layers; long wavelengths penetrate deeper | full resolution | broad overlapping responses, so noisier |
| CFA (s13) | one filter per site, interpolate | simple hardware | interpolation artefacts, each site keeps about 1/3 of the light |

**Mosaic.** With masks `M_c(x)` (1 where colour `c` is sampled), the sensor stores `m(x) = sum_c M_c(x) I_c(x)`. For Bayer, G is sampled on a quincunx lattice (50 %) and R, B on lattices of spacing 2 (25 % each): more green because the eye's luminous efficiency peaks in the middle of the spectrum (s13). Each colour plane is therefore undersampled by 2x (R, B) or sqrt(2)x (G) and **aliases** when the scene has detail near the Nyquist limit.

**Linear interpolation (s17).** Average of the nearest same-colour neighbours (4 or 2); bicubic uses 16 (s17). Writing it as a normalised convolution,
`I_c = (K_c * (M_c m)) / (K_c * M_c)`,
the denominator re-weights whichever samples exist. At a site that measured colour `c`, the kernel centre weight is 1 and every other tap lands on a missing sample, so the measurement passes through unchanged.

**Why colour moire (s18).** Fine black-and-white detail changes the R, G and B samples out of step; the interpolator reads the difference as colour. Colour artefacts are places where *chrominance changes abruptly but only transiently* (s19).

**Luminance carries sharpness (s20-25).** Blurring Cb or Cr hardly changes how sharp an image looks; blurring Y ruins it. So chroma can be filtered aggressively.

**Freeman's median method (s26-27).** Replace the chroma at each pixel by the median over a neighbourhood. A median removes spikes and leaves monotonic edges unchanged (s26), so it deletes transient fringes but keeps real colour boundaries. Algorithm on s27: (1) naive interpolation, (2) convert to YCbCr, (3) median-filter Cb and Cr, (4) reconstruct R, G, B from the sensor value and the filtered chroma.

**Malvar-He-Cutler (2004, extension).** Bilinear plus a correction from the other colours' Laplacian, e.g. `G(x at R) = G_bilinear + 1/2 (R(x) - R_avg)`. This yields fixed 5x5 linear filters that follow edges better than bilinear at the same cost class.

## 3. Algorithm
Bilinear / bicubic:
1. Build `M_c`; choose the kernel per colour (3x3 bilinear, 7x7 cubic; G uses a diamond kernel on the 45-degree lattice).
2. `num = K * (m M_c)`, `den = K * M_c` (zero padded), `I_c = num / den`.

Median (Freeman):
1. `E = bilinear(m)`.
2. `Y, Cb, Cr = ycbcr(E)`.
3. `Cb', Cr' = median_filter(Cb, Cr)` with a `median_size` window.
4. For each site that measured channel `c` with value `s`: `o = rgb(0, Cb', Cr')`, `Y' = s - o_c`, `rgb = Y' + o`. The measured channel is reproduced exactly.

Malvar: filter the mosaic with one of four fixed 5x5 kernels depending on the site type (G at R/B, R/B at G in the same row, R/B at G in the same column, R at B and B at R); keep measured samples; clip to [0, 1].

## 4. Implementation notes
* `isp.cfa`: `PATTERNS`, `masks(shape, pattern)`, `mosaic(img, pattern)`, `colorize(mosaic, pattern)`, plus `channel_index`, and display-only tiles `SONY_RGBE_TILE`, `KODAK_RGBW_TILE`, `BAYER_TILE` with `tile_colors`.
* `isp.demosaic`: `demosaic(mosaic, pattern, method, **kw)`, `METHODS = (nearest, bilinear, bicubic, median, malvar)`. `median` takes `median_size` (3), `iterations` (1) and `base` (the naive interpolator, bilinear). `malvar` takes `clip`.
* `false_color(img, reference)` = `sqrt(mean((dCb)^2 + (dCr)^2))` in JPEG YCbCr, range 0..1. Pure luminance blur costs nothing, colour fringes cost a lot. It needs a ground truth, so it is a lab metric, not a camera one.
* `blur_channel(img, 'Y' | 'Cb' | 'Cr' | 'CbCr', sigma)` is the s20-25 demonstration.
* Sensing simulators: `field_sequential(scene, motion=(dy, dx), frames=None, noise)`, `three_sensor`, `stacked_sensor(scene, mix=STACK_MIX, noise)`, `cfa_sensor(scene, pattern, method, noise)`, and `sense(scene, technology)`. Noise is additive Gaussian per recorded number (std `noise`, linear units).
* Borders: normalised convolution zero-pads and renormalises; Malvar uses reflect-101 (keeps Bayer parity). Compare methods in the interior if you care about borders.
* Pitfall: pattern naming. OpenCV names Bayer codes by the 2x2 tile read from the second row, so our `'RGGB'` is `cv2.COLOR_BayerBG2RGB` (`BGGR` -> `RG`, `GRBG` -> `GB`, `GBRG` -> `GR`). The test checks this empirically: `bilinear` agrees with OpenCV's bilinear on 16-bit data to 1.5e-5 away from the border.
* Pitfall: demosaic *linear* data. Interpolating gamma-encoded values changes the luminance of fringes.
* Cost: all methods are a few `filter2D` calls; `median` adds two 3x3 median filters. All methods run in well under a second at 256x256.
* Stacked mixing matrix `STACK_MIX` is illustrative, not a measured Foveon response: condition number 8.2; noise gain per channel 3.5, 6.3, 4.7.

## 5. Experiments
All numbers from `python scripts/make_figures_demosaic.py` (RGGB, size 256, no noise unless stated).

**`demosaic_cfa_patterns.png`.** Bayer, Sony RGB+E and Kodak RGB+W tiles (s13). Bayer: 25 % R, 50 % G, 25 % B. RGB+E adds a fourth colour for better colour; RGB+W has 50 % panchromatic sites that collect more light (less noise) at the price of harder colour reconstruction. Display only.

**`demosaic_mosaic_view.png`.** The "before demosaicking" view (s16): left, scene; then a 32x32 crop of the grey mosaic (checker texture wherever the scene is coloured, because R, G, B samples have different brightness there), the same samples in their filter colours, and the bilinear result. On the orange suit the grey checkerboard is the colour information.

**`demosaic_zoneplate.png`.** A grey zone plate reaches Nyquist at the image edge, so every colour in the output is false. False colour (RMS chroma error): nearest 0.320, bilinear 0.245, bicubic 0.265, Freeman median 0.140, Malvar 0.176. Look at the left/right/top/bottom edges: the saturated rings of nearest and bilinear are replaced by weaker, finer fringes with median and Malvar. Bicubic is not better than bilinear at colour here (0.265 vs 0.245): a wider kernel does not remove aliasing that is already in the samples.

**`demosaic_crops.png`.** The 48x48 window of the astronaut where bilinear makes the largest chroma error (rows 176-224, cols 80-128). PSNR / false colour in that window: nearest 18.2 dB / 0.092, bilinear 23.2 / 0.053, bicubic 23.4 / 0.053, median 26.9 / 0.037, Malvar 26.6 / 0.039. Nearest shows blocky 2x2 steps and strong fringes; bilinear and bicubic are smooth but fringed; median and Malvar are visibly cleaner. The faint regular grid of darker red dots they leave on the saturated orange suit is expected, not a bug: both methods assume the colour channels change together (Malvar adds a fraction of the R Laplacian to G and B; Freeman keeps the measured sample and a smoothed chroma), but on a saturated colour R varies strongly while the dim G and B channels hardly do, so their estimates at the R and B sites are off by a fixed amount and the error repeats with the 2x2 period, and the linear data make it visible (on a synthetic saturated orange ramp the mean relative error of the dim B channel is 0.04 for bilinear but 0.36 for median and 0.51 for Malvar, while on a desaturated version they beat bilinear). We checked that smooth ramps are reproduced exactly (to 1e-5) by bilinear, bicubic, median and Malvar for all four CFA patterns, and that the errors have no bias and are identical across patterns.

**`demosaic_linear_vs_median.png`** (the layout of s28). Whole "coffee" image, then a zoom on the spoon, where chrome highlights are fringed: false colour 0.0235 (linear) vs 0.0170 (median); in the zoomed window 0.0457 vs 0.0294. The coloured speckles around the highlights shrink, but the median image shows a faint pixel-level grain: luminance is recomputed per site from the measured sample against smooth chroma, so mosaic residue moves into Y. That is the price of keeping measured samples exactly.

**`demosaic_table.png`.** PSNR (dB, linear data) / false colour over the four photos:

| method | astronaut | coffee | chelsea | rocket | mean PSNR | mean fc |
|---|---|---|---|---|---|---|
| nearest | 23.21 / 0.0513 | 25.19 / 0.0418 | 31.01 / 0.0217 | 28.29 / 0.0308 | 26.93 | 0.0364 |
| bilinear | 28.91 / 0.0274 | 30.27 / 0.0235 | 35.88 / 0.0125 | 32.13 / 0.0201 | 31.80 | 0.0209 |
| bicubic | 29.48 / 0.0265 | 30.51 / 0.0235 | 36.08 / 0.0126 | 31.96 / 0.0209 | 32.01 | 0.0209 |
| median | 33.20 / 0.0176 | 33.41 / 0.0170 | 38.91 / 0.0094 | 33.40 / 0.0176 | 34.73 | 0.0154 |
| malvar | 33.15 / 0.0183 | 33.17 / 0.0183 | 38.90 / 0.0095 | 33.41 / 0.0179 | 34.66 | 0.0160 |

Median and Malvar are about 3 dB better than bilinear and cut chroma error by about 25 %. Bicubic helps PSNR slightly on average (+0.2 dB) and not at all on chroma. Median and Malvar are within 0.1 dB of each other; the median filter wins slightly on this data set, but Malvar does not need a colour-space round trip and has no free parameter.

**`demosaic_luma_chroma.png`.** "coffee" (sRGB-encoded, sigma 4 px): blur Cb: SSIM 0.970, PSNR 35.1 dB; blur Cr: 0.970, 33.9 dB; blur Y: SSIM 0.630, PSNR 22.3 dB. The two chroma-blurred images look almost like the original; the Y-blurred one is obviously soft (s20-25). Blurring both chroma planes still gives SSIM 0.944. This is why chroma filtering, as in Freeman and in JPEG 4:2:0 (L5), is cheap.

**`demosaic_sensing.png`.** One scene, four technologies (s4-12). Field sequential with a 4 px/exposure motion: PSNR 15.6 dB, false colour 0.125, with clear R/G/B ghosts at every edge. 3-sensor with noise std 0.02: 34.0 dB (error 0.020, equal to the noise). Stacked with the same noise per layer: 20.1 dB, error 0.099 (about 5x the noise, as `inv(mix)` predicts), with strong chroma noise (the zoom). CFA + bilinear with the same noise: 28.2 dB; its error mixes interpolation error and noise; the interpolation averages neighbours, which also smooths some noise. Noise-free, three-sensor, stacked and static field-sequential are all exact (error 0); the CFA is the only one that cannot be, even without noise.

**Take-home (s28): 2/3 of your data is made up.** Of 3N numbers in the RGB image, only N were measured. The remainder is an estimate, and there are better and worse ways to make it: on these images the difference between nearest and the best method is 7.8 dB.

## 6. Try it
```python
import numpy as np
from isp import scenes, cfa, demosaic as D, utils

ref = scenes.photo('astronaut', 256)
raw = cfa.mosaic(ref, 'RGGB')
for method in D.METHODS:
    out = D.demosaic(raw, 'RGGB', method)
    print(method, round(utils.psnr(out, ref), 2), round(D.false_color(out, ref), 4))

med5 = D.demosaic(raw, 'RGGB', 'median', median_size=5)        # stronger chroma filter
smear = D.sense(ref, 'field_sequential', motion=(0, 4))         # colour break-up
```
`python scripts/make_figures_demosaic.py` regenerates all figures and prints the numbers; `python -m pytest tests/test_demosaic.py -q` runs the tests.

## 7. Limitations and extensions
* The scenes have no optical low-pass filter, so they alias harshly. Real cameras blur slightly with an OLPF, which trades sharpness for fewer fringes; try `blur_channel` or a Gaussian on the scene before `mosaic`.
* The median method is chroma-only: it cannot restore resolution lost in luminance, and it smears small coloured details (a one-pixel red line is a chroma outlier and may be erased).
* `median_size` and `iterations` trade fringes against colour bleeding; try 5 and 2 on the zone plate.
* No noise model in the demosaicker. Interpolation and denoising interact (L5); in a real pipeline demosaic of noisy RAW spreads noise into neighbouring pixels.
* Extensions: edge-directed methods (Hamilton-Adams, AHD), frequency-domain demosaicking (Alleysson et al.), joint demosaic-denoise, non-Bayer patterns (RGBE, RGBW, X-Trans) using the tiles in `isp.cfa`.
* The stacked-sensor model has a single global mixing matrix; real layers vary with wavelength and depth.

## 8. References
* Lecture 3, *Sensing Color* (M. Levoy, Stanford CS 178 slides, 2010), slides 3-29.
* W. T. Freeman, "Median filter for reconstructing missing color samples", US Patent 4,724,395, 1988.
* H. S. Malvar, L.-W. He, R. Cutler, "High-quality linear interpolation for demosaicing of Bayer-patterned color images", ICASSP 2004.
* R. G. Keys, "Cubic convolution interpolation for digital image processing", IEEE Trans. ASSP 29(6), 1981.
* B. E. Bayer, "Color imaging array", US Patent 3,971,065, 1976.
* R. Lukac, K. N. Plataniotis, "Color filter arrays: design and performance analysis", IEEE Trans. Consumer Electronics 51(4), 2005.
