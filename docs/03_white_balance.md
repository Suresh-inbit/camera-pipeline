# 03 - White balance

## 1. In one paragraph
A white sheet of paper reflects whatever light falls on it: yellow under a tungsten bulb, blue in shade. The sensor records that cast faithfully, so the image looks wrong even though the camera did nothing wrong. White balance (WB) estimates the colour of the light and divides it out with three per-channel gains. It must happen on linear data, early in the pipeline (before demosaicing is fine, before tone mapping is mandatory). The hard part is the estimate: you either know a neutral object, or you guess (L4 s9-11). `isp.whitebalance` provides chromaticity tools, black-body illuminants, three ways to get the gains and the angular error to judge them.

## 2. Theory
**Chromaticity (L4 s5-6).** Dividing a colour by its brightness leaves two numbers: `r = R/(R+G+B)`, `g = G/(R+G+B)`. Neutrals of any brightness sit at `(1/3, 1/3)`. The same construction with the CIE primaries XYZ gives `x = X/(X+Y+Z)`, `y = Y/(X+Y+Z)`; unlike `rg` it encloses the whole spectral locus, so it is the standard way to name a colour. We compute the locus from the analytic multi-lobe Gaussian fit of the CIE 1931 colour matching functions by Wyman, Sloan and Shirley (2013); no data tables are needed. Check: equal-energy white integrates to `(x, y) = (0.3331, 0.3335)`, and the locus peaks at `y = 0.824` near 523 nm (true value about 0.83 at 520 nm).

**Colour temperature (L4 s7-8).** A black body at temperature `T` emits `M(lambda, T) ~ lambda^-5 / (exp(c2 / (lambda T)) - 1)`. Integrating it against the colour matching functions gives an xy point; as `T` varies these points trace the Planckian locus. Real lamps are not black bodies, so their *correlated* colour temperature (CCT) is the nearest locus point (slide 7). Slide 7-8 presets: incandescent 3200 K, cool white fluorescent 4000 K, equal energy 5000 K, midday sun/flash 6000 K, overcast/TV (D65) 6500 K, northern blue sky 7500 K. In linear sRGB (`G = 1`) the black body is `(1.91, 1, 0.37)` at 3200 K, `(1.52, 1, 0.57)` at 4000 K, `(1.06, 1, 1.05)` at 6504 K and `(0.98, 1, 1.19)` at 7500 K.

**Diagonal (von Kries) model (L4 s9).** If the light has colour `e = (e_R, e_G, e_B)` and a surface has reflectance `rho_c`, the linear sensor value is `p_c = e_c rho_c`. A grey surface has `rho_R = rho_G = rho_B`, so grey becomes `~ e`. Gains `S_c = 1/e_c` undo it. Slide 9 recipe: pick an object you believe is neutral, set `S_R = (R+G+B)/(3R)` (likewise G, B), apply the same gains everywhere. The result has R = G = B on the patch; what that grey *looks like* then depends on the camera colour space, here sRGB with a D65 white (slide 9 stresses that WB on an sRGB camera forces the object to appear 6500 K, "blueish white").

**Why linear data.** The model is multiplicative in light. After the sRGB curve each channel is raised to about 1/2.2, so `p_c` is no longer proportional to `e_c` and a single gain can no longer cancel it. Measured: a mid-grey (0.18) card under 3200 K gets exact grey from gains on linear data; with the same gains applied after sRGB encoding it ends at `R:G:B = 0.73 : 1 : 1.67`. That is the technical reason to white balance before tone mapping, and one reason (L4 s11, "capture in RAW and white balance later") RAW is so useful: no gains have been baked in, so the decision stays open.

**Estimating the light (L4 s10-11).**
* *Manual patch*: needs a known neutral object (a grey card).
* *Gray world*: assume the mean colour of the scene is grey, `S_c = mean(R,G,B) / mean_c`. "Often inappropriate": a sunset, a lawn or a sandy desert are not grey on average.
* *White patch / brightest pixel*: assume the brightest pixels are white. Fails when the pixel is saturated (clipping hides the colour ratio) or when the bright object is coloured (gold highlights).

**Angular error.** Brightness is irrelevant to colour constancy, so compare *directions*: `acos(a.b / |a||b|)` in degrees. Compare the estimated light `1/gains` with the true light, not gains with gains.

## 3. Algorithm
1. Capture or load linear RGB (or a RAW mosaic).
2. Pick an estimator: `from_patch(img, box)`, `gray_world(img)` or `white_patch(img, percentile)`. Each returns gains `(S_R, S_G, S_B)`.
3. Normalise to `S_G = 1` (default): the green channel, which carries most of the luminance, keeps its value and only R and B change.
4. Multiply: `apply_gains(img, gains)` on an RGB image, or `apply_gains_mosaic(mosaic, gains, pattern)` on the RAW mosaic, each pixel scaled by the gain of its CFA site.
5. Continue with demosaicing / colour conversion / tone mapping. Values above 1 are legal until the final clip.

## 4. Implementation notes
* **Chromaticity.** `rgb_chromaticity(img)` (black maps to `(1/3, 1/3)`), `rgb_to_xy(rgb)` (via `utils.RGB2XYZ`), `srgb_gamut_xy()`, `cie_cmf(wavelengths)`, `spectral_locus_xy()`.
* **Locus range.** `spectral_locus_xy()` returns 420-650 nm by default. The Gaussian fit is accurate in between (e.g. 520 nm gives (0.081, 0.820), true (0.074, 0.834)), but its far tails go wrong: at 700 nm it gives `x = 0.57` instead of 0.735, which bends the curve back. Ask for a different `lo`, `hi` if you want to see it.
* **Colour temperature.** `cct_to_xy(T)` and `planckian_locus_xy(Ts)` integrate Planck's law numerically (360-830 nm, 1 nm steps) against the fit: they do not use the Kim or Krystek polynomials. At 6504 K this gives `(0.3134, 0.3237)` versus D65 `(0.3127, 0.3290)`: a black body is slightly less green than the daylight illuminant, so the sRGB white is not exactly black-body white. At 3200 K it gives `(0.4225, 0.3998)` (published 0.4232, 0.3990).
* `xy_to_cct` is the numerical inverse: nearest point of a 4000-entry locus table (1000-40000 K, geometric spacing) in the CIE 1960 uv plane. Round trip over 2500-10000 K: max relative error 0.05 %. `mccamy_cct` is the closed form from McCamy 1992 (5013 K for the 5000 K black body).
* `illuminant_rgb(T)` = `XYZ2RGB @ XYZ(T)`, G = 1, unclipped (blue goes negative below ~1700 K, outside the sRGB gamut).
* **`apply_illuminant(img, T_or_rgb)`**: `out = img * e / max(e)`, with `e = illuminant_rgb(T) / illuminant_rgb(6504)`, so 6504 K is the identity (the scene is "lit by D65-neutral light") and the result stays in [0, 1]. The strongest channel is untouched and the others shrink; G scales by `1/max(e)`: 0.55 at 3200 K, 0.70 at 4000 K. Pass an RGB triple to use your own light.
* `gains_for_illuminant` returns `1/e` with G = 1, so `apply_gains(apply_illuminant(x, T), gains_for_illuminant(T)) = x / max(e)`: the scene up to that one global scale.
* `gains_to_illuminant(gains)` gives `1/gains` with G = 1: feed it to `angular_error`.
* `from_patch` follows slide 9 literally (`S_c = (R+G+B)/(3 c)`, mean over the box) and then divides by `S_G` (`normalize_green=False` gives the raw slide values). Hand example: patch `(0.6, 0.3, 0.15)` gives `S = (0.583, 1.167, 2.333)`, which maps it to `(0.35, 0.35, 0.35)`; normalised to G = 1 it is `(0.5, 1, 2)`.
* `white_patch(img, percentile=99)` averages the pixels whose R+G+B is above the percentile (not a per-channel max, which one noisy channel can ruin). Saturation is not detected; that is the failure mode on slide 11.
* `apply_gains` does not clip unless `clip=True`. `apply_gains_mosaic` accepts `'RGGB'`, `'BGGR'`, `'GRBG'`, `'GBRG'`; it tiles a 2x2 gain pattern over the mosaic, so it is exactly equivalent to indexing the gain-corrected RGB image.
* Cost: everything is O(pixels). The tests (23) run in 0.5 s.
* Pitfall: gains estimated on a clipped image are wrong, and an exposure that clips R under a warm light cannot be repaired by any gain.

## 5. Experiments
Run `python scripts/make_figures_whitebalance.py`; all numbers below are printed by it.

* `docs/figures/whitebalance_illuminants.png`: the astronaut photo and the colour chart under 3200, 4000, 6504 and 7500 K. 6504 K is the unchanged scene. Look at the grey row of the chart: warm, warm, grey, cool.
* `docs/figures/whitebalance_xy_diagram.png`: CIE xy diagram with the spectral locus and the line of purples, the sRGB triangle (primaries `(0.64, 0.33)`, `(0.30, 0.60)`, `(0.15, 0.06)`), the Planckian locus with ticks from 2000 to 10000 K and the D65 point. Markers show the experiment on the astronaut photo with a grey card under 3200 K: true light (ring), manual patch (square, on top of the ring), white patch (plus, practically on the ring) and gray world (cross, clearly off to the right and down). The xy of the true light corresponds to CCT 3296 K (the ring is slightly off the locus, since it is relative to D65), and gray world's estimate to 2460 K: it over-corrects towards blue because the orange spacesuit pulls the average to red.
* `docs/figures/whitebalance_rgb_chromaticity.png`: rg scatter of the same photo: ground truth, under 3200 K and after manual WB. The cloud drifts from mean `(r, g) = (0.501, 0.259)` to `(0.677, 0.225)` (towards the red corner) and returns exactly to `(0.501, 0.259)` after correction. Note the cloud is *not* centred on `(1/3, 1/3)` even for the true scene: it is the grey patch that should land there, not the mean.
* `docs/figures/whitebalance_methods.png`: manual patch, gray world and white patch on the same cast image (card boxed in yellow). Angular errors: no correction 29.4 deg; manual patch 0.00 deg; white patch 0.43 deg (the white helmet and flag give a nearly neutral highlight); gray world 10.2 deg (the image is mostly orange and skin tones, gray world removes part of that colour with a blue cast on the wall). For display only, the corrected images are scaled to the mean luminance of the ground truth.
* `docs/figures/whitebalance_grayworld_failure.png`: the `scenes.sunset` image, whose true light is neutral. Its mean RGB is `(1, 0.46, 0.56)` relative to R; gray world applies gains `(0.46, 1, 0.82)` and creates a strong green-blue cast: 19.4 deg error, against 8.3 deg on the colour chart (the chart is not grey on average either, but much closer). This is the "often inappropriate" case of L4 s10.
* `docs/figures/whitebalance_error_table.png`: angular error of the estimated illuminant (deg), by scene (each with a neutral card, or chart patch 20) and light.

| scene | light | manual | gray world | white patch |
|---|---|---|---|---|
| chart | 3200 K | 0.00 | 5.57 | 0.09 |
| chart | 7500 K | 0.00 | 8.23 | 0.26 |
| astronaut | 3200 K | 0.00 | 10.15 | 0.43 |
| astronaut | 7500 K | 0.00 | 14.10 | 0.51 |
| coffee | 3200 K | 0.00 | 18.81 | 0.17 |
| coffee | 7500 K | 0.00 | 34.61 | 0.33 |
| sunset | 3200 K | 0.00 | 14.68 | 19.61 |
| sunset | 7500 K | 0.00 | 18.29 | 42.93 |

(4000 K rows are in the figure.) Mean over all 12 cases: manual 0.00, gray world 15.4, white patch 7.6 deg. Manual is exact because the scene is synthetic with a perfect grey, and a real card would show some error. White patch is excellent whenever a white object is in the frame and fails on the sunset, where the brightest pixels are the saturated orange horizon: 42.9 deg at 7500 K. Gray world is mediocre everywhere and worst on scenes dominated by one colour (the coffee photo is brown and orange).
* `tests/test_whitebalance.py` checks: CMF fit (white at 1/3, locus peak 0.82-0.84, ends), D65 xy within 0.01, CCT round trip within 2 % over 2500-10000 K, reddish 3200 K / bluish 9000 K light, relight-then-correct is the scene up to the documented scale, slide 9 formula on a hand example, < 0.5 deg and R = G = B for a neutral chart patch, gray world < 1 deg on a grey-mean image but > 15 deg on the sunset, white patch correct on the chart and wrong on a saturated coloured area, mosaic gains equal to mosaicing the corrected RGB for all four CFA patterns.

## 6. Try it
```python
import numpy as np
from isp import scenes, whitebalance as W

img = scenes.load("astronaut", 256)
lit = W.apply_illuminant(img, W.PRESETS["incandescent"])      # warm cast, 3200 K
true_light = W._illum_relative(3200)

for name, gains in [("gray world", W.gray_world(lit)),
                    ("white patch", W.white_patch(lit, 99)),
                    ("ideal", W.gains_for_illuminant(3200))]:
    err = W.angular_error(W.gains_to_illuminant(gains), true_light)
    print(name, np.round(gains, 3), f"{err:.2f} deg")

balanced = W.apply_gains(lit, W.gray_world(lit), clip=True)
print(W.xy_to_cct(*W.cct_to_xy(5000)))                         # ~5000
```

## 7. Limitations and extensions
* **Single global light.** One gain triple cannot fix a scene lit by a window and a lamp; local or per-region WB does that.
* **Diagonal model.** Real sensors have overlapping spectral responses, so per-channel gains in camera RGB are only an approximation; a full 3x3 matrix (or chromatic adaptation in a cone space such as Bradford) is more accurate. We use gains because lecture 4 does.
* **Colour constancy assumptions.** Gray world, white patch and their relatives (shades-of-grey, gray-edge) trade one assumption for another. Try `mean(|x|^p)^(1/p)` with `p = 6`: it interpolates between gray world (`p = 1`) and white patch (`p = infinity`).
* **Black body only.** `illuminant_rgb` models Planckian light; daylight (CIE D series) and fluorescent lamps deviate from the locus, mainly in green-magenta.
* **Clipping.** White balance after saturation cannot recover lost colour; expose to protect highlights, and capture RAW (L4 s11).
* **Gamut.** Very low temperatures push blue below zero in sRGB.
* **Extension.** Learn the illuminant (e.g. a small network on rg chromaticity histograms), or estimate it from specular highlights.

## 8. References
* Levoy, M. *Post-processing: white balancing and tone mapping* (EE 5176 lecture 4, slides 5-11), after Stanford CS 178 (2010).
* Wyman, C., Sloan, P.-P. and Shirley, P. (2013). Simple analytic approximations to the CIE XYZ colour matching functions. *Journal of Computer Graphics Techniques* 2(2), 1-11.
* McCamy, C. S. (1992). Correlated color temperature as an explicit function of chromaticity coordinates. *Color Research & Application* 17(2), 142-144.
* von Kries, J. (1902). Chromatic adaptation. Festschrift der Albrecht-Ludwigs-Universitat.
* Buchsbaum, G. (1980). A spatial processor model for object colour perception. *Journal of the Franklin Institute* 310(1), 1-26 (gray world).
* Gijsenij, A., Gevers, T. and van de Weijer, J. (2011). Computational color constancy: survey and experiments. *IEEE Trans. Image Processing* 20(9), 2475-2489 (angular error).
