# 04 - Tone mapping (contrast correction)

## 1. In one paragraph
After white balance and demosaicing the image is still *linear*: pixel values are proportional to scene light, and a scene can span 10-15 stops while a display shows about 8 bits of an sRGB curve. Tone mapping (lecture 4, "contrast correction", s13-18) chooses the curve from linear scene values to displayed values. **Global** operators apply one curve to every pixel (gamma, sRGB, histogram equalisation); **local** operators let the curve depend on the neighbourhood, which is what Ansel Adams did with dodging and burning in the darkroom (s16-18). The stage matters because it decides how much shadow and highlight detail survives and it is where the data stops being linear. After it, sharpening and JPEG take over.

## 2. Theory
**Why a curve at all (s13).** The histogram of a low-contrast picture occupies a narrow band; a high-contrast one fills the range. Manual editing "fiddles with the histogram": move the black point to the darkest useful value, the white point to the brightest, bend the midtones. That is `levels`:
`out = clip((in - black) / (white - black), 0, 1) ** g`.

**Gamma (s14).** `out = in ** g` for `0 <= in <= 1`. `g < 1` lifts midtones, `g > 1` darkens them, 0 and 1 stay fixed. The slide calls it "simple but crude": one exponent for the whole image cannot lift a shadow without also flattening the highlights. The sRGB encoding curve is a gamma of about 1/2.2 with a linear toe (`12.92 x` below 0.0031308); `isp.utils.linear_to_srgb` is the camera default (`tonemap(..., 'srgb')`).

**Histogram equalisation.** Map each value through its own cumulative distribution, `out = CDF(in)`. If the values are continuous the output is uniformly distributed, so every grey level is used equally. On RGB, doing this per channel changes the ratios R:G:B and so the hue and saturation. We equalise the luminance `Y` and rescale the pixel, `rgb_out = rgb * Y_out / Y_in`, which keeps the colour ratios (if a channel would then exceed 1, the whole pixel is scaled down so hue is still kept).

**Global versus local (s15).** A global curve is a function `out = f(in)`: two pixels with the same value always get the same output, wherever they are. That is cheap and has no artefacts, but it cannot map a 45:1 brightness difference between two *regions* onto a display without squashing the contrast *inside* each region. A local operator uses `out = f(in, neighbourhood)`: it lowers the brightness of the bright region as a whole and lifts the dark region as a whole, while preserving the fine contrast. A darkroom print does exactly this: a card held over the shadows (dodging, "leaves print lighter") or a light on the highlights (burning, "makes print darker") changes the local exposure (s16). Compare the straight print and the toned print of Clearing Winter Storm (s17-18).

**Automatic dodge and burn (base/detail).** Work in the log of luminance, `L = log2 Y`, where multiplying light by a factor is an addition. Split `L = B + D` with a smooth base `B` (the local brightness, the "where would the card go" map) and a detail layer `D = L - B` (texture and edges). Then
`L_out = D + (1 - s)(B - B_max) + B_max = L + s (B_max - B)`.
Every pixel is brightened by `s` times how far its local brightness is below the brightest region, in stops. The detail is untouched, the base contrast is reduced by the factor `1 - s`. This is the scheme of Durand and Dorsey (2002); Reinhard et al. (2002), "Photographic Tone Reproduction", is the classic automatic dodging-and-burning operator that uses a global key plus a local adaptation level. **We implement the Durand-Dorsey base/detail operator** and borrow the idea of an anchor (`B_max`) from Reinhard.

**Halos.** If `B` is a Gaussian blur, it averages across strong edges. Next to a dark/bright edge the base on the bright side is pulled down by the dark side, so `s (B_max - B)` boosts it too much: a bright halo appears on the bright side (and a dark one on the dark side). An edge-preserving base (bilateral filter, weights `exp(-d^2/2 sigma_s^2) exp(-(L_p - L_q)^2 / 2 sigma_r^2)`, with `sigma_r` in stops) does not average across the step and the halo goes away.

## 3. Algorithm
Automatic dodge and burn (`dodge_burn`, linear in, linear out):
1. `Y = 0.2127 R + 0.7152 G + 0.0722 B`, `L = log2(Y + 1e-4)`.
2. Base `B = smooth(L)` (Gaussian of std `sigma`, or bilateral); detail `D = L - B`.
3. `L_out = D + (1 - s)(B - B_max) + B_max`; `Y_out = 2^L_out`.
4. Restore colour: `rgb_out = (rgb / Y)^c * Y_out` (`c = saturation = 1`); scale pixels with a channel above 1 down; output in [0, 1].
5. For display apply the sRGB curve (`tonemap(..., 'local')` does this).

## 4. Implementation notes
* `isp/tonemap.py`. Global: `gamma(img, g)`, `levels(img, black, white, g)`, `auto_levels(img, clip_percent)`, `histogram(img, bins)`, `hist_equalize(img, bins, mode='luminance'|'channel')` (own histogram -> CDF -> lookup, CDF interpolated between bin edges), `clahe(img, clip, tiles)` (wraps `cv2.createCLAHE` on 16-bit luminance, colour ratios kept). Local: `decompose`, `dodge_burn(img, sigma, strength, base, sigma_r, saturation, return_layers)` and the manual `dodge_burn_mask(img, mask, stops, feather)` (`out = in * 2^(stops * mask)`; +1 stop doubles linear values).
* **Domains.** `tonemap(img, method)` takes LINEAR in and returns DISPLAY (sRGB-encoded) out for every method:
  `srgb` = `linear_to_srgb`; `linear` = clip only (no encoding, shown for comparison); `gamma` = `img ** (1/2.2)` *instead of* the sRGB curve; `hist` and `clahe` = encode with sRGB first, then equalise the encoded values (so the *displayed* histogram is flat); `local` = `dodge_burn` on linear data, then sRGB. Equalising linear data would leave nearly all pixels in the darkest few display codes.
* `dodge_burn` default `sigma = 5 %` of the shorter side. The bilateral base runs on a downsampled copy (factor `sigma // 6`) and is upsampled; `sigma_r = 0.5` stops. `strength = 0` returns the (clipped) input exactly.
* Log floor `1e-4` (13 stops below white): without it, pure black pixels sit at -20 stops and dominate the base (speckles on the dark helmet).
* Pitfalls. Equalisation needs continuous values; images with large flat regions have a CDF with jumps, so the output histogram has spikes (our astronaut test has 0.059 CDF deviation, a noisy synthetic image 0.02). The gamut guard in luminance mode lowers the luminance of the few pixels that would clip, so strict luminance monotonicity holds for grey images and for unclipped colour pixels (the test checks this). `dodge_burn` leaves the sharpness question to the next stage: strong `strength` amplifies noise in the lifted shadows, so denoise first.

## 5. Experiments
All numbers are printed by `python scripts/make_figures_tonemap.py` (256 x 256 scenes).

* `docs/figures/tonemap_gamma.png` - the curve family `x^g` for `g = 1/8 ... 8` (L4 s14) with the sRGB curve dashed (close to `g = 1/2.2` except near zero), and the astronaut at `g = 0.5 / 1 / 2`. Mean display luminance 0.588 / 0.442 / 0.290: `g = 0.5` lifts the shadows (and washes out), `g = 2` crushes them.
* `docs/figures/tonemap_levels.png` - a low-contrast image (luminance 0.38..0.62, std 0.070) and the same after `auto_levels` (std 0.293) with their histograms: the histogram is the same shape, stretched to fill [0, 1].
* `docs/figures/tonemap_histeq.png` - before/after of `hist_equalize`, the two histograms and the CDFs. The output CDF follows the diagonal (max deviation 0.059 here, the image has flat regions; 0.02 and below on a continuous image, see the test); luminance std 0.070 -> 0.292.
* `docs/figures/tonemap_histeq_color.png` - a warm-tinted image equalised in luminance mode and per-channel mode. The mean r,g chromaticity error against the input is 0.0000 (luminance) versus 0.0608 (per-channel). The mean R/B ratio goes 1.58 -> 1.58 (luminance) versus 1.09 (per-channel): the per-channel version removes the cast, i.e. it is a crude (and uncontrolled) white balance, and changes the colours of everything else too. That is why we equalise luminance.
* `docs/figures/tonemap_global_vs_local.png` and the table below - `scenes.high_contrast`: a photograph with its left side 50x darker than the right (a dim interior next to a bright window). The dark side is the leftmost quarter of the image, the bright side the rightmost quarter. "Mean" is of the displayed luminance; "detail" is the std of the high-passed log2 luminance of the decoded linear light (1.5 px Gaussian), a measure of local contrast; the input has 0.219 on the left and 0.566 on the right.

| method | mean dark | mean bright | bright/dark | detail dark | detail bright |
|---|---|---|---|---|---|
| linear (clip) | 0.005 | 0.229 | 45.0 | 0.219 | 0.566 |
| sRGB | 0.053 | 0.374 | 7.02 | 0.219 | 0.566 |
| gamma 1/2.2 | 0.078 | 0.376 | 4.84 | 0.219 | 0.566 |
| hist-eq | 0.339 | 0.540 | 1.59 | 0.275 | 0.541 |
| CLAHE | 0.365 | 0.354 | 0.97 | 0.561 | 0.564 |
| dodge & burn, Gaussian | 0.378 | 0.561 | 1.49 | 0.333 | 0.566 |
| dodge & burn, bilateral | 0.419 | 0.639 | 1.52 | 0.236 | 0.432 |

  What to look for. The global curves (sRGB, gamma) keep detail contrast exactly as in the input (same last two columns: a monotone curve acting on all pixels cannot change relative local contrast in log terms) but leave the dark side at 0.05..0.08. Histogram equalisation lifts the dark side to 0.34 with a global curve, at the price of a strongly altered look. CLAHE and dodge-and-burn bring the two halves to a ratio of 1.0..1.5 instead of 7; CLAHE also raises the dark-side detail (0.56, amplifying noise and texture), dodge-and-burn keeps it near the input value (0.33 / 0.24 against 0.22). The bilateral variant gives the lowest bright-side detail (0.432), because it assigns the strongest edges to the base layer.
* `docs/figures/tonemap_halo_zoom.png` - the transition region at 3x: sRGB, Gaussian base and bilateral base. Look at the helmet and the shuttle/flag edges: the Gaussian base shows a bright rim next to dark objects, the bilateral base is clean.
* `docs/figures/tonemap_halo_profile.png` - one row through a 0.05 -> 0.8 step edge after `dodge_burn` (`sigma = 8`, `strength = 0.7`). Overshoot beyond the plateau as a fraction of the step height: **0.458 for the Gaussian base against 0.015 for the bilateral base**.
* `docs/figures/tonemap_base_detail.png` - the decomposition. The log luminance spans 13.3 stops; the bilateral base spans 13.2 stops (it keeps the large edges, e.g. the helmet, and so removes only the texture) while the Gaussian base spans 11.5 (the large edges are smeared). The detail layer of the Gaussian version has big values along strong edges; those are the future halos. The last panel is the compressed base recombined with the detail.
* `docs/figures/tonemap_mask.png` - `dodge_burn_mask`: the face region is dodged by +1 stop (mean linear luminance 0.155 -> 0.288, x1.86 instead of x2 because the mask is feathered and some pixels clip), then the bottom strip burned by -1 stop (0.071 -> 0.037).

## 6. Try it
```python
from isp import scenes, tonemap as T
img = scenes.high_contrast(256)                 # linear scene
a = T.tonemap(img, 'srgb')                      # camera default
b = T.tonemap(img, 'hist')                      # global, luminance equalised
c = T.tonemap(img, 'local', sigma=16, strength=0.7, base='bilateral')
lin = T.dodge_burn(img, strength=0.7, return_layers=True)   # (out, (base, detail))
darker = T.dodge_burn_mask(img, mask, stops=-1.0, feather=8) # mask: (H, W) in [0, 1]
```

## 7. Limitations and extensions
* Halos remain with a Gaussian base and with too-small `sigma_r` / too-large `strength`; with a bilateral base the edge threshold `sigma_r` is a free parameter (too large reverts to Gaussian).
* The operator compresses *global* brightness differences only as far as the base layer carries them; heavy `strength` gives the flat "HDR look" and amplifies shadow noise. Tone-map after denoising, or run the denoiser on the lifted shadows.
* Clipped (white) pixels carry no information; tone mapping cannot recover them, only RAW exposure can.
* Extensions: Reinhard's global operator `L / (1 + L)` with a key value; a guided-filter or local Laplacian base; per-channel saturation (`saturation < 1`) to avoid oversaturated lifted shadows; automatic `strength` from the base-layer range; tone mapping in the pipeline with the sRGB curve applied last.

## 8. References
* M. Levoy, lecture slides, *White Balancing and Tone Mapping*, 2010 (L4 s13-18: manual editing, gamma, global versus local, dodging and burning, Ansel Adams).
* F. Durand and J. Dorsey, "Fast Bilateral Filtering for the Display of High-Dynamic-Range Images", ACM SIGGRAPH 2002.
* E. Reinhard, M. Stark, P. Shirley, J. Ferwerda, "Photographic Tone Reproduction for Digital Images", ACM SIGGRAPH 2002.
* K. Zuiderveld, "Contrast Limited Adaptive Histogram Equalization", Graphics Gems IV, 1994.
* R. Gonzalez and R. Woods, *Digital Image Processing*, histogram processing.
* IEC 61966-2-1, sRGB.
