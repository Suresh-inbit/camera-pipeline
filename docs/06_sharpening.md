# 06 - Sharpening

## 1. In one paragraph
Sharpening makes edges look crisper by adding a scaled high-pass copy of the image back to itself: `S = x + lam F(x)` (L5 s15). With the 4-neighbour Laplacian as `F` the whole operation is one 3x3 kernel, the Photoshop "Custom" filters of L5 s13-14 and s16-17. It raises *acutance* (edge contrast, steepness of the edge), not resolution: no detail that the lens and sensor lost is recovered. It also amplifies noise and makes halos, which is why a camera sharpens last, after denoising, and why the strength is a taste parameter.

## 2. Theory
**Unsharp masking (L5 s15).** `F(x)` is a high-pass filtered image, `lam >= 0` a gain, `S` the output. In the classical version `F = x - G_sigma * x` (the image minus a blurred copy, the "unsharp mask"). The Laplacian version uses `F = [[0,-1,0],[-1,4,-1],[0,-1,0]]` (L5 s16), whose taps sum to 0, so flat regions give `F = 0`.

**One kernel.** Since the identity kernel is `delta`, `S = (delta + lam F) * x`:

```
lam = 1: [[ 0,-1, 0],      lam = 2: [[ 0,-2, 0],      general: centre 1 + 4 lam,
          [-1, 5,-1],                [-2, 9,-2],               4 neighbours -lam
          [ 0,-1, 0]]                [ 0,-2, 0]]
```

These are exactly the numbers typed into the Photoshop dialog on L5 s13 (centre 5) and s14 (centre 9); scale 1 and offset 0 are right because the taps sum to 1, so mean brightness is preserved.

**Frequency response.** The Laplacian has `H(wx, wy) = 4 - 2 cos wx - 2 cos wy`: 0 at DC, 4 at `(pi, 0)`, 8 at `(pi, pi)`. The sharpening operator is therefore `1 + lam H`: unity at DC (brightness and large areas untouched), growing towards the highest frequencies (`1 + 4 lam` at `(pi, 0)`, `1 + 8 lam` at `(pi, pi)`). It is a high-frequency boost, not a band-limited one. A Gaussian unsharp mask has `H = 1 - exp(-sigma^2 w^2 / 2)`, so the gain saturates at `1 + lam` and the boost starts at lower frequencies for larger sigma.

**Overshoot.** At a step edge the added `lam F` is a positive lobe on the bright side and a negative lobe on the dark side, so the result overshoots both plateaus (the "halo", the signal drawn on L5 s15). Edges become steeper *because* of that overshoot; the two cannot be separated with a linear filter.

**Noise gain.** For white noise of standard deviation `s`, a kernel `k` gives output std `s sqrt(sum k^2)`. For lam = 1 that is `sqrt(25 + 4) = 5.39`, for lam = 2 `sqrt(81 + 16) = 9.85`.

## 3. Algorithm
1. Compute the high-pass image `F(x)` (Laplacian filter, or `x - GaussianBlur(x, sigma)`).
2. Optionally zero `F` where `|F| < threshold`.
3. Output `S = x + lam F`; clip to `[0, 1]`.
4. Optionally do steps 1-3 on the Y channel of YCbCr only and convert back.

## 4. Implementation notes
* `isp.sharpen.high_pass(img, kind, sigma)`, `unsharp_mask(img, lam, kind, sigma, threshold, clip, luminance_only)`, `sharpen(img, method, lam, **kw)` (`'laplacian'` or `'unsharp'`). `LAPLACIAN`, `LAPLACIAN8`, `sharpening_kernel(lam, kind, sigma)`, `high_pass_kernel`.
* The Laplacian path equals convolution with `sharpening_kernel(lam)` in the interior (tested for lam = 1 and 2 to 1e-5). Borders are reflected (`cv2.BORDER_REFLECT_101`).
* Analysis helpers: `edge_profile(img)` samples a profile across `scenes.step_edge` along the edge normal; `overshoot(profile)` returns `max(max - hi, lo - min) / (hi - lo)`; `rise_distance(d, profile)` is the 10-90 % rise distance; `mean_gradient(img)`; `frequency_response(kernel, n)` and `unsharp_response(lam, kind, sigma, n)` (real, FFT order: `[0, 0]` DC, `[0, n//2]` is (0, pi), `[n//2, n//2]` is (pi, pi)); `noise_gain(kernel)`.
* Sharpen **display-referred** data (after tone mapping). On linear data the gamma curve changes how large the halos look in the shadows and highlights, and in practice every camera sharpens after tone mapping.
* `threshold` is a hard, per-pixel, per-channel test on `|F|`. It is a crude noise guard, not an edge-aware filter; it leaves isolated noise spikes above the threshold untouched (the speckles in the right panel of `sharpen_noise.png`).
* `luminance_only=True` changes Y by `lam F(Y)` and leaves Cb, Cr exactly as they were before the final clip, equivalent to adding the same `lam F(Y)` to R, G and B.
* Cost: one 3x3 filter, O(pixels); the Gaussian version is separable.

## 5. Experiments
All numbers are printed by `python scripts/make_figures_sharpen.py`. The photo is the 320x320 astronaut image, sRGB-encoded; the edge is `scenes.step_edge(128)` (0.2 to 0.8, 1-pixel anti-aliased ramp), unclipped when measuring overshoot.

* **`sharpen_photo.png` (L5 s12-14, s16-17 re-made).** Original, lam = 1, lam = 2 with the kernels printed, and a face crop. Mean gradient magnitude rises 0.043 -> 0.087 -> 0.117. Hair and eyelashes snap out, but the white fringe along the hair silhouette is already visible at lam = 1 and strong at lam = 2. Clipping rises from 9 % of pixels (original) to 28 % and 37 % (any channel at 0 or 1), and this clipping is why the image mean drifts from 0.449 to 0.454 and 0.460 although the kernel sums to 1.
* **`sharpen_unsharp_diagram.png` (L5 s15 with real data).** A soft pulse (0.2 -> 0.8), its Gaussian high-pass (sigma 2) with positive and negative lobes at each edge, `lam F` with lam = 1.5, and the sum. The sum overshoots the 0.8 plateau by 0.074 (12 % of the 0.6 step) and undershoots 0.2 by 0.074, and is checked in the script to equal `x + lam F` exactly.
* **`sharpen_edge_profiles.png`.** Profiles across the step edge for lam = 0 ... 3. Laplacian: overshoot 0, 24, 49, 98, 147 % for lam = 0, 0.5, 1, 2, 3 (roughly 49 % per unit lam), while the 10-90 % rise distance falls 1.60 -> 1.06 -> 0.80 -> 0.53 -> 0.39 px. Gaussian unsharp (sigma = 2) gets less overshoot for the same lam (31 % at lam = 1) and a wider, slower halo. Steeper edge and bigger halo go together.
* **`sharpen_frequency_response.png`.** Left and middle: the Laplacian operator along `wx` and along the diagonal. At lam = 1 the gain is 1.0 at DC, 5.0 at (pi, 0), 9.0 at (pi, pi); at lam = 2 it is 9 and 17. Right: Gaussian unsharp for sigma = 1, 2 and lam = 1, 2. Gain saturates at 1 + lam (1.99 at pi for sigma = 1; for sigma = 2 the plateau of 2.0 is reached already near w = 2.7), so it boosts a band from "mid" frequencies upward, while the Laplacian keeps growing to the corner.
* **`sharpen_noise.png`.** White Gaussian noise (std 0.02) added to the photo. In a smooth patch the pixel std goes 0.041 (noisy) -> 0.118 (lam = 1 Laplacian), a clear amplification; with `threshold = 0.3` it is 0.043 and PSNR vs. the clean image goes from 17.7 dB (plain sharpening) to 22.3 dB (noisy input: 34.0 dB), while the face and helmet edges are still sharpened. The pure-noise test agrees with the formula: measured 5.40 vs 5.385 (lam = 1) and 9.87 vs 9.85 (lam = 2). A threshold removes noise only where noise is below the edge strength; the isolated coloured speckles that remain are noise that exceeded it.
* **`sharpen_order.png`.** Denoise then sharpen (camera order) vs sharpen then denoise, with a small bilateral filter and a mild Gaussian unsharp (sigma 1.5, lam 1). PSNR vs. the clean image: 26.1 dB vs. 25.8 dB; vs. the *sharpened clean* image: 33.6 dB vs. 32.0 dB. The difference is small with a good bilateral filter, but sharpening first hands the denoiser noise that has already been amplified and correlated into small blobs, which an edge-preserving filter treats as structure; the sharpen-first result has the larger mean gradient (0.063 vs 0.060) partly because of that residual noise.
* **`sharpen_halos.png`.** Over-sharpening: Laplacian lam = 1 (49 % overshoot, halo about 2.5 px wide), lam = 3 (147 %, black and white outlines, a saturated "cartoon" look), and a wide Gaussian (sigma = 4, lam = 2: 80 % overshoot, but spread over about 18 px, the broad halo around the head and the "local contrast" look). Small sigma gives crisp narrow halos; large sigma gives wide soft ones.
* **`sharpen_luma_vs_rgb.png`.** A soft red-blue edge, Gaussian unsharp with lam = 2. Sharpening R, G, B independently moves the chroma: max |dCb, dCr| = 0.15 and the channels clip to 0 and 1, producing a saturated red line and a cyan line at the edge (the middle bottom panel shows the Cb change). Sharpening Y only leaves Cb and Cr unchanged (max change 0.000) and no colour fringe.
* **`sharpen_table.png`** collects noise gain, overshoot, rise distance, photo mean gradient and gain at (pi, 0) for several operators.

## 6. Try it
```python
import numpy as np
from isp import scenes, utils, sharpen as S

img = utils.linear_to_srgb(scenes.photo("astronaut", 256))     # display-referred
out = S.unsharp_mask(img, lam=1.0)                             # Laplacian: kernel [[0,-1,0],[-1,5,-1],[0,-1,0]]
soft = S.sharpen(img, "unsharp", lam=1.0, sigma=2.0)           # Gaussian unsharp mask
safe = S.unsharp_mask(img, 1.0, threshold=0.1, luminance_only=True)

d, p = S.edge_profile(S.unsharp_mask(scenes.step_edge(128), 2.0, clip=False))
print(S.overshoot(p), S.rise_distance(d, p))                   # ~0.98, ~0.53 px
print(S.noise_gain(S.sharpening_kernel(1)))                    # 5.39
```

## 7. Limitations and extensions
* **It does not restore resolution.** The linear boost gives frequencies that the lens and sampler attenuated a larger gain, but frequencies that were lost (below the noise or exactly zero, such as everything above the optical cutoff) stay lost; the response `1 + lam H` is multiplied by a transfer function that already was near zero. Edges look sharper (acutance, rise distance) without the image containing more detail. Genuine recovery needs a model (deconvolution with a known PSF, learned priors).
* **Noise.** The gain at high frequencies is the gain on noise; denoise first, and keep lam small (noise gain 5.4 at lam = 1 is already large).
* **Halos and clipping.** The overshoot is inherent. Clipped halos (white fringes on dark hair) lose information permanently.
* **Colour fringes** when RGB channels are sharpened separately; sharpen luminance only.
* Extensions: edge-aware unsharp masking (bilateral or guided-filter base layer, no halos), noise-adaptive thresholds, local-contrast (large sigma) vs. detail (small sigma) in one pipeline, deconvolution (Wiener, Richardson-Lucy) as the principled alternative, and the pipeline order in `isp.pipeline` (tone map -> denoise -> sharpen -> JPEG).

## 8. References
* Lecture 5 "Post-processing: Denoising, Sharpening, Compression", slides 12-17 (Photoshop Custom kernels, unsharp masking S = x + lam F(x)); the unsharp-masking figure is credited on the slide to NPTEL (chapter 8).
* R. C. Gonzalez and R. E. Woods, *Digital Image Processing*, 4th ed., sections 3.6 (Sharpening spatial filters: Laplacian, unsharp masking, high-boost filtering).
* A. K. Jain, *Fundamentals of Digital Image Processing*, 1989 (unsharp masking as a high-pass boost).
