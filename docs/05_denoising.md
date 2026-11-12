# 05 - Denoising

## 1. In one paragraph
Sensor noise (doc 02) is added on top of the scene, so the cheapest way to remove it is to average neighbouring pixels. A Gaussian blur does that, but it also averages across edges and "destroys" them (L5 s5). The bilateral filter of Tomasi and Manduchi keeps the averaging but weights each neighbour by how *similar* it is to the centre pixel as well as how close it is, so pixels from the other side of an edge barely contribute (L5 s6-9). In a camera it runs on linear data after demosaicing, and it can be applied more strongly to chrominance than to luminance (L5 s10). `isp.denoise` implements Gaussian, our own vectorised bilateral filter, median, a luma/chroma variant, non-local means (extension) and temporal frame averaging (L2 s43).

## 2. Theory
**Noise model.** On linear data we use signal-dependent Poisson-Gaussian noise, `var = a * s + b^2` (shot noise `a s`, read noise `b^2`; doc 02). Dark pixels have little absolute noise, bright pixels a lot. The experiments below add this noise with `b = 0.01` and `a = 0.003 / 0.01 / 0.03` (low / mid / high).

**Why blurring fails (L5 s6).** The bilateral slides assume the image is *piecewise constant plus noise*. Then nearby pixels are mostly other noisy measurements of the same value, which is exactly what averaging wants, except near an edge, where some neighbours belong to a different piece of the scene. A Gaussian cannot tell the difference. The ideal weight is "the probability that a pixel comes from the same piece of the scene".

**The filter (L5 s8-9).** For the output pixel `(i,j)` and a neighbour `(k,l)` in a window `Omega`:

`w(i,j,k,l) = exp( -((i-k)^2 + (j-l)^2) / (2 sigma_d^2)  -  ||I(i,j) - I(k,l)||^2 / (2 sigma_r^2) )`

`I_D(i,j) = sum_{k,l} I(k,l) w(i,j,k,l) / sum_{k,l} w(i,j,k,l)`

The first term is a Gaussian in the *domain* (image coordinates, `sigma_d` pixels), the second a Gaussian in the *range* (intensity, `sigma_r` in image units). "Blur in the domain and range dimensions" (L5 s7): the kernel falls off with distance from the pixel and with intensity difference from the pixel. The kernel therefore changes from pixel to pixel and is not a convolution.

Two limits fix the intuition and are unit-tested. `sigma_r -> inf` makes the range term 1 and gives the Gaussian blur of the same `sigma_d`. `sigma_r -> 0` leaves only the centre pixel, the identity. In between, `sigma_r` is the intensity step that counts as "an edge": differences below about `sigma_r` are treated as noise and averaged, differences above it as structure and kept. The output is a convex combination of input pixels, so it never leaves the input's value range.

**Luma and chroma (L5 s10).** The eye is much less sensitive to colour detail than to brightness detail. In YCbCr we filter `Y` gently and `Cb, Cr` strongly (larger `sigma_d` and `sigma_r`), which removes colour blotches without softening the picture.

## 3. Algorithm
Bilateral filter, as implemented (`isp.denoise.bilateral`):
1. Reflect-pad the image (and the optional guide) by `r` pixels.
2. For each window offset `(dy, dx)`, take the shifted view of the padded array.
3. Compute the squared colour distance to the centre pixel, `||I(k,l) - I(i,j)||^2`, summed over channels.
4. Weight `w = exp(-(dy^2+dx^2)/(2 sigma_d^2)) * exp(-d^2/(2 sigma_r^2))`.
5. Accumulate `sum w I` and `sum w`; divide at the end.

Luma/chroma (`luma_chroma`): RGB -> YCbCr; `Y <- bilateral(Y, luma_kw)`; `(Cb, Cr) <- bilateral((Cb, Cr), chroma_kw)`; back to RGB. Defaults: luma `sigma_d=1.5, sigma_r=0.05`, chroma `sigma_d=4, sigma_r=0.10`.

## 4. Implementation notes
* **API**: `gaussian(img, sigma)`, `bilateral(img, sigma_d, sigma_r, radius=None, guide=None, window='square', color_dist='l2')`, `bilateral_weights(img, center, sigma_d, sigma_r, radius)`, `median(img, size)`, `nlm(img, h)`, `average_frames(frames)`, `luma_chroma(img, luma_kw, chroma_kw)`, `estimate_noise(img)` and the dispatcher `denoise(img, method, **kw)`. Same API as conventions.md section 4, with extra optional arguments only.
* **Vectorisation.** The loop is over the `(2r+1)^2` offsets, never over pixels; each iteration is a few whole-image NumPy operations. Cost is `O(H W r^2 C)`. Measured: 256x256 RGB, radius 7: 0.34 s. Default radius is `ceil(2 sigma_d)`.
* **Gaussian reference.** `gaussian` uses the same window radius (`ceil(3 sigma)` default) and reflect-101 border as the bilateral, so the limit test `sigma_r -> inf` compares like with like (agreement below 1e-3, in practice about 1e-6).
* **Same-window caveat.** Truncating the window means the bilateral weights are not exactly a Gaussian even for `sigma_r = inf` unless the radius is the same; pass `radius=` explicitly when comparing.
* **Colour distance.** The slide uses `||.||^2`, the L2 distance over R, G, B (default). `cv2.bilateralFilter` differs in two ways: its window is a disc (`window='circle'` reproduces it) and for 3 channels it uses the *L1* distance, `(sum_c |dc|)^2` (`color_dist='l1'`). With both options our result matches OpenCV on float32 images to under 1e-4 over the whole image (measured about 1e-6 for grey and colour); with the default L2 and a disc window it differs from OpenCV by up to about 0.07 on a noisy colour photo.
* **`guide`** gives the cross (joint) bilateral filter: weights from `guide`, values from `img`. Used optionally by `luma_chroma(guide_chroma=True)`; with noisy data it was slightly worse than letting the chroma find its own edges, so it is off by default.
* **`sigma_r` has image units.** On linear data in `[0, 1]`, noise grows with the signal, so one `sigma_r` is too strong in the shadows and too weak in the highlights. Filtering after a variance-stabilising transform (e.g. sqrt or gamma) is a standard fix; not done here.
* **`estimate_noise`**: robust MAD of a high-pass response (`1.4826 MAD / 6` for the kernel `[[1,-2,1],[-2,4,-2],[1,-2,1]]`). It is biased up on textured images; it measured 0.036 on the mid-noise astronaut, an average over bright and dark areas.
* **Median** (`scipy.ndimage.median_filter`, per channel) and **NLM** (`skimage.restoration.denoise_nl_means`, patch 5, search 6, `h` and noise sigma in image units) are small wrappers. NLM is an extension, not on the slides.
* Pitfall: if `sigma_r` is below the noise std, almost no neighbour looks similar and little is smoothed; the filter degenerates to the identity (see the heat-map).

## 5. Experiments
All numbers come from `python scripts/make_figures_denoise.py`. The photo is `scenes.photo('astronaut')` (linear), PSNR in the linear domain, SSIM on the sRGB-encoded images. Gaussian, bilateral and NLM parameters are the oracle-best on a small grid (the clean image is used to pick them), so these numbers are an upper bound for what a user would tune by eye.

**`docs/figures/denoise_kernels.png`: L5 s6-7 redone.** A noisy piecewise-constant patch (a bright wedge on a dark ground, noise std 0.05). Top row: the patch, the Gaussian kernel at a pixel on the dark side right next to the edge, the bilateral weights at the same pixel, the clean patch. The Gaussian is a round blob that spills 27.6% of its weight over the edge. The bilateral window is the same blob with the bright side cut away (0.02% of the weight there). Bottom row: outputs and a row through the pixel. The Gaussian turns the step into a ramp of about 20 pixels; the bilateral keeps it a step. Flat-region noise std: noisy 0.0525, Gaussian 0.0036, bilateral 0.0094. PSNR against the clean patch: noisy 26.0 dB, Gaussian 23.4 dB (worse than doing nothing), bilateral 41.3 dB.

**`docs/figures/denoise_compare.png`: the L5 s10 layout.** Noisy, Gaussian, bilateral, luma/chroma and NLM on the mid-noise astronaut, full image plus crop.

| method | PSNR (dB) | SSIM |
|---|---|---|
| noisy | 25.38 | 0.620 |
| Gaussian (sigma 0.6) | 29.31 | 0.767 |
| bilateral (sigma_d 1.5, sigma_r 0.2) | 31.49 | 0.851 |
| luma/chroma (default parameters, untuned) | 30.26 | 0.834 |
| NLM (h 0.06), extension | 30.91 | 0.837 |

Look at the helmet rim and the suit seams in the crop: the Gaussian blurs them, the bilateral keeps them. The bilateral result has a slightly "painted", blocky look in flat areas; it flattens low-contrast texture (the skin). The best Gaussian sigma is tiny (0.6) because anything larger costs more in lost detail than it gains in noise.

**`docs/figures/denoise_heatmap.png`: parameters.** Left: PSNR over `(sigma_d, sigma_r)` on the 128x128 photo at mid noise; the star marks the optimum, `sigma_d = 1.5`, `sigma_r = 0.2`, 30.23 dB (best Gaussian: 28.24 dB, noisy: 25.47 dB). The far right column (`sigma_r = 0.5`) is the Gaussian limit and gives 25.46 dB at `sigma_d = 2`; the far left column (`sigma_r = 0.02`) is nearly the identity and gives 25.56 dB. The good region is a band in between. Right: best `sigma_r` against noise std for constant-variance noise at `sigma_d = 2`: 0.015, 0.03, 0.04, 0.12, 0.16 for noise std 0.005, 0.01, 0.02, 0.04, 0.06, so roughly `sigma_r ~ 2-3 sigma_noise`: it must clear the noise (so noise differences are averaged) while staying well below the edge contrasts.

**`docs/figures/denoise_edge_profile.png`: edge profile.** A step 0.3 -> 0.7 with noise std 0.04. Left: the row-averaged profile around the edge; right: one row. Maximum slope of the profile (ideal 0.400 per pixel) and 10-90% width:

| method | max slope | 10-90% width (px) | flat-region std |
|---|---|---|---|
| noisy | 0.395 | - | 0.0408 |
| Gaussian (sigma 2.5) | 0.063 | 6.48 | 0.0044 |
| bilateral (2.5, 0.15) | 0.371 | 0.86 | 0.0057 |
| median 7 | 0.300 | 1.73 | 0.0065 |
| NLM | 0.393 | 0.82 | 0.0036 |

The Gaussian and the bilateral filter with the same `sigma_d` remove about the same amount of flat-region noise (0.0044 vs 0.0057), but the bilateral edge is six times steeper. This is the whole point of the filter.

**`docs/figures/denoise_chroma.png`: chroma blotches.** High noise (`a = 0.03`). Top row: only the chroma (`Y` set to 0.5) in a 64x64 crop of the dark visor and the orange suit: clean, noisy, the gentle bilateral filter applied to R, G, B, and luma/chroma. Bottom row: the same crops in colour. Gentle filtering on RGB hardly touches the coloured speckle; luma/chroma removes most of it.

| image | chroma RMSE | luma RMSE | luma gradient | PSNR (dB) |
|---|---|---|---|---|
| clean | 0 | 0 | 0.0414 | inf |
| noisy | 0.0592 | 0.0598 | 0.0697 | 20.71 |
| gentle bilateral on RGB | 0.0575 | 0.0585 | 0.0669 | 20.94 |
| luma/chroma | 0.0269 | 0.0511 | 0.0592 | 24.63 |

Chroma error falls by 55% (gentle RGB filtering: 3%). The luma gradient, a sharpness measure that noise inflates above the clean value 0.0414, drops from 0.0697 to 0.0592, closer to the clean value; luma RMSE falls only from 0.0598 to 0.0511 because the gentle luma filter leaves most luma noise in place.

**`docs/figures/denoise_frames.png`: temporal averaging (L2 s43).** 1, 4 and 16 independent noisy frames of the same scene (`a = 0.03`) averaged: noise std 0.0918, 0.0459, 0.0230, i.e. ratios 1.000, 0.500, 0.251 against the predicted `1/sqrt(M)` of 1, 0.5, 0.25; PSNR 20.74, 26.76, 32.76 dB (+6 dB per 4x frames). Unlike all spatial filters this costs no resolution, but needs a static scene (or alignment) and M times the capture time.

**`docs/figures/denoise_table.png`: method x noise level** (astronaut 128x128, PSNR in dB, best parameters per column for Gaussian, bilateral and NLM; median is 3x3, luma/chroma uses the default parameters):

| method | low | mid | high |
|---|---|---|---|
| noisy | 30.35 | 25.47 | 20.80 |
| Gaussian | 31.46 | 28.24 | 25.48 |
| bilateral | 34.12 | 30.23 | 26.65 |
| median 3 | 27.16 | 26.16 | 24.40 |
| luma/chroma | 32.43 | 29.59 | 24.50 |
| NLM (ext.) | 32.84 | 29.22 | 25.58 |

The tuned bilateral filter wins at every level. Luma/chroma is untuned and loses at high noise because its luma `sigma_r = 0.05` is too small for that noise level (the optimal `sigma_r` grows with the noise), not because the idea is wrong. The median is poor on dense Gaussian noise (it is built for salt-and-pepper outliers).

**Tests** (`tests/test_denoise.py`, 12 tests): the two limits, constant image, dtype/shape/range, guide, weight window at an edge, agreement with OpenCV (grey and colour), edge slope and flat-region noise on a step, PSNR of best bilateral vs best Gaussian on a grid, luma/chroma chroma error, `1/sqrt(M)`, dispatcher and noise estimator.

## 6. Try it
```python
import numpy as np
from isp import scenes, denoise as D
from isp.utils import psnr, rng

clean = scenes.photo('astronaut', 128)
a, b = 0.01, 0.01                                   # var = a * signal + b^2
noisy = (clean + np.sqrt(a * clean + b * b) * rng(0).standard_normal(clean.shape)).astype(np.float32)

print(psnr(clean, noisy))
print(psnr(clean, D.gaussian(noisy, 1.0)))
print(psnr(clean, D.bilateral(noisy, 1.5, 0.2)))
print(psnr(clean, D.luma_chroma(noisy)))
w = D.bilateral_weights(noisy.mean(-1), (60, 60), 3.0, 0.1)   # kernel at one pixel
```
`python scripts/make_figures_denoise.py` regenerates the figures and prints the numbers.

## 7. Limitations and extensions
* **Piecewise constant is a model, not a fact.** Texture and fine low-contrast detail (skin, fabric) look like noise to the filter and are flattened, giving the cartoonish look in the crop. `sigma_r` trades noise removal for detail and the optimum depends on the image.
* **One `sigma_r` for the whole image** ignores the signal-dependent noise. Try a variance-stabilising transform (Anscombe/sqrt) first, or make `sigma_r` a function of the local signal using `estimate_noise` and the `a, b` of the sensor.
* **Cost.** Direct evaluation scales with `r^2`. The bilateral grid and Fourier/Gaussian-approximation methods make it O(1) per pixel (Paris & Durand 2006).
* **Gradient reversal / staircasing**: the bilateral filter makes plateaus, so smooth ramps turn into steps.
* **Better denoisers.** NLM (patches rather than pixels) and BM3D do clearly better at high noise; deep denoisers more so. This is "an active area of research" (L5 s10).
* **Joint demosaicing and denoising** (L5 s10): demosaicing interpolates noise into correlated blotches that are hard to remove afterwards, so doing both in one step is better than a pipeline of the two. Try denoising the mosaic or the colour differences before `isp.demosaic`.
* Try: cross-bilateral filtering with a flash image as the guide, iterating the filter, or a chroma-only denoiser followed by a luma NLM.

## 8. References
* C. Tomasi and R. Manduchi, "Bilateral filtering for gray and color images", ICCV 1998 (L5 s6-9).
* M. Levoy, lecture slides *Sensors and Noise* (L2 s41-43) and *Denoising, Sharpening, Compression* (L5 s5-10), Stanford CS 178, 2010 / course EE 5176.
* A. Buades, B. Coll and J.-M. Morel, "A non-local algorithm for image denoising", CVPR 2005 (extension).
* S. Paris and F. Durand, "A fast approximation of the bilateral filter using a signal processing approach", ECCV 2006.
* J. Immerkaer, "Fast noise variance estimation", Computer Vision and Image Understanding 64(2), 1996 (the high-pass noise estimator).
* G. Bradski, OpenCV `cv2.bilateralFilter` documentation (reference implementation used in the tests).
