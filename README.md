# Camera Pipeline and Post-processing (EE 5176 Computational Photography)

A small, readable toolkit that simulates a digital camera from photons to a JPEG file
and implements the post-processing methods of lectures 2-6: sensor noise, colour filter
arrays and demosaicing, white balance, tone mapping, denoising, sharpening and JPEG
compression. Every intermediate stage of the pipeline can be inspected.

Work in progress. See `docs/conventions.md` for the data and API contract.

```bash
python -m pytest tests -q
```
