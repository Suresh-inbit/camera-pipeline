# Camera pipeline explorer (Streamlit)

```
streamlit run app/streamlit_app.py
```

The pipeline diagram at the top of every page is the navigation: 13 boxes (label, live thumbnail of that
stage's output, data domain / bit depth) under coloured bands (Scene, Sensor, ADC, Processing, Storage; the
boxes of L2 s3). Click a box to open that stage; hover for its one-line summary. Disabled stages are greyed
and dashed. Layout: bands wrap as flex items, so at laptop width the diagram is two rows broken at a group
boundary (Scene / Sensor / ADC, then Processing / Storage). No horizontal scrolling, everything visible at once.

Pages: **Overview** (final vs ground truth, quality trace, stage table, enable/disable, order presets, reset) and
one page per stage (parameters, before/after with synchronised zoom crop, amplified difference, histograms,
metrics, an "Experiment" expander, and the stage's doc from `docs/`).

Fallback navigation: Prev / Next, the sidebar "Go to" selector (session key `stage`), deep links `?stage=<id>`.

## Files
* `streamlit_app.py` pages and layout; `_common.py` state, incremental runs, thumbnails, nav payload, docs rendering
* `stage_params.py` parameter widgets per stage; `experiments.py` the per-stage experiments (cached)
* `nav.py` + `pipeline_nav/index.html` the zero-build custom component (no npm)

## Component protocol (checked against Streamlit 1.49 `static/static/js/index.*.js`)
Component to host, via `window.parent.postMessage({isStreamlitMessage: true, type, ...}, "*")`:
`streamlit:componentReady {apiVersion: 1}`, `streamlit:setComponentValue {value, dataType: "json"}`,
`streamlit:setFrameHeight {height}`. Host to component: `{type: "streamlit:render", args, theme, ...}`.
A click sends `{stage, nonce}`; Python acts only when the nonce is new, sets `st.session_state['stage']`,
syncs `st.query_params['stage']` and reruns once.

## State and speed
The pipeline (params, enabled flags, order) and the last `PipelineResult` live in `st.session_state`; widgets
change the pipeline in `on_change` callbacks and the next run passes `previous=` so only downstream stages
recompute. Note that `StageMetrics`' cache key contains all parameters before `demosaic`, so a change to an
earlier stage recomputes everything (about 0.35 s at 256 px). A rejected change (for example disabling
`black_level`) restores the last working pipeline and shows a warning.

RAW upload appears only if `rawpy` is installed (`pip install rawpy`); it enters at `black_level`, without metrics.

Tests: `python -m pytest tests/test_app.py -q`.
