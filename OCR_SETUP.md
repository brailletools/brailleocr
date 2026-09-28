# Browser (client-side) Braille OCR setup

How to get the Braille OCR models running in a browser via
[onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/), with no backend
server — suitable for a static/GitHub Pages deployment. The reference consumer
is the `webeditor` app; the reusable pieces live in `js/` in this repo
(`@brailletools/brailleocr-web`).

For the Python CLI pipeline, see `README.md` instead.

## Architecture

Two models, matching `pipeline.py`'s two stages:

| Model | What it is | Job |
|---|---|---|
| `cell_detector.onnx` | YOLOv8n, **single class** (localization only), ~3M params | Find Braille cell boxes |
| `cell_classifier.onnx` | MobileNetV2, 6 independent sigmoid outputs | Read the 6 dot positions of each cell |

The detector deliberately does *not* classify dot patterns — it finds cells, and
the classifier reads them. That split is what lets `braille_natural` (which has
localization-only labels) contribute to detector training, and it keeps the
detector small enough to ship to a browser.

Both checkpoints live in the [`brailletools/dataset`](https://github.com/brailletools/dataset)
repo under `models/`, already exported alongside their `.pt` sources. You only
need the export step below if you have retrained them.

## Step 1: get the models

Checked out as a sibling repo (the normal local-dev layout), the `.onnx` files
are already there:

```
../dataset/models/cell_detector.onnx
../dataset/models/cell_classifier.onnx
```

Otherwise fetch them from the pinned release of the `dataset` repo (see
`dataset.version`, which `model_fetch.py` reads).

Serve them as static assets from your app, e.g. `public/models/`.

## Step 2: re-export, only if you retrained

```bash
pixi run -e export onnx
```

This runs `export_onnx.py`, which writes both `.onnx` files next to their `.pt`
sources in `../dataset/models/`. The `export` pixi environment carries `onnx` +
`onnxruntime` so they aren't runtime dependencies of `pipeline.py`.

Two export details that are load-bearing, not optimizations:

- **`dynamic=True` for the detector.** A fixed-shape export changes
  ultralytics' letterbox/aspect-ratio handling relative to the `.pt` model's
  dynamic-shape default, which silently changes detection counts — 38 vs 50
  boxes on the same image at the same threshold, back to exact parity once
  re-exported with `dynamic=True`.
- **`dynamo=False` for the classifier.** The `torch.export`-based exporter
  (default since torch 2.9) needs `onnxscript`, which isn't in this repo's
  deps. The legacy TorchScript exporter needs nothing extra and produces a
  numerically equivalent graph (max abs logit diff 5.7e-6, 100% bit-level
  agreement after `sigmoid > 0.5`).

If you retrain the detector, recheck `TARGET_CELL_PX` / `TILE_SIZE` in
`dot_pattern_utils.py` and `js/src/tiling.js` — see Step 4.

## Step 3: use the JS package

```bash
cd js && npm install
```

```js
import {
  CellDetector,
  CellClassifier,
  detectScaleNormalized,
  layoutCellsIntoLines,
  layoutToUnicodeBraille
} from '@brailletools/brailleocr-web';

const detector = await CellDetector.load('/models/cell_detector.onnx');
const classifier = await CellClassifier.load('/models/cell_classifier.onnx');

// rgbHwc: Float32Array of HWC RGB pixels, values 0-255 (e.g. from a canvas
// getImageData(), dropping the alpha channel).
const boxes = await detectScaleNormalized(detector, rgbHwc, imgW, imgH);
const bits = await classifier.classify(rgbHwc, imgW, imgH, boxes);

const cells = boxes.map((b, i) => ({ ...b, bits: bits[i] }));
const lines = layoutCellsIntoLines(cells);
const braille = layoutToUnicodeBraille(lines); // Unicode U+2800 block, '\n' between lines
```

Back-translation to English is **not** done in JS — hand the Unicode Braille to
liblouis (`liblouis-env` on the server, or a WASM liblouis build in the browser).

Run the JS tests with `cd js && npm test` (node's built-in test runner).

## Step 4: tiling is required, not optional

`detectScaleNormalized()` exists because the detector was trained *exclusively*
on tiles where cells land at `TARGET_CELL_PX` (30px) after the resize to
`TILE_SIZE` (640) — see `prepare_yolo_dataset.py`. Feeding it a whole untiled
page puts cells far outside that distribution: box *count* degrades gracefully,
but box *size* does not (roughly 3× oversized boxes, before `js/src/tiling.js`
existed).

`detectScaleNormalized()` does a first tiled pass at the default `TILE_SIZE`,
measures the median cell width, solves for the native tile size that would put
cells at `TARGET_CELL_PX`, and re-tiles only if that estimate is more than ~50%
off either way.

The constants in `js/src/tiling.js` mirror `dot_pattern_utils.py`. If they drift
apart, the whole scheme silently stops doing anything useful.

## Parity notes (JS vs Python)

The JS side is a port, and a few places match Python deliberately rather than
doing the idiomatic JS thing:

- **`imageOps.js` reimplements `cv2.resize(INTER_LINEAR)`** with half-pixel
  centre sampling instead of using Canvas `drawImage` scaling. Browsers' image
  smoothing doesn't match cv2 closely enough to reproduce ultralytics'
  letterbox output, which the detector was tuned against. It's also why the
  module is pure-JS with no DOM dependency — the same code runs in Node tests.
- **`lineLayout.js` implements banker's rounding** (`bankersRound`) because
  Python's `round()` is round-half-to-even while JS's `Math.round()` is not, and
  `insertSpaces()` is a direct port where a tie changes the number of inferred
  spaces.
- **The classifier crops with 10% padding** to match `pipeline.py`'s
  `reclassify_cells()`, and thresholds at `logit > 0` (equivalent to
  `sigmoid > 0.5`).

## What the JS path does NOT do

`pipeline.py`'s recovery machinery is out of scope for the browser port:
contrast search, grid-guided rescue, single-cell crop re-detection, pixel-level
gap recovery, indicator-gap recovery, container detection, spell-check cleanup.
Expect lower accuracy than the CLI on difficult photos. Tiling was ported
because it's load-bearing; the rest is robustness the browser path currently
trades away.

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| Boxes ~3× too large | Detection run on the untiled image — use `detectScaleNormalized()` |
| Detection counts differ from the `.pt` model | Detector exported without `dynamic=True` |
| `onnxscript` import error on export | Missing `dynamo=False` on the classifier export |
| Classifier output is noise | Input not normalized with ImageNet mean/std, or HWC not converted to CHW |
| Model fails to load | `.onnx` not served as a static asset, or wrong MIME type |
