# Welcome

The Braille OCR project handoff doc. The goal is to read embossed Braille from ordinary phone photos — pages, signage, elevator buttons — and turn it into text.

This file covers how to run everything, what we found, what
is decided, and what is half-finished. `RESEARCH.md` is the lab notebook and has
the detail; this is the map.

---

## 1. Setup

Requires [pixi](https://pixi.sh).

```bash
pixi install
pixi shell          # or prefix every command below with `pixi run`
```

Sibling repos are expected next to this one (`REPOS_ROOT` in
`dot_pattern_utils.py` is this repo's parent):

```
braille/
  brailleocr/      ← you are here
  dataset/         ← models + data (github.com/brailletools/dataset)
  liblouis-env/    ← locates lou_translate; installed automatically
```

Models (`cell_detector.pt`, `cell_classifier.pt`) resolve from
`../dataset/models/`, or download automatically from the release pinned in
`dataset.version`. See `model_fetch.py`.

Sanity check:

```bash
python pipeline.py ../dataset/data/sample-images/IMG_3153.jpeg
```

---

## 2. How the pipeline works

Read `RESEARCH.md` → **"Process we've developed"** and the Mermaid diagram in
**"Pipeline flow"**. The short version:

1. **Container detection** — classical CV finds the page/sign/button, crops away
   context.
2. **Scale-normalized tiling** — the load-bearing idea. The detector is trained
   *only* on tiles where cells land at `TARGET_CELL_PX` (30px) after the resize
   to 640. At inference we measure median cell width, solve for the tile size
   that reproduces that density, and re-tile. Train and inference must agree;
   the constants live in `dot_pattern_utils.py` and are mirrored in
   `js/src/tiling.js`. **If you change one, change both.**
3. **Detection** — YOLOv8n, single class (cell / not cell), ~3M params.
4. **Classification** — MobileNetV2, six independent sigmoid dot outputs.
5. **Recovery stages** — `grid_fill`, `crop_recover`, `gap_pixel_recover`,
   `indicator_recovery`. **See §5: these are the problem.**
6. **Layout → liblouis → spell-check → English.**

The detector/classifier split is deliberate: because the detector carries no dot
semantics, datasets with *localization-only* labels can train it. That is what
lets `braille_natural` contribute.

---

## 3. Running things

| Command | Purpose |
|---|---|
| `python pipeline.py <image\|dir>` | End-to-end OCR |
| `python bench.py --limit 6 --repeats 1 --bootstrap 2000` | Per-stage cost with confidence intervals |
| `python evaluate.py --dataset <dir>` | Cell-level precision/recall (Angelina-format labels) |

| `python experiments/crop_call_modes.py` | Batching / disk-transport micro-benchmark |
| `python experiments/edge_cell_recall.py --limit 4` | Edge vs interior recall; recovery precision |
| `python experiments/text_cer.py --limit 4` | End-to-end text CER (regression test; baseline 0.4961) |
| `python experiments/label_batch.py --source braille_natural --limit 212` | Build the labeling queue |
| `python experiments/merge_corrections.py labels/corrections --out labels/csv` | Corrections → training data |
| `pixi run -e export onnx` | Export models for the browser path |
| `pixi run lint` | ruff |

Results are stored in `bench/results/*.json`. 

---

## 4. Labeling

There is a correction-based labeling tool. Predictions are prefilled; you only
fix what is wrong; anything untouched is recorded as confirmed correct. Right now, there's no concept of leading previous labeled jsons -- you might want to change that, but not sure it's worth it.

```bash
python experiments/label_batch.py --source braille_natural --limit 212
open /tmp/braille-label/index.html
# fix pages, save each download into labels/corrections/
python experiments/merge_corrections.py labels/corrections --out labels/csv
```

Interface: click a box to edit dots (`1`–`6` toggle, `Enter` save, `x` not-a-cell),
drag to move, drag the corner to resize, drag blank paper to draw a new cell.
Amber rings mark dots the classifier was unsure about; `n` jumps to them.
"Not embossed Braille" flags an out-of-scope image.

**Priority:** `braille_natural` — 212 real-world images (signage, elevator
buttons), boxes already human-drawn, dot patterns prefilled by the classifier,
9,152 cells of which ~10% have an uncertain dot. It is our only imagery in the
deployment regime and is currently localization-only; adding dot patterns makes
it usable for the classifier and for evaluation.

**Known bias:** confirming predictions by silence inherits the model's blind
spots. A cell the model never proposed and you never notice stays missing, so
recall measured against this set is optimistic. Spot-check a few pages against
from-scratch labels.

---

## 5. Train/test protocol

Detection and dot-finding share one protocol (`RESEARCH.md` "Unified train/test
protocol"), so both stages are measured on identical pages:

- **train** — each source's own train split
- **val** — each source's own val split; DSBI publishes none, so ~15% of its
  train *pages* are carved out (by page, never by cell — cells from one page are
  near-duplicates and would leak)
- **test** — all **88 DSBI test pages**, never trained or tuned on

```bash
python extract_crops.py --sources angelina,dsbi --out /tmp/braille-crops
python train_classifier.py --crop-dir /tmp/braille-crops
python experiments/classifier_on_gt.py --limit 0 --classifier /tmp/braille-crops/cell_classifier.pt
python experiments/text_cer.py          --limit 0 --classifier /tmp/braille-crops/cell_classifier.pt
```

Nothing has been promoted into `../dataset/models/` — the pipeline still loads
the **old** classifier until someone copies a new one there. That is deliberate:
promoting a model is a decision, not a build step.

---

## 6. Where we are

### Current accuracy

See RESEARCH.md for latest results. 

Read the caveats in `RESEARCH.md` before quoting these: DSBI train and test are
the same books, so this is in-domain accuracy, not generalization. The old
89.4% was a genuine cross-dataset number.

**Always state the unit when quoting accuracy.** A cell is six binary
decisions, so per-cell accuracy is roughly per-dot to the sixth power. It is possible for per dot to be competitive with published dot-level work; the same result quoted as per cell looks far worse against a paper that reported per-dot.

### Settled findings

- **The recovery stages fabricate.** On 4 held-out DSBI pages, `crop_recover`
  added 230 cells and `gap_pixel_recover` 29, of which **zero** matched ground
  truth — even against the union of both page sides. On phone photos every
  accepted cell landed at confidence ≤ 0.0063 (median 0.0014).
  → `RESEARCH.md` "Experiment 1", "Experiment 5"
- **There is no edge-cell deficit** to justify them: detection finds 98.7% of
  line-end cells vs 99.9% in line interiors.
- **Removing `crop_recover` improves the text**: mean CER 0.586 → 0.496 across
  4 pages, and 58% faster. → "Experiment 6"
- **No efficiency win exists inside it**: batching is *slower* (crops already
  fill the model input), and the disk round trip is ~1%. → "Experiments 2 and 3"
- **Most of the profile is not measurable at small n.** Bootstrap CI widths:
  `run_detection` 8%, `crop_recover` 98%, `clean_translation` 113%. Detecting a
  20% change in a recovery stage needs ~150 images. → "Experiment 4"

### Open problems

1. **`braille_natural` is contaminated.** At least one image (`img_116`) is a
   printed reference *table* of Braille patterns, not a photograph; a heuristic
   flags 16 of 212 as suspect. It is in the detector's training data. This is
   the dataset we added to fix real-photo generalization.
   → "Dataset contamination"
2. **Possible training bug:** `dsbi_data.py` loads `+recto` and `+verso` as two
   independent images, but they are the *same scan* (identical md5). If that
   reaches `prepare_yolo_dataset.py`, the detector was taught that each side's
   real cells are background. **Unverified — check this early, it is cheap.**
3. **No confidence reaches the user.** Rescued, spell-corrected and
   high-confidence text are typographically identical. For a blind user who
   cannot check the original, silent substitution is the worst failure mode.
4. **Capture is unmeasured.** We evaluate recognition given a photo; the
   assistive-OCR literature finds capture conditions dominate. → "Future work §2"
5. **`grid_fill` loses real cells** — 6 edge cells on one page in four.
   Unexplained.

## 7. To do

- [ ] Verify the DSBI recto/verso training question 
- [ ] Decide whether `labels/` belongs here or in the `dataset` repo

---

## 8. Learnings

Things that have already cost us time:

- **A loader that finds nothing looks like a loader that found nothing important.**
  `angelina_data.py` pointed at the wrong path and returned `[]` with no error,
  silently dropping 231 training images from every pipeline that used it.
- **Partial instrumentation lies.** An early `bench.py` measured 5 of 16 stages
  and reported "5.2s/image, 80% detection". The real figures were ~30s and 11%.
- **DSBI `+recto.jpg` and `+verso.jpg` are the same file.** Scoring per side
  double-counts pages and marks real other-side cells as errors. Group by
  physical page, match against the union.
- **Timings for input-dependent stages are nearly meaningless at small n.**
  Prefer counts (forward passes, acceptance rates) — they are exact per run.
- **Absolute CER from `ablate_crop_recover.py` is not a quality figure.** The
  reference interleaves both page sides and is machine-built. Only the delta
  is meaningful.
- **Our tiling is close to published work** (SAHI, and a 2026 adaptive variant).
  What appears novel is normalizing to a target *object* pixel size and binding
  training to it; both published methods key off *image resolution*. Read them
  before claiming novelty. → "Novelty check"

## 9. Reading order

1. `RESEARCH.md` — "Problem", "Process we've developed", "Pipeline flow"
2. `RESEARCH.md` — "Measured cost" and Experiments 1–6
3. `RESEARCH.md` — "Future work" (the confidence problem is the important one)
4. `README.md` (running the CLI), `OCR_SETUP.md` (browser/ONNX path)
5. Two papers first: the 2025 improved-YOLOv11 natural-scene paper (our exact
   setting, public dataset) and BrailleBench 2026 (the text-level evaluation
   frame we lack). → "Related work"
