# brailleocr audit and work list

Started 2026-11-26. Conducted a from-scratch review of the data. Evidence is stated with each item and
marked **measured** (run on real data), **code** (read in the source) or **unverified**.

Where a finding contradicts `RESEARCH.md` or `WELCOME.md`, those documents were written
before it and are wrong on that point; see `RESEARCH.md` "Corrections (2026-11-27)".

## Done

| What | Where | Evidence |
|---|---|---|
| Sideways and upside-down photos are detected and turned | `js/src/orientation.js`; editor | measured: 5 of 5 failed photos, 40 of 40 rotated samples; no false turns on 59 Angelina photos and 20 DSBI pages; 1 of 48 signage photos wrongly turned |
| Lines survive page tilt and bend | `js/src/lineLayout.js`, `pipeline.py` | measured: 1–5° tilt now changes ~1% of output (was 25–83%) |
| One blank cell was read as two | same | measured: output had doubled spaces; cause was dividing a gap by cell width, not cell pitch |
| Unescaped LaTeX characters broke the HTML preview | `braille-bridge` v0.4.4 | measured: all 5 failed photos' text failed in latex.js before, parse after |
| HEIC photos give a clear message | `webeditor` | code only; not tried with a real HEIC in Chrome |

## A. Evaluation (everything else is measured with this)

Done 2026-11-27: `experiments/export_eval_manifest.py` writes the held-out sets
(`eval/splits.json`, committed, with file hashes); `js/eval/run.mjs` scores the *shipped*
pipeline on them (detection, per-cell accuracy, cell error rate, classifier on ground-truth
boxes by side; intervals resample pages). `text_cer.py` and `evaluate.py` were deleted (they
remain in git history); their results files under `bench/results/` are marked withdrawn in
`bench/results/README.md`.

- [x] **A1. `classifier_on_gt.py` scores each DSBI back side on the front image.** The two
  images are deskewed separately (different angles on 19 of 114 pages, all of them in the
  `Massage` book). Measured: on test pages M+11–M+20 the same cells score 52.6% the repo's
  way and 69.5% on the correct image (July classifier). The "faint back side, r = 0.82"
  finding in `RESEARCH.md` is an artifact of this.
- [x] **A2. `text_cer.py` is not a quality measure.** DSBI is Chinese Braille, translated
  here with an English table; the reference is not spell-corrected but the prediction is;
  both page sides are merged; and the back-side labels are applied to the wrong image
  (A1). Replace with a cell-level error rate on Unicode Braille (language independent),
  scored per side on the right image. English text metrics only on English data.
- [x] **A3. `evaluate.py` runs a different pipeline from the shipped one** (untiled,
  percentile margin filter, contrast search). The shipped pipeline is the JavaScript one in
  the editor. One harness should run the shipped code.
- [x] **A4. Correct `RESEARCH.md` and `WELCOME.md`** (faint back sides; CER 0.48 as an open
  question; "NSBD is a dataset we lack" — `braille_natural` is NSBD; quoted accuracy
  figures that rest on A1/A2).
- [ ] **A5. Commit the analysis scripts.** The inventory, label-alignment, rotation and tilt
  tests were run from scratch files; they should live in `experiments/` so the numbers in
  the paper can be regenerated.
- [x] **A6. Statistics.** Resample pages, not cells, for confidence intervals; state the unit
  (dot, cell, page) next to every figure. (`bench.py` already does this for timings.)

## B. Training data

Code fixed 2026-11-27 (B1–B4, B7; the shared logic is in `angelina_data.py`, `dsbi_data.py`,
`dot_pattern_utils.py`, `prepare_yolo_dataset.py`, `extract_crops.py`, with tests in
`tests/test_data_fixes.py`). **No model has been retrained yet**; the deployed models still
reflect the old data. A scratch rebuild of the detector set gives 3,478 train / 582 val /
1,691 test tiles, of which 415 are background; 19 DSBI pages (9 train) are skipped as
half-labeled. Angelina's test photos were also checked for near-duplicates of training
pages (shared cell sequences): none of the 50 share over half.

- [x] **B1. The Angelina loader trained on the authors' test set.** `uploaded/test2` is the 50
  user-upload photos the authors designate as test; the loader re-splits them 34/8/8.
  (code; upstream split files)
- [x] **B2. Blank tiles are dropped, so the detector never sees background.** Measured:
  14 of 44 Angelina non-Braille images produce detections. Those 44 images are unused.
- [x] **B3. Identical DSBI scans are used twice with different labels.** 90 pages have
  byte-identical `+recto.jpg` and `+verso.jpg`; each copy labels only one side, so each
  copy teaches the detector that the other side's cells are background. (code, file hashes)
- [x] **B4 (code done; the choice of shape is NOT settled). Two box conventions are mixed.** Box width / cell pitch is ~0.54 in DSBI *and in
  Angelina's user-upload photos* (so "Angelina" itself mixed both) and 0.73–0.77 in Angelina's
  books and handwriting and in `braille_natural`; height / width is 2.0 vs 1.5–1.6. Narrow
  pages are now rescaled to the wide shape (`canonical_boxes`). **Caution (measured with the July
  models, which were trained on a mix):** classifying ground-truth boxes got worse on the wide
  shape: Angelina uploads 97.0% -> 94.1%, DSBI front 92.1% -> 82.3%, back 75.7% -> 62.5%
  (`bench/results/2026-11-27-shipped-july-models-*-boxes.txt`). That is partly because the old
  classifier learned narrow crops, so it is not evidence that narrow is better, but it means
  the choice needs an A/B retrain (classifier and detector on narrow vs wide) before anyone
  claims wide is better. The crop padding in `extract_crops.py`/the readers is a second knob. The
  classifier sees whatever box the detector draws. (measured)
- [ ] **B5. Back-side (verso) handling.** Angelina books are double-sided but only the front
  is labeled, so back-side dots are background. DSBI is the only labeled back-side source.
  AngelinaReader reads a back side by inverting and mirroring the image; that could double
  the usable data.
- [ ] **B6. `braille_natural` contains diagrams, not photographs** (≥1 confirmed, 16 of 212
  suspected) and 6 of 48 test images have under 50% of cells found. Exclude out-of-scope
  images before training or testing.
- [x] **B7. A loader that finds nothing looks like success.** Make every loader fail loudly
  on an empty result (this already cost 231 training images once).
- [ ] **B8. Record what the shipped models were trained on.** Unverified whether the July
  detector used Angelina. The September classifier (97.4% per cell) is not in `/tmp` and
  was never promoted; deployed models date from 2026-07-08/13.
- [ ] **B9. Augmentation for capture:** tilt, perspective, blur, light direction, scale
  range. The detector currently reads only about ±5° of tilt.

## C. Model and product

- [ ] **C1. Two-sided (interpoint) pages** read as garbage (photo IMG_3816). Needs front/back
  separation: raised dots and dents together. Largest remaining gap.
- [ ] **C2. No confidence reaches the reader.** The web path marks nothing as uncertain;
  a blind reader cannot tell a guess from a read. Design with blind users.
- [ ] **C3. Strike-out runs** (all-six-dot cells, "forfor…") are read as text in the web path.
- [x] **C4. Orientation detection is JavaScript only.** No longer an issue: the Python reader
  was retired (E6), so there is one reader.
- [ ] **C5. Browser classifier resizes crops without antialiasing**; Python (PIL) antialiases.
  Only matters for large cells. (code; unverified)
- [ ] **C6. Tilts of 10° or more and strong perspective** still fail (detector, not layout).
- [ ] **C7. Capture guidance** ("move closer", "tilt less") is unmeasured and absent.

## D. Test sets and labels

- [ ] **D1. Frozen, held-out test sets**, recorded in a manifest with file hashes: your
  phone photos (consent and privacy first), Angelina's 50 user uploads, `braille_natural`
  test (cleaned), DSBI books not in training.
- [ ] **D2. `labels/` is empty.** The 662-cell verification of IMG_3153 in the notes is not
  saved anywhere.
- [ ] **D3. New English phone photos with ground truth**: embossed from known BRF files and
  photographed under varied conditions (see `PAPER_PLAN.md`).
- [ ] **D4. Licensing.** Angelina and DSBI declare no licence; `braille_natural` has none.
  Decide what can be redistributed (weights, splits) before release.

## E. Housekeeping

- [x] **E1. JavaScript tests do not run in CI** (`.github/workflows/test.yml` only runs
  ruff and pytest).
- [ ] **E2. Two locked worktrees hold uncommitted files** (`docs/` in `pipeline-notebook`,
  `RESEARCH-baselines.md` in `related-work-baselines`); decide, then remove.
- [ ] **E3. Stale remote branches** (`bug-fixlines`, `feat-add-spaces`, `onnx-support`, all
  merged).
- [ ] **E4. Package versions** (`js/package.json`, `pixi.toml` say 0.1.0; tags reach v0.1.3).
- [ ] **E5. `braille-bridge`:** Markdown prose is not escaped (affects Word/HTML export);
  lualatex tests should skip when TeX cannot start.

- [x] **E6. `pipeline.py` retired (2026-11-27).** Removed `pipeline.py`, `container_detect.py`,
  `bench.py`, `run_ocr.sh`, `OCR_SETUP.md` (stale TensorFlow.js notes),
  `experiments/edge_cell_recall.py` and `crop_call_modes.py`, and their tests. The small part
  still needed is `reader.py` (tiled detection and classifier loading, for labeling and the JS
  test fixtures). **Follow-up that needs network: `pixi.toml` still lists `scipy` and `pyspellchecker`, which
  nothing imports now. Remove them with `pixi remove --pypi scipy pyspellchecker`, which updates
  `pixi.lock` too (editing `pixi.toml` alone would break CI's locked install). `liblouis-env`
  and its CI steps stay: `experiments/score_text.py` uses it.**

## G. Ideas from the papers read on 2026-11-27 (all nine are in `../papers/`)

Each is a proposal to test with `js/eval/run.mjs` and the frozen sets, not a result.

- [~] **G1. BrailleBench (Jangra et al., arXiv 2610.36195, Oct 2026; code `github.com/kartik0806/BrailleBench`)
  is the independent English real-photo test set we lack**: 259 manually verified photos (215
  Grade 1, 44 Grade 2; signage, metal plates, books). It scores AngelinaReader at 3.26% exact
  match on Grade 1 and 0% on Grade 2, general VLMs at 0–54%, an agentic harness with crop/enlarge
  tools at 88% / 41%, and a fine-tuned 2B VLM at 87% / 61%. Check its licence, add it to
  `eval/`, and score our reader with liblouis back-translation (CER and exact match, both
  grades). Not to be confused with the text-only "BrailleBench" in `RESEARCH.md` (2608.27268).
  **Done 2026-11-27:** downloaded to `../external/BrailleBench/` (outside all repos; the repo
  declares no licence, so it is not redistributed), added as the `braillebench` set
  (`experiments/export_eval_manifest.py`, `js/eval/run.mjs`, `experiments/score_text.py`). July
  models: Grade 1 exact match 18.1% -> 29.8% after fixing scale normalization for small images
  (below), Grade 2 15.9% (n = 44, interval 4.5-27.3). Results in `bench/results/`. Many images are
  tiny crops (median 382 x 86 px) with blurred, magnified dots or white dots on black screens.
  **Runaway outputs explained and fixed:** the detector drew two boxes on one cell, which
  dragged the pitch estimate to about zero (one image: 13,716 blank cells). Layout now ignores
  gaps under one cell width and caps blanks per gap at 5: Grade 1 pooled CER 10.0 -> 0.566
  (exact match unchanged). With detector NMS 0.3 as well (G5): Grade 1 exact 35.8%
  [29.3, 42.3], CER 0.439; Grade 2 18.2%, CER 0.316.
  Still to do: the images with no output (15 Grade 1), and the 142-image `book-dataset` in the
  same repo (text labels only).
- [ ] **G2. Lighting direction is the orientation problem.** Ovodov 2021: front dots and
  back dots look alike when the image is turned 180 degrees; JACIII 2026 (Yamashita et al.):
  AngelinaReader collapses beyond about 60 degrees of lighting angle, and a detector trained
  on 12 lighting angles holds mAP50 0.98 at all of them. Capture the same pages under varied
  light on a tripod and copy registered annotations across the images (their labeling
  trick), for the capture benchmark (D3) and for training.
- [ ] **G3. Front/back separation (C1, B5):** (a) Ovodov reads a back side by flipping and
  inverting the image and running the front-side reader again; (b) BraUNet (Li et al. 2020)
  labels recto and verso characters as separate classes and gets F1 0.9966 / 0.9893 on DSBI.
  Try a two-class detector (front/back) trained on DSBI union pages, and mirror-and-invert
  back-side cells as extra front-side training data.
- [ ] **G4. Augmentation.** Ovodov: random scale +/-30%, vertical stretch +/-10%, rotation
  +/-5 degrees, horizontal flip with label swap, per-image mean/std normalization; trained on
  DSBI+Angelina together (DSBI->Angelina alone F1 0.927; mixed 0.998). Our `train_detector.py`
  sets `degrees=0` and `fliplr=0`; for the single-class detector a flip does not change any
  label, and small rotations are what the tilt tests need. Try both, and per-image
  normalization at train and inference.
- [x] **G5. Detector NMS: default changed to 0.3 (2026-11-27)** in `js/src/detector.js` and
  `reader.py`. AngelinaReader uses IoU 0.02 because cells never overlap; our JS
  detector uses 0.7. On seen sets, lower values helped signage precision (80.8% -> 85.2% at
  0.3) and BrailleBench (Grade 1 exact 29.8% -> 35.8% at 0.3). **On development sets only
  (July models): 0.1 is clearly harmful** (DSBI val detection recall 98.8% -> 72.2%: on
  two-sided pages, front and back cells overlap and one is suppressed); **0.3 is
  indistinguishable from 0.7** (books and handwriting unchanged; DSBI val cell error rate 14.7%
  -> 14.5%, 4 pages). Chosen 0.3: no measurable harm on development data, motivated
  independently by AngelinaReader; its benefit has been seen only on seen sets, and the paper
  must say so.
- [ ] **G6. Match published metrics** in the scorer: character-level IoU >= 0.5 plus correct
  class (Ovodov, Li) and dot-level precision/recall/F1, alongside ours.
- [ ] **G7. Signage.** Sun et al. 2025 (improved YOLOv11) use a *different* natural-scene
  dataset from `braille_natural`: Lu et al. 2023, 554 images annotated as Braille *regions*,
  not cells (Baidu link, or request from the authors); they detect at 640 px with no tiling and
  report P 0.942 / R 0.951 at region level, so not comparable to ours. A learned region
  detector could replace the removed container detection and help our weakest set.
- [ ] **G8. Do not rely on synthetic data for validation.** BrailleBench: a model at 96% on
  synthetic validation images scored 3.26% on real photos. Synthetic pages (liblouis text,
  paper textures, perspective, lighting) are fine as extra training data only if real photos
  stay the test. Same paper: general LLMs fail silently with plausible wrong text, which
  supports C2 (confidence reaching the reader); zero-shot LLM post-correction is unreliable
  (also 2607.11893, Korean).
- [ ] **G9. Not useful for us:** the Evetac tactile-sensor paper (2601.19079); ASAHI
  (2604.19233, aerial imagery) beyond its Cluster-DIoU-NMS and "train on full images plus
  slices" ideas (see G5; `RESEARCH.md` Novelty check already discusses slicing).

## F. Research backlog (from the original related-work notes)

- [x] The four papers that could not be read were read on 2026-11-27 (see G); the improved-YOLOv11 paper does
  not ship a cell-level dataset, only the Lu 2023 region-level one (G7).
- [ ] Compare our tiling with SAHI, including its full-image pass; classifier-head ablation
  (64-way softmax vs six sigmoids); detector-vs-classifier error attribution.

## I. Test discipline (adopted 2026-11-27)

**Rules.**

1. Settings, thresholds, box shapes, augmentations and model choices are decided on
   **development sets only**: `angelina-books-val`, `angelina-hand-val`, `dsbi-val` (4 fully
   labeled pages held out from DSBI training by `dsbi_data.val_page_names()`), plus a small set
   of our own photos set aside for development (to be made).
2. **Held-out sets** are scored once, at the end: the 142 `braillebook` photos until the
   capture benchmark (D3) exists, then the capture benchmark, which is photographed and frozen
   before anyone runs the reader on it.
3. **Seen sets** are reported, never optimized against, and called "seen during development"
   wherever they appear. Report the first-run number next to the final one and list every
   change in between.
4. A bug found by looking at a seen set may be fixed (with a synthetic test), and is listed
   as such.

`eval/splits.json` records each set's role. `js/eval/run.mjs` refuses to try `--iou` (a tuning
knob) on anything but development sets and will not touch held-out sets without `--final`. Any
new tuning option must be added to that guard.

**Contamination so far (cannot be undone).**

| Change | What was looked at |
|---|---|
| Orientation thresholds | the 10 sample pages and 5 failed photos; checked on `angelina-uploaded`, `dsbi-test`, signage |
| Lower-quartile pitch (review fix) | the same 15 photos |
| Box shape set to wide | measured on all evaluation sets |
| Small-image scaling fix | found on BrailleBench (bug, synthetic test) |
| Pitch floor and blank-cell cap | found on BrailleBench (bug, synthetic test) |
| NMS default changed 0.7 -> 0.3 | checked harmless on development sets; benefit observed on signage and BrailleBench (seen) |
| NMS threshold trials | `angelina-uploaded`, signage, `not-braille`, BrailleBench; part of a `dsbi-test` run was started and stopped, its numbers unused. Repeated on development sets only (G5) |

BrailleBench's first run (Grade 1 exact 18.1%) is the only figure from before any change
motivated by it.

**Limit.** The development pages are flat, mostly Russian books and Chinese scans. They will not
show problems with tiny crops, sideways or tilted phone photos, which is why a development
set of our own photos is needed. The July models were probably trained on some development
pages, so development scores of those models are optimistic.

## H. Retraining experiment 1: box shape (needs GPU time; not yet run)

Same data (with the B1-B3 fixes), same settings, two box shapes. The July models are only a
reference point: they differ in data too, so they cannot answer the box question. Do not add the
augmentation changes (G4) to this experiment; test them afterwards, one at a time.
Run it with `caffeinate -i bash experiments/retrain_box_shape.sh` (in a normal terminal; it
resumes if interrupted, and writes to `../retrain/box-shape`). For each of `wide` and `narrow`
the script does the following (`$R` = its output folder):

```bash
python prepare_yolo_dataset.py --single-class --box-shape $S --out $R/yolo-$S
python train_detector.py --data $R/yolo-$S/dataset.yaml --name $S     # prints where best.pt is
python extract_crops.py --sources angelina,dsbi --box-shape $S --out $R/crops-$S
python train_classifier.py --crop-dir $R/crops-$S
pixi run -e export python export_onnx.py --detector <best.pt for $S> \
    --classifier $R/crops-$S/cell_classifier.pt --out-dir $R/onnx-$S

python experiments/export_eval_manifest.py --box-shape $S --splits-out $R/splits-$S.json \
    --cells-out $R/cells-$S.json
cd js && node eval/run.mjs --cells $R/cells-$S.json --out $R/eval-$S.json --iou 0.3 \
    --detector $R/onnx-$S/cell_detector.onnx --classifier $R/onnx-$S/cell_classifier.onnx
cd .. && python experiments/score_text.py $R/eval-$S.json        # BrailleBench rows
```

Decide on the **development sets only** (section I; add `--role dev` to the `run.mjs` line when
deciding): cell error rate and per-cell accuracy on *detected* boxes (the real pipeline),
paired by page, with intervals. Run the seen sets afterwards, for the chosen shape and for the
record; skip the BrailleBench line until then. "Classifier on ground-truth boxes" differs between S by construction (different
crops), so compare it only within a shape. Both models of a shape must be used together.

The script scores development sets only and ends with a paired comparison
(`experiments/compare_results.py`). One training seed per shape: a small difference could be
run-to-run variation, so repeat with another seed before claiming one.

## Suggested order

A1–A4 → B1–B4, B7 (code only) → D1 (freeze) → retrain and re-measure → C1 and B5 → paper.
Items A and D1 must precede any number that goes in the paper.
