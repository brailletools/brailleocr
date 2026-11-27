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
- [x] **B4. Two box conventions are mixed.** Box width / cell pitch is ~0.54 in DSBI *and in
  Angelina's user-upload photos* (so "Angelina" itself mixed both) and 0.73–0.77 in Angelina's
  books and handwriting and in `braille_natural`; height / width is 2.0 vs 1.5–1.6. Narrow
  pages are now rescaled to the wide shape (`canonical_boxes`). The
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
  test fixtures). **Follow-up that needs network: `pixi.toml` still lists `scipy`,
  `pyspellchecker` and the `liblouis-env` path dependency, none of which anything imports now,
  and CI still clones `liblouis-env` (`test.yml`, `lint.yml`, `liblouis-env.version`,
  `check-liblouis-env-release.yml`). Remove them together and run `pixi lock`; editing
  `pixi.toml` alone would break CI's locked install.**

## F. Research backlog (from the original related-work notes)

- [ ] Four papers could not be read in full (DSBI, AngelinaReader, improved-YOLOv11, the
  CVPRW 2020 segmentation paper); the last two returned HTTP 403.
- [ ] Compare our tiling with SAHI, including its full-image pass; classifier-head ablation
  (64-way softmax vs six sigmoids); detector-vs-classifier error attribution.

## Suggested order

A1–A4 → B1–B4, B7 (code only) → D1 (freeze) → retrain and re-measure → C1 and B5 → paper.
Items A and D1 must precede any number that goes in the paper.
