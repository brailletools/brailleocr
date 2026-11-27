# Results files

Each `*.json` here is the output of one run of a script in this repo, kept as a record.

**Commit a result only if the notes cite it, and add it with
`python experiments/publish_results.py FULL.json bench/results/NAME.json`**, which removes
dataset transcriptions we may not redistribute (and our reading of those images) and local
file paths. Keep full copies, and trial runs, outside the repository.

## Withdrawn (2026-11-27)

These were produced by `experiments/text_cer.py` / `ablate_crop_recover.py` (text error
rate) and do not measure reading quality. They compared Chinese Braille translated with an
English table, spell-corrected only the prediction, merged both page sides, and scored some
back-side labels on the wrong image. See `RESEARCH.md` "Corrections" and `AUDIT.md` A2.
Do not cite them.

- `2026-09-22-ablation.json`, `2026-09-22-ablation-union.json`
- `2026-09-28-text-cer-after-removal.json`
- `2026-09-29-text-cer-parity.json`

## Also affected by the back-side scoring bug (AUDIT A1)

- `2026-09-28-accuracy-by-side.json`, `2026-09-28-accuracy-perdot.json`,
  `2026-09-28-accuracy.json`, `2026-09-28-classifier-on-gt.json`,
  `2026-09-28-clf-*.json`, `2026-09-29-clf-*.json`: any per-side (recto/verso) figure,
  and anything for pages M+11-M+20, is wrong. Recto-only figures on pages whose two images
  are the same file are unaffected.

## First correctly scored run of the deployed models

- `2026-11-27-shipped-july-models-narrow-boxes.{json,txt}`: `js/eval/run.mjs` on the sets in
  `eval/splits.json`, with the July ONNX models. Ground-truth boxes are in the old,
  unstandardized shape (DSBI and Angelina uploads narrow), so the "classifier on
  ground-truth boxes" figures for those two sets differ from a run on the current manifest.

- `2026-11-27-shipped-july-models-wide-boxes.{json,txt}`: same run with ground-truth boxes in the
  standardized (wide) shape. Compare with the narrow-boxes run above: detection barely changes,
  classifying ground-truth boxes gets worse for the July classifier (AUDIT B4).
- `2026-11-27-braillebench-july-models-{before,after}-small-image-fix.json`: BrailleBench
  (259 real photos with English transcriptions) scored with `experiments/score_text.py`.
- `2026-11-27-braillebench-july-models-layout-guard-nms{0.7,0.3}.json`: after the layout
  guard against duplicate boxes, at two detector NMS thresholds (AUDIT G1, G5).

- `2026-11-27-dev-nms{0.7,0.3,0.1}.json`: detector NMS threshold compared on development
  sets only (AUDIT G5, I).

**Matching rule changed on 2026-11-27.** The runs above matched detections within half a
ground-truth box width, which depends on how the boxes were drawn; `js/eval` now uses 0.375 of
the cell pitch. Do not compare their detection or cell-error figures with later runs (nor the
narrow-box figures with the wide-box ones): rerun instead.

## Still valid

- `2026-09-22-edge-recall*.json` (four pages whose two images are identical, scored on the
  union of both sides) and the `bench.py` timing results: they do not depend on the
  affected scoring.

Corrected measurements are produced by `js/eval/run.mjs`.
