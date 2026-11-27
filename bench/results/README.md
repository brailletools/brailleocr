# Results files

Each `*.json` here is the output of one run of a script in this repo, kept as a record.

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

## Still valid

- `2026-09-22-edge-recall*.json` (four pages whose two images are identical, scored on the
  union of both sides) and the `bench.py` timing results: they do not depend on the
  affected scoring.

Corrected measurements are produced by `js/eval/run.mjs`.
