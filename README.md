# brailleocr

Reads embossed Braille from phone photos. The reader that ships is JavaScript
(`js/`, published to the web editor as `@brailletools/brailleocr-web`): it detects
Braille cells with a small YOLOv8 model, classifies each cell's dots with a MobileNetV2
model, and lays the cells out in lines. Both models run in the browser through
onnxruntime-web. The Python in this repo trains and evaluates those models and supports
labeling; it is not a second reader.

`AUDIT.md` lists known bugs or unfinished validation. `RESEARCH.md` is old and has errors. Corrections are up front.

## Organization

| Path | What |
|---|---|
| `js/src/` | The reader: orientation search, scale-normalized tiled detection, classification, line layout |
| `js/eval/` | Scores that reader on the held-out sets (`run.mjs`) |
| `eval/splits.json` | Which images are in each held-out set, with file hashes |
| `angelina_data.py`, `dsbi_data.py`, `braille_natural_data.py`, `dot_pattern_utils.py` | Dataset loaders and the shared tiling and box conventions |
| `prepare_yolo_dataset.py`, `train_detector.py` | Build the detector training set and train it |
| `extract_crops.py`, `train_classifier.py` | Build the classifier crops and train it |
| `export_onnx.py` | Export both models for the browser |
| `reader.py` | Python tiled detection and classification, used only by labeling and the JS test fixtures |
| `experiments/` | Labeling tools, the evaluation manifest exporter, classifier-on-ground-truth scorer |
| `labels/` | Human-verified labels from the labeling tool |

The datasets live in the sibling `dataset` repo (`../dataset/data`, git submodules for
Angelina and DSBI), and so do the trained models (`../dataset/models`).

## Setup

Python tools use [pixi](https://pixi.sh):

```bash
pixi install
pixi shell          # or prefix commands with `pixi run`
```

The JavaScript reader and the scorer need Node 22 or later:

```bash
cd js && npm ci && npm test
```

## Evaluate on the held-out sets

This scores the reader the web editor ships, on sets fixed in `eval/splits.json`. It needs
the sibling `dataset` checkout.

```bash
python experiments/export_eval_manifest.py --cells-out /tmp/eval-cells.json
cd js
node eval/run.mjs --cells /tmp/eval-cells.json --out /tmp/eval-results.json
node eval/run.mjs --summarize /tmp/eval-results.json      # the table again, without rerunning
node eval/run.mjs --orientation ../eval/orientation.json  # photos with a known upright turn
```

**Choosing anything** (a threshold, a model, a box shape): use `--role dev` and nothing else;
see `AUDIT.md` section I. Held-out sets run only with `--final`.

Add `--sets dsbi-test,angelina-uploaded` or `--limit 5` to run part of it, and
`--detector` / `--classifier` to score other ONNX models. Every figure states its unit
(cell, dot, page) and intervals resample pages.

## Train

```bash
# detector
python prepare_yolo_dataset.py --single-class --out /tmp/braille-yolo-dataset-singleclass
python train_detector.py --data /tmp/braille-yolo-dataset-singleclass/dataset.yaml

# classifier
python extract_crops.py --sources angelina,dsbi --out /tmp/braille-crops
python train_classifier.py --crop-dir /tmp/braille-crops
python experiments/classifier_on_gt.py --limit 0 --classifier /tmp/braille-crops/cell_classifier.pt

# both models for the browser
pixi run -e export onnx
```

Promoting a trained model into `../dataset/models/` is a decision, not a build step.

## Label

`experiments/label_batch.py` pre-fills pages with the Python reader's predictions and
`experiments/label_tool.py` builds a page you correct in the browser; corrections are merged
into training data with `experiments/merge_corrections.py`. See `labels/README.md`.

## History

Until 2026-11-27 this repo also had `pipeline.py`, a Python command-line reader with
contrast search, container detection, "recovery" stages that guessed at missed cells,
spell-correction and liblouis translation. Measurements showed the recovery stages only
invented cells, the web editor never used it, and its evaluation scripts scored something
other than what ships, so it was removed (see `AUDIT.md`). It remains in git history, and
comments in `js/src` that say "port of pipeline.py" refer to it.
