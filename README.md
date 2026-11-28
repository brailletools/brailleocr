# brailleocr

Reads embossed Braille from phone photos. The reader that ships is JavaScript
(`js/`, published to the web editor as `@brailletools/brailleocr-web`): it detects
Braille cells with a small YOLOv8 model, classifies each cell's dots with a MobileNetV2
model, and lays the cells out in lines. Both models run in the browser through
onnxruntime-web. The Python here is a small second implementation of detection and classification,
used for labeling and to generate test fixtures.

Training, evaluation, labeling and experiments live in the separate, private
`braille-research` repository, along with research notes, plans and result files:
`AUDIT.md` there lists known bugs or unfinished validation, and `RESEARCH.md` is old
and has errors (corrections are up front). Comments in this code that cite those files refer
to that repository.

## Organization

| Path | What |
|---|---|
| `js/src/` | The reader: orientation search, scale-normalized tiled detection, classification, line layout |
| `reader.py` | Python tiled detection and classification, used only by labeling and the JS test fixtures |
| `dot_pattern_utils.py` | Tiling and box conventions that the readers and training must share |
| `model_fetch.py` | Finds the trained models, or downloads the pinned release |

The trained models live in the sibling `dataset` repo (`../dataset/models`).

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

## History

Until 2026-11-27 this repo also had `pipeline.py`, a Python command-line reader with
contrast search, container detection, "recovery" stages that guessed at missed cells,
spell-correction and liblouis translation. Measurements showed the recovery stages only
invented cells, the web editor never used it, and its evaluation scripts scored something
other than what ships, so it was removed (see `AUDIT.md`). It remains in git history, and
comments in `js/src` that say "port of pipeline.py" refer to it.
