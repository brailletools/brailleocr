# brailleocr

Python OCR pipeline for phone photos of embossed Braille pages. Detects Braille cells with YOLOv8, optionally reclassifies them with a MobileNetV2 cell classifier, and back-translates the result to English text.

## Setup

Requires [pixi](https://pixi.sh).

```bash
pixi install
pixi shell   # or prefix commands below with `pixi run`
```

This also installs [liblouis-env](https://github.com/brailletools/liblouis-env), which
fetches/locates the `lou_translate` binary for you (works on macOS, Linux, and Windows —
see that repo for details). No manual liblouis install step needed.

Download the YOLOv8 Braille model (first run will auto-download via ultralytics, or place manually):

```bash
mkdir -p /tmp/yolov8-braille
# The model is fetched automatically from HuggingFace on first use:
#   snoop2head/yolov8m-braille
```

Optional: download the MobileNetV2 cell classifier from the
[brailletools/dataset](https://github.com/brailletools/dataset) repo
(`models/cell_classifier.pt`) and place it at `/tmp/braille-crops/cell_classifier.pt`.
The pipeline uses it automatically if found.

## Run on an image

```bash
python pipeline.py path/to/braille_photo.jpg
```

Run on a whole directory:

```bash
python pipeline.py path/to/photos/
```

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--lang TABLE` | `en-ueb-g2.ctb` | liblouis table for back-translation |
| `--classifier PATH` | auto-detect | Path to `cell_classifier.pt` |
| `--no-contrast-search` | off | Skip multi-pass contrast tuning |
| `--no-spellcheck` | off | Disable spell-correction of output |

**Output files** are written to `/tmp/braille-yolo-results/`:

- `*_annotated.jpg` — detected cells coloured by confidence (green/yellow/red = reliable, cyan = rescued)
- `*_dots.jpg` — dot-level debug view

## Sample script

```bash
./run_ocr.sh path/to/braille_photo.jpg
```

## Evaluate on a labelled dataset

```bash
python evaluate.py --dataset path/to/labeled_dataset/
```

The dataset directory should contain `.jpg` image files and matching `.json` annotation files
(see `evaluate.py` `--help` for format details).

## Training the cell classifier

```bash
# 1. Extract crops from the Angelina dataset (clone as a submodule first)
python extract_crops.py

# 2. Train
python train_classifier.py
```

The trained model is saved to `/tmp/braille-crops/cell_classifier.pt`.
