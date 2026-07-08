#!/usr/bin/env bash
# Run the Braille OCR pipeline on one image or a directory of images.
# Usage: ./run_ocr.sh path/to/image.jpg [--lang en-ueb-g2.ctb] [--no-contrast-search]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CLASSIFIER="/tmp/braille-crops/cell_classifier.pt"

CLASSIFIER_ARG=""
if [ -f "$CLASSIFIER" ]; then
    CLASSIFIER_ARG="--classifier $CLASSIFIER"
fi

python "$SCRIPT_DIR/pipeline.py" "$@" $CLASSIFIER_ARG
