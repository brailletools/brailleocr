#!/usr/bin/env bash
# Retraining experiment 1: label box shape, wide vs narrow (AUDIT section H).
#
# Same data, settings and seed for both shapes; only the box shape differs. Each shape gets
# its own detector and classifier, exported to ONNX and scored with the shipped JavaScript
# reader on DEVELOPMENT sets only (AUDIT section I). Nothing is promoted into
# ../dataset/models.
#
# Run from the brailleocr folder, in a normal terminal (training needs the GPU):
#
#   caffeinate -i bash experiments/retrain_box_shape.sh
#
# Finished steps are skipped, and an interrupted detector run resumes, so re-running the
# same command after a stop continues where it left off. Everything goes to $OUT (default
# ../retrain/box-shape, which is outside the repository and survives reboots), including
# run.log.
#
# Settings (environment variables): OUT, SHAPES, EPOCHS_DET, EPOCHS_CLF, SMOKE=1 (tiny run
# that only checks every step works), PY and PY_EXPORT (how to start Python).
set -euo pipefail

SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
cd "$(dirname "$SELF")/.."
REPO="$PWD"
OUT="${OUT:-$REPO/../retrain/box-shape}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
SHAPES="${SHAPES:-wide narrow}"
EPOCHS_DET="${EPOCHS_DET:-100}"
EPOCHS_CLF="${EPOCHS_CLF:-30}"
PY="${PY:-pixi run python}"
PY_EXPORT="${PY_EXPORT:-pixi run -e export python}"
DET_EXTRA=()
CLF_EXTRA=()
EVAL_EXTRA=()
if [ "${SMOKE:-0}" = 1 ]; then
    EPOCHS_DET=1; EPOCHS_CLF=1
    DET_EXTRA=(--fraction 0.01 --batch 4)
    CLF_EXTRA=(--limit 200 --workers 0)
    EVAL_EXTRA=(--limit 2)
fi

# Run the rest through tee so that everything also lands in run.log; pipefail keeps the
# exit status of the work, not of tee.
if [ -z "${RETRAIN_LOGGING:-}" ]; then
    OUT="$OUT" RETRAIN_LOGGING=1 bash "$SELF" "$@" 2>&1 | tee -a "$OUT/run.log"
    exit
fi
echo "=== $(date) :: shapes: $SHAPES, output: $OUT, smoke: ${SMOKE:-0}"
git -C "$REPO" log -1 --format='code: %h %s' ; git -C "$REPO" status --short | sed 's/^/  uncommitted: /' | head -40

step() {   # step <marker> <description> <command...>: run once, remember success
    local marker="$OUT/done/$1"; shift
    local what="$1"; shift
    if [ -e "$marker" ]; then echo "--- skip (done): $what"; return; fi
    echo "--- $(date +%H:%M:%S) $what"
    "$@"
    mkdir -p "$(dirname "$marker")"; touch "$marker"
}

for S in $SHAPES; do
    # A half-built data folder must not be reused, so it is rebuilt unless marked done.
    [ -e "$OUT/done/$S.yolo-data" ] || rm -rf "$OUT/yolo-$S"
    step "$S.yolo-data" "$S: detector tiles" \
        $PY prepare_yolo_dataset.py --single-class --box-shape "$S" --out "$OUT/yolo-$S"
    step "$S.detector" "$S: train detector ($EPOCHS_DET epochs)" \
        $PY train_detector.py --data "$OUT/yolo-$S/dataset.yaml" --project "$OUT/detector" --name "$S" \
            --epochs "$EPOCHS_DET" --resume --eval-split val ${DET_EXTRA[@]+"${DET_EXTRA[@]}"}
    [ -e "$OUT/done/$S.crops" ] || rm -rf "$OUT/crops-$S"
    step "$S.crops" "$S: classifier crops" \
        $PY extract_crops.py --sources angelina,dsbi --box-shape "$S" --out "$OUT/crops-$S"
    step "$S.classifier" "$S: train classifier (up to $EPOCHS_CLF epochs)" \
        $PY train_classifier.py --crop-dir "$OUT/crops-$S" --epochs "$EPOCHS_CLF" --skip-test \
            ${CLF_EXTRA[@]+"${CLF_EXTRA[@]}"}
    step "$S.onnx" "$S: export ONNX" \
        $PY_EXPORT export_onnx.py --detector "$OUT/detector/$S/weights/best.pt" \
            --classifier "$OUT/crops-$S/cell_classifier.pt" --out-dir "$OUT/onnx-$S"
    step "$S.manifest" "$S: evaluation manifest" \
        $PY experiments/export_eval_manifest.py --box-shape "$S" \
            --splits-out "$OUT/splits-$S.json" --cells-out "$OUT/cells-$S.json"
    step "$S.eval-dev" "$S: score on development sets" \
        node js/eval/run.mjs --role dev --cells "$OUT/cells-$S.json" --out "$OUT/eval-dev-$S.json" \
            --detector "$OUT/onnx-$S/cell_detector.onnx" --classifier "$OUT/onnx-$S/cell_classifier.onnx" \
            ${EVAL_EXTRA[@]+"${EVAL_EXTRA[@]}"}
done

if [ -e "$OUT/eval-dev-wide.json" ] && [ -e "$OUT/eval-dev-narrow.json" ]; then
    echo "--- comparison on development sets"
    $PY experiments/compare_results.py "$OUT/eval-dev-wide.json" "$OUT/eval-dev-narrow.json" \
        --names wide narrow | tee "$OUT/comparison-dev.txt"
fi
echo "=== $(date) finished"
