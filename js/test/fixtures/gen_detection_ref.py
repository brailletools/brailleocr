#!/usr/bin/env python3
"""
Regenerates classifier_ref.json from a real photo, using the SAME
tiling + scale-normalization logic pipeline.py's process_container() uses
(process_container's normalize_scale branch, pipeline.py:1148-1161) --
but deliberately WITHOUT contrast search or the recovery heuristics
(grid_fill/crop_recover/gap_pixel_recover), which are out of scope for the
JS port this fixture backs (see brailleocr/js/src/tiling.js).

The previous version of this fixture was generated from a single whole-page
detector.detect() call (no tiling) -- which is exactly the out-of-distribution
scale process_container's own docstring warns against (pipeline.py:1134-1141).
That produced boxes ~3x oversized (confirmed by cropping the source photo at
a reference box's exact bounds and comparing against pipeline.py's own
save_annotated() output at the same spot). This script fixes that at the
source.

Run from the brailleocr/ repo root:
    pixi run -e dev python js/test/fixtures/gen_detection_ref.py
"""
import json
import statistics
import sys
from pathlib import Path

# pipeline.py/dot_pattern_utils.py live at the brailleocr repo root, three
# levels up from js/test/fixtures/ -- add it to sys.path so this script can
# be run directly (`python js/test/fixtures/gen_detection_ref.py` from the
# repo root) without needing to be installed as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import PIL.Image
import PIL.ImageOps
from ultralytics import YOLO

from dot_pattern_utils import REPOS_ROOT, TILE_SIZE, TARGET_CELL_PX, MIN_NATIVE_TILE
from pipeline import (
    run_detection_tiled, load_cell_classifier, reclassify_cells,
    HIGH_CONF, CONTAINER_MIN_CANDIDATES,
)

SAMPLE_IMAGE = REPOS_ROOT / 'dataset' / 'data' / 'sample-images' / 'IMG_3153.jpeg'
DETECTOR_PATH = REPOS_ROOT / 'dataset' / 'models' / 'cell_detector.pt'
CLASSIFIER_PATH = REPOS_ROOT / 'dataset' / 'models' / 'cell_classifier.pt'
OUT_PATH = Path(__file__).parent / 'classifier_ref.json'
MAX_DET = 2000


def detect_scale_normalized(img, model):
    """Direct port of process_container's normalize_scale branch
    (pipeline.py:1148-1161) -- tiling only, no contrast search, no recovery."""
    initial_cells = run_detection_tiled(img, model, MAX_DET, TILE_SIZE)
    if len(initial_cells) < CONTAINER_MIN_CANDIDATES:
        raise RuntimeError(f'only {len(initial_cells)} candidates found -- too few to size a tile from')

    initial_hi = [c for c in initial_cells if c['conf'] >= HIGH_CONF]
    size_sample = initial_hi if len(initial_hi) >= CONTAINER_MIN_CANDIDATES else initial_cells
    median_w = statistics.median(c['w'] for c in size_sample)
    native_tile_size = max(MIN_NATIVE_TILE, round(median_w * TILE_SIZE / TARGET_CELL_PX))

    if native_tile_size / TILE_SIZE > 1.5 or native_tile_size / TILE_SIZE < 0.67:
        print(f'  refining native tile size: {native_tile_size}px (median cell width {median_w:.1f}px)')
        return run_detection_tiled(img, model, MAX_DET, native_tile_size)
    return initial_cells


def main():
    print(f'Loading {SAMPLE_IMAGE.name} ...')
    img = PIL.ImageOps.exif_transpose(PIL.Image.open(SAMPLE_IMAGE)).convert('RGB')

    print(f'Loading detector from {DETECTOR_PATH} ...')
    model = YOLO(str(DETECTOR_PATH))

    print('Running scale-normalized tiled detection (no contrast search, no recovery)...')
    cells = detect_scale_normalized(img, model)
    print(f'  {len(cells)} cells detected')

    print(f'Loading classifier from {CLASSIFIER_PATH} ...')
    clf, clf_device, clf_tf = load_cell_classifier(str(CLASSIFIER_PATH))
    cells = reclassify_cells(img, cells, clf, clf_device, clf_tf)

    out = {
        'imgW': img.width,
        'imgH': img.height,
        'cells': [
            {'cx': c['cx'], 'cy': c['cy'], 'w': c['w'], 'h': c['h'], 'bits': c['bits']}
            for c in cells
        ],
    }
    OUT_PATH.write_text(json.dumps(out, indent=2))
    print(f'Wrote {OUT_PATH} ({len(cells)} cells)')

    widths = [c['w'] for c in cells]
    heights = [c['h'] for c in cells]
    print(f'  median cell size: {statistics.median(widths):.1f} x {statistics.median(heights):.1f}px'
          f'  (sanity check: should be close to TARGET_CELL_PX-scaled native size, not ~3x that)')


if __name__ == '__main__':
    main()
