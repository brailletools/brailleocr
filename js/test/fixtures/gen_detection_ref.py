#!/usr/bin/env python3
"""
Regenerates classifier_ref.json from a real photo, using reader.py's scale-normalized
tiled detection (the Python counterpart of js/src/tiling.js) with no contrast search and no
recovery heuristics, which the JS port does not have. The JS tests compare their own output
against this file.

History: an earlier version of this fixture was generated from a single whole-page detection
(no tiling), which is exactly the out-of-distribution scale the tiling exists to avoid, and
produced boxes about 3x too large. This script fixes that at the source.

Run from the brailleocr/ repo root:
    pixi run -e dev python js/test/fixtures/gen_detection_ref.py
"""
import json
import statistics
import sys
from pathlib import Path

# reader.py/dot_pattern_utils.py live at the brailleocr repo root, three
# levels up from js/test/fixtures/ -- add it to sys.path so this script can
# be run directly (`python js/test/fixtures/gen_detection_ref.py` from the
# repo root) without needing to be installed as a package.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import PIL.Image
import PIL.ImageOps
from ultralytics import YOLO

from dot_pattern_utils import REPOS_ROOT
from reader import detect_scale_normalized, load_cell_classifier, reclassify_cells

SAMPLE_IMAGE = REPOS_ROOT / 'dataset' / 'data' / 'sample-images' / 'IMG_3153.jpeg'
DETECTOR_PATH = REPOS_ROOT / 'dataset' / 'models' / 'cell_detector.pt'
CLASSIFIER_PATH = REPOS_ROOT / 'dataset' / 'models' / 'cell_classifier.pt'
OUT_PATH = Path(__file__).parent / 'classifier_ref.json'
MAX_DET = 2000


def main():
    print(f'Loading {SAMPLE_IMAGE.name} ...')
    img = PIL.ImageOps.exif_transpose(PIL.Image.open(SAMPLE_IMAGE)).convert('RGB')

    print(f'Loading detector from {DETECTOR_PATH} ...')
    model = YOLO(str(DETECTOR_PATH))

    print('Running scale-normalized tiled detection (no contrast search, no recovery)...')
    cells = detect_scale_normalized(img, model, MAX_DET)
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
