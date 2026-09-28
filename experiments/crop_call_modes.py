#!/usr/bin/env python3
"""
Experiments 2 and 3: how crop_recover() should issue its forward passes.

crop_recover() currently does, per position per contrast level:
    save a JPEG to a temp dir  →  model(path_string)

which pays a JPEG encode, a file write, and a file read+decode inside
ultralytics, and issues one single-image forward pass. With ~75 positions x 5
contrast levels that is ~375 round trips per image.

This compares three call modes on real crops taken from a real image, at the
size crop_recover() actually uses:

  path    — current behaviour: temp JPEG, pass the path          (baseline)
  array   — pass the decoded numpy array directly (experiment 3)
  batch-N — pass a list of N arrays in one call    (experiment 2)

Reports ms per crop for each, so the two changes can be costed separately:
array-vs-path isolates the disk round trip, batch-vs-array isolates
per-call overhead.

Accuracy is unaffected by construction — same pixels, same model, same
thresholds; only the transport and grouping change. That is checked here by
comparing detection counts across modes, and must be re-checked end to end
with evaluate.py before any of this lands in pipeline.py.

Usage:
  python experiments/crop_call_modes.py [image] [--crops 64] [--repeats 3]
"""

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import PIL.Image
import PIL.ImageEnhance
import PIL.ImageOps
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dot_pattern_utils import REPOS_ROOT          # noqa: E402
from model_fetch import resolve_model             # noqa: E402
from pipeline import CROP_CONF                    # noqa: E402

SAMPLE_DIR = REPOS_ROOT / 'dataset' / 'data' / 'sample-images'
TARGET_CELL_PX_IN_CROP = 96.0   # crop_recover() upscales so the cell is ~96px
MAX_DET = 20                    # crop_recover()'s value


def make_crops(img, n, cell_w):
    """
    Build n crops shaped like crop_recover()'s: 2 cells of horizontal context,
    1.5 vertically, upscaled so the target cell is ~96px. Positions are spread
    over the image — the point is realistic crop *size*, not real empties.
    """
    pad_x, pad_y = cell_w * 2.0, cell_w * 1.5 * 1.4
    scale = max(1.0, TARGET_CELL_PX_IN_CROP / cell_w)
    crops = []
    rng = np.random.default_rng(0)      # fixed seed: same crops across modes
    for _ in range(n):
        cx = rng.uniform(pad_x, img.width - pad_x)
        cy = rng.uniform(pad_y, img.height - pad_y)
        box = (int(cx - pad_x), int(cy - pad_y), int(cx + pad_x), int(cy + pad_y))
        crop = img.crop(box)
        crops.append(crop.resize((int(crop.width * scale), int(crop.height * scale)),
                                 PIL.Image.LANCZOS))
    return crops


def run_path(model, crops):
    """Current behaviour: encode each crop to a temp JPEG, pass the path."""
    n_boxes = 0
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir) / 'crop.jpg'
        for crop in crops:
            crop.save(tmp, quality=95)
            r = model(str(tmp), verbose=False, conf=CROP_CONF, max_det=MAX_DET)
            n_boxes += 0 if r[0].boxes is None else len(r[0].boxes)
    return n_boxes


def run_array(model, arrays):
    """Experiment 3: hand the decoded array straight to the model."""
    n_boxes = 0
    for arr in arrays:
        r = model(arr, verbose=False, conf=CROP_CONF, max_det=MAX_DET)
        n_boxes += 0 if r[0].boxes is None else len(r[0].boxes)
    return n_boxes


def run_batch(model, arrays, batch):
    """Experiment 2: one call per batch of arrays."""
    n_boxes = 0
    for i in range(0, len(arrays), batch):
        chunk = arrays[i:i + batch]
        for r in model(chunk, verbose=False, conf=CROP_CONF, max_det=MAX_DET):
            n_boxes += 0 if r.boxes is None else len(r.boxes)
    return n_boxes


def timeit(fn, repeats):
    times = []
    boxes = None
    for _ in range(repeats):
        t0 = time.perf_counter()
        boxes = fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times), boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('image', nargs='?')
    ap.add_argument('--crops', type=int, default=64)
    ap.add_argument('--cell-width', type=float, default=130.0,
                    help='native cell width in px; 12MP page photos measure ~130')
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--batches', default='4,8,16,32,64')
    args = ap.parse_args()

    path = Path(args.image) if args.image else sorted(
        p for p in SAMPLE_DIR.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg'})[0]
    img = PIL.ImageOps.exif_transpose(PIL.Image.open(path)).convert('RGB')
    model = YOLO(str(resolve_model('cell_detector.pt')))

    crops = make_crops(img, args.crops, args.cell_width)
    arrays = [np.asarray(c) for c in crops]
    print(f'Image:  {path.name} ({img.width}x{img.height})')
    print(f'Crops:  {len(crops)} at {crops[0].width}x{crops[0].height} px, '
          f'repeats={args.repeats}')

    model(arrays[0], verbose=False, conf=CROP_CONF, max_det=MAX_DET)  # warm up

    results = {}
    results['path'] = timeit(lambda: run_path(model, crops), args.repeats)
    results['array'] = timeit(lambda: run_array(model, arrays), args.repeats)
    for b in (int(x) for x in args.batches.split(',')):
        results[f'batch-{b}'] = timeit(lambda b=b: run_batch(model, arrays, b),
                                       args.repeats)

    base = results['path'][0]
    print(f'\n{"mode":<12}{"total":>10}{"ms/crop":>11}{"speedup":>10}{"boxes":>9}')
    print('-' * 52)
    for mode, (secs, boxes) in results.items():
        print(f'{mode:<12}{secs:9.2f}s{secs / len(crops) * 1000:10.1f}'
              f'{base / secs:9.2f}x{boxes:9d}')

    counts = {boxes for _, boxes in results.values()}
    print('\nDetection counts identical across modes: '
          + ('yes' if len(counts) == 1 else f'NO — {sorted(counts)}'))
    if len(counts) > 1:
        print('  A count difference means transport is changing results '
              '(JPEG quantisation, resize path, or batch letterboxing) —\n'
              '  investigate before adopting, it is not a free win.')


if __name__ == '__main__':
    main()
