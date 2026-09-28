#!/usr/bin/env python3
"""
Stage-level throughput benchmark for the OCR pipeline.

Rather than re-implementing the pipeline stage by stage (which drifts the
moment pipeline.py changes), this wraps the real functions in place and runs
the real process_container(). Every stage is timed where it actually executes,
including the ones that only run conditionally.

Timing is EXCLUSIVE: when a wrapped stage calls another wrapped stage, the
child's time is subtracted from the parent. That matters here because most
"recovery" stages spend their time inside run_detection() — reported
inclusively, a recovery stage and detection would each claim the same seconds,
and the shares would sum to well over 100%.

Cost splits the way it actually scales:
  FIXED per image   — decode, container detection. Scales with resolution,
                      paid once however much Braille is on the page.
  PER TILE          — detector forward passes. Scales with image area.
  PER CELL          — classifier passes (batched), layout, translation.

`detector calls` is reported per stage because that, not the stage's own
logic, is what a recovery stage actually costs.

Usage:
  python bench.py                        # sample images, 3 repeats
  python bench.py path/to/imgs --repeats 5
  python bench.py --contrast-search      # include the (expensive) contrast pass
  python bench.py --json /tmp/bench.json
"""

import argparse
import functools
import json
import random
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

import PIL.Image
import PIL.ImageOps
from ultralytics import YOLO

import pipeline
from container_detect import find_containers
from dot_pattern_utils import REPOS_ROOT, make_tile_boxes, TILE_SIZE
from model_fetch import resolve_model, resolve_classifier_path

SAMPLE_DIR = REPOS_ROOT / 'dataset' / 'data' / 'sample-images'
IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
DEFAULT_MAX_DET = 2000   # matches process_container()'s default

# Stages to instrument, in pipeline order. Names are attributes of the
# `pipeline` module; process_container() calls them as module-level globals,
# so rebinding the attribute is enough to intercept them.
STAGES = [
    'best_contrast',
    'run_detection_tiled',
    'run_detection',
    'grid_fill',
    'gap_pixel_recover',
    'reclassify_cells',
    'save_annotated',
    'group_into_lines',
    'insert_spaces',
    'save_dot_grid',
    'braille_to_text',
    'clean_translation',
]


class StageTimer:
    """
    Wraps functions to accumulate exclusive wall-clock time and call counts.

    Exclusive time uses a stack: each active call gets an accumulator that its
    wrapped children add their *inclusive* time to, and the parent subtracts
    that before recording its own.
    """

    def __init__(self):
        self.calls = {}
        self.exclusive = {}
        self.inclusive = {}
        self.forward = {}     # detector forward passes, by innermost stage
        self._stack = []
        self._names = []

    def reset(self):
        self.calls.clear()
        self.exclusive.clear()
        self.inclusive.clear()
        self.forward.clear()
        self._stack.clear()
        self._names.clear()

    def note_forward(self):
        """Record one detector forward pass against the stage running now."""
        name = self._names[-1] if self._names else '(outside any stage)'
        self.forward[name] = self.forward.get(name, 0) + 1

    def wrap(self, name, fn):
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            self._stack.append(0.0)
            self._names.append(name)
            t0 = time.perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed = time.perf_counter() - t0
                child = self._stack.pop()
                self._names.pop()
                self.calls[name] = self.calls.get(name, 0) + 1
                self.exclusive[name] = self.exclusive.get(name, 0.0) + (elapsed - child)
                self.inclusive[name] = self.inclusive.get(name, 0.0) + elapsed
                if self._stack:
                    self._stack[-1] += elapsed
        return wrapped


@contextmanager
def instrumented(timer, names):
    """Temporarily rebind pipeline.<name> to a timed wrapper."""
    originals = {n: getattr(pipeline, n) for n in names}
    for n, fn in originals.items():
        setattr(pipeline, n, timer.wrap(n, fn))
    try:
        yield
    finally:
        for n, fn in originals.items():
            setattr(pipeline, n, fn)


@contextmanager
def timed(sink, key):
    t0 = time.perf_counter()
    yield
    sink[key] = time.perf_counter() - t0


class CountingModel:
    """
    Counts every detector forward pass, wherever it is issued from.

    Counting run_detection() calls is NOT sufficient: pipeline.py invokes the
    YOLO model directly from several call sites (best_contrast,
    low_conf_repass, gap_pixel_recover and run_detection), so a
    run_detection-based count silently under-reports forward passes issued by
    recovery stages.
    """

    def __init__(self, model, timer):
        self._model = model
        self._timer = timer

    def __call__(self, *args, **kwargs):
        self._timer.note_forward()
        return self._model(*args, **kwargs)

    def __getattr__(self, item):
        # model.names etc. must still resolve — pipeline reads it to map
        # class indices to dot patterns.
        return getattr(self._model, item)


def bench_image(path, model, classifier_path, args, timer):
    """Run the full pipeline `repeats` times on one image; return timings."""
    runs = []
    meta = {'megapixels': 0.0, 'tiles': 0, 'detector_calls': 0}

    for _ in range(args.repeats):
        timer.reset()
        fixed = {}

        with timed(fixed, 'decode'):
            img = PIL.Image.open(path)
            img = PIL.ImageOps.exif_transpose(img).convert('RGB')
            img.load()          # PIL is lazy; force the decode inside the timer

        meta['megapixels'] = (img.width * img.height) / 1e6
        meta['tiles'] = len(make_tile_boxes(img.width, img.height, TILE_SIZE))

        with timed(fixed, 'find_containers'):
            find_containers(img)

        # The whole image is treated as one container: container *selection*
        # is a pipeline.main() concern, and benchmarking n containers per photo
        # would make per-image numbers depend on how many candidates a photo
        # happens to yield.
        with instrumented(timer, STAGES):
            t0 = time.perf_counter()
            pipeline.process_container(
                img, f'bench_{path.stem}', CountingModel(model, timer),
                lang_table=args.lang,
                search_contrast=args.contrast_search,
                spellcheck=not args.no_spellcheck,
                max_det=args.max_det,
                classifier_path=classifier_path,
            )
            container_total = time.perf_counter() - t0

        stages = dict(timer.exclusive)
        stages.update(fixed)
        # Time inside process_container not attributed to any wrapped stage
        # (margin filter, statistics, glue code).
        stages['unattributed'] = container_total - sum(timer.exclusive.values())
        stages['total'] = sum(fixed.values()) + container_total
        meta['detector_calls'] = sum(timer.forward.values())
        runs.append(stages)

    keys = sorted({k for r in runs for k in r})
    return {
        'path': str(path),
        **meta,
        'stages': {k: statistics.median([r.get(k, 0.0) for r in runs]) for k in keys},
        'calls': dict(timer.calls),
        'forward': dict(timer.forward),
    }


def bootstrap_ci(per_image_values, resamples, alpha=0.05, seed=0):
    """
    Percentile bootstrap CI for the mean across IMAGES.

    Images are the unit of resampling, not repeats: repeats measure machine
    noise on one input, which is small and not what we're uncertain about.
    The uncertainty that matters is across *inputs* — a recovery stage's cost
    depends on how many cells the first pass missed, which varies per photo —
    so a CI built from repeats would be confidently wrong.

    With few images the interval is wide and lumpy by construction. That is
    the honest answer, not a defect: it says how many images a run needs
    before a difference is detectable.
    """
    if len(per_image_values) < 2:
        return None
    rng = random.Random(seed)
    n = len(per_image_values)
    means = []
    for _ in range(resamples):
        sample = [per_image_values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(alpha / 2 * resamples)]
    hi = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    return lo, hi


def fmt_ms(seconds):
    return f'{seconds * 1000:.1f} ms'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('images', nargs='?', default=str(SAMPLE_DIR))
    ap.add_argument('--repeats', type=int, default=3)
    ap.add_argument('--warmup', type=int, default=1,
                    help='untimed runs first — the first forward pass pays lazy '
                         'CUDA/MPS kernel compilation that would otherwise land '
                         'entirely on image #1')
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--max-det', type=int, default=DEFAULT_MAX_DET)
    ap.add_argument('--lang', default='en-ueb-g2.ctb')
    ap.add_argument('--contrast-search', action='store_true',
                    help='include best_contrast (off by default: it re-runs '
                         'detection once per contrast level and dominates)')
    ap.add_argument('--no-spellcheck', action='store_true')
    ap.add_argument('--no-classifier', action='store_true')
    ap.add_argument('--bootstrap', type=int, default=0, metavar='N',
                    help='resample images N times (e.g. 2000) and report 95%% CIs '
                         'on per-stage cost; needs at least 2 images')
    ap.add_argument('--json')
    args = ap.parse_args()

    src = Path(args.images)
    paths = sorted(p for p in ([src] if src.is_file() else src.iterdir())
                   if p.suffix.lower() in IMG_EXTS)
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f'No images found in {src}')

    load = {}
    with timed(load, 'detector'):
        model = YOLO(str(resolve_model('cell_detector.pt')))
    classifier_path = None if args.no_classifier else resolve_classifier_path()

    print(f'Images:     {len(paths)}  (repeats={args.repeats}, warmup={args.warmup})')
    print(f'Detector:   {fmt_ms(load["detector"])} to load, max_det={args.max_det}')
    print(f'Classifier: {"off" if args.no_classifier else classifier_path}')
    print(f'Contrast search: {"on" if args.contrast_search else "off"}')

    # pipeline.main() normally creates this; save_annotated/save_dot_grid
    # write into it and we time those stages like any other.
    pipeline.OUT_DIR.mkdir(parents=True, exist_ok=True)

    warm = PIL.Image.open(paths[0]).convert('RGB')
    for _ in range(args.warmup):
        pipeline.run_detection_tiled(warm, model, args.max_det, TILE_SIZE)

    timer = StageTimer()
    results = [bench_image(p, model, classifier_path, args, timer) for p in paths]

    n = len(results)
    all_keys = sorted({k for r in results for k in r['stages']},
                      key=lambda k: -sum(r['stages'].get(k, 0) for r in results))
    grand = sum(r['stages']['total'] for r in results)

    print(f'\n{"stage":<22}{"ms/image":>11}{"share":>9}{"calls/image":>13}{"fwd/image":>11}')
    print('-' * 66)
    for k in all_keys:
        if k == 'total':
            continue
        secs = sum(r['stages'].get(k, 0.0) for r in results)
        calls = sum(r['calls'].get(k, 0) for r in results) / n
        share = secs / grand * 100
        fwd = sum(r.get('forward', {}).get(k, 0) for r in results) / n
        flag = '  ←' if share >= 10 else ''
        print(f'{k:<22}{secs / n * 1000:10.1f}{share:8.1f}%{calls:13.1f}{fwd:11.1f}{flag}')
    print('-' * 66)
    print(f'{"TOTAL":<22}{grand / n * 1000:10.1f}{100.0:8.1f}%')

    if args.bootstrap:
        print(f'\n── 95% CI over images (bootstrap, {args.bootstrap} resamples, '
              f'n={n} images) ──')
        if n < 2:
            print('  Need at least 2 images; skipped.')
        else:
            print(f'{"stage":<22}{"mean ms":>10}{"95% CI":>26}{"width":>9}')
            print('-' * 67)
            for k in all_keys[:8]:
                vals = [r['stages'].get(k, 0.0) * 1000 for r in results]
                ci = bootstrap_ci(vals, args.bootstrap)
                mean = sum(vals) / n
                width = (ci[1] - ci[0]) / mean * 100 if mean else 0
                print(f'{k:<22}{mean:10.1f}{f"[{ci[0]:.0f}, {ci[1]:.0f}]":>26}'
                      f'{width:8.0f}%')
            print('\n  CI width as % of the mean is the number to read: anything '
                  'near or above\n  100% means this sample cannot distinguish '
                  'that stage\'s cost from half or\n  double its measured value.')

    tot_tiles = sum(r['tiles'] for r in results)
    tot_mp = sum(r['megapixels'] for r in results)
    tot_fwd = sum(r['detector_calls'] for r in results)

    print('\n── Throughput ' + '─' * 41)
    print(f'End to end:     {n / grand:8.2f} images/s   ({fmt_ms(grand / n)}/image)')
    print(f'Pixels:         {tot_mp / grand:8.2f} MP/s')
    print(f'Detector calls: {tot_fwd / n:8.1f} forward passes/image '
          f'({tot_tiles / n:.0f} of them tile passes at TILE_SIZE)')

    if args.json:
        Path(args.json).write_text(json.dumps(
            {'config': vars(args), 'model_load_s': load, 'images': results}, indent=2))
        print(f'\nWrote {args.json}')


if __name__ == '__main__':
    main()
