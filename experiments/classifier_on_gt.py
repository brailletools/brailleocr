#!/usr/bin/env python3
"""
Classifier accuracy on GROUND-TRUTH crops — the published-comparison setup.

Most Braille character-recognition papers evaluate a classifier on cells that
were already segmented for it, and report per-character accuracy (roughly
95-99% in the ones surveyed in RESEARCH.md). Our end-to-end number is measured
on crops the DETECTOR produced, so it carries any framing error the detector
introduces. That makes the two numbers not quite comparable, and it hides which
stage a gap belongs to.

This runs the classifier on crops cut from ground-truth boxes, so the only thing
being measured is the classifier — the same quantity those papers report.

Splits the result three ways, because two properties of DSBI distort a single
average:
  - recto vs verso: a verso dot is a depression with an inverted shadow, and
    single-sided benchmarks contain none of them.
  - per dot vs per cell: a cell is six binary decisions (see RESEARCH.md).

Comparing this number against the end-to-end one (RESEARCH.md "Current
accuracy") separates classifier error from detector-framing error:
  gap here          → the classifier itself
  gap end-to-end only → the detector's boxes are imprecise even when "correct"

Usage:
  python experiments/classifier_on_gt.py --limit 4
"""

import argparse
import json
import sys
from pathlib import Path

import PIL.Image
import PIL.ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline                                      # noqa: E402
from dsbi_data import collect_images, load_txt       # noqa: E402
from model_fetch import resolve_classifier_path      # noqa: E402

PAD = 0.10   # matches pipeline._CLASSIFIER_CROP_PAD and extract_crops.py


def pct(a, b):
    return f'{a}/{b} ({a / b * 100:.1f}%)' if b else f'{a}/0 (n/a)'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--limit', type=int, default=4,
                    help='physical pages; 0 = all 88 DSBI test pages')
    ap.add_argument('--classifier', default=None,
                    help='path to a cell_classifier.pt (default: resolved copy)')
    ap.add_argument('--batch', type=int, default=256)
    ap.add_argument('--json')
    args = ap.parse_args()

    import torch

    pages = {}
    for jpg, txt, split in collect_images():
        if split == 'test':
            pages.setdefault(jpg.name.rsplit('+', 1)[0], [jpg, []])[1].append(txt)
    entries = list(pages.values())[:args.limit or None]
    if not entries:
        raise SystemExit('No DSBI test-split images found.')

    clf_path = args.classifier or resolve_classifier_path()
    print(f'  classifier: {clf_path}')
    clf, dev, tf = pipeline.load_cell_classifier(clf_path)
    tot = {k: 0 for k in ('cells', 'cell_ok', 'dots', 'dot_ok',
                          'recto', 'recto_ok', 'verso', 'verso_ok', 'off_by_one')}
    per_page = []

    for jpg, txts in entries:
        img = PIL.ImageOps.exif_transpose(PIL.Image.open(jpg)).convert('RGB')
        cells = []
        for txt in txts:
            side = 'verso' if '+verso' in txt.name else 'recto'
            for c in load_txt(txt, img.width, img.height):
                left, top, right, bottom = c['frac_bbox']
                cells.append({'box': (left * img.width, top * img.height,
                                      right * img.width, bottom * img.height),
                              'bits6': c['bits6'], 'side': side})
        if not cells:
            continue

        crops = []
        for c in cells:
            x0, y0, x1, y1 = c['box']
            px, py = (x1 - x0) * PAD, (y1 - y0) * PAD
            crops.append(tf(img.crop((max(0, x0 - px), max(0, y0 - py),
                                      min(img.width, x1 + px),
                                      min(img.height, y1 + py))).convert('RGB')))

        preds = []
        for i in range(0, len(crops), args.batch):
            with torch.no_grad():
                logits = clf(torch.stack(crops[i:i + args.batch]).to(dev))
            preds.extend((logits.sigmoid() > 0.5).cpu().numpy().astype(int))

        page_ok = 0
        for c, row in zip(cells, preds):
            bits = ''.join(str(b) for b in row)
            ok = bits == c['bits6']
            agree = sum(1 for a, b in zip(bits, c['bits6']) if a == b)
            tot['cells'] += 1
            tot['dots'] += 6
            tot['dot_ok'] += agree
            tot['cell_ok'] += ok
            if agree == 5:
                tot['off_by_one'] += 1
            tot[c['side']] += 1
            tot[f'{c["side"]}_ok'] += ok
            page_ok += ok
        per_page.append({'image': jpg.name, 'cells': len(cells), 'correct': page_ok})
        print(f'  {jpg.name}: {pct(page_ok, len(cells))}')

    print('\n── Classifier on ground-truth crops ' + '─' * 20)
    print(f'  per-CELL accuracy   {pct(tot["cell_ok"], tot["cells"])}')
    print(f'  per-DOT accuracy    {pct(tot["dot_ok"], tot["dots"])}')
    print(f'  recto only          {pct(tot["recto_ok"], tot["recto"])}'
          '   ← closest to a single-sided benchmark')
    print(f'  verso only          {pct(tot["verso_ok"], tot["verso"])}')
    wrong = tot['cells'] - tot['cell_ok']
    print(f'  wrong by one dot    {pct(tot["off_by_one"], wrong)} of wrong cells')

    print('\n  Compare with end-to-end on detector crops (RESEARCH.md '
          '"Current accuracy"):\n  89.0% per cell, 97.8% per dot, recto 94.3%, '
          'verso 83.5%.\n  A higher number here means the detector\'s framing '
          'costs accuracy even when\n  the cell is correctly located; a similar '
          'number means the classifier is the limit.')

    if args.json:
        Path(args.json).write_text(json.dumps(
            {'totals': tot, 'per_page': per_page}, indent=2))
        print(f'\nWrote {args.json}')


if __name__ == '__main__':
    main()
