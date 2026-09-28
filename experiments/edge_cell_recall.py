#!/usr/bin/env python3
"""
Edge-cell recall and recovery precision on labeled pages.

Experiment 1 showed crop_recover() accepts cells only at noise confidence
(max 0.0063 over 224 positions), but it could not say whether those cells were
REAL — the sample images have no ground truth. This runs the same pipeline on
labeled DSBI pages and answers the two questions that decide the stage's fate:

  1. Do we actually miss cells at line ends? (edge recall vs interior recall
     after plain detection) — i.e. is there signal there to recover at all.
  2. Of the cells each recovery stage adds, how many correspond to a real
     ground-truth cell, and how many are invented? (recovery precision)

"Edge" = a ground-truth cell among the first or last --edge-k positions of its
line; "interior" = everything else. Lines are grouped from the ground truth,
not from predictions, so a missed cell can't quietly redefine where the line
ends.

Held-out data only: DSBI's own test.txt split, since the detector was trained
on its train split.

IMPORTANT — DSBI page structure: <page>+recto.jpg and <page>+verso.jpg are the
SAME scan, byte for byte (verified by md5), and each side's .txt labels only
that side's cells. Scoring per side-file therefore (a) counts real cells from
the other side as false positives, and (b) runs each physical page twice. This
script groups by physical page and matches against the UNION of both sides'
labels. The detector cannot tell recto dots from verso dots — it finds both —
so the union is the honest ground truth for "is there a cell here".

Matching follows evaluate.py: nearest predicted cell within 0.5 x median
ground-truth cell width.

Usage:
  python experiments/edge_cell_recall.py --limit 8
  python experiments/edge_cell_recall.py --limit 20 --json out.json
"""

import argparse
import json
import statistics
import sys
from pathlib import Path

import PIL.Image
import PIL.ImageOps
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline                                                 # noqa: E402
from dsbi_data import collect_images, load_txt                  # noqa: E402
from model_fetch import resolve_model, resolve_classifier_path  # noqa: E402

ROW_BAND_FRAC = 0.55   # matches group_into_lines()
MATCH_FRAC = 0.5       # matches evaluate.py


def gt_cells(txt_path, img_w, img_h):
    """Ground-truth cells in pixel centre coordinates, for ONE side's label file."""
    out = []
    for c in load_txt(txt_path, img_w, img_h):
        left, top, right, bottom = c['frac_bbox']
        out.append({
            'cx': (left + right) / 2 * img_w,
            'cy': (top + bottom) / 2 * img_h,
            'w': (right - left) * img_w,
            'bits6': c['bits6'],
        })
    return out


def label_edges(cells, edge_k):
    """Group ground truth into lines and mark the first/last edge_k of each."""
    if not cells:
        return
    band = statistics.median(c['w'] for c in cells) * ROW_BAND_FRAC * 1.5
    rows = []
    for c in sorted(cells, key=lambda c: c['cy']):
        if rows and abs(c['cy'] - rows[-1][-1]['cy']) < band:
            rows[-1].append(c)
        else:
            rows.append([c])
    for row in rows:
        row.sort(key=lambda c: c['cx'])
        for i, c in enumerate(row):
            c['edge'] = i < edge_k or i >= len(row) - edge_k
            c['line_len'] = len(row)


def match(pred, gt, tol):
    """Nearest-centre matching, each ground-truth cell claimed at most once."""
    matched = {}
    for p in pred:
        best, best_d = None, tol
        for i, g in enumerate(gt):
            if i in matched:
                continue
            d = abs(p['cx'] - g['cx']) + abs(p['cy'] - g['cy'])
            if d < best_d:
                best, best_d = i, d
        if best is not None:
            matched[best] = p
    return matched


def pct(a, b):
    return f'{a}/{b} ({a / b * 100:.1f}%)' if b else f'{a}/0 (n/a)'


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--limit', type=int, default=8)
    ap.add_argument('--edge-k', type=int, default=1,
                    help='how many cells at each line end count as "edge" (default 1)')
    ap.add_argument('--max-det', type=int, default=2000)
    ap.add_argument('--json')
    args = ap.parse_args()

    # Group side-files by physical page: 'FM+1+recto.jpg' -> 'FM+1'.
    pages = {}
    for jpg, txt, split in collect_images():
        if split != 'test':
            continue
        pages.setdefault(jpg.name.rsplit('+', 1)[0], [jpg, []])[1].append(txt)
    entries = [(jpg, txts) for jpg, txts in pages.values()][:args.limit]
    if not entries:
        raise SystemExit('No DSBI test-split images found.')
    print(f'{len(entries)} physical pages '
          f'({sum(len(t) for _, t in entries)} side-label files, unioned)')

    model = YOLO(str(resolve_model('cell_detector.pt')))
    pipeline.OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Capture each stage's contribution by wrapping the real functions, so the
    # experiment can't drift from what the pipeline actually runs.
    captured = {}
    originals = {n: getattr(pipeline, n) for n in
                 ('run_detection_tiled', 'grid_fill', 'crop_recover', 'gap_pixel_recover')}

    def wrap_detect(*a, **k):
        cells = originals['run_detection_tiled'](*a, **k)
        captured.setdefault('detected', []).append(cells)
        return cells

    def wrap_grid(*a, **k):
        cells, empties = originals['grid_fill'](*a, **k)
        captured['after_grid'] = cells
        return cells, empties

    def wrap_crop(*a, **k):
        cells = originals['crop_recover'](*a, **k)
        captured['crop_added'] = cells
        return cells

    def wrap_gap(*a, **k):
        cells = originals['gap_pixel_recover'](*a, **k)
        captured['gap_added'] = cells
        return cells

    totals = {k: 0 for k in (
        'gt', 'gt_edge', 'gt_interior',
        'det_edge_hit', 'det_interior_hit',
        'grid_edge_hit', 'crop_added', 'crop_real', 'crop_bits_ok',
        'gap_added', 'gap_real', 'gap_bits_ok')}
    per_image = []

    for jpg, txts in entries:
        img = PIL.ImageOps.exif_transpose(PIL.Image.open(jpg)).convert('RGB')
        truth = [c for txt in txts for c in gt_cells(txt, img.width, img.height)]
        if not truth:
            continue
        label_edges(truth, args.edge_k)
        tol = statistics.median(c['w'] for c in truth) * MATCH_FRAC

        captured.clear()
        for name, fn in (('run_detection_tiled', wrap_detect), ('grid_fill', wrap_grid),
                         ('crop_recover', wrap_crop), ('gap_pixel_recover', wrap_gap)):
            setattr(pipeline, name, fn)
        try:
            pipeline.process_container(
                img, f'edge_{jpg.stem}', model, lang_table='en-ueb-g2.ctb',
                search_contrast=False, spellcheck=False, max_det=args.max_det,
                classifier_path=resolve_classifier_path())
        finally:
            for name, fn in originals.items():
                setattr(pipeline, name, fn)

        # The last tiled-detection call is the one whose output goes forward
        # (an earlier call may be the scale-measuring bootstrap pass).
        detected = captured.get('detected', [[]])[-1]
        edges = [c for c in truth if c['edge']]
        interior = [c for c in truth if not c['edge']]

        det_m = match(detected, truth, tol)
        det_edge = sum(1 for i in det_m if truth[i]['edge'])
        grid_m = match(captured.get('after_grid', []), truth, tol)
        grid_edge = sum(1 for i in grid_m if truth[i]['edge'])

        row = {'image': jpg.name, 'gt': len(truth), 'gt_edge': len(edges),
               'det_edge_hit': det_edge, 'grid_edge_hit': grid_edge}

        for stage, key in (('crop_added', 'crop'), ('gap_added', 'gap')):
            added = captured.get(stage, []) or []
            m = match(added, truth, tol)
            bits_ok = sum(1 for gi, p in m.items()
                          if p.get('bits') == truth[gi]['bits6'])
            totals[f'{key}_added'] += len(added)
            totals[f'{key}_real'] += len(m)
            totals[f'{key}_bits_ok'] += bits_ok
            row[f'{key}_added'] = len(added)
            row[f'{key}_real'] = len(m)

        totals['gt'] += len(truth)
        totals['gt_edge'] += len(edges)
        totals['gt_interior'] += len(interior)
        totals['det_edge_hit'] += det_edge
        totals['det_interior_hit'] += len(det_m) - det_edge
        totals['grid_edge_hit'] += grid_edge
        per_image.append(row)
        print(f'  {jpg.name}: {len(truth)} gt cells, edge recall '
              f'{det_edge}/{len(edges)}, crop_recover added {row["crop_added"]} '
              f'({row["crop_real"]} real)')

    print('\n── Is there signal at line ends? ' + '─' * 24)
    print(f'  interior recall (detection)  {pct(totals["det_interior_hit"], totals["gt_interior"])}')
    print(f'  EDGE recall (detection)      {pct(totals["det_edge_hit"], totals["gt_edge"])}')
    print(f'  EDGE recall (after grid_fill){pct(totals["grid_edge_hit"], totals["gt_edge"])}')

    print('\n── Recovery precision: real vs invented ' + '─' * 17)
    for key, label in (('crop', 'crop_recover'), ('gap', 'gap_pixel_recover')):
        added, real, ok = (totals[f'{key}_added'], totals[f'{key}_real'],
                           totals[f'{key}_bits_ok'])
        print(f'  {label:<20} added {added:4d}   matched a real cell {pct(real, added)}')
        print(f'  {"":<20} of those, correct dot pattern {pct(ok, real)}')

    if args.json:
        Path(args.json).write_text(json.dumps(
            {'totals': totals, 'per_image': per_image, 'edge_k': args.edge_k}, indent=2))
        print(f'\nWrote {args.json}')


if __name__ == '__main__':
    main()
