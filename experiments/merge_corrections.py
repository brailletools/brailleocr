#!/usr/bin/env python3
"""
Merge saved corrections into a labeled dataset our loaders already read.

Ground truth = predictions, with edits applied, deletions removed, additions
added; untouched predictions are confirmed correct by the labeler. This applies
that rule and writes Angelina-format CSV (`l;t;r;b;label_int`, fractional
coordinates, one row per cell) next to each image, so angelina_data.load_csv,
prepare_yolo_dataset.py and extract_crops.py consume it with no new code.

Skips cells still at '000000' (blank) — in a braille_natural page seeded from
human boxes, a cell left blank means "not yet labeled", not "an empty cell",
and writing those as real labels would poison the training set. The count of
skipped cells is reported so partial work is visible rather than silent.

Usage:
  python experiments/merge_corrections.py ~/Downloads --out /tmp/braille-labeled
  python experiments/merge_corrections.py ~/Downloads --out data/ --images ../dataset/data/braille_natural
"""

import argparse
import json
from pathlib import Path


def bits_to_label(bits6):
    """6-bit dot string -> Angelina's integer label (bit i = dot i+1)."""
    return sum(1 << i for i, b in enumerate(bits6) if b == '1')


def apply_corrections(doc):
    """Return final cells for one corrections file, plus a per-file tally."""
    by_id = {c['id']: dict(c) for c in doc['predictions']}
    corr = doc.get('corrections', {})

    for e in corr.get('edited', []):
        if e['id'] in by_id:
            by_id[e['id']]['bits'] = e['bits']
    for cell_id in corr.get('deleted', []):
        by_id.pop(cell_id, None)

    cells = list(by_id.values()) + [dict(c) for c in corr.get('added', [])]
    blank = [c for c in cells if c.get('bits', '000000') == '000000']
    return [c for c in cells if c.get('bits', '000000') != '000000'], {
        'predicted': len(doc['predictions']),
        'edited': len(corr.get('edited', [])),
        'deleted': len(corr.get('deleted', [])),
        'added': len(corr.get('added', [])),
        'blank_skipped': len(blank),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('corrections_dir')
    ap.add_argument('--out', default='/tmp/braille-labeled')
    ap.add_argument('--glob', default='*.corrections.json')
    args = ap.parse_args()

    files = sorted(Path(args.corrections_dir).glob(args.glob))
    if not files:
        raise SystemExit(f'No {args.glob} found in {args.corrections_dir}')

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    grand = {'pages': 0, 'cells': 0, 'edited': 0, 'deleted': 0,
             'added': 0, 'blank_skipped': 0}

    for path in files:
        doc = json.loads(path.read_text())
        width, height = doc['image_size']
        cells, tally = apply_corrections(doc)

        stem = Path(doc['image']).stem
        rows = []
        for c in cells:
            left = (c['cx'] - c['w'] / 2) / width
            top = (c['cy'] - c['h'] / 2) / height
            right = (c['cx'] + c['w'] / 2) / width
            bottom = (c['cy'] + c['h'] / 2) / height
            rows.append(f'{left:.6f};{top:.6f};{right:.6f};{bottom:.6f};'
                        f'{bits_to_label(c["bits"])}')
        (out_dir / f'{stem}.csv').write_text('\n'.join(rows) + '\n')

        grand['pages'] += 1
        grand['cells'] += len(cells)
        for k in ('edited', 'deleted', 'added', 'blank_skipped'):
            grand[k] += tally[k]
        print(f'  {stem}: {len(cells)} labeled cells '
              f'(+{tally["added"]} added, {tally["edited"]} edited, '
              f'{tally["deleted"]} deleted'
              + (f', {tally["blank_skipped"]} left blank' if tally['blank_skipped'] else '')
              + ')')

    print(f'\n  {grand["pages"]} pages → {out_dir}')
    print(f'  {grand["cells"]} labeled cells; corrections touched '
          f'{grand["edited"] + grand["deleted"] + grand["added"]} of them '
          f'({grand["edited"]} edited, {grand["deleted"]} deleted, {grand["added"]} added)')
    if grand['blank_skipped']:
        print(f'  {grand["blank_skipped"]} cells left blank and skipped '
              '— those pages are only partly labeled')
    if grand['cells']:
        rate = (grand['edited'] + grand['deleted'] + grand['added']) / grand['cells']
        print(f'  Correction rate: {rate * 100:.1f}% of cells needed a human touch')


if __name__ == '__main__':
    main()
