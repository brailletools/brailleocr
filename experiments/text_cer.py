#!/usr/bin/env python3
"""
End-to-end text accuracy (CER) on labeled pages.

Cell-level precision and recall cannot see whether a wrong cell became a wrong
word, so this measures the thing the user actually receives: the English text
out of the pipeline, scored against a reference built from ground truth.

Originally written as an ablation harness for crop_recover() (mean CER 0.586
with the stage vs 0.496 without, 58% slower with it — RESEARCH.md Experiment 6).
That stage was removed on 2026-09-28; the harness stayed, as the regression test
for any future change to recognition or layout. A change that improves cells but
not text is not an improvement.

Reference text is built from the ground-truth cells through the SAME layout and
liblouis path the pipeline uses (group_into_lines -> insert_spaces ->
lou_translate), so the comparison isolates recognition from translation: both
sides get identical treatment downstream, and liblouis quirks cancel out. It is
not a human transcription, so absolute CER here is a lower bound on true error
-- but the with/without DELTA, which is what the ablation is for, is sound.

This also gives us the text-level metric RESEARCH.md notes we lack (cell-level
precision/recall cannot see whether a fabricated cell became a wrong word).

DSBI page structure: <page>+recto.jpg and <page>+verso.jpg are the SAME scan
(verified by md5), with each side's .txt labeling only that side's cells. The
detector reads both sides' dots and cannot separate them, so reference text is
built from the UNION of both label files, grouped per physical page — scoring
against one side alone would treat every real other-side cell as an error.

Caveat this exposes rather than fixes: a double-sided page's union text is
interleaved recto and verso, which is not what a human reader wants. Absolute
CER here is therefore not a quality figure for the product; the with/without
DELTA is what this ablation measures.

Usage:
  python experiments/text_cer.py --limit 4 --json out.json
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import PIL.Image
import PIL.ImageOps
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline                                                 # noqa: E402
from dsbi_data import collect_images, load_txt                  # noqa: E402
from model_fetch import resolve_model, resolve_classifier_path  # noqa: E402


def levenshtein(a, b):
    """Edit distance; plain DP, inputs here are a few thousand chars."""
    if a == b:
        return 0
    if not a or not b:
        return max(len(a), len(b))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def reference_text(txt_paths, img_w, img_h, lang):
    """Ground-truth cells (union of both sides) -> Unicode Braille -> English,
    via the pipeline's own layout and translation path."""
    cells = []
    for c in [c for txt in txt_paths for c in load_txt(txt, img_w, img_h)]:
        left, top, right, bottom = c['frac_bbox']
        cells.append({
            'cx': (left + right) / 2 * img_w, 'cy': (top + bottom) / 2 * img_h,
            'w': (right - left) * img_w, 'h': (bottom - top) * img_h,
            'bits': c['bits6'], 'char': pipeline.bits_to_braille(c['bits6']),
            'conf': 1.0,
        })
    if not cells:
        return ''
    avg_w = statistics.median(c['w'] for c in cells)
    lines = []
    for line in pipeline.group_into_lines(cells):
        lines.append(''.join(c['char'] for c in pipeline.insert_spaces(line, avg_w)))
    return '\n'.join(pipeline.braille_to_text(lines, lang))


def run_pipeline(img, stem, model, args):
    """Run the pipeline once; return (text, seconds)."""
    t0 = time.perf_counter()
    text = pipeline.process_container(
        img, stem, model, lang_table=args.lang, search_contrast=False,
        spellcheck=not args.no_spellcheck, max_det=args.max_det,
        classifier_path=(args.classifier or resolve_classifier_path()))
    return (text or ''), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--limit', type=int, default=8)
    ap.add_argument('--lang', default='en-ueb-g2.ctb')
    ap.add_argument('--max-det', type=int, default=2000)
    ap.add_argument('--no-spellcheck', action='store_true')
    ap.add_argument('--classifier', default=None,
                    help='path to a cell_classifier.pt (default: resolved copy)')
    ap.add_argument('--json')
    args = ap.parse_args()

    # Group side-files by physical page: 'FM+1+recto.jpg' -> 'FM+1'.
    pages = {}
    for jpg, txt, split in collect_images():
        if split == 'test':
            pages.setdefault(jpg.name.rsplit('+', 1)[0], [jpg, []])[1].append(txt)
    entries = list(pages.values())[:args.limit]
    if not entries:
        raise SystemExit('No DSBI test-split images found.')
    print(f'{len(entries)} physical pages (both label sides unioned)')

    model = YOLO(str(resolve_model('cell_detector.pt')))
    pipeline.OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for jpg, txts in entries:
        img = PIL.ImageOps.exif_transpose(PIL.Image.open(jpg)).convert('RGB')
        ref = reference_text(txts, img.width, img.height, args.lang)
        if not ref.strip():
            continue

        text, secs = run_pipeline(img, f'cer_{jpg.stem}', model, args)
        row = {'image': jpg.name, 'ref_chars': len(ref),
               'cer': levenshtein(ref, text) / len(ref), 'seconds': secs}
        rows.append(row)
        print(f'  {jpg.name}: CER {row["cer"]:.4f}  ({secs:.1f}s)')

    if not rows:
        raise SystemExit('No pages produced reference text.')

    n = len(rows)
    mean_cer = sum(r['cer'] for r in rows) / n
    total_s = sum(r['seconds'] for r in rows)

    print(f'\n── End-to-end text, {n} pages ' + '─' * 28)
    print(f'  mean CER      {mean_cer:.4f}')
    print('  per page      ' + ', '.join(f'{r["cer"]:.3f}' for r in rows))
    print(f'  wall clock    {total_s:.1f}s ({total_s / n:.1f}s per page)')
    print('\n  Baseline after removing crop_recover (2026-09-28): mean CER 0.4961\n'
          '  over these 4 pages. A regression above that needs explaining.')

    if args.json:
        Path(args.json).write_text(json.dumps({'pages': rows}, indent=2))
        print(f'\nWrote {args.json}')


if __name__ == '__main__':
    main()
