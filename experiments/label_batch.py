#!/usr/bin/env python3
"""
Generate a queue of correction pages, with an index that tracks progress.

Labeling universe as of 2026-09-22:

  braille_natural  212 images  boxes already drawn by humans, NO dot patterns
  sample-images     10 images  nothing labeled (phone photos of pages)
  dsbi             114 pages   fully labeled already — not offered here

braille_natural is the priority: it is the only real-world (signage, elevator
button) imagery we have, its boxes are human-verified, and it is currently
localization-only, so the detector can train on it but the classifier cannot.
Adding dot patterns converts it into full training AND evaluation data.

Those pages use the dataset's own boxes (so the labeler never places a cell) and
PREPOPULATE each box's dot pattern by running the MobileNetV2 classifier on it —
which is precisely the classifier's input contract: a known cell position, six
independent dot probabilities. The labeler then fixes only what is wrong.

Automation-bias caveat: a prepopulated label invites agreement, so an error the
classifier makes confidently can survive review. Mitigation here is to surface
the model's own uncertainty — dots whose probability falls in the ambiguous band
are marked in the interface, directing attention to where the model is least
sure. It does not eliminate the bias; spot-checking a sample against
from-scratch labels is still worthwhile.

Writes one self-contained .html per image plus index.html, which links them and
shows which are done by looking for <stem>.corrections.json in --corrections.
Reopening the index after saving corrections updates the progress count.

Usage:
  python experiments/label_batch.py --source braille_natural --limit 25
  python experiments/label_batch.py --source samples
  python experiments/label_batch.py --source braille_natural --corrections ~/Downloads
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import PIL.Image
import PIL.ImageOps
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline                                      # noqa: E402
from braille_natural_data import BRAILLE_NATURAL     # noqa: E402
from dot_pattern_utils import REPOS_ROOT             # noqa: E402
from label_tool import build_page, predict           # noqa: E402
from model_fetch import resolve_model, resolve_classifier_path  # noqa: E402

SAMPLE_DIR = REPOS_ROOT / 'dataset' / 'data' / 'sample-images'
IMG_EXTS = {'.jpg', '.jpeg', '.png'}


_CLF = None


def classify_boxes(img, cells):
    """Fill in each box's dot pattern with the classifier, plus per-dot
    probabilities so the interface can flag the model's own uncertainty.

    The classifier is cached across images: loading it takes ~1.8s, which over a
    212-page batch would be six minutes of pure model loading."""
    global _CLF
    import torch
    if _CLF is None:
        _CLF = pipeline.load_cell_classifier(resolve_classifier_path())
    clf, dev, tf = _CLF
    crops = []
    for c in cells:
        pad_x, pad_y = c['w'] * 0.10, c['h'] * 0.10
        box = (max(0, c['cx'] - c['w']/2 - pad_x), max(0, c['cy'] - c['h']/2 - pad_y),
               min(img.width, c['cx'] + c['w']/2 + pad_x),
               min(img.height, c['cy'] + c['h']/2 + pad_y))
        crops.append(tf(img.crop(box).convert('RGB')))
    with torch.no_grad():
        probs = clf(torch.stack(crops).to(dev)).sigmoid().cpu().numpy()
    for c, row in zip(cells, probs):
        c['bits'] = ''.join('1' if p > 0.5 else '0' for p in row)
        c['probs'] = [round(float(p), 3) for p in row]
    return cells


def voc_boxes(xml_path):
    """Human-drawn boxes from a Pascal VOC annotation; dot pattern unknown."""
    cells = []
    for obj in ET.parse(xml_path).getroot().findall('object'):
        b = obj.find('bndbox')
        x0, y0 = float(b.find('xmin').text), float(b.find('ymin').text)
        x1, y1 = float(b.find('xmax').text), float(b.find('ymax').text)
        cells.append({'cx': (x0 + x1) / 2, 'cy': (y0 + y1) / 2,
                      'w': x1 - x0, 'h': y1 - y0,
                      'bits': '000000', 'conf': None})
    return cells


def natural_entries():
    """(image, voc annotation) pairs across braille_natural's train and test."""
    out = []
    for split in ('train', 'test'):
        root = BRAILLE_NATURAL / 'VOC_Braille' / f'natural_{split}'
        for jpg in sorted((root / 'JPEGImages').glob('*.jpg')):
            xml = root / 'Annotations' / f'{jpg.stem}.xml'
            if xml.exists():
                out.append((jpg, xml))
    return out


INDEX = '''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<title>Braille labeling queue</title><style>
 :root {{ --bg:#faf9f7; --fg:#1a1a1a; --panel:#fff; --line:#d8d4cd; --ok:#2d6a4f; }}
 @media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{
   --bg:#16151a; --fg:#ececec; --panel:#22212a; --line:#3a3845; }} }}
 body {{ margin:0 auto; max-width:820px; padding:24px 16px; background:var(--bg);
        color:var(--fg); font:15px/1.6 ui-sans-serif,system-ui,sans-serif; }}
 h1 {{ font-size:20px; }}
 .bar {{ height:8px; background:var(--line); border-radius:4px; overflow:hidden; margin:12px 0 20px; }}
 .bar i {{ display:block; height:100%; background:var(--ok); width:{pct}%; }}
 ol {{ padding-left:22px; }}
 li {{ margin:4px 0; }}
 a {{ color:inherit; }}
 .note {{ font-size:13px; opacity:.75; border-left:3px solid var(--line); padding-left:12px; }}
</style></head><body>
<h1>Braille labeling queue — {source}</h1>
<p><strong>{done}</strong> of <strong>{total}</strong> pages have corrections saved.</p>
<div class="bar"><i></i></div>
<p class="note">Open a page, fix only what is wrong, then <em>Download corrections</em>.
Save the JSON into <code>{corrections}</code> and reload this index to update progress.
Pages with nothing wrong still count: save the (empty) corrections file to record
that the page was checked.</p>
<ol>{items}</ol>
</body></html>
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source', choices=['braille_natural', 'samples'],
                    default='braille_natural')
    ap.add_argument('--limit', type=int, default=25)
    ap.add_argument('--out', default='/tmp/braille-label')
    ap.add_argument('--corrections', default=str(REPOS_ROOT / 'brailleocr' / 'labels' / 'corrections'),
                    help='where saved corrections JSONs live (for progress tracking)')
    ap.add_argument('--max-det', type=int, default=2000)
    ap.add_argument('--open', action='store_true')
    args = ap.parse_args()

    out_dir = Path(args.out)
    corrections_dir = Path(args.corrections)

    if args.source == 'braille_natural':
        entries = natural_entries()[:args.limit]
        if not entries:
            raise SystemExit(f'No braille_natural images under {BRAILLE_NATURAL}')
        model = None    # boxes come from the dataset; dots from the classifier
    else:
        entries = [(p, None) for p in sorted(SAMPLE_DIR.iterdir())
                   if p.suffix.lower() in IMG_EXTS][:args.limit]
        model = YOLO(str(resolve_model('cell_detector.pt')))

    rows = []
    stems = [jpg.stem for jpg, _ in entries]
    for i, (jpg, xml) in enumerate(entries):
        img = PIL.ImageOps.exif_transpose(PIL.Image.open(jpg)).convert('RGB')
        if xml is not None:
            cells = voc_boxes(xml)
            if cells:
                classify_boxes(img, cells)
                unsure = sum(1 for c in cells
                             if any(0.2 < p < 0.8 for p in c['probs']))
                source_note = (f'{len(cells)} human boxes, dots prefilled by '
                               f'classifier ({unsure} with an uncertain dot)')
            else:
                source_note = 'no boxes in annotation'
        else:
            cells = predict(img, model, args.max_det, True)
            source_note = f'{len(cells)} cells predicted at conf >= {pipeline.HIGH_CONF}'
        page = build_page(
            img, cells, jpg.name, out_dir / f'{jpg.stem}.label.html',
            prev_href=(f'{stems[i-1]}.label.html' if i > 0 else None),
            next_href=(f'{stems[i+1]}.label.html' if i < len(stems) - 1 else None),
            index_href='index.html', position=(i + 1, len(entries)))
        done = (corrections_dir / f'{jpg.stem}.corrections.json').exists()
        rows.append((jpg.stem, page.name, len(cells), source_note, done))
        print(f'  {jpg.name}: {source_note}')

    done_n = sum(r[4] for r in rows)
    items = ''.join(
        f'<li><a href="{name}">{stem}</a> — {n} cells{" ✓ done" if d else ""}</li>'
        for stem, name, n, _, d in rows)
    index = out_dir / 'index.html'
    index.write_text(INDEX.format(
        source=args.source, done=done_n, total=len(rows),
        pct=(done_n / len(rows) * 100) if rows else 0,
        corrections=corrections_dir, items=items))
    print(f'\n  {len(rows)} pages → {index}')
    print(f'  Progress: {done_n}/{len(rows)} have corrections in {corrections_dir}')
    if args.open:
        import webbrowser
        webbrowser.open(index.as_uri())


if __name__ == '__main__':
    main()
