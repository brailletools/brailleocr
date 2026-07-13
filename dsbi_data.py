"""
Shared loader for DSBI (Double-Sided Braille Image Dataset).

Produces the same cell-dict shape as angelina_data.load_csv() (frac_bbox,
label, bits6), so prepare_yolo_dataset.py can combine both sources uniformly.

Source layout (dataset/data/dsbi submodule, see its README.md for the full
annotation spec):
  data/**/<page>+recto.txt + <page>+recto.jpg   (also +verso variants)
  train.txt / test.txt list page basenames (e.g. "Massage\\M+1.jpg"); the
    split applies to both sides of that page.

Annotation format, verified against a real sample (dot bits render in the
standard dot1..dot6 layout — top-left, mid-left, bot-left, top-right,
mid-right, bot-right — with no reversal needed, unlike Angelina's int label):
  line 1: skew angle — ignored, since we read the pre-deskewed +recto/+verso.jpg
  line 2: vertical line positions, (left, right) pairs per cell column
  line 3: horizontal line positions, (top, mid, bottom) triplets per cell row
  each following line: "row col d1 d2 d3 d4 d5 d6" (1-indexed row/col)

A blank annotation file means that side of the page has no Braille at all
(single-sided page) — skipped, rather than treated as a labeled-empty
background image, since we can't tell "truly empty" from "not annotated".
"""

from dot_pattern_utils import REPOS_ROOT, bits6_to_label

DSBI = REPOS_ROOT / 'dataset' / 'data' / 'dsbi'
PADDING = 0.15  # fraction of each span added as bbox margin (see module docstring)


def _side_paths(page_rel):
    """'Massage\\M+1.jpg' -> [(txt, jpg), ...] for whichever sides are labeled."""
    stem = (DSBI / 'data' / page_rel.replace('\\', '/')).with_suffix('')
    pairs = []
    for side in ('recto', 'verso'):
        txt = stem.parent / f'{stem.name}+{side}.txt'
        jpg = stem.parent / f'{stem.name}+{side}.jpg'
        if txt.exists() and jpg.exists() and txt.stat().st_size > 0:
            pairs.append((txt, jpg))
    return pairs


def collect_images():
    """Return list of (img_path, txt_path, split) for every labeled DSBI side-image."""
    entries = []
    for split, list_file in (('train', 'train.txt'), ('test', 'test.txt')):
        for line in (DSBI / list_file).read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            for txt, jpg in _side_paths(line):
                entries.append((jpg, txt, split))
    return entries


def load_txt(txt_path, img_w, img_h):
    """Return cells as fractional (l, t, r, b, label, bits6) for one +recto/+verso.txt."""
    lines = txt_path.read_text().splitlines()
    vpos = list(map(int, lines[1].split()))
    hpos = list(map(int, lines[2].split()))

    cells = []
    for line in lines[3:]:
        parts = line.split()
        if len(parts) < 8:
            continue
        row, col = int(parts[0]), int(parts[1])
        bits6 = ''.join(parts[2:8])
        vi, hi = 2 * (col - 1), 3 * (row - 1)
        if vi + 1 >= len(vpos) or hi + 2 >= len(hpos):
            continue  # malformed row/col reference — skip rather than crash
        x0, x1 = vpos[vi], vpos[vi + 1]
        y0, y1 = hpos[hi], hpos[hi + 2]
        pad_x = (x1 - x0) * PADDING
        pad_y = (y1 - y0) * PADDING
        x0, x1 = max(0, x0 - pad_x), min(img_w, x1 + pad_x)
        y0, y1 = max(0, y0 - pad_y), min(img_h, y1 + pad_y)
        cells.append({
            'frac_bbox': (x0 / img_w, y0 / img_h, x1 / img_w, y1 / img_h),
            'label': bits6_to_label(bits6),
            'bits6': bits6,
        })
    return cells
