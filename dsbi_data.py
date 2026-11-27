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

import hashlib

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


def _page_names(list_file):
    path = DSBI / list_file
    if not path.exists():
        raise FileNotFoundError(f'{path} not found (is the DSBI dataset checked out?)')
    names = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not names:
        raise ValueError(f'{path} lists no pages')
    return names


def collect_images():
    """Return list of (img_path, txt_path, split) for every labeled DSBI side-image.
    A page in the split lists with no labeled side at all is an error (AUDIT B7)."""
    entries = []
    for split, list_file in (('train', 'train.txt'), ('test', 'test.txt')):
        for name in _page_names(list_file):
            sides = _side_paths(name)
            if not sides:
                raise FileNotFoundError(f'DSBI page {name!r} has no labeled side')
            for txt, jpg in sides:
                entries.append((jpg, txt, split))
    return entries


def _same_file(a, b):
    return hashlib.sha256(a.read_bytes()).digest() == hashlib.sha256(b.read_bytes()).digest()


def collect_pages():
    """
    One entry per physical page: {'page', 'split', 'kind', 'sides': {side: (jpg, txt)}}.

    DSBI stores each page as a +recto.jpg and a +verso.jpg. Whether those are one scan
    or two decides how the labels may be used (AUDIT B3):

      union   recto and verso are the SAME FILE (90 of 114 pages). Both sides' dots are
              visible in it and both label files describe it: use the union of the labels
              on one image.
      single  only one side has labels (the other file is empty, which DSBI uses for "no
              dots on that side"): that side's labels are the whole page.
      side    recto and verso are DIFFERENT files (each deskewed by its own angle). The
              other side's dots are visible but its labels are in another image's frame,
              so each image is only half labeled. Fine for classifying a side's own cells;
              wrong as a detection example or detection score.
    """
    pages = {}
    for jpg, txt, split in collect_images():
        side = 'verso' if '+verso' in jpg.name else 'recto'
        page = pages.setdefault((split, jpg.name.rsplit('+', 1)[0], str(jpg.parent)),
                                {'page': jpg.name.rsplit('+', 1)[0], 'split': split, 'sides': {}})
        page['sides'][side] = (jpg, txt)
    out = []
    for page in pages.values():
        sides = page['sides']
        if len(sides) == 1:
            page['kind'] = 'single'
        elif _same_file(sides['recto'][0], sides['verso'][0]):
            page['kind'] = 'union'
        else:
            page['kind'] = 'side'
        out.append(page)
    return sorted(out, key=lambda p: (p['split'], str(next(iter(p['sides'].values()))[0])))


def detection_pages(split=None):
    """Pages whose labels cover everything visible in one image ('union' and 'single')
    -- the only ones usable as detector training examples or detection tests."""
    return [p for p in collect_pages()
            if p['kind'] in ('union', 'single') and (split is None or p['split'] == split)]


def load_page_cells(page, img_w, img_h):
    """All labeled cells of a union/single page, tagged with their side."""
    if page['kind'] == 'side':
        raise ValueError(f"page {page['page']!r} is half labeled on each image; "
                         "load one side with load_txt() instead")
    cells = []
    for side, (_, txt) in page['sides'].items():
        cells += [{**c, 'side': side} for c in load_txt(txt, img_w, img_h)]
    return cells


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
