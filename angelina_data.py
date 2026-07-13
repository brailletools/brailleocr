"""
Loader for the Angelina single-sided labeled dataset — one of several source
datasets (see also dsbi_data.py) that get combined by extract_crops.py
(MobileNetV2 classifier crops) and prepare_yolo_dataset.py (YOLO detector
training data). Keeping per-dataset loading logic in its own module, sharing
only the generic bits in dot_pattern_utils.py, is what lets both training
pipelines add more source datasets without reshuffling existing ones.

Source layout (AngelinaDataset repo):
  handwritten/**/*.labeled.jpg + matching .csv   — all → train
  uploaded/test2/*.labeled.jpg + matching .csv   — 70/15/15 split
  books/**/*.labeled.jpg + matching .csv         — author-provided train.txt/val.txt split
    (respected as-is rather than re-split randomly, since adjacent pages
    from the same book can be near-duplicates)

CSV format: one row per cell, `;`-delimited:
  l;t;r;b;label_int
  l,t,r,b are fractional (0-1) bbox coordinates within the image.
  label_int, read as a 6-bit binary integer, encodes the dot pattern.
"""

import random
import csv

from dot_pattern_utils import REPOS_ROOT, label_to_bits6

ANGELINA = REPOS_ROOT / 'AngelinaDataset'
SEED     = 42


def load_csv(csv_path):
    """Return cells as fractional (l, t, r, b, label_int, bits6) — no image needed."""
    cells = []
    with open(csv_path) as f:
        for row in csv.reader(f, delimiter=';'):
            if len(row) < 5:
                continue
            l, t, r, b, lbl = float(row[0]), float(row[1]), float(row[2]), float(row[3]), int(row[4])
            cells.append({'frac_bbox': (l, t, r, b), 'label': lbl, 'bits6': label_to_bits6(lbl)})
    return cells


def collect_images():
    """Return list of (img_path, csv_path, split) for every labeled Angelina image."""
    entries = []

    # handwritten — all go to train (search all subdirs recursively)
    for jp in sorted((ANGELINA / 'handwritten').rglob('*.labeled.jpg')):
        cp = jp.with_suffix('.csv')
        if cp.exists():
            entries.append((jp, cp, 'train'))

    # uploaded/test2 — 70/15/15 split by image
    test2_imgs = sorted((ANGELINA / 'uploaded' / 'test2').glob('*.labeled.jpg'))
    test2_imgs = [p for p in test2_imgs if p.with_suffix('.csv').exists()]
    random.seed(SEED)
    random.shuffle(test2_imgs)
    n = len(test2_imgs)
    n_val  = max(1, round(n * 0.15))
    n_test = max(1, round(n * 0.15))
    for i, jp in enumerate(test2_imgs):
        cp = jp.with_suffix('.csv')
        if i < n_val:
            split = 'val'
        elif i < n_val + n_test:
            split = 'test'
        else:
            split = 'train'
        entries.append((jp, cp, split))

    # books — respect the dataset's own train.txt/val.txt split
    for split, list_file in (('train', 'train.txt'), ('val', 'val.txt')):
        list_path = ANGELINA / 'books' / list_file
        for line in list_path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            jp = ANGELINA / 'books' / line.replace('\\', '/')
            cp = jp.with_suffix('.csv')
            if jp.exists() and cp.exists():
                entries.append((jp, cp, split))

    return entries
