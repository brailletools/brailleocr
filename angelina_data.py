"""
Loader for the Angelina single-sided labeled dataset — one of several source
datasets (see also dsbi_data.py) that get combined by extract_crops.py
(MobileNetV2 classifier crops) and prepare_yolo_dataset.py (YOLO detector
training data). Keeping per-dataset loading logic in its own module, sharing
only the generic bits in dot_pattern_utils.py, is what lets both training
pipelines add more source datasets without reshuffling existing ones.

Source layout (AngelinaDataset repo), splits as the authors designate them:
  books/<set>/*.labeled.jpg + .csv         train.txt / val.txt
  handwritten/<set>/*.labeled.jpg + .csv   train.txt / val.txt
  uploaded/test2/*.labeled.jpg + .csv      test.txt (all of it is test)
  not_braille/misc/*.jpg                   no Braille; negatives for the detector

CSV format: one row per cell, `;`-delimited:
  l;t;r;b;label_int
  l,t,r,b are fractional (0-1) bbox coordinates within the image.
  label_int, read as a 6-bit binary integer, encodes the dot pattern.
"""

import csv

from dot_pattern_utils import REPOS_ROOT, label_to_bits6

# The dataset lives as a git submodule inside the `dataset` repo
# (dataset/.gitmodules: data/angelina -> github.com/IlyaOvodov/AngelinaDataset),
# alongside dsbi. An earlier layout had it cloned as a sibling of this repo, and
# this module still pointed there — so on a checkout following the documented
# layout the loader silently found nothing and Angelina dropped out of training
# without any error. The sibling path is kept as a fallback for anyone with the
# older layout.
_ANGELINA_SUBMODULE = REPOS_ROOT / 'dataset' / 'data' / 'angelina'
_ANGELINA_SIBLING = REPOS_ROOT / 'AngelinaDataset'
ANGELINA = (_ANGELINA_SUBMODULE if _ANGELINA_SUBMODULE.exists()
            else _ANGELINA_SIBLING)


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


def _listed(list_file):
    """Image paths named in one of the authors' split files. An empty or missing list is
    an error, never an empty source (AUDIT B7)."""
    if not list_file.exists():
        raise FileNotFoundError(f'{list_file} not found (is the Angelina dataset checked out?)')
    paths = [list_file.parent / line.strip().replace('\\', '/')
             for line in list_file.read_text().splitlines() if line.strip()]
    if not paths:
        raise ValueError(f'{list_file} lists no images')
    return paths


def collect_images():
    """
    Return list of (img_path, csv_path, split) for every labeled Angelina image, using
    the authors' own train/val/test designation (see the README in the dataset):

      books/train.txt, books/val.txt              train / val
      handwritten/train.txt, handwritten/val.txt  train / val
      uploaded/test.txt                           test (all 50 user-upload photos)

    The 50 uploaded photos are the authors' real-photo test set. This loader used to
    re-split them 34 train / 8 val / 8 test, so most of the test set was trained on
    (AUDIT B1). A missing image or label file raises rather than being skipped.
    """
    entries = []
    for subdir, split_files in (('books', (('train', 'train.txt'), ('val', 'val.txt'))),
                                ('handwritten', (('train', 'train.txt'), ('val', 'val.txt'))),
                                ('uploaded', (('test', 'test.txt'),))):
        for split, list_file in split_files:
            for jp in _listed(ANGELINA / subdir / list_file):
                cp = jp.with_suffix('.csv')
                if not jp.exists() or not cp.exists():
                    raise FileNotFoundError(f'{list_file.name}: missing {jp if not jp.exists() else cp}')
                entries.append((jp, cp, split))
    return entries


def collect_negatives():
    """Return (img_path, split) for the images with no Braille (not_braille/train.txt):
    background and printed text the detector must not fire on. The authors give no val
    split, so every eighth image (by sorted name) is held out as val."""
    paths = sorted(_listed(ANGELINA / 'not_braille' / 'train.txt'))
    for jp in paths:
        if not jp.exists():
            raise FileNotFoundError(f'not_braille: missing {jp}')
    return [(jp, 'val' if i % 8 == 7 else 'train') for i, jp in enumerate(paths)]
