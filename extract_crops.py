"""
Extract labeled cell crops for the MobileNetV2 dot classifier.

SPLIT PROTOCOL — deliberately identical to the detector's (see
prepare_yolo_dataset.py), so detection and dot-finding are trained and tested on
the same pages and a paper can describe one protocol instead of two:

  train  each source's own train split
  val    each source's own val split; DSBI publishes none, so a deterministic
         slice of its train PAGES is held out (by page, never by cell — cells
         from one page are near-duplicates, and splitting within a page would
         leak).
  test   each source's own test split. For DSBI that is the 88-page test.txt,
         the shared holdout both stages are evaluated on. Never trained or
         tuned on.

Sources:
  angelina   single-sided pages. Optional: skipped with a warning if the repo
             is not checked out alongside this one.
  dsbi       double-sided. The ONLY source with verso cells — a verso dot is a
             depression with an inverted shadow, and the classifier scored 83.7%
             on verso against 95.0% on recto precisely because it had never seen
             one (RESEARCH.md, "Why our accuracy looks lower than published
             work"). Adding DSBI train is what gives it any.

Output:
  /tmp/braille-crops/
    manifest.csv  — split, path, bits6, label_int, source, side
    train/ val/ test/   crop jpgs
"""

import argparse
import csv
import random
from pathlib import Path

import PIL.Image

import angelina_data
import dsbi_data
from dot_pattern_utils import unique_stem, bits6_to_label

OUT_DIR   = Path('/tmp/braille-crops')
CROP_SIZE = (64, 64)   # (w, h) — square, MobileNetV2-friendly
PADDING   = 0.10       # extra fraction of cell dim added on each side

# TRAIN/INFERENCE PARITY. A saved crop must be pixel-identical to what
# pipeline.reclassify_cells() hands the model at inference, or the classifier is
# trained on images it will never see. Two things have to match:
#
#   filter   inference resizes with torchvision transforms.Resize, which is
#            BILINEAR. Saving with LANCZOS (as this script used to) produces
#            visibly different pixels on a 130px->64px downsample.
#   storage  JPEG re-encoding adds artifacts at exactly the scale of a Braille
#            dot's shadow. PNG is lossless, so the bytes decode back to the
#            pixels the resize produced.
#
# Measured cost of the old mismatch: the classifier scored 99.5% on its own
# JPEG test crops but 96.6% on the same cells cropped the inference way
# (RESEARCH.md, "Train/inference preprocessing parity").
RESAMPLE  = PIL.Image.BILINEAR   # must match torchvision transforms.Resize
CROP_EXT  = 'png'                # lossless; JPEG artifacts are dot-sized
VAL_FRAC  = 0.15       # of DSBI train PAGES, since DSBI publishes no val split
SEED      = 42         # fixed so the val carve-out is reproducible across runs


def dsbi_entries():
    """(img, ann, split, source, side) for DSBI, carving a val set by page."""
    by_page = {}
    for jpg, txt, split in dsbi_data.collect_images():
        by_page.setdefault((split, jpg.name.rsplit('+', 1)[0]), []).append((jpg, txt))

    train_pages = sorted(k for k in by_page if k[0] == 'train')
    rng = random.Random(SEED)
    rng.shuffle(train_pages)
    n_val = max(1, round(len(train_pages) * VAL_FRAC))
    val_pages = set(train_pages[:n_val])

    out = []
    for key, sides in sorted(by_page.items()):
        split = 'val' if key in val_pages else key[0]
        for jpg, txt in sides:
            side = 'verso' if '+verso' in txt.name else 'recto'
            out.append((jpg, txt, split, 'dsbi', side))
    return out


def angelina_entries():
    if not angelina_data.ANGELINA.exists():
        print(f'  angelina: not found at {angelina_data.ANGELINA} — skipping. '
              'Clone AngelinaDataset alongside this repo to include it.')
        return []
    return [(jpg, csv_path, split, 'angelina', 'recto')
            for jpg, csv_path, split in angelina_data.collect_images()]


def load_cells(ann_path, img, source):
    if source == 'dsbi':
        return dsbi_data.load_txt(ann_path, img.width, img.height)
    cells = angelina_data.load_csv(ann_path)
    for c in cells:                       # angelina carries label, not bits6 key parity
        c.setdefault('bits6', c['bits6'])
    return cells


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--out', default=str(OUT_DIR))
    ap.add_argument('--sources', default='angelina,dsbi',
                    help='comma-separated subset of: angelina, dsbi')
    args = ap.parse_args()

    out_dir = Path(args.out)
    for split in ('train', 'val', 'test'):
        (out_dir / split).mkdir(parents=True, exist_ok=True)

    wanted = {s.strip() for s in args.sources.split(',')}
    entries = []
    if 'angelina' in wanted:
        entries += angelina_entries()
    if 'dsbi' in wanted:
        entries += dsbi_entries()
    if not entries:
        raise SystemExit('No sources available.')

    rows = []
    counts = {}
    for img_path, ann_path, split, source, side in entries:
        with PIL.Image.open(img_path) as img:
            img = img.convert('RGB')
            root = (dsbi_data.DSBI if source == 'dsbi' else angelina_data.ANGELINA)
            base = f'{source}_{unique_stem(img_path, root)}'
            for idx, cell in enumerate(load_cells(ann_path, img, source)):
                left, top, right, bottom = cell['frac_bbox']
                x0, y0 = left * img.width, top * img.height
                x1, y1 = right * img.width, bottom * img.height
                px, py = (x1 - x0) * PADDING, (y1 - y0) * PADDING
                crop = img.crop((max(0, x0 - px), max(0, y0 - py),
                                 min(img.width, x1 + px),
                                 min(img.height, y1 + py))).resize(CROP_SIZE,
                                                                   RESAMPLE)
                name = f'{base}_{side}_{idx:04d}.{CROP_EXT}'
                crop.save(out_dir / split / name)
                bits6 = cell['bits6']
                # Absolute path: train_classifier.CellDataset opens row['path']
                # directly, so a crop-dir-relative path would only resolve when
                # the CWD happened to be the crop dir.
                rows.append([split, str(out_dir / split / name), bits6,
                             cell.get('label', bits6_to_label(bits6)), source, side])
                counts[(source, side, split)] = counts.get((source, side, split), 0) + 1

    with open(out_dir / 'manifest.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['split', 'path', 'bits6', 'label_int', 'source', 'side'])
        w.writerows(rows)

    print(f'\n{len(rows)} crops → {out_dir}')
    print(f'{"source":<10}{"side":<8}{"split":<8}{"crops":>8}')
    for key in sorted(counts):
        print(f'{key[0]:<10}{key[1]:<8}{key[2]:<8}{counts[key]:8d}')
    for split in ('train', 'val', 'test'):
        n = sum(v for k, v in counts.items() if k[2] == split)
        print(f'  {split:5s} total: {n}')
    verso_train = sum(v for k, v in counts.items()
                      if k[1] == 'verso' and k[2] in ('train', 'val'))
    print(f'\n  verso crops available for training/tuning: {verso_train}'
          + ('  ← first verso data the classifier has ever had' if verso_train else ''))


if __name__ == '__main__':
    main()
