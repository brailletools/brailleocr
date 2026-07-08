"""
Extract labeled cell crops from single-sided Angelina images.

Sources:
  - handwritten/ang_redmi/ and handwritten/kov/  (all → train)
  - uploaded/test2/                               (70% train, 15% val, 15% test)

Output:
  /tmp/braille-crops/
    manifest.csv  — columns: split, path, bits6, label_int
    train/        — crop jpg files
    val/
    test/
"""

import csv, random, sys
from pathlib import Path
import PIL.Image, PIL.ImageOps

ANGELINA  = Path('/Users/jmankoff/Research/nonvisual/braille/AngelinaDataset')
OUT_DIR   = Path('/tmp/braille-crops')
CROP_SIZE = (64, 64)   # (w, h) — square, MobileNetV2-friendly
PADDING   = 0.10       # extra fraction of cell dim added on each side
SEED      = 42


def label_to_bits6(label_int):
    """Angelina int label → 6-char bits string (bit0=dot1 at position 0)."""
    return format(label_int, '06b')[::-1]


def load_csv(csv_path, img_w, img_h):
    cells = []
    with open(csv_path) as f:
        for row in csv.reader(f, delimiter=';'):
            if len(row) < 5:
                continue
            l, t, r, b, lbl = float(row[0]), float(row[1]), float(row[2]), float(row[3]), int(row[4])
            # pixel bbox with padding
            pw  = (r - l) * img_w * PADDING
            ph  = (b - t) * img_h * PADDING
            x0  = max(0, l * img_w - pw)
            y0  = max(0, t * img_h - ph)
            x1  = min(img_w, r * img_w + pw)
            y1  = min(img_h, b * img_h + ph)
            cells.append({'bbox': (x0, y0, x1, y1), 'label': lbl, 'bits6': label_to_bits6(lbl)})
    return cells


def collect_images():
    """Return list of (img_path, csv_path, intended_split)."""
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

    return entries


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for s in ('train', 'val', 'test'):
        (OUT_DIR / s).mkdir(exist_ok=True)

    manifest_rows = []
    counts = {'train': 0, 'val': 0, 'test': 0}
    img_counts = {'train': 0, 'val': 0, 'test': 0}

    entries = collect_images()
    print(f"Processing {len(entries)} images …")

    for img_path, csv_path, split in entries:
        img = PIL.ImageOps.exif_transpose(PIL.Image.open(img_path)).convert('RGB')
        iw, ih = img.size
        cells = load_csv(csv_path, iw, ih)
        img_counts[split] += 1

        for idx, cell in enumerate(cells):
            crop = img.crop(cell['bbox']).resize(CROP_SIZE, PIL.Image.LANCZOS)
            stem  = f"{img_path.stem}_{idx:04d}"
            fname = f"{stem}.jpg"
            out_p = OUT_DIR / split / fname
            crop.save(out_p, quality=92)
            manifest_rows.append({
                'split':     split,
                'path':      str(out_p),
                'bits6':     cell['bits6'],
                'label_int': cell['label'],
            })
            counts[split] += 1

    # Write manifest
    manifest_path = OUT_DIR / 'manifest.csv'
    with open(manifest_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['split', 'path', 'bits6', 'label_int'])
        w.writeheader()
        w.writerows(manifest_rows)

    print(f"\nDone. Crops saved to {OUT_DIR}/")
    for s in ('train', 'val', 'test'):
        print(f"  {s:5s}: {img_counts[s]:3d} images → {counts[s]:6d} cell crops")
    print(f"  Manifest: {manifest_path}")


if __name__ == '__main__':
    main()
