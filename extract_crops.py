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

import csv
from pathlib import Path
import PIL.Image
import PIL.ImageOps

from angelina_data import ANGELINA, collect_images, load_csv
from dot_pattern_utils import unique_stem

OUT_DIR   = Path('/tmp/braille-crops')
CROP_SIZE = (64, 64)   # (w, h) — square, MobileNetV2-friendly
PADDING   = 0.10       # extra fraction of cell dim added on each side


def cells_with_pixel_bbox(csv_path, img_w, img_h):
    """load_csv()'s fractional bboxes, padded and converted to pixel coords."""
    cells = []
    for cell in load_csv(csv_path):
        l, t, r, b = cell['frac_bbox']
        pw = (r - l) * img_w * PADDING
        ph = (b - t) * img_h * PADDING
        x0 = max(0,  l * img_w - pw)
        y0 = max(0,  t * img_h - ph)
        x1 = min(img_w, r * img_w + pw)
        y1 = min(img_h, b * img_h + ph)
        cells.append({'bbox': (x0, y0, x1, y1), 'label': cell['label'], 'bits6': cell['bits6']})
    return cells


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
        cells = cells_with_pixel_bbox(csv_path, iw, ih)
        img_counts[split] += 1

        for idx, cell in enumerate(cells):
            crop = img.crop(cell['bbox']).resize(CROP_SIZE, PIL.Image.LANCZOS)
            stem  = f"{unique_stem(img_path, ANGELINA)}_{idx:04d}"
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
