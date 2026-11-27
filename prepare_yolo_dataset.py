"""
Convert all combined source datasets (Angelina, DSBI, braille_natural — see
angelina_data.py / dsbi_data.py / braille_natural_data.py) into one YOLO
detection dataset.

Each source module exposes its own collect_images()/loader pair; this script
flattens them into a shared images/labels layout, tagging filenames by
source so cross-dataset collisions can't happen even though multiple
datasets reuse generic page-numbered filenames internally.

Training images are TILED, not used whole: our source photos/scans are
mostly whole pages with hundreds of cells, which — after ultralytics'
resize-to-640 — squashes each cell down to a few pixels, an unnecessarily
hard regime to learn fine 6-dot patterns in. Each source image is split into
overlapping native-pixel tiles sized so cells land at TARGET_CELL_PX after
the standard resize, matching exactly how the readers tile at inference (js/src/tiling.js, reader.py)
(see its module-level comment above TARGET_CELL_PX) — a detector trained on
whole pages and run through inference-time tiling would be seeing a scale
range at test time it never saw in training.

Changes from the first version (AUDIT.md B1-B4): Angelina uses the authors' splits (its
50 user-upload photos are TEST, not training); DSBI pages whose recto and verso are one
scan are used once with both sides' labels, and pages whose two sides are different scans
are skipped; cell-free tiles and the not_braille images are kept as background; and boxes
are brought to one shape (dot_pattern_utils.canonical_boxes).

braille_natural has no dot-pattern labels (see its module docstring), so it
only contributes when --single-class is passed.

Class scheme: class_id == bits6 interpreted per angelina_data.label_to_bits6
(0-63); class name == the 6-bit dot-pattern string, matching what
the readers expect from a detector's model.names.

Output:
  /tmp/braille-yolo-dataset[-singleclass]/
    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt   — YOLO format: class_id cx cy w h (normalized)
    dataset.yaml
"""

import argparse
import random
import statistics
from pathlib import Path

import PIL.Image
import yaml

import angelina_data
import braille_natural_data
import dsbi_data
from dot_pattern_utils import (MIN_NATIVE_TILE, TARGET_CELL_PX, TILE_SIZE, canonical_boxes,
                               make_tile_boxes, unique_stem)

N_CLASSES = 64
SEED = 42
EMPTY_TILE_FRAC = 0.15          # share of cell-free tiles (from Braille pages) kept as background
NEGATIVE_NATIVE_SIZES = (640, 1280)   # native tile sizes cut from images with no Braille
NEGATIVE_TILES_PER_SIZE = 4


def _angelina_cells(ann_path, img_path):
    return angelina_data.load_csv(ann_path)


def _dsbi_cells(page, img_path):
    """`page` is a dsbi_data page ('union' or 'single'): one image, every side's labels."""
    with PIL.Image.open(img_path) as im:
        w, h = im.size
    return dsbi_data.load_page_cells(page, w, h)


def _dsbi_entries():
    """(image, page, split) for the pages that are fully labeled in one image. Pages whose
    recto and verso are different scans (each image half labeled) cannot be detector
    examples: the unlabeled side's dots would be taught as background (AUDIT B3)."""
    pages = dsbi_data.collect_pages()
    skipped = [p for p in pages if p['kind'] == 'side']
    if skipped:
        print(f"  dsbi: skipping {len(skipped)} pages whose recto and verso are different scans "
              f"({sum(p['split'] == 'train' for p in skipped)} train) -- half labeled per image")
    # recto.jpg of a union page is the same file as verso.jpg, so either side's path will do
    return [(next(iter(p['sides'].values()))[0], p, p['split'])
            for p in pages if p['kind'] in ('union', 'single')]


def _braille_natural_cells(ann_path, img_path):
    return braille_natural_data.load_voc_xml(ann_path)


# (source tag, dataset root, collect_images(), cells-loader(ann_path, img_path), supports_multiclass)
SOURCES = [
    ('angelina', angelina_data.ANGELINA, angelina_data.collect_images, _angelina_cells, True),
    ('dsbi', dsbi_data.DSBI, _dsbi_entries, _dsbi_cells, True),
    ('braille_natural', braille_natural_data.BRAILLE_NATURAL, braille_natural_data.collect_images,
     _braille_natural_cells, braille_natural_data.SUPPORTS_MULTICLASS),
]


def _cell_center_in_tile(cell, tile_box):
    l, t, r, b = cell['frac_bbox']
    cx, cy = (l + r) / 2, (t + b) / 2
    x0, y0, x1, y1 = tile_box
    return x0 <= cx <= x1 and y0 <= cy <= y1


def _touches(cell, tile_box):
    l, t, r, b = cell['frac_bbox']
    x0, y0, x1, y1 = tile_box
    return l < x1 and r > x0 and t < y1 and b > y0


def tile_image(img, cells, img_w, img_h, empty_frac=0.0, rng=None):
    """
    Split one source image into overlapping native-pixel tiles sized so
    cells land at TARGET_CELL_PX after ultralytics' resize-to-TILE_SIZE (the
    same math the readers use at inference). Returns a list of
    (tile_img, tile_cells) — tile_cells' frac_bbox is relative to the tile,
    not the original image. A cell is assigned to whichever tile contains its
    center; tiles overlap so a cell near a boundary still lands whole in at
    least one tile.
    """
    widths_px = [(c['frac_bbox'][2] - c['frac_bbox'][0]) * img_w for c in cells]
    median_w = statistics.median(widths_px) if widths_px else TILE_SIZE
    native_tile_size = max(MIN_NATIVE_TILE, round(median_w * TILE_SIZE / TARGET_CELL_PX))

    boxes_px = make_tile_boxes(img_w, img_h, native_tile_size)
    tiles = []
    rng = rng or random.Random(SEED)
    for x0, y0, x1, y1 in boxes_px:
        tile_box_frac = (x0 / img_w, y0 / img_h, x1 / img_w, y1 / img_h)
        tw, th = x1 - x0, y1 - y0
        tile_cells = []
        for cell in cells:
            if not _cell_center_in_tile(cell, tile_box_frac):
                continue
            l, t, r, b = cell['frac_bbox']
            # Re-express relative to the tile, still fractional (of tile size)
            new_bbox = (
                max(0.0, (l * img_w - x0) / tw), max(0.0, (t * img_h - y0) / th),
                min(1.0, (r * img_w - x0) / tw), min(1.0, (b * img_h - y0) / th),
            )
            tile_cells.append({**cell, 'frac_bbox': new_bbox})
        if tile_cells:
            tiles.append((img.crop((x0, y0, x1, y1)), tile_cells))
        elif empty_frac and not any(_touches(c, tile_box_frac) for c in cells) \
                and rng.random() < empty_frac:
            # Background. A tile that merely clips a cell's edge is NOT background (that
            # cell would be taught as missing), so only tiles touching no cell qualify.
            tiles.append((img.crop((x0, y0, x1, y1)), []))
    return tiles


def negative_tiles(img, rng):
    """Cell-free tiles cut from an image with no Braille, at a few native scales."""
    img_w, img_h = img.size
    tiles = []
    for size in NEGATIVE_NATIVE_SIZES:
        size = min(size, img_w, img_h)
        for _ in range(NEGATIVE_TILES_PER_SIZE):
            x0 = rng.randint(0, img_w - size)
            y0 = rng.randint(0, img_h - size)
            tiles.append(img.crop((x0, y0, x0 + size, y0 + size)))
    return tiles


def write_label_file(cells, out_path, single_class):
    if not cells:       # background tile: an empty label file is how YOLO says "no objects"
        out_path.write_text('')
        return
    lines = []
    for cell in cells:
        l, t, r, b = cell['frac_bbox']
        cx, cy = (l + r) / 2, (t + b) / 2
        w, h = r - l, b - t
        cls = 0 if single_class else cell['label']
        lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
    out_path.write_text('\n'.join(lines) + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--single-class', action='store_true',
                     help="Collapse all 64 dot-pattern classes into one 'cell' class "
                          '(localization-only; use when a downstream classifier will '
                          'assign the actual dot pattern, as the readers do).')
    ap.add_argument('--out', default=None, help='Output dir override')
    ap.add_argument('--empty-tile-frac', type=float, default=EMPTY_TILE_FRAC,
                    help='share of cell-free tiles from Braille pages kept as background '
                         '(0 reproduces the old behaviour of dropping them)')
    ap.add_argument('--no-negatives', action='store_true',
                    help='leave out the Angelina not_braille images')
    args = ap.parse_args()

    out_dir = Path(args.out) if args.out else Path(
        '/tmp/braille-yolo-dataset-singleclass' if args.single_class else '/tmp/braille-yolo-dataset'
    )

    for split in ('train', 'val', 'test'):
        (out_dir / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out_dir / 'labels' / split).mkdir(parents=True, exist_ok=True)

    counts = {'train': 0, 'val': 0, 'test': 0}
    rng = random.Random(SEED)
    background = 0
    for tag, root, collect_images, load_cells, supports_multiclass in SOURCES:
        if not args.single_class and not supports_multiclass:
            print(f"Skipping {tag} (no dot-pattern labels, only usable with --single-class)")
            continue
        entries = collect_images()
        print(f"Processing {len(entries)} {tag} images …")
        for img_path, ann_path, split in entries:
            base_stem = f'{tag}_{unique_stem(img_path, root)}'
            with PIL.Image.open(img_path) as img:
                img.load()
                img_w, img_h = img.size
                cells = canonical_boxes(load_cells(ann_path, img_path), img_w, img_h)
                tiles = tile_image(img, cells, img_w, img_h, args.empty_tile_frac, rng)
                background += sum(1 for _, tc in tiles if not tc)
                for i, (tile_img, tile_cells) in enumerate(tiles):
                    stem = base_stem if len(tiles) == 1 else f'{base_stem}_tile{i}'
                    tile_img.save(out_dir / 'images' / split / f'{stem}.jpg', quality=92)
                    write_label_file(tile_cells, out_dir / 'labels' / split / f'{stem}.txt',
                                      args.single_class)
                    counts[split] += 1

    if not args.no_negatives:
        negatives = angelina_data.collect_negatives()
        print(f"Processing {len(negatives)} not_braille images as background …")
        for img_path, split in negatives:
            base_stem = f'negative_{unique_stem(img_path, angelina_data.ANGELINA)}'
            with PIL.Image.open(img_path) as img:
                img = img.convert('RGB')
                for i, tile in enumerate(negative_tiles(img, rng)):
                    stem = f'{base_stem}_tile{i}'
                    tile.save(out_dir / 'images' / split / f'{stem}.jpg', quality=92)
                    write_label_file([], out_dir / 'labels' / split / f'{stem}.txt', args.single_class)
                    counts[split] += 1
                    background += 1

    names = {0: 'cell'} if args.single_class else {i: format(i, '06b')[::-1] for i in range(N_CLASSES)}
    dataset_yaml = {
        'path': str(out_dir),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': names,
    }
    (out_dir / 'dataset.yaml').write_text(yaml.dump(dataset_yaml, sort_keys=False))

    print(f"\nDone. Dataset written to {out_dir}/")
    for split in ('train', 'val', 'test'):
        print(f"  {split:5s}: {counts[split]:4d} tiles")
    print(f"  background tiles (no cells): {background}")
    print(f"  dataset.yaml: {out_dir / 'dataset.yaml'}")


if __name__ == '__main__':
    main()
