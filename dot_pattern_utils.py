"""
Utilities shared across every source dataset (angelina_data.py, dsbi_data.py,
...): converting between dot-pattern representations, generating output
filenames that stay unique when multiple datasets' images get flattened into
one combined training set, and tiling a large image into overlapping
native-pixel regions (shared between prepare_yolo_dataset.py, which tiles
training images the same way, and pipeline.py, which tiles at inference —
see pipeline.py's module-level comment above TARGET_CELL_PX for why tiling
at a *native pixel* size, not a resized one, is what actually matters).
"""

from pathlib import Path

# Sibling source-dataset repos (AngelinaDataset, dataset, braille2latex, ...)
# are assumed to be checked out alongside this repo, under the same parent
# directory — e.g. .../braille/brailleocr and .../braille/AngelinaDataset.
REPOS_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TILE_OVERLAP_FRAC = 0.2

# Shared between prepare_yolo_dataset.py (tiles training images this way) and
# pipeline.py (tiles at inference the same way) — a detector trained at one
# cell-pixel-density won't perform well fed a different one at inference, so
# both sides must use the same values. TARGET_CELL_PX is a deliberate design
# choice (not fit to any particular dataset): braille cells have 6 sub-dot
# features to resolve, so aim well above what a detector can just barely
# distinguish from noise, while still fitting a useful number of cells in one
# TILE_SIZE-native tile. Revisit empirically if detection quality suggests
# otherwise.
TILE_SIZE       = 640
TARGET_CELL_PX  = 30
MIN_NATIVE_TILE = 100  # floor on the computed native tile size — guards against a
                       # pathologically small/noisy cell-size estimate producing
                       # an excessive number of tiles


def label_to_bits6(label_int):
    """Angelina-style int label → 6-char bits string (bit0=dot1 at position 0)."""
    return format(label_int, '06b')[::-1]


def bits6_to_label(bits6):
    """Inverse of label_to_bits6 — lets sources that give bits6 directly
    (e.g. DSBI) produce the same class-id scheme as Angelina's int labels."""
    return int(bits6[::-1], 2)


def unique_stem(img_path, root):
    """
    A filename stem guaranteed unique across a whole source dataset.

    Plain Path.stem collides when a dataset's subdirectories reuse generic
    page numbers (e.g. both 'mdd-redmi1/10.labeled.jpg' and 'ola/10.labeled.jpg'
    would stem to '10.labeled') — silently overwriting one when flattened into
    a single output directory. `root` should be the dataset's own root path
    (e.g. ANGELINA or DSBI), so stems also stay disambiguated *across*
    datasets once callers prefix them with a per-source tag.
    """
    rel = img_path.relative_to(root).with_suffix('')
    return str(rel).replace('/', '_')


def make_tile_boxes(img_w, img_h, tile_size, overlap_frac=DEFAULT_TILE_OVERLAP_FRAC):
    """Grid of (x0,y0,x1,y1) tiles covering (img_w, img_h), using tile_size
    native pixels per tile with overlap_frac overlap. A single tile covering
    the whole image if it already fits within tile_size."""
    if img_w <= tile_size and img_h <= tile_size:
        return [(0, 0, img_w, img_h)]

    overlap = round(tile_size * overlap_frac)
    stride = max(1, tile_size - overlap)
    xs = list(range(0, max(img_w - tile_size, 0) + 1, stride)) or [0]
    ys = list(range(0, max(img_h - tile_size, 0) + 1, stride)) or [0]
    if xs[-1] + tile_size < img_w:
        xs.append(img_w - tile_size)
    if ys[-1] + tile_size < img_h:
        ys.append(img_h - tile_size)

    return [(x, y, min(x + tile_size, img_w), min(y + tile_size, img_h))
            for y in ys for x in xs]
