"""
Utilities shared by the readers here and the training code in braille-research
(angelina_data.py, dsbi_data.py, prepare_yolo_dataset.py, ...): converting between dot-pattern representations, generating output
filenames that stay unique when multiple datasets' images get flattened into
one combined training set, and tiling a large image into overlapping
native-pixel regions (shared between prepare_yolo_dataset.py, which tiles
training images the same way, and the readers (js/src/tiling.js, reader.py), which tile at inference —
see the comment above TARGET_CELL_PX for why tiling
at a *native pixel* size, not a resized one, is what actually matters).
"""

from pathlib import Path

# Sibling source-dataset repos (AngelinaDataset, dataset, braille2latex, ...)
# are assumed to be checked out alongside this repo, under the same parent
# directory — e.g. .../braille/brailleocr and .../braille/AngelinaDataset.
REPOS_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_TILE_OVERLAP_FRAC = 0.2

# Shared between prepare_yolo_dataset.py (tiles training images this way) and
# the readers (tile at inference the same way) — a detector trained at one
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


# ── Box convention (AUDIT B4) ────────────────────────────────────────────────
# The sources draw a Braille cell's box two different ways, measured relative to the
# cell pitch (distance between neighbouring cells in a word):
#
#   narrow  DSBI, and Angelina's user-upload photos:  width ~0.54 pitch, height ~1.08
#   wide    Angelina books/handwriting, braille_natural: width ~0.73-0.77, height ~1.15-1.18
#
# (Angelina's user-upload labels follow DSBI's convention, so the loader that treats
# "Angelina" as one source was mixing both.) A detector trained on both learns an average
# box, and the classifier crops whatever box the detector draws. Narrow-convention images
# are rescaled about each box's centre to the wide convention, which most of the data
# already uses. Images that are already wide, or too sparse to measure, are untouched.
# Target shapes (width/pitch, height/pitch). 'wide' is the default. 'narrow' exists so the
# two can be compared by retraining (AUDIT B4): classifying ground-truth boxes with the old,
# mixed-data classifier was better on narrow boxes, but that classifier had learned narrow
# crops, so it does not say which shape is better.
BOX_SHAPES = {'wide': (0.75, 1.16), 'narrow': (0.54, 1.08)}
DEFAULT_BOX_SHAPE = 'wide'
BOX_SHAPE_TOLERANCE = 0.12     # a page already within this fraction of the target is left alone
MIN_CELLS_TO_MEASURE = 20


def _median(values):
    values = sorted(values)
    return values[len(values) // 2]


def box_shape_over_pitch(cells, img_w, img_h):
    """Median (width/pitch, height/pitch) of a page's boxes, or None if too few cells
    have a right-hand neighbour in the same row to measure a pitch."""
    px = [((c['frac_bbox'][0] + c['frac_bbox'][2]) / 2 * img_w,
           (c['frac_bbox'][1] + c['frac_bbox'][3]) / 2 * img_h,
           (c['frac_bbox'][2] - c['frac_bbox'][0]) * img_w,
           (c['frac_bbox'][3] - c['frac_bbox'][1]) * img_h) for c in cells]
    w_over, h_over = [], []
    for cx, cy, w, h in px:
        gaps = [ox - cx for ox, oy, _, _ in px if 0 < ox - cx < 3 * w and abs(oy - cy) < 0.3 * h]
        if gaps:
            w_over.append(w / min(gaps))
            h_over.append(h / min(gaps))
    if len(w_over) < MIN_CELLS_TO_MEASURE:
        return None
    return _median(w_over), _median(h_over)


def canonical_boxes(cells, img_w, img_h, shape=DEFAULT_BOX_SHAPE):
    """Return `cells` with boxes rescaled, about their centres, to the target shape.
    `shape` is a key of BOX_SHAPES, or None to leave the labels as the dataset drew them.
    Pages already close to the target, or too sparse to measure, are returned unchanged."""
    if shape is None:
        return cells
    target_w, target_h = BOX_SHAPES[shape]
    measured = box_shape_over_pitch(cells, img_w, img_h)
    if measured is None:
        return cells
    if abs(measured[0] - target_w) <= BOX_SHAPE_TOLERANCE * target_w \
            and abs(measured[1] - target_h) <= BOX_SHAPE_TOLERANCE * target_h:
        return cells
    scale_x = target_w / measured[0]
    scale_y = target_h / measured[1]
    out = []
    for c in cells:
        l, t, r, b = c['frac_bbox']
        cx, cy = (l + r) / 2, (t + b) / 2
        hw, hh = (r - l) / 2 * scale_x, (b - t) / 2 * scale_y
        out.append({**c, 'frac_bbox': (max(0.0, cx - hw), max(0.0, cy - hh),
                                       min(1.0, cx + hw), min(1.0, cy + hh))})
    return out


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
