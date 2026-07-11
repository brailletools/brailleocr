#!/usr/bin/env python3
"""
Test YOLOv8 Braille cell detection on sample images.

Usage:
  python test_braille_yolo.py [image_or_dir] [--lang TABLE] [--no-contrast-search]

  --lang: liblouis table name, e.g.
      en-ueb-g2.ctb  (UEB Grade 2, default)
      en-ueb-g1.ctb  (UEB Grade 1)
      en-us-g2.ctb   (US English Grade 2)

  --no-contrast-search: skip automatic contrast optimisation (faster)

Class name encoding:
  Model classes are 6-char binary strings 'b1b2b3b4b5b6' where b1=dot1
  (top-left) .. b6=dot6 (bot-right). Reversing then int(,2) gives the
  Unicode Braille offset (U+2800+val).

Detection strategy:
  1. Run model at low threshold (conf=0.15) to capture near-misses.
  2. Treat cells with conf≥HIGH_CONF as reliable; use them to fit a per-row grid.
  3. For each expected grid position with no reliable detection, promote the
     best low-conf candidate nearby ("grid-guided rescue").
  4. Annotated image shows: green/yellow/red = reliable (by confidence),
     cyan = rescued by grid fill, grey outlines = inferred spaces.
"""
import argparse
import re
import statistics
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import PIL.Image
import PIL.ImageDraw
import PIL.ImageEnhance
import PIL.ImageFont
import PIL.ImageOps
from ultralytics import YOLO
from spellchecker import SpellChecker
from liblouis_env import get_lou_translate

MODEL_PATH    = '/tmp/yolov8-braille/yolov8m.pt'
SAMPLE_DIR    = Path('/Users/jmankoff/Research/nonvisual/braille/braille2latex/sample-data')
OUT_DIR       = Path('/tmp/braille-yolo-results')
LOU_TRANSLATE = get_lou_translate()

LOW_CONF   = 0.05   # first-pass threshold — collect everything above this
HIGH_CONF  = 0.30   # threshold for "reliable" cells used to fit the grid
CROP_CONF  = 0.001  # threshold used when re-running on a single-cell crop

# Braille dot layout:  dot1 dot4 / dot2 dot5 / dot3 dot6
DOT_POS = [(0,0),(1,0),(2,0),(0,1),(1,1),(2,1)]

# ─── helpers ─────────────────────────────────────────────────────────────────

def bits_to_braille(bits6: str) -> str:
    return chr(0x2800 + int(bits6[::-1], 2))

def pil_to_cv(img):
    return cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2BGR)

def cv_to_pil(arr):
    return PIL.Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))

# ─── contrast optimisation ───────────────────────────────────────────────────

CONTRAST_VALUES = [0.7, 1.0, 1.4, 2.0, 3.0]

# Braille dot positions within a cell, as (dx, dy) fractions of (cw, ch).
# Derived from standard cell geometry: 2.5mm dot separation, ~1.6mm dot
# diameter, bounding box ≈ 4.1mm wide × 6.6mm tall (full 6-dot cell).
# Left column offset: ±1.25/4.1 ≈ ±0.30; row offsets: ±2.5/6.6 ≈ ±0.38.
BRAILLE_DOT_OFFSETS = [
    (-0.28, -0.35),  # dot 1: top-left
    (-0.28,  0.00),  # dot 2: mid-left
    (-0.28,  0.35),  # dot 3: bot-left
    ( 0.28, -0.35),  # dot 4: top-right
    ( 0.28,  0.00),  # dot 5: mid-right
    ( 0.28,  0.35),  # dot 6: bot-right
]
DOT_SAMPLE_R_FRAC = 0.10  # sampling radius as fraction of min(cw, ch)

def apply_clahe(img, clip=3.0, grid=(16, 16)):
    arr = pil_to_cv(img)
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid).apply(l)
    return cv_to_pil(cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR))

def clahe_crop(pil_crop):
    """Stronger, fine-grained CLAHE for small spatially-localised crops."""
    return apply_clahe(pil_crop, clip=6.0, grid=(4, 4))

# ─── pixel-level dot detection ───────────────────────────────────────────────

def _dot_delta(gray, cx, cy, cw, ch, dx_frac, dy_frac):
    """
    Compute the dot strength at a candidate position as the maximum of three
    lighting-specific signals:

      brightness  — dot_mean − paper_low   (dot brighter than paper)
      darkness    — paper_high − dot_mean  (dot darker than paper / shadow)
      contrast    — max(patch) − min(patch) (highlight + shadow both present)

    Using the max makes the detector robust to the actual lighting direction:
    top-lit photos score on brightness, side/bottom-lit on darkness, and
    oblique lighting that shows both scores on contrast.

    paper_low  = 25th pct of cell bounding box (dark inter-dot paper)
    paper_high = 75th pct of cell bounding box (bright paper away from dots)
    The contrast window is 1.5× DOT_SAMPLE_R_FRAC to capture both highlight
    and shadow in a single patch.

    Returns (0.0, strength) for API compatibility with calibration code.
    """
    h, w = gray.shape
    r = max(2, int(min(cw, ch) * DOT_SAMPLE_R_FRAC * 1.5))

    # Cell-level paper reference
    x1, y1 = max(0, int(cx - cw / 2)), max(0, int(cy - ch / 2))
    x2, y2 = min(w, int(cx + cw / 2)), min(h, int(cy + ch / 2))
    cell_patch = gray[y1:y2, x1:x2]
    if cell_patch.size == 0:
        return 0.0, 0.0
    paper_low  = float(np.percentile(cell_patch, 25))
    paper_high = float(np.percentile(cell_patch, 75))

    dot_x = int(cx + dx_frac * cw)
    dot_y = int(cy + dy_frac * ch)
    if not (r <= dot_x < w - r and r <= dot_y < h - r):
        return 0.0, 0.0
    patch = gray[dot_y - r:dot_y + r, dot_x - r:dot_x + r]
    if patch.size == 0:
        return 0.0, 0.0

    dot_mean   = float(np.mean(patch))
    brightness = dot_mean - paper_low    # positive → brighter than paper
    darkness   = paper_high - dot_mean   # positive → darker than paper
    contrast   = float(np.max(patch) - np.min(patch))  # highlight + shadow

    return 0.0, max(brightness, darkness, contrast)


def calibrate_dot_threshold(gray, hi_cells):
    """
    Use high-confidence cells with known dot patterns to calibrate the
    local-contrast threshold that separates a present dot from flat paper.

    A raised dot has high local contrast (highlight + shadow); absent positions
    have low contrast (uniform paper).  We use the 25th percentile of the
    present-dot distribution so a candidate must match at least the bottom
    quarter of confirmed dots — conservative enough to reject paper texture.

    Returns the threshold (floored at 5.0 to guard against degenerate images).
    """
    present_deltas, absent_deltas = [], []
    for cell in hi_cells:
        cx, cy, cw, ch = cell['cx'], cell['cy'], cell['w'], cell['h']
        bits = cell['bits']
        for i, (dx_frac, dy_frac) in enumerate(BRAILLE_DOT_OFFSETS):
            _, delta = _dot_delta(gray, cx, cy, cw, ch, dx_frac, dy_frac)
            (present_deltas if bits[i] == '1' else absent_deltas).append(delta)

    if not present_deltas:
        return 20.0  # safe fallback
    return max(float(np.percentile(present_deltas, 25)), 5.0)


def snap_to_cell_peak(gray, ex, ey, cw, ch):
    """
    Scan a ±50%×cw / ±40%×ch neighbourhood around the grid-estimated
    position (ex, ey) to find the offset that maximises the SUM of all
    6 dot-position signals.  Summing over the full cell pattern finds
    actual Braille cell centres (multiple dots bright) rather than
    random bright spots (only one signal elevated).

    Returns (refined_x, refined_y, max_single_signal).
    Step size is ~6% of the smaller cell dimension.
    """
    search_dx = int(cw * 0.20)
    search_dy = int(ch * 0.15)
    step = max(2, int(min(cw, ch) * 0.06))

    best_sum = -1.0
    best_peak = 0.0
    best_x, best_y = ex, ey

    for dy in range(-search_dy, search_dy + 1, step):
        for dx in range(-search_dx, search_dx + 1, step):
            tx, ty = ex + dx, ey + dy
            sigs = [_dot_delta(gray, tx, ty, cw, ch, dfx, dfy)[1]
                    for dfx, dfy in BRAILLE_DOT_OFFSETS]
            total = sum(sigs)
            if total > best_sum:
                best_sum = total
                best_peak = max(sigs)
                best_x, best_y = tx, ty

    return best_x, best_y, best_peak


def pixel_cell_present(gray, cx, cy, cw, ch, dot_threshold):
    """
    Return True if ≥2 of the 6 dot positions are above dot_threshold.
    Requiring 2 dots guards against single-point noise (paper texture,
    bleed-through, or a neighbour's column landing in this cell's box).
    """
    hits = 0
    for dx_frac, dy_frac in BRAILLE_DOT_OFFSETS:
        _, delta = _dot_delta(gray, cx, cy, cw, ch, dx_frac, dy_frac)
        if delta > dot_threshold:
            hits += 1
            if hits >= 2:
                return True
    return False


def gap_pixel_recover(img, model, empties, hi_cells, known_cells=None):
    """
    For 'gap' empty positions (within a line), use pixel-level dot detection
    to distinguish missed cells from word spaces, then classify any confirmed
    cells with a multi-contrast YOLO crop.

    Steps for each gap position:
      1. Snap to local brightness peak within ±50%×cw, ±40%×ch to correct for
         page tilt and uneven character spacing.
      2. Reject if the snapped position overlaps an already-detected cell.
      3. Check ≥2 dot positions above calibrated threshold; skip if not.
      4. Classify with multi-contrast YOLO crop.
    """
    if not empties:
        return []

    gray = cv2.cvtColor(pil_to_cv(img), cv2.COLOR_BGR2GRAY)
    dot_threshold = calibrate_dot_threshold(gray, hi_cells)
    known = known_cells or []

    recovered = []
    img_w, img_h = img.size

    for entry in empties:
        ex, ey, cw, ch, source = entry[:5]
        if source != 'gap':
            continue

        # Snap to the local brightness peak to correct grid positioning error.
        ex, ey, peak_sig = snap_to_cell_peak(gray, ex, ey, cw, ch)

        # Reject if the snapped position is too close to an already-detected cell.
        if any(abs(ex - c['cx']) < cw * 0.7 and abs(ey - c['cy']) < ch * 0.7
               for c in known):
            continue

        if not pixel_cell_present(gray, ex, ey, cw, ch, dot_threshold):
            continue  # word space — nothing embossed here

        # Pixel check says a cell is here; classify it with multi-contrast YOLO
        pad_x, pad_y = cw * 2.0, ch * 1.5
        x1 = max(0, int(ex - pad_x))
        y1 = max(0, int(ey - pad_y))
        x2 = min(img_w, int(ex + pad_x))
        y2 = min(img_h, int(ey + pad_y))

        crop = img.crop((x1, y1, x2, y2))
        scale = max(1.0, 96.0 / cw)
        new_size = (int(crop.width * scale), int(crop.height * scale))
        cx_target = (ex - x1) * scale
        cy_target = (ey - y1) * scale

        best_cell = None
        with tempfile.TemporaryDirectory() as tmpdir:
            for c in CONTRAST_VALUES:
                variant = PIL.ImageEnhance.Contrast(crop).enhance(c)
                crop_up = variant.resize(new_size, PIL.Image.LANCZOS)
                tmp = Path(tmpdir) / 'crop.jpg'
                crop_up.save(tmp, quality=95)
                results = model(str(tmp), verbose=False,
                                conf=CROP_CONF, max_det=20)

                if results[0].boxes is None or len(results[0].boxes) == 0:
                    continue

                for box, cls, conf in zip(results[0].boxes.xyxy,
                                          results[0].boxes.cls,
                                          results[0].boxes.conf):
                    bx1, by1, bx2, by2 = box.tolist()
                    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
                    dist = abs(bcx - cx_target) + abs(bcy - cy_target)
                    if dist < cw * scale * 0.6 and float(conf) > (
                            best_cell['conf'] if best_cell else -1):
                        bits6 = model.names[int(cls)]
                        best_cell = {
                            'cx': ex, 'cy': ey, 'h': ch, 'w': cw,
                            'char':    bits_to_braille(bits6),
                            'bits':    bits6,
                            'conf':    float(conf),
                            'rescued': True,
                        }

        if best_cell:
            recovered.append(best_cell)

    return recovered


def best_contrast(img_path, model, max_det):
    """
    Try CLAHE + contrast multipliers; pick whichever gives the most cells
    at HIGH_CONF threshold. Uses temp files so YOLO resizes consistently.
    """
    original = PIL.ImageOps.exif_transpose(PIL.Image.open(img_path)).convert('RGB')
    candidates = [('clahe', apply_clahe(original))]
    for c in CONTRAST_VALUES:
        candidates.append((f'contrast×{c}',
                           PIL.ImageEnhance.Contrast(original).enhance(c)))

    best_img, best_n, best_label = original, -1, 'original'
    with tempfile.TemporaryDirectory() as tmpdir:
        for label, candidate in candidates:
            tmp = Path(tmpdir) / 'candidate.jpg'
            candidate.save(tmp, quality=92)
            results = model(str(tmp), verbose=False, max_det=max_det,
                            conf=HIGH_CONF)
            n = len(results[0].boxes) if results[0].boxes else 0
            if n > best_n:
                best_n, best_img, best_label = n, candidate, label
    return best_img, best_n, best_label

# ─── confidence-driven second pass ───────────────────────────────────────────

def low_conf_repass(img, model, cells, max_det):
    """
    Find spatial clusters of cells that scored below HIGH_CONF. For each
    cluster, extract the bounding region from the image, apply stronger local
    CLAHE, re-run YOLO on that patch, and merge any newly high-confidence
    detections back into the cell list.

    This closes the feedback loop: the model's own uncertainty drives where
    to apply extra contrast enhancement.

    Returns the augmented cell list (original cells + newly promoted ones).
    """
    low_cells = [c for c in cells if c['conf'] < HIGH_CONF]
    if not low_cells:
        return cells

    img_w, img_h = img.size
    avg_h = statistics.median(c['h'] for c in cells)
    avg_w = statistics.median(c['w'] for c in cells)

    # Cluster low-conf cells by proximity (simple row-then-column grouping).
    # Use the same row-clustering as group_into_lines, then split each row
    # into horizontal runs separated by more than 3 cell-widths.
    rows = group_into_lines(low_cells)
    patches = []   # (x1, y1, x2, y2) image-space rectangles to re-examine

    for row in rows:
        sx = sorted(row, key=lambda c: c['cx'])
        run = [sx[0]]
        for cell in sx[1:]:
            if cell['cx'] - run[-1]['cx'] < avg_w * 3:
                run.append(cell)
            else:
                patches.append(run)
                run = [cell]
        patches.append(run)

    existing_cx = {(round(c['cx']), round(c['cy'])) for c in cells}
    new_cells = []

    for run in patches:
        # Only retry small isolated clusters — large clusters are either
        # intentional spaces or genuinely faint; aggressive CLAHE on a wide
        # patch picks up paper grain as false positives.
        if len(run) > 3:
            continue

        cx_min = min(c['cx'] for c in run) - avg_w * 1.5
        cx_max = max(c['cx'] for c in run) + avg_w * 1.5
        cy_min = min(c['cy'] for c in run) - avg_h * 1.2
        cy_max = max(c['cy'] for c in run) + avg_h * 1.2
        x1 = max(0, int(cx_min))
        y1 = max(0, int(cy_min))
        x2 = min(img_w, int(cx_max))
        y2 = min(img_h, int(cy_max))

        patch = img.crop((x1, y1, x2, y2))
        # Moderate CLAHE — enough to lift borderline cells without amplifying grain
        patch = apply_clahe(patch, clip=4.0, grid=(4, 4))

        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            patch.save(tmp.name, quality=95)
            results = model(tmp.name, verbose=False,
                            conf=HIGH_CONF, max_det=max_det)

        if not results[0].boxes:
            continue

        for box, cls, conf in zip(results[0].boxes.xyxy,
                                   results[0].boxes.cls,
                                   results[0].boxes.conf):
            bx1, by1, bx2, by2 = box.tolist()
            # Map coordinates back to full-image space
            gcx = (bx1 + bx2) / 2 + x1
            gcy = (by1 + by2) / 2 + y1
            key = (round(gcx), round(gcy))
            if key in existing_cx:
                continue   # already in the cell list
            # Check it actually overlaps the cluster area
            if not (cx_min <= gcx <= cx_max and cy_min <= gcy <= cy_max):
                continue
            bits6 = model.names[int(cls)]
            new_cells.append({
                'cx': gcx, 'cy': gcy,
                'h': by2 - by1, 'w': bx2 - bx1,
                'char': bits_to_braille(bits6),
                'bits': bits6,
                'conf': float(conf),
                'rescued': True,
            })
            existing_cx.add(key)

    return cells + new_cells


# ─── detection with grid-guided rescue ───────────────────────────────────────

def run_detection(img, model, max_det):
    """
    Run model at LOW_CONF to collect all candidates, returned as a list of
    dicts with keys: cx, cy, h, w, char, bits, conf, rescued (bool).
    """
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img.save(tmp.name, quality=92)
        results = model(tmp.name, verbose=False, max_det=max_det, conf=LOW_CONF)

    cells = []
    if results[0].boxes is None:
        return cells
    for box, cls, conf in zip(results[0].boxes.xyxy,
                               results[0].boxes.cls,
                               results[0].boxes.conf):
        x1, y1, x2, y2 = box.tolist()
        bits6 = model.names[int(cls)]
        cells.append({
            'cx': (x1+x2)/2, 'cy': (y1+y2)/2,
            'h': y2-y1,       'w': x2-x1,
            'char': bits_to_braille(bits6),
            'bits': bits6,    'conf': float(conf),
            'rescued': False,
        })
    return cells

def group_into_lines(cells):
    if not cells:
        return []
    avg_h = statistics.median(c['h'] for c in cells)
    thresh = avg_h * 0.55
    sorted_y = sorted(cells, key=lambda c: c['cy'])
    lines, cur = [], [sorted_y[0]]
    for cell in sorted_y[1:]:
        if abs(cell['cy'] - cur[-1]['cy']) < thresh:
            cur.append(cell)
        else:
            lines.append(cur)
            cur = [cell]
    lines.append(cur)
    return lines

def _char_spacing(line_x, avg_w):
    """
    Estimate true inter-character spacing for a sorted line of cells.
    Uses the 10th-percentile of inter-cell gaps so that word spaces and
    multi-cell gaps (caused by missed detections) don't inflate the estimate.
    """
    gaps = [line_x[i+1]['cx'] - line_x[i]['cx']
            for i in range(len(line_x)-1)]
    if not gaps:
        return avg_w * 1.1
    gaps_s = sorted(gaps)
    return gaps_s[max(0, len(gaps_s)//10)]   # 10th percentile


def grid_fill(all_cells):
    """
    Use high-confidence cells to fit a per-row grid, then promote the best
    low-confidence candidate for each grid position that has no reliable hit.

    Searches for missing cells in three regions per line:
      - Between consecutive detected cells (gaps)
      - Before the first detected cell (extrapolate left to text-block boundary)
      - After the last detected cell (extrapolate right to text-block boundary)

    Returns (reliable+rescued cells, list of (expected_x, expected_y, cw, ch)
    for positions where NO candidate exists even at LOW_CONF).
    """
    reliable = [c for c in all_cells if c['conf'] >= HIGH_CONF]
    low      = [c for c in all_cells if c['conf'] <  HIGH_CONF]

    if not reliable:
        return all_cells, []

    avg_h = statistics.median(c['h'] for c in reliable)
    avg_w = statistics.median(c['w'] for c in reliable)
    lines = group_into_lines(reliable)

    # Estimate the text-block left/right boundaries from all lines so we know
    # how far to extrapolate at line ends.
    line_lists = [sorted(ln, key=lambda c: c['cx']) for ln in lines
                  if len(ln) >= 3]
    if line_lists:
        text_left  = statistics.median(ln[0]['cx']  for ln in line_lists)
        text_right = statistics.median(ln[-1]['cx'] for ln in line_lists)
    else:
        text_left = text_right = None

    rescued = []
    used    = set()
    empties = []

    # Minimum confidence for rescuing ⠿ (all-dots). This cell class is the
    # model's fallback for ambiguous patterns, so low-conf ⠿ rescues are
    # almost always false positives.
    ALL_DOTS_MIN_CONF = 0.15

    def _try_position(ex, row_y, source):
        """Attempt to fill one expected grid position from low-conf pool.
        source: 'gap' (within line) or 'edge' (extrapolated beyond line end).
        """
        candidates = [
            (j, c) for j, c in enumerate(low)
            if j not in used
            and abs(c['cx'] - ex)    < cs    * 0.45
            and abs(c['cy'] - row_y) < avg_h * 0.55
            # Don't rescue an all-dots cell unless it's reasonably confident
            and not (c['bits'] == '111111' and c['conf'] < ALL_DOTS_MIN_CONF)
        ]
        if candidates:
            j, best = max(candidates, key=lambda jc: jc[1]['conf'])
            rescued.append({**best, 'rescued': True})
            used.add(j)
        else:
            empties.append((ex, row_y, avg_w, avg_h, source, CROP_CONF))

    for line in lines:
        line_x = sorted(line, key=lambda c: c['cx'])
        if len(line_x) < 2:
            continue

        cs    = _char_spacing(line_x, avg_w)
        if cs < avg_w * 0.6:
            continue
        row_y = statistics.median(c['cy'] for c in line_x)

        # ── gaps between consecutive detected cells ───────────────────────
        for i in range(len(line_x) - 1):
            gap       = line_x[i+1]['cx'] - line_x[i]['cx']
            n_missing = round(gap / cs) - 1
            for k in range(1, max(0, n_missing) + 1):
                _try_position(line_x[i]['cx'] + k * cs, row_y, 'gap')

        # ── extrapolate left of first detected cell ───────────────────────
        if text_left is not None:
            ex = line_x[0]['cx'] - cs
            while ex > text_left - cs * 0.5:
                _try_position(ex, row_y, 'edge')
                ex -= cs

        # ── extrapolate right of last detected cell ───────────────────────
        if text_right is not None:
            ex = line_x[-1]['cx'] + cs
            while ex < text_right + cs * 0.5:
                _try_position(ex, row_y, 'edge')
                ex += cs

    return reliable + rescued, empties


def crop_recover(img, model, empties):
    """
    For 'edge' grid positions (extrapolated beyond a line's first/last
    detected cell), crop the region, try every contrast level, and accept
    whichever level produces the highest-confidence detection near the
    expected centre.

    Multi-contrast search is restricted to 'edge' positions because gap
    positions (within a line) cannot be distinguished from word spaces by
    confidence alone: contrast×3.0 on empty paper reliably scores above 0.05,
    which is the same bar a faint real cell needs to clear.

    CLAHE is excluded from the crop contrast variants — it amplifies paper
    grain into false Braille cell detections on small crops.

    Returns a list of newly recovered cells.
    """
    if not empties:
        return []

    recovered = []
    img_w, img_h = img.size

    for entry in empties:
        ex, ey, cw, ch, source = entry[:5]

        if source != 'edge':
            continue

        # Crop: 2 cells of context on each side, 1.5 cell above/below
        pad_x, pad_y = cw * 2.0, ch * 1.5
        x1 = max(0, int(ex - pad_x))
        y1 = max(0, int(ey - pad_y))
        x2 = min(img_w, int(ex + pad_x))
        y2 = min(img_h, int(ey + pad_y))

        crop = img.crop((x1, y1, x2, y2))

        # Scale so the target cell is ~96 px wide in the crop
        scale    = max(1.0, 96.0 / cw)
        new_size = (int(crop.width * scale), int(crop.height * scale))

        # Expected centre in crop-space (scaled)
        cx_target = (ex - x1) * scale
        cy_target = (ey - y1) * scale

        # Try every contrast level; keep the highest-confidence detection
        # that is within 0.6 cell-widths of the expected centre.
        # CLAHE excluded — amplifies grain on small crops.
        best_cell = None

        with tempfile.TemporaryDirectory() as tmpdir:
            for c in CONTRAST_VALUES:
                variant = PIL.ImageEnhance.Contrast(crop).enhance(c)
                crop_up = variant.resize(new_size, PIL.Image.LANCZOS)
                tmp = Path(tmpdir) / 'crop.jpg'
                crop_up.save(tmp, quality=95)
                results = model(str(tmp), verbose=False,
                                conf=CROP_CONF, max_det=20)

                if results[0].boxes is None or len(results[0].boxes) == 0:
                    continue

                for box, cls, conf in zip(results[0].boxes.xyxy,
                                           results[0].boxes.cls,
                                           results[0].boxes.conf):
                    bx1, by1, bx2, by2 = box.tolist()
                    bcx, bcy = (bx1+bx2)/2, (by1+by2)/2
                    dist = abs(bcx - cx_target) + abs(bcy - cy_target)
                    if dist < cw * scale * 0.6 and float(conf) > (
                            best_cell['conf'] if best_cell else -1):
                        bits6 = model.names[int(cls)]
                        best_cell = {
                            'cx': ex, 'cy': ey,
                            'h': ch,  'w': cw,
                            'char':    bits_to_braille(bits6),
                            'bits':    bits6,
                            'conf':    float(conf),
                            'rescued': True,
                        }

        if best_cell and best_cell['conf'] >= CROP_CONF:
            recovered.append(best_cell)

    return recovered

# ─── space inference ─────────────────────────────────────────────────────────

def insert_spaces(line_cells, avg_cell_w):
    """
    Insert synthetic space cells where a gap exceeds the per-line median
    inter-character spacing by more than half a cell width.
    """
    if len(line_cells) < 2:
        return line_cells
    line = sorted(line_cells, key=lambda c: c['cx'])
    gaps = [line[i+1]['cx'] - line[i]['cx'] for i in range(len(line)-1)]
    median_gap = statistics.median(gaps)
    space_thresh = median_gap + avg_cell_w * 0.5

    result = [line[0]]
    for i in range(1, len(line)):
        gap = line[i]['cx'] - line[i-1]['cx']
        if gap > space_thresh:
            n_sp = max(1, round((gap - median_gap) / avg_cell_w))
            for _ in range(n_sp):
                result.append({'cx': None, 'cy': line[i]['cy'],
                               'h': line[i]['h'], 'char': '⠀',
                               'bits': '000000', 'conf': 0.0,
                               'is_space': True, 'rescued': False})
        result.append(line[i])
    return result

# ─── indicator-gap recovery ──────────────────────────────────────────────────

# UEB indicator cells — their class bits6 name and the liblouis \NNN/ token
# they produce when orphaned (partner cell missing).
# Capital indicator ⠠ (dot6)      → \6/
# Grade-1 indicator ⠰ (dots 56)   → \56/
# Dots-45 cell ⠘                   → \45/
# Dots-456 cell ⠸                  → \456/
# Number indicator ⠼ (dots 3456)  → \3456/
UEB_INDICATORS = {'000001', '000011', '000110', '000111', '001111'}

# Regex that matches any liblouis orphaned-indicator token e.g. \456/
_ORPHAN_RE = re.compile(r'\\(\d+)/')

def indicator_recovery(img, model, cells):
    """
    After main detection, look for UEB indicator cells whose immediately
    following cell position has an abnormally large gap — the partner cell is
    probably missing.  Run targeted crop recovery at that position.

    Returns a list of newly recovered cells (same format as crop_recover).
    """
    if not cells:
        return []

    avg_w = statistics.median(c['w'] for c in cells)
    lines = group_into_lines(cells)

    targets = []   # (ex, ey, cw, ch, 'edge') for crop_recover
    for line in lines:
        sl = sorted(line, key=lambda c: c['cx'])
        if len(sl) < 2:
            continue
        cs = _char_spacing(sl, avg_w)
        if cs < avg_w * 0.6:
            continue

        for i, cell in enumerate(sl):
            if cell['bits'] not in UEB_INDICATORS:
                continue
            if i + 1 < len(sl):
                gap = sl[i + 1]['cx'] - cell['cx']
                # More than one cell-width gap → partner cell likely missing
                if gap > cs * 1.4:
                    targets.append((cell['cx'] + cs, cell['cy'],
                                    cell['w'], cell['h'], 'edge', CROP_CONF))
            else:
                # Indicator at line end — try the position immediately after
                targets.append((cell['cx'] + cs, cell['cy'],
                                cell['w'], cell['h'], 'edge', CROP_CONF))

    if not targets:
        return []
    return crop_recover(img, model, targets)


# ─── translation ─────────────────────────────────────────────────────────────

def braille_to_text(braille_lines, table):
    results = []
    for line in braille_lines:
        stripped = line.replace('⠀', ' ').strip()
        if not stripped:
            results.append('')
            continue
        try:
            proc = subprocess.run(
                [LOU_TRANSLATE, '-b', table],
                input=stripped, capture_output=True, text=True, timeout=10)
            out = proc.stdout.strip() if proc.returncode == 0 \
                  else f'(err: {proc.stderr.strip()})'
        except Exception as e:
            out = f'(error: {e})'
        results.append(out)
    return results

# ─── post-translation cleanup ────────────────────────────────────────────────

_spell = SpellChecker()
# Domain / short words the spell checker would otherwise mangle
_spell.word_frequency.load_words([
    'skiing', 'leashes', 'vibrotactile', 'tactile', 'slope', 'slopes',
    'strap', 'straps', 'system', 'systems', 'function', 'functions',
    'signal', 'signals', 'skier', 'skiers', 'retractable',
])

def clean_translation(text, spellcheck=True):
    """
    Two-pass cleanup of liblouis back-translation output:

    1. Spell-check (optional): fix all-lowercase words (≥4 chars) that the
       spell checker recognises as errors.  Uppercase / mixed-case words are
       left alone so domain terms (ProTactile, CI, DB, vibrotactile …) are
       never stomped.

    2. Strip any remaining orphaned UEB indicator tokens (\\NNN/) that
       indicator_recovery couldn't fill.  These are artefacts of a partner cell
       being missed, not real text.
    """
    if spellcheck:
        def _fix(m):
            w = m.group(0)
            if _spell.unknown([w]):
                correction = _spell.correction(w)
                if correction and correction != w:
                    return correction
            return w
        text = re.sub(r'\b[a-z]{6,}\b', _fix, text)

    # Always strip orphaned indicator tokens — they are never real output text
    text = _ORPHAN_RE.sub('', text)

    return text


# ─── visualisation ───────────────────────────────────────────────────────────

def save_annotated(img, cells, stem):
    """
    Colour coding:
      green/yellow/red  = reliable detection (by confidence)
      cyan              = rescued by grid-guided fill (was below HIGH_CONF)
      spaces not drawn

    Dot overlay: for each cell, draws 6 small circles at the Braille dot
    positions.  Filled = dot recognised as raised ('1'); hollow = absent ('0').
    """
    ann  = img.copy()
    draw = PIL.ImageDraw.Draw(ann)
    try:
        font = PIL.ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 14)
    except Exception:
        font = PIL.ImageFont.load_default()

    for c in cells:
        if c.get('is_space'):
            continue
        if c.get('rescued'):
            color = (0, 220, 220)   # cyan
        else:
            cf = c['conf']
            color = (int(255*(1-cf)), int(255*cf), 0)

        cw = c.get('w', c['h'] * 0.6)
        ch = c['h']
        hw, hh = cw / 2, ch / 2
        cx, cy = c['cx'], c['cy']
        x1, y1 = int(cx - hw), int(cy - hh)
        x2, y2 = int(cx + hw), int(cy + hh)
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1+2, y1+1), c['char'], fill=color, font=font)

        # Draw recognised dot pattern within the cell bounding box
        bits = c.get('bits', '000000')
        dot_r = max(2, int(min(cw, ch) * 0.09))
        for dot_i, (dx_frac, dy_frac) in enumerate(BRAILLE_DOT_OFFSETS):
            dx = int(cx + dx_frac * cw)
            dy = int(cy + dy_frac * ch)
            raised = bits[dot_i] == '1'
            if raised:
                draw.ellipse([dx-dot_r, dy-dot_r, dx+dot_r, dy+dot_r],
                             fill=color, outline=color)
            else:
                draw.ellipse([dx-dot_r, dy-dot_r, dx+dot_r, dy+dot_r],
                             fill=None, outline=color)

    ann.save(OUT_DIR / f'{stem}_annotated.jpg', quality=85)

def save_dot_grid(cells, stem):
    CELL_PX, DOT_R, PAD, COLS = 60, 7, 6, 40
    try:
        sf = PIL.ImageFont.truetype('/System/Library/Fonts/Helvetica.ttc', 10)
    except Exception:
        sf = PIL.ImageFont.load_default()

    nrows  = (len(cells) + COLS - 1) // COLS
    grid   = PIL.Image.new('RGB',
                           (COLS*CELL_PX+PAD*2, nrows*CELL_PX+PAD*2),
                           (240,240,240))
    gd     = PIL.ImageDraw.Draw(grid)
    dot_w  = (CELL_PX-16) // 2
    dot_h  = (CELL_PX-20) // 3

    for idx, cell in enumerate(cells):
        col, row = idx % COLS, idx // COLS
        ox = PAD + col*CELL_PX
        oy = PAD + row*CELL_PX
        bits6   = cell['bits']
        is_sp   = cell.get('is_space',  False)
        rescued = cell.get('rescued',   False)

        for dot_i, (dr, dc) in enumerate(DOT_POS):
            cx = ox + 4 + dc*dot_w + dot_w//2
            cy = oy + 4 + dr*dot_h + dot_h//2
            if is_sp:
                fill, outline = (230,230,230), (180,180,180)
            elif rescued:
                raised  = bits6[dot_i] == '1'
                fill    = (0,160,160) if raised else (180,230,230)
                outline = (0,100,100)
            else:
                raised  = bits6[dot_i] == '1'
                fill    = (30,30,30)    if raised else (210,210,210)
                outline = (0,0,0)       if raised else (160,160,160)
            gd.ellipse([cx-DOT_R, cy-DOT_R, cx+DOT_R, cy+DOT_R],
                       fill=fill, outline=outline)

        lc = (180,180,180) if is_sp else (0,140,140) if rescued else (0,0,200)
        gd.text((ox+CELL_PX//2-5, oy+CELL_PX-14), cell['char'], fill=lc, font=sf)

    grid.save(OUT_DIR / f'{stem}_dots.jpg', quality=90)

# ─── main processing ─────────────────────────────────────────────────────────

# ─── MobileNetV2 cell classifier ─────────────────────────────────────────────

_CELL_CLASSIFIER     = None   # loaded lazily
_CELL_CLF_DEVICE     = None
_CELL_CLF_TRANSFORM  = None
_CLASSIFIER_CROP_PAD = 0.10   # matches extract_crops.py PADDING

def load_cell_classifier(model_path):
    """Load the MobileNetV2 dot classifier. Returns (model, device, transform)."""
    import torch
    from torchvision import models, transforms

    device = (
        torch.device('mps')  if torch.backends.mps.is_available() else
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('cpu')
    )
    import torch.nn as nn
    net = models.mobilenet_v2(weights=None)
    net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(net.last_channel, 6))
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std =[0.229, 0.224, 0.225]),
    ])
    return net, device, tf


def reclassify_cells(img, cells, clf, clf_device, clf_tf):
    """
    Replace YOLO's bits/char in each cell with MobileNetV2 classifier output.
    Processes in one batched forward pass.
    """
    import torch

    active = [c for c in cells if not c.get('is_space')]
    if not active:
        return cells

    iw, ih = img.size
    crops = []
    for c in active:
        pad_x = c['w'] * _CLASSIFIER_CROP_PAD
        pad_y = c['h'] * _CLASSIFIER_CROP_PAD
        x0 = max(0,  c['cx'] - c['w'] / 2 - pad_x)
        y0 = max(0,  c['cy'] - c['h'] / 2 - pad_y)
        x1 = min(iw, c['cx'] + c['w'] / 2 + pad_x)
        y1 = min(ih, c['cy'] + c['h'] / 2 + pad_y)
        crop = img.crop((x0, y0, x1, y1)).convert('RGB')
        crops.append(clf_tf(crop))

    batch = torch.stack(crops).to(clf_device)
    with torch.no_grad():
        logits = clf(batch)
        preds  = (logits.sigmoid() > 0.5).cpu().numpy().astype(int)

    for cell, pred_row in zip(active, preds):
        bits = ''.join(str(b) for b in pred_row)
        cell['bits'] = bits
        cell['char'] = bits_to_braille(bits)

    return cells


def _init_classifier(path):
    global _CELL_CLASSIFIER, _CELL_CLF_DEVICE, _CELL_CLF_TRANSFORM
    if _CELL_CLASSIFIER is None:
        print(f"  Loading cell classifier from {path} …")
        _CELL_CLASSIFIER, _CELL_CLF_DEVICE, _CELL_CLF_TRANSFORM = load_cell_classifier(path)


def process_image(img_path, model, lang_table, search_contrast,
                  spellcheck=True, max_det=2000, classifier_path=None):
    if search_contrast:
        img, n_hi, label = best_contrast(img_path, model, max_det)
        print(f"  Best contrast: {label} → {n_hi} high-conf cells")
    else:
        img = PIL.ImageOps.exif_transpose(
                PIL.Image.open(img_path)).convert('RGB')

    # Collect all candidates (low threshold)
    all_cells = run_detection(img, model, max_det)
    n_all  = len(all_cells)
    n_hi   = sum(1 for c in all_cells if c['conf'] >= HIGH_CONF)
    print(f"  Candidates: {n_all} total, {n_hi} high-conf (≥{HIGH_CONF})")

    if not all_cells:
        print("  No cells found.")
        return None

    # Keep the raw high-conf cells for dot-detection calibration below
    raw_hi_cells = [c for c in all_cells if c['conf'] >= HIGH_CONF]

    # Grid-guided rescue: promote low-conf candidates at expected positions
    cells, empties = grid_fill(all_cells)
    n_rescued = sum(1 for c in cells if c.get('rescued'))
    print(f"  After grid fill: {len(cells)} cells "
          f"({n_rescued} rescued, {len(empties)} empty positions remain)")

    # Edge crop recovery: multi-contrast YOLO on extrapolated line-end positions
    crop_cells = crop_recover(img, model, empties)
    cells += crop_cells
    if crop_cells:
        print(f"  Crop recovery: +{len(crop_cells)} additional cells rescued")

    # Gap pixel recovery: brightness-based dot detection to distinguish missed
    # mid-line cells from word spaces, then classify with YOLO
    gap_cells = gap_pixel_recover(img, model, empties, raw_hi_cells, known_cells=cells)
    cells += gap_cells
    if gap_cells:
        print(f"  Gap pixel recovery: +{len(gap_cells)} cells rescued")

    # ── Margin filter ────────────────────────────────────────────────────────
    # Remove cells that land outside the text block (e.g. ⠿ in blank margins).
    # Use 5th/95th percentile of high-conf cell positions as the bounding box.
    hi_cells = [c for c in cells if c['conf'] >= HIGH_CONF]
    if len(hi_cells) >= 20:
        xs = sorted(c['cx'] for c in hi_cells)
        ys = sorted(c['cy'] for c in hi_cells)
        n5  = max(0, len(xs) * 5  // 100)
        n95 = max(0, len(xs) * 95 // 100)
        _cx_min, _cx_max = xs[n5]  - xs[0]  * 0.05, xs[n95]  + xs[0]  * 0.05
        _cy_min, _cy_max = ys[n5]  - ys[0]  * 0.05, ys[n95]  + ys[0]  * 0.05
        avg_h = statistics.median(c['h'] for c in hi_cells)
        avg_w = statistics.median(c['w'] for c in hi_cells)
        # Generous margins: allow 3 cell-widths left/right, 2 heights top/bottom
        x_lo = xs[n5]  - avg_w * 3
        x_hi = xs[n95] + avg_w * 3
        y_lo = ys[n5]  - avg_h * 2
        y_hi = ys[n95] + avg_h * 2
        before = len(cells)
        cells = [c for c in cells
                 if x_lo <= c['cx'] <= x_hi and y_lo <= c['cy'] <= y_hi]
        n_removed = before - len(cells)
        if n_removed:
            print(f"  Margin filter: removed {n_removed} out-of-block cells")

    # ── Indicator-gap recovery ───────────────────────────────────────────────
    # For each UEB indicator cell (capital/number/grade-1 sign) that has an
    # abnormally large gap after it, the partner cell is probably missing.
    # Run targeted crop recovery there, then fold the result in.
    ind_cells = indicator_recovery(img, model, cells)
    if ind_cells:
        cells += ind_cells
        print(f"  Indicator recovery: +{len(ind_cells)} cells rescued")

    # ── Cell classifier ──────────────────────────────────────────────────────
    if classifier_path:
        _init_classifier(classifier_path)
        cells = reclassify_cells(img, cells,
                                 _CELL_CLASSIFIER, _CELL_CLF_DEVICE, _CELL_CLF_TRANSFORM)

    avg_cell_w = statistics.median(c['w'] for c in cells if not c.get('is_space'))
    save_annotated(img, cells, img_path.stem)

    lines = group_into_lines([c for c in cells if not c.get('is_space')])
    cells_with_spaces = []
    braille_lines = []
    for line in lines:
        lws = insert_spaces(line, avg_cell_w)
        cells_with_spaces.extend(lws)
        braille_lines.append(''.join(c['char'] for c in lws))

    save_dot_grid(cells_with_spaces, img_path.stem)

    translated = braille_to_text(braille_lines, lang_table)
    cleaned    = [clean_translation(t, spellcheck=spellcheck) for t in translated]
    return '\n'.join(cleaned)

# ─── entry point ─────────────────────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(exist_ok=True)
    parser = argparse.ArgumentParser(description='YOLOv8 Braille OCR tester')
    parser.add_argument('input', nargs='?', default=str(SAMPLE_DIR))
    parser.add_argument('--lang', default='en-ueb-g2.ctb')
    parser.add_argument('--no-contrast-search', action='store_true')
    parser.add_argument('--no-spellcheck', action='store_true',
                        help='Disable spell-correction of lowercase words')
    parser.add_argument('--classifier', default=None,
                        help='Path to MobileNetV2 cell_classifier.pt '
                             '(default: /tmp/braille-crops/cell_classifier.pt if it exists)')
    args = parser.parse_args()

    if args.classifier is None and Path('/tmp/braille-crops/cell_classifier.pt').exists():
        args.classifier = '/tmp/braille-crops/cell_classifier.pt'

    model  = YOLO(MODEL_PATH)
    inp    = Path(args.input)
    images = (sorted(inp.glob('*.jpeg')) + sorted(inp.glob('*.jpg'))) \
             if inp.is_dir() else [inp]

    spellcheck = not args.no_spellcheck
    print(f"Language: {args.lang}  |  Contrast search: {not args.no_contrast_search}"
          f"  |  Spell-check: {spellcheck}")
    print(f"Low-conf threshold: {LOW_CONF}  |  High-conf threshold: {HIGH_CONF}")
    print(f"Found {len(images)} image(s)\n{'='*60}")

    for img_path in images:
        print(f"\n--- {img_path.name} ---")
        text = process_image(img_path, model, args.lang,
                             not args.no_contrast_search,
                             spellcheck=spellcheck,
                             classifier_path=args.classifier)
        if text:
            print("Translated text:")
            print(text[:800])

    print(f"\n{'='*60}")
    print(f"Results in: {OUT_DIR}")
    print("  *_annotated.jpg — green/yellow/red=reliable, cyan=rescued")
    print("  *_dots.jpg      — teal dots=rescued cells, grey=inferred spaces")

if __name__ == '__main__':
    main()
