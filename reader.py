"""
Python reader core: scale-normalized tiled detection and dot classification.

What is left of the original `pipeline.py` command-line reader (retired 2026-11-27, see
AUDIT.md; it is in git history). The reader that ships is the JavaScript one in js/src,
used by the web editor. This module exists for the two jobs that still need Python:

  - braille-research's experiments/label_tool.py and label_batch.py pre-fill a page with predictions, so a
    person only corrects them (needs the PyTorch detector and classifier);
  - js/test/fixtures/gen_detection_ref.py makes the reference outputs the JS tests compare
    against, and experiments/classifier_on_gt.py there loads the classifier from here.

It deliberately has no recovery stages (grid-fill, crop and pixel recovery guessed at
cells the detector had not found; every cell they added was invented, AUDIT.md / RESEARCH.md
Experiments 1, 5, 6), no contrast search, no spell-correction and no translation.

Detection goes through a temporary JPEG, exactly as the original did, so the JS parity
fixtures stay comparable. (A JPEG round trip loses about 1% of detections compared with
passing pixels directly; RESEARCH.md Experiments 2 and 3.)
"""
import statistics
import tempfile

from dot_pattern_utils import MIN_NATIVE_TILE, TARGET_CELL_PX, TILE_SIZE, make_tile_boxes

LOW_CONF = 0.05                  # collect everything above this
HIGH_CONF = 0.30                 # reliable detections
CONTAINER_MIN_CANDIDATES = 3     # fewer detections than this are probably noise
NMS_IOU = 0.3                    # detector overlap threshold; same as js/src/detector.js
TILE_DEDUPE_IOU = 0.5            # merge threshold for cells seen in two overlapping tiles
TILE_EDGE_MARGIN_FRAC = 0.05     # a cell touching the crop edge of a tile is only partly
                                 # seen (falsely narrow): drop it and rely on the
                                 # overlapping neighbour tile
CLASSIFIER_CROP_PAD = 0.10       # must match extract_crops.PADDING

# Braille dot layout:  dot1 dot4 / dot2 dot5 / dot3 dot6


def bits_to_braille(bits6: str) -> str:
    return chr(0x2800 + int(bits6[::-1], 2))


def yolo_class_to_bits6(model, cls):
    """A single-class (localization-only) detector's class name is not a dot pattern;
    the classifier supplies the pattern, so return a blank placeholder."""
    if len(model.names) == 1:
        return '000000'
    return model.names[int(cls)]


def run_detection(img, model, max_det):
    """Detect at LOW_CONF. Returns dicts with cx, cy, h, w, char, bits, conf, rescued."""
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        img.save(tmp.name, quality=92)
        results = model(tmp.name, verbose=False, max_det=max_det, conf=LOW_CONF, iou=NMS_IOU)

    cells = []
    if results[0].boxes is None:
        return cells
    for box, cls, conf in zip(results[0].boxes.xyxy, results[0].boxes.cls, results[0].boxes.conf):
        x1, y1, x2, y2 = box.tolist()
        bits6 = yolo_class_to_bits6(model, cls)
        cells.append({
            'cx': (x1 + x2) / 2, 'cy': (y1 + y2) / 2,
            'h': y2 - y1, 'w': x2 - x1,
            'char': bits_to_braille(bits6),
            'bits': bits6, 'conf': float(conf),
            'rescued': False,
        })
    return cells


def _cell_iou(a, b):
    ax0, ay0 = a['cx'] - a['w'] / 2, a['cy'] - a['h'] / 2
    ax1, ay1 = a['cx'] + a['w'] / 2, a['cy'] + a['h'] / 2
    bx0, by0 = b['cx'] - b['w'] / 2, b['cy'] - b['h'] / 2
    bx1, by1 = b['cx'] + b['w'] / 2, b['cy'] + b['h'] / 2
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a, area_b = (ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0)
    return inter / (area_a + area_b - inter)


def _dedupe_cells(cells, iou_threshold=TILE_DEDUPE_IOU):
    """Collapse duplicate detections of one cell from overlapping tiles, keeping the
    more confident."""
    kept = []
    for c in sorted(cells, key=lambda c: c['conf'], reverse=True):
        if not any(_cell_iou(c, k) > iou_threshold for k in kept):
            kept.append(c)
    return kept


def run_detection_tiled(img, model, max_det, native_tile_size):
    """Detect on overlapping native_tile_size tiles (a single call when the image fits
    in one). See dot_pattern_utils for why the tile size is chosen the way it is."""
    boxes = make_tile_boxes(img.width, img.height, native_tile_size)
    if len(boxes) == 1:
        return run_detection(img, model, max_det)

    margin = native_tile_size * TILE_EDGE_MARGIN_FRAC
    all_cells = []
    for x0, y0, x1, y1 in boxes:
        tile_cells = run_detection(img.crop((x0, y0, x1, y1)), model, max_det)
        tw, th = x1 - x0, y1 - y0
        for c in tile_cells:
            lx0, ly0 = c['cx'] - c['w'] / 2, c['cy'] - c['h'] / 2
            lx1, ly1 = c['cx'] + c['w'] / 2, c['cy'] + c['h'] / 2
            if (lx0 < margin and x0 > 0) or (lx1 > tw - margin and x1 < img.width) or \
               (ly0 < margin and y0 > 0) or (ly1 > th - margin and y1 < img.height):
                continue
            c['cx'] += x0
            c['cy'] += y0
            all_cells.append(c)
    return _dedupe_cells(all_cells)


def detect_scale_normalized(img, model, max_det=2000, target_cell_px=TARGET_CELL_PX):
    """Tile at the default size, measure the median cell width, and re-tile at the size
    that lands cells at target_cell_px if that differs by more than 50%. Returns every
    detection at or above LOW_CONF (filter on `conf` as needed), or [] if there are too
    few to size a tile from. Mirrors js/src/tiling.js detectScaleNormalized()."""
    cells = run_detection_tiled(img, model, max_det, TILE_SIZE)
    if len(cells) < CONTAINER_MIN_CANDIDATES:
        return []
    hi = [c for c in cells if c['conf'] >= HIGH_CONF]
    sample = hi if len(hi) >= CONTAINER_MIN_CANDIDATES else cells
    median_w = statistics.median(c['w'] for c in sample)
    native = max(MIN_NATIVE_TILE, round(median_w * TILE_SIZE / target_cell_px))
    if native / TILE_SIZE > 1.5 or native / TILE_SIZE < 0.67:
        return run_detection_tiled(img, model, max_det, native)
    return cells


def load_cell_classifier(model_path):
    """Load the MobileNetV2 dot classifier. Returns (model, device, transform)."""
    import torch
    import torch.nn as nn
    from torchvision import models, transforms

    device = (
        torch.device('mps') if torch.backends.mps.is_available() else
        torch.device('cuda') if torch.cuda.is_available() else
        torch.device('cpu')
    )
    net = models.mobilenet_v2(weights=None)
    net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(net.last_channel, 6))
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return net, device, tf


def reclassify_cells(img, cells, clf, clf_device, clf_tf):
    """Replace each cell's bits/char with the classifier's, in one batched pass."""
    import torch

    active = [c for c in cells if not c.get('is_space')]
    if not active:
        return cells

    iw, ih = img.size
    crops = []
    for c in active:
        pad_x = c['w'] * CLASSIFIER_CROP_PAD
        pad_y = c['h'] * CLASSIFIER_CROP_PAD
        x0 = max(0, c['cx'] - c['w'] / 2 - pad_x)
        y0 = max(0, c['cy'] - c['h'] / 2 - pad_y)
        x1 = min(iw, c['cx'] + c['w'] / 2 + pad_x)
        y1 = min(ih, c['cy'] + c['h'] / 2 + pad_y)
        crops.append(clf_tf(img.crop((x0, y0, x1, y1)).convert('RGB')))

    with torch.no_grad():
        logits = clf(torch.stack(crops).to(clf_device))
        preds = (logits.sigmoid() > 0.5).cpu().numpy().astype(int)

    for cell, row in zip(active, preds):
        bits = ''.join(str(b) for b in row)
        cell['bits'] = bits
        cell['char'] = bits_to_braille(bits)
    return cells
