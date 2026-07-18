"""
Evaluate the OCR pipeline against any labeled Braille image dataset.

Dataset format (Angelina-compatible):
  Each image <name>.jpg must have a corresponding <name>.csv with lines:
    left;top;right;bottom;label
  where coordinates are normalised [0,1] and label is an integer 1-63
  (Braille cell encoded as 6 bits, bit0=dot1).

Metrics:
  Detection recall  = GT cells matched by a predicted cell / total GT cells
  Detection precision = matched predicted cells / total predicted cells
  Class accuracy    = predicted cells with correct 6-bit pattern / matched cells
  F1                = harmonic mean of precision and recall

Usage:
  python evaluate.py --dataset /path/to/labeled/images
  python evaluate.py --dataset /path/to/labeled/images --classifier cell_classifier.pt

Matching: nearest predicted cell within distance < 0.5 * avg_gt_cell_width.
"""

import sys
import csv
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import pipeline as tb

_MODEL = None


def _get_model():
    """Lazy so `evaluate.py --help` works without resolving a model file at all
    (resolve_model() itself is cheap when a sibling `dataset` checkout exists,
    but may fetch over the network otherwise -- keep that off the --help path)."""
    global _MODEL
    if _MODEL is None:
        _MODEL = tb.YOLO(str(tb.resolve_model('cell_detector.pt')))
    return _MODEL


def label_to_bits(label_int):
    """Convert Angelina integer label (1-63) to our 6-char bits string.
    LSB = dot1 → leftmost position in our format."""
    return format(label_int, '06b')[::-1]


def load_gt(csv_path, img_w, img_h):
    """Load ground truth cells from a CSV file.
    Returns list of dicts: cx, cy, w, h (pixel coords), bits6."""
    cells = []
    with open(csv_path) as f:
        for row in csv.reader(f, delimiter=';'):
            if len(row) < 5:
                continue
            l, t, r, b, lbl = float(row[0]), float(row[1]), float(row[2]), float(row[3]), int(row[4])
            cx = (l + r) / 2 * img_w
            cy = (t + b) / 2 * img_h
            w  = (r - l) * img_w
            h  = (b - t) * img_h
            cells.append({'cx': cx, 'cy': cy, 'w': w, 'h': h, 'bits': label_to_bits(lbl)})
    return cells


def match_cells(pred_cells, gt_cells, dist_thresh_frac=0.5):
    """
    Greedily match predicted cells to GT cells by nearest centre distance.
    dist_thresh_frac: match accepted if distance < frac * avg GT cell width.
    Returns: list of (pred_idx, gt_idx) pairs.
    """
    if not gt_cells or not pred_cells:
        return []
    avg_gt_w = np.mean([c['w'] for c in gt_cells])
    threshold = avg_gt_w * dist_thresh_frac

    pred_cx = np.array([[c['cx'], c['cy']] for c in pred_cells])
    gt_cx   = np.array([[c['cx'], c['cy']] for c in gt_cells])

    used_gt = set()
    matches = []
    for pi, pc in enumerate(pred_cx):
        dists = np.linalg.norm(gt_cx - pc, axis=1)
        dists[list(used_gt)] = np.inf
        best = int(np.argmin(dists))
        if dists[best] < threshold:
            matches.append((pi, best))
            used_gt.add(best)
    return matches


_CLF = None

def _get_clf(clf_path):
    global _CLF
    if _CLF is None and clf_path:
        net, dev, tf = tb.load_cell_classifier(clf_path)
        _CLF = (net, dev, tf)
    return _CLF


def run_pipeline(img_path, clf_path=None):
    """Full pipeline: contrast search + grid_fill + edge crop recovery + gap pixel recovery."""
    model = _get_model()
    original = tb.PIL.ImageOps.exif_transpose(tb.PIL.Image.open(img_path)).convert('RGB')
    img, _, _ = tb.best_contrast(original, model, 2000)
    all_cells = tb.run_detection(img, model, 2000)
    raw_hi    = [c for c in all_cells if c['conf'] >= tb.HIGH_CONF]
    cells, empties = tb.grid_fill(all_cells)
    edge_cells = tb.crop_recover(img, model, empties)
    cells += edge_cells
    gap_cells  = tb.gap_pixel_recover(img, model, empties, raw_hi, known_cells=cells)
    cells += gap_cells
    # Margin filter
    if raw_hi:
        hi_cx = [c['cx'] for c in raw_hi]
        hi_cy = [c['cy'] for c in raw_hi]
        xlo, xhi = np.percentile(hi_cx, 5),  np.percentile(hi_cx, 95)
        ylo, yhi = np.percentile(hi_cy, 5),  np.percentile(hi_cy, 95)
        avg_w = np.mean([c['w'] for c in raw_hi])
        avg_h = np.mean([c['h'] for c in raw_hi])
        cells = [c for c in cells
                 if xlo - avg_w < c['cx'] < xhi + avg_w
                 and ylo - avg_h < c['cy'] < yhi + avg_h]
    # Optional: reclassify with MobileNetV2
    clf = _get_clf(clf_path)
    if clf:
        cells = tb.reclassify_cells(img, cells, *clf)
    return cells


def main():
    import argparse
    ap = argparse.ArgumentParser(description='Evaluate OCR pipeline against labeled Braille images')
    ap.add_argument('--dataset',    required=True,
                    help='Directory of labeled images (each .jpg needs a matching .csv)')
    ap.add_argument('--glob',       default='*.jpg',
                    help='Glob pattern for images within --dataset (default: *.jpg)')
    ap.add_argument('--start',      type=int, default=0)
    ap.add_argument('--end',        type=int, default=None)
    ap.add_argument('--out',        default=None, help='Append per-image CSV results to this file')
    ap.add_argument('--classifier', default=None,
                    help='Path to cell_classifier.pt (auto-detected if omitted)')
    args = ap.parse_args()

    dataset_dir = Path(args.dataset)
    clf_path = args.classifier or str(tb.resolve_classifier_path())
    print(f"Using classifier: {clf_path}")

    jpg_files = sorted(dataset_dir.glob(args.glob))
    if args.end is not None:
        jpg_files = jpg_files[args.start:args.end]
    else:
        jpg_files = jpg_files[args.start:]
    outfile = open(args.out, 'a') if args.out else None
    print(f"Evaluating on {len(jpg_files)} images from {dataset_dir.name}/\n")

    total_gt   = 0
    total_pred = 0
    total_tp   = 0   # detection TP (any match)
    total_class_correct = 0

    per_image = []

    for img_path in jpg_files:
        csv_path = img_path.with_suffix('.csv')
        if not csv_path.exists():
            continue

        import PIL.Image
        with PIL.Image.open(img_path) as im:
            img_w, img_h = im.size

        gt_cells   = load_gt(csv_path, img_w, img_h)
        pred_cells = run_pipeline(img_path, clf_path=clf_path)

        matches = match_cells(pred_cells, gt_cells)
        tp = len(matches)
        class_ok = sum(1 for pi, gi in matches
                       if pred_cells[pi]['bits'] == gt_cells[gi]['bits'])

        n_gt   = len(gt_cells)
        n_pred = len(pred_cells)
        prec   = tp / n_pred if n_pred else 0.0
        rec    = tp / n_gt   if n_gt   else 0.0
        f1     = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
        ca     = class_ok / tp if tp else 0.0

        total_gt   += n_gt
        total_pred += n_pred
        total_tp   += tp
        total_class_correct += class_ok

        per_image.append((img_path.name, n_gt, n_pred, tp, prec, rec, f1, ca))
        line = (f"  {img_path.stem[:40]:40s}  gt={n_gt:3d}  pred={n_pred:3d}  "
                f"tp={tp:3d}  prec={prec:.2f}  rec={rec:.2f}  f1={f1:.2f}  cls={ca:.2f}")
        print(line, flush=True)
        if outfile:
            print(f"{img_path.name},{n_gt},{n_pred},{tp},{class_ok}", file=outfile, flush=True)

    # Aggregate
    prec = total_tp / total_pred if total_pred else 0.0
    rec  = total_tp / total_gt   if total_gt   else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    ca   = total_class_correct / total_tp if total_tp else 0.0

    print(f"\n{'='*70}")
    print(f"AGGREGATE  gt={total_gt}  pred={total_pred}  tp={total_tp}")
    print(f"  Precision : {prec:.3f}")
    print(f"  Recall    : {rec:.3f}")
    print(f"  F1        : {f1:.3f}")
    print(f"  Class acc : {ca:.3f}  (correct 6-bit pattern among matched cells)")


if __name__ == '__main__':
    main()
