"""
Classical-CV detection of candidate "container" regions in a photo — the
sign, page, or button that Braille is embossed on — as opposed to
irrelevant surrounding context (trees around a sign, the blanket a page is
lying on, other buttons in an elevator panel).

Cropping to a container before running cell detection raises the *effective*
resolution of the Braille dots after YOLO's mandatory resize to a fixed
input size, which plausibly matters more for accuracy than which detector
model is used, when the container is a small fraction of the frame.

No training/model involved: candidates come from edge detection + contour
finding, filtered to shapes that look like a container (rectangular, e.g. a
sign or page; or circular, e.g. an elevator button) — a document scanner's
auto-crop uses the same idea. Whether a candidate actually *has* Braille on
it is decided separately (see pipeline.py, which runs a quick detection pass
per candidate), since shape alone can't tell you that.
"""

import cv2
import numpy as np

MIN_AREA_FRAC = 0.005   # ignore contours smaller than this fraction of the image
MAX_AREA_FRAC = 0.95    # ignore contours near the size of the whole frame (not a sub-container)
MIN_RECTANGULARITY = 0.6
MIN_CIRCULARITY = 0.6
IOU_MERGE_THRESHOLD = 0.7  # candidates more overlapping than this collapse to one


def _auto_canny(gray, sigma=0.33):
    median = float(np.median(gray))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    return cv2.Canny(gray, lower, upper)


def _shape_score(contour):
    area = cv2.contourArea(contour)
    if area <= 0:
        return 0.0, 0.0
    x, y, w, h = cv2.boundingRect(contour)
    rectangularity = area / (w * h) if w * h else 0.0
    perimeter = cv2.arcLength(contour, closed=True)
    circularity = (4 * np.pi * area / (perimeter ** 2)) if perimeter else 0.0
    return rectangularity, circularity


def _iou(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (ax1 - ax0) * (ay1 - ay0)
    area_b = (bx1 - bx0) * (by1 - by0)
    return inter / (area_a + area_b - inter)


def find_containers(img):
    """
    Return candidate container boxes as [(x0, y0, x1, y1), ...] pixel
    coordinates in `img` (a PIL.Image), largest first. May return an empty
    list (e.g. a photo that's already tightly framed on just the Braille
    content) — callers should fall back to treating the whole image as one
    container in that case.
    """
    arr = cv2.cvtColor(np.array(img.convert('RGB')), cv2.COLOR_RGB2GRAY)
    img_area = arr.shape[0] * arr.shape[1]

    edges = _auto_canny(arr)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        if not (MIN_AREA_FRAC * img_area <= area <= MAX_AREA_FRAC * img_area):
            continue
        rectangularity, circularity = _shape_score(c)
        if rectangularity < MIN_RECTANGULARITY and circularity < MIN_CIRCULARITY:
            continue
        x, y, w, h = cv2.boundingRect(c)
        candidates.append((area, (x, y, x + w, y + h)))

    candidates.sort(key=lambda t: t[0], reverse=True)

    kept = []
    for _, box in candidates:
        if not any(_iou(box, k) > IOU_MERGE_THRESHOLD for k in kept):
            kept.append(box)
    return kept
