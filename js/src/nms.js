/** IoU of two boxes given as {cx, cy, w, h}. */
function iou(a, b) {
	const ax0 = a.cx - a.w / 2,
		ay0 = a.cy - a.h / 2,
		ax1 = a.cx + a.w / 2,
		ay1 = a.cy + a.h / 2;
	const bx0 = b.cx - b.w / 2,
		by0 = b.cy - b.h / 2,
		bx1 = b.cx + b.w / 2,
		by1 = b.cy + b.h / 2;

	const ix0 = Math.max(ax0, bx0),
		iy0 = Math.max(ay0, by0);
	const ix1 = Math.min(ax1, bx1),
		iy1 = Math.min(ay1, by1);
	if (ix1 <= ix0 || iy1 <= iy0) return 0;

	const inter = (ix1 - ix0) * (iy1 - iy0);
	const areaA = (ax1 - ax0) * (ay1 - ay0);
	const areaB = (bx1 - bx0) * (by1 - by0);
	return inter / (areaA + areaB - inter);
}

/**
 * Greedy class-agnostic NMS. `boxes` is an array of {cx, cy, w, h, conf},
 * already confidence-filtered. Returns the kept subset, highest-conf first.
 */
export function nms(boxes, iouThreshold = 0.7, maxDet = 300) {
	const sorted = [...boxes].sort((a, b) => b.conf - a.conf);
	const kept = [];
	for (const box of sorted) {
		if (kept.length >= maxDet) break;
		if (!kept.some((k) => iou(box, k) > iouThreshold)) {
			kept.push(box);
		}
	}
	return kept;
}
