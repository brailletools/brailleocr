// Ports pipeline.py's tiling + scale-normalization logic — NOT contrast
// search, NOT the grid-fill/crop-recovery/gap-pixel-recovery rescue passes,
// which stay out of scope (see the module's accuracy-fix notes). Tiling
// itself is load-bearing, not an optional robustness enhancement: the
// detector was trained exclusively on tiles where cells land at
// TARGET_CELL_PX (see prepare_yolo_dataset.py), so feeding it a whole,
// untiled page directly puts cells far outside that training distribution —
// box *count* degrades gracefully but box *size* regression does not (see
// pipeline.py's own process_container() docstring, and the ~3x oversized
// boxes this produced before this module existed).

import { cropRgbHwc } from './imageOps.js';
import { nms, iou } from './nms.js';

// Shared with prepare_yolo_dataset.py (see dot_pattern_utils.py) — the
// detector must be trained on tiles built the same way, or these two
// numbers drifting apart defeats the point.
export const TILE_SIZE = 640;
export const TARGET_CELL_PX = 30;
export const MIN_NATIVE_TILE = 100;
export const DEFAULT_TILE_OVERLAP_FRAC = 0.2;

export const HIGH_CONF = 0.3;
export const LOW_CONF = 0.05;
export const CONTAINER_MIN_CANDIDATES = 3;
export const TILE_DEDUPE_IOU = 0.5;
// Fraction of native tile size — a cell straddling a tile's own crop edge
// only gets a partial (truncated, falsely narrow) view; drop it and rely on
// the overlapping neighbor tile to detect that same cell away from its edge.
export const TILE_EDGE_MARGIN_FRAC = 0.05;
// Not from pipeline.py — a JS-side addition. Two detections of the same
// physical cell from overlapping tiles, each with a slightly different box
// estimate, can still overlap substantially below TILE_DEDUPE_IOU (0.5) and
// survive nms()'s dedup as a spurious near-duplicate pair. Anything above
// this looser threshold is treated as "likely the same cell" and resolved by
// resolveOverlaps() instead of being left as two competing detections.
export const RESIDUAL_OVERLAP_IOU = 0.15;
// Matches group_into_lines()'s row-band convention (pipeline.py:557-571):
// a cell within this fraction of the pair's own height counts as "the same
// row" for grid-fit purposes.
export const ROW_BAND_FRAC = 0.55;

function median(values) {
	const sorted = [...values].sort((a, b) => a - b);
	const mid = Math.floor(sorted.length / 2);
	return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Grid of [x0,y0,x1,y1] tiles covering (imgW, imgH), using tileSize native
 * pixels per tile with overlapFrac overlap. A single tile covering the whole
 * image if it already fits within tileSize. Direct port of
 * dot_pattern_utils.py's make_tile_boxes().
 */
export function makeTileBoxes(imgW, imgH, tileSize, overlapFrac = DEFAULT_TILE_OVERLAP_FRAC) {
	if (imgW <= tileSize && imgH <= tileSize) {
		return [[0, 0, imgW, imgH]];
	}

	const overlap = Math.round(tileSize * overlapFrac);
	const stride = Math.max(1, tileSize - overlap);

	const axisStarts = (dim) => {
		const starts = [];
		const maxStart = Math.max(dim - tileSize, 0);
		for (let x = 0; x <= maxStart; x += stride) starts.push(x);
		if (starts[starts.length - 1] + tileSize < dim) starts.push(dim - tileSize);
		return starts;
	};

	const xs = axisStarts(imgW);
	const ys = axisStarts(imgH);

	const boxes = [];
	for (const y of ys) {
		for (const x of xs) {
			boxes.push([x, y, Math.min(x + tileSize, imgW), Math.min(y + tileSize, imgH)]);
		}
	}
	return boxes;
}

// Reuses pipeline.py's own "reliable enough to fit a grid from" threshold
// (see its module docstring: cells >= HIGH_CONF are used to fit a per-row
// grid in grid_fill()) — a neighbor below this isn't trusted as grid context,
// even if it's the nearest one by position.
export const NEIGHBOR_QUALITY_CONF = HIGH_CONF;
// A candidate within this fraction of the row's typical gap from a predicted
// grid slot counts as "on the grid"; farther than that counts as off-grid.
// Not a threshold pipeline.py uses, tuned against real data.
export const SLOT_FIT_TOLERANCE_FRAC = 0.25;

/**
 * Given two candidates believed to be the same physical cell, picks whichever
 * actually sits on the row's established grid — checking each one
 * independently against a robust grid line, rather than requiring the whole
 * span between the pair's nearest neighbors to look clean.
 *
 * An earlier version required the nearest left/right neighbors to be spaced
 * apart by close to one cell-to-cell gap before trusting them at all, and
 * bailed out (left the pair unmerged) otherwise. That failed on real data: a
 * missing/undetected cell sitting elsewhere in that span (not adjacent to the
 * disputed pair at all) made the *whole* span look unreliable, even when one
 * candidate sat almost exactly on a clean grid slot relative to its own
 * immediate neighbor. Fitting one robust line across every quality neighbor
 * in the row (fixed slope = the row's typical gap; median-of-residuals
 * intercept, so no single neighbor's own position noise dominates) and
 * scoring each candidate's distance to its *nearest* predicted slot handles
 * this correctly: a real missing cell elsewhere in the row doesn't stop a
 * nearby candidate from being judged against the slot right next to it.
 *
 * Falls back to confidence only when there's no quality row context at all
 * (an isolated pair, or one surrounded only by low-confidence detections).
 * When neither candidate lands near a predicted grid slot, there's no
 * reliable evidence either way — deliberately returns null (leave both,
 * don't guess by confidence) rather than repeat that mistake.
 * @param {{cx:number,cy:number,w:number,h:number,conf:number}} a
 * @param {{cx:number,cy:number,w:number,h:number,conf:number}} b
 * @param {Array<{cx:number,cy:number,w:number,h:number,conf:number}>} others - every other cell, excluding a/b
 * @returns {object|null} the winning candidate, or null if not confidently resolvable
 */
function pickBetterGridFit(a, b, others) {
	const midCy = (a.cy + b.cy) / 2;
	const rowTolerance = Math.max(a.h, b.h) * ROW_BAND_FRAC;
	const row = others.filter((c) => Math.abs(c.cy - midCy) < rowTolerance);
	const qualityRow = row.filter((c) => c.conf >= NEIGHBOR_QUALITY_CONF);

	if (qualityRow.length === 0) {
		return a.conf >= b.conf ? a : b;
	}

	const sortedRow = [...qualityRow].sort((p, q) => p.cx - q.cx);
	const rowY = median(sortedRow.map((c) => c.cy));

	// Row-wide typical spacing, from every consecutive quality neighbor across
	// the whole row — doubles as the fixed slope for the grid line below.
	const allGaps = [];
	for (let k = 1; k < sortedRow.length; k++) allGaps.push(sortedRow[k].cx - sortedRow[k - 1].cx);
	const typicalGap = allGaps.length ? median(allGaps) : Math.max(a.w, b.w) * 1.7;

	// Robust local grid line: fixed slope = typicalGap, median-of-residuals
	// intercept across every quality neighbor in the row (anchored at the
	// leftmost one arbitrarily — any reference point gives the same line).
	const reference = sortedRow[0].cx;
	const intercepts = sortedRow.map((c) => {
		const rank = Math.round((c.cx - reference) / typicalGap);
		return c.cx - typicalGap * rank;
	});
	const intercept = median(intercepts);

	const residual = (x) => {
		const rank = Math.round((x - intercept) / typicalGap);
		return Math.abs(x - (intercept + typicalGap * rank));
	};

	const tolerance = typicalGap * SLOT_FIT_TOLERANCE_FRAC;
	const residualA = residual(a.cx);
	const residualB = residual(b.cx);
	const fitsA = residualA <= tolerance;
	const fitsB = residualB <= tolerance;

	if (fitsA && !fitsB) return a;
	if (fitsB && !fitsA) return b;
	if (!fitsA && !fitsB) return null;

	// Both land on a grid slot (e.g. one is a clean duplicate of the other) —
	// prefer the tighter fit, then whichever is closer to the row's y-baseline.
	const scoreA = residualA + Math.abs(a.cy - rowY);
	const scoreB = residualB + Math.abs(b.cy - rowY);
	return scoreA <= scoreB ? a : b;
}

/**
 * Resolves residual overlapping-candidate pairs that survive nms()'s
 * IoU-based dedup (see RESIDUAL_OVERLAP_IOU) by grid-fit rather than
 * confidence alone — see pickBetterGridFit(). Repeatedly finds the
 * highest-IoU pair still above the threshold and drops the worse-fitting
 * one, until none remain. When pickBetterGridFit() can't confidently resolve
 * a pair (returns null — the local grid context itself is unreliable, most
 * often because a missing/undetected cell sits nearby too), that pair is left
 * as-is rather than merged by a guess, and the search moves on to the next
 * pair. Expected to run zero or a handful of times per page, not a hot loop
 * — pairs like this are rare.
 */
export function resolveOverlaps(cells, iouThreshold = RESIDUAL_OVERLAP_IOU) {
	const kept = [...cells];
	const ids = new WeakMap();
	let nextId = 0;
	const idOf = (c) => {
		if (!ids.has(c)) ids.set(c, nextId++);
		return ids.get(c);
	};
	const pairKey = (x, y) => {
		const ix = idOf(x);
		const iy = idOf(y);
		return ix < iy ? `${ix}-${iy}` : `${iy}-${ix}`;
	};
	const unresolvable = new Set();

	for (;;) {
		let bestPair = null;
		let bestIou = iouThreshold;
		for (let i = 0; i < kept.length; i++) {
			for (let j = i + 1; j < kept.length; j++) {
				if (unresolvable.has(pairKey(kept[i], kept[j]))) continue;
				const v = iou(kept[i], kept[j]);
				if (v > bestIou) {
					bestIou = v;
					bestPair = [i, j];
				}
			}
		}
		if (!bestPair) break;

		const [i, j] = bestPair;
		const a = kept[i];
		const b = kept[j];
		const others = kept.filter((_, idx) => idx !== i && idx !== j);
		const winner = pickBetterGridFit(a, b, others);
		if (winner === null) {
			unresolvable.add(pairKey(a, b));
			continue;
		}
		const loserIdx = winner === a ? j : i;
		kept.splice(loserIdx, 1);
	}

	return kept;
}

/**
 * Direct port of pipeline.py's run_detection_tiled(): tiles the image at
 * nativeTileSize, runs `detector` on each tile (its own internal letterbox +
 * NMS already matches ultralytics' per-call behavior — see detector.js), drops
 * cells truncated by an *internal* tile edge, offsets survivors back to
 * full-image coordinates, then dedupes across tile-overlap zones by reusing
 * the same nms() this package already uses for within-tile NMS, just at a
 * different (looser) threshold — this is the same operation pipeline.py's
 * _dedupe_cells() performs, not a different algorithm. A final
 * resolveOverlaps() pass (not from pipeline.py) then cleans up residual
 * near-duplicate pairs nms() didn't merge, by a robust multi-neighbor grid
 * fit rather than confidence alone or a single nearest neighbor — see
 * pickBetterGridFit()'s doc comment for the two real bugs earlier, cruder
 * versions of this hit on real data (an ungated gap could imply the wrong
 * number of cell-slots; a single trusted neighbor could itself be
 * mispositioned) and how each is addressed.
 * @param {import('./detector.js').CellDetector} detector
 * @param {Float32Array} rgbHwc - HWC RGB, values 0-255, full (untiled) image
 * @returns {Promise<Array<{cx: number, cy: number, w: number, h: number, conf: number}>>}
 */
export async function detectTiled(detector, rgbHwc, imgW, imgH, nativeTileSize, opts = {}) {
	const boxes = makeTileBoxes(imgW, imgH, nativeTileSize);
	if (boxes.length === 1) {
		return detector.detect(rgbHwc, imgW, imgH, opts);
	}

	const margin = nativeTileSize * TILE_EDGE_MARGIN_FRAC;
	const allCells = [];

	for (const [x0, y0, x1, y1] of boxes) {
		const tile = cropRgbHwc(rgbHwc, imgW, imgH, x0, y0, x1, y1);
		const tileCells = await detector.detect(tile.data, tile.w, tile.h, opts);
		const tw = x1 - x0;
		const th = y1 - y0;

		for (const c of tileCells) {
			const lx0 = c.cx - c.w / 2;
			const ly0 = c.cy - c.h / 2;
			const lx1 = c.cx + c.w / 2;
			const ly1 = c.cy + c.h / 2;
			const truncatedByInternalEdge =
				(lx0 < margin && x0 > 0) ||
				(lx1 > tw - margin && x1 < imgW) ||
				(ly0 < margin && y0 > 0) ||
				(ly1 > th - margin && y1 < imgH);
			if (truncatedByInternalEdge) continue;

			allCells.push({ ...c, cx: c.cx + x0, cy: c.cy + y0 });
		}
	}

	const deduped = nms(allCells, TILE_DEDUPE_IOU, Infinity);
	return resolveOverlaps(deduped);
}

/**
 * Direct port of pipeline.py's process_container() normalize_scale branch
 * (contrast search and the grid-fill/crop-recovery/gap-pixel-recovery rescue
 * passes are out of scope — see this module's header comment). Bootstraps
 * from a first pass tiled at the plain TILE_SIZE default — always closer to
 * TARGET_CELL_PX than a whole untiled page, so the same specialized model
 * measures cell size reliably enough there — and only re-tiles at a refined
 * size if that measurement says it matters (more than 50% off either way).
 * @param {import('./detector.js').CellDetector} detector
 * @param {Float32Array} rgbHwc - HWC RGB, values 0-255, full (untiled) image
 */
export async function detectScaleNormalized(detector, rgbHwc, imgW, imgH, opts = {}) {
	const initialCells = await detectTiled(detector, rgbHwc, imgW, imgH, TILE_SIZE, opts);
	if (initialCells.length < CONTAINER_MIN_CANDIDATES) {
		// Too few candidates to trust a median-width estimate from — return
		// what little was found rather than refine tiling from noise.
		return initialCells;
	}

	const highConf = initialCells.filter((c) => c.conf >= HIGH_CONF);
	const sizeSample = highConf.length >= CONTAINER_MIN_CANDIDATES ? highConf : initialCells;
	const medianW = median(sizeSample.map((c) => c.w));
	// A tile of native size L resized to TILE_SIZE for the model scales a
	// cell by TILE_SIZE/L; solving medianW * TILE_SIZE/L = TARGET_CELL_PX
	// for L gives this.
	const nativeTileSize = Math.max(MIN_NATIVE_TILE, Math.round((medianW * TILE_SIZE) / TARGET_CELL_PX));

	const ratio = nativeTileSize / TILE_SIZE;
	if (ratio > 1.5 || ratio < 0.67) {
		return detectTiled(detector, rgbHwc, imgW, imgH, nativeTileSize, opts);
	}
	return initialCells;
}
