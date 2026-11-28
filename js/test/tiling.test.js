import { test } from 'node:test';
import assert from 'node:assert/strict';

import { makeTileBoxes, resolveOverlaps } from '../src/tiling.js';

test('makeTileBoxes returns a single tile when the image already fits', () => {
	assert.deepEqual(makeTileBoxes(500, 400, 640), [[0, 0, 500, 400]]);
});

test('makeTileBoxes covers the whole image with overlapping tiles, flush against the far edges', () => {
	const boxes = makeTileBoxes(1500, 700, 640, 0.2);
	// Every tile must be well-formed and within bounds.
	for (const [x0, y0, x1, y1] of boxes) {
		assert.ok(x0 >= 0 && y0 >= 0 && x1 <= 1500 && y1 <= 700 && x1 > x0 && y1 > y0);
	}
	// The last tile in each axis must reach exactly to the far edge (no gap left uncovered).
	const maxX1 = Math.max(...boxes.map((b) => b[2]));
	const maxY1 = Math.max(...boxes.map((b) => b[3]));
	assert.equal(maxX1, 1500);
	assert.equal(maxY1, 700);
});

test('resolveOverlaps picks the candidate that fits the row grid, even against higher confidence', () => {
	// A row of evenly-spaced cells at x = 0, 50, 100, [~150], 200, 250 (y=100,
	// w=40, h=60) with an ambiguous overlapping pair standing in for the
	// missing ~150 slot: one candidate sits exactly at the expected grid
	// position, the other is offset by 15px (still overlapping — iou ~0.45,
	// above RESIDUAL_OVERLAP_IOU). The offset one has deliberately HIGHER
	// confidence, to prove grid fit — not confidence — decides the winner.
	const neighbors = [0, 50, 100, 200, 250].map((cx) => ({ cx, cy: 100, w: 40, h: 60, conf: 0.5 }));
	const wellFit = { cx: 150, cy: 100, w: 40, h: 60, conf: 0.3, id: 'wellFit' };
	const offset = { cx: 165, cy: 100, w: 40, h: 60, conf: 0.9, id: 'offset' };

	const result = resolveOverlaps([...neighbors, wellFit, offset]);

	assert.equal(result.length, neighbors.length + 1, 'the ambiguous pair should resolve to exactly one cell');
	const kept = result.find((c) => c.id === 'wellFit' || c.id === 'offset');
	assert.equal(kept.id, 'wellFit', 'the evenly-spaced candidate should win despite lower confidence');
});

test('resolveOverlaps falls back to confidence when no row context is available', () => {
	// An isolated overlapping pair with no other nearby cells to fit a grid
	// against — nothing to prefer on position, so the higher-confidence one wins.
	const a = { cx: 500, cy: 500, w: 40, h: 60, conf: 0.2, id: 'low' };
	const b = { cx: 510, cy: 505, w: 40, h: 60, conf: 0.8, id: 'high' };

	const result = resolveOverlaps([a, b]);

	assert.equal(result.length, 1);
	assert.equal(result[0].id, 'high');
});

test('resolveOverlaps confidently picks the candidate that lands on the established grid, even across a gap with a missing cell', () => {
	// Neighbors at x = 0, 50 (normal ~50px spacing) then a big gap to 250, 300
	// (also normal spacing between themselves). A cell is almost certainly
	// missing somewhere in that 200px gap (~4x the row's 50px typical gap) --
	// but ALL FOUR confirmed neighbors land exactly on multiples of 50, so the
	// fitted grid line is fully confident, and can still correctly extrapolate
	// into the gap: a disputed candidate sitting right on one of those
	// predicted slots (150) should win over one that doesn't (165), same as
	// pipeline.py's own grid_fill extrapolates into unconfirmed positions.
	// Regression test for an earlier, overly conservative version that treated
	// ANY oversized neighbor-to-neighbor span as unresolvable, even when the
	// grid fit across the row was otherwise perfectly consistent.
	const neighbors = [0, 50, 250, 300].map((cx) => ({ cx, cy: 100, w: 40, h: 60, conf: 0.5 }));
	const onSlot = { cx: 150, cy: 100, w: 40, h: 60, conf: 0.2, id: 'onSlot' };
	const offSlot = { cx: 165, cy: 100, w: 40, h: 60, conf: 0.9, id: 'offSlot' };

	const result = resolveOverlaps([...neighbors, onSlot, offSlot]);

	assert.equal(result.length, neighbors.length + 1, 'the pair should resolve to exactly one cell');
	const kept = result.find((c) => c.id === 'onSlot' || c.id === 'offSlot');
	assert.equal(kept.id, 'onSlot', 'the candidate on the predicted grid slot should win despite lower confidence');
});

test('resolveOverlaps leaves an overlapping pair unmerged when neither candidate lands near any predicted grid slot', () => {
	// Irregular neighbor spacing (55, 205, 60 -- not clean multiples of a
	// single gap) means the fitted grid line itself is uncertain. Neither
	// disputed candidate lands close to what that uncertain line predicts, so
	// there's no reliable basis to prefer one over the other -- must be left
	// as two cells, not resolved by falling back to confidence (which
	// real-data cases upstream showed can be actively misleading).
	const neighbors = [0, 55, 260, 320].map((cx) => ({ cx, cy: 100, w: 40, h: 60, conf: 0.5 }));
	const a = { cx: 150, cy: 100, w: 40, h: 60, conf: 0.9, id: 'a' };
	const b = { cx: 165, cy: 100, w: 40, h: 60, conf: 0.2, id: 'b' };

	const result = resolveOverlaps([...neighbors, a, b]);

	assert.equal(result.length, neighbors.length + 2, 'an unresolvable pair must be left as two cells, not merged by a guess');
	assert.ok(result.some((c) => c.id === 'a'));
	assert.ok(result.some((c) => c.id === 'b'));
});

test('resolveOverlaps leaves genuinely distinct, non-overlapping cells untouched', () => {
	const cells = [0, 50, 100, 150, 200].map((cx) => ({ cx, cy: 100, w: 40, h: 60, conf: 0.5 }));
	const result = resolveOverlaps(cells);
	assert.equal(result.length, cells.length);
});

// ---- images smaller than the tile they are read at -------------------------------

import { detectTiled } from '../src/tiling.js';
import { boxDownscale, scaleOntoCanvas } from '../src/imageOps.js';

test('scaleOntoCanvas puts the scaled image top-left on a grey 640 canvas and reports the real scale', () => {
	const w = 100, h = 50;
	const src = new Float32Array(w * h * 3).fill(200);
	const c = scaleOntoCanvas(src, w, h, 0.5);
	assert.equal(c.w, 640);
	assert.equal(c.h, 640);
	assert.equal(c.scaleX, 0.5);
	assert.equal(c.scaleY, 0.5);
	assert.equal(c.data[0], 200, 'top-left pixel is the image');
	assert.equal(c.data[(24 * 640 + 49) * 3], 200, 'last image pixel');
	assert.equal(c.data[(24 * 640 + 60) * 3], 114, 'beyond the image is pad grey');
});

test('boxDownscale averages blocks', () => {
	// 4x2 image, one grey value per pixel: row 0 = 0 100 40 60, row 1 = 200 0 20 80
	const gray = [0, 100, 40, 60, 200, 0, 20, 80];
	const src = Float32Array.from(gray.flatMap((v) => [v, v, v]));
	const { data, w, h } = boxDownscale(src, 4, 2, 2);
	assert.deepEqual([w, h], [2, 1]);
	assert.deepEqual(Array.from(data.slice(0, 3)), [75, 75, 75]);   // (0+100+200+0)/4
	assert.deepEqual(Array.from(data.slice(3, 6)), [50, 50, 50]);   // (40+60+20+80)/4
});

test('a small image with big cells is shown to the detector at the density a full tile would give', async () => {
	// 1000 x 200 image whose cells are 170 px wide: native tile = 170 * 640 / 30 = 3627 px, far
	// larger than the image. The detector must see the cell at about 30 px on a 640 canvas.
	const seen = [];
	const detector = {
		async detect(rgb, w, h) {
			seen.push([w, h]);
			// pretend there is one cell, 30 canvas-px wide, centred at (100, 50) on the canvas
			return [{ cx: 100, cy: 50, w: 30, h: 46, conf: 0.9 }];
		}
	};
	const rgb = new Float32Array(1000 * 200 * 3).fill(120);
	const cells = await detectTiled(detector, rgb, 1000, 200, 3627);
	assert.deepEqual(seen, [[640, 640]]);
	const scale = 640 / 3627;
	assert.ok(Math.abs(cells[0].w - 30 / scale) < 3, `cell width back in native pixels: ${cells[0].w}`);
	assert.ok(Math.abs(cells[0].cx - 100 / scale) < 3);
});

test('an image at least as large as the tile is passed through unchanged', async () => {
	const seen = [];
	const detector = { async detect(rgb, w, h) { seen.push([w, h]); return []; } };
	await detectTiled(detector, new Float32Array(600 * 400 * 3), 600, 400, 600);
	assert.deepEqual(seen, [[600, 400]]);
});

import { detectScaleNormalized } from '../src/tiling.js';

test('the tile size follows the box width the detector was trained to see', async () => {
	// Cells 60 px wide in a 3000 px image. The first pass uses 640 px tiles; the second must
	// use 60 * 640 / target px tiles, which the fake detector observes as the tile width.
	const run = async (opts) => {
		const widths = new Set();
		const detector = {
			async detect(rgb, w) {
				widths.add(w);
				return [100, 200, 300].map((cx) => ({ cx, cy: 100, w: 60, h: 90, conf: 0.9 }));
			}
		};
		await detectScaleNormalized(detector, new Float32Array(3000 * 3000 * 3), 3000, 3000, opts);
		return [...widths];
	};
	assert.deepEqual(await run({}), [640, 1280]); // default 30 px
	assert.deepEqual(await run({ targetCellPx: 20 }), [640, 1920]);
});
