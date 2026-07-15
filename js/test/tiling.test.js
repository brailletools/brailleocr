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
