import { test } from 'node:test';
import assert from 'node:assert/strict';

import { bootstrapMean, editDistance, matchCells, unturnCell } from '../eval/metrics.js';
import { rotateImage } from '../src/orientation.js';

test('editDistance counts insertions, deletions and substitutions', () => {
	assert.equal(editDistance('', 'abc'), 3);
	assert.equal(editDistance('kitten', 'sitting'), 3);
	assert.equal(editDistance('same', 'same'), 0);
});

test('matchCells pairs each cell once, nearest first, within half a cell width', () => {
	const truth = [{ cx: 0, cy: 0, w: 20 }, { cx: 40, cy: 0, w: 20 }];
	const predicted = [{ cx: 3, cy: 0 }, { cx: 4, cy: 0 }, { cx: 41, cy: 0 }, { cx: 200, cy: 0 }];
	assert.deepEqual(matchCells(predicted, truth), [[2, 1], [0, 0]]);
});

for (const degrees of [0, 90, 180, 270]) {
	test(`unturnCell inverts rotateImage for a ${degrees} degree turn`, () => {
		// A 3x2 image with one marked pixel; find it after turning, map it back.
		const width = 3;
		const height = 2;
		const rgb = new Float32Array(width * height * 3);
		const at = { x: 2, y: 0 };
		rgb[(at.y * width + at.x) * 3] = 1;
		const turned = rotateImage({ rgb, width, height }, degrees);
		let found;
		for (let i = 0; i < turned.width * turned.height; i++) {
			if (turned.rgb[i * 3] === 1) found = { cx: i % turned.width, cy: Math.floor(i / turned.width), w: 1, h: 1 };
		}
		const back = unturnCell(found, degrees, width, height);
		assert.deepEqual([back.cx, back.cy], [at.x, at.y]);
	});
}

test('bootstrapMean resamples items and brackets the mean', () => {
	const b = bootstrapMean([0.9, 0.95, 1, 0.85, 0.92, 0.97]);
	assert.ok(b.lo <= b.mean && b.mean <= b.hi);
	assert.equal(b.n, 6);
	assert.ok(Number.isNaN(bootstrapMean([]).mean));
});
