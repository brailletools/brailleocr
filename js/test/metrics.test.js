import { test } from 'node:test';
import assert from 'node:assert/strict';

import { bootstrapMean, editDistance, matchCells, selectItems, standardBoxes, truthPitch, unturnCell } from '../eval/metrics.js';
import { rotateImage } from '../src/orientation.js';

test('editDistance counts insertions, deletions and substitutions', () => {
	assert.equal(editDistance('', 'abc'), 3);
	assert.equal(editDistance('kitten', 'sitting'), 3);
	assert.equal(editDistance('same', 'same'), 0);
});

/** A row of cells `pitch` apart, with boxes of the given width over pitch. */
const row = (widthOverPitch, n = 6, pitch = 40) =>
	Array.from({ length: n }, (_, i) => ({ cx: i * pitch, cy: 0, w: widthOverPitch * pitch, h: 1.12 * pitch }));

test('matchCells pairs each cell once, nearest first, within a fraction of the pitch', () => {
	const truth = row(0.75);
	const predicted = [{ cx: 6, cy: 0 }, { cx: 8, cy: 0 }, { cx: 41, cy: 0 }, { cx: 900, cy: 0 }];
	assert.deepEqual(matchCells(predicted, truth), [[2, 1], [0, 0]]);
});

test('ground truth drawn wide or narrow is matched with the same distance', () => {
	// 12 px off centre: inside 0.375 of a 40 px pitch (15 px), but outside half of a narrow
	// box (0.54 * 40 / 2 = 10.8 px), which is what a width-based rule would have used.
	const predicted = row(0.75).map((c) => ({ cx: c.cx + 12, cy: 0 }));
	const wide = matchCells(predicted, row(0.75));
	const narrow = matchCells(predicted, row(0.54));
	assert.equal(wide.length, 6);
	assert.deepEqual(narrow, wide);
	assert.equal(truthPitch(row(0.75)), truthPitch(row(0.54)));
	assert.deepEqual(standardBoxes(row(0.75)), standardBoxes(row(0.54)));
});

test('truthPitch ignores the other side of the page and falls back to height when sparse', () => {
	const front = row(0.75).map((c) => ({ ...c, side: 'recto' }));
	const back = row(0.75).map((c) => ({ ...c, cx: c.cx + 15, side: 'verso' }));
	assert.equal(truthPitch([...front, ...back]), 40);
	assert.ok(Math.abs(truthPitch(row(0.75, 2)) - 40) < 1e-9);
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

const sample = [
	{ set: 'a-val', role: 'dev' },
	{ set: 'b-test', role: 'seen' },
	{ set: 'c-final', role: 'held-out' }
];
const names = (r) => r.items.map((i) => i.set);

test('selectItems leaves held-out sets alone unless the run is final', () => {
	assert.deepEqual(names(selectItems(sample)), ['a-val', 'b-test']);
	assert.match(selectItems(sample, { sets: ['c-final'] }).problem, /held out/);
	assert.match(selectItems(sample, { role: 'held-out' }).problem, /held out/);
	assert.deepEqual(names(selectItems(sample, { final: true })), ['a-val', 'b-test', 'c-final']);
});

test('selectItems refuses to try a tuning option on anything but development sets', () => {
	assert.deepEqual(names(selectItems(sample, { tuning: true, role: 'dev' })), ['a-val']);
	assert.match(selectItems(sample, { tuning: true }).problem, /development sets.*b-test/);
	assert.match(selectItems(sample, { tuning: true, sets: ['b-test'] }).problem, /b-test/);
});
