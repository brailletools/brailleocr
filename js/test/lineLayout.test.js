import { test } from 'node:test';
import assert from 'node:assert/strict';

import { estimateSkew, groupIntoLines, insertSpaces, layoutCellsIntoLines, bitsToBraille, layoutToUnicodeBraille } from '../src/lineLayout.js';

test('bitsToBraille matches pipeline.py bits_to_braille() literal codepoints', () => {
	// Cross-checked by hand against chr(0x2800 + int(bits6[::-1], 2)) for each
	// case -- see lineLayout.js's doc comment for why no reversal is needed here.
	const cases = [
		['111001', 0x2827],
		['111000', 0x2807],
		['110100', 0x280b],
		['111101', 0x282f],
		['000000', 0x2800],
		['100000', 0x2801],
		['000001', 0x2820],
		['111111', 0x283f]
	];
	for (const [bits, codepoint] of cases) {
		assert.equal(bitsToBraille(bits), String.fromCodePoint(codepoint), `bits=${bits}`);
	}
});

const cell = (cx, cy, bits = '100000') => ({ cx, cy, w: 25, h: 50, bits });

/** A page of evenly pitched cells, then moved by `place`. */
function page(nLines, nCells, place = (x, y) => [x, y]) {
	const cells = [];
	for (let l = 0; l < nLines; l++) {
		for (let i = 0; i < nCells; i++) {
			const [cx, cy] = place(100 + i * 35, 100 + l * 80);
			cells.push({ ...cell(cx, cy), line: l, index: i });
		}
	}
	return cells;
}

function assertLinesIntact(lines, nLines, nCells) {
	assert.equal(lines.length, nLines);
	lines.forEach((line, l) => {
		assert.deepEqual(
			line.map((c) => [c.line, c.index]),
			Array.from({ length: nCells }, (_, i) => [l, i])
		);
	});
}

for (const degrees of [-10, -2, 1, 2, 5, 10]) {
	test(`groupIntoLines keeps lines intact on a page tilted ${degrees} degrees`, () => {
		const t = (degrees * Math.PI) / 180;
		const cells = page(12, 40, (x, y) => [
			x * Math.cos(t) - y * Math.sin(t),
			x * Math.sin(t) + y * Math.cos(t)
		]);
		assert.ok(Math.abs(estimateSkew(cells) - t) < 0.005);
		assertLinesIntact(groupIntoLines(cells), 12, 40);
	});
}

test('groupIntoLines follows lines that converge, as under perspective', () => {
	// Top line slopes down to the right and bottom line up: no single rotation levels both.
	const cells = page(10, 40, (x, y) => [x, y + ((460 - y) / 360) * 0.05 * (x - 800)]);
	assertLinesIntact(groupIntoLines(cells), 10, 40);
});

test('groupIntoLines follows a curled page', () => {
	const cells = page(10, 40, (x, y) => [x, y + 60 * Math.sin((Math.PI * (x - 100)) / 1365)]);
	assertLinesIntact(groupIntoLines(cells), 10, 40);
});

test('groupIntoLines returns the original cell objects, not de-tilted copies', () => {
	const cells = page(2, 5);
	assert.ok(groupIntoLines(cells).flat().every((c) => cells.includes(c)));
});

test('insertSpaces counts blank cells in pitches, not cell widths', () => {
	// Pitch 50 with 34px boxes (the real-photo ratio): one blank cell is a 100px
	// gap, which dividing by width reported as two.
	const line = [0, 50, 100, 200, 250].map((cx) => ({ ...cell(cx, 100), w: 34 }));
	assert.equal(insertSpaces(line, 50).filter((c) => c.isSpace).length, 1);
});

test('groupIntoLines starts a new line once the gap from the previous cell exceeds the threshold', () => {
	const line1 = [100, 110].map((cy) => ({ cx: 0, cy, w: 25, h: 50, bits: '000000' }));
	const line2 = [250, 260].map((cy) => ({ cx: 0, cy, w: 25, h: 50, bits: '000000' }));
	const lines = groupIntoLines([...line1, ...line2]);
	assert.equal(lines.length, 2);
	assert.equal(lines[0].length, 2);
	assert.equal(lines[1].length, 2);
});

test('insertSpaces leaves normally-spaced cells alone and fills a wide gap', () => {
	// pitch 30; the 150px gap holds 150/30 - 1 = 4 blank cells.
	const line = [0, 30, 60, 210, 240].map((cx) => cell(cx, 100));
	const result = insertSpaces(line, 30);

	assert.equal(result.filter((c) => c.isSpace).length, 4);
	const idx60 = result.findIndex((c) => c.cx === 60);
	const idx210 = result.findIndex((c) => c.cx === 210);
	assert.equal(idx210 - idx60, 5, 'expected 4 space markers directly between the two cells');
});

test('insertSpaces does not insert a space for a single-cell line', () => {
	const result = insertSpaces([{ cx: 0, cy: 100, w: 25, h: 50, bits: '100000' }], 25);
	assert.equal(result.length, 1);
});

test('insertSpaces takes a synthetic space marker\'s cy from the cell AFTER the gap, matching pipeline.py', () => {
	// pipeline.py's insert_spaces() uses line[i]['cy'] (the cell after the gap),
	// not the cell before it -- matters when a line isn't perfectly horizontal.
	// Needs >=3 cells so medianGap (from all gaps) differs from the one wide gap
	// being tested -- with only one gap, median(gap)==gap and no space ever fires.
	const line = [
		{ cx: 0, cy: 100, w: 25, h: 50, bits: '000000' },
		{ cx: 30, cy: 100, w: 25, h: 50, bits: '000000' },
		{ cx: 200, cy: 108, w: 25, h: 50, bits: '000000' }
	];
	const result = insertSpaces(line, 30);
	const space = result.find((c) => c.isSpace);
	assert.ok(space, 'expected a space marker for the wide 30->200 gap');
	assert.equal(space.cy, 108, 'space marker cy should match the cell after the gap, not before it');
});

test('insertSpaces rounds ties the way Python\'s round() does (round-half-to-even), not Math.round()', () => {
	// Math.round(2.5) is 3 (JS always rounds .5 up), but Python's round(2.5) is 2
	// (ties go to the nearest EVEN integer -- 2, not 3). Construct a gap where
	// (gap - medianGap) / avgCellW lands exactly on 2.5, so the two disagree.
	// pitch 40: gap 140 is 3.5 pitches -> round-half-to-even(3.5) - 1 = 3, and
	// gap 100 is 2.5 pitches -> 2 - 1 = 1 (Math.round would give 2).
	const spacesFor = (gap) =>
		insertSpaces([cell(0, 100), cell(gap, 100)], 40).filter((c) => c.isSpace).length;
	assert.equal(spacesFor(140), 3);
	assert.equal(spacesFor(100), 1);
});

test('layoutCellsIntoLines + layoutToUnicodeBraille produce two lines with a mid-line word gap', () => {
	// Same shape as pipeline.py's process_container() wiring: group into
	// lines, then infer spaces per line using one page-wide avgCellW.
	const line1 = [0, 30, 60, 210, 240].map((cx) => ({ cx, cy: 100, w: 25, h: 50, bits: '100000' }));
	const line2 = [0, 30].map((cx) => ({ cx, cy: 250, w: 25, h: 50, bits: '000001' }));
	const lines = layoutCellsIntoLines([...line1, ...line2]);

	assert.equal(lines.length, 2);
	assert.equal(lines[0].filter((c) => c.isSpace).length, 4, 'line 1 should get the wide-gap spaces');
	assert.equal(lines[1].filter((c) => c.isSpace).length, 0, 'line 2 has no wide gaps');

	const text = layoutToUnicodeBraille(lines);
	const [renderedLine1, renderedLine2] = text.split('\n');
	assert.equal(renderedLine1, '⠁⠁⠁⠀⠀⠀⠀⠁⠁');
	assert.equal(renderedLine2, '⠠⠠');
});
