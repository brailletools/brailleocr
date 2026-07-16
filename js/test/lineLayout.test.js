import { test } from 'node:test';
import assert from 'node:assert/strict';

import { groupIntoLines, insertSpaces, layoutCellsIntoLines, bitsToBraille, layoutToUnicodeBraille } from '../src/lineLayout.js';

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

test('groupIntoLines chains off the previous cell added, not a running mean or the line\'s first cell', () => {
	// Each consecutive step is 25px (< the 27.5px threshold for h=50 cells),
	// but the first-to-last drift is 75px (> threshold). A "compare to the
	// line's first cell" implementation would incorrectly split this into
	// multiple lines partway through; chaining off the previous cell keeps
	// it as one line, matching pipeline.py's group_into_lines() exactly.
	const cells = [100, 125, 150, 175].map((cy) => ({ cx: 0, cy, w: 25, h: 50, bits: '000000' }));
	const lines = groupIntoLines(cells);
	assert.equal(lines.length, 1);
	assert.equal(lines[0].length, 4);
});

test('groupIntoLines starts a new line once the gap from the previous cell exceeds the threshold', () => {
	const line1 = [100, 110].map((cy) => ({ cx: 0, cy, w: 25, h: 50, bits: '000000' }));
	const line2 = [250, 260].map((cy) => ({ cx: 0, cy, w: 25, h: 50, bits: '000000' }));
	const lines = groupIntoLines([...line1, ...line2]);
	assert.equal(lines.length, 2);
	assert.equal(lines[0].length, 2);
	assert.equal(lines[1].length, 2);
});

test('insertSpaces leaves normally-spaced cells alone and inserts multiple markers across a wide gap', () => {
	// avgCellW=25 (page-wide). Within this line: gaps [30,30,140,30], medianGap=30,
	// spaceThresh=30+25*0.5=42.5 -- only the 140px gap exceeds it.
	// nSpaces = round((140-30)/25) = round(4.4) = 4.
	const line = [0, 30, 60, 200, 230].map((cx) => ({ cx, cy: 100, w: 25, h: 50, bits: '000000' }));
	const result = insertSpaces(line, 25);

	const spaces = result.filter((c) => c.isSpace);
	const nonSpaces = result.filter((c) => !c.isSpace);
	assert.equal(spaces.length, 4);
	assert.equal(nonSpaces.length, 5);
	// Spaces must sit between the cx=60 and cx=200 cells, in order.
	const idx60 = result.findIndex((c) => c.cx === 60);
	const idx200 = result.findIndex((c) => c.cx === 200);
	assert.equal(idx200 - idx60, 5, 'expected 4 space markers directly between the two cells');
});

test('insertSpaces does not insert a space for a single-cell line', () => {
	const result = insertSpaces([{ cx: 0, cy: 100, w: 25, h: 50, bits: '100000' }], 25);
	assert.equal(result.length, 1);
});

test('layoutCellsIntoLines + layoutToUnicodeBraille produce two lines with a mid-line word gap', () => {
	// Same shape as pipeline.py's process_container() wiring: group into
	// lines, then infer spaces per line using one page-wide avgCellW.
	const line1 = [0, 30, 60, 200, 230].map((cx) => ({ cx, cy: 100, w: 25, h: 50, bits: '100000' }));
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
