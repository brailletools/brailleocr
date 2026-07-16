// Ports pipeline.py's group_into_lines()/insert_spaces() — turning a flat list
// of classified cells (each {cx, cy, w, h, bits}) into reading-order lines
// with spaces inferred between words. Pure geometry/statistics, no model
// inference — takes CellDetector/CellClassifier output as input.

/** @typedef {{cx: number, cy: number, w: number, h: number, bits: string}} Cell */
/** @typedef {{cx: null, cy: number, bits: string, isSpace: true}} SpaceMarker */

function median(values) {
	const sorted = [...values].sort((a, b) => a - b);
	const mid = Math.floor(sorted.length / 2);
	return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

// Fraction of the page's median cell height a cell must fall within of the
// *previous* cell added to the current line to join it — matches
// pipeline.py's group_into_lines() exactly (chains off the last cell added,
// not a running mean of the line, so a slight page skew doesn't make the
// threshold drift away from consecutive real neighbors).
const LINE_BAND_FRAC = 0.55;
// Extra fraction of the page's median cell width, added to a line's own
// median cell-to-cell gap, that a gap must exceed before it's treated as a
// word space rather than normal inter-cell spacing — matches pipeline.py's
// insert_spaces().
const SPACE_GAP_EXTRA_FRAC = 0.5;

/**
 * Groups cells into reading-order lines by y-position. Direct port of
 * pipeline.py's group_into_lines(): sorts by cy, then sequentially chains
 * each cell onto the current line if it's within LINE_BAND_FRAC * (page's
 * median cell height) of the *last cell added to that line* — not a running
 * mean — starting a new line otherwise. Lines are NOT sorted by cx here
 * (callers needing reading order within a line should sort, same as
 * insertSpaces() already does internally).
 * @param {Array<Cell>} cells
 * @returns {Array<Array<Cell>>}
 */
export function groupIntoLines(cells) {
	if (cells.length === 0) return [];

	const avgH = median(cells.map((c) => c.h));
	const thresh = avgH * LINE_BAND_FRAC;
	const sortedByY = [...cells].sort((a, b) => a.cy - b.cy);

	const lines = [];
	let current = [sortedByY[0]];
	for (let i = 1; i < sortedByY.length; i++) {
		const cell = sortedByY[i];
		if (Math.abs(cell.cy - current[current.length - 1].cy) < thresh) {
			current.push(cell);
		} else {
			lines.push(current);
			current = [cell];
		}
	}
	lines.push(current);
	return lines;
}

/**
 * Sorts one line's cells into reading order and inserts space markers
 * ({ bits: '000000', isSpace: true }) wherever the gap between consecutive
 * cells is wide enough to be a word space rather than normal letter spacing.
 * Direct port of pipeline.py's insert_spaces(): medianGap is computed from
 * THIS line's own consecutive-cell gaps (robust to this line's own spacing
 * quirks); avgCellW is the page-wide median cell width, passed in by the
 * caller (computed once across every cell on the page, not per line).
 * @param {Array<Cell>} lineCells
 * @param {number} avgCellW
 * @returns {Array<Cell|SpaceMarker>}
 */
export function insertSpaces(lineCells, avgCellW) {
	const line = [...lineCells].sort((a, b) => a.cx - b.cx);
	if (line.length <= 1) return line;

	const gaps = [];
	for (let i = 1; i < line.length; i++) gaps.push(line[i].cx - line[i - 1].cx);
	const medianGap = median(gaps);
	const spaceThresh = medianGap + avgCellW * SPACE_GAP_EXTRA_FRAC;

	const result = [line[0]];
	for (let i = 1; i < line.length; i++) {
		const gap = line[i].cx - line[i - 1].cx;
		if (gap > spaceThresh) {
			const nSpaces = Math.max(1, Math.round((gap - medianGap) / avgCellW));
			for (let s = 0; s < nSpaces; s++) {
				result.push({ cx: null, cy: line[i - 1].cy, bits: '000000', isSpace: true });
			}
		}
		result.push(line[i]);
	}
	return result;
}

/**
 * Convenience wrapper: groups cells into lines, then infers spaces within
 * each line, using one page-wide median cell width for every line's space
 * threshold (matches how pipeline.py's process_container() wires these two
 * functions together).
 * @param {Array<Cell>} cells
 * @returns {Array<Array<Cell|SpaceMarker>>}
 */
export function layoutCellsIntoLines(cells) {
	if (cells.length === 0) return [];
	const avgCellW = median(cells.map((c) => c.w));
	return groupIntoLines(cells).map((line) => insertSpaces(line, avgCellW));
}

/**
 * Converts a 6-char '0'/'1' bit string (index i = dot i+1, i.e. index 0 =
 * dot 1 top-left ... index 5 = dot 6 bottom-right) to its Unicode Braille
 * Patterns character. Reads left-to-right with NO string reversal — verified
 * by hand to be mathematically identical to pipeline.py's bits_to_braille()
 * (`chr(0x2800 + int(bits6[::-1], 2))`): reversing a bit string then parsing
 * it most-significant-bit-first is the same value as reading the original
 * string least-significant-bit-first, which is what summing bit[i]*2^i does
 * directly.
 * @param {string} bits6
 * @returns {string}
 */
export function bitsToBraille(bits6) {
	let value = 0;
	for (let i = 0; i < 6; i++) {
		if (bits6[i] === '1') value |= 1 << i;
	}
	return String.fromCodePoint(0x2800 + value);
}

/**
 * Renders layoutCellsIntoLines()'s output as Unicode braille text, one line
 * per array entry, joined with '\n'.
 * @param {Array<Array<Cell|SpaceMarker>>} lines
 * @returns {string}
 */
export function layoutToUnicodeBraille(lines) {
	return lines.map((line) => line.map((c) => bitsToBraille(c.bits)).join('')).join('\n');
}
