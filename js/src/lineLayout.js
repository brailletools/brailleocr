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

// Python's round() is round-half-to-even ("banker's rounding"); JS's Math.round()
// always rounds .5 up (and differs from Python on negative ties too, e.g.
// round(-1.5): Python -> -2, Math.round -> -1). insertSpaces() below needs to
// match pipeline.py's round() exactly since it's a direct port, or a tie can
// produce a different number of inferred space markers than the reference.
function bankersRound(x) {
	const floor = Math.floor(x);
	const diff = x - floor;
	if (diff < 0.5) return floor;
	if (diff > 0.5) return floor + 1;
	return floor % 2 === 0 ? floor : floor + 1;
}

// Fraction of the page's median cell height a cell may sit from where its line
// is predicted to be and still join it.
const LINE_BAND_FRAC = 0.55;
// How many of a line's most recent cells are used to estimate its local slope.
const SLOPE_WINDOW = 6;
// A line's local slope is only trusted once it spans this many cell widths;
// over a shorter run, ordinary box jitter looks like a steep slope.
const MIN_SLOPE_SPAN_CELLS = 3;
// Largest local slope, left over after removing the page-wide tilt, that a line
// is allowed to follow (perspective and page curl).
const MAX_LOCAL_SLOPE = 0.15;
// A gap this many cell pitches wide or wider contains at least one blank cell.
const SPACE_GAP_PITCHES = 1.5;
// A line needs this many gaps before its own pitch is preferred to the page's.
const MIN_GAPS_FOR_LINE_PITCH = 5;
// ...and its own pitch must be within this fraction of the page's to be believed.
const LINE_PITCH_TOLERANCE = 0.25;

/**
 * Estimates the page's tilt, in radians (positive: lines run downward to the
 * right), as the median direction from each cell to its nearest neighbor on the
 * right. Works up to about 25 degrees; quarter and half turns are handled by
 * orientation.js instead.
 * @param {Array<{cx: number, cy: number}>} cells
 * @returns {number}
 */
export function estimateSkew(cells) {
	const angles = [];
	for (const c of cells) {
		let nearest = Infinity;
		let angle = null;
		for (const o of cells) {
			const dx = o.cx - c.cx;
			const dy = o.cy - c.cy;
			if (dx <= 0 || Math.abs(dy) > 0.5 * dx) continue;
			const dist = dx * dx + dy * dy;
			if (dist < nearest) {
				nearest = dist;
				angle = Math.atan2(dy, dx);
			}
		}
		if (angle !== null) angles.push(angle);
	}
	return angles.length ? median(angles) : 0;
}

/** Predicted y at x for a line of {x, y} points, following its local slope. */
function predictY(line, cx, cellW) {
	const recent = line.slice(-SLOPE_WINDOW);
	const meanX = recent.reduce((sum, p) => sum + p.x, 0) / recent.length;
	const meanY = recent.reduce((sum, p) => sum + p.y, 0) / recent.length;
	let slope = 0;
	if (recent[recent.length - 1].x - recent[0].x >= MIN_SLOPE_SPAN_CELLS * cellW) {
		let num = 0;
		let den = 0;
		for (const p of recent) {
			num += (p.x - meanX) * (p.y - meanY);
			den += (p.x - meanX) ** 2;
		}
		slope = Math.max(-MAX_LOCAL_SLOPE, Math.min(MAX_LOCAL_SLOPE, num / den));
	}
	const last = line[line.length - 1];
	// Anchor on the recent cells' mean rather than the last cell alone, so one
	// badly placed box doesn't pull the rest of the line off course.
	return slope ? meanY + slope * (cx - meanX) : last.y;
}

/**
 * Groups cells into reading-order lines. The page-wide tilt is removed first,
 * then cells are visited left to right and each joins the line whose predicted
 * position it is closest to, so a line can bend gradually (perspective, curl).
 * Mirrors pipeline.py's group_into_lines().
 * @param {Array<Cell>} cells
 * @returns {Array<Array<Cell>>} lines from top to bottom, cells left to right
 */
export function groupIntoLines(cells) {
	if (cells.length === 0) return [];

	const angle = estimateSkew(cells);
	const cos = Math.cos(angle);
	const sin = Math.sin(angle);
	// Positions in the de-tilted frame, each remembering the cell it came from.
	const level = cells.map((cell) => ({
		cell,
		x: cell.cx * cos + cell.cy * sin,
		y: cell.cy * cos - cell.cx * sin
	}));

	const thresh = median(cells.map((c) => c.h)) * LINE_BAND_FRAC;
	const cellW = median(cells.map((c) => c.w));
	const lines = [];
	for (const point of level.sort((a, b) => a.x - b.x || a.y - b.y)) {
		let best = null;
		let bestDist = thresh;
		for (const line of lines) {
			const dist = Math.abs(point.y - predictY(line, point.x, cellW));
			if (dist < bestDist) {
				best = line;
				bestDist = dist;
			}
		}
		if (!best) {
			best = [];
			lines.push(best);
		}
		best.push(point);
	}
	// Order by height at the page's left side: a bent line's mean height depends
	// on how far across it reaches, its starting height much less so.
	const startY = (line) => median(line.slice(0, SLOPE_WINDOW).map((p) => p.y));
	return lines.sort((a, b) => startY(a) - startY(b)).map((line) => line.map((p) => p.cell));
}

function consecutiveGaps(sortedLine) {
	return sortedLine.slice(1).map((c, i) => c.cx - sortedLine[i].cx);
}

// Word spaces and missed cells make some gaps two or more pitches wide, so the
// pitch is read from the lower part of the gap distribution. The median would be
// pulled up on pages of short words, where wide gaps are half or more of all gaps;
// the lower quartile stays on ordinary letter-to-letter gaps until over three
// quarters of the gaps are wide.
const PITCH_QUANTILE = 0.25;

/**
 * The page's cell pitch: the horizontal distance between neighboring cells in a
 * word. Falls back to a multiple of cell width when no two cells share a line.
 * @param {Array<Array<Cell>>} lines
 * @param {number} avgCellW
 */
export function estimatePitch(lines, avgCellW) {
	const gaps = lines
		.flatMap((line) => consecutiveGaps([...line].sort((a, b) => a.cx - b.cx)))
		.sort((a, b) => a - b);
	return gaps.length ? gaps[Math.floor(PITCH_QUANTILE * (gaps.length - 1))] : avgCellW * 1.4;
}

/**
 * Sorts one line's cells into reading order and inserts a blank-cell marker
 * ({ bits: '000000', isSpace: true }) for every empty cell position in a gap.
 * Counted in cell pitches, not cell widths: boxes are narrower than the pitch,
 * so dividing by width turned one blank cell into two.
 * Mirrors pipeline.py's insert_spaces().
 * @param {Array<Cell>} lineCells
 * @param {number} pagePitch - from estimatePitch()
 * @returns {Array<Cell|SpaceMarker>}
 */
export function insertSpaces(lineCells, pagePitch) {
	const line = [...lineCells].sort((a, b) => a.cx - b.cx);
	if (line.length <= 1) return line;

	const gaps = consecutiveGaps(line);
	let pitch = pagePitch;
	if (gaps.length >= MIN_GAPS_FOR_LINE_PITCH) {
		const own = median(gaps);
		if (Math.abs(own - pagePitch) <= pagePitch * LINE_PITCH_TOLERANCE) pitch = own;
	}

	const result = [line[0]];
	for (let i = 1; i < line.length; i++) {
		const gap = line[i].cx - line[i - 1].cx;
		if (gap >= pitch * SPACE_GAP_PITCHES) {
			const nSpaces = Math.max(1, bankersRound(gap / pitch) - 1);
			for (let s = 0; s < nSpaces; s++) {
				result.push({ cx: null, cy: line[i].cy, bits: '000000', isSpace: true });
			}
		}
		result.push(line[i]);
	}
	return result;
}

/**
 * Convenience wrapper: groups cells into lines, then infers spaces within
 * each line, using one page-wide cell pitch (matches how pipeline.py's
 * process_container() wires these functions together).
 * @param {Array<Cell>} cells
 * @returns {Array<Array<Cell|SpaceMarker>>}
 */
export function layoutCellsIntoLines(cells) {
	if (cells.length === 0) return [];
	const lines = groupIntoLines(cells);
	const pitch = estimatePitch(lines, median(cells.map((c) => c.w)));
	return lines.map((line) => insertSpaces(line, pitch));
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
