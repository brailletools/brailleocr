// Scoring helpers for the evaluation runner. Pure functions, unit-tested.

/** Levenshtein distance between two strings. */
export function editDistance(a, b) {
	if (a === b) return 0;
	if (!a.length) return b.length;
	if (!b.length) return a.length;
	let prev = Array.from({ length: b.length + 1 }, (_, i) => i);
	for (let i = 1; i <= a.length; i++) {
		const cur = [i];
		for (let j = 1; j <= b.length; j++) {
			cur[j] = Math.min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] === b[j - 1] ? 0 : 1));
		}
		prev = cur;
	}
	return prev[b.length];
}

/**
 * One-to-one matching of predicted to ground-truth cells by centre distance,
 * nearest pairs first, within `fraction` of the mean ground-truth cell width.
 * @returns {Array<[number, number]>} [predictedIndex, groundTruthIndex]
 */
export function matchCells(predicted, truth, fraction = 0.5) {
	if (!predicted.length || !truth.length) return [];
	const limit = fraction * (truth.reduce((s, c) => s + c.w, 0) / truth.length);
	const pairs = [];
	for (let i = 0; i < predicted.length; i++) {
		for (let j = 0; j < truth.length; j++) {
			const dx = predicted[i].cx - truth[j].cx;
			if (Math.abs(dx) >= limit) continue;
			const dy = predicted[i].cy - truth[j].cy;
			const d = Math.hypot(dx, dy);
			if (d < limit) pairs.push([d, i, j]);
		}
	}
	pairs.sort((a, b) => a[0] - b[0]);
	const usedP = new Set();
	const usedT = new Set();
	const out = [];
	for (const [, i, j] of pairs) {
		if (usedP.has(i) || usedT.has(j)) continue;
		usedP.add(i);
		usedT.add(j);
		out.push([i, j]);
	}
	return out;
}

/**
 * Maps a cell found in an image turned clockwise by `degrees` back to the
 * coordinates of the original (width x height) image.
 */
export function unturnCell(cell, degrees, width, height) {
	const { cx: x, cy: y, w, h } = cell;
	if (degrees === 0) return cell;
	if (degrees === 180) return { ...cell, cx: width - 1 - x, cy: height - 1 - y };
	if (degrees === 90) return { ...cell, cx: y, cy: height - 1 - x, w: h, h: w };
	if (degrees === 270) return { ...cell, cx: width - 1 - y, cy: x, w: h, h: w };
	throw new Error(`rotation must be a multiple of 90, not ${degrees}`);
}

/** Mean, and a bootstrap 95% interval resampling whole items (pages), not cells. */
export function bootstrapMean(values, resamples = 2000, seed = 1) {
	const n = values.length;
	if (!n) return { mean: NaN, lo: NaN, hi: NaN, n };
	const mean = values.reduce((s, v) => s + v, 0) / n;
	let state = seed >>> 0;
	const rand = () => ((state = (state * 1664525 + 1013904223) >>> 0) / 2 ** 32);
	const means = [];
	for (let r = 0; r < resamples; r++) {
		let sum = 0;
		for (let i = 0; i < n; i++) sum += values[Math.floor(rand() * n)];
		means.push(sum / n);
	}
	means.sort((a, b) => a - b);
	return { mean, lo: means[Math.floor(0.025 * resamples)], hi: means[Math.floor(0.975 * resamples)], n };
}
