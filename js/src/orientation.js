// Finds which way up a photo of a braille page is. Phones photographing a page
// from above often record no meaningful orientation, and the detector and
// classifier were only trained on upright pages, so a sideways or upside-down
// photo reads as noise.
//
// The image is tried at several turns and two questions are answered:
//   1. Sideways? The share of very confident detections is far higher on the
//      axis where lines run horizontally. Both directions along an axis have to
//      be tried: an upside-down page can look as unconfident as a sideways one.
//   2. Upside down? Confidence can't settle this (lighting can favor the wrong
//      way), but text braille uses the top row of a cell (dots 1, 4) much more
//      than the bottom row (dots 3, 6), and a half turn swaps them.
//
// Thresholds come from fifteen real page photos. The caller should always let
// the user override the result.

import { detectScaleNormalized } from './tiling.js';

/** Detections at or above this confidence count toward the orientation score. */
export const SURE_CONF = 0.6;
/** An upright reading at least this good ends the search early. */
export const CLEAR_SHARE = 0.9;
export const CLEAR_RATIO = 1.1;
/** The other axis must beat the as-shot axis by this much to turn sideways. */
export const AXIS_MARGIN = 0.1;
/** The turned page's top/bottom ratio must be this much higher to flip it. */
export const FLIP_MARGIN = 0.15;
/** A turn is only made toward an axis or page with at least this confident share. */
export const MIN_SHARE_TO_TURN = 0.25;
/** Fewer cells than this is too little evidence to turn a photo. */
export const MIN_CELLS_TO_TURN = 20;

/**
 * Turns an image clockwise by a quarter-turn multiple.
 * @param {{rgb: Float32Array, width: number, height: number}} image - HWC RGB
 * @param {0|90|180|270} degrees
 * @returns {{rgb: Float32Array, width: number, height: number}}
 */
export function rotateImage(image, degrees) {
	const { rgb, width, height } = image;
	if (![0, 90, 180, 270].includes(degrees)) {
		throw new Error(`rotation must be 0, 90, 180 or 270 degrees, not ${degrees}`);
	}
	if (degrees === 0) return image;

	const outW = degrees === 180 ? width : height;
	const outH = degrees === 180 ? height : width;
	const out = new Float32Array(rgb.length);
	for (let y = 0; y < height; y++) {
		for (let x = 0; x < width; x++) {
			let dx, dy;
			if (degrees === 90) {
				dx = height - 1 - y;
				dy = x;
			} else if (degrees === 180) {
				dx = width - 1 - x;
				dy = height - 1 - y;
			} else {
				dx = y;
				dy = width - 1 - x;
			}
			const src = (y * width + x) * 3;
			const dst = (dy * outW + dx) * 3;
			out[dst] = rgb[src];
			out[dst + 1] = rgb[src + 1];
			out[dst + 2] = rgb[src + 2];
		}
	}
	return { rgb: out, width: outW, height: outH };
}

/** Share of detections at or above SURE_CONF. */
export function confidentShare(boxes) {
	return boxes.length ? boxes.filter((b) => b.conf >= SURE_CONF).length / boxes.length : 0;
}

/**
 * Raised dots in the top row divided by raised dots in the bottom row.
 * @param {string[]} bits - 6-char strings, index i = dot i+1
 */
export function topBottomRatio(bits) {
	let top = 0;
	let bottom = 0;
	for (const b of bits) {
		top += (b[0] === '1') + (b[3] === '1');
		bottom += (b[2] === '1') + (b[5] === '1');
	}
	return top / Math.max(1, bottom);
}

/**
 * Detects and classifies the cells of a page, turning the image first.
 *
 * With rotation 'auto' the turn is estimated (see this module's header). A
 * clearly upright photo costs one detection pass; anything else costs four.
 *
 * @param {import('./detector.js').CellDetector} detector
 * @param {import('./classifier.js').CellClassifier} classifier
 * @param {{rgb: Float32Array, width: number, height: number}} image
 * @param {{rotation?: 'auto'|0|90|180|270, confThreshold?: number,
 *          onStage?: (stage: 'detecting'|'orienting'|'classifying') => void}} [opts]
 * @returns {Promise<{rotation: 0|90|180|270,
 *          cells: Array<{cx:number, cy:number, w:number, h:number, conf:number, bits:string}>}>}
 *   Cell coordinates are in the turned image.
 */
export async function readPage(detector, classifier, image, opts = {}) {
	const { rotation = 'auto', onStage, ...detectOpts } = opts;

	// Turned pixels are not kept between steps: four copies of a phone photo
	// would be several hundred megabytes in the browser.
	const detect = async (degrees) => {
		const turned = rotateImage(image, degrees);
		const boxes = await detectScaleNormalized(
			detector,
			turned.rgb,
			turned.width,
			turned.height,
			detectOpts
		);
		return { degrees, boxes, share: confidentShare(boxes) };
	};
	const classify = async (candidate) => {
		const turned = rotateImage(image, candidate.degrees);
		const bits = await classifier.classify(
			turned.rgb,
			turned.width,
			turned.height,
			candidate.boxes
		);
		return { ...candidate, bits, ratio: topBottomRatio(bits) };
	};
	const result = ({ degrees, boxes, bits }) => ({
		rotation: degrees,
		cells: boxes.map((b, i) => ({ ...b, bits: bits[i] }))
	});

	onStage?.('detecting');
	const asShot = await detect(rotation === 'auto' ? 0 : rotation);
	if (rotation !== 'auto') {
		onStage?.('classifying');
		return result(await classify(asShot));
	}

	onStage?.('classifying');
	const first = await classify(asShot);
	const enough = (c) => c.boxes.length >= MIN_CELLS_TO_TURN;
	if (enough(first) && first.share >= CLEAR_SHARE && first.ratio >= CLEAR_RATIO) {
		return result(first);
	}

	onStage?.('orienting');
	const half = await detect(180);
	const quarter = await detect(90);
	const threeQuarter = await detect(270);
	const score = (...pair) => Math.max(0, ...pair.filter(enough).map((c) => c.share));
	const acrossScore = score(quarter, threeQuarter);
	const sideways =
		acrossScore >= MIN_SHARE_TO_TURN && acrossScore > score(first, half) + AXIS_MARGIN;

	onStage?.('classifying');
	if (sideways) {
		const readable = [];
		for (const c of [quarter, threeQuarter].filter(enough)) readable.push(await classify(c));
		return result(readable.reduce((a, b) => (b.ratio > a.ratio ? b : a)));
	}
	if (!enough(half) || half.share < MIN_SHARE_TO_TURN) return result(first);
	const flipped = await classify(half);
	// A page that already reads as top-heavy is never turned over.
	const flip =
		!enough(first) ||
		(first.ratio < CLEAR_RATIO && flipped.ratio > first.ratio + FLIP_MARGIN);
	return result(flip ? flipped : first);
}
