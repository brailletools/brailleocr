import { test } from 'node:test';
import assert from 'node:assert/strict';

import { readPage, rotateImage, topBottomRatio } from '../src/orientation.js';

/** 3x2 image whose red channel numbers the pixels 0..5, row by row. */
const numbered = () => {
	const rgb = new Float32Array(18);
	for (let i = 0; i < 6; i++) rgb[i * 3] = i;
	return { rgb, width: 3, height: 2 };
};
const reds = ({ rgb }) => Array.from({ length: rgb.length / 3 }, (_, i) => rgb[i * 3]);

test('rotateImage turns clockwise', () => {
	assert.deepEqual(reds(rotateImage(numbered(), 90)), [3, 0, 4, 1, 5, 2]);
	assert.deepEqual(reds(rotateImage(numbered(), 180)), [5, 4, 3, 2, 1, 0]);
	assert.deepEqual(reds(rotateImage(numbered(), 270)), [2, 5, 1, 4, 0, 3]);
	assert.equal(rotateImage(numbered(), 90).width, 2);
	assert.throws(() => rotateImage(numbered(), 45), /rotation/);
});

test('topBottomRatio compares dots 1 and 4 with dots 3 and 6', () => {
	assert.equal(topBottomRatio(['100100', '100000', '001000']), 3);
});

// Fakes standing in for the ONNX models. The image's corner pixel records which
// way it has been turned, so the fakes can answer differently for each turn.
// `upright` is the clockwise turn that puts the fake page the right way up.
function fakes(upright, nCells = 30) {
	const SIZE = 64;
	const image = { rgb: new Float32Array(SIZE * SIZE * 3), width: SIZE, height: SIZE };
	image.rgb[0] = 1; // marks the original top-left corner
	const turnOf = (rgb) => {
		const at = (x, y) => rgb[(y * SIZE + x) * 3] === 1;
		if (at(0, 0)) return 0;
		if (at(SIZE - 1, 0)) return 90;
		if (at(SIZE - 1, SIZE - 1)) return 180;
		return 270;
	};
	const detector = {
		async detect(rgb) {
			const sideways = (turnOf(rgb) - upright + 360) % 180 !== 0;
			return Array.from({ length: nCells }, (_, i) => ({
				cx: 2 + i * 2,
				cy: 32,
				w: 1,
				h: 2,
				conf: sideways && i % 2 ? 0.1 : 0.9
			}));
		}
	};
	const classifier = {
		async classify(rgb, w, h, boxes) {
			const upsideDown = turnOf(rgb) === (upright + 180) % 360;
			return boxes.map(() => (upsideDown ? '001000' : '100000'));
		}
	};
	return { detector, classifier, image };
}

for (const upright of [0, 90, 180, 270]) {
	test(`readPage turns a photo that needs ${upright} degrees`, async () => {
		const { detector, classifier, image } = fakes(upright);
		const { rotation, cells } = await readPage(detector, classifier, image);
		assert.equal(rotation, upright);
		assert.ok(cells.every((c) => c.bits === '100000'));
	});
}

test('readPage obeys an explicit rotation', async () => {
	const { detector, classifier, image } = fakes(180);
	const { rotation } = await readPage(detector, classifier, image, { rotation: 90 });
	assert.equal(rotation, 90);
});

test('readPage leaves a photo alone when there are too few cells to judge', async () => {
	const { detector, classifier, image } = fakes(180, 5);
	const { rotation } = await readPage(detector, classifier, image);
	assert.equal(rotation, 0);
});

test('readPage stops after one pass on a clearly upright page', async () => {
	const { detector, classifier, image } = fakes(0);
	// Make the fake reading clearly top-heavy and count detector calls.
	let calls = 0;
	const detect = detector.detect;
	detector.detect = (...args) => (calls++, detect(...args));
	await readPage(detector, classifier, image, { rotation: 0 });
	const onePass = calls;
	calls = 0;
	const { rotation } = await readPage(detector, classifier, image);
	assert.equal(rotation, 0);
	assert.equal(calls, onePass);
});

test('readPage does not turn toward a reading it has little confidence in', async () => {
	const { detector, classifier, image } = fakes(180);
	const detect = detector.detect;
	detector.detect = async (...args) => (await detect(...args)).map((b) => ({ ...b, conf: 0.1 }));
	const { rotation } = await readPage(detector, classifier, image);
	assert.equal(rotation, 0);
});
