import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import sharp from 'sharp';
import { CellClassifier } from '../src/classifier.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPOS_ROOT = join(__dirname, '..', '..', '..');
const SAMPLE_IMAGE = join(REPOS_ROOT, 'dataset', 'data', 'sample-images', 'IMG_3153.jpeg');
const MODEL_PATH = join(REPOS_ROOT, 'dataset', 'models', 'cell_classifier.onnx');
const REF_PATH = join(__dirname, 'fixtures', 'classifier_ref.json');

// dataset is a sibling repo, not checked out by default alongside a clean
// brailleocr checkout -- skip (not error) when its fixtures aren't present,
// rather than letting sharp()/CellClassifier.load() throw mid-test.
const FIXTURES_AVAILABLE = existsSync(SAMPLE_IMAGE) && existsSync(MODEL_PATH);
const skipReason = FIXTURES_AVAILABLE
	? false
	: 'requires a sibling `dataset` checkout (sample-images + models) -- not present in this checkout';

async function loadImageAsRgbHwc(path) {
	const { data, info } = await sharp(path).rotate().raw().toBuffer({ resolveWithObject: true });
	const rgb = new Float32Array(info.width * info.height * 3);
	for (let i = 0; i < info.width * info.height; i++) {
		rgb[i * 3] = data[i * info.channels];
		rgb[i * 3 + 1] = data[i * info.channels + 1];
		rgb[i * 3 + 2] = data[i * info.channels + 2];
	}
	return { rgb, width: info.width, height: info.height };
}

test('CellClassifier matches pipeline.py reclassify_cells() bit-for-bit-ish on real crops', { skip: skipReason }, async () => {
	const ref = JSON.parse(readFileSync(REF_PATH, 'utf-8'));
	const { rgb, width, height } = await loadImageAsRgbHwc(SAMPLE_IMAGE);
	assert.equal(width, ref.imgW);
	assert.equal(height, ref.imgH);

	const classifier = await CellClassifier.load(MODEL_PATH);
	const boxes = ref.cells.map((c) => ({ cx: c.cx, cy: c.cy, w: c.w, h: c.h }));
	const bits = await classifier.classify(rgb, width, height, boxes);

	let totalBits = 0;
	let matchingBits = 0;
	let exactCells = 0;
	for (let i = 0; i < ref.cells.length; i++) {
		const expected = ref.cells[i].bits;
		const actual = bits[i];
		if (actual === expected) exactCells++;
		for (let d = 0; d < 6; d++) {
			totalBits++;
			if (actual[d] === expected[d]) matchingBits++;
		}
	}

	const bitAgreement = matchingBits / totalBits;
	const cellAgreement = exactCells / ref.cells.length;
	console.log(
		`Bit-level agreement: ${(bitAgreement * 100).toFixed(1)}% ` +
			`(${matchingBits}/${totalBits}), exact-cell agreement: ${exactCells}/${ref.cells.length}`
	);

	// Own hand-rolled bilinear resize (not PIL's), so not bit-identical —
	// see the detector test's tolerance rationale for the same reasoning.
	assert.ok(bitAgreement > 0.97, `bit agreement ${bitAgreement} should be > 0.97`);
});

test('CellClassifier._cropAndResize produces a full, non-NaN crop for a box at the image edge', () => {
	// No real model/dataset fixtures needed: _cropAndResize never touches the
	// session, so a fake one is enough to exercise the crop-bounds math in
	// isolation. Regression test for the bounds-clamping fix — a box centered
	// right at the image corner previously let x0/y0 land at imgW/imgH (one
	// past the last valid pixel), which made the row-copy loop read past the
	// intended data and silently paste in a shorter/zero-filled subarray.
	const fakeSession = { inputNames: ['input'], outputNames: ['output'] };
	const classifier = new CellClassifier(fakeSession);
	const imgW = 10;
	const imgH = 10;
	const rgb = new Float32Array(imgW * imgH * 3).fill(100);
	const box = { cx: imgW, cy: imgH, w: 4, h: 4 };

	const crop = classifier._cropAndResize(rgb, imgW, imgH, box);

	assert.equal(crop.length, 3 * 64 * 64);
	assert.ok(crop.every((v) => Number.isFinite(v)), 'crop must contain only finite values, no NaN/undefined');
});
