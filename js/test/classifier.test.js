import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import sharp from 'sharp';
import { CellClassifier } from '../src/classifier.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPOS_ROOT = join(__dirname, '..', '..', '..');
const SAMPLE_IMAGE = join(REPOS_ROOT, 'dataset', 'data', 'sample-images', 'IMG_3153.jpeg');
const MODEL_PATH = join(REPOS_ROOT, 'dataset', 'models', 'cell_classifier.onnx');
const REF_PATH = join(__dirname, 'fixtures', 'classifier_ref.json');

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

test('CellClassifier matches pipeline.py reclassify_cells() bit-for-bit-ish on real crops', async () => {
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
