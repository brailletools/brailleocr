import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import sharp from 'sharp';
import { CellDetector } from '../src/detector.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const REPOS_ROOT = join(__dirname, '..', '..', '..');
const SAMPLE_IMAGE = join(REPOS_ROOT, 'dataset', 'data', 'sample-images', 'IMG_3153.jpeg');
const MODEL_PATH = join(REPOS_ROOT, 'dataset', 'models', 'cell_detector.onnx');

async function loadImageAsRgbHwc(path) {
	// .rotate() with no args: auto-orient from EXIF, matching how ultralytics
	// loads images via cv2.imread (which respects EXIF orientation) — see
	// scoping notes on why this matters (letterbox output shape differs
	// entirely between portrait/landscape interpretation of the same file).
	const { data, info } = await sharp(path)
		.rotate()
		.raw()
		.toBuffer({ resolveWithObject: true });
	const rgb = new Float32Array(info.width * info.height * 3);
	for (let i = 0; i < info.width * info.height; i++) {
		rgb[i * 3] = data[i * info.channels];
		rgb[i * 3 + 1] = data[i * info.channels + 1];
		rgb[i * 3 + 2] = data[i * info.channels + 2];
	}
	return { rgb, width: info.width, height: info.height };
}

test('CellDetector matches the Python/ultralytics reference on a real sample image', async () => {
	const { rgb, width, height } = await loadImageAsRgbHwc(SAMPLE_IMAGE);
	console.log(`Loaded image: ${width}x${height}`);

	const detector = await CellDetector.load(MODEL_PATH);
	const boxes = await detector.detect(rgb, width, height, { confThreshold: 0.15, iouThreshold: 0.7 });

	console.log(`Detected ${boxes.length} boxes (reference: 38)`);
	const top3 = [...boxes].sort((a, b) => b.conf - a.conf).slice(0, 3);
	for (const b of top3) {
		console.log(
			`  cx=${b.cx.toFixed(1)} cy=${b.cy.toFixed(1)} w=${b.w.toFixed(1)} h=${b.h.toFixed(1)} conf=${b.conf.toFixed(3)}`
		);
	}

	// Reference (pipeline.py / ultralytics on cell_detector.onnx, conf=0.15,
	// iou=0.7): 38 boxes, top box xyxy=[2333.5, 2907.9, 2451.3, 3092.5] conf=0.82.
	// Tolerance of 2 rather than exact equality: this JS port reimplements
	// bilinear resize by hand rather than depending on cv2, so it isn't
	// bit-identical — a handful of borderline candidates right at the conf/IoU
	// threshold can land on either side. pipeline.py's own grid-fill rescue
	// exists precisely because single-box noise like this is expected and
	// tolerated at the orchestration layer, not something the detector itself
	// needs to reproduce exactly.
	assert.ok(
		Math.abs(boxes.length - 38) <= 2,
		`expected ~38 boxes (±2), got ${boxes.length}`
	);

	const refCx = (2333.5 + 2451.3) / 2;
	const refCy = (2907.9 + 3092.5) / 2;
	const match = boxes.find((b) => Math.abs(b.cx - refCx) < 5 && Math.abs(b.cy - refCy) < 5);
	assert.ok(match, `expected a box near (${refCx.toFixed(1)}, ${refCy.toFixed(1)})`);
	assert.ok(Math.abs(match.conf - 0.82) < 0.02, `conf ${match.conf} should be close to 0.82`);
});
