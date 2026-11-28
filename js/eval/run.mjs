#!/usr/bin/env node
// Scores the SHIPPED pipeline (readPage + layout, the code the web editor runs) on the
// held-out sets listed by experiments/export_eval_manifest.py.
//
//   node eval/run.mjs --cells /tmp/eval-cells.json --out /tmp/eval-results.json \
//        [--sets dsbi-test,angelina-uploaded] [--limit 5] [--rotation auto|0]
//        [--detector PATH] [--classifier PATH] [--data ../../dataset/data] [--iou 0.7]
//        [--role dev|seen|test|held-out] [--final]
//
// Test discipline (AUDIT section I): a tuning option (--iou) only runs on 'dev' sets, and
// held-out sets only run with --final.
//   node eval/run.mjs --summarize /tmp/eval-results.json
//   node eval/run.mjs --orientation ../../braille-research/eval/orientation.json
//
// Units are stated in every line of the summary. Intervals resample pages.

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { dirname, join, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';

import { CellDetector } from '../src/detector.js';
import { CellClassifier } from '../src/classifier.js';
import { readPage } from '../src/orientation.js';
import { layoutCellsIntoLines, layoutToUnicodeBraille } from '../src/lineLayout.js';
import { bootstrapMean, editDistance, matchCells, selectItems, standardBoxes, unturnCell } from './metrics.js';

const here = dirname(fileURLToPath(import.meta.url));

function args() {
	const out = {};
	const argv = process.argv.slice(2);
	for (let i = 0; i < argv.length; i++) {
		if (argv[i].startsWith('--')) out[argv[i].slice(2)] = argv[i + 1]?.startsWith('--') || argv[i + 1] === undefined ? true : argv[++i];
	}
	return out;
}

async function loadRgb(path) {
	const { data, info } = await sharp(path).rotate().removeAlpha().toColourspace('srgb').raw().toBuffer({ resolveWithObject: true });
	return { rgb: Float32Array.from(data), width: info.width, height: info.height };
}

const brailleOf = (cells) => layoutToUnicodeBraille(layoutCellsIntoLines(cells));

async function scoreItem(item, image, models, rotationOpt, detectOpts = {}) {
	const { detector, classifier } = models;
	const t0 = Date.now();
	const page = await readPage(detector, classifier, image, { rotation: rotationOpt, confThreshold: 0.15, ...detectOpts });
	const seconds = (Date.now() - t0) / 1000;
	const predicted = page.cells.map((c) => unturnCell(c, page.rotation, image.width, image.height));
	const truth = item.cells;
	const row = { id: item.id, set: item.set, kind: item.kind ?? null, rotation: page.rotation, predicted: predicted.length, truth: truth.length, seconds };

	if (item.text !== undefined) {
		// Text set (BrailleBench): keep the Braille the reader produced; experiments/score_text.py
		// back-translates and scores it.
		row.braille = brailleOf(predicted);
		row.reference = item.text;
		row.grade = item.grade;
		return row;
	}
	if (item.set === 'not-braille') return row;
	if (item.detection === false) {
		row.detection = false;
	} else {
		const matches = matchCells(predicted, truth);
		row.matched = matches.length;
		row.recall = truth.length ? matches.length / truth.length : null;
		row.precision = predicted.length ? matches.length / predicted.length : null;
		if (truth.every((c) => c.bits)) {
			let correct = 0;
			for (const [i, j] of matches) correct += predicted[i].bits === truth[j].bits;
			row.cellAccuracyDetected = matches.length ? correct / matches.length : null;
			row.cellsWrong = matches.length - correct;
			// Cell error rate: the same layout code on both sides, so what differs is the
			// cells themselves (missed, extra, misread), not how they are arranged.
			const ref = brailleOf(standardBoxes(truth));
			const hyp = brailleOf(predicted);
			row.cellErrorRate = ref.length ? editDistance(ref, hyp) / ref.length : null;
		}
	}
	if (truth.length && truth.every((c) => c.bits)) {
		// Classify the ground-truth boxes on the upright image: isolates the classifier.
		const upright = page.rotation === 0 ? image : await loadRgb(join(models.dataRoot, item.image));
		const bits = await classifier.classify(upright.rgb, upright.width, upright.height, truth);
		const by = {};
		truth.forEach((c, k) => {
			const side = c.side ?? 'front';
			by[side] ??= { n: 0, ok: 0 };
			by[side].n++;
			by[side].ok += bits[k] === c.bits;
		});
		row.classifyTruth = by;
	}
	return row;
}

function summarize(rows) {
	const lines = [];
	const sets = [...new Set(rows.map((r) => r.set))];
	for (const set of sets) {
		const rs = rows.filter((r) => r.set === set);
		const fmt = (b, pct = true) => (Number.isNaN(b.mean) ? 'n/a' : `${pct ? (100 * b.mean).toFixed(1) + '%' : b.mean.toFixed(2)} [${pct ? (100 * b.lo).toFixed(1) : b.lo.toFixed(2)}, ${pct ? (100 * b.hi).toFixed(1) : b.hi.toFixed(2)}]`);
		lines.push(`\n${set}: ${rs.length} images, ${rs.reduce((s, r) => s + r.truth, 0)} labeled cells`);
		const turned = rs.filter((r) => r.rotation !== 0).length;
		lines.push(`  turned by orientation search: ${turned} of ${rs.length} images (these sets are upright, so each is a mistake)`);
		if (rs.every((r) => r.braille !== undefined)) {
			lines.push('  text set: run experiments/score_text.py on this results file for CER and exact match');
			continue;
		}
		if (set === 'not-braille') {
			lines.push(`  images with any detection: ${rs.filter((r) => r.predicted > 0).length} of ${rs.length}; detections: ${rs.reduce((s, r) => s + r.predicted, 0)}`);
			continue;
		}
		const det = rs.filter((r) => r.recall != null);
		if (det.length) {
			lines.push(`  detection recall   (per page, mean, 95% CI over pages): ${fmt(bootstrapMean(det.map((r) => r.recall)))}`);
			lines.push(`  detection precision (per page): ${fmt(bootstrapMean(det.filter((r) => r.precision != null).map((r) => r.precision)))}`);
		}
		const acc = rs.filter((r) => r.cellAccuracyDetected != null);
		if (acc.length) lines.push(`  per-CELL accuracy on detected cells (per page): ${fmt(bootstrapMean(acc.map((r) => r.cellAccuracyDetected)))}`);
		const cer = rs.filter((r) => r.cellErrorRate != null);
		if (cer.length) lines.push(`  cell error rate (Levenshtein over Braille, per page; lower is better): ${fmt(bootstrapMean(cer.map((r) => r.cellErrorRate)))}`);
		const sides = {};
		for (const r of rs) for (const [side, v] of Object.entries(r.classifyTruth ?? {})) {
			sides[side] ??= { n: 0, ok: 0 };
			sides[side].n += v.n;
			sides[side].ok += v.ok;
		}
		for (const [side, v] of Object.entries(sides)) lines.push(`  classifier on ground-truth boxes, ${side}: ${(100 * v.ok / v.n).toFixed(1)}% per CELL (${v.ok}/${v.n}, pooled)`);
		const secs = rs.map((r) => r.seconds);
		lines.push(`  time per image, Node: median ${secs.sort((a, b) => a - b)[secs.length >> 1].toFixed(1)} s`);
	}
	return lines.join('\n');
}

// Orientation check: photos with a known upright turn (braille-research/eval/orientation.json).
async function checkOrientation(path) {
	const { items } = JSON.parse(readFileSync(path, 'utf8'));
	const base = resolve(dirname(path), '..');
	const models = await loadModels();
	let right = 0;
	for (const it of items) {
		const image = await loadRgb(join(base, it.image));
		const page = await readPage(models.detector, models.classifier, image, { confThreshold: 0.15 });
		const ok = page.rotation === it.expected_rotation;
		right += ok;
		console.log(`${it.id}: expected ${it.expected_rotation}, chose ${page.rotation}${ok ? '' : '   <-- WRONG'}`);
	}
	console.log(`orientation correct: ${right} of ${items.length} images`);
}

const a = args();
if (a.summarize) {
	console.log(summarize(JSON.parse(readFileSync(a.summarize, 'utf8')).rows));
	process.exit(0);
}
const repos = resolve(here, '..', '..', '..');
const pick = (explicit, name) => explicit ?? [join(repos, 'webeditor/static/models', name), join(repos, 'dataset/models', name)].find(existsSync);
const detectorPath = pick(a.detector, 'cell_detector.onnx');
const classifierPath = pick(a.classifier, 'cell_classifier.onnx');
async function loadModels(dataRoot) {
	if (!detectorPath || !classifierPath) throw new Error('no ONNX models found; pass --detector and --classifier');
	return { detector: await CellDetector.load(detectorPath), classifier: await CellClassifier.load(classifierPath), dataRoot };
}
if (a.orientation) {
	await checkOrientation(a.orientation);
	process.exit(0);
}
if (!a.cells || !a.out) {
	console.error('usage: node eval/run.mjs --cells FILE --out FILE [--sets a,b] [--limit N] [--rotation auto|0]\n       node eval/run.mjs --summarize FILE');
	process.exit(2);
}

const manifest = JSON.parse(readFileSync(a.cells, 'utf8'));
const dataRoot = a.data ? resolve(a.data) : join(repos, manifest.data_root);
const wanted = a.sets ? a.sets.split(',') : null;
const rotationOpt = a.rotation === '0' ? 0 : 'auto';
const { items: chosen, problem } = selectItems(manifest.items, {
	sets: wanted,
	role: a.role,
	tuning: Boolean(a.iou),
	final: Boolean(a.final)
});
if (problem) {
	console.error(problem);
	process.exit(2);
}
let items = chosen;
if (a.limit) {
	const per = {};
	items = items.filter((i) => (per[i.set] = (per[i.set] ?? 0) + 1) <= Number(a.limit));
}

const models = await loadModels(dataRoot);
const rows = [];
// Model paths are recorded relative to the folder holding the repositories, so a results file
// that gets committed doesn't carry anyone's home directory.
function results() {
	return JSON.stringify({ detectorPath: relative(repos, detectorPath), classifierPath: relative(repos, classifierPath), rotation: rotationOpt, rows });
}
for (const [k, item] of items.entries()) {
	const image = await loadRgb(join(item.root ? join(repos, item.root) : dataRoot, item.image));
	if (image.width !== item.width || image.height !== item.height) throw new Error(`${item.image}: size ${image.width}x${image.height} does not match the manifest ${item.width}x${item.height}`);
	rows.push(await scoreItem(item, image, models, rotationOpt, a.iou ? { iouThreshold: Number(a.iou) } : {}));
	if ((k + 1) % 10 === 0 || k + 1 === items.length) {
		console.error(`${k + 1}/${items.length}`);
		writeFileSync(a.out, results());
	}
}
writeFileSync(a.out, results());
console.log(summarize(rows));
