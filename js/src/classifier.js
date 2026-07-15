import * as ort from 'onnxruntime-web';
import { bilinearResize } from './imageOps.js';

const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD = [0.229, 0.224, 0.225];
const CROP_PAD_FRAC = 0.1; // matches pipeline.py's _CLASSIFIER_CROP_PAD

/**
 * Wraps the exported cell_classifier.onnx (MobileNetV2, 6 independent
 * sigmoid outputs — one per Braille dot position), matching pipeline.py's
 * reclassify_cells().
 */
export class CellClassifier {
	constructor(session) {
		this.session = session;
	}

	static async load(modelUrl) {
		const session = await ort.InferenceSession.create(modelUrl);
		return new CellClassifier(session);
	}

	/**
	 * @param {Float32Array} rgbHwc - full image, HWC RGB, values 0-255
	 * @param {number} imgW
	 * @param {number} imgH
	 * @param {Array<{cx: number, cy: number, w: number, h: number}>} boxes
	 * @returns {Promise<string[]>} one 6-char '0'/'1' bit string per box
	 */
	async classify(rgbHwc, imgW, imgH, boxes) {
		if (boxes.length === 0) return [];

		const batch = new Float32Array(boxes.length * 3 * 64 * 64);
		for (let i = 0; i < boxes.length; i++) {
			const crop = this._cropAndResize(rgbHwc, imgW, imgH, boxes[i]);
			batch.set(crop, i * 3 * 64 * 64);
		}

		const input = new ort.Tensor('float32', batch, [boxes.length, 3, 64, 64]);
		const outputs = await this.session.run({ [this.session.inputNames[0]]: input });
		const logits = outputs[this.session.outputNames[0]].data;

		const results = [];
		for (let i = 0; i < boxes.length; i++) {
			let bits = '';
			for (let d = 0; d < 6; d++) {
				bits += logits[i * 6 + d] > 0 ? '1' : '0'; // logit>0 === sigmoid>0.5
			}
			results.push(bits);
		}
		return results;
	}

	/** Crop with 10% padding (pipeline.py parity), resize to 64x64, normalize, CHW. */
	_cropAndResize(rgbHwc, imgW, imgH, box) {
		const padX = box.w * CROP_PAD_FRAC;
		const padY = box.h * CROP_PAD_FRAC;
		// x0/y0 are clamped to the last valid pixel index (not just >= 0), and
		// x1/y1 are derived to always be strictly greater than x0/y0 -- a box
		// extending to or past the image edge would otherwise let x0 === imgW
		// (or y0 === imgH), one past the last valid column/row, which made the
		// row-copy loop below read past the intended data and silently paste in
		// a shorter/zero-filled subarray instead of real pixel data.
		const x0 = Math.min(imgW - 1, Math.max(0, Math.round(box.cx - box.w / 2 - padX)));
		const y0 = Math.min(imgH - 1, Math.max(0, Math.round(box.cy - box.h / 2 - padY)));
		const x1 = Math.max(x0 + 1, Math.min(imgW, Math.round(box.cx + box.w / 2 + padX)));
		const y1 = Math.max(y0 + 1, Math.min(imgH, Math.round(box.cy + box.h / 2 + padY)));
		const cropW = x1 - x0;
		const cropH = y1 - y0;

		const cropped = new Float32Array(cropW * cropH * 3);
		for (let y = 0; y < cropH; y++) {
			const srcRow = (y0 + y) * imgW * 3 + x0 * 3;
			const dstRow = y * cropW * 3;
			cropped.set(rgbHwc.subarray(srcRow, srcRow + cropW * 3), dstRow);
		}

		const resized = bilinearResize(cropped, cropW, cropH, 64, 64, 3);

		const chw = new Float32Array(3 * 64 * 64);
		const plane = 64 * 64;
		for (let i = 0; i < plane; i++) {
			chw[i] = (resized[i * 3] / 255 - IMAGENET_MEAN[0]) / IMAGENET_STD[0];
			chw[plane + i] = (resized[i * 3 + 1] / 255 - IMAGENET_MEAN[1]) / IMAGENET_STD[1];
			chw[2 * plane + i] = (resized[i * 3 + 2] / 255 - IMAGENET_MEAN[2]) / IMAGENET_STD[2];
		}
		return chw;
	}
}
