import * as ort from 'onnxruntime-web';
import { letterbox, hwcToChwNormalized } from './imageOps.js';
import { nms } from './nms.js';

/**
 * Wraps the exported cell_detector.onnx (YOLOv8, single "cell" class,
 * exported with dynamic=True — see brailleocr/export_onnx.py for why that's
 * required for parity with the .pt model, not just a size optimization).
 */
export class CellDetector {
	constructor(session) {
		this.session = session;
	}

	static async load(modelUrl) {
		const session = await ort.InferenceSession.create(modelUrl);
		return new CellDetector(session);
	}

	/**
	 * @param {Float32Array} rgbHwc - HWC RGB pixel data, values 0-255
	 * @param {number} srcW
	 * @param {number} srcH
	 * @param {{confThreshold?: number, iouThreshold?: number, maxDet?: number}} [opts]
	 * @returns {Array<{cx: number, cy: number, w: number, h: number, conf: number}>}
	 *   Boxes in original-image pixel coordinates.
	 */
	async detect(rgbHwc, srcW, srcH, opts = {}) {
		const { confThreshold = 0.05, iouThreshold = 0.7, maxDet = 300 } = opts;

		const lb = letterbox(rgbHwc, srcW, srcH, 640, 640);
		const chw = hwcToChwNormalized(lb.data, lb.w, lb.h);

		const input = new ort.Tensor('float32', chw, [1, 3, lb.h, lb.w]);
		const outputs = await this.session.run({ [this.session.inputNames[0]]: input });
		const raw = outputs[this.session.outputNames[0]];
		const [, numAttrs, numAnchors] = raw.dims;
		const data = raw.data;

		// Raw layout: (1, 5, numAnchors) — row 0-3 = cx,cy,w,h (already decoded
		// to letterboxed-image pixel space by the export), row 4 = confidence
		// (sigmoid already applied). See scoping notes: this is the standard
		// ultralytics ONNX export layout for a single-class detector.
		const candidates = [];
		for (let i = 0; i < numAnchors; i++) {
			const conf = data[4 * numAnchors + i];
			if (conf < confThreshold) continue;
			candidates.push({
				cx: data[0 * numAnchors + i],
				cy: data[1 * numAnchors + i],
				w: data[2 * numAnchors + i],
				h: data[3 * numAnchors + i],
				conf
			});
		}
		if (numAttrs !== 5) {
			throw new Error(`Unexpected detector output shape: (_, ${numAttrs}, ${numAnchors})`);
		}

		const kept = nms(candidates, iouThreshold, maxDet);

		// Map from letterboxed 640-space back to original image coordinates.
		return kept.map((b) => ({
			cx: (b.cx - lb.padLeft) / lb.scale,
			cy: (b.cy - lb.padTop) / lb.scale,
			w: b.w / lb.scale,
			h: b.h / lb.scale,
			conf: b.conf
		}));
	}
}
