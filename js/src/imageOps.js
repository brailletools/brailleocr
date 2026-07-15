// Pure-JS, platform-agnostic image ops (no Canvas/DOM, no Node-only libs) so
// the exact same code path runs in the browser and in Node test harnesses.
// Deliberately reimplements cv2.resize(INTER_LINEAR)'s half-pixel-center
// bilinear algorithm rather than relying on Canvas's drawImage scaling —
// browsers' built-in image smoothing does not match cv2's resampling closely
// enough to reproduce ultralytics' letterbox output, which the detector was
// trained/tuned against.

/**
 * Bilinear resize matching cv2.resize(INTER_LINEAR)'s half-pixel-center
 * sampling convention: output pixel x maps to input coordinate
 * (x + 0.5) * scale - 0.5, clamped to the source bounds.
 * @param {Float32Array} src - HWC, values 0-255
 */
export function bilinearResize(src, srcW, srcH, dstW, dstH, channels) {
	const dst = new Float32Array(dstW * dstH * channels);
	const scaleX = srcW / dstW;
	const scaleY = srcH / dstH;

	for (let dy = 0; dy < dstH; dy++) {
		const sy = Math.min(Math.max((dy + 0.5) * scaleY - 0.5, 0), srcH - 1);
		const y0 = Math.floor(sy);
		const y1 = Math.min(y0 + 1, srcH - 1);
		const wy = sy - y0;

		for (let dx = 0; dx < dstW; dx++) {
			const sx = Math.min(Math.max((dx + 0.5) * scaleX - 0.5, 0), srcW - 1);
			const x0 = Math.floor(sx);
			const x1 = Math.min(x0 + 1, srcW - 1);
			const wx = sx - x0;

			const rowY0 = y0 * srcW;
			const rowY1 = y1 * srcW;
			const dstBase = (dy * dstW + dx) * channels;

			for (let c = 0; c < channels; c++) {
				const v00 = src[(rowY0 + x0) * channels + c];
				const v01 = src[(rowY0 + x1) * channels + c];
				const v10 = src[(rowY1 + x0) * channels + c];
				const v11 = src[(rowY1 + x1) * channels + c];
				const top = v00 + (v01 - v00) * wx;
				const bot = v10 + (v11 - v10) * wx;
				dst[dstBase + c] = top + (bot - top) * wy;
			}
		}
	}
	return dst;
}

/**
 * Replicates ultralytics' LetterBox(new_shape, auto=True, stride=32,
 * scaleup=True): resize preserving aspect ratio so the image fits within
 * new_shape, then pad only up to the nearest stride multiple (not all the
 * way to a fixed square) with grey (114,114,114) — see brailleocr's
 * export_onnx.py / scoping notes for why this matters: a fixed-shape
 * letterbox silently changes detection counts relative to the .pt model.
 *
 * @param {Float32Array} src - HWC RGB, values 0-255
 * @returns {{data: Float32Array, w: number, h: number, padLeft: number, padTop: number, scale: number}}
 */
export function letterbox(src, srcW, srcH, newW = 640, newH = 640, stride = 32, padValue = 114) {
	const r = Math.min(newH / srcH, newW / srcW);
	const unpadW = Math.round(srcW * r);
	const unpadH = Math.round(srcH * r);

	let dw = newW - unpadW;
	let dh = newH - unpadH;
	dw = dw % stride;
	dh = dh % stride;
	dw /= 2;
	dh /= 2;

	const resized = bilinearResize(src, srcW, srcH, unpadW, unpadH, 3);

	const outW = unpadW + Math.round(dw - 0.1) + Math.round(dw + 0.1);
	const outH = unpadH + Math.round(dh - 0.1) + Math.round(dh + 0.1);
	const padLeft = Math.round(dw - 0.1);
	const padTop = Math.round(dh - 0.1);

	const out = new Float32Array(outW * outH * 3).fill(padValue);
	for (let y = 0; y < unpadH; y++) {
		const srcRow = y * unpadW * 3;
		const dstRow = (y + padTop) * outW * 3 + padLeft * 3;
		out.set(resized.subarray(srcRow, srcRow + unpadW * 3), dstRow);
	}

	return { data: out, w: outW, h: outH, padLeft, padTop, scale: r };
}

/**
 * Raw HWC sub-region copy (no resize/normalize) -- used by tiling.js to crop
 * a tile out of a full page before running the detector on it. Bounds are
 * clamped the same way classifier.js's _cropAndResize() clamps its padded
 * crop: x0/y0 to the last valid pixel index (not just >= 0), x1/y1 derived
 * to always be strictly greater, so a caller-supplied box that runs to or
 * past the image edge can never produce an out-of-bounds read.
 * @param {Float32Array} rgbHwc - HWC RGB, values 0-255
 * @returns {{data: Float32Array, w: number, h: number}}
 */
export function cropRgbHwc(rgbHwc, imgW, imgH, x0, y0, x1, y1) {
	const cx0 = Math.min(imgW - 1, Math.max(0, Math.round(x0)));
	const cy0 = Math.min(imgH - 1, Math.max(0, Math.round(y0)));
	const cx1 = Math.max(cx0 + 1, Math.min(imgW, Math.round(x1)));
	const cy1 = Math.max(cy0 + 1, Math.min(imgH, Math.round(y1)));
	const w = cx1 - cx0;
	const h = cy1 - cy0;

	const out = new Float32Array(w * h * 3);
	for (let y = 0; y < h; y++) {
		const srcRow = (cy0 + y) * imgW * 3 + cx0 * 3;
		const dstRow = y * w * 3;
		out.set(rgbHwc.subarray(srcRow, srcRow + w * 3), dstRow);
	}
	return { data: out, w, h };
}

/** HWC Float32Array (0-255) -> CHW Float32Array (0-1), NCHW batch of 1. */
export function hwcToChwNormalized(hwc, w, h) {
	const chw = new Float32Array(3 * w * h);
	const plane = w * h;
	for (let i = 0; i < plane; i++) {
		chw[i] = hwc[i * 3] / 255;
		chw[plane + i] = hwc[i * 3 + 1] / 255;
		chw[2 * plane + i] = hwc[i * 3 + 2] / 255;
	}
	return chw;
}
