#!/usr/bin/env python3
"""
Export the detector and classifier in dataset/models/ to ONNX, for the
client-side (onnxruntime-web) OCR path in webeditor.

Usage:
  pixi run -e export python export_onnx.py

Requires the `export` pixi environment (onnx + onnxruntime), separate from
the default environment so these aren't a runtime dependency of pipeline.py.
"""
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO

from dot_pattern_utils import REPOS_ROOT

MODELS_DIR = REPOS_ROOT / 'dataset' / 'models'


def export_detector():
    """
    dynamic=True is required, not optional: a fixed-shape export changes
    ultralytics' letterbox/aspect-ratio handling relative to the .pt model's
    dynamic-shape default, which silently changes detection counts (verified:
    38 vs 50 boxes on the same image at the same conf threshold, collapsing
    to exact parity once re-exported with dynamic=True). See scoping notes.
    """
    src = MODELS_DIR / 'cell_detector.pt'
    model = YOLO(str(src))
    out_path = model.export(format='onnx', simplify=True, imgsz=640, dynamic=True)
    print(f'Detector exported: {out_path}')


def export_classifier():
    """
    Rebuilds the exact architecture from pipeline.py's load_cell_classifier()
    — the .pt file is a bare state_dict, not a scriptable checkpoint, so the
    module structure has to match exactly or load_state_dict silently fails
    to load some/all layers depending on strictness.
    """
    src = MODELS_DIR / 'cell_classifier.pt'
    dst = MODELS_DIR / 'cell_classifier.onnx'

    net = models.mobilenet_v2(weights=None)
    net.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(net.last_channel, 6))
    net.load_state_dict(torch.load(src, map_location='cpu'))
    net.eval()

    dummy = torch.randn(1, 3, 64, 64)
    # dynamo=False: the new torch.export-based exporter (default since torch
    # 2.9) requires the `onnxscript` package, which isn't in this repo's
    # deps. The legacy TorchScript-based exporter needs no extra dependency
    # and produces a numerically-verified-identical graph (see scoping notes:
    # max abs logit diff 5.7e-6, 100% bit-level agreement after sigmoid>0.5).
    torch.onnx.export(
        net, dummy, str(dst),
        export_params=True, opset_version=17,
        input_names=['input'], output_names=['logits'],
        dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}},
        dynamo=False,
    )
    print(f'Classifier exported: {dst}')


if __name__ == '__main__':
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    export_detector()
    export_classifier()
