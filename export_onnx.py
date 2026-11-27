#!/usr/bin/env python3
"""
Export the detector and classifier in dataset/models/ to ONNX, for the
client-side (onnxruntime-web) OCR path in webeditor.

Usage:
  pixi run -e export onnx

Requires the `export` pixi environment (onnx + onnxruntime), separate from
the default environment so these aren't a runtime dependency of reader.py.
"""

import argparse
import shutil
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from ultralytics import YOLO

from dot_pattern_utils import REPOS_ROOT

MODELS_DIR = REPOS_ROOT / 'dataset' / 'models'


def export_detector(src=None, out_dir=None):
    """
    dynamic=True is required, not optional: a fixed-shape export changes
    ultralytics' letterbox/aspect-ratio handling relative to the .pt model's
    dynamic-shape default, which silently changes detection counts (verified:
    38 vs 50 boxes on the same image at the same conf threshold, collapsing
    to exact parity once re-exported with dynamic=True). See scoping notes.
    """
    src = Path(src or MODELS_DIR / 'cell_detector.pt')
    model = YOLO(str(src))
    # ultralytics writes next to the .pt file; move it if another folder was asked for
    out_path = Path(model.export(format='onnx', simplify=True, imgsz=640, dynamic=True))
    if out_dir:
        out_path = Path(shutil.move(str(out_path), str(Path(out_dir) / 'cell_detector.onnx')))
    print(f'Detector exported: {out_path}')


def export_classifier(src=None, out_dir=None):
    """
    Rebuilds the exact architecture from reader.py's load_cell_classifier()
    — the .pt file is a bare state_dict, not a scriptable checkpoint, so the
    module structure has to match exactly or load_state_dict silently fails
    to load some/all layers depending on strictness.
    """
    src = Path(src or MODELS_DIR / 'cell_classifier.pt')
    dst = Path(out_dir or MODELS_DIR) / 'cell_classifier.onnx'

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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--detector', help='detector .pt (default: dataset/models/cell_detector.pt)')
    ap.add_argument('--classifier', help='classifier .pt (default: dataset/models/cell_classifier.pt)')
    ap.add_argument('--out-dir', help='where to write the .onnx files (default: dataset/models, '
                    'which promotes them -- use another folder for experiments)')
    args = ap.parse_args()
    if (args.detector or args.classifier) and not args.out_dir:
        ap.error('--out-dir is required with --detector/--classifier, so an experiment '
                 'cannot overwrite the published models')
    out_dir = Path(args.out_dir) if args.out_dir else MODELS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    export_detector(args.detector, args.out_dir)
    export_classifier(args.classifier, out_dir)


if __name__ == '__main__':
    main()
