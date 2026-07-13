"""
Fine-tune a COCO-pretrained YOLOv8 detector on the Angelina braille-cell
dataset (see prepare_yolo_dataset.py).

Base model is chosen to be small enough for a browser export (yolov8n by
default; pass --base yolov8s.pt for the mid-sized option) while the existing
pipeline.py / HF snoop2head/yolov8m-braille model stays available as the
higher-accuracy CLI option.

IMPORTANT: fliplr is forced to 0. A horizontally-flipped braille cell is a
*different*, still-valid-looking dot pattern (e.g. dot1 <-> dot4), so the
default YOLO left-right flip augmentation would teach the model wrong
associations between appearance and class.

Output: /tmp/braille-yolo-detector/train/weights/best.pt
"""

import argparse

import torch
from ultralytics import YOLO

DATASET_YAML = '/tmp/braille-yolo-dataset/dataset.yaml'
PROJECT      = '/tmp/braille-yolo-detector'

device = (
    'mps'  if torch.backends.mps.is_available() else
    'cuda' if torch.cuda.is_available() else
    'cpu'
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='yolov8n.pt', help='COCO-pretrained base checkpoint')
    ap.add_argument('--epochs', type=int, default=100)
    ap.add_argument('--imgsz', type=int, default=640)
    ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--data', default=DATASET_YAML, help='dataset.yaml path')
    ap.add_argument('--name', default='train', help='run name under ' + PROJECT)
    args = ap.parse_args()

    model = YOLO(args.base)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=PROJECT,
        name=args.name,
        fliplr=0.0,   # see module docstring — do not flip braille cells
        flipud=0.0,
        degrees=0.0,  # dot positions are orientation-sensitive
    )

    metrics = model.val(data=args.data, split='test')
    print(f"\nTest set: mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")


if __name__ == '__main__':
    main()
