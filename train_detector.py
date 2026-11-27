"""
Fine-tune a COCO-pretrained YOLOv8 detector on the Angelina braille-cell
dataset (see prepare_yolo_dataset.py).

Base model is chosen to be small enough for a browser export (yolov8n by
default; pass --base yolov8s.pt for the mid-sized option) while the existing
HF snoop2head/yolov8m-braille model remains the larger third-party model to compare
against.

IMPORTANT: fliplr is forced to 0. A horizontally-flipped braille cell is a
*different*, still-valid-looking dot pattern (e.g. dot1 <-> dot4), so the
default YOLO left-right flip augmentation would teach the model wrong
associations between appearance and class.

Output: /tmp/braille-yolo-detector/train/weights/best.pt
"""

import argparse
from pathlib import Path

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
    ap.add_argument('--name', default='train', help='run name under --project')
    ap.add_argument('--project', default=PROJECT,
                    help='output folder; the default is under /tmp, which does not survive reboots')
    ap.add_argument('--resume', action='store_true',
                    help='continue an interrupted run of this --project/--name from its last.pt')
    ap.add_argument('--fraction', type=float, default=1.0,
                    help='share of the training tiles to use (for smoke tests)')
    ap.add_argument('--eval-split', default='val', choices=['val', 'test'],
                    help="split scored after training. 'val' is development data; 'test' holds "
                         'sets that must not influence any choice (AUDIT section I)')
    args = ap.parse_args()

    run_dir = Path(args.project).resolve() / args.name
    last = run_dir / 'weights' / 'last.pt'
    finished = False
    if args.resume and last.exists():
        # ultralytics marks a completed run's checkpoint with epoch -1 and cannot resume it
        finished = torch.load(last, map_location='cpu', weights_only=False).get('epoch', -1) == -1
        model = YOLO(str(last))
        if finished:
            print(f'{run_dir} already finished training; only scoring it')
        else:
            model.train(resume=True)
    else:
        model = YOLO(args.base)
        model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=device,
            project=str(run_dir.parent),
            name=run_dir.name,
            exist_ok=True,   # so the weights are where --project/--name says, not name2, name3
            fraction=args.fraction,
            fliplr=0.0,   # see module docstring — do not flip braille cells
            flipud=0.0,
            degrees=0.0,  # dot positions are orientation-sensitive
        )

    # without project/name ultralytics writes its plots to ./runs, inside the repository
    metrics = model.val(data=args.data, split=args.eval_split, device=device,
                        project=str(run_dir.parent), name=f'{run_dir.name}-eval', exist_ok=True)
    print(f"\n{args.eval_split} split: mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
    print(f"best weights: {run_dir / 'weights' / 'best.pt'}")


if __name__ == '__main__':
    main()
