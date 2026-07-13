from pathlib import Path
import PIL.Image, PIL.ImageOps
from ultralytics import YOLO
import pipeline

model = YOLO('/tmp/braille-yolo-detector/tiled-full/weights/best.pt')
sample_dir = Path('../dataset/data/sample-images')
images = sorted(sample_dir.glob('*.jpeg'))

out_dir = Path('/tmp/sample_translations')
out_dir.mkdir(exist_ok=True)

for img_path in images:
    print(f"\n=== {img_path.name} ===")
    img = PIL.ImageOps.exif_transpose(PIL.Image.open(img_path)).convert('RGB')
    text = pipeline.process_container(
        img, img_path.stem, model, 'en-ueb-g2.ctb', True,
        spellcheck=True, classifier_path='/tmp/braille-crops/cell_classifier.pt',
        normalize_scale=True
    )
    if text:
        (out_dir / f'{img_path.stem}.txt').write_text(text)
        print(text[:500])
    else:
        print("  (no text detected)")

print("\nDone. Annotated images in:", pipeline.OUT_DIR)
print("Text output in:", out_dir)
