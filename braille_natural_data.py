"""
Loader for the braille_natural dataset — real-world "natural scene" photos
of braille (embossed signage, elevator buttons, etc.), as opposed to
Angelina/DSBI's flat page scans/photos. This is the source the original HF
snoop2head/yolov8m-braille model used that we were missing, which plausibly
explains why our own detector generalized poorly to real phone photos
despite good held-out mAP on Angelina/DSBI.

Localization-only: every box is labeled with a single generic class
("Braille" in VOC_Braille, "braille" in Org, unlabeled in ICDAR_Braille — all
three are just different annotation-format exports of the same 164 train +
48 test images, verified by identical filename sets). No dot-pattern info
is available from this source, so it can only feed the single-class
detector, never the 64-class detector or the classifier.

Source layout (dataset/data/braille_natural):
  VOC_Braille/natural_{train,test}/JPEGImages/*.jpg
  VOC_Braille/natural_{train,test}/Annotations/*.xml   — Pascal VOC format
"""

import xml.etree.ElementTree as ET

from dot_pattern_utils import REPOS_ROOT

BRAILLE_NATURAL = REPOS_ROOT / 'dataset' / 'data' / 'braille_natural'
SUPPORTS_MULTICLASS = False


def collect_images():
    """Return list of (img_path, xml_path, split) for every labeled image."""
    entries = []
    for split, subdir in (('train', 'natural_train'), ('test', 'natural_test')):
        base = BRAILLE_NATURAL / 'VOC_Braille' / subdir
        for xp in sorted((base / 'Annotations').glob('*.xml')):
            jp = base / 'JPEGImages' / f'{xp.stem}.jpg'
            if jp.exists():
                entries.append((jp, xp, split))
    return entries


def load_voc_xml(xml_path):
    """Return cells as fractional (l, t, r, b) — no label/bits6 (see module docstring)."""
    root = ET.parse(xml_path).getroot()
    size = root.find('size')
    img_w, img_h = int(size.find('width').text), int(size.find('height').text)

    cells = []
    for obj in root.findall('object'):
        box = obj.find('bndbox')
        x0, y0 = float(box.find('xmin').text), float(box.find('ymin').text)
        x1, y1 = float(box.find('xmax').text), float(box.find('ymax').text)
        cells.append({
            'frac_bbox': (x0 / img_w, y0 / img_h, x1 / img_w, y1 / img_h),
            'label': None,
            'bits6': None,
        })
    return cells
