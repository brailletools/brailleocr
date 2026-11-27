#!/usr/bin/env python3
"""
Build the evaluation manifest: which images are in which held-out set, with the
ground-truth cells for each, as pixel boxes in the orientation the image is read in.

Two outputs:

  eval/splits.json    small, committed. Set name -> [{id, image, sha256, ...}]. Fixes WHICH
                      images are in each set, so a result can be tied to exact files.
  --cells-out FILE    large, not committed. The same items plus every ground-truth cell,
                      read by js/eval/run.mjs.

The sets follow each dataset's own designation, not this repo's training loaders (the
Angelina loader re-splits the authors' test images into train/val/test -- AUDIT B1):

  angelina-uploaded   all 50 photos the authors designate as test (uploaded/test.txt)
  angelina-books-val  the authors' validation books (front side only is labeled)
  angelina-hand-val   the authors' validation handwritten pages
  dsbi-test           the 88 test pages. Each page has a recto and a verso image. On 90 of
                      114 pages they are the same file, so one image carries both sides'
                      labels (a 'union' item). On the rest each side is deskewed
                      separately; each image then has only half its dots labeled, so
                      page-level detection scores are invalid there. Those sides are kept
                      as 'side' items with detection=false, usable for classifying
                      ground-truth boxes only. A page with only one labeled side
                      ('single': the other annotation file is empty, which DSBI uses for
                      "no dots on that side") is fully labeled and valid for detection.
  braille-natural-test  48 signage photos, boxes only (no dot labels)
  not-braille         44 images with no Braille; any detection is a false alarm
  braillebench        259 real photos with English transcriptions (external; see
                      braillebench_items). Scored as text, not as cells.

Usage:
  python experiments/export_eval_manifest.py --cells-out /tmp/eval-cells.json
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import PIL.Image
import PIL.ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import angelina_data                                      # noqa: E402
import braille_natural_data                               # noqa: E402
import dsbi_data                                          # noqa: E402
from dot_pattern_utils import BOX_SHAPES, DEFAULT_BOX_SHAPE, REPOS_ROOT, canonical_boxes   # noqa: E402

BOX_SHAPE = DEFAULT_BOX_SHAPE   # set from --box-shape in main()

DATA_ROOT = REPOS_ROOT / 'dataset' / 'data'
SPLITS_OUT = Path(__file__).resolve().parent.parent / 'eval' / 'splits.json'


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path):
    return str(Path(path).resolve().relative_to(DATA_ROOT.resolve()))


def oriented_size(path):
    with PIL.Image.open(path) as im:
        return PIL.ImageOps.exif_transpose(im).size


def to_pixels(cells, width, height, keep_side=False):
    out = []
    for c in cells:
        left, top, right, bottom = c['frac_bbox']
        cell = {'cx': (left + right) / 2 * width, 'cy': (top + bottom) / 2 * height,
                'w': (right - left) * width, 'h': (bottom - top) * height}
        if c.get('bits6'):
            cell['bits'] = c['bits6']
        if keep_side and c.get('side'):
            cell['side'] = c['side']
        out.append(cell)
    return out


def item(set_name, image, cells, **extra):
    width, height = oriented_size(image)
    cells = canonical_boxes(cells, width, height, BOX_SHAPE)   # same box shape the training data uses
    return {'set': set_name, 'id': f'{set_name}:{rel(image)}', 'image': rel(image),
            'sha256': sha256(image), 'width': width, 'height': height,
            'cells': to_pixels(cells, width, height), **extra}


def listed(list_file):
    root = list_file.parent
    return [root / line.strip().replace('\\', '/')
            for line in list_file.read_text().splitlines() if line.strip()]


def angelina_items():
    base = angelina_data.ANGELINA
    for name, list_file in (('angelina-uploaded', base / 'uploaded' / 'test.txt'),
                            ('angelina-books-val', base / 'books' / 'val.txt'),
                            ('angelina-hand-val', base / 'handwritten' / 'val.txt')):
        paths = listed(list_file)
        if not paths:
            raise SystemExit(f'{list_file} lists no images')  # AUDIT B7: never silently empty
        for jpg in paths:
            csv_path = jpg.with_suffix('.csv')
            if not csv_path.exists():
                raise SystemExit(f'missing labels for {jpg}')
            yield item(name, jpg, angelina_data.load_csv(csv_path))


def dsbi_items():
    """One item per fully labeled image, from dsbi_data.collect_pages() -- the same logic
    that decides which DSBI pages may train the detector (AUDIT B3). Test pages form
    'dsbi-test'; the pages held out from training form 'dsbi-val'."""
    for page in dsbi_data.collect_pages():
        if page['split'] not in ('test', 'val'):
            continue
        set_name = f"dsbi-{page['split']}"
        if page['kind'] == 'side':
            members = [(side, jpg, [txt]) for side, (jpg, txt) in page['sides'].items()]
        else:
            jpg = next(iter(page['sides'].values()))[0]
            members = [('both', jpg, [txt for _, txt in page['sides'].values()])]
        for side, jpg, txts in members:
            width, height = oriented_size(jpg)
            cells = []
            for txt in txts:
                tag = 'verso' if '+verso' in txt.name else 'recto'
                cells += [{**c, 'side': tag} for c in dsbi_data.load_txt(txt, width, height)]
            cells = canonical_boxes(cells, width, height, BOX_SHAPE)
            yield {'set': set_name, 'id': f'{set_name}:{rel(jpg)}', 'image': rel(jpg),
                   'sha256': sha256(jpg), 'width': width, 'height': height,
                   'cells': to_pixels(cells, width, height, keep_side=True),
                   'kind': page['kind'], 'detection': page['kind'] != 'side'}


def natural_items():
    for jpg, xml, split in braille_natural_data.collect_images():
        if split == 'test':
            yield item('braille-natural-test', jpg, braille_natural_data.load_voc_xml(xml),
                       labels='boxes-only')


def negative_items():
    for jpg, _ in angelina_data.collect_negatives():
        yield item('not-braille', jpg, [])


EXTERNAL = REPOS_ROOT / 'external' / 'BrailleBench' / 'real-braille-images'


def braillebench_items():
    """BrailleBench (Jangra et al., arXiv 2610.36195; github.com/Kartik0806/BrailleBench):
    259 real photos with manually verified English transcriptions, 215 Grade 1 and 44
    Grade 2. Images are small crops of a line or a few words, not whole pages. Downloaded
    separately into external/BrailleBench/ (the repo declares no licence, so it is not
    redistributed here); this set is skipped when that folder is absent.

    Items carry `text` (the reference) and `grade`; scored by experiments/score_text.py."""
    labels = EXTERNAL / 'benchmark' / 'labels.jsonl'
    if not labels.exists():
        print(f'  braillebench: {labels} not found -- skipping')
        return
    for line in labels.read_text().splitlines():
        row = json.loads(line)
        jpg = EXTERNAL / 'benchmark' / row['image_path']
        width, height = oriented_size(jpg)
        yield {'set': 'braillebench', 'id': f"braillebench:{row['image_path']}",
               'root': str(EXTERNAL.relative_to(REPOS_ROOT) / 'benchmark'),
               'image': row['image_path'], 'sha256': sha256(jpg), 'width': width,
               'height': height, 'cells': [], 'text': row['ground_truth'],
               'grade': row['grade'], 'kind': 'text', 'detection': False}


def braillebook_items():
    """The 142 'book-dataset' photos in the BrailleBench repository, kept as a held-out set:
    they had not been run through the reader when this set was defined (2026-11-27), and
    js/eval/run.mjs leaves held-out sets alone unless --final is given."""
    labels = EXTERNAL / 'book-dataset' / 'labels.json'
    if not labels.exists():
        print(f'  braillebook: {labels} not found -- skipping')
        return
    for row in json.loads(labels.read_text()):
        image = EXTERNAL / 'book-dataset' / row['image_path']
        width, height = oriented_size(image)
        yield {'set': 'braillebook', 'id': f"braillebook:{row['image_path']}",
               'root': str(EXTERNAL.relative_to(REPOS_ROOT) / 'book-dataset'),
               'image': row['image_path'], 'sha256': sha256(image), 'width': width,
               'height': height, 'cells': [], 'text': row['ground_truth'],
               'grade': row.get('grade'), 'kind': 'text', 'detection': False}


# What each set may be used for (AUDIT section I):
#   dev       settings and models are chosen on these
#   test      report only; never choose anything by these scores (none at present: every
#             would-be test set has been looked at, so it is 'seen')
#   seen      meant as a test, but results were looked at while changing the reader
#   held-out  not yet run; scored once, at the end
ROLES = {'angelina-books-val': 'dev', 'angelina-hand-val': 'dev', 'dsbi-val': 'dev',
         'angelina-uploaded': 'seen', 'dsbi-test': 'seen', 'braille-natural-test': 'seen',
         'not-braille': 'seen', 'braillebench': 'seen', 'braillebook': 'held-out'}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cells-out', required=True, help='large manifest with ground-truth cells')
    ap.add_argument('--splits-out', default=str(SPLITS_OUT))
    ap.add_argument('--box-shape', default=DEFAULT_BOX_SHAPE, choices=[*BOX_SHAPES, 'none'],
                    help='shape of the ground-truth boxes; match the models being scored')
    args = ap.parse_args()
    global BOX_SHAPE
    BOX_SHAPE = None if args.box_shape == 'none' else args.box_shape

    items = [*angelina_items(), *dsbi_items(), *natural_items(), *negative_items(),
             *braillebench_items(), *braillebook_items()]
    for it in items:
        it['role'] = ROLES[it['set']]      # a set without a role is an error, on purpose
    sets = {}
    for it in items:
        sets.setdefault(it['set'], []).append(it)

    # The committed split list names files and hashes only: no ground truth, and in particular
    # no transcriptions from datasets we are not licensed to redistribute (AUDIT D4).
    splits = {name: [{k: v for k, v in it.items() if k not in ('cells', 'text')} | {'n_cells': len(it['cells'])}
                     for it in members] for name, members in sets.items()}
    Path(args.splits_out).write_text(json.dumps(
        {'data_root': 'dataset/data', 'sets': splits}, indent=1) + '\n')
    Path(args.cells_out).write_text(json.dumps({'data_root': 'dataset/data', 'items': items}))

    print(f'{"set":<22}{"items":>7}{"cells":>9}')
    for name, members in sets.items():
        print(f'{name:<22}{len(members):>7}{sum(len(m["cells"]) for m in members):>9}')
    print(f'wrote {args.splits_out} and {args.cells_out}')


if __name__ == '__main__':
    main()
