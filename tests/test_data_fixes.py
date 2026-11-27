"""
Tests for the training-data fixes in AUDIT.md B1-B4 and B7. Synthetic inputs where
possible; the ones that read the real datasets skip when the `dataset` checkout is absent
(as in CI).
"""
import random

import PIL.Image
import pytest

import angelina_data
import dsbi_data
import prepare_yolo_dataset as pyd
from dot_pattern_utils import box_shape_over_pitch, canonical_boxes

W, H = 1000, 1000
PITCH = 50          # px between neighbouring cells in a row


def grid(w_over_pitch, h_over_pitch, rows=6, cols=12):
    cells = []
    for r in range(rows):
        for c in range(cols):
            cx, cy = 100 + c * PITCH, 100 + r * 90
            w, h = w_over_pitch * PITCH, h_over_pitch * PITCH
            cells.append({'frac_bbox': ((cx - w / 2) / W, (cy - h / 2) / H,
                                        (cx + w / 2) / W, (cy + h / 2) / H), 'label': 1})
    return cells


def test_narrow_boxes_are_rescaled_to_the_canonical_shape_about_their_centres():
    narrow = grid(0.54, 1.08)
    out = canonical_boxes(narrow, W, H)
    w_over, h_over = box_shape_over_pitch(out, W, H)
    assert w_over == pytest.approx(0.75, abs=0.01)
    assert h_over == pytest.approx(1.16, abs=0.01)
    for a, b in zip(narrow, out):
        assert (a['frac_bbox'][0] + a['frac_bbox'][2]) == pytest.approx(b['frac_bbox'][0] + b['frac_bbox'][2])
        assert (a['frac_bbox'][1] + a['frac_bbox'][3]) == pytest.approx(b['frac_bbox'][1] + b['frac_bbox'][3])


def test_wide_boxes_and_sparse_pages_are_left_alone():
    wide = grid(0.75, 1.16)
    assert canonical_boxes(wide, W, H) is wide
    sparse = grid(0.54, 1.08, rows=1, cols=5)          # too few cells to measure a pitch
    assert canonical_boxes(sparse, W, H) is sparse


def test_word_gaps_do_not_change_the_measured_pitch():
    cells = [c for i, c in enumerate(grid(0.54, 1.08)) if i % 12 not in (4, 8)]   # two word gaps per row
    assert box_shape_over_pitch(cells, W, H)[0] == pytest.approx(0.54, abs=0.01)


def test_cell_free_tiles_are_kept_as_background_only_when_they_touch_no_cell():
    image = PIL.Image.new('RGB', (2000, 1000), 'white')
    # one block of cells in the top-left corner; most tiles are far from it
    cells = [c for c in grid(0.75, 1.16, rows=2, cols=4)]
    for c in cells:   # fractions of the 2000x1000 image
        l, t, r, b = c['frac_bbox']
        c['frac_bbox'] = (l * W / 2000, t, r * W / 2000, b)
    none = pyd.tile_image(image, cells, 2000, 1000, empty_frac=0.0)
    kept = pyd.tile_image(image, cells, 2000, 1000, empty_frac=1.0, rng=random.Random(0))
    assert all(tc for _, tc in none), 'by default cell-free tiles are dropped, as before'
    empties = [tile_cells for _, tile_cells in kept if not tile_cells]
    assert empties, 'with empty_frac=1 the cell-free tiles are kept'
    assert len(kept) > len(none)


def test_negative_tiles_have_the_requested_scales_and_are_reproducible():
    image = PIL.Image.new('RGB', (1600, 1200), 'white')
    a = pyd.negative_tiles(image, random.Random(1))
    b = pyd.negative_tiles(image, random.Random(1))
    assert [t.size for t in a] == [t.size for t in b]
    assert {t.size[0] for t in a} == {640, 1200}      # 1280 is capped at the image's short side


def test_empty_label_file_is_written_for_background(tmp_path):
    out = tmp_path / 'tile.txt'
    pyd.write_label_file([], out, single_class=True)
    assert out.read_text() == ''


needs_dataset = pytest.mark.skipif(not angelina_data.ANGELINA.exists() or not dsbi_data.DSBI.exists(),
                                   reason='requires a sibling `dataset` checkout with the Angelina and DSBI submodules')


@needs_dataset
def test_angelina_uses_the_authors_splits_and_all_50_user_uploads_are_test():
    entries = angelina_data.collect_images()
    by_split = {s: [e for e in entries if e[2] == s] for s in ('train', 'val', 'test')}
    assert len(by_split['test']) == 50
    assert all('test2' in str(jp) for jp, _, _ in by_split['test'])
    assert not any('test2' in str(jp) for jp, _, _ in by_split['train'] + by_split['val'])


@needs_dataset
def test_dsbi_pages_are_classified_by_whether_recto_and_verso_are_one_scan():
    pages = dsbi_data.collect_pages()
    assert len({(p['split'], p['page']) for p in pages}) == len(pages) == 114
    kinds = {p['kind'] for p in pages}
    assert kinds == {'union', 'single', 'side'}
    for p in pages:
        if p['kind'] == 'union':
            (rj, _), (vj, _) = p['sides']['recto'], p['sides']['verso']
            assert rj.read_bytes() == vj.read_bytes()
    with pytest.raises(ValueError):
        dsbi_data.load_page_cells(next(p for p in pages if p['kind'] == 'side'), 100, 100)
    assert all(p['kind'] != 'side' for p in dsbi_data.detection_pages())


@needs_dataset
def test_an_empty_or_missing_split_list_is_an_error_not_an_empty_source(tmp_path):
    (tmp_path / 'train.txt').write_text('')
    with pytest.raises(ValueError):
        angelina_data._listed(tmp_path / 'train.txt')
    with pytest.raises(FileNotFoundError):
        angelina_data._listed(tmp_path / 'absent.txt')


def test_both_shapes_can_be_targeted_and_none_leaves_labels_alone():
    for shape, (tw, th) in (('wide', (0.75, 1.16)), ('narrow', (0.54, 1.08))):
        for start in (grid(0.54, 1.08), grid(0.75, 1.16)):
            out = canonical_boxes(start, W, H, shape)
            w_over, h_over = box_shape_over_pitch(out, W, H)
            assert w_over == pytest.approx(tw, abs=0.02)
            assert h_over == pytest.approx(th, abs=0.02)
    original = grid(0.54, 1.08)
    assert canonical_boxes(original, W, H, None) is original
    with pytest.raises(KeyError):
        canonical_boxes(original, W, H, 'square')


@needs_dataset
def test_dsbi_validation_pages_come_from_training_pages_and_are_fully_labeled():
    names = dsbi_data.val_page_names()
    assert len(names) == dsbi_data.VAL_PAGES
    assert names <= set(dsbi_data._page_names('train.txt'))
    assert not names & set(dsbi_data._page_names('test.txt'))
    assert dsbi_data.val_page_names() == names, 'the same pages every time'
    val = [p for p in dsbi_data.collect_pages() if p['split'] == 'val']
    assert len(val) == dsbi_data.VAL_PAGES
    assert all(p['kind'] in ('union', 'single') for p in val)
    # the detector and the classifier must hold out the same pages
    import extract_crops
    crop_val = {jpg.name.rsplit('+', 1)[0] for jpg, _, split, _, _ in extract_crops.dsbi_entries() if split == 'val'}
    assert crop_val == {p['page'] for p in val}
