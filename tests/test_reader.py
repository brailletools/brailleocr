"""reader.py: the pieces that are pure logic (no model weights needed)."""
import PIL.Image
import pytest

import reader
from dot_pattern_utils import TARGET_CELL_PX, TILE_SIZE


def cell(cx, cy, w=20, h=30, conf=0.9):
    return {'cx': cx, 'cy': cy, 'w': w, 'h': h, 'conf': conf}


def test_bits_to_braille_uses_dot_1_as_the_lowest_bit():
    assert reader.bits_to_braille('100000') == '⠁'
    assert reader.bits_to_braille('000001') == '⠠'
    assert reader.bits_to_braille('111111') == '⠿'
    assert reader.bits_to_braille('000000') == '⠀'


def test_dedupe_keeps_the_more_confident_of_two_overlapping_detections():
    a, b, far = cell(100, 100, conf=0.4), cell(102, 101, conf=0.8), cell(300, 100)
    kept = reader._dedupe_cells([a, b, far])
    assert b in kept and far in kept and a not in kept


class OneClassModel:
    names = {0: 'cell'}


def test_a_single_class_detector_gives_a_blank_placeholder_pattern():
    assert reader.yolo_class_to_bits6(OneClassModel(), 0) == '000000'


def test_too_few_detections_means_no_tile_size_can_be_chosen(monkeypatch):
    monkeypatch.setattr(reader, 'run_detection_tiled', lambda *a: [cell(1, 1), cell(2, 2)])
    assert reader.detect_scale_normalized(PIL.Image.new('RGB', (10, 10)), None) == []


@pytest.mark.parametrize('cell_w, retiles', [(TARGET_CELL_PX, False),          # already at target density
                                             (TARGET_CELL_PX * 1.2, False),    # within 50%
                                             (TARGET_CELL_PX * 3, True),       # cells 3x too big
                                             (TARGET_CELL_PX * 0.4, True)])    # cells far too small
def test_tiles_are_refined_only_when_the_cell_size_is_far_from_target(monkeypatch, cell_w, retiles):
    sizes = []

    def fake(img, model, max_det, native):
        sizes.append(native)
        return [cell(i * 30, 10, w=cell_w) for i in range(10)]

    monkeypatch.setattr(reader, 'run_detection_tiled', fake)
    reader.detect_scale_normalized(PIL.Image.new('RGB', (10, 10)), None)
    assert (len(sizes) == 2) == retiles
    if retiles:
        assert sizes[1] == round(cell_w * TILE_SIZE / TARGET_CELL_PX)
