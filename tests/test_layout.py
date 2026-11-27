"""Line grouping and spacing in pipeline.py. Mirrors js/test/lineLayout.test.js."""
import math

import pytest

import pipeline


def cell(cx, cy, **extra):
    return {'cx': cx, 'cy': cy, 'w': 25, 'h': 50, 'bits': '100000', **extra}


def page(n_lines, n_cells, place=lambda x, y: (x, y)):
    cells = []
    for line in range(n_lines):
        for i in range(n_cells):
            cx, cy = place(100 + i * 35, 100 + line * 80)
            cells.append(cell(cx, cy, line=line, index=i))
    return cells


def assert_lines_intact(lines, n_lines, n_cells):
    assert len(lines) == n_lines
    for l, line in enumerate(lines):
        assert [(c['line'], c['index']) for c in line] == [(l, i) for i in range(n_cells)]


@pytest.mark.parametrize('degrees', [-10, -2, 1, 2, 5, 10])
def test_lines_survive_tilt(degrees):
    t = math.radians(degrees)
    cells = page(12, 40, lambda x, y: (x * math.cos(t) - y * math.sin(t),
                                       x * math.sin(t) + y * math.cos(t)))
    assert abs(pipeline.estimate_skew(cells) - t) < 0.005
    assert_lines_intact(pipeline.group_into_lines(cells), 12, 40)


def test_lines_survive_perspective():
    cells = page(10, 40, lambda x, y: (x, y + (460 - y) / 360 * 0.05 * (x - 800)))
    assert_lines_intact(pipeline.group_into_lines(cells), 10, 40)


def test_lines_survive_curl():
    cells = page(10, 40, lambda x, y: (x, y + 60 * math.sin(math.pi * (x - 100) / 1365)))
    assert_lines_intact(pipeline.group_into_lines(cells), 10, 40)


def test_grouping_returns_the_original_cells():
    cells = page(2, 5)
    assert all(any(c is o for o in cells) for line in pipeline.group_into_lines(cells) for c in line)


def test_blank_cells_are_counted_in_pitches_not_cell_widths():
    # Pitch 50, boxes 34 wide (the real-photo ratio): one blank cell is a 100px
    # gap, which dividing by width reported as two.
    line = [{**cell(cx, 100), 'w': 34} for cx in (0, 50, 100, 200, 250)]
    result = pipeline.insert_spaces(line, 50)
    assert sum(1 for c in result if c.get('is_space')) == 1


def test_wide_gap_gets_several_blank_cells():
    line = [cell(cx, 100) for cx in (0, 30, 60, 210, 240)]
    result = pipeline.insert_spaces(line, 30)
    assert sum(1 for c in result if c.get('is_space')) == 4
