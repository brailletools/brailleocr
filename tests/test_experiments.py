"""
Tests for the labeling tools in experiments/ (the corrections -> ground-truth rule and the
page a labeler gets). Synthetic inputs and no models, so they run in CI in under a second.

History: this file once also tested the timing, ground-truth matching and edge-recall
scripts, which were written after two measurement bugs produced confidently wrong results
that ran cleanly and printed plausible numbers (RESEARCH.md records both). Those scripts
measured the recovery stages of the Python pipeline, which was retired (AUDIT.md).
"""
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'experiments'))

from merge_corrections import apply_corrections, bits_to_label   # noqa: E402


# ── merge_corrections.py: the corrections → ground truth rule ────────────────

def doc(predictions, **corrections):
    return {'image': 'x.jpg', 'image_size': [100, 100],
            'predictions': predictions, 'corrections': corrections}


def test_untouched_predictions_are_confirmed_correct():
    """The premise of correction-based labeling: silence means 'correct'."""
    preds = [{'id': 0, 'cx': 5, 'cy': 5, 'w': 4, 'h': 6, 'bits': '100000'},
             {'id': 1, 'cx': 15, 'cy': 5, 'w': 4, 'h': 6, 'bits': '110000'}]
    cells, tally = apply_corrections(doc(preds))
    assert len(cells) == 2
    assert tally['edited'] == tally['deleted'] == tally['added'] == 0


def test_edits_deletions_and_additions_are_applied():
    preds = [{'id': 0, 'cx': 5, 'cy': 5, 'w': 4, 'h': 6, 'bits': '100000'},
             {'id': 1, 'cx': 15, 'cy': 5, 'w': 4, 'h': 6, 'bits': '110000'}]
    cells, tally = apply_corrections(doc(
        preds,
        edited=[{'id': 0, 'bits': '111000'}],
        deleted=[1],
        added=[{'cx': 25, 'cy': 5, 'w': 4, 'h': 6, 'bits': '000001'}]))
    bits = sorted(c['bits'] for c in cells)
    assert bits == ['000001', '111000']
    assert tally == {'predicted': 2, 'edited': 1, 'deleted': 1,
                     'added': 1, 'blank_skipped': 0}


def test_blank_cells_are_skipped_not_written_as_labels():
    """A cell left at 000000 is UNLABELED, not an empty cell. Writing it as a
    real label would teach the classifier that blank paper is a valid pattern."""
    preds = [{'id': 0, 'cx': 5, 'cy': 5, 'w': 4, 'h': 6, 'bits': '000000'},
             {'id': 1, 'cx': 15, 'cy': 5, 'w': 4, 'h': 6, 'bits': '110000'}]
    cells, tally = apply_corrections(doc(preds))
    assert [c['bits'] for c in cells] == ['110000']
    assert tally['blank_skipped'] == 1


def test_bits_to_label_matches_the_dataset_convention():
    """bit i of the string is dot i+1; the integer is the 6-bit value."""
    assert bits_to_label('000000') == 0
    assert bits_to_label('100000') == 1     # dot1
    assert bits_to_label('000001') == 32    # dot6
    assert bits_to_label('111111') == 63
    assert bits_to_label('101000') == 1 + 4


def test_bits_to_label_round_trips_through_dot_pattern_utils():
    """Guards against the experiment code and the training loaders disagreeing
    about dot order — a silent way to mislabel an entire dataset."""
    from dot_pattern_utils import label_to_bits6
    for label in range(64):
        assert bits_to_label(label_to_bits6(label)) == label


# ── label_tool.py: the page a labeler actually gets ─────────────────────────

def test_build_page_is_self_contained_and_carries_the_cells(tmp_path):
    import PIL.Image
    from label_tool import build_page

    img = PIL.Image.new('RGB', (200, 120), 'white')
    cells = [{'cx': 50.0, 'cy': 60.0, 'w': 20.0, 'h': 30.0,
              'bits': '101010', 'conf': 0.91, 'probs': [0.9] * 6}]
    out = build_page(img, cells, 'page.jpg', tmp_path / 'page.label.html',
                     next_href='next.html', index_href='index.html',
                     position=(1, 3))
    html = out.read_text()

    assert 'data:image/jpeg;base64,' in html      # image embedded, works offline
    assert 'http://' not in html and 'https://' not in html   # no network deps
    assert '101010' in html
    assert 'next.html' in html and 'index.html' in html

    payload = json.loads(html.split('const DATA = ')[1].split(';\n')[0])
    assert payload['cells'][0]['bits'] == '101010'
    assert payload['cells'][0]['id'] == 0
    assert payload['display_scale'] <= 1.0
