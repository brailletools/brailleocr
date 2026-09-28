"""
Tests for the experiment harnesses in experiments/ and for bench.py.

These exist because two measurement bugs in this code produced confidently wrong
results before anyone noticed (RESEARCH.md records both):

  - bench.py counted detector forward passes by counting run_detection() calls,
    while the pipeline invoked the model from five call sites. A stage issuing
    630 forward passes was reported as issuing 0.
  - edge_cell_recall.py scored DSBI +recto and +verso as separate pages, but
    they are the same scan with complementary labels, so real cells from the
    other side counted as errors.

Neither was caught by running the code — both ran cleanly and printed plausible
numbers. So these tests target the *logic that turns observations into claims*:
timing attribution, ground-truth matching, and the corrections→labels rule.
They use synthetic inputs and no models, so they run in CI in under a second.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'experiments'))

import bench                                      # noqa: E402
from edge_cell_recall import label_edges, match   # noqa: E402
from merge_corrections import apply_corrections, bits_to_label   # noqa: E402
from text_cer import levenshtein                  # noqa: E402


# ── bench.py: exclusive timing and forward-pass attribution ──────────────────

def test_stage_timer_subtracts_child_time():
    """A parent's exclusive time must exclude a wrapped child's time, or the
    shares sum past 100% and a recovery stage's cost gets double-counted."""
    timer = bench.StageTimer()

    def child():
        for _ in range(200000):
            pass

    def parent():
        child_w()

    child_w = timer.wrap('child', child)
    parent_w = timer.wrap('parent', parent)
    parent_w()

    assert timer.calls == {'parent': 1, 'child': 1}
    # Parent does nothing but call the child, so its exclusive time is tiny
    # while its inclusive time contains the child's.
    assert timer.exclusive['parent'] < timer.exclusive['child']
    assert timer.inclusive['parent'] >= timer.inclusive['child']


def test_forward_passes_attributed_to_innermost_stage():
    """Counting model calls at the model catches every call site; counting
    run_detection() calls does not (the bug this replaced)."""
    timer = bench.StageTimer()

    class FakeYOLO:
        names = {0: '100000'}

        def __call__(self, *a, **k):
            return ['result']

    counting = bench.CountingModel(FakeYOLO(), timer)

    def stage_that_calls_model_directly():
        for _ in range(3):
            counting('image.jpg')

    wrapped = timer.wrap('sneaky_stage', stage_that_calls_model_directly)
    counting('outside')          # before any stage opens
    wrapped()

    assert timer.forward['sneaky_stage'] == 3
    assert timer.forward['(outside any stage)'] == 1
    assert sum(timer.forward.values()) == 4


def test_counting_model_passes_attributes_through():
    """pipeline reads model.names to map class indices to dot patterns."""
    class FakeYOLO:
        names = {0: '101010'}

    m = bench.CountingModel(FakeYOLO(), bench.StageTimer())
    assert m.names == {0: '101010'}


def test_bootstrap_ci_brackets_the_mean_and_needs_two_points():
    values = [10.0, 12.0, 11.0, 30.0, 9.0, 10.5]
    lo, hi = bench.bootstrap_ci(values, resamples=500)
    mean = sum(values) / len(values)
    assert lo <= mean <= hi
    assert bench.bootstrap_ci([5.0], resamples=100) is None


def test_bootstrap_ci_is_wider_for_noisier_input():
    """The CI must actually respond to spread — this is what told us a stage
    whose cost varies per image was not measurable at n=6."""
    tight = bench.bootstrap_ci([10.0] * 6, resamples=800)
    loose = bench.bootstrap_ci([1.0, 50.0, 2.0, 40.0, 3.0, 30.0], resamples=800)
    assert (tight[1] - tight[0]) < (loose[1] - loose[0])


# ── edge_cell_recall.py: which cells are "edge", and what counts as a match ──

def cell(cx, cy, w=10.0, bits='100000'):
    return {'cx': cx, 'cy': cy, 'w': w, 'bits6': bits}


def test_label_edges_marks_first_and_last_of_each_line():
    cells = [cell(x, 100.0) for x in (10, 20, 30, 40)] + \
            [cell(x, 200.0) for x in (10, 20, 30)]
    label_edges(cells, edge_k=1)
    row1 = sorted([c for c in cells if c['cy'] == 100.0], key=lambda c: c['cx'])
    assert [c['edge'] for c in row1] == [True, False, False, True]
    row2 = sorted([c for c in cells if c['cy'] == 200.0], key=lambda c: c['cx'])
    assert [c['edge'] for c in row2] == [True, False, True]


def test_label_edges_groups_rows_by_vertical_band_not_exact_y():
    """Real pages are slightly skewed; cells on one line differ in y."""
    cells = [cell(10, 100.0), cell(20, 101.5), cell(30, 103.0)]
    label_edges(cells, edge_k=1)
    assert all(c['line_len'] == 3 for c in cells)
    assert [c['edge'] for c in sorted(cells, key=lambda c: c['cx'])] == \
           [True, False, True]


def test_match_claims_each_ground_truth_cell_once():
    """Two predictions near one ground-truth cell must not both count as hits,
    or precision is silently inflated."""
    gt = [cell(100.0, 100.0)]
    pred = [{'cx': 101.0, 'cy': 100.0}, {'cx': 102.0, 'cy': 100.0}]
    matched = match(pred, gt, tol=10.0)
    assert len(matched) == 1


def test_match_respects_the_distance_tolerance():
    gt = [cell(100.0, 100.0)]
    assert match([{'cx': 400.0, 'cy': 100.0}], gt, tol=10.0) == {}


def test_match_on_empty_inputs():
    assert match([], [cell(1.0, 1.0)], tol=5.0) == {}
    assert match([{'cx': 1.0, 'cy': 1.0}], [], tol=5.0) == {}


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


# ── text_cer.py ──────────────────────────────────────────────────────────────

def test_levenshtein_basics():
    assert levenshtein('', '') == 0
    assert levenshtein('abc', 'abc') == 0
    assert levenshtein('abc', '') == 3
    assert levenshtein('kitten', 'sitting') == 3


def test_cer_of_doubled_text_exceeds_one():
    """Sanity check on the regime that made an early ablation report CER 2.1:
    when the output contains far more text than the reference, CER > 1."""
    ref = 'hello'
    assert levenshtein(ref, ref + ref + ref) / len(ref) > 1.0


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
