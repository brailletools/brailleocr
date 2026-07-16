"""
Tests for model_fetch.py -- see https://github.com/brailletools/brailleocr/issues/4.

The fetch-from-release path is exercised for real (not mocked): that's the
whole point of this module (letting CI, or any checkout with no sibling
`dataset` repo, actually resolve a working model), so a real network fetch
against the pinned release is the only test that would have caught the repo's
previous complete lack of a fallback.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))
import model_fetch as mf


def test_resolve_model_prefers_a_sibling_dataset_checkout():
    # This dev machine has `dataset` checked out as a sibling of `brailleocr`
    # -- the common local-dev layout model_fetch.py's sibling-path branch is
    # for. Skip (not fail) if that assumption doesn't hold here.
    sibling = mf.REPOS_ROOT / 'dataset' / 'models' / 'cell_detector.pt'
    if not sibling.exists():
        pytest.skip('no sibling `dataset` checkout on this machine')
    assert mf.resolve_model('cell_detector.pt') == sibling


def test_resolve_model_fetches_and_caches_when_no_sibling_checkout_exists(tmp_path, monkeypatch):
    # Point REPOS_ROOT at an empty temp dir with no `dataset` subfolder, so
    # the sibling-path branch is unavailable and resolve_model() must fall
    # back to a real fetch from the pinned brailletools/dataset release.
    monkeypatch.setattr(mf, 'REPOS_ROOT', tmp_path)
    cache_root = tmp_path / 'cache'
    monkeypatch.setattr(mf.platformdirs, 'user_cache_path', lambda name: cache_root / name)

    path = mf.resolve_model('cell_classifier.pt')
    assert path.exists()
    assert path.stat().st_size > 1_000_000  # real weights file, not an error page/stub
    first_mtime = path.stat().st_mtime

    # Second call must hit the cache, not re-download -- verify by making any
    # further network call fail loudly if attempted.
    def _boom(*a, **kw):
        raise AssertionError('should not re-download an already-cached model')

    monkeypatch.setattr(mf, '_download', _boom)
    path_again = mf.resolve_model('cell_classifier.pt')
    assert path_again == path
    assert path_again.stat().st_mtime == first_mtime


def test_resolve_model_raises_model_not_found_error_when_fetch_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(mf, 'REPOS_ROOT', tmp_path)
    monkeypatch.setattr(mf.platformdirs, 'user_cache_path', lambda name: tmp_path / 'cache' / name)

    def _fail(url, dest):
        raise mf.requests.exceptions.ConnectionError('simulated network failure')

    monkeypatch.setattr(mf, '_download', _fail)
    with pytest.raises(mf.ModelNotFoundError):
        mf.resolve_model('cell_detector.pt')


def test_resolve_classifier_path_prefers_durable_over_scratch(tmp_path, monkeypatch):
    fake_repos_root = tmp_path / 'repos'
    durable_dir = fake_repos_root / 'dataset' / 'models'
    durable_dir.mkdir(parents=True)
    durable_path = durable_dir / 'cell_classifier.pt'
    durable_path.write_bytes(b'durable')

    scratch_path = tmp_path / 'scratch' / 'cell_classifier.pt'
    scratch_path.parent.mkdir(parents=True)
    scratch_path.write_bytes(b'scratch')

    monkeypatch.setattr(mf, 'REPOS_ROOT', fake_repos_root)

    assert mf.resolve_classifier_path(scratch=scratch_path) == durable_path


def test_resolve_classifier_path_falls_back_to_scratch_when_no_durable_copy(tmp_path, monkeypatch):
    fake_repos_root = tmp_path / 'repos'  # no dataset/models dir at all
    scratch_path = tmp_path / 'scratch' / 'cell_classifier.pt'
    scratch_path.parent.mkdir(parents=True)
    scratch_path.write_bytes(b'scratch')

    monkeypatch.setattr(mf, 'REPOS_ROOT', fake_repos_root)

    assert mf.resolve_classifier_path(scratch=scratch_path) == scratch_path


def test_resolve_classifier_path_falls_back_to_fetch_when_neither_exists(tmp_path, monkeypatch):
    fake_repos_root = tmp_path / 'repos'
    scratch_path = tmp_path / 'scratch' / 'cell_classifier.pt'  # never created

    monkeypatch.setattr(mf, 'REPOS_ROOT', fake_repos_root)
    cache_root = tmp_path / 'cache'
    monkeypatch.setattr(mf.platformdirs, 'user_cache_path', lambda name: cache_root / name)

    resolved = mf.resolve_classifier_path(scratch=scratch_path)
    assert resolved == cache_root / 'brailleocr' / mf.DATASET_VERSION / 'cell_classifier.pt'
    assert resolved.exists()
