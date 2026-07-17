"""
Resolves trained model weights (cell_detector.pt, cell_classifier.pt) without
assuming the `dataset` repo is checked out as a sibling directory -- mirrors
liblouis-env's ensure_installed() pattern (see
../liblouis-env/python/src/liblouis_env/fetch.py): prefer a local copy if one's
already there (fast, no network -- the common local-dev case where all
brailletools repos are checked out side by side), otherwise fetch the pinned
release version on demand and cache it, so this repo works the same way in CI
or in any deployment with no filesystem relationship to `dataset`.

See https://github.com/brailletools/brailleocr/issues/4.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import platformdirs
import requests

from dot_pattern_utils import REPOS_ROOT

DATASET_VERSION = (Path(__file__).resolve().parent / 'dataset.version').read_text().strip()


class ModelNotFoundError(RuntimeError):
    """Raised when a model file could not be located locally or fetched."""


def _cache_dir() -> Path:
    d = platformdirs.user_cache_path('brailleocr') / DATASET_VERSION
    d.mkdir(parents=True, exist_ok=True)
    return d


def _download(url: str, dest: Path) -> None:
    # Download to a temp file first, then rename into place atomically -- an
    # interrupted download must not leave a corrupt file sitting at `dest`
    # that a later run's dest.exists() check would treat as complete.
    tmp = dest.with_suffix(dest.suffix + '.part')
    try:
        with requests.get(url, stream=True, timeout=60) as r:
            r.raise_for_status()
            with open(tmp, 'wb') as f:
                shutil.copyfileobj(r.raw, f)
        tmp.replace(dest)
    finally:
        if tmp.exists():
            tmp.unlink()


def resolve_model(filename: str) -> Path:
    """
    Resolves a trained model file (e.g. 'cell_detector.pt'), preferring a
    sibling `dataset` checkout and falling back to fetching+caching it from
    the pinned brailletools/dataset release (DATASET_VERSION) otherwise.
    """
    sibling = REPOS_ROOT / 'dataset' / 'models' / filename
    if sibling.exists():
        return sibling

    cached = _cache_dir() / filename
    if cached.exists():
        return cached

    url = f'https://raw.githubusercontent.com/brailletools/dataset/{DATASET_VERSION}/models/{filename}'
    print(f"  Fetching {filename} from brailletools/dataset@{DATASET_VERSION} …")
    try:
        _download(url, cached)
    except requests.RequestException as e:
        raise ModelNotFoundError(
            f"{filename} not found in a sibling `dataset` checkout ({sibling}), and "
            f"could not be fetched from {url}: {e}"
        ) from e
    return cached


DEFAULT_SCRATCH_CLASSIFIER = Path('/tmp/braille-crops/cell_classifier.pt')


def resolve_classifier_path(scratch: Path = DEFAULT_SCRATCH_CLASSIFIER) -> Path:
    """
    cell_classifier.pt resolution shared by pipeline.py and evaluate.py, in
    the same priority order pipeline.py used before this module existed:
    1. sibling `dataset` checkout (REPOS_ROOT/dataset/models/cell_classifier.pt)
       -- the published, versioned copy.
    2. `scratch` (default /tmp/braille-crops/cell_classifier.pt) -- a
       freshly-trained classifier not yet copied into dataset/models/.
       Overridable for testing, so tests don't read/write the real path a
       local training workflow uses.
    3. fetch+cache from the pinned brailletools/dataset release.
    """
    durable = REPOS_ROOT / 'dataset' / 'models' / 'cell_classifier.pt'
    if durable.exists():
        return durable
    if scratch.exists():
        return scratch
    return resolve_model('cell_classifier.pt')
