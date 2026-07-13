"""
Smoke tests for the two CLI entry points (pipeline.py, evaluate.py).

These only exercise --help — importing the module and parsing arguments — rather
than running actual OCR. A full run needs a real photo, the YOLOv8 model
(auto-downloaded from HuggingFace on first use), and optionally the MobileNetV2
classifier, none of which are appropriate to require in CI. --help still catches
the most common way this repo breaks: a bad import, or a dependency that's used
but not declared/installed.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def run_help(script):
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / script), '--help'],
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_pipeline_help():
    result = run_help('pipeline.py')
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout.lower()


def test_evaluate_help():
    result = run_help('evaluate.py')
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout.lower()
    assert '--dataset' in result.stdout
