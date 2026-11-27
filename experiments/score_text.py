#!/usr/bin/env python3
"""
Score a text set (BrailleBench) from a js/eval/run.mjs results file.

The runner keeps the Unicode Braille the reader produced for each image. This script
back-translates it with liblouis (Grade 1 images with en-ueb-g1, Grade 2 with en-ueb-g2,
as the images are labeled) and compares it with the reference transcription using the
benchmark's own definitions (run-evals/ocr_eval.py in the BrailleBench repo):

  CER, WER   edit distance / reference length, pooled over the images of a grade
  exact      the whole transcription is identical
  normalized exact: after NFKC, lowercasing and whitespace collapsing

Intervals resample images. Comparable with the BrailleBench paper's Table 1 for the same
grade, with two differences to keep in mind: we translate with liblouis here, and our reader
is a Braille reader (cells, then translation), not a model that outputs text directly.

Usage:
  python experiments/score_text.py /tmp/eval-braillebench.json [--out rows.json]
"""
import argparse
import json
import random
import re
import statistics
import subprocess
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

TABLES = {1: 'en-ueb-g1.ctb', 2: 'en-ueb-g2.ctb'}


def normalize(text):
    return re.sub(r'\s+', ' ', unicodedata.normalize('NFKC', str(text or '')).lower()).strip()


def edit_distance(a, b):
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def translate(braille, table, lou):
    """Unicode Braille (lines separated by newlines) -> English, line by line."""
    out = []
    for line in braille.split('\n'):
        line = line.replace('⠀', ' ').strip()
        if not line:
            continue
        proc = subprocess.run([lou, '-b', table], input=line, capture_output=True, text=True, timeout=20)
        out.append(proc.stdout.strip() if proc.returncode == 0 else '')
    return ' '.join(out).strip()


def bootstrap(values, resamples=2000, seed=1):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return sum(values) / n, means[int(0.025 * resamples)], means[int(0.975 * resamples)]


def score(rows, lou):
    scored = []
    for r in rows:
        if 'braille' not in r:
            continue
        pred = translate(r['braille'], TABLES[r['grade']], lou)
        ref = r['reference']
        scored.append({**r, 'predicted': pred,
                       'exact': pred == ref,
                       'norm_exact': normalize(pred) == normalize(ref),
                       'chars': edit_distance(ref, pred), 'ref_chars': max(1, len(ref)),
                       'words': edit_distance(ref.split(), pred.split()),
                       'ref_words': max(1, len(ref.split()))})
    return scored


def report(scored):
    lines = []
    for grade in sorted({r['grade'] for r in scored}):
        rs = [r for r in scored if r['grade'] == grade]
        n = len(rs)
        em = bootstrap([float(r['exact']) for r in rs])
        nem = bootstrap([float(r['norm_exact']) for r in rs])
        cer = sum(r['chars'] for r in rs) / sum(r['ref_chars'] for r in rs)
        wer = sum(r['words'] for r in rs) / sum(r['ref_words'] for r in rs)
        none = sum(1 for r in rs if r['predicted'] == '')
        lines.append(f'Grade {grade}: {n} images')
        lines.append(f'  CER {cer:.3f}   WER {wer:.3f}   (pooled; lower is better)')
        lines.append(f'  exact match      {100 * em[0]:.1f}% [{100 * em[1]:.1f}, {100 * em[2]:.1f}]')
        lines.append(f'  normalized exact {100 * nem[0]:.1f}% [{100 * nem[1]:.1f}, {100 * nem[2]:.1f}]')
        lines.append(f'  images with no output: {none}; median image time {statistics.median(r["seconds"] for r in rs):.1f} s')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('results')
    ap.add_argument('--out', help='write per-image predictions here')
    args = ap.parse_args()

    from liblouis_env import get_lou_translate
    rows = json.loads(Path(args.results).read_text())['rows']
    scored = score(rows, get_lou_translate())
    if not scored:
        raise SystemExit('no text rows in that results file (run js/eval/run.mjs --sets braillebench)')
    print(report(scored))
    if args.out:
        Path(args.out).write_text(json.dumps(scored, indent=1, ensure_ascii=False))
        print(f'wrote {args.out}')


if __name__ == '__main__':
    main()
