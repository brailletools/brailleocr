#!/usr/bin/env python3
"""
Paired comparison of two js/eval/run.mjs results files over the same images.

For each set and metric it reports both means and the mean per-image difference (B - A)
with a 95% interval from resampling images. An interval that includes zero means the run
cannot tell the two apart on that set.

Usage:
  python experiments/compare_results.py A.json B.json [--names wide narrow]
"""
import argparse
import json
import random
from pathlib import Path

METRICS = (('cellErrorRate', 'cell error rate (lower is better)'),
           ('cellAccuracyDetected', 'per-cell accuracy on detected cells'),
           ('recall', 'detection recall'),
           ('precision', 'detection precision'))


def paired(a_rows, b_rows, key):
    """{set: [(a, b), ...]} over images present in both runs with a value for `key`."""
    b_by_id = {r['id']: r for r in b_rows}
    out = {}
    for r in a_rows:
        other = b_by_id.get(r['id'])
        if other is None or r.get(key) is None or other.get(key) is None:
            continue
        out.setdefault(r['set'], []).append((r[key], other[key]))
    return out


def interval(values, resamples=4000, seed=1):
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return sum(values) / n, means[int(0.025 * resamples)], means[int(0.975 * resamples)]


def report(a_rows, b_rows, names):
    lines = []
    for key, label in METRICS:
        sets = paired(a_rows, b_rows, key)
        if not sets:
            continue
        lines.append(f'\n{label}')
        for name, pairs in sets.items():
            a = sum(x for x, _ in pairs) / len(pairs)
            b = sum(y for _, y in pairs) / len(pairs)
            d, lo, hi = interval([y - x for x, y in pairs])
            verdict = 'no clear difference' if lo <= 0 <= hi else 'differs'
            lines.append(f'  {name:<22}{len(pairs):>4} images  {names[0]} {100 * a:5.1f}%  {names[1]} {100 * b:5.1f}%'
                         f'  difference {100 * d:+5.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]  {verdict}')
    return '\n'.join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--names', nargs=2, default=['A', 'B'])
    args = ap.parse_args()
    rows = [json.loads(Path(p).read_text())['rows'] for p in (args.a, args.b)]
    print(report(rows[0], rows[1], args.names))
    print('\nFew pages (dsbi-val has 4) give wide or unreliable intervals.')


if __name__ == '__main__':
    main()
