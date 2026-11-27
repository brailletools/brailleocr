#!/usr/bin/env python3
"""
Copy a results file into bench/results/ in a form that is safe to commit.

Removes what must not be published: transcriptions from third-party datasets whose licence
does not allow redistribution (BrailleBench declares none), the Braille we read from their
images (which amounts to the same text), and local file paths. Numbers and image ids stay,
so every score can still be compared per image. Keep the full file locally: score_text.py
needs the removed fields to rescore.

Usage:
  python experiments/publish_results.py FULL.json bench/results/NAME.json
"""
import json
import sys
from pathlib import Path

REMOVE = ('reference', 'braille', 'predicted_text', 'text')


def publishable(data):
    def clean(row):
        row = {k: v for k, v in row.items() if k not in REMOVE}
        # score_text.py stores its translation as `predicted` (a string); run.mjs uses the
        # same key for the number of detected cells, which stays.
        if isinstance(row.get('predicted'), str):
            del row['predicted']
        return row

    if isinstance(data, list):                      # score_text.py --out
        return [clean(r) for r in data]
    out = {**data, 'rows': [clean(r) for r in data['rows']]}
    for key in ('detectorPath', 'classifierPath'):
        if key in out and Path(out[key]).is_absolute():
            out[key] = '/'.join(Path(out[key]).parts[-3:])
    return out


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    source, dest = Path(sys.argv[1]), Path(sys.argv[2])
    dest.write_text(json.dumps(publishable(json.loads(source.read_text()))) + '\n')
    print(f'wrote {dest}')


if __name__ == '__main__':
    main()
