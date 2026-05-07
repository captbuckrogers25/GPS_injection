#!/usr/bin/env python3
"""
filter_catalog.py
=================
Post-process a GPS injection catalog by splitting events on the MLT of the
initial dispersionless signature:

  Nightside  (onset_mlt < 6  or  onset_mlt > 18):
    Classical injection candidates driven by substorm/dipolarisation.
    Written to  <stem>_nightside.json.

  Dayside  (6 ≤ onset_mlt ≤ 18):
    Possible radial diffusion / ULF-wave transport events.
    Written to  <stem>_radial_candidates.json.

  No dispersionless platforms:
    Excluded from both outputs (classification is ambiguous without at least
    one dispersionless platform to anchor the onset MLT).

The filter reads onset_mlt from each catalog record, which is the MLT of
the earliest-detected platform in the cluster.  In the vast majority of
events this platform is also dispersionless, so onset_mlt is a reliable
proxy for the location of the initial dispersionless signature.

Usage
-----
    python3 filter_catalog.py \\
        --input  catalogs/injections_20160901T000000_20161001T000000.json \\
        --output-dir  catalogs/filtered/

Multiple input files are accepted; each produces its own pair of outputs.
"""

import argparse
import json
import sys
from pathlib import Path


_NIGHTSIDE_LABEL = 'injection'
_DAYSIDE_LABEL   = 'possible_radial'

# MLT boundary separating nightside from dayside (hours)
_MLT_DAWN = 6.0
_MLT_DUSK = 18.0


def _classify(event):
    """Return 'injection', 'possible_radial', or None (exclude).

    Returns None when dispersionless_platforms is empty/absent, because there
    is no dispersionless onset to place in MLT.
    """
    if not event.get('dispersionless_platforms'):
        return None
    mlt = event.get('onset_mlt')
    if mlt is None:
        return None
    if mlt < _MLT_DAWN or mlt > _MLT_DUSK:
        return _NIGHTSIDE_LABEL
    return _DAYSIDE_LABEL


def filter_catalog(input_path, output_dir):
    """Split one catalog file into nightside and radial-candidate outputs.

    Parameters
    ----------
    input_path : path-like
    output_dir : Path

    Returns
    -------
    tuple: (n_nightside, n_dayside, n_excluded, nightside_path, dayside_path)
    """
    with open(input_path) as f:
        events = json.load(f)

    nightside = []
    dayside   = []
    excluded  = 0

    for event in events:
        label = _classify(event)
        if label is None:
            excluded += 1
            continue
        tagged = dict(event, classification=label)
        if label == _NIGHTSIDE_LABEL:
            nightside.append(tagged)
        else:
            dayside.append(tagged)

    stem = Path(input_path).stem
    night_path = output_dir / f'{stem}_nightside.json'
    day_path   = output_dir / f'{stem}_radial_candidates.json'

    with open(night_path, 'w') as f:
        json.dump(nightside, f, indent=2)
    with open(day_path, 'w') as f:
        json.dump(dayside, f, indent=2)

    return len(nightside), len(dayside), excluded, night_path, day_path


def main():
    parser = argparse.ArgumentParser(
        description='Split a GPS injection catalog by onset MLT (nightside vs. radial).',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--input', required=True, nargs='+',
                        metavar='JSON',
                        help='Input catalog file(s) produced by find_injections.py')
    parser.add_argument('--output-dir', required=True,
                        metavar='DIR',
                        help='Directory for filtered output catalogs')
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for raw_path in args.input:
        path = Path(raw_path)
        if not path.is_file():
            print(f'Warning: {path} not found — skipping.', file=sys.stderr)
            continue

        n, d, excl, night_path, day_path = filter_catalog(path, output_dir)
        total = n + d + excl
        print(f'{path.name}  ({total} events total)')
        print(f'  Nightside injections      : {n:4d}  →  {night_path}')
        print(f'  Radial candidates         : {d:4d}  →  {day_path}')
        print(f'  Excluded (no disp. onset) : {excl:3d}')


if __name__ == '__main__':
    main()
