#!/usr/bin/env python3
"""
find_injections.py
==================
Detect GPS electron injection events within a given time interval and write
the catalog to disk.

Usage
-----
    python3 find_injections.py \\
        --start 2016-09-27T00:00:00 \\
        --end   2016-09-28T00:00:00 \\
        --output-dir ./catalogs/

The script discovers all spacecraft directories under the GPS data root,
selects the files whose 7-day window overlaps the requested interval, runs
the full detection pipeline, filters the results to events whose onset falls
within [start, end), and writes a JSON catalog.

GPS file naming convention assumed: {spacecraft}_{YYMMDD}_v1.10.ascii
All 2-digit years are interpreted as 20YY.

Optional arguments let you override the default detection thresholds; see
--help for the full list.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from gps_reader import read_gps_multi
from gps_injection_detect import detect_injections, write_catalog


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_DATA_DIR = Path('/home/buck/data/gps')
FILE_DURATION_DAYS = 7


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ts(s):
    """Parse an ISO 8601 string to a UTC-aware pd.Timestamp."""
    t = pd.Timestamp(s)
    return t.tz_localize('UTC') if t.tzinfo is None else t.tz_convert('UTC')


def _file_start(path):
    """Return the UTC start date encoded in a GPS filename (YYMMDD → 20YYMMDD)."""
    date_str = Path(path).stem.split('_')[1]   # e.g. '160925'
    yy, mm, dd = int(date_str[:2]), int(date_str[2:4]), int(date_str[4:6])
    return pd.Timestamp(year=2000 + yy, month=mm, day=dd, tz='UTC')


def find_overlapping_files(data_dir, t_start, t_end):
    """Return sorted list of GPS files whose 7-day window overlaps [t_start, t_end)."""
    file_dur = pd.Timedelta(days=FILE_DURATION_DAYS)
    files = []
    for sc_dir in sorted(data_dir.iterdir()):
        if not sc_dir.is_dir():
            continue
        for f in sorted(sc_dir.glob('*_v1.10.ascii')):
            try:
                fs = _file_start(f)
            except (IndexError, ValueError):
                continue
            if fs < t_end and (fs + file_dur) > t_start:
                files.append(f)
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Detect GPS injection events in a time interval.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--start', required=True,
                        metavar='ISO8601',
                        help='Interval start (e.g. 2016-09-27T00:00:00)')
    parser.add_argument('--end', required=True,
                        metavar='ISO8601',
                        help='Interval end   (e.g. 2016-09-28T00:00:00)')
    parser.add_argument('--output-dir', required=True,
                        metavar='DIR',
                        help='Directory for catalog output')
    parser.add_argument('--data-dir', default=str(DEFAULT_DATA_DIR),
                        metavar='DIR',
                        help='GPS data root (spacecraft subdirs live here)')
    parser.add_argument('--format', choices=['json', 'jsonl'], default='json',
                        help='Catalog file format')
    # Detection parameters
    parser.add_argument('--snr-threshold', type=float, default=3.0,
                        metavar='FLOAT',
                        help='SNR threshold for inflection detection')
    parser.add_argument('--l-max', type=float, default=6.0,
                        metavar='FLOAT',
                        help='Maximum L-shell to include')
    parser.add_argument('--min-platforms', type=int, default=3,
                        metavar='INT',
                        help='Minimum platforms required for a valid cluster')
    parser.add_argument('--n-channels', type=int, default=3,
                        metavar='INT',
                        help='Minimum channels required per platform')
    args = parser.parse_args()

    try:
        t_start = _parse_ts(args.start)
        t_end   = _parse_ts(args.end)
    except Exception as e:
        sys.exit(f"Error parsing time arguments: {e}")

    if t_end <= t_start:
        sys.exit("Error: --end must be after --start.")

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.is_dir():
        sys.exit(f"Error: data directory not found: {data_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Discover files
    # ------------------------------------------------------------------
    print(f"Interval : {t_start}  →  {t_end}")
    print(f"Data dir : {data_dir}")

    files = find_overlapping_files(data_dir, t_start, t_end)
    if not files:
        sys.exit("No GPS files found covering the requested interval.")

    platforms_found = sorted({f.parent.name for f in files})
    print(f"Files    : {len(files)} across {len(platforms_found)} platform(s): "
          f"{', '.join(platforms_found)}")

    # ------------------------------------------------------------------
    # 2. Load
    # ------------------------------------------------------------------
    print("Loading  : reading GPS files …")
    data, _ = read_gps_multi(files)

    # ------------------------------------------------------------------
    # 3. Detect
    # ------------------------------------------------------------------
    print("Detect   : running pipeline …")
    events = detect_injections(
        data,
        snr_threshold=args.snr_threshold,
        l_max=args.l_max,
        min_platforms=args.min_platforms,
        n_channels_required=args.n_channels,
        data_files=[str(f) for f in files],
    )

    # ------------------------------------------------------------------
    # 4. Filter to the requested window
    # ------------------------------------------------------------------
    in_window = [e for e in events if t_start <= e['onset_utc'] < t_end]

    print(f"Results  : {len(events)} event(s) detected in loaded data; "
          f"{len(in_window)} with onset in requested interval.")

    if in_window:
        for e in in_window:
            print(f"           {e['onset_utc'].isoformat()}  "
                  f"n={e['n_platforms']}  "
                  f"platforms={e['cluster_platforms']}")

    # ------------------------------------------------------------------
    # 5. Write catalog
    # ------------------------------------------------------------------
    start_tag = t_start.strftime('%Y%m%dT%H%M%S')
    end_tag   = t_end.strftime('%Y%m%dT%H%M%S')
    ext       = args.format
    out_path  = output_dir / f"injections_{start_tag}_{end_tag}.{ext}"

    write_catalog(in_window, out_path, format=args.format)
    print(f"Catalog  : {out_path}  ({len(in_window)} event(s))")


if __name__ == '__main__':
    main()
