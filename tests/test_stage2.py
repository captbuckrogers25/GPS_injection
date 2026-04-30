"""
Tests for Phase 3 Stage 2: find_coincidences (multi-platform coincidence clustering).

Run with:
    cd /home/buck/GPS_injection && python -m pytest tests/test_stage2.py -v

Paper event validation (tests 8-9) requires GPS data under /home/buck/data/gps/.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from gps_injection_detect import (
    _UnionFind,
    find_coincidences,
    find_inflections,
    merge_channels,
)
from gps_reader import read_gps_multi

DATA_DIR = Path('/home/buck/data/gps')

# Files for paper Event 1 (2016-09-27, onset ~14:48 UTC)
EVENT1_PLATFORMS = ['ns54', 'ns62', 'ns55', 'ns61', 'ns68', 'ns60', 'ns72', 'ns73']
EVENT1_FILES = [DATA_DIR / sc / f'{sc}_160925_v1.10.ascii' for sc in EVENT1_PLATFORMS]

# Files for paper Event 2 (2022-04-10, onset ~04:30 UTC)
EVENT2_PLATFORMS = ['ns65', 'ns53', 'ns68', 'ns63', 'ns56', 'ns71', 'ns70', 'ns57']
EVENT2_FILES = [DATA_DIR / sc / f'{sc}_220410_v1.10.ascii' for sc in EVENT2_PLATFORMS]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

_T0 = pd.Timestamp('2020-01-01T00:00:00', tz='UTC')


def _make_merged_df(rows):
    """Build a minimal merged_candidates DataFrame.

    rows : list of (platform, t_offset_s, mlt)
    """
    records = [
        {
            'platform': p,
            'time_onset': _T0 + pd.Timedelta(seconds=t),
            'l_shell': 5.0,
            'mlt': mlt,
            'n_channels_fired': 3,
            'channels_fired': [1, 2, 3],
            'channel_times': {},
            'max_snr': 5.0,
            'min_snr': 2.0,
        }
        for p, t, mlt in rows
    ]
    return pd.DataFrame(records)


def _run_pipeline(files):
    """Run Phases 1+2 for a list of files; return concatenated merged candidates."""
    data, _ = read_gps_multi(files)
    merged_list = []
    for platform, df in data.items():
        infl = find_inflections(df, scale='log', snr_threshold=1.5, l_max=6.0)
        mc = merge_channels(infl, n_channels_required=3, time_tolerance_s=480)
        if not mc.empty:
            merged_list.append(mc)
    if not merged_list:
        return pd.DataFrame()
    return pd.concat(merged_list, ignore_index=True)


# ---------------------------------------------------------------------------
# Test 1: UnionFind sanity
# ---------------------------------------------------------------------------

def test_unionfind_find_union():
    uf = _UnionFind(5)
    # Initially each element is its own root
    assert all(uf.find(i) == i for i in range(5))

    uf.union(0, 1)
    assert uf.find(0) == uf.find(1)
    assert uf.find(2) != uf.find(0)

    uf.union(2, 3)
    assert uf.find(2) == uf.find(3)
    assert uf.find(0) != uf.find(2)

    # Connect the two components
    uf.union(1, 2)
    assert uf.find(0) == uf.find(3)


def test_unionfind_groups():
    uf = _UnionFind(6)
    uf.union(0, 1)
    uf.union(1, 2)
    uf.union(3, 4)
    groups = uf.groups()
    sizes = sorted(len(g) for g in groups)
    assert sizes == [1, 2, 3], f"Expected group sizes [1,2,3], got {sizes}"


def test_unionfind_path_compression():
    # Create a chain 0→1→2→3→4 and verify find still returns the correct root
    uf = _UnionFind(5)
    for i in range(4):
        uf.union(i, i + 1)
    root = uf.find(0)
    assert all(uf.find(i) == root for i in range(5))


# ---------------------------------------------------------------------------
# Test 2: Synthetic dispersionless cluster
# ---------------------------------------------------------------------------

def test_dispersionless_cluster():
    """Three platforms at MLT 23.5 / 0.0 / 0.5 with onsets within 240 s → one cluster."""
    df = _make_merged_df([
        ('P1', 0,   23.5),
        ('P2', 120,  0.0),
        ('P3', 200,  0.5),
    ])
    clusters = find_coincidences(df, simul_tolerance_s=240, min_platforms=3)
    assert len(clusters) == 1, f"Expected 1 cluster, got {len(clusters)}"
    assert clusters[0]['platform'].nunique() == 3


# ---------------------------------------------------------------------------
# Test 3: Synthetic dispersed cluster
# ---------------------------------------------------------------------------

def test_dispersed_cluster():
    """Three platforms at MLT 0 / 4 / 8 with drift-consistent onset delays → one cluster."""
    # T_drift_max = 5400 s; for Δφ=4h: Δt_max = (4/24)*5400+240 = 1140 s
    df = _make_merged_df([
        ('P1',    0,  0.0),
        ('P2',  800,  4.0),
        ('P3', 1600,  8.0),
    ])
    clusters = find_coincidences(df, simul_tolerance_s=240, min_platforms=3)
    assert len(clusters) == 1, f"Expected 1 cluster, got {len(clusters)}"
    assert clusters[0]['platform'].nunique() == 3


# ---------------------------------------------------------------------------
# Test 4: Synthetic uncorrelated — zero clusters
# ---------------------------------------------------------------------------

def test_uncorrelated_no_cluster():
    """Onsets hours apart → all pairs fail the time constraint → zero clusters."""
    df = _make_merged_df([
        ('P1',     0,  6.0),
        ('P2',  7200, 14.0),
        ('P3', 14400, 22.0),
    ])
    clusters = find_coincidences(df, simul_tolerance_s=240, min_platforms=3)
    assert len(clusters) == 0, f"Expected 0 clusters, got {len(clusters)}"


# ---------------------------------------------------------------------------
# Test 5: MLT wraparound
# ---------------------------------------------------------------------------

def test_mlt_wraparound():
    """Two platforms straddling the 0/24 boundary with simultaneous onsets → cluster."""
    df = _make_merged_df([
        ('P1',   0, 23.5),
        ('P2', 100,  0.5),
    ])
    clusters = find_coincidences(df, simul_tolerance_s=240, min_platforms=2)
    assert len(clusters) == 1, (
        f"Expected wraparound pair to cluster, got {len(clusters)} clusters"
    )


# ---------------------------------------------------------------------------
# Test 6: Westward wrong direction
# ---------------------------------------------------------------------------

def test_westward_rejected():
    """Later platform is clearly westward → pair rejected → zero clusters."""
    # P2 is 2 h west of P1 and arrives 500 s later — kinematically wrong direction
    df = _make_merged_df([
        ('P1',   0, 5.0),
        ('P2', 500, 3.0),
    ])
    clusters = find_coincidences(df, simul_tolerance_s=240, min_platforms=2)
    assert len(clusters) == 0, (
        f"Westward pair should not cluster, got {len(clusters)} clusters"
    )


# ---------------------------------------------------------------------------
# Test 7: Mixed dispersionless + dispersed — all five in one cluster
# ---------------------------------------------------------------------------

def test_mixed_cluster():
    """Two simultaneous near midnight (dispersionless) + three delayed eastward."""
    df = _make_merged_df([
        ('P1',    0, 23.8),  # dispersionless pair
        ('P2',   50,  0.2),
        ('P3',  300,  2.0),  # slightly east and delayed
        ('P4',  750,  5.0),
        ('P5', 1500,  9.0),
    ])
    clusters = find_coincidences(df, simul_tolerance_s=240, min_platforms=3)
    assert len(clusters) == 1, f"Expected 1 cluster, got {len(clusters)}"
    assert clusters[0]['platform'].nunique() == 5


# ---------------------------------------------------------------------------
# Test 8: Paper Event 1 — 2016-09-27, onset ~14:48 UTC
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not all(f.exists() for f in EVENT1_FILES),
    reason="Event 1 data files not found under /home/buck/data/gps/",
)
def test_paper_event1_cluster():
    t_expected = pd.Timestamp('2016-09-27 14:48:00', tz='UTC')
    t_tol = pd.Timedelta('30min')

    merged = _run_pipeline(EVENT1_FILES)
    assert not merged.empty, "No merged candidates produced for Event 1 platforms"

    clusters = find_coincidences(merged, simul_tolerance_s=240, min_platforms=3)
    assert len(clusters) > 0, "find_coincidences returned no clusters for Event 1"

    # Find the cluster containing candidates nearest the expected onset time
    event_clusters = [
        c for c in clusters
        if ((c['time_onset'] - t_expected).abs() < t_tol).any()
    ]
    assert len(event_clusters) > 0, (
        f"No cluster has a candidate within ±30 min of {t_expected}.\n"
        f"Cluster onset ranges: "
        + str([(c['time_onset'].min(), c['time_onset'].max()) for c in clusters])
    )

    # Dominant event cluster: most platforms
    best = max(event_clusters, key=lambda c: c['platform'].nunique())
    found_platforms = set(best['platform'].unique())
    expected = set(EVENT1_PLATFORMS)
    n_found = len(found_platforms & expected)

    assert n_found >= 3, (
        f"Event 1 cluster contains only {n_found}/8 expected platforms.\n"
        f"Found: {sorted(found_platforms & expected)}\n"
        f"Missing: {sorted(expected - found_platforms)}"
    )


# ---------------------------------------------------------------------------
# Test 9: Paper Event 2 — 2022-04-10, onset ~04:30 UTC
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not all(f.exists() for f in EVENT2_FILES),
    reason="Event 2 data files not found under /home/buck/data/gps/",
)
def test_paper_event2_cluster():
    t_expected = pd.Timestamp('2022-04-10 04:30:00', tz='UTC')
    t_tol = pd.Timedelta('30min')

    merged = _run_pipeline(EVENT2_FILES)
    assert not merged.empty, "No merged candidates produced for Event 2 platforms"

    clusters = find_coincidences(merged, simul_tolerance_s=240, min_platforms=3)
    assert len(clusters) > 0, "find_coincidences returned no clusters for Event 2"

    event_clusters = [
        c for c in clusters
        if ((c['time_onset'] - t_expected).abs() < t_tol).any()
    ]
    assert len(event_clusters) > 0, (
        f"No cluster has a candidate within ±30 min of {t_expected}.\n"
        f"Cluster onset ranges: "
        + str([(c['time_onset'].min(), c['time_onset'].max()) for c in clusters])
    )

    best = max(event_clusters, key=lambda c: c['platform'].nunique())
    found_platforms = set(best['platform'].unique())
    expected = set(EVENT2_PLATFORMS)
    n_found = len(found_platforms & expected)

    assert n_found >= 3, (
        f"Event 2 cluster contains only {n_found}/8 expected platforms.\n"
        f"Found: {sorted(found_platforms & expected)}\n"
        f"Missing: {sorted(expected - found_platforms)}"
    )
